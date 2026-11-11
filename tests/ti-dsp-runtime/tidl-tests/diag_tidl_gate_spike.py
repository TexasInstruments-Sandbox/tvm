#!/usr/bin/env python
"""Cost and fidelity of the TIDL_relaxAllowNode gate on real models.

For each model, partitions through the gate (with ``allow_fn=lambda c: 1`` as
the no-gate baseline) and reports partition time, gate call count, process
max-RSS growth (the library appends to a global diagnostics list on every
call), and fidelity: every accepted match has a synthesised composite
structurally equal to a composite in the partitioned module.

Usage:
    python diag_tidl_gate_spike.py [resnet18 mobilenet_v2 yolov5n yolov8n]
"""

import os
import resource
import sys
import time
from collections import Counter

from tvm import relax
from tvm.relax.backend.tidl import TIDLOffloadCompiler, _gate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp_utils import tidl_tools_config  # noqa: E402
from test_tidl_allownode import _composites_in, _equal_ignoring_attrs  # noqa: E402


def _torchvision_module(name):
    import torch  # noqa: PLC0415
    from torch.export import export  # noqa: PLC0415
    from torchvision import models  # noqa: PLC0415

    from tvm.relax.frontend.torch import from_exported_program  # noqa: PLC0415

    model = getattr(models, name)(weights="DEFAULT").eval()
    with torch.no_grad():
        ep = export(model, (torch.randn(1, 3, 224, 224),))
        mod = from_exported_program(ep, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)
    return mod, dict(zip(mod["main"].params[1:], params["main"]))


def _yolo_module(name):
    from test_yolo_dsp import YOLO_MODELS, _create_yolo_model_unbound  # noqa: PLC0415

    version = next(v for n, v in YOLO_MODELS if n == name)
    mod, params, _, _ = _create_yolo_model_unbound(name, version)
    return mod, params


def _load(name):
    if name.startswith("yolo"):
        return _yolo_module(name)
    return _torchvision_module(name)


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(name, artifacts="/tmp/tidl_gate_spike"):
    mod, params = _load(name)
    compiler = TIDLOffloadCompiler({**tidl_tools_config(), "artifacts_dir": artifacts})
    prepared = compiler.prepare(mod, params)

    base = TIDLOffloadCompiler({**compiler.config, "allow_fn": lambda call: 1})
    t0 = time.time()
    base.partition(prepared)
    t_base = time.time() - t0

    rss0 = _rss_mb()
    t0 = time.time()
    with _gate.recording() as records:
        out = compiler.partition(prepared)
    t_gate = time.time() - t0
    rss1 = _rss_mb()

    reals = _composites_in(out)
    unmatched = Counter()
    for res in records:
        if not res.verdict:
            continue
        if not any(
            str(r.attrs["Composite"]) == res.pattern and _equal_ignoring_attrs(res.composite, r)
            for r in reals
        ):
            unmatched[res.pattern] += 1

    declined = Counter(res.pattern for res in records if not res.verdict)
    print(f"== {name}")
    print(
        f"   partition: no gate {t_base:.2f}s, with gate {t_gate:.2f}s, "
        f"{len(records)} gate calls, {sum(declined.values())} declined"
    )
    print(f"   maxrss growth during gated partition: {rss1 - rss0:+.1f} MB")
    print(
        f"   fidelity: {sum(unmatched.values())} accepted matches with no identical fused composite"
    )
    for pat, n in unmatched.most_common():
        print(f"     NOT IDENTICAL {pat} x{n}")
    for pat, n in declined.most_common():
        print(f"     declined {pat:32s} x{n}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["resnet18", "mobilenet_v2"]:
        try:
            run(m)
        except Exception as e:  # noqa: BLE001
            print(f"== {m}: SKIPPED ({type(e).__name__}: {str(e)[:200]})")
