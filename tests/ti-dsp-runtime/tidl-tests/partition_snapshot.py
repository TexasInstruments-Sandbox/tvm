"""Partition snapshots: the ordered composites of every TIDL subgraph per model.

Track 1 moves every offload-eligibility check from Python to the TIDL library.
A snapshot taken before a check moves, diffed after, shows exactly which ops
the move started or stopped offloading. Each difference must be classified
(see the Track 1 plan, section 5.7):

1. the old Python check accepted an op that TIDL now correctly rejects;
2. a missed port: the Python rule has no C++ equivalent yet;
3. the Python bound was stricter than TIDL's; a new composite needs a
   deliberate accept/decline decision plus a numeric check.

Regenerate with ``python diag_tidl_partition_snapshot.py --write`` and review
the diff; ``test_tidl_partition_snapshot.py`` fails on any unexplained change.
"""

import inspect
import json
import os

from tvm import relax
from tvm.relax.backend.tidl import TIDLOffloadCompiler
from tvm.relax.backend.tidl.tidl import _extract_composite_calls, _find_tidl_subgraphs

SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "snapshots", "partition_snapshots.json"
)

REAL_MODELS = ["resnet18", "mobilenet_v2", "yolov5n", "yolov8n"]


def _shape(sinfo):
    if isinstance(sinfo, relax.TensorType) and sinfo.shape is not None:
        return [int(d) for d in sinfo.shape]
    if isinstance(sinfo, relax.TupleType):
        return [_shape(f) for f in sinfo.fields]
    return None


def snapshot_module(mod):
    """Sorted list of subgraphs; each is the ordered ``[composite, out_shape]`` list."""
    subgraphs = []
    for _, func in _find_tidl_subgraphs(mod):
        subgraphs.append(
            [
                [str(fn.attrs["Composite"]), _shape(call.ty)]
                for fn, call, _ in _extract_composite_calls(func)
            ]
        )
    return sorted(subgraphs, key=json.dumps)


def layer_model_builders():
    """``{name: builder}`` for the per-layer offload test models."""
    import test_tidl_layer_offload as layer_models  # noqa: PLC0415

    return {
        name[len("_build_") : -len("_model")]: fn
        for name, fn in inspect.getmembers(layer_models, inspect.isfunction)
        if name.startswith("_build_")
        and name.endswith("_model")
        and not inspect.signature(fn).parameters
    }


def _torchvision(name):
    import torch  # noqa: PLC0415
    from torch.export import export  # noqa: PLC0415
    from torchvision import models  # noqa: PLC0415

    from tvm.relax.frontend.torch import from_exported_program  # noqa: PLC0415

    model = getattr(models, name)(weights="DEFAULT").eval()
    with torch.no_grad():
        mod = from_exported_program(
            export(model, (torch.randn(1, 3, 224, 224),)), keep_params_as_input=True
        )
    mod, params = relax.frontend.detach_params(mod)
    return mod, dict(zip(mod["main"].params[1:], params["main"]))


def _yolo(name):
    from test_yolo_dsp import YOLO_MODELS, _create_yolo_model_unbound  # noqa: PLC0415

    mod, params, _, _ = _create_yolo_model_unbound(
        name, next(v for n, v in YOLO_MODELS if n == name)
    )
    return mod, params


def partition_real_model(name, tidl_tools_path, artifacts_dir="/tmp/tidl_snapshot"):
    mod, params = _yolo(name) if name.startswith("yolo") else _torchvision(name)
    compiler = TIDLOffloadCompiler(
        {"tidl_tools_path": tidl_tools_path, "artifacts_dir": artifacts_dir}
    )
    return compiler.partition(compiler.prepare(mod, params))


def partition_layer_model(builder):
    from tvm.relax.backend.tidl import partition_for_tidl  # noqa: PLC0415

    return partition_for_tidl(builder(), {})


def load_snapshots():
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)
