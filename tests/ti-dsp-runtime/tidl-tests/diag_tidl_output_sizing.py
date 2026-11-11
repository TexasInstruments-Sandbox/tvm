#!/usr/bin/env python
"""Report what a TIDL subgraph needs for its output, and whether it fits.

TIDL writes more bytes to a subgraph output than the logical tensor holds: it
appends ``outPadCh`` padding channels plus one spare, each ``outChannelPitch``
elements.  ``PadTIDLOutputStorage`` grows the allocation to cover that, using
sizes read from the subgraph's ``params_1.bin``.

This prints, per output, the descriptor fields that drive the requirement, the
bytes required, and the logical tensor size -- then runs the model twice on the
board and checks the two results agree.  Use it when:

  * a TIDL version bump changes ``outPadCh`` / ``outChannelPitch`` / element
    type, and you want to see the new requirement before trusting the pass;
  * a model returns plausible-looking but wrong data on the second and later
    inferences, which is the signature of an under-sized output allocation;
  * you need to confirm ``profile`` (repeat=2) agrees with ``run`` for a model.

The check compares the c7x_compute verbs: ``run`` infers once, ``profile`` sets
repeat=2 and returns the *second* inference, so a mismatch means state or memory
from the first inference leaked into the second.

Usage:
    python diag_tidl_output_sizing.py --model conv8 --board j722s-evm
    python diag_tidl_output_sizing.py --model mv2   --board j722s-evm

Requires TI_CGT_C7000_PATH, a staged TIDL tools directory and a board running c7x_compute
firmware.  Hardware runs must not overlap with a test suite.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TESTS_DIR / "dsp-cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsp_utils import (  # noqa: E402
    add_board_arg,
    get_current_board,
    run_dsp_dload,
    tidl_tools_config,
)

_HW = 16


def _build_conv(channels: int):
    """A conv whose output width is set by --model convN, for sizing sweeps."""
    import tvm
    from tvm.relax.frontend import nn

    class ConvReluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2D(3, channels, 3, 1, 1, bias=False)

        def main(self, x):
            return nn.relu(self.conv(x))

    input_spec = {"x": nn.spec.Tensor((1, 3, _HW, _HW), "float32")}
    mod, param_spec = ConvReluModel().export_tvm(spec={"main": input_spec})
    np.random.seed(42)
    params = [
        tvm.runtime.tensor(np.random.rand(*p.shape).astype("float32"), device=tvm.cpu())
        for _, p in param_spec
    ]
    param_dict = dict(zip(mod["main"].params[1:], params))
    input_data = np.random.randn(1, 3, _HW, _HW).astype("float32")
    calib = [np.random.randn(1, 3, _HW, _HW).astype("float32") for _ in range(2)]
    return mod, param_dict, input_data, calib


def _build_mv2():
    from test_tidl_mv2_e2e import _CALIB_INPUTS, _create_mv2

    mod, param_dict, _torch_model, input_data = _create_mv2()
    calib = _CALIB_INPUTS or [
        np.random.rand(1, 3, 224, 224).astype("float32") for _ in range(2)
    ]
    return mod, param_dict, input_data, calib


def _report_sizing(outputs, logical_bytes: int) -> None:
    """Print the TIDL output info and the required-vs-logical comparison."""
    from tvm.relax.backend.tidl import output_info

    print("\n" + "=" * 72)
    print("OUTPUT SIZING (TIDL_relaxGetOutputInfo)")
    print("=" * 72)
    for field in ("num_channels", "num_batches", "dim1", "dim2", "pad_ch",
                  "channel_pitch", "element_type", "buf_size", "name"):
        print(f"  {field:18s} = {[getattr(o, field) for o in outputs]}")

    required = output_info.output_byte_sizes(outputs)
    print(f"\n  required per slot : {required}")
    print(f"  logical tensor    : {logical_bytes} bytes")
    for slot, need in sorted(required.items()):
        verdict = "EXCEEDS the tensor -- allocation must be grown" if need > logical_bytes \
            else "fits inside the tensor"
        print(f"  slot {slot}: {need} bytes {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="conv8",
                        help="conv<N> for a conv with N output channels, or mv2")
    add_board_arg(parser)
    args = parser.parse_args()

    if args.board is None:
        parser.error("--board is required (this runs on hardware)")
    for var in ("TI_CGT_C7000_PATH",):
        if os.environ.get(var) is None:
            parser.error(f"{var} is not set")

    from tvm.relax.backend.tidl import TIDLOffloadCompiler

    work_dir = Path(tempfile.mkdtemp(prefix=f"diag_sizing_{args.model}_"))
    if args.model == "mv2":
        mod, param_dict, input_data, calib = _build_mv2()
    elif args.model.startswith("conv"):
        mod, param_dict, input_data, calib = _build_conv(int(args.model[4:]))
    else:
        parser.error(f"unknown --model {args.model!r}")

    compiler = TIDLOffloadCompiler(
        config={
            "board": get_current_board(),
            "artifacts_dir": str(work_dir / "tidl_artifacts"),
            **tidl_tools_config(),
            "num_calibration_frames": len(calib),
            "calibration_inputs": calib,
        }
    )
    result = compiler.build(mod, params=param_dict, exec_mode="c7x_dload",
                            build_dir=str(work_dir / "build"))
    if not result.artifacts:
        print("model did not offload to TIDL; nothing to report")
        return 1

    single, stdout, _ = run_dsp_dload(result.module_path, result.weights_path,
                                      [input_data], embedded_weights=True,
                                      profile=False)
    repeated, _, _ = run_dsp_dload(result.module_path, result.weights_path,
                                   [input_data], embedded_weights=True,
                                   profile=True)
    single = np.asarray(single).reshape(-1)
    repeated = np.asarray(repeated).reshape(-1)
    logical = single.size * single.dtype.itemsize

    for sg in result.artifacts.values():
        _report_sizing(sg["outputs"], logical)

    print("\n" + "=" * 72)
    print(f"REPEAT CHECK ({single.size} elements = {logical} bytes)")
    print("=" * 72)
    ndiff = int((single != repeated).sum())
    print(f"  run vs profile: {ndiff} of {single.size} elements differ")
    if ndiff == 0:
        print("  -> repeat=2 agrees with a single inference")
        return 0

    n = len(single)
    shift = next(
        (s for s in range(1, n) if np.array_equal(repeated[s:], single[: n - s])), None
    )
    if shift is not None:
        print(f"  -> the second inference's tail is a verbatim copy of the head "
              f"from element {shift} ({shift * 4} bytes).")
        print("     That is the under-sized-output signature: TIDL wrote past the "
              "allocation.")
    else:
        first = int(np.flatnonzero(single != repeated)[0])
        print(f"  -> differs from element {first} but not as a head-copy; "
              "inspect the DSP output below.")
    if stdout:
        tail = [ln for ln in stdout.splitlines() if "tidl" in ln.lower()][-10:]
        if tail:
            print("\n  last TIDL lines from the DSP:")
            for line in tail:
                print(f"    {line.strip()}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
