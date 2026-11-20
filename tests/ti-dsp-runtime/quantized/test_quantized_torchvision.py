#!/usr/bin/env python
"""
Quantized TorchVision classification models — MMALIB DSP test.

Phase 1 of porting tests/cstatic/cl_torchvision.py to quantized/: get every
TorchVision ImageNet classification model compiling and running correctly
with -mmalib=1, via PT2E quantization (C7xMMAQuantizer). This is a
correctness sweep, not an offload-maximization pass -- some models will
have ops that fall back to the scalar (non-MMA) path, which is fine here.

Reuses infrastructure directly rather than duplicating it:
  - cl_torchvision.py's load_model_with_preprocessing/load_image/
    get_all_classification_models (dynamic model discovery + correct
    per-model preprocessing)
  - pt2e-tests/pt2e_utils.py's e2e_quantize_and_import (quantize ->
    import), the same mechanism pt2e-tests/test_mobilenet_v2_pt2e_dsp.py
    already uses for mobilenet_v2.

Each model must give the reference's top-1 class, or a class the reference
scores within _TOP1_TIE of it, and with MMALIB stay within _MAX_ABS_DIFF of its
logits. Each run adds a row to results/sweep_metrics.csv
(under $WORKSPACE, default the working directory): top-1 of both, the
reference's top-1 margin, max |DSP - reference| and cosine of the logits, and
cycles on hardware. A model in _KNOWN_FAILURES that fails in a listed mode is
reported as xfail with the measured reason.

4 models are excluded: their int8 weight size alone exceeds the 256 MiB
AM67A DLOAD DDR heap (DDR_C7X_1_LOCAL_HEAP). 8 more are excluded for a
different reason: runtime DDR pool exhaustion at inference time (not
weight size). 1 more (maxvit_t) is excluded because it hits a TVM pass
bug after quantization. See _EXCLUDED below and quantized/README.md.

Usage:
    pytest test_quantized_torchvision.py -v --dsp-mode=c7x_host --mmalib
    pytest test_quantized_torchvision.py -v --dsp-mode=c7x_host --mmalib -k resnet50
    pytest test_quantized_torchvision.py -v --dsp-mode=c7x_dload --mmalib
    python test_quantized_torchvision.py --model resnet50 --dsp-mode c7x_host --mmalib
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_QUANTIZED_DIR = Path(__file__).parent
_CSTATIC_DIR = _QUANTIZED_DIR.parent.parent / "cstatic"
sys.path.insert(0, str(_CSTATIC_DIR))
from cl_torchvision import (  # noqa: E402
    get_all_classification_models,
    load_image,
    load_model_with_preprocessing,
)

_PT2E_TESTS_DIR = _QUANTIZED_DIR.parent / "pt2e-tests"
sys.path.insert(0, str(_PT2E_TESTS_DIR))
from dsp_utils import add_board_arg, compile_and_run_dsp, get_target_string  # noqa: E402
from pt2e_utils import e2e_quantize_and_import  # noqa: E402
from quant_recipe import BASELINE_TEST_IMAGE  # noqa: E402

pytestmark = [pytest.mark.c7x_only]

logger = logging.getLogger(__name__)

_TEST_IMAGE = _CSTATIC_DIR / "test_images" / "dog.jpg"

# Exceed the 256 MiB AM67A DLOAD DDR heap (DDR_C7X_1_LOCAL_HEAP) at pure
# int8 weight size alone, before any runtime/workspace overhead:
#   regnet_y_128gf 615MB, vit_h_14 603MB, vit_l_32 292MB, vit_l_16 290MB.
_EXCLUDED_WEIGHT_SIZE = {"regnet_y_128gf", "vit_h_14", "vit_l_32", "vit_l_16"}

# Not a weight-size problem -- these compile and run most of the way
# through, then exhaust the 352 MiB TVM DDR pool (workspace + weights +
# DLOAD segments, shared) at a late layer. Measured shortfall at the
# first failing allocation: convnext_base ~0.2 KB (requested 0.2 MB,
# 0.2 MB free of 369.1 MB -- the tightest margin in this set),
# convnext_large 2.8 KB, efficientnet_b6 2.74 MB, efficientnet_b7
# 43.6 KB, swin_b ~355 KB, swin_v2_b ~1.07 MB, swin_v2_s ~54.6 KB,
# swin_v2_t ~376.6 KB. Not limited to one architecture family -- the
# swin_* models are transformers, not CNNs. convnext_base only started
# reaching this OOM once its 7x7 depthwise convs correctly declined
# MMALIB offload (see ti_mmalib_qdq_dwconv.py's MMA-panel-fit check) and
# fell back to the scalar path instead of aborting the DSP earlier on
# an unsupported kernel geometry -- previously this OOM was unreachable,
# masked by that abort. See quantized/README.md.
_EXCLUDED_DDR_OOM = {
    "convnext_base",
    "convnext_large",
    "efficientnet_b6",
    "efficientnet_b7",
    "swin_b",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
}

# The original quantizer crash (quantize_per_tensor on the int64
# relative_position_index buffer) is fixed -- see _is_float_tensor in
# c7x_mma_quantizer.py. maxvit_t now gets past quantization but fails
# later in TVM's own pipeline: `tvm.error.InternalError: Check failed:
# (opt) is false: The struct info of Tuple must be TupleStructInfo, but
# expression lv7 has struct info R.Tensor(...)` in
# ti_eliminate_qdq_transparent.py. Not root-caused yet. See README.md.
_EXCLUDED_QUANT_BUG = {"maxvit_t"}

_EXCLUDED = _EXCLUDED_WEIGHT_SIZE | _EXCLUDED_DDR_OOM | _EXCLUDED_QUANT_BUG

TORCHVISION_MODELS = sorted(
    name for name, _ in get_all_classification_models() if name not in _EXCLUDED
)


# Elementwise bound on |DSP - reference| logits with MMALIB, on top of the top-1
# match every model must give: twice the largest difference of a passing model
# outside _KNOWN_FAILURES, measured on c7x_host and BeagleY-AI, rounded up to a
# multiple of 0.25, as _ATOL is in the per-model tests. It catches a miscompile
# that leaves top-1 intact; sweep_metrics.csv records each model's actual
# difference. Re-measure when a change legitimately moves the numbers
# (calibration, quantizer, kernel rounding).
_MAX_ABS_DIFF = 8.5

# A DSP top-1 other than the reference's still passes when the reference scores
# the DSP's class within this many logits of its own top-1: the two are a tie
# that the DSP's rounding can break either way. On dog.jpg several models'
# reference has classes 258 and 259 tied (efficientnet_v2_s/m/l by exactly 0,
# vgg13_bn by 0.087), and with the 8-bit MMALIB requant multiplier
# efficientnet_v2_s and vgg13_bn picked the other class at cosines of 0.9975 and
# 0.9996. Twice the largest such margin, rounded up to a multiple of 0.25; every
# other model's reference margin is far above it.
_TOP1_TIE = 0.25

# Models that fail the check, or pass it only because the reference's top-1 margin
# exceeds a known DSP error, in the modes they do so, with what was measured
# (dog.jpg, BASELINE_TEST_IMAGE recipe). They still run and record their metrics;
# a failure is reported as xfail with the reason, a pass as a pass. Remove an
# entry once its model's logits match the reference. See quantized-sweep.md.
_KNOWN_FAILURES: dict[str, tuple[tuple[str, ...], str]] = {}


def _metrics(dsp_out: np.ndarray, ref: np.ndarray) -> dict:
    """DSP vs fake-quant reference logits. ref_margin is the reference's top-1 logit
    minus its second, which tells a near-tie from a real top-1 mismatch;
    dsp_top1_gap is the reference's top-1 logit minus its logit for the DSP's top-1
    (0 when they match)."""
    d, r = dsp_out.reshape(-1).astype(np.float64), ref.reshape(-1).astype(np.float64)
    top2 = np.sort(r)[-2:]
    denom = np.linalg.norm(d) * np.linalg.norm(r)
    return {
        "top1_match": int(np.argmax(d)) == int(np.argmax(r)),
        "ref_top1": int(np.argmax(r)),
        "dsp_top1": int(np.argmax(d)),
        "ref_margin": float(top2[1] - top2[0]),
        "dsp_top1_gap": float(r.max() - r[int(np.argmax(d))]),
        "max_abs_diff": float(np.abs(d - r).max()),
        "cosine": float(d @ r / denom) if denom else float("nan"),
    }


_METRIC_FIELDS = [
    "model",
    "dsp_mode",
    "mmalib",
    "top1_match",
    "ref_top1",
    "dsp_top1",
    "ref_margin",
    "dsp_top1_gap",
    "max_abs_diff",
    "cosine",
    "cycles",
]


def _record_metrics(model_name: str, dsp_mode: str, mmalib: bool, m: dict, cycles: int):
    """Add or replace this model's row in results/sweep_metrics.csv. Written per call,
    like cycles.csv (quantized/conftest.py), so rows survive a forked test process."""
    out_file = Path(os.environ.get("WORKSPACE", ".")) / "results" / "sweep_metrics.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if out_file.exists():
        with open(out_file, newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["model"], row["dsp_mode"], row["mmalib"])] = row
    row = {"model": model_name, "dsp_mode": dsp_mode, "mmalib": str(mmalib), "cycles": cycles}
    row.update(m)
    rows[(model_name, dsp_mode, str(mmalib))] = row
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))


def _run_test(model_name: str, dsp_mode: str, mmalib: bool, record_cycles) -> None:
    model, preproc, _ = load_model_with_preprocessing(model_name)
    img = load_image(str(_TEST_IMAGE))
    # NCHW, as the DSP is given: the transform returns a channels-last view.
    example = (preproc(img).unsqueeze(0).contiguous(),)

    mod, input_np, ref_np = e2e_quantize_and_import(model, example, recipe=BASELINE_TEST_IMAGE)

    target = get_target_string(dsp_mode, use_cpp_api=True) + (" -mmalib=1" if mmalib else "")
    results = compile_and_run_dsp(
        mod=mod, input_data=input_np, target_string=target, execution_mode=dsp_mode
    )
    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_out = results[result_key].reshape(ref_np.shape).astype(np.float32)
    # c7x_dload_cycles is only populated for hardware runs.
    cycles = int(results.get("c7x_dload_cycles", 0) or 0)
    if mmalib:
        record_cycles(f"{model_name}_pt2e_i8", cycles)

    m = _metrics(dsp_out, ref_np)
    _record_metrics(model_name, dsp_mode, mmalib, m, cycles)
    logger.info(
        "%s vs reference: top-1 %s (ref %d, DSP %d, ref margin %.3f), max_abs_diff = %.4f, "
        "cosine = %.6f",
        model_name,
        "match" if m["top1_match"] else "MISMATCH",
        m["ref_top1"],
        m["dsp_top1"],
        m["ref_margin"],
        m["max_abs_diff"],
        m["cosine"],
    )
    failure = None
    if not m["top1_match"] and m["dsp_top1_gap"] <= _TOP1_TIE:
        logger.info(
            "%s: DSP top-1 %d is within %.3f of the reference's %d (tie <= %.2f)",
            model_name,
            m["dsp_top1"],
            m["dsp_top1_gap"],
            m["ref_top1"],
            _TOP1_TIE,
        )
    elif not m["top1_match"]:
        failure = (
            f"{model_name}: top-1 mismatch: DSP={m['dsp_top1']}, ref={m['ref_top1']} "
            f"(the reference scores the DSP's class {m['dsp_top1_gap']:.3f} below its top-1, "
            f"> tie {_TOP1_TIE}; reference margin {m['ref_margin']:.3f})"
        )
    # The generic path without MMALIB is checked on top-1 alone.
    elif mmalib and m["max_abs_diff"] > _MAX_ABS_DIFF:
        failure = f"{model_name}: max |DSP - reference| {m['max_abs_diff']:.3f} > {_MAX_ABS_DIFF}"
    if failure is None:
        return
    modes, reason = _KNOWN_FAILURES.get(model_name, ((), ""))
    if dsp_mode in modes:
        pytest.xfail(f"{failure}; known: {reason}")
    pytest.fail(failure)


@pytest.mark.parametrize("model_name", TORCHVISION_MODELS, ids=TORCHVISION_MODELS)
def test_quantized_torchvision_dsp(model_name, dsp_mode, mmalib, record_cycles):
    """Test a quantized TorchVision classification model on DSP."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")
    if not mmalib:
        # This is an MMALIB offload sweep (see module docstring) -- restrict
        # the large parametrized model set to the --mmalib stage rather than
        # tripling its cost across Host / c7x_dload / MMALIB c7x_dload.
        pytest.skip("quantized torchvision sweep requires --mmalib (see module docstring)")
    _run_test(model_name, dsp_mode, mmalib, record_cycles)


def main():
    parser = argparse.ArgumentParser(description="Quantized TorchVision DSP Test")
    parser.add_argument("--model", required=True, choices=TORCHVISION_MODELS)
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    add_board_arg(parser)
    parser.add_argument("--mmalib", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    def _record_cycles(name, cycles):
        if cycles:
            print(f"  cycles: {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)")

    print(f"Quantized {args.model} DSP Test (mode: {args.dsp_mode}, mmalib={args.mmalib})")
    _run_test(args.model, args.dsp_mode, args.mmalib, _record_cycles)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
