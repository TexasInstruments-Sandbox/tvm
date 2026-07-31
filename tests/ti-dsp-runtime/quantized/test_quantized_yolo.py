"""
Quantized YOLO object detection DSP test.

Tests INT8 quantized YOLOv5/v8/26 (PT2E QDQ + MMALIB offload) on DSP
comparing against the PyTorch fake-quantized reference (quantized_gm), like
the other tests in this directory. This replaces tidl-tests/test_yolo_dsp.py's
TIDL offload path: MMALIB bakes quantization scales into the compiled code at
build time, so there is no separate PC-calibration inference step to
disagree with the DSP (the source of TIDL's documented DFL-softmax NaN).

yolo26 has no DFL (reg_max=1, direct box regression) and defaults to an
NMS-free "one2one" deploy head; YOLOWrapper (model_utils.py) forces its
auxiliary "one2many" head instead, which still yields a raw per-anchor
detection tensor structurally analogous to v5/v8's.

All 4 variants return a single raw detection tensor; NMS is not exercised
here (same scope as the source test). Detection tensors mix small-magnitude
box regression with bounded-range class scores, so — like the source test —
pass/fail uses cosine similarity rather than element-wise tolerance.

--mmalib is required (tests skip without it): the non-MMALIB path runs
_ConvertLayoutNHWC, whose NCHW fallback for YOLO's 3D detection-head
reshapes leaves PassTimingInstrument's profile stack unbalanced and crashes
compile_for_dsp. MMALIB skips NHWC conversion entirely, so it isn't hit.

Usage:
    pytest test_quantized_yolo.py -v --dsp-mode=c7x_host --mmalib
    pytest test_quantized_yolo.py -v --dsp-mode=c7x_dload --mmalib
    pytest test_quantized_yolo.py -v --dsp-mode=c7x_host --mmalib -k yolov8n
    python test_quantized_yolo.py --model yolov8n --dsp-mode c7x_host --mmalib
"""

import argparse
import logging
import sys

import numpy as np
import pytest
import torch
from dsp_utils import (
    assert_dsp_comparison,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)
from model_utils import create_quantized_yolo_model

pytestmark = [pytest.mark.c7x_only]

logger = logging.getLogger(__name__)

# (model_name, version) — version is "v5", "v8", or "v26"
YOLO_MODELS = [
    ("yolov5n", "v5"),
    ("yolov5s", "v5"),
    ("yolov8n", "v8"),
    ("yolov8s", "v8"),
    ("yolo26n", "v26"),
]


def _skip_yolo_if_no_ultralytics(model_name):
    """Skip if ultralytics is not installed.

    v5, v8, and v26 all depend on the ultralytics package: v8/v26 directly
    import ultralytics.YOLO, and v5's torch.hub entry point also requires it
    (the modern yolov5 hubconf.py imports from it).
    """
    pytest.importorskip(
        "ultralytics",
        reason=f"{model_name} requires the ultralytics package",
    )


def _run_test(
    model_name: str,
    version: str,
    dsp_mode: str = "c7x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = True,
    profile_layers: bool = False,
    profile: bool = False,
    mmalib: bool = False,
) -> dict:
    tvm_mod, quantized_gm, input_data = create_quantized_yolo_model(model_name, version)

    with torch.no_grad():
        torch_result = quantized_gm(torch.from_numpy(input_data)).numpy()

    target_string = get_target_string(
        dsp_mode, profile_layers=profile_layers, use_cpp_api=use_cpp_api
    )
    if mmalib:
        target_string += " -mmalib=1"

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
        profile_layers=profile_layers,
        profile=profile,
    )

    # Populate *_vs_ref_max_diff for display, then override *_vs_ref_passed
    # with cosine similarity — detection tensors mix small box-regression
    # values with bounded class scores, so element-wise tolerance is not a
    # meaningful pass/fail signal (see module docstring).
    comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=2e1)
    flat_ref = torch_result.flatten()
    for key in list(comparison.keys()):
        if not key.endswith("_passed"):
            continue
        result_key = key.replace("_vs_ref_passed", "_result")
        if result_key in dsp_results:
            flat_dsp = dsp_results[result_key].flatten()
            cos_sim = float(
                np.dot(flat_ref, flat_dsp)
                / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_dsp) + 1e-10)
            )
            comparison[key] = cos_sim > 0.90
            comparison[key.replace("_passed", "_cos_sim")] = cos_sim

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


@pytest.mark.parametrize(
    "model_spec",
    YOLO_MODELS,
    ids=[m[0] for m in YOLO_MODELS],
)
def test_quantized_yolo_dsp(
    model_spec, dsp_mode, dsp_timeout, use_cpp_api, profile_layers, profile, mmalib, record_cycles
):
    """Test quantized YOLO model on DSP comparing against PyTorch reference."""
    model_name, version = model_spec

    _skip_yolo_if_no_ultralytics(model_name)
    if not mmalib:
        # The non-MMALIB path runs _ConvertLayoutNHWC, whose NCHW fallback
        # (needed for YOLO's 3D detection-head reshapes) leaves
        # PassTimingInstrument's profile stack unbalanced, crashing
        # compile_for_dsp at render() time. MMALIB skips NHWC conversion
        # entirely, so this is the only supported path for YOLO here.
        pytest.skip("quantized YOLO requires --mmalib (see module docstring)")

    results = _run_test(
        model_name=model_name,
        version=version,
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
        profile=profile,
        mmalib=mmalib,
    )
    record_cycles(f"{model_name}_int8", results["dsp_results"].get("c7x_dload_cycles", 0))
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


def main():
    model_names = [m[0] for m in YOLO_MODELS]
    parser = argparse.ArgumentParser(description="Quantized YOLO DSP Test")
    parser.add_argument("--model", required=True, choices=model_names)
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    parser.add_argument("--mmalib", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    version = next(v for n, v in YOLO_MODELS if n == args.model)

    print(f"Quantized {args.model} DSP Test (mode: {args.dsp_mode})")
    results = _run_test(
        model_name=args.model,
        version=version,
        dsp_mode=args.dsp_mode,
        profile_layers=args.profile,
        profile=args.profile,
        mmalib=args.mmalib,
    )

    dsp_results = results["dsp_results"]
    torch_result = results["torch_result"]

    for key in ("c7x_host_result", "c7x_dload_result"):
        if key in dsp_results:
            diff = np.max(np.abs(dsp_results[key] - torch_result))
            passed = results["comparison"].get(key.replace("_result", "_vs_ref_passed"), False)
            cos_sim = results["comparison"].get(key.replace("_result", "_vs_ref_cos_sim"))
            status = "PASS" if passed else "FAIL"
            cos_sim_str = f"{cos_sim:.6f}" if cos_sim is not None else "n/a"
            print(f"  {key}: max_diff={diff:.2e} cos_sim={cos_sim_str} [{status}]")

    return 0 if all(v for k, v in results["comparison"].items() if k.endswith("_passed")) else 1


if __name__ == "__main__":
    sys.exit(main())
