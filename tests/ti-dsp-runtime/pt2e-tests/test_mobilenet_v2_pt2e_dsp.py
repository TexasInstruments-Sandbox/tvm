"""MobileNetV2 end-to-end PT2E classification test via C7xMMAQuantizer.

Exercises all four int8 MMALIB kernels in a single model:
  mmalib_conv2d_i8, mmalib_depthwise_conv2d_i8,
  c7x_int8_residual_add_relu, mmalib_matmul_bias_i8.

Two assertions on a real image (tests/cstatic/test_images/dog.jpg):

  1. No-MMALIB (generic int8 codegen): top-1 class matches PyTorch quantized
     reference.  Validates the C7xMMAQuantizer + TVM import pipeline end-to-end.

  2. MMALIB: max_diff ≤ 20.0 vs the same reference.  Validates that MMALIB
     compiles and produces a numerically reasonable result.  Top-1 is not
     asserted here — MMALIB integer arithmetic diverges from float-simulated
     int8 across 17+ bottleneck blocks, matching known behavior in
     quantized/test_quantized_mobilenet_v2.py (atol=25.0 with XNNPACKQuantizer).

Usage:
    cd tests/ti-dsp-runtime
    export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
    pytest --rootdir=. pt2e-tests/test_mobilenet_v2_pt2e_dsp.py --dsp-mode=c7x_host -v
    pytest --rootdir=. pt2e-tests/test_mobilenet_v2_pt2e_dsp.py --dsp-mode=c7x_dload -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from pt2e_utils import e2e_quantize_and_import  # noqa: E402

_CSTATIC_DIR = Path(__file__).parent.parent.parent / "cstatic"
sys.path.insert(0, str(_CSTATIC_DIR))
from cl_torchvision import load_image, load_model_with_preprocessing  # noqa: E402

_TEST_IMAGE = _CSTATIC_DIR / "test_images" / "dog.jpg"


@pytest.mark.quick
@pytest.mark.c7x_only
def test_mobilenet_v2_pt2e_i8(dsp_mode, record_cycles):
    """MobileNetV2 int8 via C7xMMAQuantizer: pipeline correctness + MMALIB sanity."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model, preproc, _ = load_model_with_preprocessing("mobilenet_v2")
    img = load_image(str(_TEST_IMAGE))
    example = (preproc(img).unsqueeze(0),)

    # 10 calibration batches with randn (same count as quantized/model_utils.py).
    mod, input_np, ref_np = e2e_quantize_and_import(
        model, example, dtype="int8", n_calibration_batches=10
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"

    # --- Part 1: no-MMALIB -------------------------------------------------------
    # Validates C7xMMAQuantizer + TVM import.  Top-1 must match PyTorch reference.
    results_no = compile_and_run_dsp(
        mod=mod,
        input_data=input_np,
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    dsp_no = results_no[result_key].reshape(ref_np.shape).astype(np.float32)
    assert np.argmax(dsp_no) == np.argmax(ref_np), (
        f"no-MMALIB top-1 mismatch: DSP={int(np.argmax(dsp_no))}, "
        f"ref={int(np.argmax(ref_np))}"
    )

    # --- Part 2: MMALIB ----------------------------------------------------------
    # Validates all four MMALIB kernels compile and produce a reasonable result.
    # max_diff ≤ 20.0 matches the tolerance in quantized/test_quantized_*.py.
    results_mm = compile_and_run_dsp(
        mod=mod,
        input_data=input_np,
        target_string=get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1",
        execution_mode=dsp_mode,
    )
    dsp_mm = results_mm[result_key].reshape(ref_np.shape).astype(np.float32)
    max_diff = float(np.abs(dsp_mm - ref_np).max())
    assert max_diff <= 20.0, (
        f"MMALIB output diverges from PyTorch quantized reference: "
        f"max_diff={max_diff:.3f} > 20.0"
    )

    record_cycles("mobilenet_v2_pt2e_i8", results_mm.get("c7x_dload_cycles", 0))
