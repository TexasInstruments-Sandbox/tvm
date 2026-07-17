"""End-to-end tests for C7xMMAQuantizer → TVM c_static MMALIB pipeline.

Validates the full flow:
  float PyTorch model
  → C7xMMAQuantizer (prepare_pt2e / calibrate / convert_pt2e)
  → from_exported_program (Relax QDQ IR)
  → c_static -mcpu=c7x -mmalib=1   (FuseMMALIBQDQ* passes)
  → MMALIB kernel on c7x_host / c7x_dload

Correctness: DSP int8 output is compared against the PyTorch quantized model
running on CPU. Both use identical quantization parameters, so the only allowed
divergence is integer rounding: max_diff ≤ 2.

Usage:
    cd tests/ti-dsp-runtime
    pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_host -v
    pytest --rootdir=. pt2e-tests/ -m quick --dsp-mode=c7x_dload -v
"""

import pytest
import torch
import torch.nn as nn
from pt2e_utils import e2e_quantize_and_import as _e2e_quantize_and_import
from pt2e_utils import run_and_check as _run_and_check

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_conv2d_i8(dsp_mode, record_cycles):
    """Conv2d int8: C7xMMAQuantizer → MMALIB mmalib_conv2d_i8."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1)).eval()
    example = (torch.randn(1, 3, 56, 56),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example)
    _run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_conv2d_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_depthwise_conv2d_i8(dsp_mode, record_cycles):
    """Depthwise conv2d int8: C7xMMAQuantizer → MMALIB mmalib_depthwise_conv2d_i8."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8)).eval()
    example = (torch.randn(1, 8, 56, 56),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example)
    _run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_dwconv2d_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_linear_i8(dsp_mode, record_cycles):
    """Linear (with bias) int8: C7xMMAQuantizer → MMALIB mmalib_matmul_bias_i8."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = nn.Sequential(nn.Linear(64, 128)).eval()
    example = (torch.randn(1, 64),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example)
    _run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_linear_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_linear_i8_no_bias(dsp_mode, record_cycles):
    """Linear (no bias) int8: C7xMMAQuantizer → MMALIB mmalib_matmul_i8."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = nn.Sequential(nn.Linear(64, 128, bias=False)).eval()
    example = (torch.randn(1, 64),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example)
    _run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_linear_i8_no_bias")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_residual_add_i8(dsp_mode, record_cycles):
    """Residual add int8: C7xMMAQuantizer → c7x_int8_residual_add_relu (no-relu variant)."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1)

        def forward(self, x):
            return self.conv(x) + x

    model = ResidualBlock().eval()
    example = (torch.randn(1, 8, 16, 16),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example)
    _run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_residual_add_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_linear_3d_i8(dsp_mode, record_cycles):
    """3D linear int8 (LLM-style [batch, seq, K]): C7xMMAQuantizer → MMALIB FC."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = nn.Sequential(nn.Linear(64, 128)).eval()
    example = (torch.randn(1, 4, 64),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example)
    _run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_linear_3d_i8")


# ---------------------------------------------------------------------------
# Int16 end-to-end tests
#
# Tolerance is higher than int8 because uint8 scale/shift requantization has
# coarser precision for int16 accumulators (error scales with √K).
# max_diff=10 is conservative for typical layer sizes; single layers are ≤6.
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_conv2d_i16(dsp_mode, record_cycles):
    """Conv2d int16: C7xMMAQuantizer(int16) → mmalib_conv2d_i16."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    # C_in and C_out must be multiples of 16 (MMA_SIZE_I16)
    model = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1)).eval()
    example = (torch.randn(1, 32, 28, 28),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example, dtype="int16")
    _run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles,
        "e2e_conv2d_i16", max_diff=10,
    )


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_depthwise_conv2d_i16(dsp_mode, record_cycles):
    """Depthwise conv2d int16 (3×3 only): C7xMMAQuantizer(int16) → mmalib_depthwise_conv2d_i16."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    # MMALIB only supports 3×3 for int16 depthwise (MMALIB-882)
    model = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32)).eval()
    example = (torch.randn(1, 32, 28, 28),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example, dtype="int16")
    _run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles,
        "e2e_dwconv2d_i16", max_diff=10,
    )


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_linear_i16(dsp_mode, record_cycles):
    """Linear int16: C7xMMAQuantizer(int16) → mmalib_matmul_bias_i16."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    # K and N must be multiples of 16 (MMA_SIZE_I16)
    model = nn.Sequential(nn.Linear(64, 64)).eval()
    example = (torch.randn(1, 64),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example, dtype="int16")
    _run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles,
        "e2e_linear_i16", max_diff=10,
    )


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_residual_add_i16(dsp_mode, record_cycles):
    """Residual add int16: C7xMMAQuantizer(int16) → c7x_int16_residual_add_relu."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 32, 3, padding=1)

        def forward(self, x):
            return self.conv(x) + x

    model = ResidualBlock().eval()
    example = (torch.randn(1, 32, 16, 16),)
    mod, input_np, ref_np = _e2e_quantize_and_import(model, example, dtype="int16")
    _run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles,
        "e2e_residual_add_i16", max_diff=10,
    )
