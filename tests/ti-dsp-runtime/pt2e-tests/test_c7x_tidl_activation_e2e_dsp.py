"""End-to-end DSP tests for TIDL activation, avg_pool, and layer_norm fusions.

Validates the full flow:
  float PyTorch model
  → C7xMMAQuantizer (prepare_pt2e / calibrate / convert_pt2e)
  → from_exported_program (Relax QDQ IR)
  → c_static -mcpu=c7x -mmalib=1
      (FuseQDQToTIDLActivation / FuseQDQToC7xAvgPool / FuseQDQToTIDLLayerNorm)
  → tidl_int8_* kernel on c7x_host / c7x_dload

Correctness: DSP output is compared against the PyTorch quantized reference.
Tolerance max_diff ≤ 2 allows ±1 LSB rounding difference (same as MMALIB tests).

Usage:
    cd tests/ti-dsp-runtime
    pytest --rootdir=. pt2e-tests/test_c7x_tidl_activation_e2e_dsp.py \\
        -m quick --dsp-mode=c7x_host -v
    pytest --rootdir=. pt2e-tests/test_c7x_tidl_activation_e2e_dsp.py \\
        -m quick --dsp-mode=c7x_dload -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pt2e_utils import e2e_quantize_and_import, run_and_check


# ---------------------------------------------------------------------------
# Non-linear activation tests (FuseQDQToTIDLActivation)
#
# Each model is Linear → activation → Linear so that both the activation
# input and output are surrounded by quantized layers, giving a clean
# dq → act → q pattern for the fusion pass to match.
# ---------------------------------------------------------------------------


class _ActModel(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.act = act_fn
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_gelu_i8(dsp_mode, record_cycles):
    """gelu int8: C7xMMAQuantizer → tidl_int8_gelu."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = _ActModel(nn.GELU()).eval()
    example = (torch.randn(1, 32),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_gelu_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_silu_i8(dsp_mode, record_cycles):
    """silu int8: C7xMMAQuantizer → tidl_int8_silu."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = _ActModel(nn.SiLU()).eval()
    example = (torch.randn(1, 32),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_silu_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_hardsigmoid_i8(dsp_mode, record_cycles):
    """hardsigmoid int8: C7xMMAQuantizer → tidl_int8_hardsigmoid."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = _ActModel(nn.Hardsigmoid()).eval()
    example = (torch.randn(1, 32),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_hardsigmoid_i8")


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_hardswish_i8(dsp_mode, record_cycles):
    """hardswish int8: C7xMMAQuantizer → tidl_int8_hardswish."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model = _ActModel(nn.Hardswish()).eval()
    example = (torch.randn(1, 32),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_hardswish_i8")


# ---------------------------------------------------------------------------
# Average pooling tests (FuseQDQToC7xAvgPool)
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_global_avg_pool_i8(dsp_mode, record_cycles):
    """Global avg pool (adaptive 1×1) int8: → c7x_int8_global_avg_pool."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1)

        def forward(self, x):
            return F.adaptive_avg_pool2d(self.conv(x), (1, 1))

    model = M().eval()
    example = (torch.randn(1, 8, 16, 16),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_global_avg_pool_i8"
    )


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_avg_pool2d_i8(dsp_mode, record_cycles):
    """Spatial avg_pool2d int8: → c7x_int8_avg_pool."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1)

        def forward(self, x):
            return F.avg_pool2d(self.conv(x), kernel_size=3, stride=1, padding=1)

    model = M().eval()
    example = (torch.randn(1, 8, 16, 16),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_avg_pool2d_i8"
    )


# ---------------------------------------------------------------------------
# Layer norm test (FuseQDQToTIDLLayerNorm)
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.c7x_only
def test_e2e_layer_norm_i8(dsp_mode, record_cycles):
    """layer_norm int8: C7xMMAQuantizer → tidl_int8_layer_norm.

    The kernel runs float32 internally (dequant → normalize → requant),
    so accuracy should be the same as the activation tests (max_diff ≤ 2).
    """
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 32)
            self.ln = nn.LayerNorm(32)
            self.out = nn.Linear(32, 16)

        def forward(self, x):
            return self.out(self.ln(self.fc(x)))

    model = M().eval()
    example = (torch.randn(1, 32),)
    mod, input_np, ref_np = e2e_quantize_and_import(model, example)
    run_and_check(
        mod, input_np, ref_np, dsp_mode, record_cycles, "e2e_layer_norm_i8"
    )
