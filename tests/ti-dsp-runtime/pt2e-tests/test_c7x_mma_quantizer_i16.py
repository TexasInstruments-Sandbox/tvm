"""Pure-Python tests for the int16 PT2E quantization pipeline.

Validates the end-to-end int16 flow from C7xMMAQuantizer → from_exported_program
→ MMALIB i16 QDQ fusion passes, without requiring a DSP or TI toolchain.

Two kinds of checks:
  1. Annotation / PyTorch-level — C7xMMAQuantizer produces correct int16 Q/DQ graphs
  2. TVM import + fusion — from_exported_program round-trips correctly and the
     FuseMMALIBQDQ*I16 passes produce the expected extern calls

Run with:
    cd tests/ti-dsp-runtime
    pytest --rootdir=. pt2e-tests/test_c7x_mma_quantizer_i16.py -m quick -v
"""

import pytest
import torch
import torch.nn as nn
from pt2e_utils import e2e_quantize_and_import, quantize_pt2e

import tvm
from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer
from tvm.relax.transform.ti_mmalib_passes import get_mmalib_qdq_passes

# ---------------------------------------------------------------------------
# Shared test models (dimensions satisfy i16 alignment: multiples of 16)
# ---------------------------------------------------------------------------


class ConvI16(nn.Module):
    """Regular conv2d — exercises FuseMMALIBQDQConv2dI16."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class DwConvI16(nn.Module):
    """Depthwise conv2d (3×3 only for i16) — exercises FuseMMALIBQDQDwConv2dI16."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1, groups=32)

    def forward(self, x):
        return self.conv(x)


class LinearI16(nn.Module):
    """Linear layer — exercises FuseMMALIBQDQFCI16."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        return self.fc(x)


class ResidualI16(nn.Module):
    """Conv + skip connection — exercises FuseInt16ResidualAdd."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1)

    def forward(self, x):
        return self.conv(x) + x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_i16_mmalib_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply the full MMALIB QDQ pass list (int8 + int16) to a Relax module."""
    for p in get_mmalib_qdq_passes():
        mod = p(mod)
    return mod


def _fused_kernel_names(mod: tvm.IRModule) -> list[str]:
    """Return name hints of all PrimFuncs the pipeline produced."""
    return [str(gv.name_hint) for gv in mod.functions]


# ---------------------------------------------------------------------------
# 1. Annotation: int16 Q/DQ graph produced correctly
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_int16_produces_int16_quantize_on_activations():
    """C7xMMAQuantizer(int16) annotates activations with torch.int16 dtype."""
    q = quantize_pt2e(ConvI16(), (torch.randn(1, 32, 8, 8),), C7xMMAQuantizer("int16"))
    int16_quant_found = any(
        "quantize_per_tensor" in str(n.target)
        and len(n.args) >= 6
        and n.args[-1] == torch.int16
        for n in q.graph.nodes
        if n.op == "call_function"
    )
    assert int16_quant_found, "no int16 quantize_per_tensor node found"


@pytest.mark.quick
def test_int16_weight_zero_point_is_zero():
    """Weight zero_points must be 0 for int16 (MMALIB symmetric-weight requirement)."""
    q = quantize_pt2e(ConvI16(), (torch.randn(1, 32, 8, 8),), C7xMMAQuantizer("int16"))
    found = False
    for node in q.graph.nodes:
        if "dequantize_per_channel" in str(node.target):
            zp_node = node.args[2]
            assert zp_node.op == "get_attr"
            zp = getattr(q, zp_node.target)
            assert torch.all(zp == 0), f"weight zero_point not zero: {zp}"
            found = True
    assert found, "no dequantize_per_channel node found"


@pytest.mark.quick
def test_int16_activation_zero_point_is_zero():
    """Activation zero_points must be 0 (int16 symmetric-only requirement)."""
    q = quantize_pt2e(ConvI16(), (torch.randn(1, 32, 8, 8),), C7xMMAQuantizer("int16"))
    for node in q.graph.nodes:
        if "quantize_per_tensor" in str(node.target) and node.op == "call_function":
            zp_val = node.args[2]
            # zp may be a get_attr node or a literal 0
            if hasattr(zp_val, "op") and zp_val.op == "get_attr":
                zp = getattr(q, zp_val.target)
                assert int(zp) == 0, f"activation zero_point not zero: {zp}"
            else:
                assert zp_val == 0, f"activation zero_point not zero: {zp_val}"


# ---------------------------------------------------------------------------
# 2. TVM import: from_exported_program handles int16 zero_points correctly
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_int16_tvm_import_succeeds():
    """from_exported_program must not crash on an int16 PT2E graph.

    Before the zero_point dtype fix, TVM's relax.dequantize rejected int16
    zero_points with 'datatype should be int8 or float16'.
    """
    mod, _, _ = e2e_quantize_and_import(
        ConvI16(), (torch.randn(1, 32, 8, 8),), dtype="int16"
    )
    assert "main" in [str(gv.name_hint) for gv in mod.functions]


@pytest.mark.quick
def test_int16_tvm_import_zero_point_is_int8():
    """Zero_points in the imported Relax module must be int8 (TVM qdq.cc constraint).

    The exported_program_translator must cast the PyTorch int16 zero_points to
    int8; the value is always 0 for symmetric schemes so this is lossless.
    """
    mod, _, _ = e2e_quantize_and_import(
        ConvI16(), (torch.randn(1, 32, 8, 8),), dtype="int16"
    )
    # Walk the function body and check every Constant used as a zero_point
    # argument to relax.dequantize or relax.quantize.
    dequantize_op = relax.op.dequantize
    quantize_op = relax.op.quantize
    bad_zp_dtypes = set()

    def _check_func(func):
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if val.op not in (dequantize_op, quantize_op):
                    continue
                zp = val.args[2]
                if isinstance(zp, relax.Constant):
                    dtype = str(zp.data.dtype)
                    if dtype not in ("int8", "float16"):
                        bad_zp_dtypes.add(dtype)

    for gv, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _check_func(func)

    assert not bad_zp_dtypes, (
        f"zero_point constants with invalid dtype found: {bad_zp_dtypes}. "
        f"Expected int8 only."
    )


# ---------------------------------------------------------------------------
# 3. Fusion passes: correct i16 MMALIB kernels are emitted
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_int16_conv2d_fusion_fires():
    """FuseMMALIBQDQConv2dI16 must produce a mmalib_conv2d function."""
    mod, _, _ = e2e_quantize_and_import(
        ConvI16(), (torch.randn(1, 32, 8, 8),), dtype="int16"
    )
    mod = _run_i16_mmalib_passes(mod)
    names = _fused_kernel_names(mod)
    i16_conv = [n for n in names if "mmalib_conv2d" in n]
    assert i16_conv, (
        f"FuseMMALIBQDQConv2dI16 did not fire. PrimFuncs: {names}"
    )


@pytest.mark.quick
def test_int16_linear_fusion_fires():
    """FuseMMALIBQDQFCI16 must produce a mmalib_fc_i16 function."""
    mod, _, _ = e2e_quantize_and_import(
        LinearI16(), (torch.randn(1, 64),), dtype="int16"
    )
    mod = _run_i16_mmalib_passes(mod)
    names = _fused_kernel_names(mod)
    i16_fc = [n for n in names if "mmalib_fc_i16" in n or "mmalib_matmul_bias_i16" in n]
    assert i16_fc, (
        f"FuseMMALIBQDQFCI16 did not fire. PrimFuncs: {names}"
    )


@pytest.mark.quick
def test_int16_residual_add_fusion_fires():
    """FuseInt16ResidualAdd must produce an i16_residual_add function."""
    mod, _, _ = e2e_quantize_and_import(
        ResidualI16(), (torch.randn(1, 32, 8, 8),), dtype="int16"
    )
    mod = _run_i16_mmalib_passes(mod)
    names = _fused_kernel_names(mod)
    res_add = [n for n in names if "residual_add" in n and "i16" in n]
    assert res_add, (
        f"FuseInt16ResidualAdd did not fire. PrimFuncs: {names}"
    )


@pytest.mark.quick
def test_int16_dwconv2d_fusion_fires():
    """FuseMMALIBQDQDwConv2dI16 must produce a mmalib_dwconv2d function."""
    mod, _, _ = e2e_quantize_and_import(
        DwConvI16(), (torch.randn(1, 32, 8, 8),), dtype="int16"
    )
    mod = _run_i16_mmalib_passes(mod)
    names = _fused_kernel_names(mod)
    dwconv = [n for n in names if "mmalib_dwconv2d" in n or "mmalib_depthwise" in n]
    assert dwconv, (
        f"FuseMMALIBQDQDwConv2dI16 did not fire. PrimFuncs: {names}"
    )


@pytest.mark.quick
def test_int16_5x5_dwconv2d_not_fused():
    """5×5 depthwise i16 must NOT be fused (MMALIB-882: unsupported kernel size).

    The check function rejects it; the layer falls through to float computation.
    """

    class DwConv5x5(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 32, 5, padding=2, groups=32)

        def forward(self, x):
            return self.conv(x)

    mod, _, _ = e2e_quantize_and_import(DwConv5x5(), (torch.randn(1, 32, 8, 8),), dtype="int16")
    mod = _run_i16_mmalib_passes(mod)
    names = _fused_kernel_names(mod)
    # 5×5 depthwise should NOT produce an i16 depthwise kernel
    i16_dw = [n for n in names if "mmalib_dwconv2d_i16" in n or "mmalib_depthwise_conv2d_i16" in n]
    assert not i16_dw, (
        f"5×5 i16 depthwise was incorrectly fused (MMALIB-882 should block it). "
        f"PrimFuncs: {names}"
    )


@pytest.mark.quick
def test_int16_i8_passes_not_triggered_on_int16_graph():
    """Int8 fusion passes must not match an int16 QDQ graph.

    FuseMMALIBQDQConv2d (int8) should not fire on a pure int16 model.
    """
    mod, _, _ = e2e_quantize_and_import(
        ConvI16(), (torch.randn(1, 32, 8, 8),), dtype="int16"
    )
    mod = _run_i16_mmalib_passes(mod)
    names = _fused_kernel_names(mod)
    # mmalib_conv2d (no suffix) would be the int8 kernel name
    i8_conv = [n for n in names if n == "mmalib_conv2d"]
    assert not i8_conv, (
        f"Int8 conv2d pass fired on int16 graph (check function too permissive). "
        f"PrimFuncs: {names}"
    )
