"""
Int16 residual add fusion tests — both operand orders.

FuseInt16ResidualAdd matches:
    dequantize(x_i16) + dequantize(skip_i16) -> [relu] -> quantize(out_i16)

For int16, all zero-points must be 0 (symmetric quantization enforced by
C7xMMAQuantizer(dtype="int16")).  Both operand orders are tested since
TVM's DFPattern is not commutative.

Usage:
    pytest test_mmalib_residual_add_i16_dsp.py -v --dsp-mode=c7x_host
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax
from tvm.relax import TensorStructInfo

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _make_residual_add_i16_model(
    shape: tuple,
    x_scale: float,
    skip_scale: float,
    o_scale: float,
    has_relu: bool,
    swapped: bool,
    seed: int = 42,
):
    """Build a Relax model with a single int16 quantized residual add.

    All zero-points are 0 — int16 activation quantization is always
    symmetric in C7xMMAQuantizer(dtype="int16").

    When ``swapped=False``: add(x_dq, skip_dq) — standard order.
    When ``swapped=True``:  add(skip_dq, x_dq) — skip-first order.
    """
    rng = np.random.default_rng(seed)
    x_data = rng.integers(-1000, 1000, size=shape, dtype=np.int16)
    skip_data = rng.integers(-1000, 1000, size=shape, dtype=np.int16)

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", TensorStructInfo(shape, "int16"))
    skip_var = relax.Var("skip", TensorStructInfo(shape, "int16"))

    # TVM's dequantize requires int8 zero-point; zp=0 in int8 is correct here.
    x_scale_c = relax.Constant(np.array(x_scale, dtype=np.float32))
    x_zp_c = relax.Constant(np.array(0, dtype=np.int8))
    skip_scale_c = relax.Constant(np.array(skip_scale, dtype=np.float32))
    skip_zp_c = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_c = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_c = relax.Constant(np.array(0, dtype=np.int8))

    with bb.function("main", [x_var, skip_var], attrs={"num_input": 2}):
        with bb.dataflow():
            x_dq = bb.emit(relax.op.dequantize(x_var, x_scale_c, x_zp_c))
            skip_dq = bb.emit(relax.op.dequantize(skip_var, skip_scale_c, skip_zp_c))

            if swapped:
                add_out = bb.emit(relax.op.add(skip_dq, x_dq))
            else:
                add_out = bb.emit(relax.op.add(x_dq, skip_dq))

            if has_relu:
                add_out = bb.emit(relax.op.nn.relu(add_out))

            result = bb.emit(
                relax.op.quantize(add_out, o_scale_c, o_zp_c, out_dtype="int16")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize(), x_data, skip_data


def _numpy_residual_add_i16(
    x_data, skip_data, x_scale, skip_scale, o_scale, has_relu
):
    """Float-domain reference for int16 residual add (symmetric, zp=0)."""
    x_f = x_data.astype(np.float64) * x_scale
    skip_f = skip_data.astype(np.float64) * skip_scale
    out_f = x_f + skip_f
    if has_relu:
        out_f = np.maximum(0.0, out_f)
    out_q = np.round(out_f / o_scale)
    return np.clip(out_q, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_residual_add_i16(dsp_mode, swapped, has_relu, record_cycles):
    """Compile and run an int16 residual add model."""
    shape = (1, 16, 8, 8)
    x_scale, skip_scale, o_scale = 0.001, 0.002, 0.0015

    mod, x_data, skip_data = _make_residual_add_i16_model(
        shape, x_scale, skip_scale, o_scale, has_relu=has_relu, swapped=swapped
    )
    ref = _numpy_residual_add_i16(x_data, skip_data, x_scale, skip_scale, o_scale, has_relu)

    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=(x_data, skip_data),
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_out = results.get(result_key)
    assert dsp_out is not None, f"No {result_key} in results"
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(shape)

    label = (
        f"residual_add_i16_{'relu_' if has_relu else ''}{'swapped' if swapped else 'normal'}"
    )
    record_cycles(label, results.get("c7x_dload_cycles", 0))

    return dsp_out_i16, ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_residual_add_i16_order_a(dsp_mode, record_cycles):
    """Standard order: add(x_dq, skip_dq) → relu → quantize [int16]."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add_i16(
        dsp_mode, swapped=False, has_relu=True, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"i16 residual add order-A: max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_i16_order_b(dsp_mode, record_cycles):
    """Swapped order: add(skip_dq, x_dq) → relu → quantize [int16].

    Verifies FuseInt16ResidualAdd handles skip-first operand order.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add_i16(
        dsp_mode, swapped=True, has_relu=True, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"i16 residual add order-B (swapped): max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_i16_no_relu(dsp_mode, record_cycles):
    """Standard order without relu: add(x_dq, skip_dq) → quantize [int16]."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add_i16(
        dsp_mode, swapped=False, has_relu=False, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"i16 residual add no-relu: max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_i16_both_orders_agree(dsp_mode, record_cycles):
    """Order A and Order B must produce bit-identical results [int16]."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    out_a, _ = _run_residual_add_i16(
        dsp_mode, swapped=False, has_relu=True, record_cycles=record_cycles
    )
    out_b, _ = _run_residual_add_i16(
        dsp_mode, swapped=True, has_relu=True, record_cycles=record_cycles
    )
    diff = np.abs(out_a.astype(np.int32) - out_b.astype(np.int32))
    assert diff.max() == 0, f"i16 residual add: order A vs B differ, max_diff={diff.max()}"
