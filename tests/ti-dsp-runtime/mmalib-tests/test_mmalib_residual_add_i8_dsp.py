"""
Int8 residual add fusion tests — both operand orders.

FuseInt8ResidualAdd matches:
    dequantize(x) + dequantize(skip) -> [relu] -> quantize

TVM's DFPattern is NOT commutative, so the add operand order matters.
Some models emit add(x, skip), others emit add(skip, x).  This test
verifies that both orders are fused correctly, producing the same
integer-arithmetic result.

Usage:
    pytest test_mmalib_residual_add_i8_dsp.py -v --dsp-mode=c7x_host
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
# Model builders
# ---------------------------------------------------------------------------


def _make_residual_add_model(
    shape: tuple,
    x_scale: float,
    x_zp: int,
    skip_scale: float,
    skip_zp: int,
    o_scale: float,
    o_zp: int,
    has_relu: bool,
    swapped: bool,
    seed: int = 42,
):
    """Build a Relax model containing a single quantized residual add.

    When ``swapped=False`` the add is emitted as add(x_dq, skip_dq).
    When ``swapped=True``  the add is emitted as add(skip_dq, x_dq).

    This exercises both pattern variants in FuseInt8ResidualAdd.
    """
    rng = np.random.default_rng(seed)
    x_data = rng.integers(-10, 10, size=shape, dtype=np.int8)
    skip_data = rng.integers(-10, 10, size=shape, dtype=np.int8)

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", TensorStructInfo(shape, "int8"))
    skip_var = relax.Var("skip", TensorStructInfo(shape, "int8"))

    x_scale_c = relax.Constant(np.array(x_scale, dtype=np.float32))
    x_zp_c = relax.Constant(np.array(x_zp, dtype=np.int8))
    skip_scale_c = relax.Constant(np.array(skip_scale, dtype=np.float32))
    skip_zp_c = relax.Constant(np.array(skip_zp, dtype=np.int8))
    o_scale_c = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_c = relax.Constant(np.array(o_zp, dtype=np.int8))

    with bb.function("main", [x_var, skip_var], attrs={"num_input": 2}):
        with bb.dataflow():
            x_dq = bb.emit(relax.op.dequantize(x_var, x_scale_c, x_zp_c))
            skip_dq = bb.emit(relax.op.dequantize(skip_var, skip_scale_c, skip_zp_c))

            # Emit add in the requested operand order
            if swapped:
                # add(skip_dq, x_dq) — skip connection is first arg
                add_out = bb.emit(relax.op.add(skip_dq, x_dq))
            else:
                # add(x_dq, skip_dq) — x is first arg (standard order)
                add_out = bb.emit(relax.op.add(x_dq, skip_dq))

            if has_relu:
                add_out = bb.emit(relax.op.nn.relu(add_out))

            result = bb.emit(relax.op.quantize(add_out, o_scale_c, o_zp_c, out_dtype="int8"))
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize(), x_data, skip_data


def _numpy_residual_add_i8(
    x_data,
    skip_data,
    x_scale,
    x_zp,
    skip_scale,
    skip_zp,
    o_scale,
    o_zp,
    has_relu,
):
    """Reference implementation: float-domain residual add with requantize."""
    # Dequantize both inputs to float
    x_f = (x_data.astype(np.float64) - x_zp) * x_scale
    skip_f = (skip_data.astype(np.float64) - skip_zp) * skip_scale

    # Add (order-independent)
    out_f = x_f + skip_f

    if has_relu:
        out_f = np.maximum(0.0, out_f)

    # Requantize to int8
    out_q = np.round(out_f / o_scale) + o_zp
    return np.clip(out_q, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_residual_add(dsp_mode, swapped, has_relu, record_cycles):
    """Compile and run a residual add model, return (dsp_output, ref_output)."""
    shape = (1, 8, 16, 16)  # small enough to run quickly on any device
    x_scale, x_zp = 0.04, -3
    skip_scale, skip_zp = 0.06, 2
    o_scale, o_zp = 0.05, 0

    mod, x_data, skip_data = _make_residual_add_model(
        shape,
        x_scale,
        x_zp,
        skip_scale,
        skip_zp,
        o_scale,
        o_zp,
        has_relu=has_relu,
        swapped=swapped,
    )

    ref = _numpy_residual_add_i8(
        x_data,
        skip_data,
        x_scale,
        x_zp,
        skip_scale,
        skip_zp,
        o_scale,
        o_zp,
        has_relu=has_relu,
    )

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

    dsp_out_i8 = dsp_out.astype(np.int8).reshape(shape)

    label = f"residual_add_i8_{'relu_' if has_relu else ''}{'swapped' if swapped else 'normal'}"
    record_cycles(label, results.get("c7x_dload_cycles", 0))

    return dsp_out_i8, ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_residual_add_i8_order_a(dsp_mode, record_cycles):
    """Standard order: add(x_dq, skip_dq) → relu → quantize."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add(
        dsp_mode, swapped=False, has_relu=True, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"Order-A residual add: max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_i8_order_b(dsp_mode, record_cycles):
    """Swapped order: add(skip_dq, x_dq) → relu → quantize.

    This verifies the fix for models where the skip-connection tensor is
    placed as the *first* argument of the add node.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add(
        dsp_mode, swapped=True, has_relu=True, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"Order-B (swapped) residual add: max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_i8_no_relu_order_a(dsp_mode, record_cycles):
    """Standard order without relu: add(x_dq, skip_dq) → quantize."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add(
        dsp_mode, swapped=False, has_relu=False, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"Order-A no-relu residual add: max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_i8_no_relu_order_b(dsp_mode, record_cycles):
    """Swapped order without relu: add(skip_dq, x_dq) → quantize."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    dsp_out, ref = _run_residual_add(
        dsp_mode, swapped=True, has_relu=False, record_cycles=record_cycles
    )
    diff = np.abs(dsp_out.astype(np.int32) - ref.astype(np.int32))
    assert diff.max() <= 1, f"Order-B (swapped) no-relu residual add: max_diff={diff.max()}"


@pytest.mark.quick
def test_residual_add_both_orders_agree(dsp_mode, record_cycles):
    """Order A and Order B must produce bit-identical results.

    Since addition is commutative and the fixed-point arithmetic tracks
    per-operand scale/zp independently, the two orderings must agree.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")

    out_a, _ = _run_residual_add(
        dsp_mode, swapped=False, has_relu=True, record_cycles=record_cycles
    )
    out_b, _ = _run_residual_add(dsp_mode, swapped=True, has_relu=True, record_cycles=record_cycles)
    diff = np.abs(out_a.astype(np.int32) - out_b.astype(np.int32))
    assert diff.max() == 0, f"Order A and Order B differ: max_diff={diff.max()}"
