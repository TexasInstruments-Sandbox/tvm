"""Unit tests for the c7x_int8_silu_f32out kernel.

Invokes the kernel via call_extern with known inputs and verifies output
against a numpy reference. Tests are independent of the
FuseQDQToC7xSiluF32Out pass.

c7x_int8_silu_f32out: same self-gated SiLU math as c7x_int8_silu
(out = x_f * sigmoid(x_f), x_f = (in[i] - zx) * sx), but writes the
float32 value directly -- no output zero-point/scale, no clamp/requantize.
Same Taylor-exp + reciprocal-refinement vectorized path as c7x_int8_silu
(see test_silu_kernel.py); only the final store differs.

Because there is no requantize step to absorb 1-ULP float32 differences
between numpy's exact reciprocal and the hardware's seed+Newton-Raphson
approximation, comparisons use a tight np.allclose tolerance rather than
exact equality (unlike test_silu_kernel.py, whose int8 output rounds those
differences away).

Usage:
    pytest test_silu_f32out_kernel.py -v --dsp-mode=c7x_host
    pytest test_silu_f32out_kernel.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import bulk_tail_split, compile_and_run_dsp, get_target_string  # noqa: E402

# ---------------------------------------------------------------------------
# Reference implementation (Taylor exp / reciprocal copied from
# test_silu_kernel.py -- same kernel-side math, just no output rq_f/clamp).
# ---------------------------------------------------------------------------

_FLT_MAX = np.finfo(np.float32).max


def _exp_taylor(x):
    f32 = np.float32
    x = x.astype(f32)
    ln2 = f32(0.693147180559945)
    one_by_ln2 = f32(1.44269504090)
    one_by_6 = f32(0.1666667)
    one_by_24 = f32(0.0416667)
    pkd_one_by_65536 = f32(0.0000152587890625)

    y = one_by_ln2 * x
    yI = np.round(y).astype(np.int32)
    yf = y - yI.astype(f32)

    r1 = yf * ln2
    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r2 * r2
    two_pw_f = f32(1.0) + r1 + r2 * f32(0.5) + r3 * one_by_6 + r4 * one_by_24

    def _pow2_shift16(amt):
        pos = amt > 0
        shift_l = np.left_shift(np.int32(1 << 16), np.clip(amt, 0, 30))
        shift_r = np.right_shift(np.int32(1 << 16), np.clip(-amt, 0, 30))
        return np.where(pos, shift_l, shift_r)

    # Two chained safe rings (each clamped to +/-14, the widest single
    # shift that can't overflow int32) extend the exact range to yI in
    # [-28,28] instead of one unclamped ring -- see c7x_qdq_common.h's
    # exp_taylor for why.
    yI_lo = np.clip(yI, -14, 14)
    excess = yI - yI_lo
    excess_lo = np.clip(excess, -14, 14)

    e_pw_x = two_pw_f * _pow2_shift16(yI_lo).astype(f32) * pkd_one_by_65536
    e_pw_x = e_pw_x * _pow2_shift16(excess_lo).astype(f32) * pkd_one_by_65536
    e_pw_x = np.where(yI < -28, f32(0.0), e_pw_x)
    e_pw_x = np.where(yI > 28, f32(_FLT_MAX), e_pw_x)
    return e_pw_x


def _vec_recip(v):
    f32 = np.float32
    return (f32(1.0) / v.astype(np.float64)).astype(f32)


def _numpy_silu_f32out(inp, zx, sx):
    f32 = np.float32
    n = len(inp)
    nvec8 = bulk_tail_split(n, vec_width=8)
    zx32, sx32 = f32(zx), f32(sx)
    one = f32(1.0)

    x = (inp.astype(f32) - zx32) * sx32
    sig = _vec_recip(_exp_taylor(-x) + one)
    y_bulk = (x * sig)[:nvec8]

    x_tail = (inp[nvec8:].astype(np.float64) - zx) * sx
    y_tail = x_tail / (1.0 + np.exp(-x_tail))

    out = np.empty(n, dtype=np.float32)
    out[:nvec8] = y_bulk
    out[nvec8:] = y_tail.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_silu_f32out_module(n, zx, sx):
    n_v = int(n)
    zx_v = int(zx)
    sx_v = float(sx)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "c7x_int8_silu_f32out",
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", n_v),
                tir.IntImm("int32", zx_v),
                tir.FloatImm("float32", sx_v),
            )

        return te.extern([n_v], [x_t], fcompute, name="silu_f32out_out", dtype="float32")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([n_v], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint="c7x_int8_silu_f32out")
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_silu_f32out(dsp_mode, inp, zx, sx):
    mod = _build_silu_f32out_module(len(inp), zx, sx)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[inp],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_silu_f32out_scalar_tail(dsp_mode):
    """n=37 -- 4 full 4x-unrolled vector groups (32 elements) + 5-element
    scalar tail, no elements through the single-vector remainder path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(20)
    inp = rng.integers(-128, 127, 37, dtype=np.int8)
    zx, sx = 0, 0.05
    ref = _numpy_silu_f32out(inp, zx, sx)
    out, _ = _run_silu_f32out(dsp_mode, inp, zx, sx)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.quick
def test_silu_f32out_asymmetric_zp(dsp_mode):
    """Non-zero zero-point -- tests zp subtraction."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(21)
    inp = rng.integers(-128, 127, 128, dtype=np.int8)
    zx, sx = -10, 0.04
    ref = _numpy_silu_f32out(inp, zx, sx)
    out, _ = _run_silu_f32out(dsp_mode, inp, zx, sx)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.quick
def test_silu_f32out_single_vector_remainder(dsp_mode):
    """n=104 -- 13 vector groups: 12 through the 4x-unrolled main loop, 1
    through the single-vector remainder loop, zero scalar tail (13*8=104)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(22)
    inp = rng.integers(-128, 127, 104, dtype=np.int8)
    zx, sx = 0, 0.06
    ref = _numpy_silu_f32out(inp, zx, sx)
    out, _ = _run_silu_f32out(dsp_mode, inp, zx, sx)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.core
def test_silu_f32out_c2f_block_size(dsp_mode, record_cycles):
    """C=32, H=W=80 (n=204,800) -- a real yolo26n C2f-block SiLU-before-split
    shape (see yolo_head_qdq_movement_fusion.md's silu_f32out motivation)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(23)
    n = 32 * 80 * 80
    inp = rng.integers(-128, 127, n, dtype=np.int8)
    zx, sx = 0, 0.0545687770843505859
    ref = _numpy_silu_f32out(inp, zx, sx)
    out, cycles = _run_silu_f32out(dsp_mode, inp, zx, sx)
    record_cycles("silu_f32out_n204800", cycles)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)
    if cycles:
        print(
            f"\n  c7x_int8_silu_f32out n={n}: {cycles:,} cycles ({cycles / n:.2f} cycles/element)"
        )
