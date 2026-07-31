"""Unit tests for the c7x_int8_silu kernel.

Invokes the kernel via call_extern with known inputs and verifies output
against a numpy reference. Tests are independent of the
FuseQDQToC7xActivation pass.

c7x_int8_silu: quantized SiLU  out[i] = quant(x_f * sigmoid(x_f))
  where x_f = (in[i] - zx) * sx, sigmoid(x_f) = 1 / (1 + exp(-x_f))

The vectorized kernel (src/runtime/ti_dsp/kernels/c7x_activation.cpp) has
no vectorized transcendental intrinsic to call, so exp(x) is computed via a
4th-order Taylor-series polynomial with range reduction, and the final
1/(1+exp(-x)) division is computed via a hardware reciprocal-approximation
instruction refined by two Newton-Raphson iterations (a plain `/` on the
vector float type was tried first and rejected -- it compiles to eight
sequential scalar divide-subroutine calls per vector op instead of a real
vector instruction). Same bulk/tail split convention as
test_activation_kernels.py's hardswish reference: the vectorized bulk
(n // 8 * 8 elements) must reproduce the kernel's own float32 arithmetic
(the Taylor exp + reciprocal-refinement pipeline) rather than an idealized
scalar formula, while the true scalar tail (n % 8 remainder) uses the
kernel's plain expf()-based scalar path.

Usage:
    pytest test_silu_kernel.py -v --dsp-mode=c7x_host
    pytest test_silu_kernel.py -v --dsp-mode=c7x_dload
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
# Reference implementation
# ---------------------------------------------------------------------------

_FLT_MAX = np.finfo(np.float32).max


def _exp_taylor(x):
    """4th-order Taylor-series exp with range reduction, matching the
    kernel's exp_taylor() bit-for-bit in float32 (round-to-nearest integer
    split via np.round, matching hardware VSPINT)."""
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

    pos = yI > 0
    shift_l = np.left_shift(np.int32(1 << 16), np.clip(yI, 0, 30))
    shift_r = np.right_shift(np.int32(1 << 16), np.clip(-yI, 0, 30))
    shift = np.where(pos, shift_l, shift_r)

    e_pw_x = two_pw_f * shift.astype(f32) * pkd_one_by_65536
    e_pw_x = np.where(yI < -16, f32(0.0), e_pw_x)
    e_pw_x = np.where(yI > 14, f32(_FLT_MAX), e_pw_x)
    return e_pw_x


def _vec_recip(v):
    """__recip (~8-bit mantissa seed) + 2 Newton-Raphson iterations, same
    structure as the kernel's vec_recip(). Converges to full float32
    precision regardless of the exact hardware seed (quadratic convergence
    from any correctly-exponented seed), so a plain accurate reciprocal in
    float32 is a faithful proxy for the hardware result."""
    f32 = np.float32
    return (f32(1.0) / v.astype(np.float64)).astype(f32)


def _numpy_silu(inp, zx, sx, zy, sy):
    f32 = np.float32
    n = len(inp)
    nvec8 = bulk_tail_split(n, vec_width=8)
    zx32, sx32, zy32 = f32(zx), f32(sx), f32(zy)
    one = f32(1.0)
    lo, hi, invsy = f32(-128.0), f32(127.0), one / f32(sy)

    x = (inp.astype(f32) - zx32) * sx32
    sig = _vec_recip(_exp_taylor(-x) + one)
    y = x * sig

    q_bulk = np.clip(y[:nvec8] * invsy + zy32, lo, hi)
    bulk_out = np.round(q_bulk).astype(np.int32)  # ties-to-even, matches __float_to_int

    x_tail = inp[nvec8:].astype(np.float64)
    y_tail = (x_tail - zx) * sx
    y_tail = y_tail / (1.0 + np.exp(-y_tail))
    tail_out = np.trunc(y_tail / float(sy) + 0.5).astype(np.int32) + zy

    out = np.empty(n, dtype=np.int32)
    out[:nvec8] = bulk_out
    out[nvec8:] = tail_out
    return np.clip(out, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_silu_module(n, zx, sx, zy, sy):
    n_v = int(n)
    zx_v = int(zx)
    sx_v = float(sx)
    zy_v = int(zy)
    sy_v = float(sy)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "c7x_int8_silu",
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", n_v),
                tir.IntImm("int32", zx_v),
                tir.FloatImm("float32", sx_v),
                tir.IntImm("int32", zy_v),
                tir.FloatImm("float32", sy_v),
            )

        return te.extern([n_v], [x_t], fcompute, name="silu_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([n_v], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint="c7x_int8_silu")
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_silu(dsp_mode, inp, zx, sx, zy, sy):
    mod = _build_silu_module(len(inp), zx, sx, zy, sy)
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
def test_silu_scalar_tail(dsp_mode):
    """n=37 -- 4 full 4x-unrolled vector groups (32 elements) + 5-element
    scalar tail, no elements through the single-vector remainder path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(10)
    inp = rng.integers(-128, 127, 37, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.05, 0, 0.05
    ref = _numpy_silu(inp, zx, sx, zy, sy)
    out, _ = _run_silu(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_silu_asymmetric_zp(dsp_mode):
    """Non-zero zero-points -- tests zp subtraction and output offset."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(11)
    inp = rng.integers(-128, 127, 128, dtype=np.int8)
    zx, sx, zy, sy = -10, 0.04, 5, 0.03
    ref = _numpy_silu(inp, zx, sx, zy, sy)
    out, _ = _run_silu(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_silu_single_vector_remainder(dsp_mode):
    """n=104 -- 13 vector groups: 12 through the 4x-unrolled main loop, 1
    through the single-vector remainder loop, zero scalar tail (13*8=104)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(12)
    inp = rng.integers(-128, 127, 104, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.06, 0, 0.045
    ref = _numpy_silu(inp, zx, sx, zy, sy)
    out, _ = _run_silu(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_silu_mid_size(dsp_mode):
    """n=94_080 (56×56×30) -- exercises the inner vectorized loop."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(13)
    n = 56 * 56 * 30
    inp = rng.integers(-128, 127, n, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.03, 0, 0.02
    ref = _numpy_silu(inp, zx, sx, zy, sy)
    out, _ = _run_silu(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_silu_large(dsp_mode, record_cycles):
    """n=200_704 (112×112×16) -- same size as hardswish's largest test, for
    a directly comparable cycles/element figure between the two kernels."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(14)
    n = 112 * 112 * 16
    inp = rng.integers(-128, 127, n, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.025, 0, 0.018
    ref = _numpy_silu(inp, zx, sx, zy, sy)
    out, cycles = _run_silu(dsp_mode, inp, zx, sx, zy, sy)
    record_cycles("silu_n200704", cycles)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        print(f"\n  c7x_int8_silu n={n}: {cycles:,} cycles ({cycles / n:.2f} cycles/element)")


# Real (zx=0, sx, zy=0, sy≈sx) values pulled from a compiled yolov8s model's
# actual c7x_int8_silu call sites (57 sites, zero_point always 0/symmetric):
# sx ranged [0.045304, 0.526710] -> dequantized input magnitude in [5.75, 66.9].
_REAL_YOLO_SCALES = [0.045304, 0.09, 0.25, 0.526710]


@pytest.mark.core
@pytest.mark.parametrize("sx", _REAL_YOLO_SCALES)
def test_silu_real_yolo_scales(dsp_mode, sx):
    """Full int8 input range (-128..127) against real scales observed in a
    compiled yolov8s model -- the actual operating conditions this kernel
    was optimized for."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    inp = np.arange(-128, 128, dtype=np.int8)
    zx, zy, sy = 0, 0, sx
    ref = _numpy_silu(inp, zx, sx, zy, sy)
    out, _ = _run_silu(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
