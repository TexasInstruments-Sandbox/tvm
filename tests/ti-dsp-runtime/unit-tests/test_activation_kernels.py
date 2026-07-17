"""Unit tests for c7x_int8_hardswish and tidl_int8_channel_scale_multiply kernels.

Invokes each kernel via call_extern with known inputs and verifies output against
a numpy reference.  Tests are independent of the FuseQDQToTIDLActivation pass.

c7x_int8_hardswish: quantized hardswish  out[i] = quant(x_f * clip(x_f/6+0.5,0,1))
  where x_f = (in[i] - zx) * sx

tidl_int8_channel_scale_multiply: SE-block excitation × feature-map
  out[c][j] = sat_i8(round((exc[c]-ze)*se*(fm[c*HW+j]-zf)*sf/so) + zo)

Bug found and fixed (2026-07-17): all 8 tests below used to fail on c7x_host
(smaller, but nonzero, mismatches on c7x_dload hardware too). Root cause was
in the kernel, not the test: tidl_activation_wrappers.cpp gated
`#include <c7x.h>` on `#ifdef __C7524__` -- but on the c7x_host g++
toolchain, __C7524__ is defined *by* <c7x.h> itself, so the guard was always
false and the header was never included, silently compiling out both
hardswish_vec and channel_scale_multiply_vec. c7x_host therefore ran the
scalar float rq_f fallback for every element, not the SE + Q13/float
vectorized path -- a completely different algorithm from what these
references modeled. This is the same chicken-and-egg #include bug already
fixed elsewhere (see c7x_quantize.cpp / c7x_avgpool_wrappers.cpp /
c7x_pool_relu_wrappers.cpp); it just hadn't been applied to this file yet.
Fixed by making the include unconditional. Real hardware (c7x_dload) was
never affected -- the cross-compiler predefines __C7524__ as a builtin
before this file is parsed, so it already ran the vectorized path; the
small residual hardware mismatches that motivated this investigation were
a separate, genuine test-reference issue (below).

With the kernel fix in place, c7x_host now runs the same vectorized paths
as hardware, so the references must model those exactly rather than an
idealized scalar formula:
  - _numpy_channel_scale_multiply replicates channel_scale_multiply_vec's
    Q13 fixed-point integer arithmetic bit-for-bit (bare, unrounded right
    shifts), the same convention test_concat_kernel.py's
    _numpy_concat_rescale already uses for c7x_int8_concat_rescale.
  - _numpy_hardswish splits by vectorized-bulk vs. scalar-tail, since
    hardswish_vec itself uses two different rounding rules for the two
    regions (round-to-nearest-even via hardware __float_to_int in the
    bulk; round-half-up via rq_f in the tail -- see
    test_input_normalize_quantize_kernel.py's _numpy_quantize_rgb for the
    same pattern), and keeps the bulk path in float32 throughout
    (matching __float8 arithmetic, including multiplying by a
    precomputed reciprocal 1/sy rather than dividing) -- mixing in
    float64 scale/zp scalars produced spurious mismatches exactly at
    rounding ties that weren't really ties in the kernel's own float32
    computation.

Usage:
    pytest test_activation_kernels.py -v --dsp-mode=c7x_host
    pytest test_activation_kernels.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _numpy_hardswish(inp, zx, sx, zy, sy):
    # hardswish_vec's vectorized bulk requantizes via a bare
    # vf*vinvsy+vzy then hardware __float_to_int (VSPINT, documented as
    # round-to-nearest-even); its own scalar tail -- inside the same
    # function -- instead calls rq_f: (int32_t)(y/scale + 0.5f), i.e.
    # trunc(x+0.5), round-half-up. Two different rounding rules within
    # one kernel invocation depending on which elements land in the
    # vectorized bulk (HW // 8 * 8 elements) vs. the tail (n % 8) --
    # reproduce both exactly rather than assuming one rounding rule for
    # the whole array (same rigor as
    # test_input_normalize_quantize_kernel.py's _numpy_quantize_rgb).
    #
    # The bulk path must also stay in float32 throughout (matching the
    # kernel's __float8 arithmetic) and multiply by a precomputed
    # reciprocal (vinvsy = 1/sy, one float32 division) rather than
    # dividing by sy per element -- mixing in float64 scale/zp scalars or
    # using a true division for the bulk reference introduces a different
    # rounding error than the kernel's, which shows up as spurious
    # mismatches exactly at exact.5 ties (e.g. q=62.5) that aren't really
    # ties in the kernel's own float32 computation.
    f32 = np.float32
    n = len(inp)
    nvec8 = (n // 8) * 8
    zx32, sx32, zy32 = f32(zx), f32(sx), f32(zy)
    inv6, half, zero, one = f32(1.0 / 6.0), f32(0.5), f32(0.0), f32(1.0)
    lo, hi, invsy = f32(-128.0), f32(127.0), one / f32(sy)

    x = (inp.astype(f32) - zx32) * sx32
    y = x * np.clip(x * inv6 + half, zero, one).astype(f32)

    q_bulk = np.clip(y[:nvec8] * invsy + zy32, lo, hi)
    bulk_out = np.round(q_bulk).astype(np.int32)  # ties-to-even, matches __float_to_int

    tail_out = np.trunc(y[nvec8:].astype(np.float64) / float(sy) + 0.5).astype(np.int32) + zy

    out = np.empty(n, dtype=np.int32)
    out[:nvec8] = bulk_out
    out[nvec8:] = tail_out
    return np.clip(out, -128, 127).astype(np.int8)


def _numpy_channel_scale_multiply(exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out):
    # c7x_host and c7x_dload both compile with __C7524__ defined, so both
    # actually execute channel_scale_multiply_vec's Q13 fixed-point integer
    # path (tidl_activation_wrappers.cpp), never the float rq_f-based
    # scalar fallback (that only exists for non-__C7524__ targets, never
    # built here). Replicate that Q13 algorithm exactly -- same per-channel
    # scale_q rounding, same bare (floor, not round-to-nearest) right
    # shifts for the offset and every per-element product -- following
    # test_concat_kernel.py's _numpy_concat_rescale, which models
    # c7x_int8_concat_rescale's identical Q13-bare-shift pattern the same
    # way.
    SHIFT = 13
    out = np.zeros(C * H_W, dtype=np.int32)
    for c in range(C):
        scale_f = float(exc[c] - z_exc) * s_exc * s_feat / s_out
        scale_q = int(scale_f * (1 << SHIFT) + 0.5)
        offset = int(z_out) - (int(z_feat) * scale_q >> SHIFT)
        for j in range(H_W):
            feat = int(fm[c * H_W + j])
            v = (feat * scale_q >> SHIFT) + offset
            out[c * H_W + j] = np.clip(v, -128, 127)
    return out.astype(np.int8)


# ---------------------------------------------------------------------------
# Module builders
# ---------------------------------------------------------------------------


def _build_hardswish_module(n, zx, sx, zy, sy):
    n_v = int(n)
    zx_v = int(zx)
    sx_v = float(sx)
    zy_v = int(zy)
    sy_v = float(sy)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "c7x_int8_hardswish",
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", n_v),
                tir.IntImm("int32", zx_v),
                tir.FloatImm("float32", sx_v),
                tir.IntImm("int32", zy_v),
                tir.FloatImm("float32", sy_v),
            )

        return te.extern([n_v], [x_t], fcompute, name="hardswish_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([n_v], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint="c7x_int8_hardswish")
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_channel_scale_multiply_module(C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out):
    C_v = int(C)
    HW_v = int(H_W)
    se_v = float(s_exc)
    ze_v = int(z_exc)
    sf_v = float(s_feat)
    zf_v = int(z_feat)
    so_v = float(s_out)
    zo_v = int(z_out)

    def te_kernel(exc_t, fm_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "tidl_int8_channel_scale_multiply",
                ins[0].data,
                ins[1].data,
                outs[0].data,
                tir.IntImm("int32", C_v),
                tir.IntImm("int32", HW_v),
                tir.FloatImm("float32", se_v),
                tir.IntImm("int32", ze_v),
                tir.FloatImm("float32", sf_v),
                tir.IntImm("int32", zf_v),
                tir.FloatImm("float32", so_v),
                tir.IntImm("int32", zo_v),
            )

        return te.extern(
            [C_v * HW_v], [exc_t, fm_t], fcompute, name="channel_scale_mul_out", dtype="int8"
        )

    bb = relax.BlockBuilder()
    exc_var = relax.Var("exc", relax.TensorStructInfo([C_v], "int8"))
    fm_var = relax.Var("fm", relax.TensorStructInfo([C_v * HW_v], "int8"))
    with bb.function("main", [exc_var, fm_var], attrs={"num_input": 2}):
        with bb.dataflow():
            result = bb.emit_te(
                te_kernel, exc_var, fm_var, primfunc_name_hint="tidl_int8_channel_scale_multiply"
            )
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _run_hardswish(dsp_mode, inp, zx, sx, zy, sy):
    mod = _build_hardswish_module(len(inp), zx, sx, zy, sy)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[inp],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


def _run_channel_scale_multiply(
    dsp_mode, exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out
):
    mod = _build_channel_scale_multiply_module(C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[exc, fm],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


# ---------------------------------------------------------------------------
# c7x_int8_hardswish tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_hardswish_scalar_tail(dsp_mode):
    """n=37 — exercises only the scalar tail path (< 8 elements per iteration)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    inp = rng.integers(-128, 127, 37, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.05, 0, 0.05
    ref = _numpy_hardswish(inp, zx, sx, zy, sy)
    out, _ = _run_hardswish(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_hardswish_asymmetric_zp(dsp_mode):
    """Non-zero zero-points — tests zp subtraction and output offset."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    inp = rng.integers(-128, 127, 128, dtype=np.int8)
    zx, sx, zy, sy = -10, 0.04, 5, 0.03
    ref = _numpy_hardswish(inp, zx, sx, zy, sy)
    out, _ = _run_hardswish(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_hardswish_mid_size(dsp_mode):
    """n=94_080 (56×56×30) — exercises the inner vectorized loop."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    n = 56 * 56 * 30
    inp = rng.integers(-128, 127, n, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.03, 0, 0.02
    ref = _numpy_hardswish(inp, zx, sx, zy, sy)
    out, _ = _run_hardswish(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_hardswish_large(dsp_mode, record_cycles):
    """n=200_704 (112×112×16) — largest hardswish layer in MobileNetV3."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    n = 112 * 112 * 16
    inp = rng.integers(-128, 127, n, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.025, 0, 0.018
    ref = _numpy_hardswish(inp, zx, sx, zy, sy)
    out, cycles = _run_hardswish(dsp_mode, inp, zx, sx, zy, sy)
    record_cycles("hardswish_n200704", cycles)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        print(f"\n  c7x_int8_hardswish n={n}: {cycles:,} cycles ({cycles / n:.2f} cycles/element)")


# ---------------------------------------------------------------------------
# tidl_int8_channel_scale_multiply tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_csm_small_tail(dsp_mode):
    """C=16, H_W=11 — exercises the scalar tail in the inner H_W loop."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(4)
    C, H_W = 16, 11
    exc = rng.integers(-128, 127, C, dtype=np.int8)
    fm = rng.integers(-128, 127, C * H_W, dtype=np.int8)
    s_exc, z_exc, s_feat, z_feat, s_out, z_out = 0.04, 0, 0.03, 0, 0.05, 0
    ref = _numpy_channel_scale_multiply(exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out)
    out, _ = _run_channel_scale_multiply(
        dsp_mode, exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out
    )
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_csm_zero_excitation(dsp_mode):
    """All excitation values equal z_exc — all outputs should equal z_out."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    C, H_W = 32, 16
    z_exc = 5
    exc = np.full(C, z_exc, dtype=np.int8)
    fm = np.arange(C * H_W, dtype=np.int8)
    z_out = 3
    s_exc, s_feat, z_feat, s_out = 0.03, 0.02, 0, 0.04
    ref = _numpy_channel_scale_multiply(exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out)
    out, _ = _run_channel_scale_multiply(
        dsp_mode, exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out
    )
    assert np.all(out.flatten() == z_out), (
        f"expected all {z_out}, got unique={np.unique(out.flatten())}"
    )
    assert np.array_equal(out.flatten(), ref)


@pytest.mark.core
def test_csm_mobilenet_28x28(dsp_mode):
    """C=120, H_W=784 — 120 channels × 28×28, MobileNetV3 SE block."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(5)
    C, H_W = 120, 28 * 28
    exc = rng.integers(-128, 127, C, dtype=np.int8)
    fm = rng.integers(-128, 127, C * H_W, dtype=np.int8)
    s_exc, z_exc, s_feat, z_feat, s_out, z_out = 0.02, -3, 0.03, 2, 0.04, 0
    ref = _numpy_channel_scale_multiply(exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out)
    out, _ = _run_channel_scale_multiply(
        dsp_mode, exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out
    )
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_csm_mobilenet_14x14(dsp_mode):
    """C=480, H_W=196 — 480 channels × 14×14."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(6)
    C, H_W = 480, 14 * 14
    exc = rng.integers(-128, 127, C, dtype=np.int8)
    fm = rng.integers(-128, 127, C * H_W, dtype=np.int8)
    s_exc, z_exc, s_feat, z_feat, s_out, z_out = 0.015, 0, 0.025, -5, 0.035, 1
    ref = _numpy_channel_scale_multiply(exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out)
    out, _ = _run_channel_scale_multiply(
        dsp_mode, exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out
    )
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_csm_mobilenet_7x7(dsp_mode, record_cycles):
    """C=960, H_W=49 — largest SE block in MobileNetV3."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(7)
    C, H_W = 960, 7 * 7
    exc = rng.integers(-128, 127, C, dtype=np.int8)
    fm = rng.integers(-128, 127, C * H_W, dtype=np.int8)
    s_exc, z_exc, s_feat, z_feat, s_out, z_out = 0.012, 0, 0.020, 0, 0.025, 0
    ref = _numpy_channel_scale_multiply(exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out)
    out, cycles = _run_channel_scale_multiply(
        dsp_mode, exc, fm, C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out
    )
    record_cycles("csm_C960_HW49", cycles)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        n = C * H_W
        print(
            f"\n  tidl_int8_channel_scale_multiply C={C} H_W={H_W}: "
            f"{cycles:,} cycles ({cycles / n:.2f} cycles/element)"
        )
