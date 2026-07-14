"""Unit tests for c7x_int8_avg_pool kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference. Tests are independent of FuseQDQToC7xAvgPool.

The kernel's scalar path (used for the 1-pixel border of the fast path, and
for any config other than kH=kW=3/sH=sW=1/pH=pW=1) matches a float
dequantize -> mean -> requantize formula with round-half-up rounding
(np.trunc(y/scale + 0.5), matching the kernel's (int32_t)(y/scale + 0.5f)).

The fast path's interior (away from the 1-pixel pad border, where all 9
window taps are valid) uses Q13 fixed-point instead: this is a *truncating*
right shift, not round-to-nearest -- same convention as
c7x_int8_requantize_clamp / c7x_int8_concat_rescale, whose unit tests also
match the kernel's exact integer arithmetic rather than an idealized
rounding. The reference below reproduces both paths bit-for-bit.

Usage:
    pytest test_avgpool_kernel.py -v --dsp-mode=c7x_host
    pytest test_avgpool_kernel.py -v --dsp-mode=c7x_dload
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
# Numpy reference
# ---------------------------------------------------------------------------

_KERNEL = "c7x_int8_avg_pool"
_SHIFT = 13


def _rq(y, zy, sy):
    """Matches the kernel's scalar rq(): (int32_t)(y/sy + 0.5f) + zy, clamped."""
    v = np.trunc(y / sy + 0.5).astype(np.int64) + zy
    return int(np.clip(v, -128, 127))


def _numpy_avg_pool_scalar(x2d, kH, kW, sH, sW, pH, pW, zx, sx, zy, sy):
    """Exact reference for the kernel's scalar path (any config, any pixel).

    count_include_pad=True: divisor is always the fixed kH*kW, regardless of
    how many window taps land outside the input (those taps contribute 0).
    """
    H_in, W_in = x2d.shape
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    inv_k = sx / (kH * kW)
    xi = x2d.astype(np.int64)
    out = np.zeros((H_out, W_out), dtype=np.int8)
    for ph in range(H_out):
        ih0 = ph * sH - pH
        for pw in range(W_out):
            iw0 = pw * sW - pW
            s = 0
            for kh in range(kH):
                ih = ih0 + kh
                if ih < 0 or ih >= H_in:
                    continue
                for kw in range(kW):
                    iw = iw0 + kw
                    if iw < 0 or iw >= W_in:
                        continue
                    s += int(xi[ih, iw]) - zx
            out[ph, pw] = _rq(s * inv_k, zy, sy)
    return out


def _numpy_avg_pool_fastpath(x2d, zx, sx, zy, sy):
    """Exact reference for the kernel's 3x3/stride=1/pad=1 fast path.

    Border (rows/cols 0 and H-1/W-1): scalar rq() path, unchanged.
    Interior [1, H-2] x [1, W-2]: Q13 fixed-point, matching
    avg_pool_interior_vec exactly (truncating >> _SHIFT, not rounded).
    """
    H, W = x2d.shape
    out = _numpy_avg_pool_scalar(x2d, 3, 3, 1, 1, 1, 1, zx, sx, zy, sy)
    if H < 3 or W < 3:
        return out

    # Match the kernel's float32 (not float64) arithmetic exactly: inv_k and
    # combined_scale are both `float` in C, and scale_q's rounding is
    # sensitive to that precision -- a float64 reference can round scale_q
    # to a different integer than the kernel, biasing every interior pixel.
    inv_k32 = np.float32(sx) / np.float32(9.0)
    combined_scale32 = inv_k32 / np.float32(sy)
    scale_q = int(combined_scale32 * np.float32(1 << _SHIFT) + np.float32(0.5))
    xi = x2d.astype(np.int64)
    for ph in range(1, H - 1):
        for pw in range(1, W - 1):
            sum9 = int(xi[ph - 1 : ph + 2, pw - 1 : pw + 2].sum())
            v = ((sum9 - 9 * zx) * scale_q) >> _SHIFT
            v += zy
            out[ph, pw] = np.clip(v, -128, 127)
    return out


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_avg_pool_module(N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, zx, sx, zy, sy):
    _N, _C = N, C
    _H_in, _W_in, _H_out, _W_out = H_in, W_in, H_out, W_out
    _kH, _kW, _sH, _sW, _pH, _pW = kH, kW, sH, sW, pH, pW
    _zx, _sx, _zy, _sy = zx, float(sx), zy, float(sy)

    def te_avg_pool(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", _N),
                tir.IntImm("int32", _C),
                tir.IntImm("int32", _H_in),
                tir.IntImm("int32", _W_in),
                tir.IntImm("int32", _H_out),
                tir.IntImm("int32", _W_out),
                tir.IntImm("int32", _kH),
                tir.IntImm("int32", _kW),
                tir.IntImm("int32", _sH),
                tir.IntImm("int32", _sW),
                tir.IntImm("int32", _pH),
                tir.IntImm("int32", _pW),
                tir.IntImm("int32", _zx),
                tir.FloatImm("float32", _sx),
                tir.IntImm("int32", _zy),
                tir.FloatImm("float32", _sy),
            )

        return te.extern(
            [_N, _C, _H_out, _W_out],
            [x_t],
            fcompute,
            name="avgpool_out",
            dtype="int8",
        )

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([N, C, H_in, W_in], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_avg_pool, x_var, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_avg_pool(
    dsp_mode, x, N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, zx, sx, zy, sy
):
    mod = _build_avg_pool_module(
        N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, zx, sx, zy, sy
    )
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=x,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    out = results[f"{dsp_mode}_result"]
    cycles = results.get("c7x_dload_cycles", 0)
    return out.reshape(N, C, H_out, W_out), cycles


def _check(dsp_mode, x2d, kH, kW, sH, sW, pH, pW, zx, sx, zy, sy, ref):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    H_in, W_in = x2d.shape
    H_out, W_out = ref.shape
    x = x2d.reshape(1, 1, H_in, W_in)
    out, _ = _run_avg_pool(
        dsp_mode,
        x,
        1,
        1,
        H_in,
        W_in,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
        zx,
        sx,
        zy,
        sy,
    )
    out2d = out.reshape(H_out, W_out)
    assert np.array_equal(out2d, ref), (
        f"max_err={np.abs(out2d.astype(int) - ref.astype(int)).max()}\ngot=\n{out2d}\nref=\n{ref}"
    )


# ---------------------------------------------------------------------------
# Tests: 3x3/stride=1/pad=1 fast path
# ---------------------------------------------------------------------------


_FASTPATH_CASES = {
    # symmetric zero-points: exercises both the Q13 interior path and the
    # scalar border path in a single call
    "interior_and_border": (0, 16, 16, 0, 0.05, 0, 0.045),
    # asymmetric zx != 0, zy != 0 -- the exact case the rejected TIDL
    # spatialAvgPool C7x kernel would get wrong (its vectorized exec path
    # has no zero-point term; see quantized_model_optimization.md Step 12)
    "asymmetric_zero_points": (1, 20, 20, 12, 0.037, -8, 0.031),
    # W=6 < 8: the vertical-sum stage has zero full 8-wide SE vectors, so
    # both stages run entirely through their scalar-tail paths
    "below_vector_width": (2, 6, 6, 3, 0.04, -2, 0.036),
    # W=20, not a multiple of 8: exercises vectorized chunks plus a
    # remainder in both the vertical-sum and horizontal-sum stages
    "above_vector_width": (3, 20, 20, 0, 0.028, 5, 0.033),
    # 3x3 image: interior is a single center pixel, everything else is the
    # 1-pixel border -- the smallest shape the fast path accepts
    "minimum_size_border_dominated": (4, 3, 3, -4, 0.05, 2, 0.041),
}


@pytest.mark.quick
@pytest.mark.parametrize(
    "seed, H, W, zx, sx, zy, sy", _FASTPATH_CASES.values(), ids=_FASTPATH_CASES.keys()
)
def test_avg_pool_fastpath(dsp_mode, seed, H, W, zx, sx, zy, sy):
    rng = np.random.default_rng(seed)
    x2d = rng.integers(-128, 127, (H, W), dtype=np.int8)
    ref = _numpy_avg_pool_fastpath(x2d, zx, sx, zy, sy)
    _check(dsp_mode, x2d, 3, 3, 1, 1, 1, 1, zx, sx, zy, sy, ref)


@pytest.mark.core
def test_avg_pool_fastpath_inceptionv3_size(dsp_mode, record_cycles):
    """InceptionV3 branch_pool: 35x35, the largest spatial avg_pool in the
    current model suite (docs/dsp/quantized_model_optimization.md Bottleneck B)."""
    rng = np.random.default_rng(5)
    H = W = 35
    x2d = rng.integers(-128, 127, (H, W), dtype=np.int8)
    zx, sx, zy, sy = -2, 0.021, 1, 0.019
    ref = _numpy_avg_pool_fastpath(x2d, zx, sx, zy, sy)
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    x = x2d.reshape(1, 1, H, W)
    out, cycles = _run_avg_pool(dsp_mode, x, 1, 1, H, W, H, W, 3, 3, 1, 1, 1, 1, zx, sx, zy, sy)
    record_cycles("avgpool_inceptionv3_35x35", cycles)
    out2d = out.reshape(H, W)
    assert np.array_equal(out2d, ref), (
        f"max_err={np.abs(out2d.astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        print(f"\n  c7x_int8_avg_pool 35x35: {cycles:,} cycles")


# ---------------------------------------------------------------------------
# Tests: scalar fallback (non-3x3/stride=1/pad=1 configs)
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_avg_pool_scalar_fallback_2x2_stride2(dsp_mode):
    """kH=kW=2, sH=sW=2 (standard downsampling avg pool): not the fast
    path's shape, so the whole image should use the scalar rq() path."""
    rng = np.random.default_rng(6)
    H = W = 16
    x2d = rng.integers(-128, 127, (H, W), dtype=np.int8)
    zx, sx, zy, sy = 4, 0.03, -1, 0.028
    ref = _numpy_avg_pool_scalar(x2d, 2, 2, 2, 2, 0, 0, zx, sx, zy, sy)
    _check(dsp_mode, x2d, 2, 2, 2, 2, 0, 0, zx, sx, zy, sy, ref)


@pytest.mark.quick
def test_avg_pool_scalar_fallback_3x3_stride1_no_padding(dsp_mode):
    """kH=kW=3, sH=sW=1, but pH=pW=0 (no padding, H_out != H_in): fails the
    fast path's same-size condition, so falls back to the scalar path."""
    rng = np.random.default_rng(7)
    H = W = 16
    x2d = rng.integers(-128, 127, (H, W), dtype=np.int8)
    zx, sx, zy, sy = -3, 0.036, 2, 0.03
    ref = _numpy_avg_pool_scalar(x2d, 3, 3, 1, 1, 0, 0, zx, sx, zy, sy)
    _check(dsp_mode, x2d, 3, 3, 1, 1, 0, 0, zx, sx, zy, sy, ref)
