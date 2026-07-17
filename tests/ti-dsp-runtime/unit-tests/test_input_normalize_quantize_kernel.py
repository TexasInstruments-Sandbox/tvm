"""Unit tests for c7x_int8_quantize_rgb kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference. Tests are independent of the
FuseInputNormalizeQuantize fusion pass (see test_input_normalize_quantize_pass.py
for that).

The kernel computes, per channel c and per batch n:
  out[n,c,:] = clamp(round(in[n,c,:] * inv_scale_c + offset_c), -128, 127)

Rounding convention differs by element position within each channel plane,
inherited unchanged from c7x_int8_quantize (this kernel reuses the exact
same quantize_vec/quantize_scalar helpers): the vectorized bulk
(HW // 8 * 8 elements) rounds via __float_to_int/VSPINT, which is
round-half-to-even; the scalar tail (HW % 8 remaining elements per plane)
rounds via the truncating (int32_t)(v >= 0 ? v + 0.5f : v - 0.5f), which is
round-half-away-from-zero. These agree everywhere except exact .5 ties, so
the reference below reproduces both paths bit-for-bit rather than one
rounding rule for the whole tensor (same rigor as test_avgpool_kernel.py's
handling of its own two-rounding-mode kernel).

Small-HW hardware bug (found and fixed): c7x_int8_quantize's (and this
kernel's shared quantize_vec helper's) SE-vectorized path used to return
wrong (~zero-input) results on real c7x_dload hardware for small
per-plane sizes (empirically, HW < 64). Root cause: the main 4x-unrolled
loop was preceded by `#pragma MUST_ITERATE(1,,)`, asserting at least 1
iteration -- but the loop's trip count is nvec4/4, and nvec4 = nvec & ~3
is exactly 0 whenever HW < 32, making the pragma's claim false. Per TI's
compiler docs, MUST_ITERATE's effect is to let the compiler eliminate the
unpipelined safety-fallback loop that otherwise correctly handles small/
zero trip counts -- exactly the condition this loop needed. Removing the
(invalid) pragma from c7x_quantize.cpp's quantize_vec fixed every
previously-failing case in the characterization matrix (HW=8,16,24,31,63),
including the HW=63 case, which failed only in its scalar tail -- outside
what the vector loop's pragma should plausibly have affected, so the fix
resolving it too was not fully anticipated going in. Confirmed directly
against the unmodified (pre-fix) c7x_int8_quantize kernel in isolation, so
the bug predated and was independent of c7x_int8_quantize_rgb /
FuseInputNormalizeQuantize (Step 16).

Usage:
    pytest test_input_normalize_quantize_kernel.py -v --dsp-mode=c7x_host
    pytest test_input_normalize_quantize_kernel.py -v --dsp-mode=c7x_dload
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

_KERNEL = "c7x_int8_quantize_rgb"

# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------


def _numpy_quantize_rgb(x, params):
    """x: (N,3,H,W) float32. params: [(inv_scale_c, offset_c), ...] len 3.

    Splits each channel plane into its vectorized bulk (round-half-to-even,
    matching __float_to_int) and scalar tail (round-half-away-from-zero,
    matching the truncating v +/- 0.5 convention) -- see module docstring.
    """
    N, C, H, W = x.shape
    HW = H * W
    nvec8 = (HW // 8) * 8
    xf = x.reshape(N, C, HW)
    out = np.empty((N, C, HW), dtype=np.int8)
    for c, (inv_scale_c, offset_c) in enumerate(params):
        v = xf[:, c].astype(np.float32) * np.float32(inv_scale_c) + np.float32(offset_c)
        qi = np.empty(v.shape, dtype=np.float64)
        qi[:, :nvec8] = np.round(v[:, :nvec8])  # vectorized bulk: ties-to-even
        tail = v[:, nvec8:]
        qi[:, nvec8:] = np.where(
            tail >= 0, np.floor(tail + np.float32(0.5)), np.ceil(tail - np.float32(0.5))
        )
        out[:, c] = np.clip(qi, -128, 127).astype(np.int8)
    return out.reshape(N, C, H, W)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_module(N, H, W, params):
    _N, _HW = int(N), int(H * W)
    (_is0, _off0), (_is1, _off1), (_is2, _off2) = params

    def te_quantize_rgb(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", _N),
                tir.IntImm("int32", _HW),
                tir.FloatImm("float32", _is0),
                tir.FloatImm("float32", _off0),
                tir.FloatImm("float32", _is1),
                tir.FloatImm("float32", _off1),
                tir.FloatImm("float32", _is2),
                tir.FloatImm("float32", _off2),
            )

        return te.extern([N, 3, H, W], [x_t], fcompute, name="quantize_rgb_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([N, 3, H, W], "float32"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_quantize_rgb, x_var, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run(dsp_mode, x, params):
    N, _, H, W = x.shape
    mod = _build_module(N, H, W, params)
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
    return out.reshape(N, 3, H, W), cycles


def _check(dsp_mode, x, params):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    ref = _numpy_quantize_rgb(x, params)
    out, _ = _run(dsp_mode, x, params)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


# transform_input's actual torchvision constants (Inception3/GoogLeNet).
_TRANSFORM_INPUT_AFFINE = [
    (0.229 / 0.5, (0.485 - 0.5) / 0.5),
    (0.224 / 0.5, (0.456 - 0.5) / 0.5),
    (0.225 / 0.5, (0.406 - 0.5) / 0.5),
]


def _folded_params(affine, scale, zp):
    inv_scale = 1.0 / scale
    return [(a * inv_scale, b * inv_scale + zp) for a, b in affine]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_transform_input_shape(dsp_mode):
    """Real transform_input affine + a representative quantize (scale/zp),
    exercising the 4x-unrolled vector loop (H*W=64, multiple of 8)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, (1, 3, 8, 8)).astype(np.float32)
    params = _folded_params(_TRANSFORM_INPUT_AFFINE, scale=0.01, zp=-3)
    _check(dsp_mode, x, params)


@pytest.mark.quick
@pytest.mark.parametrize("h,w", [(1, 1), (1, 5), (2, 3), (3, 3), (7, 9)])
def test_below_and_non_multiple_of_8(dsp_mode, h, w):
    """HW below 8 or not a multiple of 8 -- exercises the scalar tail path
    (and, for HW < 8, no vector iterations at all). These per-plane sizes
    are exactly the range the MUST_ITERATE fix (see module docstring) was
    verified against on real c7x_dload hardware."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-3.0, 3.0, (1, 3, h, w)).astype(np.float32)
    params = _folded_params(_TRANSFORM_INPUT_AFFINE, scale=0.01, zp=-3)
    _check(dsp_mode, x, params)


@pytest.mark.quick
def test_batch_n_greater_than_1(dsp_mode):
    """N>1: confirms the kernel's channel-plane indexing (plane_idx % 3)
    correctly threads through multiple batches, not just N=1. HW=25 per
    plane is in the small-HW range covered by the MUST_ITERATE fix."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-3.0, 3.0, (4, 3, 5, 5)).astype(np.float32)
    params = _folded_params(_TRANSFORM_INPUT_AFFINE, scale=0.01, zp=-3)
    _check(dsp_mode, x, params)


@pytest.mark.quick
def test_extremes_saturate(dsp_mode):
    """Values that push well past +-127 after the affine -- checks clamp."""
    x = np.full((1, 3, 8, 8), 100.0, dtype=np.float32)
    x[:, 0] *= -1  # channel 0 saturates low, channels 1/2 saturate high
    params = _folded_params(_TRANSFORM_INPUT_AFFINE, scale=0.01, zp=-3)
    _check(dsp_mode, x, params)


@pytest.mark.core
def test_inceptionv3_input_size(dsp_mode, record_cycles):
    """3x299x299 -- InceptionV3/GoogLeNet's actual transform_input input
    size (docs/dsp/quantized_model_optimization.md Step 16)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    x = rng.uniform(-3.0, 3.0, (1, 3, 299, 299)).astype(np.float32)
    params = _folded_params(_TRANSFORM_INPUT_AFFINE, scale=0.01, zp=-3)
    ref = _numpy_quantize_rgb(x, params)
    out, cycles = _run(dsp_mode, x, params)
    record_cycles("input_normalize_quantize_rgb_299x299", cycles)
    assert np.array_equal(out, ref)
    if cycles:
        n = x.size
        print(
            f"\n  c7x_int8_quantize_rgb n={n}: {cycles:,} cycles ({cycles / n:.2f} cycles/element)"
        )
