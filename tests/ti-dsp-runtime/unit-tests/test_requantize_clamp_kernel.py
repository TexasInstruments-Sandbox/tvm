"""Unit tests for c7x_int8_requantize_clamp kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference. Tests are independent of the
ti_fuse_qdq_c7x_relu.py fusion pass.

The kernel computes out[i] = clamp(round(in[i] * combined_scale), clip_lo,
clip_hi) in Q13 fixed point, rounding half up:

  scale_q = (int32_t)(combined_scale * 2^13 + 0.5f)     [float32]
  out[i]  = clamp((in[i] * scale_q + 2^12) >> 13, clip_lo, clip_hi)

Usage:
    pytest test_requantize_clamp_kernel.py -v --dsp-mode=c7x_host
    pytest test_requantize_clamp_kernel.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tirx

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402

_KERNEL = "c7x_int8_requantize_clamp"
_SHIFT = 13

# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------


def _scale_q(combined_scale):
    # The kernel computes scale_q in float32; match it exactly.
    f32 = np.float32
    return int(f32(combined_scale) * f32(1 << _SHIFT) + f32(0.5))


def _numpy_requantize_clamp(x, combined_scale, clip_lo, clip_hi):
    v = (x.astype(np.int64) * _scale_q(combined_scale) + (1 << (_SHIFT - 1))) >> _SHIFT
    return np.clip(v, clip_lo, clip_hi).astype(np.int8)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_module(n, combined_scale, clip_lo, clip_hi):
    n_val = int(n)
    scale_val = float(combined_scale)
    clip_lo_val = int(clip_lo)
    clip_hi_val = int(clip_hi)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tirx.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                outs[0].data,
                tirx.IntImm("int32", n_val),
                tirx.FloatImm("float32", scale_val),
                tirx.IntImm("int32", clip_lo_val),
                tirx.IntImm("int32", clip_hi_val),
            )

        return te.extern([n_val], [x_t], fcompute, name="requantize_clamp_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorType([n_val], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _check(dsp_mode, x, combined_scale, clip_lo=-128, clip_hi=127):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    ref = _numpy_requantize_clamp(x, combined_scale, clip_lo, clip_hi)
    mod = _build_module(len(x), combined_scale, clip_lo, clip_hi)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=x,
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].flatten()
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_requantize_clamp_rounds_not_floors(dsp_mode):
    """Every int8 input at a scale whose products are rarely exact, so a
    floored shift would differ from the reference on most elements."""
    x = np.arange(-128, 128, dtype=np.int8)
    floor_ref = np.clip((x.astype(np.int64) * _scale_q(0.37)) >> _SHIFT, -128, 127)
    assert not np.array_equal(_numpy_requantize_clamp(x, 0.37, -128, 127), floor_ref)
    _check(dsp_mode, x, 0.37)


@pytest.mark.quick
def test_requantize_clamp_ties(dsp_mode):
    """combined_scale=0.5 puts every odd input exactly on a .5 tie, which
    rounds half up (toward +inf) for both signs."""
    x = np.arange(-128, 128, dtype=np.int8)
    _check(dsp_mode, x, 0.5)


@pytest.mark.quick
def test_requantize_clamp_relu6_bounds(dsp_mode):
    """ReLU6-shaped clip after a rescale (the pattern the pass lowers)."""
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 127, 256, dtype=np.int8)
    _check(dsp_mode, x, 1.7, 0, 45)


@pytest.mark.quick
@pytest.mark.parametrize("n", [1, 5, 7, 9, 15, 17, 63])
def test_requantize_clamp_non_multiple_of_8(dsp_mode, n):
    """n % 8 != 0 -- exercises the scalar tail path in the vectorized kernel."""
    rng = np.random.default_rng(2)
    x = rng.integers(-128, 127, n, dtype=np.int8)
    _check(dsp_mode, x, 0.83)
