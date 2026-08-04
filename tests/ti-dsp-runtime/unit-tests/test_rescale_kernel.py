"""Unit tests for the c7x_int8_rescale kernel.

Invokes the kernel via call_extern with known inputs and verifies output
against a numpy reference.  Tests are independent of the
FuseQDQToC7xMovement pass.

c7x_int8_rescale: flat int8->int8 Q13 affine rescale
  out[i] = sat_i8(((in[i] - zx) * scale_q >> 13) + offset)
  scale_q = round(sx / sy * 2^13), offset = zy - ((zx * scale_q) >> 13)

Transparent fast path fires when sx == sy and zx == zy (memcpy). Shape-
agnostic: the kernel only sees a flat element count, so it backs any
dequantize -> reshape -> quantize chain regardless of tensor rank.

Usage:
    pytest test_rescale_kernel.py -v --dsp-mode=c7x_host
    pytest test_rescale_kernel.py -v --dsp-mode=c7x_dload
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

_KERNEL = "c7x_int8_rescale"
_SHIFT = 13


def _numpy_rescale(inp, zx, sx, zy, sy):
    """Reference using the kernel's Q13 fixed-point arithmetic for exact
    match -- same convention as test_concat_kernel.py's
    _numpy_concat_rescale (single slot, no HW/channel split)."""
    if sx == sy and zx == zy:
        return inp.copy()
    scale_q = np.int32(int(sx / sy * (1 << _SHIFT) + 0.5))
    offset = int(zy) - int(np.int64(zx) * int(scale_q) >> _SHIFT)
    result = np.clip((inp.astype(np.int32) * int(scale_q) >> _SHIFT) + offset, -128, 127).astype(
        np.int8
    )
    return result


def _build_rescale_module(n, zx, sx, zy, sy):
    n_v = int(n)
    zx_v, sx_v, zy_v, sy_v = int(zx), float(sx), int(zy), float(sy)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", n_v),
                tir.IntImm("int32", zx_v),
                tir.FloatImm("float32", sx_v),
                tir.IntImm("int32", zy_v),
                tir.FloatImm("float32", sy_v),
            )

        return te.extern([n_v], [x_t], fcompute, name="rescale_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([n_v], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_rescale(dsp_mode, inp, zx, sx, zy, sy):
    mod = _build_rescale_module(len(inp), zx, sx, zy, sy)
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
def test_rescale_transparent(dsp_mode):
    """Matching scale/zp -- exercises the memcpy fast path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    inp = rng.integers(-128, 127, 256, dtype=np.int8)
    s, z = 0.05, -3
    ref = _numpy_rescale(inp, z, s, z, s)
    out, _ = _run_rescale(dsp_mode, inp, z, s, z, s)
    assert np.array_equal(out.flatten(), ref)


@pytest.mark.quick
def test_rescale_mismatched_scale(dsp_mode):
    """Different input/output scale and zero-point -- exercises the Q13 rescale path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    inp = rng.integers(-128, 127, 256, dtype=np.int8)
    zx, sx, zy, sy = 5, 0.03, -2, 0.05
    ref = _numpy_rescale(inp, zx, sx, zy, sy)
    out, _ = _run_rescale(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_rescale_scalar_tail(dsp_mode):
    """n=37, not a multiple of 8 -- exercises the scalar tail path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    inp = rng.integers(-128, 127, 37, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.04, 3, 0.06
    ref = _numpy_rescale(inp, zx, sx, zy, sy)
    out, _ = _run_rescale(dsp_mode, inp, zx, sx, zy, sy)
    assert np.array_equal(out.flatten(), ref)


@pytest.mark.core
def test_rescale_large(dsp_mode, record_cycles):
    """n=200_704 (112x112x16) -- exercises the inner vectorized loop at scale."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    n = 112 * 112 * 16
    inp = rng.integers(-128, 127, n, dtype=np.int8)
    zx, sx, zy, sy = 0, 0.025, 0, 0.018
    ref = _numpy_rescale(inp, zx, sx, zy, sy)
    out, cycles = _run_rescale(dsp_mode, inp, zx, sx, zy, sy)
    record_cycles("rescale_n200704", cycles)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        print(f"\n  c7x_int8_rescale n={n}: {cycles:,} cycles ({cycles / n:.2f} cycles/element)")
