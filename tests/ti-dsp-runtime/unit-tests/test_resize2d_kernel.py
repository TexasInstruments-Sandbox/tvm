"""Unit tests for the c7x_int8_resize2d_nearest2x kernel.

Invokes the kernel via call_extern with known inputs and verifies output
against a numpy reference.  Tests are independent of the
FuseQDQToC7xMovement pass.

c7x_int8_resize2d_nearest2x: NCHW int8 nearest-neighbor 2x spatial upsample
  out[c, 2h+dh, 2w+dw] = in[c, h, w]   for dh, dw in {0, 1}

Pure int8 gather -- no scale/zero-point parameters. Matches
relax.image.resize2d(method="nearest_neighbor",
coordinate_transformation_mode="half_pixel", rounding_method="round")
for an exact 2x upsample (see c7x_rescale.h for the derivation showing
half_pixel+round collapses to floor(dst/2) at scale factor 2).

Usage:
    pytest test_resize2d_kernel.py -v --dsp-mode=c7x_host
    pytest test_resize2d_kernel.py -v --dsp-mode=c7x_dload
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

_KERNEL = "c7x_int8_resize2d_nearest2x"


def _numpy_resize2d_nearest2x(inp_chw):
    """inp_chw: [C, H, W] int8 array. Returns [C, 2H, 2W]."""
    return np.repeat(np.repeat(inp_chw, 2, axis=1), 2, axis=2)


def _build_resize2d_module(C, H, W):
    C_v, H_v, W_v = int(C), int(H), int(W)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", C_v),
                tir.IntImm("int32", H_v),
                tir.IntImm("int32", W_v),
            )

        return te.extern(
            [C_v, 2 * H_v, 2 * W_v], [x_t], fcompute, name="resize2d_out", dtype="int8"
        )

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([C_v, H_v, W_v], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_resize2d(dsp_mode, inp_chw):
    C, H, W = inp_chw.shape
    mod = _build_resize2d_module(C, H, W)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[inp_chw],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_resize2d_small(dsp_mode):
    """C=2, H=W=4 -- small, easy to hand-verify shape correctness."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    C, H, W = 2, 4, 4
    inp = rng.integers(-128, 127, (C, H, W), dtype=np.int8)
    ref = _numpy_resize2d_nearest2x(inp)
    out, _ = _run_resize2d(dsp_mode, inp)
    out = out.reshape(C, 2 * H, 2 * W)
    assert np.array_equal(out, ref)


@pytest.mark.quick
def test_resize2d_odd_width(dsp_mode):
    """W=11 (not a multiple of 8) -- exercises the scalar-loop tail within a row."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    C, H, W = 3, 5, 11
    inp = rng.integers(-128, 127, (C, H, W), dtype=np.int8)
    ref = _numpy_resize2d_nearest2x(inp)
    out, _ = _run_resize2d(dsp_mode, inp)
    out = out.reshape(C, 2 * H, 2 * W)
    assert np.array_equal(out, ref)


@pytest.mark.quick
def test_resize2d_single_channel(dsp_mode):
    """C=1 -- boundary case for the channel loop."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    inp = np.arange(-128, -128 + 6 * 6, dtype=np.int8).reshape(1, 6, 6)
    ref = _numpy_resize2d_nearest2x(inp)
    out, _ = _run_resize2d(dsp_mode, inp)
    out = out.reshape(1, 12, 12)
    assert np.array_equal(out, ref)


@pytest.mark.core
def test_resize2d_fpn_10to20(dsp_mode, record_cycles):
    """C=256, 10x10 -> 20x20 -- yolo26n's FPN P5->P4 upsample shape."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    C, H, W = 256, 10, 10
    inp = rng.integers(-128, 127, (C, H, W), dtype=np.int8)
    ref = _numpy_resize2d_nearest2x(inp)
    out, cycles = _run_resize2d(dsp_mode, inp)
    record_cycles("resize2d_C256_10to20", cycles)
    out = out.reshape(C, 2 * H, 2 * W)
    assert np.array_equal(out, ref)
    if cycles:
        n = C * (2 * H) * (2 * W)
        print(
            f"\n  c7x_int8_resize2d_nearest2x C={C} 10->20: "
            f"{cycles:,} cycles ({cycles / n:.2f} cycles/output-element)"
        )


@pytest.mark.core
def test_resize2d_fpn_20to40(dsp_mode, record_cycles):
    """C=128, 20x20 -> 40x40 -- yolo26n's FPN P4->P3 upsample shape."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    C, H, W = 128, 20, 20
    inp = rng.integers(-128, 127, (C, H, W), dtype=np.int8)
    ref = _numpy_resize2d_nearest2x(inp)
    out, cycles = _run_resize2d(dsp_mode, inp)
    record_cycles("resize2d_C128_20to40", cycles)
    out = out.reshape(C, 2 * H, 2 * W)
    assert np.array_equal(out, ref)
    if cycles:
        n = C * (2 * H) * (2 * W)
        print(
            f"\n  c7x_int8_resize2d_nearest2x C={C} 20->40: "
            f"{cycles:,} cycles ({cycles / n:.2f} cycles/output-element)"
        )
