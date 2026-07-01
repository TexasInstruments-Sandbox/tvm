"""Tile-consistency test for mmalib_conv2d_i8 OC tiling.

Verifies that calling mmalib_conv2d_i8 twice with C_out=oc_tile and
numpy-sliced weight/bias/scale/shift/output pointers produces bit-identical
results to a single call with C_out=128.

This validates the pointer-slice semantics used by mmalib_conv2d_i8_sliced
and InjectMMALIBDMA's OC-tiling path before testing them on hardware.

Usage:
    pytest test_mmalib_oc_tile_consistency.py -v --dsp-mode=c7x_host
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir
from tvm.ir.module import IRModule  # noqa: F401 (used in type hint string)

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a minimal Relax module calling mmalib_conv2d_i8 directly
# ---------------------------------------------------------------------------


def _build_conv_module(
    input_np: np.ndarray,
    kernel_np: np.ndarray,
    bias_np: np.ndarray,
    scale_np: np.ndarray,
    shift_np: np.ndarray,
    C_in: int, H_in: int, W_in: int,
    C_out: int, KH: int, KW: int,
    stride: int, padding: int,
):
    """Relax module that calls mmalib_conv2d_i8 with the given constants."""
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1

    _C_in, _H_in, _W_in = C_in, H_in, W_in
    _C_out, _KH, _KW = C_out, KH, KW
    _stride, _padding = stride, padding
    _H_out, _W_out = H_out, W_out

    kernel_c = relax.Constant(kernel_np)
    bias_c   = relax.Constant(bias_np)
    scale_c  = relax.Constant(scale_np)
    shift_c  = relax.Constant(shift_np)

    def te_conv(x_t, k_t, b_t, s_t, sh_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32", "mmalib_conv2d_i8",
                ins[0].data, ins[1].data, ins[2].data, ins[3].data, ins[4].data,
                outs[0].data,
                tir.IntImm("int32", _C_in), tir.IntImm("int32", _H_in),
                tir.IntImm("int32", _W_in), tir.IntImm("int32", _C_out),
                tir.IntImm("int32", _KH),   tir.IntImm("int32", _KW),
                tir.IntImm("int32", _stride), tir.IntImm("int32", _stride),
                tir.IntImm("int32", _padding), tir.IntImm("int32", _padding),
                tir.IntImm("int32", _padding), tir.IntImm("int32", _padding),
            )
        return te.extern(
            [1, _C_out, _H_out, _W_out],
            [x_t, k_t, b_t, s_t, sh_t],
            fcompute,
            name="conv_out",
            dtype="int8",
        )

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([1, C_in, H_in, W_in], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            out = bb.emit_te(
                te_conv, x_var, kernel_c, bias_c, scale_c, shift_c,
                primfunc_name_hint="mmalib_conv2d_i8",
            )
            result = bb.emit_output(out)
        bb.emit_func_output(result)
    return bb.finalize()


def _run_conv(dsp_mode, input_np, kernel_np, bias_np, scale_np, shift_np,
              C_in, H_in, W_in, C_out, KH, KW, stride, padding):
    mod = _build_conv_module(
        input_np, kernel_np, bias_np, scale_np, shift_np,
        C_in, H_in, W_in, C_out, KH, KW, stride, padding,
    )
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_np,
        target_string=target,
        execution_mode=dsp_mode,
        profile=False,
    )
    return results[f"{dsp_mode}_result"].reshape(1, C_out,
                                                  (H_in + 2*padding - KH)//stride + 1,
                                                  (W_in + 2*padding - KW)//stride + 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_oc_tile_consistency_3x3(dsp_mode):
    """Full C_out=128 call == two C_out=64 calls with sliced constants."""
    if dsp_mode == "c7x_dload":
        pytest.skip("tile-consistency is a compile-time property; c7x_host sufficient")

    rng = np.random.default_rng(0)
    C_in, H_in, W_in = 32, 14, 14
    C_out, KH, KW = 128, 3, 3
    stride, padding = 1, 1
    oc_tile = 64

    input_np  = rng.integers(-64, 64, (1, C_in, H_in, W_in), dtype=np.int8)
    kernel_np = rng.integers(-32, 32, (C_out, C_in, KH, KW), dtype=np.int8)
    bias_np   = rng.integers(-1000, 1000, C_out, dtype=np.int32)
    scale_np  = rng.integers(64, 200, C_out, dtype=np.uint8)
    shift_np  = rng.integers(0, 8, C_out, dtype=np.uint8)

    # Full call
    full = _run_conv(dsp_mode, input_np, kernel_np, bias_np, scale_np, shift_np,
                     C_in, H_in, W_in, C_out, KH, KW, stride, padding)

    # Tile 0: channels [0, 64)
    tile0 = _run_conv(dsp_mode, input_np,
                      kernel_np[:oc_tile], bias_np[:oc_tile],
                      scale_np[:oc_tile], shift_np[:oc_tile],
                      C_in, H_in, W_in, oc_tile, KH, KW, stride, padding)

    # Tile 1: channels [64, 128)
    tile1 = _run_conv(dsp_mode, input_np,
                      kernel_np[oc_tile:], bias_np[oc_tile:],
                      scale_np[oc_tile:], shift_np[oc_tile:],
                      C_in, H_in, W_in, oc_tile, KH, KW, stride, padding)

    assert np.array_equal(full[:, :oc_tile, :, :], tile0), \
        f"Tile 0 mismatch: max_diff={np.abs(full[:, :oc_tile].astype(int) - tile0.astype(int)).max()}"
    assert np.array_equal(full[:, oc_tile:, :, :], tile1), \
        f"Tile 1 mismatch: max_diff={np.abs(full[:, oc_tile:].astype(int) - tile1.astype(int)).max()}"


@pytest.mark.quick
def test_oc_tile_consistency_non_divisible(dsp_mode):
    """C_out=96 with oc_tile=64: tile 0 has 64ch, tile 1 has 32ch (tail tile)."""
    if dsp_mode == "c7x_dload":
        pytest.skip("c7x_host sufficient for tile-consistency check")

    rng = np.random.default_rng(1)
    C_in, H_in, W_in = 16, 8, 8
    C_out, KH, KW = 96, 3, 3
    stride, padding = 1, 1
    oc_tile = 64

    input_np  = rng.integers(-64, 64, (1, C_in, H_in, W_in), dtype=np.int8)
    kernel_np = rng.integers(-32, 32, (C_out, C_in, KH, KW), dtype=np.int8)
    bias_np   = rng.integers(-500, 500, C_out, dtype=np.int32)
    scale_np  = rng.integers(100, 180, C_out, dtype=np.uint8)
    shift_np  = rng.integers(0, 6, C_out, dtype=np.uint8)

    full = _run_conv(dsp_mode, input_np, kernel_np, bias_np, scale_np, shift_np,
                     C_in, H_in, W_in, C_out, KH, KW, stride, padding)

    tile0 = _run_conv(dsp_mode, input_np,
                      kernel_np[:oc_tile], bias_np[:oc_tile],
                      scale_np[:oc_tile], shift_np[:oc_tile],
                      C_in, H_in, W_in, oc_tile, KH, KW, stride, padding)

    tail = C_out - oc_tile  # = 32
    tile1 = _run_conv(dsp_mode, input_np,
                      kernel_np[oc_tile:], bias_np[oc_tile:],
                      scale_np[oc_tile:], shift_np[oc_tile:],
                      C_in, H_in, W_in, tail, KH, KW, stride, padding)

    assert np.array_equal(full[:, :oc_tile, :, :], tile0)
    assert np.array_equal(full[:, oc_tile:, :, :], tile1)
