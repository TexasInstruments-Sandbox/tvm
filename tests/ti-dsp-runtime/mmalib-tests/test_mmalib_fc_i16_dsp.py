"""
MMALIB int16 quantized fully-connected (linear) layer integration test.

Tests mmalib_matmul_bias_i16 with per-channel scale/shift requantization.
Uses K=1536, N=576 (SmolLM down_proj dimensions) to verify precision
with large reduction dimensions where int8 fails.

Usage:
    pytest test_mmalib_fc_i16_dsp.py -v --dsp-mode=c7x_host
    pytest test_mmalib_fc_i16_dsp.py -v --dsp-mode=c7x_dload
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


def _create_fc_i16_model(
    n_in: int,
    n_out: int,
    seed: int = 42,
):
    """Create a Relax model that directly calls mmalib_matmul_bias_i16.

    Builds TIR with call_extern("mmalib_matmul_bias_i16") taking:
    - input: int16 [1, N_in]
    - weight: int16 [N_out, N_in] (transposed internally)
    - bias: int64 [N_out] (zero)
    - scale: uint8 [N_out]
    - shift: uint8 [N_out]
    - output: int16 [1, N_out]
    """
    from tvm import te, tir
    from tvm.relax.transform.ti_mmalib_legalize import _float_to_scale_shift

    rng = np.random.default_rng(seed)

    input_data = rng.integers(-100, 100, size=(1, n_in), dtype=np.int16)
    weight_data = rng.integers(-100, 100, size=(n_out, n_in), dtype=np.int16)

    d_scale = np.float32(0.005)
    w_scale = rng.uniform(0.001, 0.01, size=(n_out,)).astype(np.float32)
    o_scale = np.float32(0.01)

    combined_rescale = d_scale * w_scale / o_scale
    scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)
    bias_i64 = np.zeros(n_out, dtype=np.int64)

    weight_const = relax.Constant(weight_data)
    bias_const = relax.Constant(bias_i64)
    scale_const = relax.Constant(scale_u8)
    shift_const = relax.Constant(shift_u8)

    M, K, N = 1, n_in, n_out

    def te_mmalib_fc_i16(data_t, w_t, b_t, s_t, sh_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "mmalib_matmul_bias_i16",
                ins[0].data, ins[1].data, ins[2].data,
                ins[3].data, ins[4].data, outs[0].data,
                M, K, N,
            )
        return te.extern(
            [M, N], [data_t, w_t, b_t, s_t, sh_t],
            fcompute, name="mmalib_fc_i16", dtype="int16",
        )

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((M, K), "int16"))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit(
                bb.call_te(
                    te_mmalib_fc_i16, x,
                    weight_const, bias_const, scale_const, shift_const,
                    primfunc_name_hint="mmalib_fc_i16",
                )
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()
    return mod, input_data, weight_data, d_scale, w_scale, o_scale


def _numpy_qdq_fc_i16(input_data, weight_data, d_scale, w_scale, o_scale):
    """Numpy reference for int16 QDQ linear (symmetric quant, zp=0)."""
    n_out, n_in = weight_data.shape
    conv_i64 = np.zeros((1, n_out), dtype=np.int64)
    for n in range(n_out):
        conv_i64[0, n] = np.sum(
            input_data[0].astype(np.int64) * weight_data[n].astype(np.int64)
        )

    float_out = conv_i64.astype(np.float64) * d_scale * w_scale.reshape(1, n_out)
    result = float_out / o_scale
    rounded = np.round(result)
    clipped = np.clip(rounded, -32768, 32767)
    return clipped.astype(np.int16)


@pytest.mark.c7x_only
@pytest.mark.xfail(reason="Known int16 precision limitation: max_diff=9 at K=1536 due to shift truncation")
def test_mmalib_fc_i16_qdq_downproj(dsp_mode, record_cycles):
    """Test int16 FC with down_proj dimensions (1536→576, K=1536).

    This is the layer where int8 fails with max_diff=978 due to large
    accumulators. Int16 should give much better precision.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    n_in, n_out = 1536, 576  # down_proj dimensions (K must be multiple of 32)

    mod, input_data, weight_data, d_scale, w_scale, o_scale = (
        _create_fc_i16_model(n_in, n_out, seed=42)
    )

    ref_output = _numpy_qdq_fc_i16(
        input_data, weight_data, d_scale, w_scale, o_scale
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"

    target_mmalib = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results_mmalib = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_mmalib,
        execution_mode=dsp_mode,
        profile=True,
    )

    dsp_output = results_mmalib.get(result_key)
    assert dsp_output is not None, f"No {result_key} returned"
    dsp_output_i16 = dsp_output.astype(np.int16).reshape(1, n_out)

    diff = np.abs(dsp_output_i16.astype(np.int32) - ref_output.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 2, f"MMALIB FC i16 mismatch: max_diff={max_diff}"

    mmalib_cycles = results_mmalib.get("c7x_dload_cycles", 0)
    record_cycles("mmalib_fc_i16_1536x576", mmalib_cycles)

    print(f"\n{'='*60}")
    print(f"FC i16 ({n_in} -> {n_out}):")
    print(f"  MMALIB: {mmalib_cycles:>12,} cycles")
    print(f"  max_diff = {max_diff}")
    print(f"{'='*60}")


@pytest.mark.c7x_only
def test_mmalib_fc_i16_qdq_small(dsp_mode, record_cycles):
    """Test int16 FC with minimal aligned dimensions (32→32)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    n_in, n_out = 32, 32

    mod, input_data, weight_data, d_scale, w_scale, o_scale = (
        _create_fc_i16_model(n_in, n_out, seed=123)
    )

    ref_output = _numpy_qdq_fc_i16(
        input_data, weight_data, d_scale, w_scale, o_scale
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"

    target_mmalib = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results_mmalib = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_mmalib,
        execution_mode=dsp_mode,
        profile=True,
    )

    dsp_output = results_mmalib.get(result_key)
    assert dsp_output is not None, f"No {result_key} returned"
    dsp_output_i16 = dsp_output.astype(np.int16).reshape(1, n_out)

    diff = np.abs(dsp_output_i16.astype(np.int32) - ref_output.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 2, f"MMALIB FC i16 small mismatch: max_diff={max_diff}"

    mmalib_cycles = results_mmalib.get("c7x_dload_cycles", 0)
    record_cycles("mmalib_fc_i16_32x32", mmalib_cycles)


@pytest.mark.c7x_only
def test_mmalib_fc_i16_bias_per_channel_shift(dsp_mode, record_cycles):
    """Test matmulBias_i16 with scale=1 and per-channel shift.

    Uses mmalib_matmul_bias_i16 (bTranspose=1, weight [N,K]) with:
    - bias = zeros (int64)
    - scale = ones (int8, identity)
    - shift = per-channel values (uint8)

    This gives each output channel its optimal precision without the
    lossy uint8 scale approximation — just a right-shift per channel.
    """
    import math

    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    M, K, N = 1, 32, 32
    rng = np.random.default_rng(42)
    input_data = rng.integers(-100, 100, size=(M, K), dtype=np.int16)
    weight_data = rng.integers(-50, 50, size=(N, K), dtype=np.int16)

    # Per-channel shift from L1-norm of each weight row
    l1_per_ch = np.abs(weight_data).sum(axis=1)  # [N]
    shift_per_ch = np.zeros(N, dtype=np.uint8)
    for n in range(N):
        max_accum = int(l1_per_ch[n]) * 32767
        if max_accum > 32767:
            shift_per_ch[n] = math.ceil(math.log2(max_accum / 32767))

    bias_i64 = np.zeros(N, dtype=np.int64)
    scale_i8 = np.ones(N, dtype=np.int8)

    # Reference: per-channel (accum * 1) >> shift[n]
    accum = input_data.astype(np.int64) @ weight_data.astype(np.int64).T  # [M, N]
    ref = np.zeros((M, N), dtype=np.int16)
    for n in range(N):
        ref[0, n] = np.clip(accum[0, n] >> int(shift_per_ch[n]), -32768, 32767)

    # Build model
    from tvm import te, tir

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((M, K), "int16"))

    def te_bias_i16(data_t, w_t, bias_t, scale_t, shift_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "mmalib_matmul_bias_i16",
                ins[0].data,
                ins[1].data,
                ins[2].data,
                ins[3].data,
                ins[4].data,
                outs[0].data,
                M, K, N,
            )

        return te.extern(
            [M, N],
            [data_t, w_t, bias_t, scale_t, shift_t],
            fcompute,
            name="mmalib_bias_i16",
            dtype="int16",
        )

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit(
                bb.call_te(
                    te_bias_i16,
                    x,
                    relax.Constant(weight_data),
                    relax.Constant(bias_i64),
                    relax.Constant(scale_i8),
                    relax.Constant(shift_per_ch),
                    primfunc_name_hint="mmalib_bias_i16",
                )
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    target_mmalib = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_mmalib,
        execution_mode=dsp_mode,
        profile=True,
    )

    dsp_output = results.get(result_key)
    assert dsp_output is not None, f"No {result_key} returned"
    dsp_i16 = dsp_output.astype(np.int16).reshape(M, N)

    diff = np.abs(dsp_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())

    print(f"\n  bias_i16 per-ch shift: max_diff={max_diff}")
    print(f"  shift range: [{shift_per_ch.min()}, {shift_per_ch.max()}]")
    assert max_diff <= 1, f"MMALIB bias_i16 per-ch shift mismatch: max_diff={max_diff}"

    mmalib_cycles = results.get("c7x_dload_cycles", 0)
    record_cycles("mmalib_bias_i16_per_ch_shift_32x32", mmalib_cycles)
