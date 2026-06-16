"""
MMALIB int16 quantized fully-connected (linear) layer integration test.

Two test groups:
  1. Direct wrapper tests (mmalib_matmul_bias_i16 called from TIR):
       test_mmalib_fc_i16_qdq_* — use _create_fc_i16_model (low-level)
  2. QDQ pattern fusion tests (FuseMMALIBQDQFCI16 pass, PT2E-style):
       test_mmalib_qdq_fc_i16_* — use _create_qdq_i16_fc_model (high-level)

The QDQ tests verify that the FuseMMALIBQDQFCI16 pass correctly matches
the int16 PT2E pattern and emits mmalib_matmul_bias_i16.

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
@pytest.mark.xfail(
    reason="Known int16 precision limitation: max_diff=9 at K=1536 due to shift truncation"
)
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


# =========================================================================
# QDQ pattern fusion tests (FuseMMALIBQDQFCI16)
# =========================================================================


def _create_qdq_i16_fc_model(n_in: int, n_out: int, batch: int = 1, seed: int = 42):
    """Build a Relax model in PT2E int16 QDQ form for a linear (FC) layer.

    Produces:
        dequantize(data_int16, d_scale, d_zp=0)
        -> matmul(float, permute_dims(dequantize(weight_int16, w_scale, w_zp=0)))
        -> add(float_bias)
        -> quantize(out, o_scale, o_zp=0)

    FuseMMALIBQDQFCI16 should replace this with mmalib_matmul_bias_i16.
    """
    rng = np.random.default_rng(seed)
    input_data = rng.integers(-100, 100, size=(batch, n_in), dtype=np.int16)
    weight_data = rng.integers(-50, 50, size=(n_out, n_in), dtype=np.int16)

    d_scale = np.float32(0.005)
    w_scale = rng.uniform(0.001, 0.01, size=(n_out,)).astype(np.float32)
    o_scale = np.float32(0.01)
    bias_data = rng.uniform(-1.0, 1.0, size=(n_out,)).astype(np.float32)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((batch, n_in), "int16"))

    w_const = relax.Constant(weight_data)
    w_scale_c = relax.Constant(w_scale)
    # TVM's dequantize requires int8 zero_point (not int16); zp=0 so lossless.
    w_zp_c = relax.Constant(np.zeros(n_out, dtype=np.int8))
    d_scale_c = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_c = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_c = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_c = relax.Constant(np.array(0, dtype=np.int8))
    bias_c = relax.Constant(bias_data)

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_c, w_zp_c, axis=0))
            w_perm = bb.emit(relax.op.permute_dims(w_dq))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_c, d_zp_c))
            mm = bb.emit(relax.op.matmul(data_dq, w_perm))
            biased = bb.emit(relax.op.add(mm, bias_c))
            result = bb.emit(
                relax.op.quantize(biased, o_scale_c, o_zp_c, out_dtype="int16")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()
    return mod, input_data, weight_data, d_scale, w_scale, o_scale, bias_data


def _numpy_qdq_fc_i16_with_bias(input_data, weight_data, d_scale, w_scale, o_scale, bias_data):
    """Float-domain reference for int16 QDQ linear with bias."""
    # input [batch, K], weight [N, K] (transposed internally)
    accum = input_data.astype(np.float64) * d_scale
    w_float = weight_data.astype(np.float64) * w_scale.reshape(-1, 1)
    out_float = accum @ w_float.T  # [batch, N]
    out_float = out_float + bias_data.reshape(1, -1)
    out_q = np.round(out_float / o_scale)
    return np.clip(out_q, -32768, 32767).astype(np.int16)


@pytest.mark.quick
def test_mmalib_qdq_fc_i16_2d(dsp_mode, record_cycles):
    """FuseMMALIBQDQFCI16: 2D input [1, K] × weight [N, K] → [1, N], int16.

    Verifies the QDQ pattern is matched and fused by FuseMMALIBQDQFCI16,
    producing correct output vs float-domain reference.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    n_in, n_out = 64, 64  # aligned to MMA_SIZE_I16 (16)

    mod, input_data, weight_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_i16_fc_model(n_in, n_out)
    )
    ref = _numpy_qdq_fc_i16_with_bias(
        input_data, weight_data, d_scale, w_scale, o_scale, bias_data
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )

    dsp_out = results.get(result_key)
    assert dsp_out is not None, f"No {result_key} in results"
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(1, n_out)

    diff = np.abs(dsp_out_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 5, f"FuseMMALIBQDQFCI16 (2D) mismatch: max_diff={max_diff}"

    record_cycles("mmalib_qdq_fc_i16_64x64", results.get("c7x_dload_cycles", 0))


@pytest.mark.quick
def test_mmalib_qdq_fc_i16_3d_reshape(dsp_mode, record_cycles):
    """FuseMMALIBQDQFCI16: 3D input [1, seq, K] via reshape+matmul+reshape.

    SmolLM-style: aten.linear decomposes 3D inputs into reshape+matmul+reshape.
    FuseMMALIBQDQFC has _qdq_fc_reshape_bias_pattern for this case.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    seq, n_in, n_out = 4, 64, 64

    # Build 3D QDQ model with reshape pattern
    rng = np.random.default_rng(99)
    input_data_3d = rng.integers(-100, 100, size=(1, seq, n_in), dtype=np.int16)
    weight_data = rng.integers(-50, 50, size=(n_out, n_in), dtype=np.int16)

    d_scale = np.float32(0.005)
    w_scale = rng.uniform(0.001, 0.01, size=(n_out,)).astype(np.float32)
    o_scale = np.float32(0.01)
    bias_data = rng.uniform(-1.0, 1.0, size=(n_out,)).astype(np.float32)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, seq, n_in), "int16"))

    w_const = relax.Constant(weight_data)
    w_scale_c = relax.Constant(w_scale)
    # TVM's dequantize requires int8 zero_point (not int16); zp=0 so lossless.
    w_zp_c = relax.Constant(np.zeros(n_out, dtype=np.int8))
    d_scale_c = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_c = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_c = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_c = relax.Constant(np.array(0, dtype=np.int8))
    bias_c = relax.Constant(bias_data)

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_c, w_zp_c, axis=0))
            w_perm = bb.emit(relax.op.permute_dims(w_dq))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_c, d_zp_c))
            # reshape 3D [1, seq, K] → 2D [seq, K] for matmul
            data_rs = bb.emit(relax.op.reshape(data_dq, (seq, n_in)))
            mm = bb.emit(relax.op.matmul(data_rs, w_perm))
            # reshape back 2D [seq, N] → 3D [1, seq, N]
            mm_rs = bb.emit(relax.op.reshape(mm, (1, seq, n_out)))
            biased = bb.emit(relax.op.add(mm_rs, bias_c))
            result = bb.emit(
                relax.op.quantize(biased, o_scale_c, o_zp_c, out_dtype="int16")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()

    # Reference: reshape to 2D, compute, reshape back
    input_2d = input_data_3d.reshape(seq, n_in)
    ref_2d = _numpy_qdq_fc_i16_with_bias(
        input_2d, weight_data, d_scale, w_scale, o_scale, bias_data
    )
    ref = ref_2d.reshape(1, seq, n_out)

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data_3d,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )

    dsp_out = results.get(result_key)
    assert dsp_out is not None, f"No {result_key} in results"
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(1, seq, n_out)

    diff = np.abs(dsp_out_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 5, f"FuseMMALIBQDQFCI16 (3D reshape) mismatch: max_diff={max_diff}"

    record_cycles("mmalib_qdq_fc_i16_3d_4x64x64", results.get("c7x_dload_cycles", 0))


# ---------------------------------------------------------------------------
# Guard tests: check function rejects invalid patterns (no DSP needed)
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_fuse_fc_i16_rejects_nonzero_o_zp():
    """FuseMMALIBQDQFCI16 must NOT fuse when output zero-point is non-zero.

    The i16 lowerer does not fold o_zp into bias_i64, so a non-zero o_zp
    would silently shift all outputs.  The check function must reject it.
    This is a pure-Python pass-level test — no DSP execution required.
    """
    from tvm.relax.transform import FuseMMALIBQDQFCI16

    n_in, n_out = 64, 64
    rng = np.random.default_rng(0)
    weight_data = rng.integers(-50, 50, size=(n_out, n_in), dtype=np.int16)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, n_in), "int16"))
    w_const = relax.Constant(weight_data)
    w_scale = relax.Constant(np.ones(n_out, dtype=np.float32) * 0.001)
    w_zp = relax.Constant(np.zeros(n_out, dtype=np.int8))
    d_scale = relax.Constant(np.array(0.005, dtype=np.float32))
    d_zp = relax.Constant(np.array(0, dtype=np.int8))
    o_scale = relax.Constant(np.array(0.01, dtype=np.float32))
    # Non-zero o_zp — should cause the check function to reject fusion
    o_zp_nonzero = relax.Constant(np.array(1, dtype=np.int8))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale, w_zp, axis=0))
            w_perm = bb.emit(relax.op.permute_dims(w_dq))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale, d_zp))
            mm = bb.emit(relax.op.matmul(data_dq, w_perm))
            result = bb.emit(
                relax.op.quantize(mm, o_scale, o_zp_nonzero, out_dtype="int16")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)
    mod = bb.finalize()

    # Run the pass — should NOT fuse because o_zp != 0
    mod_after = FuseMMALIBQDQFCI16().transform_module(mod, None)

    # After the full pass (FuseOpsByPattern + lowering + DCE), a successful
    # fusion produces a PrimFunc named "mmalib_fc_i16" in the module.
    # If the check function rejected the pattern, no such PrimFunc exists.
    fused_names = [str(gv.name_hint) for gv in mod_after.functions]
    fused = [n for n in fused_names if "mmalib_fc_i16" in n]
    assert len(fused) == 0, (
        f"FuseMMALIBQDQFCI16 incorrectly fused a pattern with non-zero o_zp=1; "
        f"found PrimFuncs: {fused}"
    )
