"""
MMALIB int16 QDQ conv2d integration test.

Tests the FuseMMALIBQDQConv2dI16 pass that matches the int16 PT2E pattern:
    dequantize(data_i16) -> conv2d(float, dequantize(weight_i16)) -> add(bias) -> quantize

Fused into a single mmalib_conv2d_i16 call with per-channel bias (int64),
scale (uint8), and shift (uint8).

Produced by C7xMMAQuantizer(dtype="int16") on e.g. ResNet-class models.

Usage:
    pytest test_mmalib_conv2d_i16_dsp.py -v --dsp-mode=c7x_host
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

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _create_qdq_i16_conv2d_model(
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kh: int,
    kw: int,
    padding: int,
    seed: int = 42,
):
    """Create a Relax model in PT2E int16 QDQ form.

    Produces:
        dequantize(data_int16, d_scale, d_zp=0)  [symmetric, d_zp always 0]
        -> conv2d(float_data, dequantize(weight_int16, w_scale, w_zp=0))
        -> add(float_bias)
        -> quantize(out, o_scale, o_zp=0)

    This mirrors what C7xMMAQuantizer(dtype="int16") produces for a Conv2d layer.
    """
    rng = np.random.default_rng(seed)

    input_data = rng.integers(-100, 100, size=(1, c_in, h, w), dtype=np.int16)
    # Weights as int16 (C7xMMAQuantizer quantizes weights to int16)
    kernel_data = rng.integers(-100, 100, size=(c_out, c_in, kh, kw), dtype=np.int16)

    # Quantization parameters — all zero-points are 0 (symmetric)
    d_scale = np.float32(0.002)
    w_scale = rng.uniform(0.001, 0.005, size=(c_out,)).astype(np.float32)
    o_scale = np.float32(0.003)

    bias_data = rng.uniform(-1.0, 1.0, size=(c_out,)).astype(np.float32)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int16"))

    w_const = relax.Constant(kernel_data)
    w_scale_const = relax.Constant(w_scale)
    # TVM's dequantize requires zero_point to be int8 (not int16).
    # For symmetric int16 quant, zp is always 0 so int8(0) is correct.
    w_zp_const = relax.Constant(np.zeros(c_out, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias_data.reshape(1, c_out, 1, 1))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            conv = bb.emit(
                relax.op.nn.conv2d(
                    data_dq,
                    w_dq,
                    strides=(1, 1),
                    padding=(padding, padding, padding, padding),
                    dilation=(1, 1),
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                )
            )
            biased = bb.emit(relax.op.add(conv, bias_const))
            result = bb.emit(
                relax.op.quantize(biased, o_scale_const, o_zp_const, out_dtype="int16")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()
    return mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data


# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------


def _numpy_qdq_conv2d_i16(input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding):
    """Float-domain reference for the int16 QDQ conv2d (symmetric quant, zp=0)."""
    _, c_in, h_in, w_in = input_data.shape
    c_out, _, kh, kw = kernel_data.shape

    h_out = h_in + 2 * padding - kh + 1
    w_out = w_in + 2 * padding - kw + 1

    # The QDQ graph computes:
    # float_data = int16_data * d_scale
    # float_weight[ch] = int16_weight[ch] * w_scale[ch]
    # float_conv[ch] = sum(float_data * float_weight[ch]) + float_bias[ch]
    # int16_out = round(clip(float_conv / o_scale, -32768, 32767))
    padded = np.pad(
        input_data.astype(np.int64),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    conv_i64 = np.zeros((1, c_out, h_out, w_out), dtype=np.int64)
    for co in range(c_out):
        for ci in range(c_in):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch = padded[0, ci, oh : oh + kh, ow : ow + kw]
                    conv_i64[0, co, oh, ow] += np.sum(
                        patch * kernel_data[co, ci].astype(np.int64)
                    )

    float_conv = conv_i64.astype(np.float64) * d_scale * w_scale.reshape(1, c_out, 1, 1)
    float_biased = float_conv + bias_data.reshape(1, c_out, 1, 1)
    float_out = float_biased / o_scale
    rounded = np.round(float_out)
    clipped = np.clip(rounded, -32768, 32767)
    return clipped.astype(np.int16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_mmalib_conv2d_i16_qdq_small(dsp_mode, record_cycles):
    """Single-layer int16 QDQ conv2d — small ResNet-style dimensions.

    Verifies FuseMMALIBQDQConv2dI16 produces correct output vs the
    float-domain reference. max_diff ≤ 5 due to uint8 scale/shift
    requantization approximation.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    c_in, h, w = 32, 28, 28
    c_out, kh, kw = 32, 3, 3
    padding = 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_i16_conv2d_model(c_in, h, w, c_out, kh, kw, padding)
    )

    ref = _numpy_qdq_conv2d_i16(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding
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
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(1, c_out, h, w)

    diff = np.abs(dsp_out_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 5, f"MMALIB int16 conv2d mismatch: max_diff={max_diff}"

    record_cycles("mmalib_conv2d_i16_qdq_small", results.get("c7x_dload_cycles", 0))


@pytest.mark.quick
def test_mmalib_conv2d_i16_qdq_resnet_layer(dsp_mode, record_cycles):
    """ResNet-18 layer1 size: 64ch 56x56 3x3 — typical production shape."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    c_in, h, w = 64, 56, 56
    c_out, kh, kw = 64, 3, 3
    padding = 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_i16_conv2d_model(c_in, h, w, c_out, kh, kw, padding)
    )

    ref = _numpy_qdq_conv2d_i16(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding
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
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(1, c_out, h, w)

    diff = np.abs(dsp_out_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())
    # Threshold ≤10: for K=C_in×KH×KW=576, uint8 scale/shift requantization
    # introduces ≈√576≈24 max error in the worst case; observed ≈6 in practice.
    assert max_diff <= 10, f"MMALIB int16 conv2d (64ch 56x56) mismatch: max_diff={max_diff}"

    record_cycles("mmalib_conv2d_i16_qdq_64ch", results.get("c7x_dload_cycles", 0))
