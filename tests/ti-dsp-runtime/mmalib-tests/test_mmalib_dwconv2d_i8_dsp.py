"""
MMALIB int8 quantized depthwise conv2d integration test.

Tests the MMALIB QDQ fusion path for depthwise convolution (groups=C_in):
    dequantize(data) -> conv2d(groups=C) -> add(bias) -> quantize
Fused into a single MMALIB depthwise call with per-channel bias/scale/shift.

Usage:
    pytest test_mmalib_dwconv2d_i8_dsp.py -v --dsp-mode=c7x_host
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


def _create_qdq_dwconv2d_model(
    channels: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    seed: int = 42,
):
    """Create a Relax model with depthwise conv2d in PT2E QDQ form.

    Produces:
        dequantize(data_int8, d_scale, d_zp=0)
        -> conv2d(groups=channels, float_data, dequantize(weight_int8, w_scale, w_zp=0))
        -> add(float_bias)
        -> quantize(out, o_scale, o_zp=0)
    """
    rng = np.random.default_rng(seed)

    # int8 input and depthwise weights [C_out, 1, KH, KW]
    input_data = rng.integers(-4, 4, size=(1, channels, h, w), dtype=np.int8)
    kernel_data = rng.integers(-4, 4, size=(channels, 1, kh, kw), dtype=np.int8)

    # Quantization parameters (symmetric: zp=0)
    d_scale = np.float32(0.05)
    w_scale = rng.uniform(0.01, 0.1, size=(channels,)).astype(np.float32)
    o_scale = np.float32(0.08)

    # Float bias
    bias_data = rng.uniform(-1.0, 1.0, size=(channels,)).astype(np.float32)

    # Build Relax IR
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, channels, h, w), "int8"))

    w_const = relax.Constant(kernel_data)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(channels, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias_data.reshape(1, channels, 1, 1))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            conv = bb.emit(
                relax.op.nn.conv2d(
                    data_dq,
                    w_dq,
                    strides=(stride, stride),
                    padding=(padding, padding, padding, padding),
                    dilation=(1, 1),
                    groups=channels,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                )
            )
            biased = bb.emit(relax.op.add(conv, bias_const))
            result = bb.emit(
                relax.op.quantize(biased, o_scale_const, o_zp_const, out_dtype="int8")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()
    return mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data


def _numpy_qdq_dwconv2d_i8(
    input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, stride, padding
):
    """Numpy reference for PT2E QDQ depthwise conv2d (symmetric quant, zp=0)."""
    _, channels, h_in, w_in = input_data.shape
    _, _, kh, kw = kernel_data.shape

    h_out = (h_in + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    padded = np.pad(
        input_data.astype(np.int32),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    conv_i32 = np.zeros((1, channels, h_out, w_out), dtype=np.int32)
    for ch in range(channels):
        for oh in range(h_out):
            for ow in range(w_out):
                ih = oh * stride
                iw = ow * stride
                patch = padded[0, ch, ih : ih + kh, iw : iw + kw]
                conv_i32[0, ch, oh, ow] = np.sum(
                    patch * kernel_data[ch, 0].astype(np.int32)
                )

    # Scale to float domain
    float_conv = conv_i32.astype(np.float64) * d_scale * w_scale.reshape(1, channels, 1, 1)
    float_biased = float_conv + bias_data.reshape(1, channels, 1, 1)

    # Requantize
    float_out = float_biased / o_scale
    rounded = np.round(float_out)
    clipped = np.clip(rounded, -128, 127)
    return clipped.astype(np.int8)


@pytest.mark.c7x_only
def test_mmalib_dwconv2d_i8_qdq(dsp_mode, record_cycles):
    """Test int8 depthwise conv2d 3x3 stride 1 via MMALIB."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    channels, h, w = 64, 56, 56
    kh, kw = 3, 3
    stride, padding = 1, 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_dwconv2d_model(channels, h, w, kh, kw, stride, padding)
    )

    ref_output = _numpy_qdq_dwconv2d_i8(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, stride, padding
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

    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    dsp_output_i8 = dsp_output.astype(np.int8).reshape(1, channels, h_out, w_out)

    diff = np.abs(dsp_output_i8.astype(np.int32) - ref_output.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 2, f"MMALIB depthwise conv2d mismatch: max_diff={max_diff}"

    mmalib_cycles = results_mmalib.get("c7x_dload_cycles", 0)
    record_cycles("mmalib_dwconv2d_i8_3x3_s1", mmalib_cycles)

    print(f"\n{'='*60}")
    print(f"Depthwise conv2d i8 ({channels}ch, {h}x{w}, 3x3 s1):")
    print(f"  MMALIB: {mmalib_cycles:>12,} cycles")
    print(f"  max_diff = {max_diff}")
    print(f"{'='*60}")


@pytest.mark.c7x_only
def test_mmalib_dwconv2d_i8_qdq_stride2(dsp_mode, record_cycles):
    """Test int8 depthwise conv2d 3x3 stride 2 via MMALIB."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    channels, h, w = 128, 28, 28
    kh, kw = 3, 3
    stride, padding = 2, 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_dwconv2d_model(channels, h, w, kh, kw, stride, padding)
    )

    ref_output = _numpy_qdq_dwconv2d_i8(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, stride, padding
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

    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    dsp_output_i8 = dsp_output.astype(np.int8).reshape(1, channels, h_out, w_out)

    diff = np.abs(dsp_output_i8.astype(np.int32) - ref_output.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 2, f"MMALIB depthwise conv2d stride2 mismatch: max_diff={max_diff}"

    mmalib_cycles = results_mmalib.get("c7x_dload_cycles", 0)
    record_cycles("mmalib_dwconv2d_i8_3x3_s2", mmalib_cycles)

    print(f"\n{'='*60}")
    print(f"Depthwise conv2d i8 ({channels}ch, {h}x{w}, 3x3 s2):")
    print(f"  MMALIB: {mmalib_cycles:>12,} cycles")
    print(f"  max_diff = {max_diff}")
    print(f"{'='*60}")
