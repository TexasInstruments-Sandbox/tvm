"""
MMALIB int16 QDQ depthwise conv2d integration test.

Tests the FuseMMALIBQDQDwConv2dI16 pass that matches the int16 PT2E pattern:
    dequantize(data_i16) -> conv2d(groups=C, dequantize(weight_i16)) -> add(bias) -> quantize

Fused into a single mmalib_depthwise_conv2d_i16 call with per-group bias (int64),
scale (uint8), and shift (uint8).

Produced by C7xMMAQuantizer(dtype="int16") on MobileNet V2/V3 style models.

Note: MMALIB only supports 3×3 kernels for INT16 depthwise (MMALIB-882 tracks
5×5/7×7). The check function in FuseMMALIBQDQDwConv2dI16 enforces this constraint.

Usage:
    pytest test_mmalib_dwconv_i16_dsp.py -v --dsp-mode=c7x_host
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


def _create_qdq_i16_dwconv2d_model(
    channels: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    seed: int = 42,
):
    """Build a Relax model in PT2E int16 QDQ form for a depthwise conv2d layer.

    Produces:
        dequantize(data_int16, d_scale, d_zp=0)   [symmetric, d_zp always 0]
        -> conv2d(float, dequantize(weight_int16, w_scale, w_zp=0), groups=channels)
        -> add(float_bias)
        -> quantize(out, o_scale, o_zp=0)

    This mirrors what C7xMMAQuantizer(dtype="int16") produces for a depthwise layer.
    """
    rng = np.random.default_rng(seed)

    input_data = rng.integers(-100, 100, size=(1, channels, h, w), dtype=np.int16)
    # Weights: [channels, 1, KH, KW] for depthwise (groups=channels)
    kernel_data = rng.integers(-50, 50, size=(channels, 1, kh, kw), dtype=np.int16)

    # Quantization parameters — all zero-points are 0 (symmetric)
    d_scale = np.float32(0.002)
    w_scale = rng.uniform(0.001, 0.005, size=(channels,)).astype(np.float32)
    o_scale = np.float32(0.003)
    bias_data = rng.uniform(-0.5, 0.5, size=(channels,)).astype(np.float32)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, channels, h, w), "int16"))

    w_const = relax.Constant(kernel_data)
    w_scale_const = relax.Constant(w_scale)
    # TVM's dequantize requires int8 zero-point even for int16 data.
    # For symmetric int16 quantization, zp=0 in int8 is equivalent.
    w_zp_const = relax.Constant(np.zeros(channels, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias_data.reshape(1, channels, 1, 1))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            # Dequantize weight (per-channel, axis=0)
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            # Dequantize data (per-tensor)
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            # Depthwise conv2d: groups=channels, Ni=1, No=1 per group
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
                relax.op.quantize(biased, o_scale_const, o_zp_const, out_dtype="int16")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize(), input_data, kernel_data, d_scale, w_scale, o_scale, bias_data


# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------


def _numpy_qdq_dwconv2d_i16(
    input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, stride, padding
):
    """Float-domain reference for int16 QDQ depthwise conv2d (symmetric, zp=0)."""
    _, channels, h_in, w_in = input_data.shape
    _, _, kh, kw = kernel_data.shape

    h_out = (h_in + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    # The QDQ graph computes per-channel float-domain depthwise convolution:
    # float_data = int16_data * d_scale
    # float_weight[ch] = int16_weight[ch, 0] * w_scale[ch]
    # float_conv[ch] = depthwise_conv(float_data[ch], float_weight[ch])
    # int16_out = round(clip((float_conv + float_bias) / o_scale, -32768, 32767))

    padded = np.pad(
        input_data.astype(np.int64),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    result = np.zeros((1, channels, h_out, w_out), dtype=np.float64)
    for ch in range(channels):
        for oh in range(h_out):
            for ow in range(w_out):
                hs, he = oh * stride, oh * stride + kh
                ws, we = ow * stride, ow * stride + kw
                patch = padded[0, ch, hs:he, ws:we]
                acc = np.sum(patch * kernel_data[ch, 0].astype(np.int64))
                result[0, ch, oh, ow] = acc * d_scale * w_scale[ch]

    result += bias_data.reshape(1, channels, 1, 1)
    result /= o_scale
    return np.clip(np.round(result), -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_mmalib_dwconv_i16_qdq_3x3_stride1(dsp_mode, record_cycles):
    """Int16 QDQ depthwise conv2d — 3×3 kernel, stride 1.

    Verifies FuseMMALIBQDQDwConv2dI16 fuses and produces correct output.
    max_diff ≤ 5 accounts for uint8 scale/shift requantization approximation.
    """
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    channels, h, w = 32, 28, 28
    kh, kw, stride, padding = 3, 3, 1, 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_i16_dwconv2d_model(channels, h, w, kh, kw, stride, padding)
    )
    ref = _numpy_qdq_dwconv2d_i16(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, stride, padding
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
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(1, channels, h, w)

    diff = np.abs(dsp_out_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 5, f"MMALIB i16 dwconv (3x3 stride1): max_diff={max_diff}"

    record_cycles("mmalib_dwconv_i16_3x3_stride1", results.get("c7x_dload_cycles", 0))


@pytest.mark.quick
def test_mmalib_dwconv_i16_qdq_3x3_stride2(dsp_mode, record_cycles):
    """Int16 QDQ depthwise conv2d — 3×3 kernel, stride 2."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    channels, h, w = 32, 28, 28
    kh, kw, stride, padding = 3, 3, 2, 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_i16_dwconv2d_model(channels, h, w, kh, kw, stride, padding)
    )
    ref = _numpy_qdq_dwconv2d_i16(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, stride, padding
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
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    dsp_out_i16 = dsp_out.astype(np.int16).reshape(1, channels, h_out, w_out)

    diff = np.abs(dsp_out_i16.astype(np.int32) - ref.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 5, f"MMALIB i16 dwconv (3x3 stride2): max_diff={max_diff}"

    record_cycles("mmalib_dwconv_i16_3x3_stride2", results.get("c7x_dload_cycles", 0))


# NOTE: 5×5 and 7×7 INT16 depthwise are NOT supported by MMALIB
# (MMALIB-882: only 3×3 is implemented for INT16 in convolve_col_smallNo_highPrecision).
# FuseMMALIBQDQDwConv2dI16 rejects 5×5/7×7 INT16 at compile time; those layers
# fall through to generic float32 processing.  No test needed here.
