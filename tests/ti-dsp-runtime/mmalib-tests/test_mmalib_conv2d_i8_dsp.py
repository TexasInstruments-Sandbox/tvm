"""
MMALIB int8 quantized conv2d integration test.

Tests the MMALIB QDQ fusion path that matches the PT2E quantization pattern:
    dequantize(data) -> conv2d(float, dequantize(weight)) -> add(bias) -> quantize
Fused into a single MMALIB call with per-channel bias/scale/shift.

Usage:
    pytest test_mmalib_conv2d_i8_dsp.py -v --dsp-mode=c7x_host
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


def _create_qdq_conv2d_model(
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kh: int,
    kw: int,
    padding: int,
    seed: int = 42,
):
    """Create a Relax model in PT2E QDQ form (pre-FuseQDQToInt8Conv2D).

    Produces:
        dequantize(data_int8, d_scale, d_zp=0)
        -> conv2d(float_data, dequantize(weight_int8, w_scale, w_zp=0))
        -> add(float_bias)
        -> quantize(out, o_scale, o_zp=0)
    """
    rng = np.random.default_rng(seed)

    # int8 input and weights
    input_data = rng.integers(-4, 4, size=(1, c_in, h, w), dtype=np.int8)
    kernel_data = rng.integers(-4, 4, size=(c_out, c_in, kh, kw), dtype=np.int8)

    # Quantization parameters (symmetric: zp=0)
    d_scale = np.float32(0.05)
    w_scale = rng.uniform(0.01, 0.1, size=(c_out,)).astype(np.float32)
    o_scale = np.float32(0.08)

    # Float bias (in real-value domain, as PT2E produces)
    bias_data = rng.uniform(-1.0, 1.0, size=(c_out,)).astype(np.float32)

    # Build Relax IR in PT2E QDQ form
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int8"))

    w_const = relax.Constant(kernel_data)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(c_out, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias_data.reshape(1, c_out, 1, 1))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            # Dequantize weight (per-channel, axis=0)
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            # Dequantize data (per-tensor, scalar scale/zp)
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            # Float conv2d
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
            # Add float bias
            biased = bb.emit(relax.op.add(conv, bias_const))
            # Quantize output (per-tensor)
            result = bb.emit(
                relax.op.quantize(biased, o_scale_const, o_zp_const, out_dtype="int8")
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()
    return mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data


def _numpy_qdq_conv2d_i8(input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding):
    """Numpy reference for the PT2E QDQ conv2d (symmetric quant, zp=0)."""
    _, c_in, h_in, w_in = input_data.shape
    c_out, _, kh, kw = kernel_data.shape

    h_out = h_in + 2 * padding - kh + 1
    w_out = w_in + 2 * padding - kw + 1

    # Float-domain computation (what the original QDQ graph computes):
    # float_data = int8_data * d_scale
    # float_weight = int8_weight * w_scale[ch]
    # float_conv[ch] = sum(float_data * float_weight[ch])
    #                = sum(int8_data * int8_weight[ch]) * d_scale * w_scale[ch]
    # float_out = float_conv + float_bias
    # int8_out = round(float_out / o_scale)

    padded = np.pad(
        input_data.astype(np.int32),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    conv_i32 = np.zeros((1, c_out, h_out, w_out), dtype=np.int32)
    for co in range(c_out):
        for ci in range(c_in):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch = padded[0, ci, oh : oh + kh, ow : ow + kw]
                    conv_i32[0, co, oh, ow] += np.sum(
                        patch * kernel_data[co, ci].astype(np.int32)
                    )

    # Scale to float domain
    float_conv = conv_i32.astype(np.float64) * d_scale * w_scale.reshape(1, c_out, 1, 1)
    float_biased = float_conv + bias_data.reshape(1, c_out, 1, 1)

    # Requantize
    float_out = float_biased / o_scale
    rounded = np.round(float_out)
    clipped = np.clip(rounded, -128, 127)
    return clipped.astype(np.int8)


def test_mmalib_conv2d_i8_qdq(dsp_mode, record_cycles):
    """Test int8 conv2d with and without MMALIB, report cycle comparison."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    # ResNet-18 layer2 size: 64->64 channels, 56x56 spatial, 3x3 kernel
    c_in, h, w = 64, 56, 56
    c_out, kh, kw = 64, 3, 3
    padding = 1

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_conv2d_model(c_in, h, w, c_out, kh, kw, padding)
    )

    ref_output = _numpy_qdq_conv2d_i8(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"

    # --- Run WITH MMALIB ---
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
    dsp_output_i8 = dsp_output.astype(np.int8).reshape(1, c_out, h, w)

    diff = np.abs(dsp_output_i8.astype(np.int32) - ref_output.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 2, f"MMALIB int8 conv2d mismatch: max_diff={max_diff}"

    mmalib_cycles = results_mmalib.get("c7x_dload_cycles", 0)
    record_cycles("mmalib_conv2d_i8_qdq", mmalib_cycles)

    # --- Run WITHOUT MMALIB (loop-based, only on hardware) ---
    loop_cycles = 0
    if dsp_mode == "c7x_dload":
        # Plain int8 conv2d without QDQ (avoids layout conversion issues)
        rng = np.random.default_rng(42)
        plain_input = rng.integers(-4, 4, size=(1, c_in, h, w), dtype=np.int8)
        plain_kernel = rng.integers(-4, 4, size=(c_out, c_in, kh, kw), dtype=np.int8)

        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int8"))
        w_const = relax.Constant(plain_kernel)
        with bb.function("main", [x], attrs={"num_input": 1}):
            with bb.dataflow():
                out = bb.emit(
                    relax.op.nn.conv2d(
                        x, w_const, strides=(1, 1),
                        padding=(padding, padding, padding, padding),
                        dilation=(1, 1), groups=1,
                        data_layout="NCHW", kernel_layout="OIHW",
                        out_dtype="int8",
                    )
                )
                bb.emit_output(out)
            bb.emit_func_output(out)
        plain_mod = bb.finalize()

        target_loop = get_target_string(dsp_mode, use_cpp_api=True)
        results_loop = compile_and_run_dsp(
            mod=plain_mod,
            input_data=plain_input,
            target_string=target_loop,
            execution_mode=dsp_mode,
            profile=True,
        )
        loop_cycles = results_loop.get("c7x_dload_cycles", 0)
        record_cycles("loop_conv2d_i8", loop_cycles)

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Conv2d i8 (64x56x56 -> 64x3x3) cycle comparison:")
    print(f"  MMALIB (MMA coprocessor): {mmalib_cycles:>12,} cycles")
    if loop_cycles:
        print(f"  Loop-based (C7x scalar):  {loop_cycles:>12,} cycles")
        print(f"  Speedup:                  {loop_cycles / max(mmalib_cycles, 1):>12.1f}x")
    print(f"{'='*60}")
