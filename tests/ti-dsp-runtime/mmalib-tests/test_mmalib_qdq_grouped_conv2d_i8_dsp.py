"""MMALIB int8 grouped conv2d QDQ *pass* integration test.

Unlike test_mmalib_conv2d_i8_grouped_loop_dsp.py and
test_mmalib_loop_only_chain_dsp.py (which build a lowered call_extern to
mmalib_conv2d_i8_grouped_loop by hand), this test builds the PT2E QDQ
pattern directly (dequantize -> conv2d(groups>1) -> add -> quantize) and
lets the standard `-mmalib=1` pipeline run FuseMMALIBQDQConv2d itself --
exercising the actual Python-side lowering logic this diff added
(_check_mmalib_qdq_conv2d's allow_groups path, the C_in = C_in_per_group *
groups reconstruction, and the per-channel weight-sum/zero-point-correction
computation in _MMALIBQDQLowerer._lower) end-to-end, the way ResNeXt101's
real graph actually exercises it.

See docs/dsp/quantized_model_optimization.md Step 13.

Usage:
    pytest test_mmalib_qdq_grouped_conv2d_i8_dsp.py -v --dsp-mode=c7x_host
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


def _create_qdq_grouped_conv2d_model(
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kh: int,
    kw: int,
    padding: int,
    stride: int,
    groups: int,
    seed: int = 42,
):
    """PT2E QDQ form (pre-FuseQDQToInt8Conv2D), groups>1:
    dequantize(data_int8, d_scale, d_zp=0)
      -> conv2d(float_data, dequantize(weight_int8, w_scale, w_zp=0), groups)
      -> add(float_bias) -> quantize(out, o_scale, o_zp=0)
    """
    rng = np.random.default_rng(seed)
    c_in_g = c_in // groups

    input_data = rng.integers(-4, 4, size=(1, c_in, h, w), dtype=np.int8)
    kernel_data = rng.integers(-4, 4, size=(c_out, c_in_g, kh, kw), dtype=np.int8)

    d_scale = np.float32(0.05)
    w_scale = rng.uniform(0.01, 0.1, size=(c_out,)).astype(np.float32)
    o_scale = np.float32(0.08)
    bias_data = rng.uniform(-1.0, 1.0, size=(c_out,)).astype(np.float32)

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
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            conv = bb.emit(
                relax.op.nn.conv2d(
                    data_dq,
                    w_dq,
                    strides=(stride, stride),
                    padding=(padding, padding, padding, padding),
                    dilation=(1, 1),
                    groups=groups,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                )
            )
            biased = bb.emit(relax.op.add(conv, bias_const))
            result = bb.emit(relax.op.quantize(biased, o_scale_const, o_zp_const, out_dtype="int8"))
            bb.emit_output(result)
        bb.emit_func_output(result)

    mod = bb.finalize()
    return mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data


def _numpy_qdq_grouped_conv2d_i8(
    input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding, stride, groups
):
    """Numpy reference for the PT2E QDQ grouped conv2d (symmetric quant, zp=0)."""
    _, _, h_in, w_in = input_data.shape
    c_out, c_in_g, kh, kw = kernel_data.shape
    c_out_g = c_out // groups

    h_out = (h_in + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    padded = np.pad(
        input_data.astype(np.int32),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    conv_i32 = np.zeros((1, c_out, h_out, w_out), dtype=np.int32)
    for g in range(groups):
        ci0 = g * c_in_g
        co0 = g * c_out_g
        for co in range(c_out_g):
            for ci in range(c_in_g):
                for oh in range(h_out):
                    for ow in range(w_out):
                        ih0, iw0 = oh * stride, ow * stride
                        patch = padded[0, ci0 + ci, ih0 : ih0 + kh, iw0 : iw0 + kw]
                        conv_i32[0, co0 + co, oh, ow] += int(
                            np.sum(patch * kernel_data[co0 + co, ci].astype(np.int32))
                        )

    float_conv = conv_i32.astype(np.float64) * d_scale * w_scale.reshape(1, c_out, 1, 1)
    float_biased = float_conv + bias_data.reshape(1, c_out, 1, 1)
    float_out = float_biased / o_scale
    rounded = np.round(float_out)
    clipped = np.clip(rounded, -128, 127)
    return clipped.astype(np.int8)


def _check_qdq_grouped_conv2d(dsp_mode, c_in, h, w, c_out, kh, kw, padding, stride, groups):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    mod, input_data, kernel_data, d_scale, w_scale, o_scale, bias_data = (
        _create_qdq_grouped_conv2d_model(c_in, h, w, c_out, kh, kw, padding, stride, groups)
    )
    ref_output = _numpy_qdq_grouped_conv2d_i8(
        input_data, kernel_data, d_scale, w_scale, o_scale, bias_data, padding, stride, groups
    )
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    target_mmalib = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_mmalib,
        execution_mode=dsp_mode,
        profile=False,
    )
    dsp_output = results.get(result_key)
    assert dsp_output is not None, f"No {result_key} returned"
    dsp_output_i8 = dsp_output.astype(np.int8).reshape(1, c_out, h_out, w_out)

    diff = np.abs(dsp_output_i8.astype(np.int32) - ref_output.astype(np.int32))
    max_diff = int(diff.max())
    assert max_diff <= 2, (
        f"MMALIB grouped QDQ conv2d mismatch (c_in={c_in} c_out={c_out} "
        f"groups={groups} stride={stride}): max_diff={max_diff}"
    )


@pytest.mark.quick
def test_qdq_grouped_conv2d_stride1(dsp_mode):
    """groups=32, stride=1 -- exercises FuseMMALIBQDQConv2d's allow_groups
    eligibility path and _emit_grouped_loop end-to-end."""
    _check_qdq_grouped_conv2d(dsp_mode, 64, 14, 14, 64, 3, 3, 1, 1, 32)


@pytest.mark.quick
def test_qdq_grouped_conv2d_stride2(dsp_mode):
    """groups=32, stride=2 stage-transition shape, same as
    layer4.0.conv2 in real ResNeXt101-32x8d."""
    _check_qdq_grouped_conv2d(dsp_mode, 64, 14, 14, 128, 3, 3, 1, 2, 32)
