"""Unit tests for MMALIB QDQ scale/shift validation.

The MMALIB QDQ fusion lowerers previously accepted a per-tensor weight scale
(size-1) and silently produced a length-1 scale/shift array the kernels index
per output channel, and ``_float_to_scale_shift`` silently emitted
garbage for out-of-range rescale values.  Both cases must now decline
fusion instead of emitting a wrong ``mmalib_*`` call.

Pure Relax IR-level tests (no hardware / DSP build).
"""

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo

pytestmark = pytest.mark.quick


def _build_qdq_i16_conv2d_model(w_scale, o_scale):
    """dequant(data_i16) -> conv2d(_, dequant(w_i16)) -> relu -> quantize."""
    c_in, h, w = 2, 4, 4
    c_out, kh, kw = 2, 3, 3
    rng = np.random.default_rng(42)
    kernel = rng.integers(-100, 100, size=(c_out, c_in, kh, kw), dtype=np.int16)
    d_scale = np.float32(0.002)

    w_scale_arr = np.asarray(w_scale, dtype=np.float32)
    per_channel = w_scale_arr.ndim == 1

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int16"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale_arr)
    if per_channel:
        w_zp_const = relax.Constant(np.zeros(c_out, dtype=np.int8))
    else:
        w_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            if per_channel:
                w_dq = bb.emit(
                    relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0)
                )
            else:
                w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            conv = bb.emit(
                relax.op.nn.conv2d(
                    data_dq,
                    w_dq,
                    strides=(1, 1),
                    padding=(1, 1, 1, 1),
                    dilation=(1, 1),
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                )
            )
            relu_out = bb.emit(relax.op.nn.relu(conv))
            q = bb.emit(
                relax.op.quantize(relu_out, o_scale_const, o_zp_const, out_dtype="int16")
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestScaleShiftValidation:
    def test_per_tensor_weight_scale_declines(self):
        """A size-1 weight scale must not produce a length-1 scale/shift."""
        mod = _build_qdq_i16_conv2d_model(w_scale=np.float32(0.003), o_scale=np.float32(0.003))
        new_mod = tvm.relax.transform.FuseMMALIBQDQConv2dI16()(mod)
        assert "mmalib_conv2d_i16" not in new_mod.script()

    def test_out_of_range_rescale_declines(self):
        """A rescale below 2^-31 must decline rather than emit garbage."""
        mod = _build_qdq_i16_conv2d_model(
            w_scale=np.array([0.003, 0.003], dtype=np.float32), o_scale=np.float32(1e6)
        )
        new_mod = tvm.relax.transform.FuseMMALIBQDQConv2dI16()(mod)
        assert "mmalib_conv2d_i16" not in new_mod.script()
