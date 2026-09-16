"""Unit tests for int16 MMALIB QDQ ReLU fusion.

The int16 conv2d/depthwise-conv2d QDQ fusion lowerers previously emitted a
full-range ``[-32768, 32767]`` clip for the ReLU variant.  On int16 that
clip is a no-op, so ReLU was silently dropped (negative pre-activations
survived).  The fix clips at the output zero-point instead: ``[0, 32767]``
(int16 is symmetric, so ``o_zp=0`` is enforced in the check functions).

Pure Relax IR-level tests (no hardware / DSP build), mirroring
``test_relu_pass.py``: build a small PT2E-style QDQ model, run the fusion
pass, and inspect the emitted clip bounds in ``mod.script()``.
"""

import re

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo

pytestmark = pytest.mark.quick


def _clip_bounds(text):
    """Return the two scalar bounds of the fused ``R.clip(x, lo, hi)`` call."""
    m = re.search(r"R\.clip\([^,]+,\s*([^,]+),\s*([^)]+)\)", text)
    assert m is not None, f"R.clip not found in fused IR:\n{text}"
    return m.group(1).strip(), m.group(2).strip()


def _build_qdq_i16_conv2d_relu_model():
    """dequant(data_i16) -> conv2d(_, dequant(w_i16)) -> relu -> quantize."""
    c_in, h, w = 2, 4, 4
    c_out, kh, kw = 2, 3, 3
    rng = np.random.default_rng(42)
    kernel = rng.integers(-100, 100, size=(c_out, c_in, kh, kw), dtype=np.int16)
    d_scale = np.float32(0.002)
    w_scale = rng.uniform(0.001, 0.005, size=(c_out,)).astype(np.float32)
    o_scale = np.float32(0.003)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int16"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(c_out, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
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


def _build_qdq_i16_dwconv2d_relu_model():
    """dequant(data_i16) -> depthwise conv2d -> relu -> quantize."""
    channels, h, w = 2, 4, 4
    kh, kw = 3, 3
    rng = np.random.default_rng(42)
    kernel = rng.integers(-50, 50, size=(channels, 1, kh, kw), dtype=np.int16)
    d_scale = np.float32(0.002)
    w_scale = rng.uniform(0.001, 0.005, size=(channels,)).astype(np.float32)
    o_scale = np.float32(0.003)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, channels, h, w), "int16"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(channels, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(d_scale, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            conv = bb.emit(
                relax.op.nn.conv2d(
                    data_dq,
                    w_dq,
                    strides=(1, 1),
                    padding=(1, 1, 1, 1),
                    dilation=(1, 1),
                    groups=channels,
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


class TestConv2dI16Relu:
    def test_relu_clips_at_output_zero_point(self):
        mod = _build_qdq_i16_conv2d_relu_model()
        new_mod = tvm.relax.transform.FuseMMALIBQDQConv2dI16()(mod)
        text = new_mod.script()
        assert "mmalib_conv2d_i16" in text
        assert "R.nn.relu" not in text
        lo, hi = _clip_bounds(text)
        assert "0" in lo and "32768" not in lo
        assert "32767" in hi


class TestDwConv2dI16Relu:
    def test_relu_clips_at_output_zero_point(self):
        mod = _build_qdq_i16_dwconv2d_relu_model()
        new_mod = tvm.relax.transform.FuseMMALIBQDQDwConv2dI16()(mod)
        text = new_mod.script()
        assert "mmalib_depthwise_conv2d_i16" in text
        assert "R.nn.relu" not in text
        lo, hi = _clip_bounds(text)
        assert "0" in lo and "32768" not in lo
        assert "32767" in hi
