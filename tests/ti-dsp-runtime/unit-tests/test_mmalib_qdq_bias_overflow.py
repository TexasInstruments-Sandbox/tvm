"""Regression tests for MMALIB QDQ bias-fold int32 overflow.

A collapsed input/weight scale (e.g. float32 eps for a constant tensor)
makes ``dw_scale = d_scale * w_scale`` tiny, so ``bias / dw_scale`` overflows
int32 during the accumulator-scale bias fold.  The fusion lowerers must
decline (fall back to the non-MMALIB path) instead of emitting a
``RuntimeWarning: invalid value encountered in cast`` and discarding a
garbage int32 bias.

Pure Relax IR-level tests (no hardware / DSP build).
"""

import warnings

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo
from tvm.relax.transform import (
    FuseMMALIBQDQConv2d,
    FuseMMALIBQDQDwConv2d,
    FuseMMALIBQDQFC,
)

pytestmark = pytest.mark.quick

# float32 epsilon: the scale a zero-range (constant) tensor collapses to.
_EPS = np.finfo(np.float32).eps


def _run_pass(pass_obj, mod):
    """Run a fusion pass, returning (out_mod, overflow_warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pass_obj(mod)
    overflow = [w for w in caught if "invalid value encountered in cast" in str(w.message)]
    return out, overflow


def _build_qdq_i8_conv2d_bias_model():
    """dequant(data) -> conv2d(_, dequant(w)) -> add(bias) -> quantize (int8)."""
    c_in, h, w = 4, 4, 4
    c_out, kh, kw = 4, 3, 3
    rng = np.random.default_rng(42)
    kernel = rng.integers(-100, 100, size=(c_out, c_in, kh, kw), dtype=np.int8)
    bias = np.array([0.5, -0.25, 0.75, -0.5], dtype=np.float32).reshape(1, c_out, 1, 1)

    w_scale = np.full(c_out, _EPS, dtype=np.float32)
    o_scale = np.float32(0.006)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int8"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(c_out, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(_EPS, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias)

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
            add = bb.emit(relax.op.add(conv, bias_const))
            q = bb.emit(relax.op.quantize(add, o_scale_const, o_zp_const, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_qdq_i8_dwconv2d_bias_model():
    """dequant(data) -> conv2d(groups=C) -> add(bias) -> quantize (int8)."""
    channels, h, w = 4, 6, 6
    kh, kw = 3, 3
    rng = np.random.default_rng(42)
    kernel = rng.integers(-100, 100, size=(channels, 1, kh, kw), dtype=np.int8)
    bias = np.array([0.5, -0.25, 0.75, -0.5], dtype=np.float32).reshape(1, channels, 1, 1)

    w_scale = np.full(channels, _EPS, dtype=np.float32)
    o_scale = np.float32(0.006)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, channels, h, w), "int8"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(channels, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(_EPS, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias)

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
            add = bb.emit(relax.op.add(conv, bias_const))
            q = bb.emit(relax.op.quantize(add, o_scale_const, o_zp_const, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_qdq_i8_fc_bias_model():
    """dequant(data) -> matmul(_, permute_dims(dequant(w))) -> add(bias) -> quantize (int8)."""
    m, k, n_out = 1, 32, 32
    rng = np.random.default_rng(42)
    kernel = rng.integers(-100, 100, size=(n_out, k), dtype=np.int8)
    bias = np.linspace(-0.5, 0.5, n_out, dtype=np.float32).reshape(1, n_out)

    w_scale = np.full(n_out, _EPS, dtype=np.float32)
    o_scale = np.float32(0.006)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((m, k), "int8"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(n_out, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(_EPS, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(o_scale, dtype=np.float32))
    o_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    bias_const = relax.Constant(bias)

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            w_dq = bb.emit(relax.op.dequantize(w_const, w_scale_const, w_zp_const, axis=0))
            w_perm = bb.emit(relax.op.permute_dims(w_dq, axes=[1, 0]))
            data_dq = bb.emit(relax.op.dequantize(x, d_scale_const, d_zp_const))
            mm = bb.emit(relax.op.matmul(data_dq, w_perm))
            add = bb.emit(relax.op.add(mm, bias_const))
            q = bb.emit(relax.op.quantize(add, o_scale_const, o_zp_const, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestMMALIBBiasFoldOverflow:
    def test_conv2d_bias_fold_overflow_declines(self):
        """Collapsed conv2d scales must decline without an int32 cast warning."""
        out, overflow = _run_pass(FuseMMALIBQDQConv2d(), _build_qdq_i8_conv2d_bias_model())
        assert "mmalib_conv2d_i8" not in out.script()
        assert not overflow, f"bias fold emitted RuntimeWarning: {overflow[0].message}"

    def test_dwconv2d_bias_fold_overflow_declines(self):
        """Collapsed depthwise scales must decline without an int32 cast warning."""
        out, overflow = _run_pass(FuseMMALIBQDQDwConv2d(), _build_qdq_i8_dwconv2d_bias_model())
        assert "mmalib_depthwise_conv2d_i8" not in out.script()
        assert not overflow, f"bias fold emitted RuntimeWarning: {overflow[0].message}"

    def test_fc_bias_fold_overflow_declines(self):
        """Collapsed FC scales must decline without an int32 cast warning."""
        out, overflow = _run_pass(FuseMMALIBQDQFC(), _build_qdq_i8_fc_bias_model())
        assert "mmalib_matmul_bias_i8" not in out.script()
        assert not overflow, f"bias fold emitted RuntimeWarning: {overflow[0].message}"
