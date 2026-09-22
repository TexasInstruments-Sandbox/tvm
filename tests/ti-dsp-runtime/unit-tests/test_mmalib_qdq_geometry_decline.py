"""Regression tests for MMALIB QDQ conv2d/depthwise-conv2d geometry declines.

Three real MMALIB_CNN_* kernel limitations, confirmed against the vendor
source and reproduced on real c7x_dload hardware (10/79 model failures in
quantized/test_quantized_torchvision.py --mmalib), all surfacing as an
abrupt DSP abort ("Function call failed") rather than a compile-time
decline:

  1. Regular conv2d (mmalib_conv2d_i8): stride>1 in both dims requires
     padTop==0 or padBottom in (0, (KH-1)//2) -- AlexNet's conv1 (11x11,
     stride 4, pad=2/2 symmetric) satisfies neither.
  2. Depthwise conv2d (mmalib_depthwise_conv2d_i8): KH*KW*sizeof(int8)
     must fit in one 32-byte MMA B-panel row -- ConvNeXt's 7x7 depthwise
     (49 > 32) never fits, regardless of stride/padding.
  3. Depthwise conv2d, stride==2: H_in+padTop+padBottom must be even --
     one EfficientNet-b1/b3/b4/b5 layer lands on H_in=15, pad=2 (sum=19,
     odd).

The fusion passes must decline these geometries at compile time (fall
back to the existing, already-proven scalar/non-MMALIB QDQ conv2d path)
instead of emitting a call_extern that aborts the DSP at runtime.

Pure Relax IR-level tests (no hardware / DSP build) -- see
test_mmalib_qdq_bias_overflow.py for the same pattern.
"""

import numpy as np
import pytest

from tvm import relax
from tvm.relax import TensorStructInfo
from tvm.relax.transform import FuseMMALIBQDQConv2d, FuseMMALIBQDQDwConv2d

pytestmark = pytest.mark.quick

_O_SCALE = np.float32(0.006)


def _build_qdq_conv2d_model(
    c_in, h, w, c_out, kh, kw, stride, padding, groups=1
):
    """dequant(data) -> conv2d(_, dequant(w)) -> add(bias) -> quantize (int8)."""
    rng = np.random.default_rng(42)
    c_in_per_group = c_in // groups
    kernel = rng.integers(-100, 100, size=(c_out, c_in_per_group, kh, kw), dtype=np.int8)
    bias = np.linspace(-0.5, 0.5, c_out, dtype=np.float32).reshape(1, c_out, 1, 1)
    w_scale = np.full(c_out, 0.01, dtype=np.float32)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int8"))
    w_const = relax.Constant(kernel)
    w_scale_const = relax.Constant(w_scale)
    w_zp_const = relax.Constant(np.zeros(c_out, dtype=np.int8))
    d_scale_const = relax.Constant(np.array(0.01, dtype=np.float32))
    d_zp_const = relax.Constant(np.array(0, dtype=np.int8))
    o_scale_const = relax.Constant(np.array(_O_SCALE, dtype=np.float32))
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
                    strides=(stride, stride),
                    padding=padding,
                    dilation=(1, 1),
                    groups=groups,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                )
            )
            add = bb.emit(relax.op.add(conv, bias_const))
            q = bb.emit(relax.op.quantize(add, o_scale_const, o_zp_const, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_qdq_dwconv2d_model(channels, h, w, kh, kw, stride, padding):
    """dequant(data) -> conv2d(groups=C) -> add(bias) -> quantize (int8)."""
    return _build_qdq_conv2d_model(
        channels, h, w, channels, kh, kw, stride, padding, groups=channels
    )


class TestMMALIBConv2dGeometryDecline:
    def test_stride4_asymmetric_pad_declines(self):
        """AlexNet conv1 shape: 11x11/stride4/pad=2 symmetric -- neither
        padTop==0 nor padBottom in (0, (KH-1)//2=5) -- must decline."""
        mod = _build_qdq_conv2d_model(
            c_in=3, h=224, w=224, c_out=64, kh=11, kw=11, stride=4, padding=(2, 2, 2, 2)
        )
        out = FuseMMALIBQDQConv2d()(mod)
        assert "mmalib_conv2d_i8" not in out.script()

    def test_stride2_same_padding_still_fuses(self):
        """Ordinary stride-2 conv with exact SAME-derived padding
        ((KH-1)//2 == 1 for a 3x3 kernel) must still fuse -- regression
        guard against being overly conservative."""
        mod = _build_qdq_conv2d_model(
            c_in=3, h=32, w=32, c_out=16, kh=3, kw=3, stride=2, padding=(1, 1, 1, 1)
        )
        out = FuseMMALIBQDQConv2d()(mod)
        assert "mmalib_conv2d_i8" in out.script()

    def test_stride2_zero_padding_still_fuses(self):
        """padTop==0 branch of the rule must still fuse regardless of KH."""
        mod = _build_qdq_conv2d_model(
            c_in=3, h=35, w=35, c_out=16, kh=3, kw=3, stride=2, padding=(0, 0, 0, 0)
        )
        out = FuseMMALIBQDQConv2d()(mod)
        assert "mmalib_conv2d_i8" in out.script()

    def test_valid_stride1_wide_kernel_declines(self):
        """Inception_v3's exact second-stem-conv shape: 3x3/stride1,
        zero padding ("VALID"). MMALIB's own validColsOut formula is
        padding-agnostic, so this trips validColsOut > H_out*W_out --
        must decline."""
        mod = _build_qdq_conv2d_model(
            c_in=32, h=149, w=149, c_out=32, kh=3, kw=3, stride=1, padding=(0, 0, 0, 0)
        )
        out = FuseMMALIBQDQConv2d()(mod)
        assert "mmalib_conv2d_i8" not in out.script()

    def test_same_padding_stride1_still_fuses(self):
        """Ordinary SAME-padded 3x3/stride1 conv (the overwhelmingly
        common case across the whole torchvision sweep) must still
        fuse -- regression guard against being overly conservative."""
        mod = _build_qdq_conv2d_model(
            c_in=32, h=149, w=149, c_out=32, kh=3, kw=3, stride=1, padding=(1, 1, 1, 1)
        )
        out = FuseMMALIBQDQConv2d()(mod)
        assert "mmalib_conv2d_i8" in out.script()

    def test_1x1_stride1_no_padding_still_fuses(self):
        """1x1 pointwise stride1 conv (used everywhere) must still fuse
        regardless of padding -- the (KW-1) term is zero, so it never
        trips the stride==1 rule."""
        mod = _build_qdq_conv2d_model(
            c_in=256, h=56, w=56, c_out=64, kh=1, kw=1, stride=1, padding=(0, 0, 0, 0)
        )
        out = FuseMMALIBQDQConv2d()(mod)
        assert "mmalib_conv2d_i8" in out.script()


class TestMMALIBDwConv2dGeometryDecline:
    def test_7x7_exceeds_mma_panel_declines(self):
        """ConvNeXt depthwise shape: 7x7/stride1 -- Ni*Fr*Fc*sizeof(int8)
        = 49 > the 32-byte MMA panel row width -- must decline regardless
        of allowed_kh_sizes including 7."""
        mod = _build_qdq_dwconv2d_model(channels=96, h=56, w=56, kh=7, kw=7, stride=1, padding=(3, 3, 3, 3))
        out = FuseMMALIBQDQDwConv2d()(mod)
        assert "mmalib_depthwise_conv2d_i8" not in out.script()

    def test_5x5_stride2_odd_featuremap_declines(self):
        """EfficientNet-b1 shape: 5x5/stride2, H_in=15/pad=2 -> sum=19
        (odd) -- must decline."""
        mod = _build_qdq_dwconv2d_model(channels=672, h=15, w=15, kh=5, kw=5, stride=2, padding=(2, 2, 2, 2))
        out = FuseMMALIBQDQDwConv2d()(mod)
        assert "mmalib_depthwise_conv2d_i8" not in out.script()

    def test_5x5_stride2_even_featuremap_still_fuses(self):
        """Same shape but H_in=16/pad=2 -> sum=20 (even) -- must still
        fuse -- regression guard isolating the odd-sum trigger."""
        mod = _build_qdq_dwconv2d_model(channels=672, h=16, w=16, kh=5, kw=5, stride=2, padding=(2, 2, 2, 2))
        out = FuseMMALIBQDQDwConv2d()(mod)
        assert "mmalib_depthwise_conv2d_i8" in out.script()

    def test_3x3_still_fuses(self):
        """Ordinary 3x3/stride1 depthwise (mobilenet-style) must be
        unaffected by either new check."""
        mod = _build_qdq_dwconv2d_model(channels=32, h=112, w=112, kh=3, kw=3, stride=1, padding=(1, 1, 1, 1))
        out = FuseMMALIBQDQDwConv2d()(mod)
        assert "mmalib_depthwise_conv2d_i8" in out.script()
