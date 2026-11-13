"""Strided int8 QDQ conv2d through the -mmalib=1 pipeline, against a numpy reference.

MMALIB_CNN_convolveBias_row, as mmalib_conv2d_i8 calls it, aborts at stride 3 and
returns wrong values at stride 4 and above, so FuseMMALIBQDQConv2d declines those
strides and the scalar QDQ conv path computes them. The patch convs of ConvNeXt,
Swin (4x4/stride 4) and ViT (16x16/stride 16) are the cases that occur in real
models. ConvNeXt's 2x2/stride-2 downsampling conv stays on MMALIB.

Usage:
    pytest test_mmalib_conv2d_stride_dsp.py -v --dsp-mode=c7x_host
"""

import pytest
from test_mmalib_qdq_grouped_conv2d_i8_dsp import _check_qdq_grouped_conv2d


@pytest.mark.parametrize(
    "c_in,h,c_out,k,stride",
    [
        (3, 32, 64, 4, 4),
        (3, 64, 64, 16, 16),
        (3, 33, 64, 3, 3),
        (16, 16, 32, 2, 2),
    ],
    ids=["4x4_s4_patch", "16x16_s16_patch", "3x3_s3", "2x2_s2_downsample"],
)
def test_qdq_conv2d_stride(dsp_mode, c_in, h, c_out, k, stride):
    _check_qdq_grouped_conv2d(dsp_mode, c_in, h, h, c_out, k, k, 0, stride, 1)
