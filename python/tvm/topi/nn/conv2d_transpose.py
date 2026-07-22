# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Transposed 2D convolution operators (sometimes called Deconvolution)."""
import collections

from tvm import te

from ..utils import simplify
from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n, f"Input can only have {n} elements, but got {len(x)} instead: {x}."
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def conv2d_transpose_nchw(
    Input, Filter, strides, padding, out_dtype, output_padding, dilation=(1, 1)
):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]

    strides : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : tuple of ints
        Used to get the right output shape for gradients

    dilation : tuple of two ints
        The spatial dilation along height and width

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return declaration_conv2d_transpose_impl(
        Input, Filter, strides, padding, out_dtype, output_padding=output_padding, dilation=dilation
    )


def conv2d_transpose_nchw_preprocess(
    data, kernel, strides, padding, out_dtype, output_padding, dilation=(1, 1)
):
    """Preprocess data and kernel to make the compute pattern
    of conv2d_transpose the same as conv2d"""
    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilation
    opad_h, opad_w = output_padding
    assert opad_h < stride_h and opad_w < stride_w
    # effective (dilated) filter size, used everywhere the raw filter size
    # would otherwise appear -- reduces to filter_h/filter_w when dilation=1
    eff_filter_h = (filter_h - 1) * dilation_h + 1
    eff_filter_w = (filter_w - 1) * dilation_w + 1
    # dilate data
    data_dilate = dilate(data, [1, 1, stride_h, stride_w], name="data_dilate")
    # pad data
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        padding, (eff_filter_h, eff_filter_w)
    )
    bpad_top = eff_filter_h - 1 - fpad_top
    bpad_bottom = eff_filter_h - 1 - fpad_bottom + opad_h
    bpad_left = eff_filter_w - 1 - fpad_left
    bpad_right = eff_filter_w - 1 - fpad_right + opad_w
    data_pad = pad(
        data_dilate, [0, 0, bpad_top, bpad_left], [0, 0, bpad_bottom, bpad_right], name="data_pad"
    )
    # transform kernel layout from IOHW to OIHW, and rotate kernel by 180 degrees
    kernel_transform = te.compute(
        (out_c, in_c, filter_h, filter_w),
        lambda o, i, h, w: kernel[i][o][filter_h - 1 - h][filter_w - 1 - w],
        name="kernel_transform",
    )
    return data_pad, kernel_transform


def declaration_conv2d_transpose_impl(
    data, kernel, strides, padding, out_dtype, output_padding, dilation=(1, 1)
):
    """Implementation of conv2d transpose"""
    data_pad, kernel_transform = conv2d_transpose_nchw_preprocess(
        data, kernel, strides, padding, out_dtype, output_padding, dilation=dilation
    )
    dilation_h, dilation_w = dilation
    batch, in_c, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape
    eff_filter_h = (filter_h - 1) * dilation_h + 1
    eff_filter_w = (filter_w - 1) * dilation_w + 1

    # convolution stage
    out_c = simplify(out_c)

    out_h = simplify(in_h - eff_filter_h + 1)
    out_w = simplify(in_w - eff_filter_w + 1)
    dc = te.reduce_axis((0, in_c), name="dc")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    Output = te.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: te.sum(
            data_pad[b, dc, h + dh * dilation_h, w + dw * dilation_w].astype(out_dtype)
            * kernel_transform[c, dc, dh, dw].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="conv2d_transpose_nchw",
    )

    return Output


def group_conv2d_transpose_nchw(
    data, kernel, stride, padding, out_dtype, output_padding, groups, dilation=(1, 1)
):
    """Group convolution operator in NCHW layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [in_channel, out_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : tuple of ints
        Used to get the right output shape for gradients

    groups : int
        number of groups

    dilation : int or a list/tuple of two ints
        Dilation size, or [dilation_height, dilation_width]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if groups == 1:
        return conv2d_transpose_nchw(
            data, kernel, stride, padding, out_dtype, output_padding, dilation=dilation
        )

    # some pre-processing and prelimnary checks
    if out_dtype is None:
        out_dtype = data.dtype

    batch, in_channels, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    assert (
        in_channels % groups == 0
    ), f"input channels {in_channels} must divide group size {groups}"
    # assert out_c % groups == 0, f"output channels {in_c} must divide group size {groups}"

    strides = _pair(stride)
    # padding = _pair(padding)
    # output_padding = _pair(output_padding)
    dilation = _pair(dilation)

    stride_h, stride_w = strides
    dilation_h, dilation_w = dilation
    opad_h, opad_w = output_padding
    assert (
        opad_h < stride_h and opad_w < stride_w
    ), f"[{output_padding}] opad_h:{opad_h} < stride_h:{stride_h} \
        and opad_w:{opad_w} < stride_w:{stride_w} does not satisfy."
    # effective (dilated) filter size, used everywhere the raw filter size
    # would otherwise appear -- reduces to filter_h/filter_w when dilation=1
    eff_filter_h = (filter_h - 1) * dilation_h + 1
    eff_filter_w = (filter_w - 1) * dilation_w + 1
    # dilate data
    data_dilate = dilate(data, [1, 1, stride_h, stride_w], name="data_dilate")
    # pad data
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        padding, (eff_filter_h, eff_filter_w)
    )
    bpad_top = eff_filter_h - 1 - fpad_top
    bpad_bottom = eff_filter_h - 1 - fpad_bottom + opad_h
    bpad_left = eff_filter_w - 1 - fpad_left
    bpad_right = eff_filter_w - 1 - fpad_right + opad_w
    data_pad = pad(
        data_dilate, [0, 0, bpad_top, bpad_left], [0, 0, bpad_bottom, bpad_right], name="data_pad"
    )
    # transform kernel layout from IOHW to OIHW, and rotate kernel by 180 degrees
    kernel_transform = te.compute(
        (out_c, in_channels, filter_h, filter_w),
        lambda i, o, h, w: kernel[o][i][filter_h - 1 - h][filter_w - 1 - w],
        name="kernel_transform",
    )

    batch, in_channels, in_h, in_w = data_pad.shape
    out_c, _, filter_h, filter_w = kernel_transform.shape

    # convolution stage
    out_channels = simplify(out_c * groups)

    out_h = simplify(in_h - eff_filter_h + 1)
    out_w = simplify(in_w - eff_filter_w + 1)
    dc = te.reduce_axis((0, in_channels // groups), name="dc")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    # data: batch, in_channels, out_h, out_w
    # weight: out_channels // G, in_channels, out_h, out_w
    return te.compute(
        (batch, out_channels, out_h, out_w),
        lambda b, c, h, w: te.sum(
            data_pad[
                b,
                c // (out_channels // groups) * (in_channels // groups) + dc,
                h + dh * dilation_h,
                w + dw * dilation_w,
            ].astype(out_dtype)
            * kernel_transform[
                c % (out_channels // groups),
                c // (out_channels // groups) * (in_channels // groups) + dc,
                dh,
                dw,
            ].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="group_conv2d_transpose_nchw",
    )
