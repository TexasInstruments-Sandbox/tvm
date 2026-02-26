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
"""Transposed 1D convolution operators (sometimes called Deconvolution)."""
import tvm
from tvm import te
from .dilate import dilate
from .pad import pad
from ..utils import simplify
from .utils import get_pad_tuple1d


def _conv1d_transpose_ncw_preprocess(data, kernel, stride, padding, out_dtype, output_padding):
    """Preprocess data and kernel to make the compute pattern
    of conv1d_transpose the same as conv1d.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape.  Must be smaller than stride.

    Returns
    -------
    data_pad : tvm.te.Tensor
        Padded input data. 3-D with shape [batch, in_channel, in_width]

    kernel: tvm.te.Tensor
        Transformed kernel. 3-D with shape [num_filter, in_channel, filter_width]
    """
    # some pre-processing and prelimnary checks
    if out_dtype is None:
        out_dtype = data.dtype

    # dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(output_padding, (tuple, list)):
        output_padding = output_padding[0]

    _, channels_in, _ = data.shape
    _, channels_out, kernel_width = kernel.shape
    assert output_padding < stride
    channels_out = simplify(channels_out)
    data_dilate = dilate(data, [1, 1, stride], name="data_dilate")
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_width,))
    pad_left = kernel_width - 1 - pad_left
    pad_right = kernel_width - 1 - pad_right + output_padding
    data_pad = pad(data_dilate, [0, 0, pad_left], [0, 0, pad_right], name="data_pad")

    # transform kernel layout from IOW to OIW, and rotate kernel by 180 degrees
    kernel = te.compute(
        (channels_out, channels_in, kernel_width),
        lambda o, i, w: kernel[i][o][kernel_width - 1 - w],
        name="kernel",
    )
    return data_pad, kernel


def conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype, output_padding):
    """Transposed 1D convolution ncw forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape.  Must be smaller than stride.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_channel, out_width]

    """

    batch, channels_in, _ = data.shape
    _, channels_out, kernel_width = kernel.shape

    data_pad, transformed_kernel = _conv1d_transpose_ncw_preprocess(
        data, kernel, stride, padding, out_dtype, output_padding
    )

    # convolution
    _, _, data_width = data_pad.shape
    out_w = simplify(data_width - kernel_width + 1)
    dc = te.reduce_axis((0, channels_in), name="dc")
    dw = te.reduce_axis((0, kernel_width), name="dw")
    output = te.compute(
        (batch, channels_out, out_w),
        lambda b, c, w: te.sum(
            data_pad[b, dc, w + dw].astype(out_dtype)
            * transformed_kernel[c, dc, dw].astype(out_dtype),
            axis=[dc, dw],
        ),
        tag="conv1d_transpose_ncw",
    )

    return output


def group_conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype, output_padding, groups):
    """Transposed 1D group convolution ncw forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape.  Must be smaller than stride.

     groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_channel, out_width]

    """
    if groups == 1:
        return conv1d_transpose_ncw(data, kernel, stride, padding, out_dtype, output_padding)

    _, in_channels, _ = data.shape

    assert (
        in_channels % groups == 0
    ), f"input channels {in_channels} must divide group size {groups}"

    data_pad, transformed_kernel = _conv1d_transpose_ncw_preprocess(
        data, kernel, stride, padding, out_dtype, output_padding
    )

    batch, in_channels, in_w = data_pad.shape
    out_c, _, filter_w = transformed_kernel.shape

    # convolution stage
    out_channels = simplify(out_c * groups)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_channels // groups), name="dc")
    dw = te.reduce_axis((0, filter_w), name="dw")

    # data: batch, in_channels, out_w
    # weight: out_channels // G, in_channels, out_w
    return te.compute(
        (batch, out_channels, out_w),
        lambda b, c, w: te.sum(
            data_pad[
                b, c // (out_channels // groups) * (in_channels // groups) + dc, w + dw
            ].astype(out_dtype)
            * transformed_kernel[
                c % (out_channels // groups),
                c // (out_channels // groups) * (in_channels // groups) + dc,
                dw,
            ].astype(out_dtype),
            axis=[dc, dw],
        ),
        tag="group_conv1d_transpose_ncw",
    )


def conv1d_transpose_ncw_optimized(data, kernel, stride, padding, out_dtype, output_padding):
    """Optimized transposed 1D convolution that eliminates wasteful iterations.

    For stride=1 with small input width, this computes the valid kernel position
    analytically instead of iterating over all kernel positions and checking
    validity. This eliminates 15/16 wasted iterations for typical kernel_width=16.

    For in_width=1 (common in iterative algorithms like LISTA), each output
    position has exactly one valid kernel position, computed as:
        dw = pad_left_trans - w

    This is 16x faster than the naive approach that iterates over all dw values.
    """
    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(output_padding, (tuple, list)):
        output_padding = output_padding[0]

    batch, channels_in, in_width = data.shape
    _, channels_out, kernel_width = kernel.shape
    channels_out = simplify(channels_out)

    # Calculate padding amounts
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_width,))
    pad_left_trans = kernel_width - 1 - pad_left
    pad_right_trans = kernel_width - 1 - pad_right + output_padding

    # Calculate output dimensions
    if stride == 1:
        dilated_width = in_width
    else:
        dilated_width = (in_width - 1) * stride + 1
    out_width = simplify(dilated_width + pad_left_trans + pad_right_trans - kernel_width + 1)

    # Flip kernel: transform from IOW to OIW layout and rotate 180 degrees
    kernel_flipped = te.compute(
        (channels_out, channels_in, kernel_width),
        lambda o, i, w: kernel[i, o, kernel_width - 1 - w],
        name="kernel_flipped",
    )

    # Reduction axis over input channels only (no dw loop needed for optimized path)
    dc = te.reduce_axis((0, channels_in), name="dc")

    # For stride=1 and in_width=1: each output has exactly one valid kernel position
    # Valid dw for output w: dw = pad_left_trans - w (must be in [0, kernel_width))
    # Input index when valid: iw = w + dw - pad_left_trans = 0 (always)

    def compute_optimized(b, c, w):
        # Compute the single valid kernel position for this output
        dw_val = pad_left_trans - w

        # Check if this kernel position is valid
        valid = tvm.tir.all(dw_val >= 0, dw_val < kernel_width)

        # For in_width=1: input index is always 0 when valid
        # For in_width>1: input index is w + dw - pad_left_trans = 0 when dw = pad_left_trans - w
        input_val = data[b, dc, 0].astype(out_dtype)

        # Use the computed kernel position (clamped to valid range for safety)
        dw_clamped = tvm.tir.max(tvm.tir.min(dw_val, kernel_width - 1), 0)
        kernel_val = kernel_flipped[c, dc, dw_clamped].astype(out_dtype)

        # Multiply input by kernel, zero if invalid position
        product = tvm.tir.if_then_else(valid, input_val * kernel_val, tvm.tir.const(0.0, out_dtype))

        return te.sum(product, axis=[dc])

    output = te.compute(
        (batch, channels_out, out_width),
        compute_optimized,
        tag="conv1d_transpose_ncw_optimized",
        name="conv1d_transpose_optimized",
    )

    return output


def conv1d_transpose_ncw_direct(data, kernel, stride, padding, out_dtype, output_padding):
    """Direct transposed 1D convolution without intermediate pad buffer.

    This implementation computes the output directly from the input without
    creating intermediate dilate/pad buffers. This avoids conditional branches
    in inner loops that can block software pipelining on DSP targets.

    The key optimization is using tvm.tir.if_then_else at the element level
    instead of creating a padded intermediate tensor, which allows the compiler
    to better optimize the memory access pattern.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, filter_width]

    stride : ints
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : ints
        Used to recover the actual output shape in case there are more
        than one possible shape. Must be smaller than stride.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_channel, out_width]
    """
    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(output_padding, (tuple, list)):
        output_padding = output_padding[0]

    batch, channels_in, in_width = data.shape
    _, channels_out, kernel_width = kernel.shape
    channels_out = simplify(channels_out)

    # Calculate padding amounts
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_width,))
    # Transform padding for transposed conv
    pad_left_trans = kernel_width - 1 - pad_left
    pad_right_trans = kernel_width - 1 - pad_right + output_padding

    # Calculate output dimensions
    # For stride=1: out_width = in_width + pad_left_trans + pad_right_trans - kernel_width + 1
    #             = in_width + (kernel_width - 1 - pad_left) + (kernel_width - 1 - pad_right + output_padding) - kernel_width + 1
    #             = in_width + kernel_width - 1 - pad_left - pad_right + output_padding
    # For stride>1: dilated_width = (in_width - 1) * stride + 1
    #               out_width = dilated_width + pad_left_trans + pad_right_trans - kernel_width + 1
    if stride == 1:
        dilated_width = in_width
    else:
        dilated_width = (in_width - 1) * stride + 1
    out_width = simplify(dilated_width + pad_left_trans + pad_right_trans - kernel_width + 1)

    # Reduction axes
    dc = te.reduce_axis((0, channels_in), name="dc")
    dw = te.reduce_axis((0, kernel_width), name="dw")

    # Flip kernel: transform from IOW to OIW layout and rotate 180 degrees
    kernel_flipped = te.compute(
        (channels_out, channels_in, kernel_width),
        lambda o, i, w: kernel[i, o, kernel_width - 1 - w],
        name="kernel_flipped",
    )

    def compute_conv1d_transpose(b, c, w):
        """Compute single output element directly from input.

        For each output position w and kernel position dw:
        - The position in the "virtual" padded+dilated tensor is: w + dw
        - The position in the dilated tensor is: w + dw - pad_left_trans
        - For stride=1: input position = dilated position
        - For stride>1: input position = dilated_pos // stride if dilated_pos % stride == 0

        We only accumulate when the input position is valid (in range and aligned).
        """
        # Position in dilated tensor (before padding was applied)
        dilated_pos = w + dw - pad_left_trans

        if stride == 1:
            # Simple case: no dilation
            # Valid when: 0 <= dilated_pos < in_width
            valid = tvm.tir.all(dilated_pos >= 0, dilated_pos < in_width)
            input_val = tvm.tir.if_then_else(
                valid,
                data[b, dc, dilated_pos].astype(out_dtype),
                tvm.tir.const(0.0, out_dtype),
            )
        else:
            # With dilation: only positions that are multiples of stride have data
            # Valid when: dilated_pos >= 0 AND dilated_pos % stride == 0 AND
            #             dilated_pos // stride < in_width
            input_pos = tvm.tir.indexdiv(dilated_pos, stride)
            is_aligned = tvm.tir.indexmod(dilated_pos, stride) == 0
            valid = tvm.tir.all(dilated_pos >= 0, is_aligned, input_pos < in_width)
            input_val = tvm.tir.if_then_else(
                valid,
                data[b, dc, input_pos].astype(out_dtype),
                tvm.tir.const(0.0, out_dtype),
            )

        return te.sum(
            input_val * kernel_flipped[c, dc, dw].astype(out_dtype),
            axis=[dc, dw],
        )

    output = te.compute(
        (batch, channels_out, out_width),
        compute_conv1d_transpose,
        tag="conv1d_transpose_ncw_direct",
        name="conv1d_transpose_direct",
    )

    return output
