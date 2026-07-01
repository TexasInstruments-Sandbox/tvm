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
#Begin TI
"""ROI Align operator"""
from tvm import te, tir
from tvm.topi.utils import get_const_tuple


def roi_align(data, rois, pooled_size, spatial_scale, sample_ratio, mode, spatial_offset=0.5):
    """ROI align operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]
    rois : tvm.te.Tensor
        2-D with shape [num_roi, 5] with format [batch_idx, x1, y1, x2, y2]
    pooled_size : tuple of ints
        Output height and width
    spatial_scale : float
        Spatial scale factor
    sample_ratio : int
        Sampling ratio (0 for adaptive)
    mode : int
        0 for average, 1 for max
    spatial_offset : float
        Pixel shift for coordinate transformation (default 0.5 for half_pixel mode)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_roi, channel, pooled_size_h, pooled_size_w]
    """
    avg_mode = mode == 0
    dtype = rois.dtype
    _, channel, height, width = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    if sample_ratio > 0:
        max_grid_size = sample_ratio
    else:
        max_grid_size = max(int(height / pooled_size_h) + 2, int(width / pooled_size_w) + 2)

    def _bilinear(i, c, y, x):
        outside = tir.any(y < -1.0, x < -1.0, y > height, x > width)
        y_clamped = te.min(te.max(y, 0.0), height - 1)
        x_clamped = te.min(te.max(x, 0.0), width - 1)

        y_low = y_clamped.astype("int32")
        x_low = x_clamped.astype("int32")
        y_high = y_low + 1
        x_high = x_low + 1

        y_high = te.min(y_high, height - 1)
        x_high = te.min(x_high, width - 1)

        wy_h = y_clamped - y_low.astype(dtype)
        wx_h = x_clamped - x_low.astype(dtype)
        wy_l = 1.0 - wy_h
        wx_l = 1.0 - wx_h

        val = (
            wx_l * wy_l * data[i, c, y_low, x_low]
            + wx_h * wy_l * data[i, c, y_low, x_high]
            + wx_l * wy_h * data[i, c, y_high, x_low]
            + wx_h * wy_h * data[i, c, y_high, x_high]
        )
        return tir.if_then_else(outside, tir.const(0.0, dtype), val)

    def _sample(i, c, ph, pw):
        roi = rois[i]
        batch_index = roi[0].astype("int32")
        roi_start_w = roi[1] * spatial_scale - spatial_offset
        roi_start_h = roi[2] * spatial_scale - spatial_offset
        roi_end_w = roi[3] * spatial_scale - spatial_offset
        roi_end_h = roi[4] * spatial_scale - spatial_offset

        roi_h = te.max(roi_end_h - roi_start_h, tir.const(1.0, dtype))
        roi_w = te.max(roi_end_w - roi_start_w, tir.const(1.0, dtype))

        bin_h = roi_h / pooled_size_h
        bin_w = roi_w / pooled_size_w

        if sample_ratio > 0:
            roi_bin_grid_h = roi_bin_grid_w = tir.const(sample_ratio, "int32")
        else:
            roi_bin_grid_h = te.ceil(roi_h / pooled_size_h).astype("int32")
            roi_bin_grid_w = te.ceil(roi_w / pooled_size_w).astype("int32")

        rh = te.reduce_axis((0, max_grid_size), name="rh")
        rw = te.reduce_axis((0, max_grid_size), name="rw")

        y_start = roi_start_h + ph.astype(dtype) * bin_h
        x_start = roi_start_w + pw.astype(dtype) * bin_w

        valid = (rh < roi_bin_grid_h) & (rw < roi_bin_grid_w)
        count = (roi_bin_grid_h * roi_bin_grid_w).astype(dtype)

        if avg_mode:
            return te.sum(
                tir.if_then_else(
                    valid,
                    _bilinear(
                        batch_index,
                        c,
                        y_start + (rh.astype(dtype) + 0.5) * bin_h / roi_bin_grid_h.astype(dtype),
                        x_start + (rw.astype(dtype) + 0.5) * bin_w / roi_bin_grid_w.astype(dtype),
                    )
                    / count,
                    tir.const(0.0, dtype),
                ),
                axis=[rh, rw],
            )
        else:  # max mode
            return te.max(
                tir.if_then_else(
                    valid,
                    _bilinear(
                        batch_index,
                        c,
                        y_start + (rh.astype(dtype) + 0.5) * bin_h / roi_bin_grid_h.astype(dtype),
                        x_start + (rw.astype(dtype) + 0.5) * bin_w / roi_bin_grid_w.astype(dtype),
                    ),
                    tir.const(float("-inf"), dtype),
                ),
                axis=[rh, rw],
            )

    return te.compute(
        (num_roi, channel, pooled_size_h, pooled_size_w), _sample, tag="pool,roi_align"
    )
#End TI
