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
# pylint: disable=invalid-name
"""MMALIB int16 legalization and shared helpers for TI C7x MMA.

Provides:
  - MMALIBLegalize: custom legalization for int16 matmul/conv2d via
    LegalizeOps(customize_legalize_map=...).
  - _check_conv2d_mmalib_constraints: shared eligibility check (used by
    both this module and ti_mmalib_qdq_fusion.py).
  - _float_to_scale_shift: per-channel float→uint8 scale/shift conversion
    (used by ti_mmalib_qdq_fusion.py).

Data layout: NCHW throughout.
MMALIB's conv kernel (convolveBias_row) operates on planar channel-first
data — each input channel is a contiguous H*W block. This matches NCHW.
When -mmalib=1 is set, the pipeline skips ConvertLayoutNHWC so that all
ops (conv, relu, add, pool) stay in NCHW. Layout conversion happens at
network I/O boundaries only.
"""

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext

from .legalize_ops.linear_algebra import _matmul
from .legalize_ops.nn import _nn_conv2d

# =======================================================================
# Int16 matmul legalization
# =======================================================================


def _is_mmalib_eligible(call: relax.Call) -> bool:
    """Check if a matmul call can be handled by MMALIB."""
    lhs_sinfo = call.args[0].struct_info
    rhs_sinfo = call.args[1].struct_info

    if not isinstance(lhs_sinfo, relax.TensorStructInfo):
        return False
    if not isinstance(rhs_sinfo, relax.TensorStructInfo):
        return False

    if lhs_sinfo.dtype != "int16" or rhs_sinfo.dtype != "int16":
        return False

    lhs_shape = lhs_sinfo.shape
    rhs_shape = rhs_sinfo.shape
    if lhs_shape is None or rhs_shape is None:
        return False

    if lhs_sinfo.ndim != 2 or rhs_sinfo.ndim != 2:
        return False

    from .ti_mmalib_constants import MMA_SIZE_I16

    mma_size_i16 = MMA_SIZE_I16
    for s in lhs_shape:
        if not isinstance(s, tir.IntImm):
            return False
        if int(s) % mma_size_i16 != 0:
            return False
    for s in rhs_shape:
        if not isinstance(s, tir.IntImm):
            return False
        if int(s) % mma_size_i16 != 0:
            return False

    return True


def _mmalib_matmul_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    """Legalize R.matmul to MMALIB extern call when eligible."""
    if not _is_mmalib_eligible(call):
        return _matmul(bb, call)

    lhs_sinfo = call.args[0].struct_info
    rhs_sinfo = call.args[1].struct_info
    M = int(lhs_sinfo.shape[0])
    K = int(lhs_sinfo.shape[1])
    N = int(rhs_sinfo.shape[1])

    def te_mmalib_matmul(a: te.Tensor, b: te.Tensor) -> te.Tensor:
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "mmalib_matmul_i16",
                ins[0].data,
                ins[1].data,
                outs[0].data,
                M,
                K,
                N,
                0,
            )

        return te.extern(
            (M, N),
            [a, b],
            fcompute,
            name="mmalib_matmul",
            dtype="int16",
        )

    return bb.call_te(
        te_mmalib_matmul,
        call.args[0],
        call.args[1],
        primfunc_name_hint="mmalib_matmul",
    )


# =======================================================================
# Int16 conv2d legalization
# =======================================================================


def _check_conv2d_mmalib_constraints(
    attrs, data_sinfo, kernel_sinfo, allow_groups: bool = False
) -> bool:
    """Shared MMALIB conv2d eligibility check for both int8 and int16.

    MMALIB constraints:
      - dilation must be 1x1
      - groups must be 1, unless allow_groups=True (see below)
      - strides must be symmetric (strideX == strideY)
      - N must be 1
      - all shapes must be static

    allow_groups: when True, permits groups>1 for genuinely grouped
    (partial-channel) convolution — e.g. ResNeXt101's cardinality=32
    bottleneck convs — routed to a per-group call_extern loop by the
    caller (see ti_mmalib_qdq_fusion.py). True
    depthwise (groups == C_in) is excluded here: that's
    ti_mmalib_qdq_dwconv.py's job, not this path's. Defaults to False so
    every other existing caller (int16 QDQ conv, int16 plain-legalize)
    keeps rejecting groups>1 unchanged.
    """
    if list(attrs.dilation) != [1, 1]:
        return False

    strides = [int(s) for s in attrs.strides]
    if strides[0] != strides[1]:
        return False

    if data_sinfo.shape is None or kernel_sinfo.shape is None:
        return False
    for s in data_sinfo.shape:
        if not isinstance(s, tir.IntImm):
            return False
    for s in kernel_sinfo.shape:
        if not isinstance(s, tir.IntImm):
            return False

    data_layout = tir.layout(attrs.data_layout)

    if attrs.groups != 1:
        if not allow_groups or attrs.groups <= 0:
            return False
        kernel_layout = tir.layout(attrs.kernel_layout)
        c_in = int(data_sinfo.shape[data_layout.index_of("C")])
        c_out = int(kernel_sinfo.shape[kernel_layout.index_of("O")])
        if attrs.groups == c_in:
            return False  # true depthwise — ti_mmalib_qdq_dwconv.py's job
        if c_in % attrs.groups != 0 or c_out % attrs.groups != 0:
            return False

    if int(data_sinfo.shape[data_layout.index_of("N")]) != 1:
        return False

    return True


def _is_conv2d_mmalib_eligible(call: relax.Call) -> bool:
    """Check if a conv2d call can be handled by MMALIB (int16 path)."""
    data_sinfo = call.args[0].struct_info
    kernel_sinfo = call.args[1].struct_info

    if not isinstance(data_sinfo, relax.TensorStructInfo):
        return False
    if not isinstance(kernel_sinfo, relax.TensorStructInfo):
        return False
    if data_sinfo.dtype != "int16" or kernel_sinfo.dtype != "int16":
        return False
    if data_sinfo.ndim != 4 or kernel_sinfo.ndim != 4:
        return False

    return _check_conv2d_mmalib_constraints(call.attrs, data_sinfo, kernel_sinfo)


def _mmalib_conv2d_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    """Legalize R.nn.conv2d to MMALIB extern call when eligible (int16)."""
    if not _is_conv2d_mmalib_eligible(call):
        return _nn_conv2d(bb, call)

    attrs = call.attrs
    data_sinfo = call.args[0].struct_info
    kernel_sinfo = call.args[1].struct_info

    data_layout = tir.layout(attrs.data_layout)
    kernel_layout = tir.layout(attrs.kernel_layout)

    C_in = int(data_sinfo.shape[data_layout.index_of("C")])
    H_in = int(data_sinfo.shape[data_layout.index_of("H")])
    W_in = int(data_sinfo.shape[data_layout.index_of("W")])
    C_out = int(kernel_sinfo.shape[kernel_layout.index_of("O")])
    KH = int(kernel_sinfo.shape[kernel_layout.index_of("H")])
    KW = int(kernel_sinfo.shape[kernel_layout.index_of("W")])

    strides = [int(s) for s in attrs.strides]
    stride_h, stride_w = strides[0], strides[1]

    padding = [int(p) for p in attrs.padding]
    if len(padding) == 2:
        pad_top, pad_left = padding[0], padding[1]
        pad_bottom, pad_right = padding[0], padding[1]
    else:
        pad_top, pad_left, pad_bottom, pad_right = padding

    H_out = (H_in + pad_top + pad_bottom - KH) // stride_h + 1
    W_out = (W_in + pad_left + pad_right - KW) // stride_w + 1

    # Identity params: zero bias, scale=1, shift=0 for all channels.
    # The QDQ fusion pass (FuseMMALIBQDQConv2dI16) will pass real per-channel
    # values; this legalize path handles plain float32 conv2d with no requant.
    bias_const = relax.Constant(np.zeros(C_out, dtype=np.int64))
    scale_const = relax.Constant(np.ones(C_out, dtype=np.uint8))
    shift_const = relax.Constant(np.zeros(C_out, dtype=np.uint8))

    def te_mmalib_conv2d(
        data: te.Tensor,
        weight: te.Tensor,
        bias_t: te.Tensor,
        scale_t: te.Tensor,
        shift_t: te.Tensor,
    ) -> te.Tensor:
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "mmalib_conv2d_i16",
                ins[0].data,  # input
                ins[1].data,  # kernel
                ins[2].data,  # bias  (int64[C_out])
                ins[3].data,  # scale (uint8[C_out])
                ins[4].data,  # shift (uint8[C_out])
                outs[0].data,  # output
                C_in,
                H_in,
                W_in,
                C_out,
                KH,
                KW,
                stride_h,
                stride_w,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
            )

        return te.extern(
            (1, C_out, H_out, W_out),
            [data, weight, bias_t, scale_t, shift_t],
            fcompute,
            name="mmalib_conv2d",
            dtype="int16",
        )

    return bb.call_te(
        te_mmalib_conv2d,
        call.args[0],
        call.args[1],
        bias_const,
        scale_const,
        shift_const,
        primfunc_name_hint="mmalib_conv2d",
    )


# =======================================================================
# Shared helpers (used by ti_mmalib_qdq_fusion.py)
# =======================================================================


def _float_to_scale_shift(rescale: np.ndarray):
    """Convert per-channel float rescale to (int8 scale, uint8 shift).

    Finds (s, sh) per channel such that s * 2^(-sh) ≈ rescale[ch],
    where s is in [1, 127] (signed int8 positive range) and sh is in [0, 31].

    MMALIB's matrixMatrixMultiplyBias expects signed int8 scale values
    (confirmed by MMALIB test case 8 which declares scale as int8_t).
    """
    n_channels = rescale.shape[0]
    scale_out = np.zeros(n_channels, dtype=np.uint8)
    shift_out = np.zeros(n_channels, dtype=np.uint8)

    for ch in range(n_channels):
        r = float(rescale[ch])
        if r <= 0:
            scale_out[ch] = 0
            shift_out[ch] = 0
            continue

        best_err = float("inf")
        best_s, best_sh = 1, 0
        for sh in range(32):
            s_float = r * (1 << sh)
            s_int = int(round(s_float))
            if s_int < 1:
                continue
            if s_int > 127:
                break
            actual = s_int / (1 << sh)
            err = abs(actual - r) / r
            if err < best_err:
                best_err = err
                best_s = s_int
                best_sh = sh

        scale_out[ch] = best_s
        shift_out[ch] = best_sh

    return scale_out, shift_out


# =======================================================================
# Public passes
# =======================================================================


@tvm.transform.module_pass(opt_level=0, name="MMALIBLegalize")
class MMALIBLegalize:
    """Wraps LegalizeOps with MMALIB custom legalization for int16 ops."""

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        custom_map = {
            "relax.matmul": _mmalib_matmul_legalize,
            "relax.nn.conv2d": _mmalib_conv2d_legalize,
        }
        return relax.transform.LegalizeOps(customize_legalize_map=custom_map)(mod)
