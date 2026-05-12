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
"""MMALIB QDQ fusion: partition int8 quantized depthwise conv2d for TI C7x MMA.

Matches the PT2E QDQ pattern for depthwise conv2d (groups == C_in):

    dequantize(data_int8, d_scale, d_zp)
      -> conv2d(float, dequantize(weight_int8, w_scale, w_zp), groups=C_in)
      -> [add(float_bias)] -> [relu]
      -> quantize(out, o_scale, o_zp)

Replaces the full sequence with a single call_extern("mmalib_depthwise_conv2d_i8")
that computes per-channel depthwise convolution with fused bias, requantization,
and saturation.

Uses MMALIB_CNN_convolve_col_smallNo_highPrecision kernel. Weight
reordering is handled at runtime by the C wrapper via reorderWeights_exec.
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_mmalib_legalize import _float_to_scale_shift

logger = logging.getLogger(__name__)


# =========================================================================
# Pattern definitions (same structure as conv2d QDQ, but for depthwise)
# =========================================================================


def _qdq_dwconv2d_bias_relu_pattern():
    """dequant(data) -> conv2d(groups=C) -> add(bias) -> relu -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)

    conv = is_op("relax.nn.conv2d")(data_dq, w_dq)
    bias = wildcard()
    add_out = is_op("relax.add")(conv, bias)
    relu_out = is_op("relax.nn.relu")(add_out)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(relu_out, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "bias": bias,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_dwconv2d


def _qdq_dwconv2d_bias_pattern():
    """dequant(data) -> conv2d(groups=C) -> add(bias) -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)

    conv = is_op("relax.nn.conv2d")(data_dq, w_dq)
    bias = wildcard()
    add_out = is_op("relax.add")(conv, bias)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(add_out, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "bias": bias,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_dwconv2d


def _qdq_dwconv2d_relu_pattern():
    """dequant(data) -> conv2d(groups=C) -> relu -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)

    conv = is_op("relax.nn.conv2d")(data_dq, w_dq)
    relu_out = is_op("relax.nn.relu")(conv)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(relu_out, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_dwconv2d


def _qdq_dwconv2d_pattern():
    """dequant(data) -> conv2d(groups=C) -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)

    conv = is_op("relax.nn.conv2d")(data_dq, w_dq)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(conv, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_dwconv2d


# =========================================================================
# Check function
# =========================================================================


def _is_const_zero(expr) -> bool:
    if isinstance(expr, relax.Constant):
        return np.all(expr.data.numpy() == 0)
    return False


def _check_mmalib_qdq_dwconv2d(ctx) -> bool:
    """Validate MMALIB depthwise eligibility for a QDQ conv2d pattern match."""
    w_zp = ctx.annotated_expr["w_zp"]
    if not _is_const_zero(w_zp):
        return False

    w = ctx.annotated_expr["w_int8"]
    if isinstance(w, relax.Constant):
        if w.data.dtype != "int8":
            return False
    elif hasattr(w, "struct_info") and hasattr(w.struct_info, "dtype"):
        if str(w.struct_info.dtype) != "int8":
            return False
    else:
        return False

    data = ctx.annotated_expr["data"]
    if isinstance(data, relax.Constant):
        return False
    if hasattr(data, "struct_info") and hasattr(data.struct_info, "dtype"):
        if str(data.struct_info.dtype) != "int8":
            return False

    conv = ctx.annotated_expr["conv"]
    if not isinstance(conv, relax.Call):
        return False

    attrs = conv.attrs
    # Must be depthwise: groups == C_in
    if hasattr(data, "struct_info") and data.struct_info.shape is not None:
        data_shape = data.struct_info.shape
    else:
        return False
    data_layout = tir.layout(attrs.data_layout)
    c_in = int(data_shape[data_layout.index_of("C")])
    if attrs.groups != c_in:
        return False

    # Kernel shape from weight
    if isinstance(w, relax.Constant):
        kernel_shape = w.data.shape
    elif hasattr(w, "struct_info") and w.struct_info.shape is not None:
        kernel_shape = [int(s) for s in w.struct_info.shape]
    else:
        return False

    kernel_layout = tir.layout(attrs.kernel_layout)
    kh = int(kernel_shape[kernel_layout.index_of("H")])
    kw = int(kernel_shape[kernel_layout.index_of("W")])

    # Supported kernel sizes: 3x3, 5x5, 7x7; must be square
    if kh != kw:
        return False
    if kh not in (3, 5, 7):
        return False

    # Strides: 1 or 2, symmetric
    strides = [int(s) for s in attrs.strides]
    if strides[0] != strides[1]:
        return False
    if strides[0] not in (1, 2):
        return False
    # stride==2 requires kernel 3x3 or 5x5
    if strides[0] == 2 and kh > 5:
        return False

    # Dilation must be 1x1
    if list(attrs.dilation) != [1, 1]:
        return False

    # N must be 1
    if int(data_shape[data_layout.index_of("N")]) != 1:
        return False

    # All shapes must be static
    for s in data_shape:
        if not isinstance(s, tir.IntImm):
            return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================

_COMPOSITE_PREFIXES = (
    "mmalib.dwconv2d_i8_qdq_bias_relu",
    "mmalib.dwconv2d_i8_qdq_bias",
    "mmalib.dwconv2d_i8_qdq_relu",
    "mmalib.dwconv2d_i8_qdq",
)


@mutator
class _MMALIBQDQDwConvLowerer(PyExprMutator):
    """Replace MMALIB QDQ depthwise composite functions with call_tir."""

    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.count = 0

    def visit_call_(self, call: relax.Call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)

        func = self.builder_.get()[call.op]
        if not isinstance(func, relax.Function):
            return super().visit_call_(call)
        if "Composite" not in func.attrs:
            return super().visit_call_(call)

        name = str(func.attrs["Composite"])
        if not any(name == prefix for prefix in _COMPOSITE_PREFIXES):
            return super().visit_call_(call)

        has_bias = "bias" in name
        has_relu = "relu" in name
        return self._lower(call, func, has_bias=has_bias, has_relu=has_relu)

    def _lower(self, call, func, has_bias, has_relu):
        param_to_arg = dict(zip(func.params, call.args))
        roles = self._extract_roles(func, has_bias)

        required = ["data", "w_int8", "w_scale", "d_scale", "o_scale", "conv_attrs"]
        if any(r not in roles for r in required):
            logger.warning("Could not identify roles in MMALIB depthwise composite")
            return super().visit_call_(call)

        data_arg = param_to_arg[roles["data"]]

        w_int8_arg = param_to_arg[roles["w_int8"]]
        w_int8_np = w_int8_arg.data.numpy() if isinstance(w_int8_arg, relax.Constant) else None
        if w_int8_np is None:
            return super().visit_call_(call)

        w_scale_arg = param_to_arg[roles["w_scale"]]
        if not isinstance(w_scale_arg, relax.Constant):
            return super().visit_call_(call)
        w_scale_np = w_scale_arg.data.numpy().flatten()

        d_scale_arg = param_to_arg[roles["d_scale"]]
        if not isinstance(d_scale_arg, relax.Constant):
            return super().visit_call_(call)
        d_scale_val = float(d_scale_arg.data.numpy())

        o_scale_arg = param_to_arg[roles["o_scale"]]
        if not isinstance(o_scale_arg, relax.Constant):
            return super().visit_call_(call)
        o_scale_val = float(o_scale_arg.data.numpy())

        d_zp_val = 0
        if "d_zp" in roles:
            d_zp_arg = param_to_arg[roles["d_zp"]]
            if isinstance(d_zp_arg, relax.Constant):
                d_zp_val = int(d_zp_arg.data.numpy())

        o_zp_val = 0
        if "o_zp" in roles:
            o_zp_arg = param_to_arg[roles["o_zp"]]
            if isinstance(o_zp_arg, relax.Constant):
                o_zp_val = int(o_zp_arg.data.numpy())

        bias_np = None
        if has_bias and "bias" in roles:
            bias_arg = param_to_arg[roles["bias"]]
            if isinstance(bias_arg, relax.Constant):
                bias_np = bias_arg.data.numpy().flatten()

        # Extract dimensions
        attrs = roles["conv_attrs"]
        # Weight is [C_out, 1, KH, KW] for depthwise (OIHW)
        channels = w_int8_np.shape[0]
        KH, KW = w_int8_np.shape[2], w_int8_np.shape[3]

        data_sinfo = data_arg.struct_info
        data_layout = tir.layout(attrs.data_layout)
        H_in = int(data_sinfo.shape[data_layout.index_of("H")])
        W_in = int(data_sinfo.shape[data_layout.index_of("W")])

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

        # --- Compute MMALIB parameters ---

        # Per-channel weight sum for zero-point correction (depthwise: sum over 1*KH*KW)
        weight_sum = w_int8_np.astype(np.int32).reshape(channels, -1).sum(axis=1)
        zp_correction = (np.int32(-d_zp_val) * weight_sum).astype(np.int32)

        # Bias in accumulator scale
        if bias_np is not None:
            dw_scale = d_scale_val * w_scale_np[:channels]
            bias_accum = np.round(bias_np[:channels] / dw_scale).astype(np.int32)
        else:
            bias_accum = np.zeros(channels, dtype=np.int32)

        bias_i32 = (bias_accum + zp_correction).astype(np.int32)

        if o_zp_val != 0:
            combined_rescale_for_ozp = d_scale_val * w_scale_np[:channels] / o_scale_val
            bias_i32 = (bias_i32 + np.round(o_zp_val / combined_rescale_for_ozp)).astype(np.int32)

        # Requantization scale
        combined_rescale = d_scale_val * w_scale_np[:channels] / o_scale_val
        scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)

        # Pass natural-order weights [C, 1, KH, KW] flattened to [C * KH * KW].
        # The C wrapper calls reorderWeights_exec at runtime.
        nat_weights = w_int8_np.reshape(channels, KH * KW).flatten()

        # Build relax constants
        weights_relax = relax.Constant(nat_weights)
        bias_relax = relax.Constant(bias_i32)
        scale_relax = relax.Constant(scale_u8)
        shift_relax = relax.Constant(shift_u8)

        def te_mmalib_dwconv2d_i8(
            data_t: te.Tensor,
            weight_t: te.Tensor,
            bias_t: te.Tensor,
            scale_t: te.Tensor,
            shift_t: te.Tensor,
        ) -> te.Tensor:
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "mmalib_depthwise_conv2d_i8",
                    ins[0].data,
                    ins[1].data,
                    ins[2].data,
                    ins[3].data,
                    ins[4].data,
                    outs[0].data,
                    channels,
                    H_in,
                    W_in,
                    KH,
                    KW,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    channels,
                )

            return te.extern(
                (1, channels, H_out, W_out),
                [data_t, weight_t, bias_t, scale_t, shift_t],
                fcompute,
                name="mmalib_dwconv2d",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_mmalib_dwconv2d_i8,
            data_arg,
            weights_relax,
            bias_relax,
            scale_relax,
            shift_relax,
            primfunc_name_hint="mmalib_dwconv2d",
        )

        if has_relu:
            result = relax.op.clip(result, relax.PrimValue(0), relax.PrimValue(127))

        self.count += 1
        logger.info(
            "MMALIB QDQ dwconv2d fusion #%d: %d channels %dx%d (stride=%d)%s%s",
            self.count,
            channels,
            KH,
            KW,
            stride_h,
            " +bias" if has_bias else "",
            " +relu" if has_relu else "",
        )
        return result

    @staticmethod
    def _extract_roles(func, has_bias):
        """Walk composite body to map function params to their roles."""
        roles = {}

        conv2d_data_var = None
        conv2d_weight_var = None
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if not hasattr(val.op, "name"):
                    continue
                if val.op.name == "relax.nn.conv2d":
                    roles["conv_attrs"] = val.attrs
                    conv2d_data_var = val.args[0]
                    conv2d_weight_var = val.args[1]
                    break

        if conv2d_data_var is None:
            return roles

        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if not hasattr(val.op, "name"):
                    continue

                op_name = val.op.name
                if op_name == "relax.dequantize":
                    if binding.var.same_as(conv2d_data_var):
                        roles["data"] = val.args[0]
                        roles["d_scale"] = val.args[1]
                        roles["d_zp"] = val.args[2]
                    elif binding.var.same_as(conv2d_weight_var):
                        roles["w_int8"] = val.args[0]
                        roles["w_scale"] = val.args[1]
                        roles["w_zp"] = val.args[2]
                elif has_bias and op_name == "relax.add":
                    roles["bias"] = val.args[1]
                elif op_name == "relax.quantize":
                    roles["o_scale"] = val.args[1]
                    roles["o_zp"] = val.args[2]
        return roles


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseMMALIBQDQDwConv2d")
class FuseMMALIBQDQDwConv2d:
    """Fuse PT2E QDQ depthwise conv2d patterns into MMALIB calls.

    This pass matches depthwise conv2d ops (groups == C_in) in the
    original PT2E quantized graph and replaces them with MMALIB
    depthwise kernel calls. Non-eligible ops (wrong kernel size,
    dilation, etc.) fall through to subsequent QDQ passes.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = [
            ("mmalib.dwconv2d_i8_qdq_bias_relu", *_qdq_dwconv2d_bias_relu_pattern()),
            ("mmalib.dwconv2d_i8_qdq_bias", *_qdq_dwconv2d_bias_pattern()),
            ("mmalib.dwconv2d_i8_qdq_relu", *_qdq_dwconv2d_relu_pattern()),
            ("mmalib.dwconv2d_i8_qdq", *_qdq_dwconv2d_pattern()),
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _MMALIBQDQDwConvLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseMMALIBQDQDwConv2d: fused %d layers", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
