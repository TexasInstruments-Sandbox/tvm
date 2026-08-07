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
"""MMALIB QDQ fusion: partition int8 quantized conv2d for TI C7x MMA.

Matches the PT2E QDQ pattern BEFORE FuseQDQToInt8Conv2D runs:

    dequantize(data_int8, d_scale, d_zp)
      -> conv2d(float, dequantize(weight_int8, w_scale, w_zp))
      -> [add(float_bias)] -> [relu]
      -> quantize(out, o_scale, o_zp)

Replaces the full sequence with a single call_extern to
"mmalib_conv2d_i8" (groups==1) or "mmalib_conv2d_i8_grouped_loop"
(groups>1, e.g. ResNeXt101's cardinality convs) that computes
int8*int8->int32 convolution with fused bias, requantization, and
optional ReLU clipping.

Supports asymmetric activation zero points (d_zp != 0) by folding the
correction into the bias term at compile time.
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import propagate_span
from .ti_mmalib_legalize import (
    _check_conv2d_mmalib_constraints,
    _float_to_scale_shift,
    _resolve_constant_tensor,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Pattern definitions
# =========================================================================


def _qdq_conv2d_bias_relu_pattern():
    """dequant(data) -> conv2d(_, dequant(w)) -> add(bias) -> relu -> quantize"""
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
    return quant_out, annotations, _check_mmalib_qdq_conv2d


def _qdq_conv2d_bias_pattern():
    """dequant(data) -> conv2d(_, dequant(w)) -> add(bias) -> quantize"""
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
    return quant_out, annotations, _check_mmalib_qdq_conv2d


def _qdq_conv2d_relu_pattern():
    """dequant(data) -> conv2d(_, dequant(w)) -> relu -> quantize"""
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
    return quant_out, annotations, _check_mmalib_qdq_conv2d


def _qdq_conv2d_pattern():
    """dequant(data) -> conv2d(_, dequant(w)) -> quantize"""
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
    return quant_out, annotations, _check_mmalib_qdq_conv2d


# =========================================================================
# Check function
# =========================================================================


def _is_const_zero(expr) -> bool:
    """Check if expression is a constant zero (scalar or vector)."""
    if isinstance(expr, relax.Constant):
        return np.all(expr.data.numpy() == 0)
    return False


def _check_mmalib_qdq_conv2d(ctx) -> bool:
    """Validate MMALIB eligibility for a QDQ conv2d pattern match."""
    # w_zp must be 0 (symmetric weight quantization required by MMALIB)
    w_zp = ctx.annotated_expr["w_zp"]
    if not _is_const_zero(w_zp):
        return False

    # Bias (if present) must resolve to a compile-time constant -- e.g.
    # PT2E's conv-bias decomposition wraps it in reshape(bias_const, (1, C,
    # 1, 1)) rather than passing the constant directly. _resolve_constant_
    # tensor() (used by _lower() below) unwraps that; if it *still* can't
    # resolve to a constant, reject the match here rather than silently
    # lowering with a dropped (all-zero) bias.
    if "bias" in ctx.annotated_expr:
        if _resolve_constant_tensor(ctx.annotated_expr["bias"]) is None:
            return False

    # w_int8 must be int8
    w = ctx.annotated_expr["w_int8"]
    if isinstance(w, relax.Constant):
        if w.data.dtype != "int8":
            return False
    elif hasattr(w, "struct_info") and hasattr(w.struct_info, "dtype"):
        if str(w.struct_info.dtype) != "int8":
            return False
    else:
        return False

    # data must be int8
    data = ctx.annotated_expr["data"]
    if isinstance(data, relax.Constant):
        return False
    if hasattr(data, "struct_info") and hasattr(data.struct_info, "dtype"):
        if str(data.struct_info.dtype) != "int8":
            return False

    # Extract conv2d call and validate constraints
    conv = ctx.annotated_expr["conv"]
    if not isinstance(conv, relax.Call):
        return False

    data_dq_sinfo = conv.args[0].struct_info
    w_dq_sinfo = conv.args[1].struct_info
    if not isinstance(data_dq_sinfo, relax.TensorStructInfo):
        return False
    if not isinstance(w_dq_sinfo, relax.TensorStructInfo):
        return False

    # Conv2d operates on float (dequantized) tensors; we need int8 shapes.
    # The int8 shapes match float shapes for the spatial/channel dims.
    # Get kernel shape from the int8 weight directly.
    if isinstance(w, relax.Constant):
        kernel_shape = w.data.shape
    elif hasattr(w, "struct_info") and w.struct_info.shape is not None:
        kernel_shape = [int(s) for s in w.struct_info.shape]
    else:
        return False

    if hasattr(data, "struct_info") and data.struct_info.shape is not None:
        data_shape = data.struct_info.shape
    else:
        return False

    # Build mock struct_info for the constraint checker (needs int8 shapes)
    try:
        data_sinfo = relax.TensorStructInfo(data_shape, "int8")
        kernel_sinfo = relax.TensorStructInfo(kernel_shape, "int8")
    except Exception:
        return False

    return _check_conv2d_mmalib_constraints(conv.attrs, data_sinfo, kernel_sinfo, allow_groups=True)


# =========================================================================
# Composite lowering (PyExprMutator)
# =========================================================================

_COMPOSITE_PREFIXES = (
    "mmalib.conv2d_i8_qdq_bias_relu",
    "mmalib.conv2d_i8_qdq_bias",
    "mmalib.conv2d_i8_qdq_relu",
    "mmalib.conv2d_i8_qdq",
)


@mutator
class _MMALIBQDQLowerer(PyExprMutator):
    """Replace MMALIB QDQ composite functions with call_tir to
    mmalib_conv2d_i8 (groups==1) or mmalib_conv2d_i8_grouped_loop
    (groups>1)."""

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
        """Lower composite to call_tir via _emit_conv2d: mmalib_conv2d_i8
        for groups==1, mmalib_conv2d_i8_grouped_loop for groups>1."""
        param_to_arg = dict(zip(func.params, call.args))
        roles = self._extract_roles(func, has_bias)

        required = ["data", "w_int8", "w_scale", "d_scale", "o_scale", "conv_attrs"]
        if any(r not in roles for r in required):
            logger.warning("Could not identify roles in MMALIB composite")
            return super().visit_call_(call)

        # Map roles to actual constant values
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
            resolved = _resolve_constant_tensor(bias_arg, lookup=self.lookup_binding)
            if resolved is not None:
                bias_np = resolved.flatten()

        # Extract conv2d dimensions from attrs and weight shape
        attrs = roles["conv_attrs"]
        # Weight is OIHW (default kernel_layout for conv2d). For groups>1,
        # PyTorch/Relax's grouped-conv weight layout is [C_out, C_in/groups,
        # KH, KW] — this dim is the per-group input channel count, which
        # equals the full C_in only when groups==1.
        C_out, C_in_per_group, KH, KW = w_int8_np.shape
        groups = int(attrs.groups)
        C_in = C_in_per_group * groups

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

        # Per-channel weight sum for zero-point correction
        weight_sum = w_int8_np.astype(np.int32).reshape(C_out, -1).sum(axis=1)

        # Zero-point correction folded into bias
        zp_correction = (np.int32(-d_zp_val) * weight_sum).astype(np.int32)

        # Bias in accumulator scale
        if bias_np is not None:
            dw_scale = d_scale_val * w_scale_np[:C_out]
            bias_accum = np.round(bias_np[:C_out] / dw_scale).astype(np.int32)
        else:
            bias_accum = np.zeros(C_out, dtype=np.int32)

        bias_i32 = (bias_accum + zp_correction).astype(np.int32)

        # Output zero-point correction
        if o_zp_val != 0:
            combined_rescale_for_ozp = d_scale_val * w_scale_np[:C_out] / o_scale_val
            bias_i32 = (bias_i32 + np.round(o_zp_val / combined_rescale_for_ozp)).astype(np.int32)

        # Requantization scale
        combined_rescale = d_scale_val * w_scale_np[:C_out] / o_scale_val
        scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)

        # Build relax constants
        kernel_relax = relax.Constant(w_int8_np)
        bias_relax = relax.Constant(bias_i32)
        scale_relax = relax.Constant(scale_u8)
        shift_relax = relax.Constant(shift_u8)

        if groups == 1:
            result = self._emit_conv2d(
                "mmalib_conv2d_i8",
                "mmalib_conv2d",
                data_arg,
                kernel_relax,
                bias_relax,
                scale_relax,
                shift_relax,
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
                H_out,
                W_out,
            )
        else:
            # See docs/dsp/quantized_model_optimization.md Step 13: MMALIB's
            # numGroupsPerKernel field on the bias-fused convolveBias_row
            # kernel is real (substantial hardware-intrinsics code exists)
            # but has zero validation coverage anywhere in TI's own software
            # -- TIDL always pairs grouped mode with the non-bias
            # convolve_row kernel instead, never this one. Hardware testing
            # confirmed the concern: chaining this kernel's native grouped
            # calls within one inference produced intermittent silent data
            # corruption and hangs on real c7x_dload hardware (never on
            # c7x_host, which runs a different reference implementation).
            # Per the plan's hard gate, the native path is abandoned
            # entirely -- every grouped conv layer uses
            # mmalib_conv2d_i8_grouped_loop, a single call_extern whose C++
            # implementation loops over groups internally via the already-
            # proven conv2d_impl (see mmalib_wrappers.cpp).
            result = self._emit_conv2d(
                "mmalib_conv2d_i8_grouped_loop",
                "mmalib_conv2d_grouped_loop",
                data_arg,
                kernel_relax,
                bias_relax,
                scale_relax,
                shift_relax,
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
                H_out,
                W_out,
                groups=groups,
            )

        if has_relu:
            result = relax.op.clip(result, relax.PrimValue(0), relax.PrimValue(127))

        self.count += 1
        logger.info(
            "MMALIB QDQ fusion #%d: conv2d %dx%dx%d->%d (stride=%d)%s%s",
            self.count,
            C_in,
            KH,
            KW,
            C_out,
            stride_h,
            " +bias" if has_bias else "",
            " +relu" if has_relu else "",
        )
        return propagate_span(result, roles["conv_call"])

    def _emit_conv2d(
        self,
        extern_name,
        name_hint,
        data_arg,
        kernel_relax,
        bias_relax,
        scale_relax,
        shift_relax,
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
        H_out,
        W_out,
        groups=None,
    ):
        """Emit a call_tir to `extern_name`: mmalib_conv2d_i8 (groups=1,
        pass groups=None) or mmalib_conv2d_i8_grouped_loop (groups>1,
        pass the group count -- appended as the extern call's final arg;
        see mmalib_wrappers.cpp for why a single call per layer, not one
        per group unrolled here with strided_slice/concat, is required to
        keep cl7x's compile time tractable -- docs/dsp/
        quantized_model_optimization.md Step 13).
        """

        def fcompute(ins, outs):
            args = [
                "int32",
                extern_name,
                ins[0].data,
                ins[1].data,
                ins[2].data,
                ins[3].data,
                ins[4].data,
                outs[0].data,
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
            ]
            if groups is not None:
                args.append(groups)
            return tir.call_extern(*args)

        def te_mmalib_conv2d(data_t, weight_t, bias_t, scale_t, shift_t):
            return te.extern(
                (1, C_out, H_out, W_out),
                [data_t, weight_t, bias_t, scale_t, shift_t],
                fcompute,
                name=name_hint,
                dtype="int8",
            )

        return self.builder_.call_te(
            te_mmalib_conv2d,
            data_arg,
            kernel_relax,
            bias_relax,
            scale_relax,
            shift_relax,
            primfunc_name_hint=name_hint,
            primfunc_attrs={"c7x_offload_backend": "mmalib"},
        )

    @staticmethod
    def _extract_roles(func, has_bias):
        """Walk composite body to map function params to their roles.

        Uses conv2d's args to disambiguate data vs weight dequantize:
        conv2d(data_dq_var, weight_dq_var) tells us which dequantize
        binding produces data and which produces weight.
        """
        roles = {}

        # First pass: find conv2d to identify data_dq and weight_dq vars
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
                    roles["conv_call"] = val
                    conv2d_data_var = val.args[0]
                    conv2d_weight_var = val.args[1]
                    break

        if conv2d_data_var is None:
            return roles

        # Second pass: trace dequantize bindings via conv2d's args
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


@tvm.transform.module_pass(opt_level=0, name="FuseMMALIBQDQConv2d")
class FuseMMALIBQDQConv2d:
    """Fuse PT2E QDQ conv2d patterns into MMALIB int8 conv2d calls.

    This pass runs BEFORE FuseQDQToInt8Conv2D and matches the original
    PT2E quantized graph structure directly. Non-eligible conv2d ops
    (wrong alignment, dilation, groups, etc.) are not matched and fall
    through to subsequent QDQ passes.

    Supports:
      - Per-channel weight quantization
      - Asymmetric activation zero points (folded into bias)
      - Float bias (converted to int32 at compile time)
      - Optional fused ReLU (emitted as clip after call_tir)
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Phase 1: pattern match into composite functions
        patterns = [
            ("mmalib.conv2d_i8_qdq_bias_relu", *_qdq_conv2d_bias_relu_pattern()),
            ("mmalib.conv2d_i8_qdq_bias", *_qdq_conv2d_bias_pattern()),
            ("mmalib.conv2d_i8_qdq_relu", *_qdq_conv2d_relu_pattern()),
            ("mmalib.conv2d_i8_qdq", *_qdq_conv2d_pattern()),
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: lower composites to call_tir
        lowerer = _MMALIBQDQLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseMMALIBQDQConv2d: fused %d layers", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
