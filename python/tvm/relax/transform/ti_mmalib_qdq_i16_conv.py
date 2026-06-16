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
"""MMALIB QDQ fusion: partition int16 quantized conv2d for TI C7x MMA.

Mirrors FuseMMALIBQDQConv2d (ti_mmalib_qdq_fusion.py) for int16 precision.

Matches the PT2E int16 QDQ pattern BEFORE FuseQDQToInt8Conv2D runs:

    dequantize(data_int16, d_scale, d_zp=0)     # always symmetric
      -> conv2d(float, dequantize(weight_int16, w_scale, w_zp=0))
      -> [add(float_bias)] -> [relu]
      -> quantize(out, o_scale, o_zp=0)

Replaces the full sequence with a single call_extern("mmalib_conv2d_i16")
that computes int16*int16->int64 convolution with fused per-channel bias,
requantization (uint8 scale/shift), and optional ReLU clipping.

Key differences from the int8 version:
  - Input/output/weight dtype: int16 (not int8)
  - Bias dtype: int64 (wider accumulator)
  - Alignment constraint: C_out % MMA_SIZE_I16 == 0 (half of i8)
  - d_zp must be exactly 0 (int16 only supports symmetric activation quant)
  - No asymmetric zero-point correction needed (d_zp=0 always)
  - Clip bounds for optional ReLU: (-32768, 32767)

Produced by C7xMMAQuantizer(dtype="int16").
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_mmalib_legalize import _check_conv2d_mmalib_constraints, _float_to_scale_shift
from .ti_mmalib_qdq_fusion import _MMALIBQDQLowerer as _I8Lowerer

logger = logging.getLogger(__name__)


# =========================================================================
# Pattern definitions (identical structure to i8 patterns; dtype is checked
# in the eligibility function rather than the pattern itself)
# =========================================================================


def _qdq_i16_conv2d_bias_relu_pattern():
    """dequant(data_i16) -> conv2d(_, dequant(w_i16)) -> add(bias) -> relu -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_i16 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_i16, w_scale, w_zp)

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
        "w_int8": w_i16,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "bias": bias,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_i16_conv2d


def _qdq_i16_conv2d_bias_pattern():
    """dequant(data_i16) -> conv2d(_, dequant(w_i16)) -> add(bias) -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_i16 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_i16, w_scale, w_zp)

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
        "w_int8": w_i16,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "bias": bias,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_i16_conv2d


def _qdq_i16_conv2d_relu_pattern():
    """dequant(data_i16) -> conv2d(_, dequant(w_i16)) -> relu -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_i16 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_i16, w_scale, w_zp)

    conv = is_op("relax.nn.conv2d")(data_dq, w_dq)
    relu_out = is_op("relax.nn.relu")(conv)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(relu_out, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_i16,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_i16_conv2d


def _qdq_i16_conv2d_pattern():
    """dequant(data_i16) -> conv2d(_, dequant(w_i16)) -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_i16 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_i16, w_scale, w_zp)

    conv = is_op("relax.nn.conv2d")(data_dq, w_dq)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(conv, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_i16,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "conv": conv,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_i16_conv2d


# =========================================================================
# Check function
# =========================================================================


def _is_const_zero(expr) -> bool:
    """Check if expression is a constant zero (scalar or vector)."""
    if isinstance(expr, relax.Constant):
        return np.all(expr.data.numpy() == 0)
    return False


def _check_mmalib_qdq_i16_conv2d(ctx) -> bool:
    """Validate MMALIB eligibility for an int16 QDQ conv2d pattern match.

    Additional int16-specific constraint vs the int8 version:
      - d_zp must be 0 (int16 activations are always symmetric).
        Asymmetric activation quantization is not supported for int16.
    """
    # Weight zero-point must be 0 (symmetric, required by MMALIB)
    w_zp = ctx.annotated_expr["w_zp"]
    if not _is_const_zero(w_zp):
        return False

    # Activation zero-point must be 0 for int16 (symmetric only)
    d_zp = ctx.annotated_expr["d_zp"]
    if not _is_const_zero(d_zp):
        return False

    # Output zero-point must be 0 for int16.
    # The lowerer does not fold o_zp into bias_i64; a non-zero value would
    # shift all outputs incorrectly.  Direct [] access: all conv2d patterns
    # include "o_zp" in annotations.
    if not _is_const_zero(ctx.annotated_expr["o_zp"]):
        return False

    # Weight must be int16
    w = ctx.annotated_expr["w_int8"]  # keyed "w_int8" for compat with _extract_roles
    if isinstance(w, relax.Constant):
        if w.data.dtype != "int16":
            return False
    elif hasattr(w, "struct_info") and hasattr(w.struct_info, "dtype"):
        if str(w.struct_info.dtype) != "int16":
            return False
    else:
        return False

    # Activation must be int16 (not a constant — it's the dynamic input)
    data = ctx.annotated_expr["data"]
    if isinstance(data, relax.Constant):
        return False
    if hasattr(data, "struct_info") and hasattr(data.struct_info, "dtype"):
        if str(data.struct_info.dtype) != "int16":
            return False

    # Extract conv2d call to validate spatial/channel constraints
    conv = ctx.annotated_expr["conv"]
    if not isinstance(conv, relax.Call):
        return False

    data_dq_sinfo = conv.args[0].struct_info
    w_dq_sinfo = conv.args[1].struct_info
    if not isinstance(data_dq_sinfo, relax.TensorStructInfo):
        return False
    if not isinstance(w_dq_sinfo, relax.TensorStructInfo):
        return False

    # Build mock struct_info with int16 dtype for the constraint checker
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

    try:
        data_sinfo = relax.TensorStructInfo(data_shape, "int16")
        kernel_sinfo = relax.TensorStructInfo(kernel_shape, "int16")
    except Exception:
        return False

    from .ti_mmalib_constants import MMA_SIZE_I16

    return _check_conv2d_mmalib_constraints(
        conv.attrs, data_sinfo, kernel_sinfo, mma_size=MMA_SIZE_I16
    )


# =========================================================================
# Composite lowering
# =========================================================================

_PATTERN_REGISTRY = [
    ("mmalib.conv2d_i16_qdq_bias_relu", _qdq_i16_conv2d_bias_relu_pattern),
    ("mmalib.conv2d_i16_qdq_bias", _qdq_i16_conv2d_bias_pattern),
    ("mmalib.conv2d_i16_qdq_relu", _qdq_i16_conv2d_relu_pattern),
    ("mmalib.conv2d_i16_qdq", _qdq_i16_conv2d_pattern),
]
_COMPOSITE_NAMES = frozenset(name for name, _ in _PATTERN_REGISTRY)


@mutator
class _MMALIB_QDQI16Conv2dLowerer(PyExprMutator):
    """Replace MMALIB int16 QDQ composite functions with call_tir to mmalib_conv2d_i16."""

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
        if name not in _COMPOSITE_NAMES:
            return super().visit_call_(call)

        has_bias = "bias" in name
        has_relu = "relu" in name
        return self._lower(call, func, has_bias=has_bias, has_relu=has_relu)

    def _lower(self, call, func, has_bias, has_relu):
        """Lower composite to call_tir with mmalib_conv2d_i16."""
        param_to_arg = dict(zip(func.params, call.args))
        # Reuse the i8 lowerer's _extract_roles — it's dtype-independent
        roles = _I8Lowerer._extract_roles(func, has_bias)

        required = ["data", "w_int8", "w_scale", "d_scale", "o_scale", "conv_attrs"]
        if any(r not in roles for r in required):
            logger.warning("Could not identify roles in MMALIB i16 conv2d composite")
            return super().visit_call_(call)

        data_arg = param_to_arg[roles["data"]]

        w_i16_arg = param_to_arg[roles["w_int8"]]
        w_i16_np = w_i16_arg.data.numpy() if isinstance(w_i16_arg, relax.Constant) else None
        if w_i16_np is None:
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

        # int16 always uses symmetric activation (d_zp=0 enforced by quantizer)
        # so no zero-point correction is needed in the bias term.

        bias_np = None
        if has_bias and "bias" in roles:
            bias_arg = param_to_arg[roles["bias"]]
            if isinstance(bias_arg, relax.Constant):
                bias_np = bias_arg.data.numpy().flatten()

        # Extract conv2d dimensions
        attrs = roles["conv_attrs"]
        C_out, C_in, KH, KW = w_i16_np.shape

        data_sinfo = data_arg.struct_info
        data_layout = tir.layout(attrs.data_layout)
        H_in = int(data_sinfo.shape[data_layout.index_of("H")])
        W_in = int(data_sinfo.shape[data_layout.index_of("W")])

        strides = [int(s) for s in attrs.strides]
        stride_h, stride_w = strides[0], strides[1]

        padding = [int(p) for p in attrs.padding]
        if len(padding) == 2:
            # TVM 2-element form: (pad_h, pad_w)
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        else:
            pad_top, pad_left, pad_bottom, pad_right = padding

        H_out = (H_in + pad_top + pad_bottom - KH) // stride_h + 1
        W_out = (W_in + pad_left + pad_right - KW) // stride_w + 1

        # --- Compute MMALIB int16 parameters ---

        # Bias in accumulator scale (int64 for int16 accumulators)
        dw_scale = d_scale_val * w_scale_np[:C_out]
        if bias_np is not None:
            bias_i64 = np.round(bias_np[:C_out] / dw_scale).astype(np.int64)
        else:
            bias_i64 = np.zeros(C_out, dtype=np.int64)

        # Per-channel requantization scale and shift (same formula as int8)
        combined_rescale = dw_scale / o_scale_val
        scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)

        kernel_relax = relax.Constant(w_i16_np)
        bias_relax = relax.Constant(bias_i64)
        scale_relax = relax.Constant(scale_u8)
        shift_relax = relax.Constant(shift_u8)

        def te_mmalib_conv2d_i16(
            data_t: te.Tensor,
            weight_t: te.Tensor,
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
                [data_t, weight_t, bias_t, scale_t, shift_t],
                fcompute,
                # Use "mmalib_conv2d" (not "mmalib_conv2d_i16") as the output
                # buffer variable name — the C codegen would otherwise emit
                # `void* mmalib_conv2d_i16 = output.data;` which shadows the
                # extern function declaration of the same name.
                name="mmalib_conv2d",
                dtype="int16",
            )

        result = self.builder_.call_te(
            te_mmalib_conv2d_i16,
            data_arg,
            kernel_relax,
            bias_relax,
            scale_relax,
            shift_relax,
            primfunc_name_hint="mmalib_conv2d_i16",
        )

        if has_relu:
            # Clip to int16 range rather than int8 range
            result = relax.op.clip(result, relax.PrimValue(-32768), relax.PrimValue(32767))

        self.count += 1
        logger.info(
            "MMALIB i16 QDQ fusion #%d: conv2d %dx%dx%d->%d (stride=%d)%s%s",
            self.count,
            C_in,
            KH,
            KW,
            C_out,
            stride_h,
            " +bias" if has_bias else "",
            " +relu" if has_relu else "",
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseMMALIBQDQConv2dI16")
class FuseMMALIBQDQConv2dI16:
    """Fuse PT2E int16 QDQ conv2d patterns into MMALIB int16 conv2d calls.

    Mirrors FuseMMALIBQDQConv2d for int16 precision. Must run before
    FuseQDQToInt8Conv2D to see the intact PT2E QDQ graph.

    Requires C7xMMAQuantizer(dtype="int16") or equivalent quantizer that
    produces int16 per-channel symmetric weight quantization and int16
    per-tensor symmetric activation quantization (d_zp=0).
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = [(name, *factory()) for name, factory in _PATTERN_REGISTRY]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _MMALIB_QDQI16Conv2dLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseMMALIBQDQConv2dI16: fused %d layers", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
