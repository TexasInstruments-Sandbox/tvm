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
"""MMALIB QDQ fusion: partition int16 quantized depthwise conv2d for TI C7x MMA.

Mirrors FuseMMALIBQDQDwConv2d (ti_mmalib_qdq_dwconv.py) for int16 precision.

Matches the PT2E int16 QDQ pattern for depthwise conv2d (groups == C_in):

    dequantize(data_int16, d_scale, d_zp=0)     # always symmetric for int16
      -> conv2d(float, dequantize(weight_int16, w_scale, w_zp=0), groups=C_in)
      -> [add(float_bias)] -> [relu]
      -> quantize(out, o_scale, o_zp=0)

Key differences from the int8 version:
  - data/weight/output dtype: int16
  - d_zp must be 0 (int16 activations are always symmetric)
  - bias: int64 per-group (wider accumulator for int16 inputs)
  - no zero-point correction needed (d_zp=0 always)
  - saturation clip bounds: (-32768, 32767) instead of (-128, 127)
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import propagate_span
from .ti_mmalib_legalize import _float_to_scale_shift, _resolve_constant_tensor
from .ti_mmalib_qdq_dwconv import (
    _check_dwconv2d_geometry,
    _MMALIBQDQDwConvLowerer,
    _qdq_dwconv2d_bias_pattern,
    _qdq_dwconv2d_bias_relu_pattern,
    _qdq_dwconv2d_pattern,
    _qdq_dwconv2d_relu_pattern,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Int16 check function
# =========================================================================


def _is_const_zero(expr) -> bool:
    if isinstance(expr, relax.Constant):
        return np.all(expr.data.numpy() == 0)
    return False


def _check_mmalib_qdq_dwconv2d_i16(ctx) -> bool:
    """Validate MMALIB depthwise eligibility for an int16 QDQ pattern match.

    Same constraints as the int8 version (_check_mmalib_qdq_dwconv2d) plus:
      - weight dtype must be int16
      - data dtype must be int16
      - d_zp must be 0 (int16 activations are always symmetric)
    """
    # Weight zero-point must be 0 (symmetric, required by MMALIB)
    w_zp = ctx.annotated_expr["w_zp"]
    if not _is_const_zero(w_zp):
        return False

    # Activation zero-point must be 0 for int16 (symmetric only)
    d_zp = ctx.annotated_expr["d_zp"]
    if not _is_const_zero(d_zp):
        return False

    # Output zero-point must be 0 — the i16 lowerer does not fold o_zp.
    # Direct [] access: all depthwise patterns include "o_zp" in annotations.
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

    # Data must be int16 (not a constant — it's the dynamic input)
    data = ctx.annotated_expr["data"]
    if isinstance(data, relax.Constant):
        return False
    if hasattr(data, "struct_info") and hasattr(data.struct_info, "dtype"):
        if str(data.struct_info.dtype) != "int16":
            return False

    # Bias (if present) must resolve to a compile-time constant -- see the
    # matching comment in ti_mmalib_qdq_fusion.py's _check_mmalib_qdq_conv2d.
    if "bias" in ctx.annotated_expr:
        if _resolve_constant_tensor(ctx.annotated_expr["bias"]) is None:
            return False

    # Shared geometry constraints (groups==C_in, strides, dilation, N==1, static shapes).
    # INT16 depthwise: MMALIB only supports 3×3 kernels (MMALIB-882 tracks 5×5/7×7),
    # so allowed_kh_sizes=(3,).  max_kh_stride2 is irrelevant since kh is always 3.
    return _check_dwconv2d_geometry(ctx, allowed_kh_sizes=(3,), max_kh_stride2=3)


# =========================================================================
# Int16 pattern factories
#
# Structurally identical to the int8 patterns; the only difference is the
# check function.  Reuse the int8 pattern bodies and replace the check fn.
# =========================================================================


def _qdq_dwconv2d_i16_bias_relu_pattern():
    pat, annotations, _ = _qdq_dwconv2d_bias_relu_pattern()
    return pat, annotations, _check_mmalib_qdq_dwconv2d_i16


def _qdq_dwconv2d_i16_bias_pattern():
    pat, annotations, _ = _qdq_dwconv2d_bias_pattern()
    return pat, annotations, _check_mmalib_qdq_dwconv2d_i16


def _qdq_dwconv2d_i16_relu_pattern():
    pat, annotations, _ = _qdq_dwconv2d_relu_pattern()
    return pat, annotations, _check_mmalib_qdq_dwconv2d_i16


def _qdq_dwconv2d_i16_pattern():
    pat, annotations, _ = _qdq_dwconv2d_pattern()
    return pat, annotations, _check_mmalib_qdq_dwconv2d_i16


# =========================================================================
# Pattern registry
# =========================================================================

_PATTERN_REGISTRY = [
    ("mmalib.dwconv2d_i16_qdq_bias_relu", _qdq_dwconv2d_i16_bias_relu_pattern),
    ("mmalib.dwconv2d_i16_qdq_bias", _qdq_dwconv2d_i16_bias_pattern),
    ("mmalib.dwconv2d_i16_qdq_relu", _qdq_dwconv2d_i16_relu_pattern),
    ("mmalib.dwconv2d_i16_qdq", _qdq_dwconv2d_i16_pattern),
]
_COMPOSITE_NAMES = frozenset(name for name, _ in _PATTERN_REGISTRY)


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _MMALIBQDQDwConvI16Lowerer(PyExprMutator):
    """Replace MMALIB int16 QDQ depthwise composites with call_tir."""

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
        """Lower composite to call_tir with mmalib_depthwise_conv2d_i16."""
        param_to_arg = dict(zip(func.params, call.args))
        # Reuse the i8 lowerer's _extract_roles — it's dtype-independent
        roles = _MMALIBQDQDwConvLowerer._extract_roles(func, has_bias)

        required = ["data", "w_int8", "w_scale", "d_scale", "o_scale", "conv_attrs"]
        if any(r not in roles for r in required):
            logger.warning("Could not identify roles in MMALIB i16 depthwise composite")
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

        # int16 always uses symmetric activation (d_zp=0), so no ZP correction.

        bias_np = None
        if has_bias and "bias" in roles:
            bias_arg = param_to_arg[roles["bias"]]
            resolved = _resolve_constant_tensor(bias_arg, lookup=self.lookup_binding)
            if resolved is not None:
                bias_np = resolved.flatten()

        # Extract dimensions: weight is [C_out, 1, KH, KW] (OIHW) for depthwise
        attrs = roles["conv_attrs"]
        channels = w_i16_np.shape[0]
        KH, KW = w_i16_np.shape[2], w_i16_np.shape[3]

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

        # Bias in int64 accumulator scale (no ZP correction since d_zp=0)
        dw_scale = d_scale_val * w_scale_np[:channels]
        if bias_np is not None:
            bias_i64 = np.round(bias_np[:channels] / dw_scale).astype(np.int64)
        else:
            bias_i64 = np.zeros(channels, dtype=np.int64)

        # Per-channel requantization: same formula as int8
        combined_rescale = dw_scale / o_scale_val
        scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)

        # Natural-order int16 weights [C, KH*KW], flattened to [C*KH*KW].
        # The C wrapper reorders at runtime via reorderWeights_exec.
        nat_weights = w_i16_np.reshape(channels, KH * KW).flatten()

        weights_relax = relax.Constant(nat_weights)
        bias_relax = relax.Constant(bias_i64)
        scale_relax = relax.Constant(scale_u8)
        shift_relax = relax.Constant(shift_u8)

        def te_mmalib_dwconv2d_i16(
            data_t: te.Tensor,
            weight_t: te.Tensor,
            bias_t: te.Tensor,
            scale_t: te.Tensor,
            shift_t: te.Tensor,
        ) -> te.Tensor:
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "mmalib_depthwise_conv2d_i16",
                    ins[0].data,  # input  [1, C, H_in, W_in], int16
                    ins[1].data,  # weights [C*KH*KW], int16 (natural order)
                    ins[2].data,  # bias  [C], int64
                    ins[3].data,  # scale [C], uint8
                    ins[4].data,  # shift [C], uint8
                    outs[0].data,  # output [1, C, H_out, W_out], int16
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
                    channels,  # num_groups == channels for depthwise
                )

            return te.extern(
                (1, channels, H_out, W_out),
                [data_t, weight_t, bias_t, scale_t, shift_t],
                fcompute,
                # Use "mmalib_dwconv2d" not "mmalib_depthwise_conv2d_i16" to
                # avoid the C codegen creating a local variable that shadows
                # the extern function declaration of the same name.
                name="mmalib_dwconv2d",
                dtype="int16",
            )

        result = self.builder_.call_te(
            te_mmalib_dwconv2d_i16,
            data_arg,
            weights_relax,
            bias_relax,
            scale_relax,
            shift_relax,
            primfunc_name_hint="mmalib_dwconv2d_i16",
            primfunc_attrs={"c7x_offload_backend": "mmalib"},
        )

        # Attach the span to the conv2d call itself, before any relu clip
        # wraps it -- otherwise propagate_span would attach it to the
        # trailing clip node instead (result gets reassigned below), so the
        # visualizer's source lookup on the MMALIB conv2d node it actually
        # highlights would come up empty.
        result = propagate_span(result, roles["conv_call"])
        if has_relu:
            # Clip to int16 range (not int8)
            result = relax.op.clip(result, relax.PrimValue(-32768), relax.PrimValue(32767))

        self.count += 1
        logger.info(
            "MMALIB i16 QDQ dwconv2d fusion #%d: %d channels %dx%d (stride=%d)%s%s",
            self.count,
            channels,
            KH,
            KW,
            stride_h,
            " +bias" if has_bias else "",
            " +relu" if has_relu else "",
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseMMALIBQDQDwConv2dI16")
class FuseMMALIBQDQDwConv2dI16:
    """Fuse PT2E int16 QDQ depthwise conv2d patterns into MMALIB int16 calls.

    Mirrors FuseMMALIBQDQDwConv2d for int16 precision.  Requires:
      - weight dtype int16, per-channel symmetric (w_zp=0)
      - activation dtype int16, per-tensor symmetric (d_zp=0)
      - kernel sizes 3×3, 5×5, or 7×7; strides 1-2; dilation 1

    Emits mmalib_depthwise_conv2d_i16 which calls
    MMALIB_CNN_convolve_col_smallNo_highPrecision with MMALIB_INT16 datatype.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = [(name, *factory()) for name, factory in _PATTERN_REGISTRY]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _MMALIBQDQDwConvI16Lowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseMMALIBQDQDwConv2dI16: fused %d layers", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
