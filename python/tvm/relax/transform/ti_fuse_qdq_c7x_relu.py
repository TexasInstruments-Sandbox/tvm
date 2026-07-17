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
"""Fuse QDQ-wrapped relu / clip into vectorizable int8 C kernel calls.

Why this pass runs BEFORE EliminateQDQTransparent
-------------------------------------------------
relu and clip are "transparent" ops: when input and output quantization
parameters are identical, dq→op→q simplifies to op(int8) with no float
conversion.  EliminateQDQTransparent (step 3) exploits this and removes
the Q/DQ wrappers, leaving a bare int8 op lowered by LegalizeOps to a
slow scalar TIR loop (~4–75 cycles/element).

Patterns matched
----------------
1. dq(x_i8, d_scale, d_zp) → relax.nn.relu → q(out, o_scale, o_zp)
   → call_extern c7x_int8_relu(in, out, n, clip_lo=d_zp)
   Transparent condition: d_zp == o_zp

2. dq(x_i8, d_scale, d_zp) → relax.clip(dq, a_min, a_max) → q(out, o_scale, o_zp)
   → call_extern c7x_int8_clamp(in, out, n,
         clip_lo=round(a_min/d_scale)+d_zp,
         clip_hi=round(a_max/d_scale)+d_zp)
   Transparent condition: d_zp == o_zp AND d_scale ≈ o_scale
   Covers ReLU6 (a_min=0, a_max=6) and any general symmetric clip.
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

_COMPOSITE_NAME = "tidl_act.relu"
_COMPOSITE_NAME_CLAMP = "tidl_act.clamp"           # transparent: d_scale ≈ o_scale
_COMPOSITE_NAME_REQCLAMP = "tidl_act.reqclamp"      # non-transparent: d_scale ≠ o_scale


# =========================================================================
# Pattern definition
# =========================================================================


def _make_relu_pattern():
    """dq → relax.nn.relu → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    relu_out = is_op("relax.nn.relu")(dq)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(relu_out, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _check_relu(ctx) -> bool:
    """Require int8 input, compile-time QDQ constants, and transparent scales.

    The transparent condition (d_zp == o_zp) is required because the kernel
    does no float conversion.  C7xMMAQuantizer annotates relu with
    SharedQuantizationSpec (output shares input params), so this always holds
    for quantized models produced by that quantizer.
    """
    x = ctx.annotated_expr["x"]
    if isinstance(x, relax.Constant):
        if x.data.dtype != "int8":
            return False
    elif hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        if str(x.struct_info.dtype) != "int8":
            return False
    else:
        return False

    for name in ("d_scale", "d_zp", "o_scale", "o_zp"):
        if not isinstance(ctx.annotated_expr[name], relax.Constant):
            return False

    # Transparent condition: zero-points must match (clip preserves scale).
    d_zp = ctx.annotated_expr["d_zp"]
    o_zp = ctx.annotated_expr["o_zp"]
    if int(d_zp.data.numpy()) != int(o_zp.data.numpy()):
        return False

    return True


def _make_clamp_pattern():
    """dq → relax.clip(_, a_min, a_max) → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    a_min, a_max = wildcard(), wildcard()
    clipped = is_op("relax.clip")(dq, a_min, a_max)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(clipped, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z,
                "a_min": a_min, "a_max": a_max,
                "o_scale": o_s, "o_zp": o_z}


def _get_scalar_float(expr) -> float:
    """Extract a float value from relax.Constant or relax.PrimValue."""
    if isinstance(expr, relax.Constant):
        return float(expr.data.numpy())
    if isinstance(expr, relax.PrimValue):
        return float(expr.value.value)
    return None


def _check_clamp(ctx) -> bool:
    """Transparent dq→clip→q: d_scale ≈ o_scale AND d_zp == o_zp.

    The int8 clip bounds are computed from the clip float bounds using the
    (shared) scale.  Transparent means the QDQ context can be eliminated.
    """
    x = ctx.annotated_expr["x"]
    if isinstance(x, relax.Constant):
        if x.data.dtype != "int8":
            return False
    elif hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        if str(x.struct_info.dtype) != "int8":
            return False
    else:
        return False

    for name in ("d_scale", "d_zp", "o_scale", "o_zp"):
        if not isinstance(ctx.annotated_expr[name], relax.Constant):
            return False
    # a_min/a_max may be Constant or PrimValue
    if _get_scalar_float(ctx.annotated_expr["a_min"]) is None:
        return False
    if _get_scalar_float(ctx.annotated_expr["a_max"]) is None:
        return False

    d_zp = int(ctx.annotated_expr["d_zp"].data.numpy())
    o_zp = int(ctx.annotated_expr["o_zp"].data.numpy())
    if d_zp != o_zp:
        return False

    d_scale = float(ctx.annotated_expr["d_scale"].data.numpy())
    o_scale = float(ctx.annotated_expr["o_scale"].data.numpy())
    if not np.isclose(d_scale, o_scale, rtol=1e-5):
        return False

    return True


def _check_reqclamp(ctx) -> bool:
    """Non-transparent dq→clip→q: d_scale ≠ o_scale, requires rescale.

    Only matches when d_zp == o_zp == 0 (symmetric quantization) so that
    the operation reduces to clamp(round(in * combined_scale), lo, hi).
    """
    x = ctx.annotated_expr["x"]
    if isinstance(x, relax.Constant):
        if x.data.dtype != "int8":
            return False
    elif hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        if str(x.struct_info.dtype) != "int8":
            return False
    else:
        return False

    for name in ("d_scale", "d_zp", "o_scale", "o_zp"):
        if not isinstance(ctx.annotated_expr[name], relax.Constant):
            return False
    if _get_scalar_float(ctx.annotated_expr["a_min"]) is None:
        return False
    if _get_scalar_float(ctx.annotated_expr["a_max"]) is None:
        return False

    # Require symmetric quantization (both zero-points == 0) for simplicity.
    d_zp = int(ctx.annotated_expr["d_zp"].data.numpy())
    o_zp = int(ctx.annotated_expr["o_zp"].data.numpy())
    if d_zp != 0 or o_zp != 0:
        return False

    # Must NOT be transparent (transparent case is handled by _check_clamp).
    d_scale = float(ctx.annotated_expr["d_scale"].data.numpy())
    o_scale = float(ctx.annotated_expr["o_scale"].data.numpy())
    if np.isclose(d_scale, o_scale, rtol=1e-5):
        return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _ReluLowerer(PyExprMutator):
    """Lower TIDL relu composites to call_extern."""

    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.count = 0

    def visit_call_(self, call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)

        func = self.builder_.get()[call.op]
        if not isinstance(func, relax.Function):
            return super().visit_call_(call)
        if "Composite" not in func.attrs:
            return super().visit_call_(call)
        composite = str(func.attrs["Composite"])
        if composite == _COMPOSITE_NAME:
            return self._lower(call, func)
        if composite == _COMPOSITE_NAME_CLAMP:
            return self._lower_clamp(call, func)
        if composite == _COMPOSITE_NAME_REQCLAMP:
            return self._lower_reqclamp(call, func)
        return super().visit_call_(call)

    def _lower(self, call, func):
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = None
        d_zp_val = None
        x_sinfo = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if not isinstance(val, relax.Call):
                continue
            if not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)
            if "dequantize" in op_name:
                x_param = val.args[0]
                x_arg = param_to_arg.get(x_param, x_param)
                z = param_to_arg.get(val.args[2], val.args[2])
                d_zp_val = int(z.data.numpy())
                if hasattr(binding.var, "struct_info"):
                    x_sinfo = binding.var.struct_info

        if x_arg is None or d_zp_val is None or x_sinfo is None:
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)

        out_shape = [int(s) for s in call_sinfo.shape]
        n_v = 1
        for s in out_shape:
            n_v *= s

        _n = n_v
        _clip_lo = int(d_zp_val)

        def te_relu(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_relu",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", _n),
                    tir.IntImm("int32", _clip_lo),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="tidl_relu_out",
                dtype="int8",
            )

        result = self.builder_.call_te(te_relu, x_arg, primfunc_name_hint="c7x_int8_relu")
        self.count += 1
        logger.debug("Fused c7x_int8_relu: n=%d clip_lo=%d", n_v, _clip_lo)
        return result

    def _lower_clamp(self, call, func):
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = None
        d_scale_val = None
        d_zp_val = None
        a_min_float = None
        a_max_float = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if not isinstance(val, relax.Call):
                continue
            if not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)
            if "dequantize" in op_name:
                x_param = val.args[0]
                x_arg = param_to_arg.get(x_param, x_param)
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                d_scale_val = float(s.data.numpy())
                d_zp_val = int(z.data.numpy())
            elif "clip" in op_name:
                lo = param_to_arg.get(val.args[1], val.args[1])
                hi = param_to_arg.get(val.args[2], val.args[2])
                a_min_float = _get_scalar_float(lo)
                a_max_float = _get_scalar_float(hi)

        if any(v is None for v in [x_arg, d_scale_val, d_zp_val, a_min_float, a_max_float]):
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)

        out_shape = [int(s) for s in call_sinfo.shape]
        n_v = 1
        for s in out_shape:
            n_v *= s

        # Convert float bounds to int8 using input quantization parameters.
        def _to_int8(float_val):
            return max(-128, min(127, int(round(float_val / d_scale_val)) + d_zp_val))

        _n = n_v
        _clip_lo = _to_int8(a_min_float)
        _clip_hi = _to_int8(a_max_float)

        def te_clamp(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_clamp",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", _n),
                    tir.IntImm("int32", _clip_lo),
                    tir.IntImm("int32", _clip_hi),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="tidl_clamp_out",
                dtype="int8",
            )

        result = self.builder_.call_te(te_clamp, x_arg, primfunc_name_hint="c7x_int8_clamp")
        self.count += 1
        logger.debug(
            "Fused c7x_int8_clamp: n=%d clip_lo=%d clip_hi=%d "
            "(float [%.4g, %.4g] scale=%.4g)",
            n_v, _clip_lo, _clip_hi, a_min_float, a_max_float, d_scale_val,
        )
        return result

    def _lower_reqclamp(self, call, func):
        """Lower non-transparent dq→clip→q to c7x_int8_requantize_clamp."""
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = None
        d_scale_val = None
        o_scale_val = None
        a_min_float = None
        a_max_float = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if not isinstance(val, relax.Call):
                continue
            if not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)
            if "dequantize" in op_name:
                x_param = val.args[0]
                x_arg = param_to_arg.get(x_param, x_param)
                s = param_to_arg.get(val.args[1], val.args[1])
                d_scale_val = float(s.data.numpy())
            elif "clip" in op_name:
                lo = param_to_arg.get(val.args[1], val.args[1])
                hi = param_to_arg.get(val.args[2], val.args[2])
                a_min_float = _get_scalar_float(lo)
                a_max_float = _get_scalar_float(hi)
            elif "quantize" in op_name and "de" not in op_name:
                s = param_to_arg.get(val.args[1], val.args[1])
                o_scale_val = float(s.data.numpy())

        if any(v is None for v in [x_arg, d_scale_val, o_scale_val, a_min_float, a_max_float]):
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)

        out_shape = [int(s) for s in call_sinfo.shape]
        n_v = 1
        for s in out_shape:
            n_v *= s

        _n = n_v
        _combined_scale = float(d_scale_val / o_scale_val)
        _clip_lo = max(-128, min(127, int(round(a_min_float / o_scale_val))))
        _clip_hi = max(-128, min(127, int(round(a_max_float / o_scale_val))))

        def te_reqclamp(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_requantize_clamp",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", _n),
                    tir.FloatImm("float32", _combined_scale),
                    tir.IntImm("int32", _clip_lo),
                    tir.IntImm("int32", _clip_hi),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="tidl_reqclamp_out",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_reqclamp, x_arg, primfunc_name_hint="c7x_int8_requantize_clamp"
        )
        self.count += 1
        logger.debug(
            "Fused c7x_int8_requantize_clamp: n=%d combined_scale=%.4g "
            "clip_lo=%d clip_hi=%d (float [%.4g, %.4g])",
            n_v, _combined_scale, _clip_lo, _clip_hi, a_min_float, a_max_float,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xRelu")
class FuseQDQToC7xRelu:
    """Fuse QDQ-wrapped relu into a c7x_int8_relu C kernel call.

    Must run before EliminateQDQTransparent: that pass removes the Q/DQ
    context that this pass needs to identify and match the pattern.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat_relu, ann_relu = _make_relu_pattern()
        pat_clamp, ann_clamp = _make_clamp_pattern()
        # reqclamp reuses the same DFPattern as clamp; check fn differentiates
        pat_req, ann_req = _make_clamp_pattern()
        patterns = [
            (_COMPOSITE_NAME,          pat_relu,  ann_relu,  _check_relu),
            (_COMPOSITE_NAME_CLAMP,    pat_clamp, ann_clamp, _check_clamp),
            (_COMPOSITE_NAME_REQCLAMP, pat_req,   ann_req,   _check_reqclamp),
        ]

        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _ReluLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseQDQToC7xRelu: fused %d relu/clamp ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
