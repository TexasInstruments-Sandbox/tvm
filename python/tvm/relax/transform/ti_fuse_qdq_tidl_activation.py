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
"""Fuse QDQ-wrapped activation ops into int8 TIDL activation kernel calls.

Handles four non-linear activations annotated by C7xMMAQuantizer._TIDL_ACT_OPS.
Each PT2E activation annotation wraps the full aten op expansion with a single
dequantize before and a single quantize after.  After from_exported_program the
Relax IR has these sub-graph structures:

  gelu:         dq → R.nn.gelu → q
  silu:         dq → sigmoid(dq) → multiply(dq, sigmoid) → q
  hardsigmoid:  dq → add(dq, 3) → clip → clip → divide(_, 6) → q
  hardswish:    dq → add(dq, 3) → clip → clip → multiply(dq, clip) → divide → q

Replaces each with call_extern to the corresponding tidl_int8_* C wrapper.

Kernels: src/runtime/ti_dsp/kernels/tidl_activation_wrappers.c
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

_COMPOSITE_PREFIX = "tidl_act."


# =========================================================================
# Pattern definitions
# =========================================================================


def _make_gelu_pattern():
    """dq → R.nn.gelu → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    act = is_op("relax.nn.gelu")(dq)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(act, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _make_silu_pattern():
    """dq → sigmoid(dq) → multiply(dq, sigmoid) → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    sig = is_op("relax.sigmoid")(dq)
    mul = is_op("relax.multiply")(dq, sig)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(mul, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _make_hardsigmoid_pattern():
    """dq → add(dq, 3) → clip → clip → divide(_, 6) → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    added = is_op("relax.add")(dq, wildcard())
    clipped1 = is_op("relax.clip")(added, wildcard(), wildcard())
    clipped2 = is_op("relax.clip")(clipped1, wildcard(), wildcard())
    divided = is_op("relax.divide")(clipped2, wildcard())
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(divided, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _make_hardswish_pattern():
    """dq → add(dq, 3) → clip → clip → multiply(dq, clip) → divide → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    added = is_op("relax.add")(dq, wildcard())
    clipped1 = is_op("relax.clip")(added, wildcard(), wildcard())
    clipped2 = is_op("relax.clip")(clipped1, wildcard(), wildcard())
    mul = is_op("relax.multiply")(dq, clipped2)
    divided = is_op("relax.divide")(mul, wildcard())
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(divided, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


# (composite_name, pattern_factory)
_PATTERN_REGISTRY = [
    ("tidl_act.gelu", _make_gelu_pattern),
    ("tidl_act.silu", _make_silu_pattern),
    ("tidl_act.hardsigmoid", _make_hardsigmoid_pattern),
    ("tidl_act.hardswish", _make_hardswish_pattern),
]

_COMPOSITE_NAMES = frozenset(name for name, _ in _PATTERN_REGISTRY)


def _check_activation(ctx) -> bool:
    """Require int8 input and compile-time quantization constants."""
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
    return True


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _ActivationLowerer(PyExprMutator):
    """Lower TIDL activation composites to call_extern."""

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

        name = str(func.attrs["Composite"])
        if name not in _COMPOSITE_NAMES:
            return super().visit_call_(call)

        return self._lower(call, func, name)

    def _lower(self, call, func, composite_name):
        param_to_arg = dict(zip(func.params, call.args))

        # Walk the composite body to extract the single dq and q parameters.
        # For all activation patterns there is exactly one dequantize (entry)
        # and one quantize (exit); the intermediate ops vary by activation type.
        x_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None

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
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if x_arg is None or o_scale_val is None:
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)
        output_shape = [int(s) for s in call_sinfo.shape]

        n_elem = 1
        for s in output_shape:
            n_elem *= s

        # "tidl_act.gelu" → "tidl_int8_gelu"
        act_suffix = composite_name[len(_COMPOSITE_PREFIX):]
        extern_name = f"tidl_int8_{act_suffix}"

        d_zp_v = int(d_zp_val)  # type: ignore[arg-type]
        d_scale_v = float(d_scale_val)  # type: ignore[arg-type]
        o_zp_v = int(o_zp_val)  # type: ignore[arg-type]
        o_scale_v = float(o_scale_val)
        n = n_elem

        def te_activation(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", n),
                    tir.IntImm("int32", d_zp_v),
                    tir.FloatImm("float32", d_scale_v),
                    tir.IntImm("int32", o_zp_v),
                    tir.FloatImm("float32", o_scale_v),
                )

            return te.extern(output_shape, [x_t], fcompute, name="tidl_act_out", dtype="int8")

        result = self.builder_.call_te(te_activation, x_arg, primfunc_name_hint=extern_name)

        self.count += 1
        logger.debug(
            "Fused %s: n=%d d_zp=%d d_scale=%.6g o_zp=%d o_scale=%.6g",
            extern_name,
            n_elem,
            d_zp_v,
            d_scale_v,
            o_zp_v,
            o_scale_v,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToTIDLActivation")
class FuseQDQToTIDLActivation:
    """Fuse QDQ-wrapped activation ops into tidl_int8_* C kernel calls.

    Handles: gelu, silu, hardsigmoid, hardswish.
    Requires int8 input and compile-time quantization constants (satisfied
    by the PT2E pipeline after convert_pt2e).

    Applicable to both MMALIB and non-MMALIB C7x targets.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = []
        for composite_name, factory in _PATTERN_REGISTRY:
            pat, annotations = factory()
            patterns.append((composite_name, pat, annotations, _check_activation))

        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _ActivationLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseQDQToTIDLActivation: fused %d activation ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
