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
"""Fuse QDQ-wrapped layer normalization into a single int8 C kernel call.

Matches the PT2E pattern for layer_norm after C7xMMAQuantizer annotation:

    dq(x_i8, d_scale, d_zp) → R.nn.layer_norm(dq, weight, bias) → q(out, o_scale, o_zp)

where `weight` and `bias` are float32 constants.

Replaces with call_extern to c7x_int8_layer_norm which:
  - Accepts int8 input + float32 weight/bias + quantization parameters
  - Runs normalization in float32 (dequant → norm → requant)
  - Produces int8 output

Kernel: src/runtime/ti_dsp/kernels/c7x_norm.c
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

_COMPOSITE_NAME = "tidl_norm.layer_norm"


# =========================================================================
# Pattern definition
# =========================================================================


def _make_layer_norm_pattern():
    """dq(x_i8) → R.nn.layer_norm(dq, weight, bias) → q(out)"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    w = wildcard()
    b = wildcard()
    ln = is_op("relax.nn.layer_norm")(dq, w, b)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(ln, o_s, o_z)
    return q, {
        "x": x,
        "w": w,
        "b": b,
        "d_scale": d_s,
        "d_zp": d_z,
        "o_scale": o_s,
        "o_zp": o_z,
    }


def _check_layer_norm(ctx) -> bool:
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
class _LayerNormLowerer(PyExprMutator):
    """Lower layer_norm composite to call_extern."""

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

        if str(func.attrs["Composite"]) != _COMPOSITE_NAME:
            return super().visit_call_(call)

        return self._lower(call, func)

    def _lower(self, call, func):
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = w_arg = b_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None
        eps_val = 1e-5

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
                pass  # x_sinfo not needed; shape taken from call.struct_info
            elif "layer_norm" in op_name and x_arg is not None:
                w_param = val.args[1]
                b_param = val.args[2]
                w_arg = param_to_arg.get(w_param, w_param)
                b_arg = param_to_arg.get(b_param, b_param)
                if hasattr(val, "attrs") and val.attrs is not None:
                    eps_val = float(val.attrs.epsilon)
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if any(v is None for v in (x_arg, w_arg, b_arg, o_scale_val)):
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)
        out_shape = [int(s) for s in call_sinfo.shape]

        # Compute outer_size (all dims except last = norm dimension)
        # and norm_size (last dim).
        if len(out_shape) < 2:
            return super().visit_call_(call)
        norm_size_v = out_shape[-1]
        outer_size_v = 1
        for s in out_shape[:-1]:
            outer_size_v *= s

        d_zp_v = int(d_zp_val)  # type: ignore[arg-type]
        d_scale_v = float(d_scale_val)  # type: ignore[arg-type]
        o_zp_v = int(o_zp_val)  # type: ignore[arg-type]
        o_scale_v = float(o_scale_val)  # type: ignore[arg-type]
        outer_v = outer_size_v
        norm_v = norm_size_v
        eps_v = float(eps_val)

        def te_layer_norm(x_t, w_t, b_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_layer_norm",
                    ins[0].data,   # in (int8)
                    ins[1].data,   # weight (float32)
                    ins[2].data,   # bias (float32)
                    outs[0].data,  # out (int8)
                    tir.IntImm("int32", outer_v),
                    tir.IntImm("int32", norm_v),
                    tir.FloatImm("float32", eps_v),
                    tir.IntImm("int32", d_zp_v),
                    tir.FloatImm("float32", d_scale_v),
                    tir.IntImm("int32", o_zp_v),
                    tir.FloatImm("float32", o_scale_v),
                )

            return te.extern(
                out_shape, [x_t, w_t, b_t], fcompute,
                name="tidl_norm_out", dtype="int8",
            )

        result = self.builder_.call_te(
            te_layer_norm, x_arg, w_arg, b_arg,
            primfunc_name_hint="c7x_int8_layer_norm",
        )

        self.count += 1
        logger.debug(
            "Fused c7x_int8_layer_norm: outer=%d norm=%d eps=%.2e",
            outer_size_v, norm_size_v, eps_val,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xLayerNorm")
class FuseQDQToC7xLayerNorm:
    """Fuse QDQ-wrapped layer_norm into c7x_int8_layer_norm C kernel call.

    Requires int8 input, float32 weight/bias, and compile-time quantization
    constants.  The normalization runs in float32 internally.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat, annotations = _make_layer_norm_pattern()
        patterns = [(_COMPOSITE_NAME, pat, annotations, _check_layer_norm)]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _LayerNormLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseQDQToC7xLayerNorm: fused %d layer_norm ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
