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
"""Fuse the input float32→int8 quantize into a vectorized C7x kernel call.

Why this pass is needed
-----------------------
PT2E quantization inserts R.quantize(x_float32, scale, zp, out_dtype="int8")
as the first op in the graph.  TVM's default quantize lowering (LegalizeOps)
produces a scalar TIR loop at ~75 cycles/element (11.3M for 150K inputs).

This pass intercepts R.quantize before LegalizeOps and replaces it with a
call_extern to c7x_int8_quantize — a C7x vectorized kernel that uses
VMPYSP+VADDSP+VMAXSP+VMINSP+VSPINT+VSTWSVPACKB (8 elements per cycle on C7524)
achieving ~50K cycles, a ~225× improvement.

Pattern matched:
  R.quantize(x_float32, scale_const, zp_const, out_dtype="int8", axis=0)
  → call_extern c7x_int8_quantize(in, out, n, inv_scale, zp)

inv_scale = 1.0f / scale is precomputed here so the kernel has no division.

Only per-tensor quantize is matched (scalar scale constant, rank-0 or shape []).
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

_COMPOSITE_NAME = "c7x.input_quantize"


def is_per_tensor_scalar_constant(expr) -> bool:
    """True if expr is a compile-time constant with scalar (rank-0) shape.

    Shared by FuseInputQuantize and FuseInputNormalizeQuantize to guard a
    trailing quantize's scale/zp: both kernels these lower to take one
    shared (scale, zp) pair for the whole tensor, not per-channel.
    """
    if not isinstance(expr, relax.Constant):
        return False
    shape = expr.struct_info.shape
    return shape is None or len(shape) == 0


# =========================================================================
# Pattern definition
# =========================================================================


def _make_quantize_pattern():
    """R.quantize(x_float32, scale_const, zp_const, out_dtype='int8')"""
    x = wildcard()
    scale = wildcard()
    zp = wildcard()
    q = is_op("relax.quantize")(x, scale, zp)
    return q, {"x": x, "scale": scale, "zp": zp}


def _check_quantize(ctx) -> bool:
    """Match only the model's input float32→int8 quantize with constant scale/zp.

    The initial input quantize has the model's float32 input as its operand,
    which is a function parameter (relax.Var).  Intermediate quantize ops have
    dataflow-binding outputs as their operands (relax.DataflowVar).  Excluding
    DataflowVar inputs limits the match to the float32 model input only.
    """
    x = ctx.annotated_expr["x"]

    # In ctx.annotated_expr, the model's float32 input is a relax.Var
    # (function parameter).  Intermediate activations are relax.Call
    # (e.g. the output of dequantize) or relax.TupleGetItem.
    # Only accept Var to match the initial model input quantize.
    if not isinstance(x, relax.Var):
        return False
    if hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        if str(x.struct_info.dtype) != "float32":
            return False
    else:
        return False

    # scale and zp must be compile-time per-tensor scalar constants
    scale = ctx.annotated_expr["scale"]
    zp = ctx.annotated_expr["zp"]
    if not is_per_tensor_scalar_constant(scale) or not isinstance(zp, relax.Constant):
        return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _QuantizeLowerer(PyExprMutator):
    """Lower c7x.input_quantize composites to call_extern c7x_int8_quantize."""

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

        x_arg = None
        scale_val = None
        zp_val = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if not isinstance(val, relax.Call):
                continue
            if not hasattr(val.op, "name"):
                continue
            if "quantize" not in str(val.op.name):
                continue
            # Found the quantize op — extract x, scale, zp
            x_param = val.args[0]
            x_arg = param_to_arg.get(x_param, x_param)
            s = param_to_arg.get(val.args[1], val.args[1])
            z = param_to_arg.get(val.args[2], val.args[2])
            if isinstance(s, relax.Constant) and isinstance(z, relax.Constant):
                scale_val = float(s.data.numpy())
                zp_val = int(z.data.numpy())
            break

        if x_arg is None or scale_val is None or scale_val == 0.0:
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)

        out_shape = [int(s) for s in call_sinfo.shape]
        n_elems = 1
        for s in out_shape:
            n_elems *= s

        _n = n_elems
        _inv_scale = float(1.0 / scale_val)
        _zp = int(zp_val)

        def te_quantize(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_quantize",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", _n),
                    tir.FloatImm("float32", _inv_scale),
                    tir.IntImm("int32", _zp),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="quantize_out",
                dtype="int8",
            )

        result = self.builder_.call_te(te_quantize, x_arg, primfunc_name_hint="c7x_int8_quantize")
        self.count += 1
        logger.debug(
            "Fused c7x_int8_quantize: n=%d scale=%.6g inv_scale=%.6g zp=%d",
            n_elems,
            scale_val,
            _inv_scale,
            _zp,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseInputQuantize")
class FuseInputQuantize:
    """Replace R.quantize(float32→int8) with call_extern to c7x_int8_quantize.

    Must run before LegalizeOps (which lowers R.quantize to a scalar TIR loop)
    and before EliminateQDQTransparent.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat, annotations = _make_quantize_pattern()
        patterns = [(_COMPOSITE_NAME, pat, annotations, _check_quantize)]

        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _QuantizeLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseInputQuantize: fused %d quantize ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
