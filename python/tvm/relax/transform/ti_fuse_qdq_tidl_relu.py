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
"""Fuse QDQ-wrapped relu into a vectorizable int8 C kernel call.

Why this pass runs BEFORE EliminateQDQTransparent
-------------------------------------------------
relu is a "transparent" op (clip is monotone): when input and output
quantization parameters are identical, dq→relu→q simplifies to relu(int8)
with no float conversion.  EliminateQDQTransparent (step 3) exploits this
and removes the Q/DQ wrappers, leaving a bare int8 relu that is then lowered
by LegalizeOps to a scalar TIR loop.  That scalar loop is slow (~4–10 M
cycles per relu on AM67A C7x @ 1 GHz).

By running at step 2.5 (after CanonicalizeBindings/DCE, before
EliminateQDQTransparent), this pass intercepts the QDQ-wrapped form and
replaces it with a call_extern to c7x_int8_relu — a flat C loop that
cl7x auto-vectorizes with SIMD int8 max instructions.

Why clip_lo = d_zp
-------------------
relu clips at float 0.0.  In the int8 quantized domain, float 0.0 is
represented by the zero_point (d_zp).  For symmetric quantization d_zp=0
so this reduces to max(x, 0) on int8, but the formula generalises to
asymmetric quant.

Pattern matched:
  dq(x_i8, d_scale, d_zp) → relax.nn.relu → q(out, o_scale, o_zp)
  → call_extern c7x_int8_relu(in, out, n, clip_lo=d_zp)

Kernel: src/runtime/ti_dsp/kernels/tidl_pool_relu_wrappers.c
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

_COMPOSITE_NAME = "tidl_act.relu"


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
        if str(func.attrs["Composite"]) != _COMPOSITE_NAME:
            return super().visit_call_(call)

        return self._lower(call, func)

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


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToTIDLRelu")
class FuseQDQToTIDLRelu:
    """Fuse QDQ-wrapped relu into a c7x_int8_relu C kernel call.

    Must run before EliminateQDQTransparent: that pass removes the Q/DQ
    context that this pass needs to identify and match the pattern.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat, annotations = _make_relu_pattern()
        patterns = [(_COMPOSITE_NAME, pat, annotations, _check_relu)]

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
            logger.info("FuseQDQToTIDLRelu: fused %d relu ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
