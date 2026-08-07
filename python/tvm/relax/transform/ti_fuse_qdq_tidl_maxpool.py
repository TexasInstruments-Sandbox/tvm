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
"""Fuse QDQ-wrapped max_pool2d into a vectorizable int8 C kernel call.

Why this pass runs BEFORE EliminateQDQTransparent
-------------------------------------------------
max_pool2d is a "transparent" op (max is monotone): when input and output
quantization parameters are identical, dq→max_pool2d→q simplifies to
max_pool2d(int8) with no float conversion.  EliminateQDQTransparent (step 3)
exploits this and removes the Q/DQ wrappers, leaving a bare int8
max_pool2d that is then lowered by LegalizeOps to a scalar TIR loop.
That scalar loop is slow (~37 M cycles for 112×112×64 on AM67A C7x @ 1 GHz).

By running at step 2.5 (after CanonicalizeBindings/DCE, before
EliminateQDQTransparent), this pass intercepts the QDQ-wrapped form and
replaces it with a call_extern to c7x_int8_max_pool_tidl — a simple C loop that
cl7x can auto-vectorize with SIMD int8 instructions.  EliminateQDQTransparent
then finds no remaining max_pool2d QDQ patterns.

When the firmware has no TIDL-backed kernels (target attr `tidl-kernels=0`,
e.g. a no-TIDL beagley-ai build), this pass falls back to the native scalar
`c7x_int8_max_pool` kernel instead -- functionally correct but unvectorized.

Pattern matched:
  dq(x_i8, d_scale, d_zp) → relax.nn.max_pool2d → q(out, o_scale, o_zp)
  → call_extern c7x_int8_max_pool_tidl(in, out, N, C, H_in, W_in, H_out, W_out,
                                   kH, kW, sH, sW, pH, pW)

Kernel: src/runtime/ti_dsp/mmalib/tidl_maxpool_wrapper.cpp
        (fallback: src/runtime/ti_dsp/kernels/c7x_pool_relu.cpp)
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import find_composite_span, propagate_span

logger = logging.getLogger(__name__)

_COMPOSITE_NAME = "tidl_pool.max_pool2d"


# =========================================================================
# Pattern definition
# =========================================================================


def _make_max_pool_pattern():
    """dq → relax.nn.max_pool2d → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    pool = is_op("relax.nn.max_pool2d")(dq)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(pool, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _check_maxpool(ctx) -> bool:
    """Require int8 input, compile-time QDQ constants, and transparent scales.

    The transparent condition (d_zp == o_zp) is required because the kernel
    does no float conversion.  C7xMMAQuantizer annotates max_pool2d with
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

    # Transparent condition: input and output zero-points must match.
    d_zp = ctx.annotated_expr["d_zp"]
    o_zp = ctx.annotated_expr["o_zp"]
    if int(d_zp.data.numpy()) != int(o_zp.data.numpy()):
        return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _MaxPoolLowerer(PyExprMutator):
    """Lower TIDL max_pool2d composites to call_extern."""

    def __init__(self, mod: IRModule, use_tidl_maxpool: bool = True):
        super().__init__(mod)
        self.count = 0
        self.use_tidl_maxpool = use_tidl_maxpool

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
        pool_call = None
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
                if hasattr(binding.var, "struct_info"):
                    x_sinfo = binding.var.struct_info
            elif "max_pool" in op_name:
                pool_call = val

        if x_arg is None or pool_call is None or x_sinfo is None:
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)

        if not isinstance(x_sinfo, relax.TensorStructInfo) or not x_sinfo.shape:
            return super().visit_call_(call)

        in_shape = [int(s) for s in x_sinfo.shape]
        out_shape = [int(s) for s in call_sinfo.shape]
        if len(in_shape) != 4 or len(out_shape) != 4:
            return super().visit_call_(call)

        N_v, C_v, H_in_v, W_in_v = in_shape
        _, _, H_out_v, W_out_v = out_shape

        attrs = pool_call.attrs
        kH_v = int(attrs.pool_size[0])
        kW_v = int(attrs.pool_size[1])
        sH_v = int(attrs.strides[0])
        sW_v = int(attrs.strides[1])
        # padding may be [pH, pW] (2-element) or [pT, pL, pB, pR] (4-element);
        # take top/left values (symmetric padding assumed for quantized CNNs).
        pH_v = int(attrs.padding[0])
        pW_v = int(attrs.padding[1])

        _N, _C = N_v, C_v
        _H_in, _W_in, _H_out, _W_out = H_in_v, W_in_v, H_out_v, W_out_v
        _kH, _kW, _sH, _sW, _pH, _pW = kH_v, kW_v, sH_v, sW_v, pH_v, pW_v
        kernel_name = "c7x_int8_max_pool_tidl" if self.use_tidl_maxpool else "c7x_int8_max_pool"

        def te_max_pool(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    kernel_name,
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", _N),
                    tir.IntImm("int32", _C),
                    tir.IntImm("int32", _H_in),
                    tir.IntImm("int32", _W_in),
                    tir.IntImm("int32", _H_out),
                    tir.IntImm("int32", _W_out),
                    tir.IntImm("int32", _kH),
                    tir.IntImm("int32", _kW),
                    tir.IntImm("int32", _sH),
                    tir.IntImm("int32", _sW),
                    tir.IntImm("int32", _pH),
                    tir.IntImm("int32", _pW),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="tidl_max_pool_out",
                dtype="int8",
            )

        result = self.builder_.call_te(te_max_pool, x_arg, primfunc_name_hint=kernel_name)
        self.count += 1
        logger.debug(
            "Fused %s: N=%d C=%d H_in=%d W_in=%d "
            "kH=%d kW=%d sH=%d sW=%d pH=%d pW=%d",
            kernel_name,
            N_v,
            C_v,
            H_in_v,
            W_in_v,
            kH_v,
            kW_v,
            sH_v,
            sW_v,
            pH_v,
            pW_v,
        )
        return propagate_span(result, find_composite_span(func))


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToTIDLMaxPool")
class FuseQDQToTIDLMaxPool:
    """Fuse QDQ-wrapped max_pool2d into a c7x max-pool C kernel call.

    Emits c7x_int8_max_pool_tidl by default, or the scalar c7x_int8_max_pool
    fallback when use_tidl_maxpool is False (see Parameters).

    Must run before EliminateQDQTransparent: that pass removes the Q/DQ
    context that this pass needs to identify and match the pattern.

    Parameters
    ----------
    use_tidl_maxpool : bool
        Whether the firmware has the TIDL-backed max pool kernel available.
        When False (no-TIDL builds), falls back to the native scalar
        c7x_int8_max_pool kernel instead.
    """

    def __init__(self, use_tidl_maxpool: bool = True):
        self.use_tidl_maxpool = use_tidl_maxpool

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat, annotations = _make_max_pool_pattern()
        patterns = [(_COMPOSITE_NAME, pat, annotations, _check_maxpool)]

        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _MaxPoolLowerer(mod, self.use_tidl_maxpool)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseQDQToTIDLMaxPool: fused %d max_pool2d ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
