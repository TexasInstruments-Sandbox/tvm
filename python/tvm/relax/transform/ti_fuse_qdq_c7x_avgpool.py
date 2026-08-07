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
"""Fuse QDQ-wrapped average pooling ops into int8 C kernel calls.

Handles three Relax sub-graph patterns produced by C7xMMAQuantizer for
adaptive_avg_pool2d and avg_pool2d:

  Global avg pool (adaptive output (1,1)):
    dq(x_i8) → R.mean(axis=[-1,-2], keepdims=True) → q(out)
    → call_extern c7x_int8_global_avg_pool(in, out, N, C, H, W, zx, sx, zy, sy)

  Adaptive avg pool (output != (1,1)):
    dq(x_i8) → R.nn.adaptive_avg_pool2d → q(out)
    → call_extern c7x_int8_avg_pool(in, out, N, C, H_in, W_in, H_out, W_out,
                                    kH, kW, sH, sW, pH, pW, zx, sx, zy, sy)

  Spatial avg pool:
    dq(x_i8) → R.nn.avg_pool2d → q(out)
    → call_extern c7x_int8_avg_pool(...)

Kernels: src/runtime/ti_dsp/kernels/c7x_avgpool.cpp

Neither kernel calls into the TIDL algo library (see Step 12 in
docs/dsp/quantized_model_optimization.md for why the equivalent
TIDL_spatialAvgPool_ixX_oxX_* API isn't usable here), hence composite names
and the pass itself use the `c7x_`/`C7x` prefix rather than `tidl_`.
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

_COMPOSITE_PREFIX = "c7x_pool."


# =========================================================================
# Pattern definitions
# =========================================================================


def _make_global_avg_pool_pattern():
    """dq → R.mean(axis=[-1,-2], keepdims=True) → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    mean_out = is_op("relax.mean")(dq)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(mean_out, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _make_adaptive_avg_pool_pattern():
    """dq → R.nn.adaptive_avg_pool2d → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    pool = is_op("relax.nn.adaptive_avg_pool2d")(dq)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(pool, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _make_avg_pool_pattern():
    """dq → R.nn.avg_pool2d → q"""
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    pool = is_op("relax.nn.avg_pool2d")(dq)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(pool, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


_PATTERN_REGISTRY = [
    ("c7x_pool.global_avg", _make_global_avg_pool_pattern),
    ("c7x_pool.adaptive_avg", _make_adaptive_avg_pool_pattern),
    ("c7x_pool.avg_pool2d", _make_avg_pool_pattern),
]

_COMPOSITE_NAMES = frozenset(name for name, _ in _PATTERN_REGISTRY)


def _check_pool(ctx) -> bool:
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
class _AvgPoolLowerer(PyExprMutator):
    """Lower C7x avg pool composites to call_extern."""

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

        x_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None
        pool_call = None  # the pool op binding value
        x_sinfo = None  # input tensor struct_info (for shape)

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
                # Resolve the input shape from the dequant output struct_info
                if hasattr(binding.var, "struct_info"):
                    x_sinfo = binding.var.struct_info
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())
            elif any(k in op_name for k in ("avg_pool", "mean")):
                pool_call = val

        if x_arg is None or o_scale_val is None or pool_call is None:
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)
        out_shape = [int(s) for s in call_sinfo.shape]

        # Decode input shape (NCHW)
        if x_sinfo is None or not isinstance(x_sinfo, relax.TensorStructInfo) or not x_sinfo.shape:
            return super().visit_call_(call)
        in_shape = [int(s) for s in x_sinfo.shape]
        if len(in_shape) != 4 or len(out_shape) != 4:
            return super().visit_call_(call)

        N_v, C_v, H_in_v, W_in_v = in_shape
        _, _, H_out_v, W_out_v = out_shape

        composite_span = find_composite_span(func)
        if composite_name == "c7x_pool.global_avg":
            return propagate_span(
                self._lower_global(
                    x_arg,
                    out_shape,
                    N_v,
                    C_v,
                    H_in_v,
                    W_in_v,
                    d_zp_val,
                    d_scale_val,
                    o_zp_val,
                    o_scale_val,
                ),
                composite_span,
            )
        else:
            return propagate_span(
                self._lower_spatial(
                    x_arg,
                    out_shape,
                    pool_call,
                    N_v,
                    C_v,
                    H_in_v,
                    W_in_v,
                    H_out_v,
                    W_out_v,
                    composite_name,
                    d_zp_val,
                    d_scale_val,
                    o_zp_val,
                    o_scale_val,
                ),
                composite_span,
            )

    def _lower_global(self, x_arg, out_shape, N, C, H, W, d_zp, d_scale, o_zp, o_scale):
        def te_global_avg_pool(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_global_avg_pool",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", N),
                    tir.IntImm("int32", C),
                    tir.IntImm("int32", H),
                    tir.IntImm("int32", W),
                    tir.IntImm("int32", d_zp),
                    tir.FloatImm("float32", d_scale),
                    tir.IntImm("int32", o_zp),
                    tir.FloatImm("float32", o_scale),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="c7x_pool_out",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_global_avg_pool, x_arg, primfunc_name_hint="c7x_int8_global_avg_pool"
        )
        self.count += 1
        logger.debug("Fused c7x_int8_global_avg_pool: N=%d C=%d H=%d W=%d", N, C, H, W)
        return result

    def _lower_spatial(
        self,
        x_arg,
        out_shape,
        pool_call,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        composite_name,
        d_zp,
        d_scale,
        o_zp,
        o_scale,
    ):
        # Extract kernel_size, stride, padding
        if composite_name == "c7x_pool.avg_pool2d":
            attrs = pool_call.attrs
            kH_v = int(attrs.pool_size[0])
            kW_v = int(attrs.pool_size[1])
            sH_v = int(attrs.strides[0])
            sW_v = int(attrs.strides[1])
            pH_v = int(attrs.padding[0])
            pW_v = int(attrs.padding[1])
        else:
            # adaptive_avg_pool2d: compute kernel from input/output ratio
            kH_v = H_in // H_out
            kW_v = W_in // W_out
            sH_v = kH_v
            sW_v = kW_v
            pH_v = 0
            pW_v = 0

        def te_avg_pool(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_avg_pool",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", N),
                    tir.IntImm("int32", C),
                    tir.IntImm("int32", H_in),
                    tir.IntImm("int32", W_in),
                    tir.IntImm("int32", H_out),
                    tir.IntImm("int32", W_out),
                    tir.IntImm("int32", kH_v),
                    tir.IntImm("int32", kW_v),
                    tir.IntImm("int32", sH_v),
                    tir.IntImm("int32", sW_v),
                    tir.IntImm("int32", pH_v),
                    tir.IntImm("int32", pW_v),
                    tir.IntImm("int32", d_zp),
                    tir.FloatImm("float32", d_scale),
                    tir.IntImm("int32", o_zp),
                    tir.FloatImm("float32", o_scale),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="c7x_pool_out",
                dtype="int8",
            )

        result = self.builder_.call_te(te_avg_pool, x_arg, primfunc_name_hint="c7x_int8_avg_pool")
        self.count += 1
        logger.debug(
            "Fused c7x_int8_avg_pool: N=%d C=%d kH=%d kW=%d sH=%d sW=%d",
            N,
            C,
            kH_v,
            kW_v,
            sH_v,
            sW_v,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xAvgPool")
class FuseQDQToC7xAvgPool:
    """Fuse QDQ-wrapped average pooling into c7x_int8_*_avg_pool C kernel calls.

    Handles adaptive_avg_pool2d (global and non-global) and avg_pool2d.
    Requires int8 input and compile-time quantization constants.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = []
        for composite_name, factory in _PATTERN_REGISTRY:
            pat, annotations = factory()
            patterns.append((composite_name, pat, annotations, _check_pool))

        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _AvgPoolLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseQDQToC7xAvgPool: fused %d pool ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
