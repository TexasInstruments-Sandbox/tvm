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
"""Integer residual add fusion for quantized models.

Matches the PT2E pattern for residual (skip) connections:

    dequantize(x, scale_x, zp_x) + dequantize(skip, scale_skip, zp_skip)
      -> [relu] -> quantize(out, scale_out, zp_out)

Replaces with a single call_extern("tvm_int8_residual_add_relu") that
computes the equivalent in integer arithmetic:

    out[i] = sat_i8(max(0, (x[i]-zp_x)*M_x + (skip[i]-zp_skip)*M_skip) >> shift + zp_out)

Where M_x, M_skip, shift are compile-time constants derived from the
quantization scales.  This eliminates float32 intermediate computation
and reduces per-layer cost from ~5-11M cycles to ~100-500K cycles on C7x.
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

_SHIFT_BITS = 15


# =========================================================================
# Pattern definitions
# =========================================================================


def _residual_add_relu_pattern():
    """dequant(x) + dequant(skip) -> relu -> quantize"""
    x = wildcard()
    x_scale = wildcard()
    x_zp = wildcard()
    x_dq = is_op("relax.dequantize")(x, x_scale, x_zp)

    skip = wildcard()
    skip_scale = wildcard()
    skip_zp = wildcard()
    skip_dq = is_op("relax.dequantize")(skip, skip_scale, skip_zp)

    add_out = is_op("relax.add")(x_dq, skip_dq)
    relu_out = is_op("relax.nn.relu")(add_out)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(relu_out, o_scale, o_zp)

    annotations = {
        "x": x,
        "x_scale": x_scale,
        "x_zp": x_zp,
        "skip": skip,
        "skip_scale": skip_scale,
        "skip_zp": skip_zp,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_residual_add


def _residual_add_pattern():
    """dequant(x) + dequant(skip) -> quantize (no relu)"""
    x = wildcard()
    x_scale = wildcard()
    x_zp = wildcard()
    x_dq = is_op("relax.dequantize")(x, x_scale, x_zp)

    skip = wildcard()
    skip_scale = wildcard()
    skip_zp = wildcard()
    skip_dq = is_op("relax.dequantize")(skip, skip_scale, skip_zp)

    add_out = is_op("relax.add")(x_dq, skip_dq)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(add_out, o_scale, o_zp)

    annotations = {
        "x": x,
        "x_scale": x_scale,
        "x_zp": x_zp,
        "skip": skip,
        "skip_scale": skip_scale,
        "skip_zp": skip_zp,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_residual_add


# =========================================================================
# Check function
# =========================================================================


def _check_residual_add(ctx) -> bool:
    """Validate integer residual add eligibility."""
    x = ctx.annotated_expr["x"]
    skip = ctx.annotated_expr["skip"]

    # Both inputs must be int8
    for tensor in (x, skip):
        if isinstance(tensor, relax.Constant):
            if tensor.data.dtype != "int8":
                return False
        elif hasattr(tensor, "struct_info") and hasattr(tensor.struct_info, "dtype"):
            if str(tensor.struct_info.dtype) != "int8":
                return False
        else:
            return False

    # Scales must be constants (for compile-time computation)
    for name in ("x_scale", "skip_scale", "o_scale"):
        val = ctx.annotated_expr[name]
        if not isinstance(val, relax.Constant):
            return False

    # Zero points must be constants
    for name in ("x_zp", "skip_zp", "o_zp"):
        val = ctx.annotated_expr[name]
        if not isinstance(val, relax.Constant):
            return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================

_COMPOSITE_PREFIXES = (
    "int8_residual.add_relu",
    "int8_residual.add",
)


@mutator
class _ResidualAddLowerer(PyExprMutator):
    """Replace int8 residual add composite functions with call_tir."""

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

        has_relu = "relu" in name
        return self._lower(call, func, has_relu=has_relu)

    def _lower(self, call, func, has_relu):
        param_to_arg = dict(zip(func.params, call.args))

        # Walk the composite function to find dequantize/add/quantize
        x_arg, x_scale_val, x_zp_val = None, None, None
        skip_arg, skip_scale_val, skip_zp_val = None, None, None
        o_scale_val, o_zp_val = None, None
        output_shape = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if isinstance(val, relax.Call):
                op_name = str(val.op) if hasattr(val.op, "name") else ""
                if "dequantize" in op_name:
                    data_param = val.args[0]
                    scale_param = val.args[1]
                    zp_param = val.args[2]
                    data_resolved = param_to_arg.get(data_param, data_param)
                    scale_resolved = param_to_arg.get(scale_param, scale_param)
                    zp_resolved = param_to_arg.get(zp_param, zp_param)
                    if x_arg is None:
                        x_arg = data_resolved
                        x_scale_val = float(scale_resolved.data.numpy())
                        x_zp_val = int(zp_resolved.data.numpy())
                    else:
                        skip_arg = data_resolved
                        skip_scale_val = float(scale_resolved.data.numpy())
                        skip_zp_val = int(zp_resolved.data.numpy())
                elif "quantize" in op_name:
                    scale_param = val.args[1]
                    zp_param = val.args[2]
                    o_scale_val = float(param_to_arg.get(scale_param, scale_param).data.numpy())
                    o_zp_val = int(param_to_arg.get(zp_param, zp_param).data.numpy())

        if x_arg is None or skip_arg is None or o_scale_val is None:
            return super().visit_call_(call)

        # Get output shape from the quantize output struct_info
        call_sinfo = call.struct_info
        if isinstance(call_sinfo, relax.TensorStructInfo) and call_sinfo.shape:
            output_shape = [int(s) for s in call_sinfo.shape]
        else:
            return super().visit_call_(call)

        num_elements = 1
        for s in output_shape:
            num_elements *= s

        # Compute integer multipliers
        M_x = int(round(x_scale_val / o_scale_val * (1 << _SHIFT_BITS)))
        M_skip = int(round(skip_scale_val / o_scale_val * (1 << _SHIFT_BITS)))

        # Pack params: [M_x(i32), M_skip(i32), shift(i32), zp_x, zp_skip, zp_out]
        params = np.zeros(16, dtype=np.uint8)
        params[0:4] = np.array([M_x], dtype=np.int32).view(np.uint8)
        params[4:8] = np.array([M_skip], dtype=np.int32).view(np.uint8)
        params[8:12] = np.array([_SHIFT_BITS], dtype=np.int32).view(np.uint8)
        params[12] = np.uint8(np.int8(x_zp_val).view(np.uint8))
        params[13] = np.uint8(np.int8(skip_zp_val).view(np.uint8))
        params[14] = np.uint8(np.int8(o_zp_val).view(np.uint8))
        params_const = relax.Constant(params)

        has_relu_int = 1 if has_relu else 0
        n_elem = num_elements

        def te_int8_residual_add(x_t, skip_t, params_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "tvm_int8_residual_add_relu",
                    ins[0].data,
                    ins[1].data,
                    ins[2].data,
                    outs[0].data,
                    n_elem,
                    has_relu_int,
                )

            return te.extern(
                output_shape,
                [x_t, skip_t, params_t],
                fcompute,
                name="int8_residual_add",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_int8_residual_add,
            x_arg,
            skip_arg,
            params_const,
            primfunc_name_hint="int8_residual_add",
        )

        self.count += 1
        logger.debug(
            "Fused residual add: M_x=%d M_skip=%d shift=%d relu=%d elems=%d",
            M_x,
            M_skip,
            _SHIFT_BITS,
            has_relu,
            num_elements,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseInt8ResidualAdd")
class FuseInt8ResidualAdd:
    """Fuse quantized residual add patterns into integer-only operations.

    Matches dequantize(x) + dequantize(skip) -> [relu] -> quantize and
    replaces with a call_extern to tvm_int8_residual_add_relu that
    computes the equivalent using fixed-point arithmetic.

    Applicable to any quantized model with residual/skip connections
    (ResNet, MobileNetV2, EfficientNet, etc.).
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = [
            ("int8_residual.add_relu", *_residual_add_relu_pattern()),
            ("int8_residual.add", *_residual_add_pattern()),
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _ResidualAddLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseInt8ResidualAdd: fused %d residual adds", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
