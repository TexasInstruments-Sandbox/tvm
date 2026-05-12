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
"""MMALIB QDQ fusion: partition int8 quantized linear (FC) for TI C7x MMA.

Matches the PT2E QDQ pattern for fully-quantized linear layers:

    dequantize(data_int8, d_scale, d_zp)
      -> matmul(float, permute_dims(dequantize(weight_int8, w_scale, w_zp)))
      -> [add(float_bias)]
      -> quantize(out, o_scale, o_zp)

Replaces with a single call_extern("mmalib_matmul_bias_i8") that computes
int8 × int8 → int32 matmul with fused bias, per-channel requantization via
MMALIB_LINALG_matrixMatrixMultiplyBias (bTranspose=1, no weight reorder).
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_mmalib_constants import MMA_SIZE_I8
from .ti_mmalib_legalize import _float_to_scale_shift

logger = logging.getLogger(__name__)


# =========================================================================
# Pattern definitions
# =========================================================================


def _qdq_fc_bias_pattern():
    """dequant(data) -> matmul(_, permute_dims(dequant(w))) -> add(bias) -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)
    w_perm = is_op("relax.permute_dims")(w_dq)

    mm = is_op("relax.matmul")(data_dq, w_perm)
    bias = wildcard()
    add_out = is_op("relax.add")(mm, bias)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(add_out, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "w_perm": w_perm,
        "mm": mm,
        "bias": bias,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_fc


def _qdq_fc_pattern():
    """dequant(data) -> matmul(_, permute_dims(dequant(w))) -> quantize"""
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)
    w_perm = is_op("relax.permute_dims")(w_dq)

    mm = is_op("relax.matmul")(data_dq, w_perm)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(mm, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "w_perm": w_perm,
        "mm": mm,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_fc


def _qdq_fc_reshape_bias_pattern():
    """dequant(data) -> reshape -> matmul(...) -> reshape -> add(bias) -> quantize

    Handles 3D inputs where aten.linear decomposes to reshape+matmul+reshape.
    """
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)
    data_rs = is_op("relax.reshape")(data_dq, wildcard())

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)
    w_perm = is_op("relax.permute_dims")(w_dq)

    mm = is_op("relax.matmul")(data_rs, w_perm)
    mm_rs = is_op("relax.reshape")(mm, wildcard())
    bias = wildcard()
    add_out = is_op("relax.add")(mm_rs, bias)

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(add_out, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "w_perm": w_perm,
        "mm": mm,
        "bias": bias,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_fc


def _qdq_fc_reshape_pattern():
    """dequant(data) -> reshape -> matmul(...) -> reshape -> quantize

    Handles 3D inputs where aten.linear decomposes to reshape+matmul+reshape.
    """
    data = wildcard()
    d_scale = wildcard()
    d_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, d_scale, d_zp)
    data_rs = is_op("relax.reshape")(data_dq, wildcard())

    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)
    w_perm = is_op("relax.permute_dims")(w_dq)

    mm = is_op("relax.matmul")(data_rs, w_perm)
    mm_rs = is_op("relax.reshape")(mm, wildcard())

    o_scale = wildcard()
    o_zp = wildcard()
    quant_out = is_op("relax.quantize")(mm_rs, o_scale, o_zp)

    annotations = {
        "data": data,
        "d_scale": d_scale,
        "d_zp": d_zp,
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "w_perm": w_perm,
        "mm": mm,
        "o_scale": o_scale,
        "o_zp": o_zp,
    }
    return quant_out, annotations, _check_mmalib_qdq_fc


# =========================================================================
# Check function
# =========================================================================


def _is_const_zero(expr) -> bool:
    if isinstance(expr, relax.Constant):
        return np.all(expr.data.numpy() == 0)
    return False


def _check_mmalib_qdq_fc(ctx) -> bool:
    """Validate MMALIB matmul_bias eligibility for a QDQ FC pattern."""
    w_zp = ctx.annotated_expr["w_zp"]
    if not _is_const_zero(w_zp):
        return False

    w = ctx.annotated_expr["w_int8"]
    if isinstance(w, relax.Constant):
        if w.data.dtype != "int8":
            return False
    elif hasattr(w, "struct_info") and hasattr(w.struct_info, "dtype"):
        if str(w.struct_info.dtype) != "int8":
            return False
    else:
        return False

    data = ctx.annotated_expr["data"]
    if isinstance(data, relax.Constant):
        return False
    if hasattr(data, "struct_info") and hasattr(data.struct_info, "dtype"):
        if str(data.struct_info.dtype) != "int8":
            return False

    # Weight shape [N_out, K] — both K and N must be multiples of 64
    if isinstance(w, relax.Constant):
        w_shape = w.data.shape
    elif hasattr(w, "struct_info") and w.struct_info.shape is not None:
        w_shape = [int(s) for s in w.struct_info.shape]
    else:
        return False

    if len(w_shape) != 2:
        return False
    n_out, k = w_shape[0], w_shape[1]
    if k % MMA_SIZE_I8 != 0:
        return False
    if n_out % MMA_SIZE_I8 != 0:
        return False

    # permute_dims must be a standard transpose (last two dims)
    w_perm = ctx.annotated_expr["w_perm"]
    if isinstance(w_perm, relax.Call):
        if w_perm.attrs.axes is not None:
            expected = list(range(len(w_shape)))
            expected[-1], expected[-2] = expected[-2], expected[-1]
            if list(w_perm.attrs.axes) != expected:
                return False

    # Bias (if present) must be a Constant
    if "bias" in ctx.annotated_expr:
        b = ctx.annotated_expr["bias"]
        if not isinstance(b, relax.Constant):
            return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================

_COMPOSITE_PREFIXES = (
    "mmalib.fc_i8_qdq_reshape_bias",
    "mmalib.fc_i8_qdq_reshape",
    "mmalib.fc_i8_qdq_bias",
    "mmalib.fc_i8_qdq",
)


@mutator
class _MMALIBQDQFCLowerer(PyExprMutator):
    """Replace MMALIB QDQ FC composite functions with call_tir."""

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

        has_bias = "bias" in name
        return self._lower(call, func, has_bias=has_bias)

    def _lower(self, call, func, has_bias):
        param_to_arg = dict(zip(func.params, call.args))
        roles = self._extract_roles(func, has_bias)

        required = ["data", "w_int8", "w_scale", "d_scale", "o_scale"]
        if any(r not in roles for r in required):
            logger.warning("Could not identify roles in MMALIB FC composite")
            return super().visit_call_(call)

        data_arg = param_to_arg[roles["data"]]

        w_int8_arg = param_to_arg[roles["w_int8"]]
        w_int8_np = w_int8_arg.data.numpy() if isinstance(w_int8_arg, relax.Constant) else None
        if w_int8_np is None:
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

        d_zp_val = 0
        if "d_zp" in roles:
            d_zp_arg = param_to_arg[roles["d_zp"]]
            if isinstance(d_zp_arg, relax.Constant):
                d_zp_val = int(d_zp_arg.data.numpy())

        o_zp_val = 0
        if "o_zp" in roles:
            o_zp_arg = param_to_arg[roles["o_zp"]]
            if isinstance(o_zp_arg, relax.Constant):
                o_zp_val = int(o_zp_arg.data.numpy())

        bias_np = None
        if has_bias and "bias" in roles:
            bias_arg = param_to_arg[roles["bias"]]
            if isinstance(bias_arg, relax.Constant):
                bias_np = bias_arg.data.numpy().flatten()

        # Weight shape: [N_out, K]
        N_out, K = w_int8_np.shape

        # Data shape: [..., K] -> M = product of leading dims
        data_sinfo = data_arg.struct_info
        data_shape = [int(s) for s in data_sinfo.shape]
        M = 1
        for d in data_shape[:-1]:
            M *= d

        # --- Compute MMALIB parameters ---

        # Per-channel weight sum for zero-point correction
        weight_sum = w_int8_np.astype(np.int32).sum(axis=1)
        zp_correction = (np.int32(-d_zp_val) * weight_sum).astype(np.int32)

        # Bias in accumulator scale
        if bias_np is not None:
            dw_scale = d_scale_val * w_scale_np[:N_out]
            bias_accum = np.round(bias_np[:N_out] / dw_scale).astype(np.int32)
        else:
            bias_accum = np.zeros(N_out, dtype=np.int32)

        bias_i32 = (bias_accum + zp_correction).astype(np.int32)

        if o_zp_val != 0:
            combined_rescale_for_ozp = d_scale_val * w_scale_np[:N_out] / o_scale_val
            bias_i32 = (bias_i32 + np.round(o_zp_val / combined_rescale_for_ozp)).astype(np.int32)

        # Requantization scale
        combined_rescale = d_scale_val * w_scale_np[:N_out] / o_scale_val
        scale_u8, shift_u8 = _float_to_scale_shift(combined_rescale)

        # Build relax constants (weight passed as-is, no reorder needed)
        weight_relax = relax.Constant(w_int8_np)
        bias_relax = relax.Constant(bias_i32)
        scale_relax = relax.Constant(scale_u8)
        shift_relax = relax.Constant(shift_u8)

        def te_mmalib_fc_i8(
            data_t: te.Tensor,
            weight_t: te.Tensor,
            bias_t: te.Tensor,
            scale_t: te.Tensor,
            shift_t: te.Tensor,
        ) -> te.Tensor:
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "mmalib_matmul_bias_i8",
                    ins[0].data,
                    ins[1].data,
                    ins[2].data,
                    ins[3].data,
                    ins[4].data,
                    outs[0].data,
                    M,
                    K,
                    N_out,
                )

            return te.extern(
                data_shape[:-1] + [N_out],
                [data_t, weight_t, bias_t, scale_t, shift_t],
                fcompute,
                name="mmalib_fc",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_mmalib_fc_i8,
            data_arg,
            weight_relax,
            bias_relax,
            scale_relax,
            shift_relax,
            primfunc_name_hint="mmalib_fc",
        )

        self.count += 1
        logger.info(
            "MMALIB QDQ FC fusion #%d: %dx%d%s",
            self.count,
            K,
            N_out,
            " +bias" if has_bias else "",
        )
        return result

    @staticmethod
    def _extract_roles(func, has_bias):
        """Walk composite body to map function params to their roles."""
        roles = {}

        # Find matmul to identify data and weight dequantize vars
        matmul_data_var = None
        matmul_weight_var = None  # This is the permute_dims output
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if not hasattr(val.op, "name"):
                    continue
                if val.op.name == "relax.matmul":
                    matmul_data_var = val.args[0]
                    matmul_weight_var = val.args[1]
                    break

        if matmul_data_var is None:
            return roles

        # Trace matmul data input back through reshape (for 3D patterns)
        data_dq_var = matmul_data_var
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                if not binding.var.same_as(matmul_data_var):
                    continue
                val = binding.value
                if isinstance(val, relax.Call) and hasattr(val.op, "name"):
                    if val.op.name == "relax.reshape":
                        data_dq_var = val.args[0]
                break

        # Find permute_dims to get the dequantize var for weights
        weight_dq_var = None
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if not hasattr(val.op, "name"):
                    continue
                if val.op.name == "relax.permute_dims":
                    if binding.var.same_as(matmul_weight_var):
                        weight_dq_var = val.args[0]
                        break

        if weight_dq_var is None:
            return roles

        # Trace dequantize bindings
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if not hasattr(val.op, "name"):
                    continue

                op_name = val.op.name
                if op_name == "relax.dequantize":
                    if binding.var.same_as(data_dq_var):
                        roles["data"] = val.args[0]
                        roles["d_scale"] = val.args[1]
                        roles["d_zp"] = val.args[2]
                    elif binding.var.same_as(weight_dq_var):
                        roles["w_int8"] = val.args[0]
                        roles["w_scale"] = val.args[1]
                        roles["w_zp"] = val.args[2]
                elif has_bias and op_name == "relax.add":
                    # Identify bias as the add operand that is a function param
                    for arg in (val.args[1], val.args[0]):
                        for p in func.params:
                            if arg.same_as(p):
                                roles["bias"] = p
                                break
                        if "bias" in roles:
                            break
                    else:
                        roles["bias"] = val.args[1]
                elif op_name == "relax.quantize":
                    roles["o_scale"] = val.args[1]
                    roles["o_zp"] = val.args[2]
        return roles


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseMMALIBQDQFC")
class FuseMMALIBQDQFC:
    """Fuse PT2E QDQ linear/FC patterns into MMALIB matmul_bias calls.

    Uses MMALIB_LINALG_matrixMatrixMultiplyBias with bTranspose=1 which
    accepts weights in natural [N_out, K] layout without reordering.
    Non-eligible ops (wrong alignment, non-constant weight) fall through
    to subsequent passes.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        patterns = [
            ("mmalib.fc_i8_qdq_reshape_bias", *_qdq_fc_reshape_bias_pattern()),
            ("mmalib.fc_i8_qdq_reshape", *_qdq_fc_reshape_pattern()),
            ("mmalib.fc_i8_qdq_bias", *_qdq_fc_bias_pattern()),
            ("mmalib.fc_i8_qdq", *_qdq_fc_pattern()),
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _MMALIBQDQFCLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseMMALIBQDQFC: fused %d layers", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
