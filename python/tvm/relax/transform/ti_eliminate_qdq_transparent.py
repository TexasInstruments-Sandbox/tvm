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
"""Eliminate redundant QDQ around quantization-transparent ops.

PT2E quantization wraps non-quantizable ops in dequantize/quantize pairs
even when the scales are identical (the op doesn't change quantization
semantics). This pass removes the float conversion for such ops,
letting them operate directly on int8 data.

Handles ops where `dequant(x, S, Z) -> op -> quant(out, S, Z)` with
identical scales/zps can be replaced by `op(x_int8)`:

  - max_pool2d: max is monotonic
  - reshape, flatten, permute_dims: pure shape ops
  - relu: max(0, x) monotonic (only when zp=0, symmetric quant)
"""

import logging

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

_SCALE_RTOL = 1e-5


def _scales_match(s1, s2) -> bool:
    if s1 == 0 and s2 == 0:
        return True
    return abs(s1 - s2) / max(abs(s1), abs(s2)) < _SCALE_RTOL


# =========================================================================
# Pattern definitions
# =========================================================================


def _make_qdq_pattern(op_name, extra_args=0):
    """Create a dequant → op → quant pattern.

    extra_args: number of additional wildcard args after dq (e.g., 1 for reshape's shape arg).
    """
    x = wildcard()
    s_in = wildcard()
    zp_in = wildcard()
    dq = is_op("relax.dequantize")(x, s_in, zp_in)
    if extra_args == 0:
        op_out = is_op(op_name)(dq)
    elif extra_args == 1:
        op_out = is_op(op_name)(dq, wildcard())
    else:
        args = [dq] + [wildcard() for _ in range(extra_args)]
        op_out = is_op(op_name)(*args)
    s_out = wildcard()
    zp_out = wildcard()
    quant = is_op("relax.quantize")(op_out, s_out, zp_out)
    annotations = {
        "x": x,
        "s_in": s_in,
        "zp_in": zp_in,
        "s_out": s_out,
        "zp_out": zp_out,
    }
    return quant, annotations


def _make_qdq_concat_pattern():
    """Create dequant(...) + dequant(...) → concat → quant pattern."""
    # Use wildcard for the tuple arg (captures all dequantized inputs)
    pat_tuple = wildcard()
    concat = is_op("relax.concat")(pat_tuple)
    s_out = wildcard()
    zp_out = wildcard()
    quant = is_op("relax.quantize")(concat, s_out, zp_out)
    annotations = {"s_out": s_out, "zp_out": zp_out}
    return quant, annotations


def _check_scales_match(ctx) -> bool:
    """Validate that input and output quantization scales match."""
    s_in = ctx.annotated_expr["s_in"]
    s_out = ctx.annotated_expr["s_out"]
    zp_in = ctx.annotated_expr["zp_in"]
    zp_out = ctx.annotated_expr["zp_out"]

    if not isinstance(s_in, relax.Constant) or not isinstance(s_out, relax.Constant):
        return False
    if not isinstance(zp_in, relax.Constant) or not isinstance(zp_out, relax.Constant):
        return False

    sv_in = float(s_in.data.numpy())
    sv_out = float(s_out.data.numpy())
    zv_in = int(zp_in.data.numpy())
    zv_out = int(zp_out.data.numpy())

    return _scales_match(sv_in, sv_out) and zv_in == zv_out


def _check_scales_match_relu(ctx) -> bool:
    """Validate scales match AND zp=0 (required for relu transparency)."""
    if not _check_scales_match(ctx):
        return False
    zp_out = ctx.annotated_expr["zp_out"]
    return int(zp_out.data.numpy()) == 0


def _check_input_int8(ctx) -> bool:
    """Validate input is int8 and scales match."""
    if not _check_scales_match(ctx):
        return False
    x = ctx.annotated_expr["x"]
    if isinstance(x, relax.Constant):
        return x.data.dtype == "int8"
    if hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        return str(x.struct_info.dtype) == "int8"
    return False


def _check_input_int8_relu(ctx) -> bool:
    """Validate input is int8, scales match, and zp=0 for relu."""
    if not _check_scales_match_relu(ctx):
        return False
    x = ctx.annotated_expr["x"]
    if isinstance(x, relax.Constant):
        return x.data.dtype == "int8"
    if hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        return str(x.struct_info.dtype) == "int8"
    return False


def _check_concat(ctx) -> bool:
    """Validate concat output scale is constant (input scales checked in lowerer)."""
    s_out = ctx.annotated_expr["s_out"]
    zp_out = ctx.annotated_expr["zp_out"]
    return isinstance(s_out, relax.Constant) and isinstance(zp_out, relax.Constant)


# =========================================================================
# Composite lowering
# =========================================================================

_COMPOSITE_PREFIX = "qdq_transparent."


@mutator
class _QDQTransparentLowerer(PyExprMutator):
    """Replace QDQ-transparent composites with direct int8 ops."""

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
        if not name.startswith(_COMPOSITE_PREFIX):
            return super().visit_call_(call)

        return self._lower(call, func)

    def _lower(self, call, func):
        """Replace composite with the int8 op applied to original input."""
        param_to_arg = dict(zip(func.params, call.args))

        # Collect all dequantize inputs (int8 tensors) and the main op
        int8_inputs = []  # ordered list of int8 inputs from dequantize ops
        op_call = None
        # Map: binding var -> its resolved value (for tracing tuple args)
        var_to_val = {}

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            var_to_val[binding.var] = val
            if not isinstance(val, relax.Call):
                continue
            op_name = str(val.op.name) if hasattr(val.op, "name") else ""
            if "dequantize" in op_name:
                x_param = val.args[0]
                int8_inputs.append(param_to_arg.get(x_param, x_param))
            elif "quantize" not in op_name:
                op_call = val

        if not int8_inputs or op_call is None:
            return super().visit_call_(call)

        # Determine op type and rebuild args
        op_name = str(op_call.op.name) if hasattr(op_call.op, "name") else ""

        if "concat" in op_name:
            # Concat: replace tuple of dequantized inputs with tuple of int8
            new_tuple = relax.Tuple(int8_inputs)
            new_args = [new_tuple] + [
                param_to_arg.get(a, a) for a in op_call.args[1:]
            ]
        else:
            # Single-input ops: first int8 input + remaining args
            new_args = [int8_inputs[0]] + [
                param_to_arg.get(a, a) for a in op_call.args[1:]
            ]

        result = relax.Call(
            op_call.op, new_args, op_call.attrs, op_call.sinfo_args, span=op_call.span
        )
        result = self.builder_.emit(result)

        self.count += 1
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="EliminateQDQTransparent")
class EliminateQDQTransparent:
    """Eliminate redundant dequantize/quantize around transparent ops.

    For ops where int8 computation gives identical results to the
    float32 roundtrip (max_pool, reshape, relu with zp=0),
    removes the dequantize/quantize wrapper so the op runs on int8
    directly.  This reduces memory bandwidth by 4x and eliminates
    type conversion overhead.

    Applicable to any quantized model from PT2E (not MMALIB-specific).
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Phase 1: pattern-match into composites
        _P = _COMPOSITE_PREFIX
        _C = _check_input_int8
        patterns = [
            (_P + "max_pool2d", *_make_qdq_pattern("relax.nn.max_pool2d"), _C),
            (_P + "reshape", *_make_qdq_pattern("relax.reshape", extra_args=1), _C),
            (_P + "permute_dims", *_make_qdq_pattern("relax.permute_dims"), _C),
            (_P + "flatten", *_make_qdq_pattern("relax.flatten"), _C),
            (_P + "relu", *_make_qdq_pattern("relax.nn.relu"), _check_input_int8_relu),
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: lower composites to direct int8 ops
        lowerer = _QDQTransparentLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info(
                "EliminateQDQTransparent: eliminated %d QDQ wrappers",
                lowerer.count,
            )
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
