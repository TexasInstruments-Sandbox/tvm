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
"""Integer residual add fusion for quantized models (int8 and int16).

Matches the PT2E pattern for residual (skip) connections:

    dequantize(x, scale_x, zp_x) + dequantize(skip, scale_skip, zp_skip)
      -> [relu] -> quantize(out, scale_out, zp_out)

Replaces with a single call_extern to a fixed-point integer kernel:
  - int8 inputs  → c7x_int8_residual_add_relu  → int8 output
  - int16 inputs → c7x_int16_residual_add_relu → int16 output

The kernel computes:
    out[i] = sat(((x[i]-zp_x)*M_x + (skip[i]-zp_skip)*M_skip) >> shift + zp_out)

Where M_x, M_skip, shift are compile-time constants derived from the
quantization scales.  This eliminates float32 intermediate computation
and reduces per-layer cost from ~5-11M cycles to ~100-500K cycles on C7x.

Both operand orders (add(x, skip) and add(skip, x)) are handled for each
dtype since TVM's DFPattern matcher is not commutative.
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


# -------------------------------------------------------------------------
# Why four patterns?
#
# TVM's DFPattern matching is NOT commutative — is_op("relax.add")(a, b)
# only matches add(a, b), not add(b, a).  PyTorch's torch.export produces
# residual connections in one of two operand orders depending on the model
# architecture and the position of the skip branch in the graph:
#
#   Order A:  add(conv_out, skip)  — conv result is first arg
#   Order B:  add(skip, conv_out)  — skip connection is first arg
#
# We need separate patterns for each order.  The lowering arithmetic is
# identical for both: addition is commutative, and each operand's scale
# and zero-point are tracked independently, so swapping the labels produces
# the same numerical result.
# -------------------------------------------------------------------------


def _residual_add_relu_pattern():
    """Order A: add(x_dq, skip_dq) → relu → quantize."""
    x = wildcard()
    x_scale = wildcard()
    x_zp = wildcard()
    x_dq = is_op("relax.dequantize")(x, x_scale, x_zp)

    skip = wildcard()
    skip_scale = wildcard()
    skip_zp = wildcard()
    skip_dq = is_op("relax.dequantize")(skip, skip_scale, skip_zp)

    # x_dq is the first argument to add (e.g. conv output)
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
    """Order A: add(x_dq, skip_dq) → quantize (no relu)."""
    x = wildcard()
    x_scale = wildcard()
    x_zp = wildcard()
    x_dq = is_op("relax.dequantize")(x, x_scale, x_zp)

    skip = wildcard()
    skip_scale = wildcard()
    skip_zp = wildcard()
    skip_dq = is_op("relax.dequantize")(skip, skip_scale, skip_zp)

    # x_dq is the first argument to add (e.g. conv output)
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


def _residual_add_relu_pattern_swapped():
    """Order B: add(skip_dq, x_dq) → relu → quantize (skip connection first).

    Some models (e.g. certain MobileNet variants) place the skip-connection
    tensor as the *first* argument of the add node.  TVM's pattern matcher
    does not treat add as commutative, so we need this explicit variant.

    The lowering logic is unchanged: per-operand scale/zp are tracked
    independently, so the integer arithmetic produces the same result
    regardless of which operand the labels "x" and "skip" are assigned to.
    """
    x = wildcard()
    x_scale = wildcard()
    x_zp = wildcard()
    x_dq = is_op("relax.dequantize")(x, x_scale, x_zp)

    skip = wildcard()
    skip_scale = wildcard()
    skip_zp = wildcard()
    skip_dq = is_op("relax.dequantize")(skip, skip_scale, skip_zp)

    # skip_dq is the first argument to add (skip connection comes first)
    add_out = is_op("relax.add")(skip_dq, x_dq)
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


def _residual_add_pattern_swapped():
    """Order B: add(skip_dq, x_dq) → quantize (skip connection first, no relu).

    See _residual_add_relu_pattern_swapped for the rationale.
    """
    x = wildcard()
    x_scale = wildcard()
    x_zp = wildcard()
    x_dq = is_op("relax.dequantize")(x, x_scale, x_zp)

    skip = wildcard()
    skip_scale = wildcard()
    skip_zp = wildcard()
    skip_dq = is_op("relax.dequantize")(skip, skip_scale, skip_zp)

    # skip_dq is the first argument to add (skip connection comes first)
    add_out = is_op("relax.add")(skip_dq, x_dq)

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


def _check_dtype(ctx, expected_dtype: str) -> bool:
    """Shared eligibility check for a given element dtype."""
    x = ctx.annotated_expr["x"]
    skip = ctx.annotated_expr["skip"]

    for tensor in (x, skip):
        if isinstance(tensor, relax.Constant):
            if tensor.data.dtype != expected_dtype:
                return False
        elif hasattr(tensor, "struct_info") and hasattr(tensor.struct_info, "dtype"):
            if str(tensor.struct_info.dtype) != expected_dtype:
                return False
        else:
            return False

    # Scales and zero-points must be compile-time constants
    for name in ("x_scale", "skip_scale", "o_scale", "x_zp", "skip_zp", "o_zp"):
        val = ctx.annotated_expr[name]
        if not isinstance(val, relax.Constant):
            return False

    return True


def _check_residual_add(ctx) -> bool:
    """Validate int8 residual add eligibility."""
    return _check_dtype(ctx, "int8")


def _check_residual_add_i16(ctx) -> bool:
    """Validate int16 residual add eligibility.

    Additional constraint vs int8: zero-points must all be 0.
    C7xMMAQuantizer(dtype="int16") enforces symmetric activation quantization
    (d_zp=0), so this constraint is always satisfied for correct int16 PT2E
    graphs but guards against misuse.
    """
    if not _check_dtype(ctx, "int16"):
        return False

    # Int16 activation quantization is symmetric only — all zp must be 0.
    for name in ("x_zp", "skip_zp", "o_zp"):
        zp = ctx.annotated_expr[name]
        if isinstance(zp, relax.Constant):
            if int(zp.data.numpy()) != 0:
                return False
        else:
            return False

    return True


# =========================================================================
# Int16 pattern factories
#
# Structurally identical to the int8 patterns; the only difference is the
# check function (_check_residual_add_i16 enforces int16 dtype and zp=0).
# We reuse the int8 pattern bodies and replace the check function.
# =========================================================================


def _residual_add_i16_relu_pattern():
    """Order A: add(x_dq, skip_dq) → relu → quantize [int16]."""
    pat, annotations, _ = _residual_add_relu_pattern()
    return pat, annotations, _check_residual_add_i16


def _residual_add_i16_relu_pattern_swapped():
    """Order B: add(skip_dq, x_dq) → relu → quantize [int16]."""
    pat, annotations, _ = _residual_add_relu_pattern_swapped()
    return pat, annotations, _check_residual_add_i16


def _residual_add_i16_pattern():
    """Order A: add(x_dq, skip_dq) → quantize [int16, no relu]."""
    pat, annotations, _ = _residual_add_pattern()
    return pat, annotations, _check_residual_add_i16


def _residual_add_i16_pattern_swapped():
    """Order B: add(skip_dq, x_dq) → quantize [int16, no relu]."""
    pat, annotations, _ = _residual_add_pattern_swapped()
    return pat, annotations, _check_residual_add_i16


# =========================================================================
# Pattern registry
# =========================================================================

# Single source of truth for all residual-add pattern variants.
# Each entry is (composite_name, pattern_factory).
# - transform_module builds the FuseOpsByPattern patterns list from this.
# - _ResidualAddLowerer looks up names in _COMPOSITE_NAMES to decide
#   whether to lower a composite function, and which call_extern to emit.
# Adding a new variant: append one entry here; nothing else needs updating.
#
# Name convention:
#   "int8_residual.*"  → emits c7x_int8_residual_add_relu  (int8 output)
#   "int16_residual.*" → emits c7x_int16_residual_add_relu (int16 output)
_PATTERN_REGISTRY = [
    # --- int8 variants (Phase 2a) ---
    ("int8_residual.add_relu", _residual_add_relu_pattern),
    ("int8_residual.add_relu_swapped", _residual_add_relu_pattern_swapped),
    ("int8_residual.add", _residual_add_pattern),
    ("int8_residual.add_swapped", _residual_add_pattern_swapped),
    # --- int16 variants (Phase 2c) ---
    ("int16_residual.add_relu", _residual_add_i16_relu_pattern),
    ("int16_residual.add_relu_swapped", _residual_add_i16_relu_pattern_swapped),
    ("int16_residual.add", _residual_add_i16_pattern),
    ("int16_residual.add_swapped", _residual_add_i16_pattern_swapped),
]

# Derived set of names for O(1) membership checks in the lowerer.
_COMPOSITE_NAMES = frozenset(name for name, _ in _PATTERN_REGISTRY)

# =========================================================================
# Composite lowering
# =========================================================================


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
        if name not in _COMPOSITE_NAMES:
            return super().visit_call_(call)

        has_relu = "relu" in name
        # Detect output dtype from the name prefix — int8 vs int16 kernels
        # use different call_extern targets and output dtypes.
        is_i16 = name.startswith("int16_residual.")
        return self._lower(call, func, has_relu=has_relu, is_i16=is_i16)

    def _lower(self, call, func, has_relu, is_i16=False):
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
        # Select kernel and output dtype based on operand dtype.
        extern_name = "c7x_int16_residual_add_relu" if is_i16 else "c7x_int8_residual_add_relu"
        out_dtype = "int16" if is_i16 else "int8"
        hint = "i16_residual_add" if is_i16 else "int8_residual_add"

        def te_residual_add(x_t, skip_t, params_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    extern_name,
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
                name=hint,
                dtype=out_dtype,
            )

        result = self.builder_.call_te(
            te_residual_add,
            x_arg,
            skip_arg,
            params_const,
            primfunc_name_hint=hint,
        )

        self.count += 1
        logger.debug(
            "Fused %s residual add: M_x=%d M_skip=%d shift=%d relu=%d elems=%d",
            out_dtype,
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


def _run_residual_add_pass(mod: IRModule, dtype_prefix: str, pass_name: str) -> IRModule:
    """Shared implementation: fuse residual-add patterns for one dtype prefix.

    Selects patterns whose composite name starts with `dtype_prefix` from
    the registry, runs FuseOpsByPattern, then lowers the matched composites.

    `dtype_prefix` must end with '.' (e.g. "int8_residual." not "int8_residual")
    so it cannot accidentally match entries with a different suffix.
    """
    assert dtype_prefix.endswith("."), (
        f"dtype_prefix must end with '.' to avoid partial matches, got {dtype_prefix!r}"
    )
    patterns = [
        (name, *factory())
        for name, factory in _PATTERN_REGISTRY
        if name.startswith(dtype_prefix)
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
        logger.info("%s: fused %d residual adds", pass_name, lowerer.count)
        mod = relax.transform.DeadCodeElimination()(mod)

    return mod


@tvm.transform.module_pass(opt_level=0, name="FuseInt8ResidualAdd")
class FuseInt8ResidualAdd:
    """Fuse int8 quantized residual add patterns into integer-only operations.

    Matches dequantize(a_i8) + dequantize(b_i8) -> [relu] -> quantize and
    replaces with a call_extern to c7x_int8_residual_add_relu using
    compile-time fixed-point scale/zp parameters.

    Both operand orders (add(a, b) and add(b, a)) are handled since
    TVM's pattern matcher is not commutative.

    Applicable to int8 quantized models with skip connections (ResNet, MobileNet, etc.).
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        return _run_residual_add_pass(mod, "int8_residual.", "FuseInt8ResidualAdd")


@tvm.transform.module_pass(opt_level=0, name="FuseInt16ResidualAdd")
class FuseInt16ResidualAdd:
    """Fuse int16 quantized residual add patterns into integer-only operations.

    Mirrors FuseInt8ResidualAdd for int16 precision. Requires symmetric
    activation quantization (all zero-points must be 0), which is enforced
    by C7xMMAQuantizer(dtype="int16").

    Emits call_extern to c7x_int16_residual_add_relu; output dtype is int16.

    Applicable to int16 quantized models with skip connections (MobileNet V2/V3, etc.).
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        return _run_residual_add_pass(mod, "int16_residual.", "FuseInt16ResidualAdd")
