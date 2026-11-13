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

Replaces each with call_extern to the corresponding c7x_int8_* C wrapper.

An activation's float result can also be read by an op the quantizer leaves in
float: a concat, or a squeeze-excite multiply by a [1,C,1,1] gate. The quantized
PyTorch graph gives that reader the float activation, rounded once at the
reader's own output scale. Before matching, _unshare_shared_activations gives
each such reader its own copy of the activation chain, so it is computed from
the activation's int8 input rather than from a requantized copy of its output
(which would add a rounding, and clipping wherever that copy's scale is narrower
than the activation's range):

  concat reader:    the copy is quantized at the concat's output scale, lowered
                    to the same c7x_int8_* kernel, and copied by the concat as is.
  multiply reader:  dq -> activation -> multiply(gate) -> q is lowered to
                    c7x_int8_{silu,hardswish}_gate_multiply.

Kernels: src/runtime/ti_dsp/kernels/c7x_activation.c
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tirx
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import CallPattern, is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import find_composite_span, propagate_span
from .ti_mmalib_legalize import _call_extern_checked, _scale_value, _zero_point_value

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


def _make_silu_f32out_pattern():
    """dq → sigmoid(dq) → multiply(dq, sigmoid) — same self-gated SiLU as
    _make_silu_pattern, but the composite ends at the multiply itself: no
    trailing quantize, float32 output.

    This is the C2f-block shape (YOLOv8/v5/26 backbone bottleneck blocks):
    Conv → SiLU → split into two channel halves, each routed to a different
    downstream branch before a later concat+quantize. _make_silu_pattern
    requires a direct trailing quantize, which this shape never has -- the
    SiLU'd float32 value feeds `split`/`concat` movement instead. Confirmed
    via direct IR inspection of compiled yolo26n/yolov8n: every remaining
    (post-FuseQDQToC7xActivation) bare sigmoid+multiply pair in both graphs
    matches exactly this shape, immediately consumed by R.split or R.concat.
    """
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    sig = is_op("relax.sigmoid")(dq)
    mul = is_op("relax.multiply")(dq, sig)
    return mul, {"x": x, "d_scale": d_s, "d_zp": d_z}


def _make_dfl_softmax_pattern():
    """dq(x[B,A,K,N]) → permute_dims → softmax → q → [B,K,A,N].

    YOLOv8's DFL (Distribution Focal Loss) head: softmax over the K=16
    reg_max distribution bins, independently for each of B*A*N (batch,
    box-coordinate, anchor) positions. The real compiled shape always has
    a permute_dims between the dequantize and the softmax (confirmed via
    direct IR inspection) -- softmax's own axis attr applies to the
    *permuted* tensor, not the original [B,A,K,N] layout, so the lowerer
    validates the permute's axes + softmax's axis together and derives
    B/A/K/N from the pre-permute shape (see c7x_softmax.h for why fusing
    the permute in, rather than materializing it, is the actual win here).
    """
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    perm = is_op("relax.permute_dims")(dq)
    sm = is_op("relax.nn.softmax")(perm)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(sm, o_s, o_z)
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


def _make_hardswish_pattern_commuted():
    """Same as _make_hardswish_pattern but with multiply operands swapped.

    PT2E exports some layers as multiply(clip, dq) rather than multiply(dq, clip).
    FuseOpsByPattern is non-commutative so both orderings must be registered.
    """
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    added = is_op("relax.add")(dq, wildcard())
    clipped1 = is_op("relax.clip")(added, wildcard(), wildcard())
    clipped2 = is_op("relax.clip")(clipped1, wildcard(), wildcard())
    mul = is_op("relax.multiply")(clipped2, dq)  # operands swapped vs canonical
    divided = is_op("relax.divide")(mul, wildcard())
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(divided, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _make_hardswish_chain(commuted: bool):
    """dq → add(dq, 3) → clip → clip → multiply → divide: hardswish's float output.

    commuted: multiply(clip, dq) rather than multiply(dq, clip) (PT2E exports both).
    """
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    added = is_op("relax.add")(dq, wildcard())
    clipped1 = is_op("relax.clip")(added, wildcard(), wildcard())
    clipped2 = is_op("relax.clip")(clipped1, wildcard(), wildcard())
    mul = (
        is_op("relax.multiply")(clipped2, dq) if commuted else is_op("relax.multiply")(dq, clipped2)
    )
    act = is_op("relax.divide")(mul, wildcard())
    return act, {"x": x, "d_scale": d_s, "d_zp": d_z}


# Float output of each activation chain _unshare_shared_activations can copy, and
# that a gate-multiply composite starts from. _make_silu_f32out_pattern is exactly
# SiLU's chain.
_ACTIVATION_CHAINS = {
    "silu": [_make_silu_f32out_pattern],
    "hardswish": [lambda: _make_hardswish_chain(False), lambda: _make_hardswish_chain(True)],
}


def _make_gate_multiply_pattern(chain_factory, act_first: bool):
    """dq → activation → multiply(activation, gate) → q: a squeeze-excite output.

    gate is the [1,C,1,1] float excitation (a sigmoid, or a dequantized hardsigmoid),
    computed outside the composite. act_first: the multiply's operand order.
    """
    act, annotations = chain_factory()
    gate = wildcard()
    mul = is_op("relax.multiply")(act, gate) if act_first else is_op("relax.multiply")(gate, act)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(mul, o_s, o_z)
    return q, {**annotations, "act": act, "gate": gate, "o_scale": o_s, "o_zp": o_z}


def _gate_multiply_entries():
    """(composite name, pattern factory, activation) for every chain and operand order."""
    entries = []
    for act_name, chains in _ACTIVATION_CHAINS.items():
        for i, chain in enumerate(chains):
            for act_first in (True, False):
                name = f"tidl_act.{act_name}_gate_multiply_{i}{'' if act_first else '_commuted'}"
                entries.append(
                    (name, lambda c=chain, a=act_first: _make_gate_multiply_pattern(c, a), act_name)
                )
    return entries


def _make_channel_scale_multiply_pattern():
    """dq(in1) * dq(in2) → q — SE-block excitation × feature-map multiply.

    One input has trailing [1,1] spatial dims (excitation from hardsigmoid);
    the other is the full [C,H,W] feature map.  The check function verifies
    this shape constraint and the lowerer determines which is which at compile
    time.
    """
    in1, s1, z1 = wildcard(), wildcard(), wildcard()
    in2, s2, z2 = wildcard(), wildcard(), wildcard()
    dq1 = is_op("relax.dequantize")(in1, s1, z1)
    dq2 = is_op("relax.dequantize")(in2, s2, z2)
    mul = is_op("relax.multiply")(dq1, dq2)
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(mul, o_s, o_z)
    return q, {
        "in1": in1,
        "s1": s1,
        "z1": z1,
        "in2": in2,
        "s2": s2,
        "z2": z2,
        "o_scale": o_s,
        "o_zp": o_z,
    }


def _make_channel_scale_multiply_pattern_commuted():
    """Same as _make_channel_scale_multiply_pattern with swapped multiply operands."""
    in1, s1, z1 = wildcard(), wildcard(), wildcard()
    in2, s2, z2 = wildcard(), wildcard(), wildcard()
    dq1 = is_op("relax.dequantize")(in1, s1, z1)
    dq2 = is_op("relax.dequantize")(in2, s2, z2)
    mul = is_op("relax.multiply")(dq2, dq1)  # operands swapped
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(mul, o_s, o_z)
    return q, {
        "in1": in1,
        "s1": s1,
        "z1": z1,
        "in2": in2,
        "s2": s2,
        "z2": z2,
        "o_scale": o_s,
        "o_zp": o_z,
    }


# (composite_name, pattern_factory)
# NOTE: "tidl_act.silu_f32out" is deliberately NOT run as part of this
# pass's own Round 1 -- see FuseQDQToC7xSiluF32Out below for why it needs
# to run later, after FuseQDQToC7xMovement. It's still listed here (rather
# than in a wholly separate registry) so _COMPOSITE_NAMES/_ActivationLowerer
# recognize and lower it correctly when that later pass invokes the same
# _run_patterns/_ActivationLowerer machinery.
_GATE_MULTIPLY_ENTRIES = _gate_multiply_entries()

# Composite name → activation, for the fused squeeze-excite composites.
_GATE_MULTIPLY_ACT = {name: act for name, _, act in _GATE_MULTIPLY_ENTRIES}

# Gate-multiply composites come first: they contain a whole activation chain, which
# the single-activation patterns would otherwise claim.
_PATTERN_REGISTRY = [
    *[(name, factory) for name, factory, _ in _GATE_MULTIPLY_ENTRIES],
    ("tidl_act.gelu", _make_gelu_pattern),
    ("tidl_act.silu", _make_silu_pattern),
    ("tidl_act.silu_f32out", _make_silu_f32out_pattern),
    ("tidl_act.hardsigmoid", _make_hardsigmoid_pattern),
    ("tidl_act.hardswish", _make_hardswish_pattern),
    ("tidl_act.hardswish_commuted", _make_hardswish_pattern_commuted),
    ("tidl_act.channel_scale_multiply", _make_channel_scale_multiply_pattern),
    ("tidl_act.channel_scale_multiply_commuted", _make_channel_scale_multiply_pattern_commuted),
    ("tidl_act.dfl_softmax", _make_dfl_softmax_pattern),
]

_COMPOSITE_NAMES = frozenset(name for name, _ in _PATTERN_REGISTRY)

# Composite names that use the two-input channel_scale_multiply lowering path.
_CHANNEL_SCALE_MULTIPLY_NAMES = frozenset(
    {
        "tidl_act.channel_scale_multiply",
        "tidl_act.channel_scale_multiply_commuted",
    }
)

# Composite names with no trailing quantize (float32 output) -- use
# _check_silu_f32out instead of _check_activation.
_F32OUT_NAMES = frozenset({"tidl_act.silu_f32out"})

# Composite names that use the dedicated DFL-softmax lowering path.
_DFL_SOFTMAX_NAMES = frozenset({"tidl_act.dfl_softmax"})


def _check_activation(ctx) -> bool:
    """Require int8 input and compile-time quantization constants."""
    x = ctx.annotated_expr["x"]
    if isinstance(x, tvm.ir.GenericConst):
        if x.value.dtype != "int8":
            return False
    elif hasattr(x, "ty") and hasattr(x.ty, "dtype"):
        if str(x.ty.dtype) != "int8":
            return False
    else:
        return False

    # Per-tensor constants only: the kernels take one scale and zero point each.
    for name in ("d_scale", "d_zp", "o_scale", "o_zp"):
        arg = ctx.annotated_expr[name]
        if not isinstance(arg, tvm.ir.GenericConst) or arg.value.numpy().size != 1:
            return False
    return True


# Activations with a kernel for a tensor quantized per channel (c7x_int8_<act>_per_channel).
_PER_CHANNEL_COMPOSITES = frozenset(
    {"tidl_act.silu", "tidl_act.hardswish", "tidl_act.hardswish_commuted"}
)


def _check_activation_per_channel(ctx) -> bool:
    """_check_activation, or the same with per-channel scales: one per channel of a
    [1, C, H, W] input, and one zero point for all channels."""
    if _check_activation(ctx):
        return True
    x = ctx.annotated_expr["x"]
    ty = getattr(x, "ty", None)
    if not isinstance(ty, relax.TensorType) or str(ty.dtype) != "int8" or ty.ndim != 4:
        return False
    shape = list(ty.shape) if ty.shape is not None else []
    if len(shape) != 4 or not all(isinstance(s, tirx.IntImm) for s in shape) or int(shape[0]) != 1:
        return False
    n_ch = int(shape[1])
    scales = [_scale_value(ctx.annotated_expr[k], n_ch) for k in ("d_scale", "o_scale")]
    zps = [_zero_point_value(ctx.annotated_expr[k]) for k in ("d_zp", "o_zp")]
    return all(s is not None for s in scales) and all(z is not None for z in zps)


def _check_silu_f32out(ctx) -> bool:
    """Same as _check_activation, minus the o_scale/o_zp check -- this
    pattern has no trailing quantize, so there is no output scale/zp to
    validate."""
    x = ctx.annotated_expr["x"]
    if isinstance(x, tvm.ir.GenericConst):
        if x.value.dtype != "int8":
            return False
    elif hasattr(x, "ty") and hasattr(x.ty, "dtype"):
        if str(x.ty.dtype) != "int8":
            return False
    else:
        return False

    for name in ("d_scale", "d_zp"):
        if not isinstance(ctx.annotated_expr[name], tvm.ir.GenericConst):
            return False
    return True


def _gate_shapes(act_ty, gate_ty):
    """(C, H*W) if act is int8-able [1,C,H,W] and gate is float32 [1,C,1,1], else None."""
    if not isinstance(act_ty, relax.TensorType) or not isinstance(gate_ty, relax.TensorType):
        return None
    if act_ty.shape is None or gate_ty.shape is None or str(gate_ty.dtype) != "float32":
        return None
    a, g = list(act_ty.shape), list(gate_ty.shape)
    if len(a) != 4 or len(g) != 4 or not all(isinstance(s, tirx.IntImm) for s in a + g):
        return None
    n, c, h, w = (int(s) for s in a)
    if n != 1 or [int(s) for s in g] != [1, c, 1, 1] or h * w <= 1:
        return None
    return c, h * w


def _check_gate_multiply(ctx) -> bool:
    """Require _check_activation's conditions, a [1,C,H,W] activation and a float32
    [1,C,1,1] gate."""
    if not _check_activation(ctx):
        return False
    act = ctx.annotated_expr["act"]
    gate = ctx.annotated_expr["gate"]
    return _gate_shapes(getattr(act, "ty", None), getattr(gate, "ty", None)) is not None


def _check_channel_scale_multiply(ctx) -> bool:
    """Require two int8 inputs, compile-time QDQ constants, and one input with
    trailing [1,1] spatial dims (excitation) and the other with H×W > 1."""
    for inp_name in ("in1", "in2"):
        inp = ctx.annotated_expr[inp_name]
        if isinstance(inp, tvm.ir.GenericConst):
            if inp.value.dtype != "int8":
                return False
        elif hasattr(inp, "ty") and hasattr(inp.ty, "dtype"):
            if str(inp.ty.dtype) != "int8":
                return False
        else:
            return False

    for name in ("s1", "z1", "s2", "z2", "o_scale", "o_zp"):
        if not isinstance(ctx.annotated_expr[name], tvm.ir.GenericConst):
            return False

    # Verify shape constraint: exactly one input has trailing [1, 1] dims.
    in1 = ctx.annotated_expr["in1"]
    in2 = ctx.annotated_expr["in2"]
    in1_si = getattr(in1, "ty", None)
    in2_si = getattr(in2, "ty", None)
    if in1_si is None or in2_si is None:
        return False
    in1_shape = getattr(in1_si, "shape", None)
    in2_shape = getattr(in2_si, "shape", None)
    if in1_shape is None or in2_shape is None:
        return False
    if len(in1_shape) < 4 or len(in2_shape) < 4:
        return False
    for s in [*in1_shape, *in2_shape]:
        if not isinstance(s, tirx.IntImm):
            return False
    s1_last2 = int(in1_shape[-2]) == 1 and int(in1_shape[-1]) == 1
    s2_last2 = int(in2_shape[-2]) == 1 and int(in2_shape[-1]) == 1
    # XOR: exactly one has trailing [1,1]
    return s1_last2 != s2_last2


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _ActivationLowerer(PyExprMutator):
    """Lower TIDL activation composites to call_extern."""

    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.count = 0
        # Maps emitted Tuple Var → list of field Vars, for TupleGetItem inlining.
        self._tuple_fields: dict = {}

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

        if name in _CHANNEL_SCALE_MULTIPLY_NAMES:
            return self._lower_channel_scale_multiply(call, func)
        if name in _DFL_SOFTMAX_NAMES:
            return self._lower_dfl_softmax(call, func)
        if name in _GATE_MULTIPLY_ACT:
            return self._lower_gate_multiply(call, func, _GATE_MULTIPLY_ACT[name])
        return self._lower_single_input(call, func, name)

    def _lower_gate_multiply(self, call, func, act_name):
        """Lower a gate-multiply composite to c7x_int8_<act>_gate_multiply.

        Inputs: the activation's int8 input and the float32 [1,C,1,1] gate (the
        multiply operand that is a composite parameter). The dequantize gives the
        input qparams; the final quantize the output qparams.
        """
        param_to_arg = dict(zip(func.params, call.args))
        bindings = func.body.blocks[0].bindings
        values = {b.var: b.value for b in bindings}

        dq = next(
            (
                b.value
                for b in bindings
                if isinstance(b.value, relax.Call)
                and getattr(b.value.op, "name", "") == "relax.dequantize"
            ),
            None,
        )
        q = bindings[-1].value
        if dq is None or not (
            isinstance(q, relax.Call) and getattr(q.op, "name", "") == "relax.quantize"
        ):
            return super().visit_call_(call)
        mul = values.get(q.args[0])
        if not (isinstance(mul, relax.Call) and getattr(mul.op, "name", "") == "relax.multiply"):
            return super().visit_call_(call)
        gate_param = next((a for a in mul.args if a in param_to_arg), None)
        act_var = next((a for a in mul.args if a in values), None)
        if gate_param is None or act_var is None:
            return super().visit_call_(call)

        def const(expr):
            return param_to_arg.get(expr, expr).value.numpy()

        x_arg = param_to_arg.get(dq.args[0], dq.args[0])
        gate_arg = param_to_arg[gate_param]
        shapes = _gate_shapes(act_var.ty, gate_param.ty)
        if shapes is None or not isinstance(call.ty, relax.TensorType):
            return super().visit_call_(call)
        n_ch, hw = shapes
        d_scale_v, d_zp_v = float(const(dq.args[1])), int(const(dq.args[2]))
        o_scale_v, o_zp_v = float(const(q.args[1])), int(const(q.args[2]))
        output_shape = [int(s) for s in call.ty.shape]
        extern_name = f"c7x_int8_{act_name}_gate_multiply"

        def te_gate_multiply(x_t, gate_t):
            def fcompute(ins, outs):
                return tirx.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    ins[1].data,
                    outs[0].data,
                    tirx.IntImm("int32", n_ch),
                    tirx.IntImm("int32", hw),
                    tirx.IntImm("int32", d_zp_v),
                    tirx.FloatImm("float32", d_scale_v),
                    tirx.IntImm("int32", o_zp_v),
                    tirx.FloatImm("float32", o_scale_v),
                )

            return te.extern(
                output_shape, [x_t, gate_t], fcompute, name="tidl_act_gate_out", dtype="int8"
            )

        result = self.builder_.call_te(
            te_gate_multiply, x_arg, gate_arg, primfunc_name_hint=extern_name
        )
        self.count += 1
        logger.debug(
            "Fused %s: C=%d HW=%d d_zp=%d d_scale=%.6g o_zp=%d o_scale=%.6g",
            extern_name,
            n_ch,
            hw,
            d_zp_v,
            d_scale_v,
            o_zp_v,
            o_scale_v,
        )
        return propagate_span(result, find_composite_span(func))

    def _lower_per_channel(self, call, x_arg, composite_name, d_scale, d_zp, o_scale, o_zp, span):
        """Lower a silu/hardswish composite quantized per channel to
        c7x_int8_<act>_per_channel, with the scales as float32 [C] tensors."""
        if not isinstance(call.ty, relax.TensorType) or not call.ty.shape:
            return super().visit_call_(call)
        output_shape = [int(s) for s in call.ty.shape]
        if len(output_shape) != 4 or output_shape[0] != 1:
            return super().visit_call_(call)
        n_ch, hw = output_shape[1], output_shape[2] * output_shape[3]
        act_suffix = composite_name[len(_COMPOSITE_PREFIX) :].removesuffix("_commuted")
        extern_name = f"c7x_int8_{act_suffix}_per_channel"
        sx = relax.const(np.broadcast_to(np.float32(d_scale), (n_ch,)).astype(np.float32))
        sy = relax.const(np.broadcast_to(np.float32(o_scale), (n_ch,)).astype(np.float32))

        def te_activation(x_t, sx_t, sy_t):
            def fcompute(ins, outs):
                return tirx.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    ins[1].data,
                    ins[2].data,
                    outs[0].data,
                    tirx.IntImm("int32", n_ch),
                    tirx.IntImm("int32", hw),
                    tirx.IntImm("int32", d_zp),
                    tirx.IntImm("int32", o_zp),
                )

            return te.extern(
                output_shape, [x_t, sx_t, sy_t], fcompute, name="tidl_act_out", dtype="int8"
            )

        result = self.builder_.call_te(te_activation, x_arg, sx, sy, primfunc_name_hint=extern_name)
        self.count += 1
        logger.debug("Fused %s: C=%d HW=%d", extern_name, n_ch, hw)
        return propagate_span(result, span)

    def _lower_silu_f32out(self, call, call_sinfo, x_arg, d_scale_val, d_zp_val, span):
        """Lower a silu_f32out composite to c7x_int8_silu_f32out.

        No trailing quantize, so no is_tuple_out companion-dequantize case to
        handle the way _lower_single_input's int8-output path does: a tuple
        struct_info here would mean the multiply's float32 result itself has
        an external consumer, which isn't observed on the real compiled
        yolo26n/yolov8n graphs (every self-gated SiLU without a trailing
        quantize is consumed exactly once, by the split/concat that follows
        it). Decline rather than guess at the right reconstruction if it
        ever occurs.
        """
        if isinstance(call_sinfo, relax.TupleType):
            return super().visit_call_(call)
        if not isinstance(call_sinfo, relax.TensorType) or not call_sinfo.shape:
            return super().visit_call_(call)

        output_shape = [int(s) for s in call_sinfo.shape]
        n_elem = 1
        for s in output_shape:
            n_elem *= s

        extern_name = "c7x_int8_silu_f32out"
        d_zp_v = int(d_zp_val)
        d_scale_v = float(d_scale_val)
        n = n_elem

        def te_silu_f32out(x_t):
            def fcompute(ins, outs):
                return tirx.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    outs[0].data,
                    tirx.IntImm("int32", n),
                    tirx.IntImm("int32", d_zp_v),
                    tirx.FloatImm("float32", d_scale_v),
                )

            return te.extern(output_shape, [x_t], fcompute, name="tidl_act_f32out", dtype="float32")

        result = self.builder_.call_te(te_silu_f32out, x_arg, primfunc_name_hint=extern_name)
        self.count += 1
        logger.debug(
            "Fused %s: n=%d d_zp=%d d_scale=%.6g", extern_name, n_elem, d_zp_v, d_scale_v
        )
        return propagate_span(result, span)

    def _lower_single_input(self, call, func, composite_name):
        """Lower single-input activation composites (gelu/silu/hardsigmoid/hardswish/
        silu_f32out)."""
        param_to_arg = dict(zip(func.params, call.args))
        is_f32out = composite_name in _F32OUT_NAMES

        x_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None
        axes = []  # (scale, axis) of the dequantize and quantize
        n_ch = None  # the input's channel count, for per-channel scales

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
                x_ty = getattr(x_arg, "ty", None)
                if isinstance(x_ty, relax.TensorType) and x_ty.ndim == 4 and x_ty.shape is not None:
                    n_ch = int(list(x_ty.shape)[1])
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                d_scale_val, d_zp_val = _scale_value(s, n_ch), _zero_point_value(z)
                axes.append((d_scale_val, int(getattr(val.attrs, "axis", -1))))
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val, o_zp_val = _scale_value(s, n_ch), _zero_point_value(z)
                axes.append((o_scale_val, int(getattr(val.attrs, "axis", -1))))

        if x_arg is None or d_scale_val is None or d_zp_val is None:
            return super().visit_call_(call)
        if not is_f32out and (o_scale_val is None or o_zp_val is None):
            return super().visit_call_(call)

        call_sinfo = call.ty
        composite_span = find_composite_span(func)

        if any(isinstance(s, np.ndarray) for s, _ in axes):
            # Per-channel scales: only on the channel axis of a [1, C, H, W] tensor.
            if is_f32out or composite_name not in _PER_CHANNEL_COMPOSITES:
                return super().visit_call_(call)
            if any(isinstance(s, np.ndarray) and a not in (1, -3) for s, a in axes):
                return super().visit_call_(call)
            return self._lower_per_channel(
                call,
                x_arg,
                composite_name,
                d_scale_val,
                d_zp_val,
                o_scale_val,
                o_zp_val,
                composite_span,
            )

        if is_f32out:
            return self._lower_silu_f32out(
                call, call_sinfo, x_arg, d_scale_val, d_zp_val, composite_span
            )

        # Determine output shape.  For hardswish layers whose float32 intermediate
        # is shared with an SE-block multiply, FuseOpsByPattern creates a
        # tuple-output composite: R.Tuple(int8_tensor, float32_tensor).
        # We emit c7x_int8_hardswish for the int8 part and re-dequantize it
        # to recover the float32 part for the downstream SE multiply.
        is_tuple_out = isinstance(call_sinfo, relax.TupleType)
        if is_tuple_out:
            # Find the int8 field.
            int8_idx = next(
                (
                    i
                    for i, f in enumerate(call_sinfo.fields)
                    if isinstance(f, relax.TensorType) and str(f.dtype) == "int8"
                ),
                None,
            )
            if int8_idx is None:
                return super().visit_call_(call)
            int8_sinfo = call_sinfo.fields[int8_idx]
            if not int8_sinfo.shape:
                return super().visit_call_(call)
            output_shape = [int(s) for s in int8_sinfo.shape]
        elif not isinstance(call_sinfo, relax.TensorType) or not call_sinfo.shape:
            return super().visit_call_(call)
        else:
            output_shape = [int(s) for s in call_sinfo.shape]

        n_elem = 1
        for s in output_shape:
            n_elem *= s

        # Map composite suffix → extern function name.
        act_suffix = composite_name[len(_COMPOSITE_PREFIX) :].removesuffix("_commuted")
        extern_name = f"c7x_int8_{act_suffix}"

        d_zp_v = int(d_zp_val)  # type: ignore[arg-type]
        d_scale_v = float(d_scale_val)  # type: ignore[arg-type]
        o_zp_v = int(o_zp_val)  # type: ignore[arg-type]
        o_scale_v = float(o_scale_val)
        n = n_elem

        def te_activation(x_t):
            def fcompute(ins, outs):
                return tirx.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    outs[0].data,
                    tirx.IntImm("int32", n),
                    tirx.IntImm("int32", d_zp_v),
                    tirx.FloatImm("float32", d_scale_v),
                    tirx.IntImm("int32", o_zp_v),
                    tirx.FloatImm("float32", o_scale_v),
                )

            return te.extern(output_shape, [x_t], fcompute, name="tidl_act_out", dtype="int8")

        int8_result = propagate_span(
            self.builder_.call_te(te_activation, x_arg, primfunc_name_hint=extern_name),
            composite_span,
        )
        self.count += 1

        if is_tuple_out:
            # Reconstruct the float32 output by dequantizing the int8 result.
            # The downstream SE multiply will see dq(int8_hardswish) × dq(int8_sig),
            # which the channel_scale_multiply pass (Round 2) can then match.
            float32_result = self.builder_.emit(
                relax.op.dequantize(
                    int8_result,
                    relax.const(o_scale_v, "float32"),
                    relax.const(o_zp_v, "int8"),
                )
            )
            out_fields = []
            for i, f in enumerate(call_sinfo.fields):
                if i == int8_idx:
                    out_fields.append(int8_result)
                else:
                    out_fields.append(float32_result)
            result = self.builder_.emit(relax.Tuple(out_fields))
            self._tuple_fields[result] = out_fields
            logger.debug("Fused %s (tuple output): int8+float32", extern_name)
            return result

        logger.debug(
            "Fused %s: n=%d d_zp=%d d_scale=%.6g o_zp=%d o_scale=%.6g",
            extern_name,
            n_elem,
            d_zp_v,
            d_scale_v,
            o_zp_v,
            o_scale_v,
        )
        return int8_result

    def _lower_channel_scale_multiply(self, call, func):
        """Lower channel_scale_multiply composites to c7x_int8_channel_scale_multiply."""
        param_to_arg = dict(zip(func.params, call.args))

        # Collect the two dequantize ops and one quantize op from the composite.
        dq_entries = []  # list of (input_arg, scale_val, zp_val)
        o_scale_val = o_zp_val = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if not isinstance(val, relax.Call) or not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)
            if "dequantize" in op_name:
                inp = param_to_arg.get(val.args[0], val.args[0])
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                dq_entries.append((inp, float(s.value.numpy()), int(z.value.numpy())))
            elif "quantize" in op_name and dq_entries:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.value.numpy())
                o_zp_val = int(z.value.numpy())

        if len(dq_entries) < 2 or o_scale_val is None:
            return super().visit_call_(call)

        call_sinfo = call.ty
        if not isinstance(call_sinfo, relax.TensorType) or not call_sinfo.shape:
            return super().visit_call_(call)

        # Determine which dq input is excitation [1,C,1,1] and which is feature map.
        (arg0, s0, z0), (arg1, s1, z1) = dq_entries[0], dq_entries[1]
        shape0 = [int(x) for x in arg0.ty.shape]
        shape1 = [int(x) for x in arg1.ty.shape]

        if shape0[-2] == 1 and shape0[-1] == 1:
            exc_arg, fm_arg = arg0, arg1
            s_exc, z_exc, s_fm, z_fm = s0, z0, s1, z1
            fm_shape = shape1
        else:
            exc_arg, fm_arg = arg1, arg0
            s_exc, z_exc, s_fm, z_fm = s1, z1, s0, z0
            fm_shape = shape0

        # C is the channel dimension; H_W is the product of remaining spatial dims.
        C = fm_shape[1]
        H_W = 1
        for d in fm_shape[2:]:
            H_W *= d

        output_shape = [int(s) for s in call_sinfo.shape]

        s_exc_v = float(s_exc)
        z_exc_v = int(z_exc)
        s_fm_v = float(s_fm)
        z_fm_v = int(z_fm)
        s_out_v = float(o_scale_val)
        z_out_v = int(o_zp_val)
        C_v = int(C)
        H_W_v = int(H_W)

        def te_channel_scale_multiply(exc_t, fm_t):
            def fcompute(ins, outs):
                return tirx.call_extern(
                    "int32",
                    "c7x_int8_channel_scale_multiply",
                    ins[0].data,
                    ins[1].data,
                    outs[0].data,
                    tirx.IntImm("int32", C_v),
                    tirx.IntImm("int32", H_W_v),
                    tirx.FloatImm("float32", s_exc_v),
                    tirx.IntImm("int32", z_exc_v),
                    tirx.FloatImm("float32", s_fm_v),
                    tirx.IntImm("int32", z_fm_v),
                    tirx.FloatImm("float32", s_out_v),
                    tirx.IntImm("int32", z_out_v),
                )

            return te.extern(
                output_shape,
                [exc_t, fm_t],
                fcompute,
                name="channel_scale_mul_out",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_channel_scale_multiply,
            exc_arg,
            fm_arg,
            primfunc_name_hint="c7x_int8_channel_scale_multiply",
        )
        self.count += 1
        logger.debug(
            "Fused c7x_int8_channel_scale_multiply: C=%d H_W=%d",
            C_v,
            H_W_v,
        )
        return propagate_span(result, find_composite_span(func))

    def _lower_dfl_softmax(self, call, func):
        """Lower a dfl_softmax composite to c7x_int8_dfl_softmax.

        Unlike _lower_single_input's patterns, the intermediate permute_dims
        and softmax ops here carry attrs (axes / axis) that must match the
        one real shape this kernel implements -- validated here by walking
        the composite's bindings directly, following the same convention as
        ti_fuse_qdq_c7x_movement.py's resize2d attr validation (the pattern's
        own check() only validates dtype/constant-ness of the wildcards;
        op-specific attribute validation happens in the lowerer, which can
        decline by falling through to the generic path rather than guessing).
        """
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None
        permute_axes = None
        softmax_axis = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if not isinstance(val, relax.Call) or not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)
            if "dequantize" in op_name:
                x_param = val.args[0]
                x_arg = param_to_arg.get(x_param, x_param)
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                d_scale_val = float(s.value.numpy())
                d_zp_val = int(z.value.numpy())
            elif op_name == "relax.permute_dims":
                axes = val.attrs["axes"] if val.attrs else None
                permute_axes = [int(a) for a in axes] if axes is not None else None
            elif op_name == "relax.nn.softmax":
                softmax_axis = int(val.attrs["axis"])
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.value.numpy())
                o_zp_val = int(z.value.numpy())

        if x_arg is None or d_scale_val is None or o_scale_val is None:
            return super().visit_call_(call)

        x_sinfo = getattr(x_arg, "ty", None)
        if not isinstance(x_sinfo, relax.TensorType) or not x_sinfo.shape:
            return super().visit_call_(call)
        pre_shape = list(x_sinfo.shape)
        if len(pre_shape) != 4 or any(not isinstance(s, tirx.IntImm) for s in pre_shape):
            return super().visit_call_(call)

        # Only the one real shape this kernel implements: permute swaps
        # axes 1 and 2 ([0,2,1,3]), softmax reduces the permuted tensor's
        # axis 1 -- i.e. the pre-permute axis 2 ("K", reg_max bins). Any
        # other permute/axis combination declines to fuse rather than
        # mis-deriving B/A/K/N.
        if permute_axes != [0, 2, 1, 3] or softmax_axis not in (1, -3):
            return super().visit_call_(call)

        B_v, A_v, K_v, N_v = (int(s) for s in pre_shape)
        d_zp_v = int(d_zp_val)
        d_scale_v = float(d_scale_val)
        o_zp_v = int(o_zp_val)
        o_scale_v = float(o_scale_val)

        def te_dfl_softmax(x_t):
            def fcompute(ins, outs):
                # The kernel returns -1 when its workspace allocation fails.
                return _call_extern_checked(
                    "int32",
                    "c7x_int8_dfl_softmax",
                    ins[0].data,
                    outs[0].data,
                    tirx.IntImm("int32", B_v),
                    tirx.IntImm("int32", A_v),
                    tirx.IntImm("int32", K_v),
                    tirx.IntImm("int32", N_v),
                    tirx.IntImm("int32", d_zp_v),
                    tirx.FloatImm("float32", d_scale_v),
                    tirx.IntImm("int32", o_zp_v),
                    tirx.FloatImm("float32", o_scale_v),
                )

            return te.extern(
                [B_v, K_v, A_v, N_v], [x_t], fcompute, name="dfl_softmax_out", dtype="int8"
            )

        result = self.builder_.call_te(
            te_dfl_softmax, x_arg, primfunc_name_hint="c7x_int8_dfl_softmax"
        )
        self.count += 1
        logger.debug(
            "Fused c7x_int8_dfl_softmax: B=%d A=%d K=%d N=%d", B_v, A_v, K_v, N_v
        )
        return propagate_span(result, find_composite_span(func))


# =========================================================================
# Public pass
# =========================================================================


def _is_op(expr, name: str) -> bool:
    return isinstance(expr, relax.Call) and getattr(expr.op, "name", "") == name


def _is_scalar(expr) -> bool:
    ty = getattr(expr, "ty", None)
    return isinstance(ty, relax.TensorType) and ty.ndim == 0


def _match_activation_chain(expr, values):
    """The chain's calls if expr is an activation's float output, else None.

    Only call patterns belong to the chain: a wildcard (the int8 input, a scale)
    can match a call too, and copying it would duplicate the producer.
    """
    for chains in _ACTIVATION_CHAINS.values():
        for chain in chains:
            pat, _ = chain()
            matched = pat.extract_matched_expr(expr, values)
            if matched is not None:
                return [e for p, e in matched.items() if isinstance(p, CallPattern)]
    return None


def _plan_unsharing(func: relax.Function) -> dict:
    """Map each float reader binding var to (concat quantize, [(activation var, chain calls)]).

    Only for an activation output that has its own quantize reader. Readers:
      concat whose only user is a per-tensor quantize -> (that quantize, every such
          activation it reads)
      multiply by a [1,C,1,1] gate whose only user is a quantize -> (None, [the activation])
      a second quantize (another scale: a depthwise conv's per-channel input and a
          residual add's per-tensor one) -> (None, [the activation])
    Other readers keep the tuple-output path and are logged.
    """
    bindings = [b for block in func.body.blocks for b in block.bindings]
    values = {b.var: b.value for b in bindings}
    users: dict = {b.var: [] for b in bindings}
    for b in bindings:

        def note(e, reader=b.var):
            if isinstance(e, relax.Var) and e in users:
                users[e].append(reader)

        relax.analysis.post_order_visit(b.value, note)

    def only_quantize_reader(var):
        readers = users.get(var, [])
        return (
            values[readers[0]]
            if len(readers) == 1 and _is_op(values[readers[0]], "relax.quantize")
            else None
        )

    plan: dict = {}
    for b in bindings:
        readers = users[b.var]
        if len(readers) < 2 or not any(_is_op(values[r], "relax.quantize") for r in readers):
            continue
        chain = _match_activation_chain(b.value, values)
        if chain is None:
            continue
        own = next(r for r in readers if _is_op(values[r], "relax.quantize"))
        for r in dict.fromkeys(readers):  # a concat can read the activation twice
            rv = values[r]
            if _is_op(rv, "relax.quantize"):
                if r is not own:
                    plan[r] = (None, [(b.var, chain)])
                continue
            q = only_quantize_reader(r)
            if _is_op(rv, "relax.concat") and q is not None:
                # The copy takes the concat's qparams, so they must be per-tensor.
                if all(_is_scalar(a) for a in q.args[1:3]):
                    plan.setdefault(r, (q, []))[1].append((b.var, chain))
                else:
                    logger.info("Activation %s: concat reader is quantized per channel", b.var.name)
            elif _is_op(rv, "relax.multiply") and q is not None:
                gate = rv.args[1] if rv.args[0].same_as(b.var) else rv.args[0]
                if _gate_shapes(b.var.ty, getattr(gate, "ty", None)) is not None:
                    plan[r] = (None, [(b.var, chain)])
                else:
                    logger.info(
                        "Activation %s: multiply reader is not a [1,C,1,1] gate", b.var.name
                    )
            else:
                logger.info(
                    "Activation %s: float reader %s keeps a requantized copy",
                    b.var.name,
                    getattr(getattr(rv, "op", None), "name", type(rv).__name__),
                )
    return plan


@mutator
class _Unsharer(PyExprMutator):
    """Rewrite each planned reader to read its own copy of the activation chain."""

    def __init__(self, mod: IRModule, plan: dict, values: dict):
        super().__init__(mod)
        self.plan = plan
        self.values = values  # the original function's var → bound value
        self.count = 0

    def _copy_chain(self, act_var, chain):
        """Re-emit the chain's calls as fresh bindings. Everything outside the chain
        (the int8 input, the constants) is read through the usual remapping."""
        copies: dict = {}

        def chain_index(expr):
            value = self.values.get(expr) if isinstance(expr, relax.Var) else expr
            if not isinstance(value, relax.Call):
                return None
            return next((i for i, c in enumerate(chain) if c.same_as(value)), None)

        def copy(expr):
            i = chain_index(expr)
            if i is None:
                return self.visit_expr(expr)
            if i not in copies:
                call = chain[i]
                new_args = [copy(a) for a in call.args]
                copies[i] = self.builder_.emit(
                    relax.Call(call.op, new_args, call.attrs, call.ty_args)
                )
            return copies[i]

        return copy(act_var)

    def visit_var_binding_(self, binding: relax.VarBinding):
        entry = self.plan.get(binding.var)
        if entry is None:
            return super().visit_var_binding_(binding)
        concat_q, acts = entry
        copies = []  # [(original activation var, its copy)]
        for act_var, chain in acts:
            copy = self._copy_chain(act_var, chain)
            if concat_q is not None:
                # Quantize the copy at the concat's output scale: the concat then
                # copies it as is, and the result equals quantizing the concatenated
                # float.
                scale, zp = self.visit_expr(concat_q.args[1]), self.visit_expr(concat_q.args[2])
                q = self.builder_.emit(relax.Call(concat_q.op, [copy, scale, zp], concat_q.attrs))
                copy = self.builder_.emit(
                    relax.op.dequantize(
                        q, scale, zp, axis=int(concat_q.attrs.axis), out_dtype="float32"
                    )
                )
            copies.append((act_var, copy))

        def swap(expr):
            if isinstance(expr, relax.Var):
                for act_var, copy in copies:
                    if expr.same_as(act_var):
                        return copy
            return self.visit_expr(expr)

        value = binding.value
        if _is_op(value, "relax.concat"):
            fields = value.args[0].fields if isinstance(value.args[0], relax.Tuple) else None
            if fields is None:
                return super().visit_var_binding_(binding)
            new_value = relax.Call(value.op, [relax.Tuple([swap(f) for f in fields])], value.attrs)
        else:
            new_value = relax.Call(value.op, [swap(a) for a in value.args], value.attrs)
        new_var = (
            self.builder_.emit(new_value)
            if isinstance(binding.var, relax.DataflowVar)
            else self.builder_.emit_output(new_value)
        )
        self.set_var_remap(binding.var, new_var)
        self.count += 1


def _unshare_shared_activations(mod: IRModule) -> IRModule:
    """Give each float reader of a shared activation output its own activation copy.

    See the module docstring. The original activation keeps its quantize readers.
    """
    total = 0
    for gv, func in list(mod.functions_items()):
        if not isinstance(func, relax.Function) or (func.attrs and "Composite" in func.attrs):
            continue
        plan = _plan_unsharing(func)
        if not plan:
            continue
        values = {b.var: b.value for block in func.body.blocks for b in block.bindings}
        unsharer = _Unsharer(mod, plan, values)
        new_func = unsharer.visit_expr(func)
        unsharer.builder_.update_func(gv, new_func)
        mod = unsharer.builder_.get()
        total += unsharer.count
    if total:
        logger.info("FuseQDQToC7xActivation: gave %d float readers their own activation", total)
    return mod


def _run_activation_patterns(mod: IRModule, pattern_list: list):
    """Run FuseOpsByPattern + lowering for one set of patterns.

    Returns (mod, count, tuple_fields) where tuple_fields maps emitted
    Tuple Vars to their field Var lists (for TupleGetItem inlining).

    Module-level (not a method) so both FuseQDQToC7xActivation and
    FuseQDQToC7xSiluF32Out can call it -- @tvm.transform.module_pass wraps
    the decorated class into an opaque Pass object, so a classmethod/
    staticmethod defined on FuseQDQToC7xActivation is not reachable from
    outside its own transform_module after decoration.
    """
    mod = relax.transform.FuseOpsByPattern(pattern_list, bind_constants=False)(mod)
    lowerer = _ActivationLowerer(mod)
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            if "Composite" in (func.attrs or {}):
                continue
            func = lowerer.visit_expr(func)
            lowerer.builder_.update_func(gv, func)
    mod = lowerer.builder_.get()
    if lowerer.count > 0:
        mod = relax.transform.DeadCodeElimination()(mod)
    return mod, lowerer.count, lowerer._tuple_fields


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xActivation")
class FuseQDQToC7xActivation:
    """Fuse QDQ-wrapped activation ops into c7x_int8_* C kernel calls.

    Handles: gelu, silu, hardsigmoid, hardswish.
    Requires int8 input and compile-time quantization constants (satisfied
    by the PT2E pipeline after convert_pt2e).

    Applicable to both MMALIB and non-MMALIB C7x targets.
    """

    @staticmethod
    def _inline_tuple_getitems(mod: IRModule, _ignored: dict = None) -> IRModule:
        """Inline TupleGetItem(tuple_var, i) → direct field Var for Tuple bindings.

        After Round 1, hardswish with a shared SE-multiply output is lowered to:
          int8_hs    = c7x_int8_hardswish(...)
          float32_hs = dequantize(int8_hs, ...)
          result_var = Tuple([int8_hs, float32_hs])
          tgi_1      = TupleGetItem(result_var, 1)
          se_mul     = multiply(tgi_1, dq_sig)   # TupleGetItem blocks FuseOpsByPattern

        After inlining, tgi_1 = float32_hs.  FuseOpsByPattern (Round 2) then traces
        multiply(tgi_1, ...) → float32_hs → dequantize(...) and matches the pattern.

        Scans the module directly rather than using _tuple_fields, which is stale
        after DeadCodeElimination recreates Var objects with new Python identities.
        Uses same_as() on vars from the same pre-mutation function, which is reliable.
        """

        @mutator
        class _Inliner(PyExprMutator):
            def __init__(self, m):
                super().__init__(m)
                # Pre-scanned per function.
                # _tuple_map: Tuple-binding var → [field0, field1, ...]
                # _var_vals:  all var → binding value (for one-hop resolution)
                self._tuple_map: dict = {}
                self._var_vals: dict = {}

            def pre_scan(self, func):
                """Populate _tuple_map and _var_vals from function bindings."""
                self._tuple_map = {}
                self._var_vals = {}
                for block in func.body.blocks:
                    for binding in block.bindings:
                        self._var_vals[binding.var] = binding.value
                        if isinstance(binding.value, relax.Tuple):
                            self._tuple_map[binding.var] = list(binding.value.fields)

            def _resolve_tuple_fields(self, var):
                """Follow the Var→Var chain until a Tuple binding is found.

                After Round 1, PyExprMutator emits multiple levels of indirection:
                  result_var    = Tuple([int8_hs, float32_hs])   <- _tuple_map
                  composite_mid = result_var                     <- Var→Var
                  composite_out = composite_mid                  <- Var→Var

                TupleGetItem accesses composite_out; we must walk the chain
                to reach result_var before checking _tuple_map.
                """
                cur = var
                for _ in range(8):
                    for tvar, fields in self._tuple_map.items():
                        if cur.same_as(tvar):
                            return fields
                    # Follow one more Var→Var hop.
                    nxt = None
                    for bvar, bval in self._var_vals.items():
                        if cur.same_as(bvar):
                            if isinstance(bval, relax.Var):
                                nxt = bval
                            break
                    if nxt is None:
                        break
                    cur = nxt
                return None

            def visit_tuple_getitem_(self, op):
                old_src = op.tuple_value
                if not isinstance(old_src, relax.Var):
                    return super().visit_tuple_getitem_(op)
                fields = self._resolve_tuple_fields(old_src)
                if fields is None:
                    return super().visit_tuple_getitem_(op)
                field = fields[op.index]
                if not isinstance(field, relax.Var):
                    return super().visit_tuple_getitem_(op)
                # Only inline the float32 dequantize field (not the int8 field).
                # Critically, emit a FRESH dequantize rather than pointing at the
                # shared float32_hs.  float32_hs is already in the Tuple expression
                # so it has multiple users; FuseOpsByPattern (Round 2) would see a
                # node shared between the Tuple group and the SE-multiply group and
                # report a cyclic-group dependency.  A fresh dequantize node (same
                # parameters, separate allocation) is unshared and can be cleanly
                # fused into the channel_scale_multiply composite.
                for bvar, bval in self._var_vals.items():
                    if not field.same_as(bvar):
                        continue
                    if not isinstance(bval, relax.Call):
                        break
                    if not (hasattr(bval.op, "name") and "dequantize" in str(bval.op.name)):
                        break
                    # Re-emit the dequantize with remapped args (new node, not shared).
                    new_args = [self.visit_expr(a) for a in bval.args]
                    return self.builder_.emit(
                        relax.Call(bval.op, new_args, bval.attrs, bval.ty_args)
                    )
                return super().visit_tuple_getitem_(op)

        inliner = _Inliner(mod)
        changed = False
        for gv, func in mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            inliner.pre_scan(func)
            if not inliner._tuple_map:
                continue
            new_func = inliner.visit_expr(func)
            inliner.builder_.update_func(gv, new_func)
            changed = True

        if not changed:
            return mod
        mod = inliner.builder_.get()
        mod = relax.transform.DeadCodeElimination()(mod)
        # Flatten any Var→Var chains introduced by the TupleGetItem substitution
        # (e.g. new_tgi = fresh_dq_var).  Without this, FuseOpsByPattern sees the
        # intermediate Var in the multiply args and its group-dependency tracking
        # produces a false cycle between the SE-multiply group and the Tuple group.
        mod = relax.transform.CanonicalizeBindings()(mod)
        return mod

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Round 1: single-input activations (gelu/silu/hardsigmoid/hardswish).
        # Hardswish layers whose float32 intermediate is shared with an SE-block
        # multiply produce a tuple-output composite; the lowerer handles this by
        # emitting c7x_int8_hardswish + an explicit dequantize for the float32 part.
        round1 = []
        round2 = []
        for composite_name, factory in _PATTERN_REGISTRY:
            if composite_name in _F32OUT_NAMES:
                # Handled by FuseQDQToC7xSiluF32Out, later in the pipeline
                # (after FuseQDQToC7xMovement) -- see that pass's docstring.
                continue
            pat, annotations = factory()
            if composite_name in _CHANNEL_SCALE_MULTIPLY_NAMES:
                check = _check_channel_scale_multiply
                round2.append((composite_name, pat, annotations, check))
            else:
                if composite_name in _GATE_MULTIPLY_ACT:
                    check = _check_gate_multiply
                elif composite_name in _PER_CHANNEL_COMPOSITES:
                    check = _check_activation_per_channel
                else:
                    check = _check_activation
                round1.append((composite_name, pat, annotations, check))

        # Give each float reader of a shared activation output its own copy of the
        # activation, so Round 1 matches it without a tuple-output composite.
        mod = _unshare_shared_activations(mod)
        mod, n1, tuple_fields = _run_activation_patterns(mod, round1)

        # Round 1 hardswish composites with a shared SE-multiply output produce
        # Tuple([int8_hs, float32_hs]).  Downstream SE multiplies access float32_hs
        # via TupleGetItem which FuseOpsByPattern cannot trace through.  Inline those
        # TupleGetItem bindings so Round 2 sees dq(int8_hs) directly as the multiply
        # arg.
        mod = self._inline_tuple_getitems(mod, tuple_fields)

        # Round 2: channel_scale_multiply.  After Round 1 + TupleGetItem inlining,
        # all SE-block multiply instances expose dq(int8_hs)[1,C,H,W] × dq(int8_sig)
        # [1,C,1,1] → q directly, so FuseOpsByPattern can match them.
        mod, n2, _ = _run_activation_patterns(mod, round2)

        total = n1 + n2
        if total > 0:
            logger.info("FuseQDQToC7xActivation: fused %d activation ops", total)

        return mod


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xSiluF32Out")
class FuseQDQToC7xSiluF32Out:
    """Fuse the no-trailing-quantize self-gated SiLU composite (dq → sigmoid
    → multiply(self), float32 output) into c7x_int8_silu_f32out.

    Must run AFTER FuseQDQToC7xMovement, not as part of FuseQDQToC7xActivation's
    own Round 1 (where every other single-input activation pattern runs).
    FuseQDQToC7xMovement's FPN upsample-concat pattern matches this exact
    dq→sigmoid→multiply shape directly (as its branch-1 sub-structure feeding
    resize2d) and needs to see it raw. Running this pass before Movement
    would silently break Step 2's FPN fusion: confirmed by direct IR
    inspection -- c7x_int8_fpn_upsample_concat stopped appearing in the
    compiled yolo26n/yolov8n graphs entirely when this pattern ran first,
    because it greedily consumed the resize2d-feeding SiLU chains before
    Movement's own FuseOpsByPattern call ever saw them.

    This composite covers the C2f-block shape instead: a SiLU'd feature map
    that feeds a `split`/`concat` rather than a resize2d, which Movement's
    own patterns don't (and shouldn't) match. By running after Movement,
    this pass only ever sees what Movement's FPN pattern didn't already
    consume, plus everything that was never resize2d-adjacent to begin with.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat, annotations = _make_silu_f32out_pattern()
        entry = ("tidl_act.silu_f32out", pat, annotations, _check_silu_f32out)
        mod, count, _ = _run_activation_patterns(mod, [entry])
        if count > 0:
            logger.info("FuseQDQToC7xSiluF32Out: fused %d ops", count)
        return mod
