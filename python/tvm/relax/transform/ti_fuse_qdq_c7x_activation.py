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

Kernels: src/runtime/ti_dsp/kernels/c7x_activation.c
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

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
_PATTERN_REGISTRY = [
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


def _check_silu_f32out(ctx) -> bool:
    """Same as _check_activation, minus the o_scale/o_zp check -- this
    pattern has no trailing quantize, so there is no output scale/zp to
    validate."""
    x = ctx.annotated_expr["x"]
    if isinstance(x, relax.Constant):
        if x.data.dtype != "int8":
            return False
    elif hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype"):
        if str(x.struct_info.dtype) != "int8":
            return False
    else:
        return False

    for name in ("d_scale", "d_zp"):
        if not isinstance(ctx.annotated_expr[name], relax.Constant):
            return False
    return True


def _check_channel_scale_multiply(ctx) -> bool:
    """Require two int8 inputs, compile-time QDQ constants, and one input with
    trailing [1,1] spatial dims (excitation) and the other with H×W > 1."""
    for inp_name in ("in1", "in2"):
        inp = ctx.annotated_expr[inp_name]
        if isinstance(inp, relax.Constant):
            if inp.data.dtype != "int8":
                return False
        elif hasattr(inp, "struct_info") and hasattr(inp.struct_info, "dtype"):
            if str(inp.struct_info.dtype) != "int8":
                return False
        else:
            return False

    for name in ("s1", "z1", "s2", "z2", "o_scale", "o_zp"):
        if not isinstance(ctx.annotated_expr[name], relax.Constant):
            return False

    # Verify shape constraint: exactly one input has trailing [1, 1] dims.
    in1 = ctx.annotated_expr["in1"]
    in2 = ctx.annotated_expr["in2"]
    in1_si = getattr(in1, "struct_info", None)
    in2_si = getattr(in2, "struct_info", None)
    if in1_si is None or in2_si is None:
        return False
    in1_shape = getattr(in1_si, "shape", None)
    in2_shape = getattr(in2_si, "shape", None)
    if in1_shape is None or in2_shape is None:
        return False
    if len(in1_shape) < 4 or len(in2_shape) < 4:
        return False
    for s in [*in1_shape, *in2_shape]:
        if not isinstance(s, tir.IntImm):
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
        return self._lower_single_input(call, func, name)

    def _lower_silu_f32out(self, call, call_sinfo, x_arg, d_scale_val, d_zp_val):
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
        if isinstance(call_sinfo, relax.TupleStructInfo):
            return super().visit_call_(call)
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
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
                return tir.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", n),
                    tir.IntImm("int32", d_zp_v),
                    tir.FloatImm("float32", d_scale_v),
                )

            return te.extern(output_shape, [x_t], fcompute, name="tidl_act_f32out", dtype="float32")

        result = self.builder_.call_te(te_silu_f32out, x_arg, primfunc_name_hint=extern_name)
        self.count += 1
        logger.debug(
            "Fused %s: n=%d d_zp=%d d_scale=%.6g", extern_name, n_elem, d_zp_v, d_scale_v
        )
        return result

    def _lower_single_input(self, call, func, composite_name):
        """Lower single-input activation composites (gelu/silu/hardsigmoid/hardswish/
        silu_f32out)."""
        param_to_arg = dict(zip(func.params, call.args))
        is_f32out = composite_name in _F32OUT_NAMES

        x_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None

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
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if x_arg is None or d_scale_val is None:
            return super().visit_call_(call)
        if not is_f32out and o_scale_val is None:
            return super().visit_call_(call)

        call_sinfo = call.struct_info

        if is_f32out:
            return self._lower_silu_f32out(call, call_sinfo, x_arg, d_scale_val, d_zp_val)

        # Determine output shape.  For hardswish layers whose float32 intermediate
        # is shared with an SE-block multiply, FuseOpsByPattern creates a
        # tuple-output composite: R.Tuple(int8_tensor, float32_tensor).
        # We emit c7x_int8_hardswish for the int8 part and re-dequantize it
        # to recover the float32 part for the downstream SE multiply.
        is_tuple_out = isinstance(call_sinfo, relax.TupleStructInfo)
        if is_tuple_out:
            # Find the int8 field.
            int8_idx = next(
                (
                    i
                    for i, f in enumerate(call_sinfo.fields)
                    if isinstance(f, relax.TensorStructInfo) and str(f.dtype) == "int8"
                ),
                None,
            )
            if int8_idx is None:
                return super().visit_call_(call)
            int8_sinfo = call_sinfo.fields[int8_idx]
            if not int8_sinfo.shape:
                return super().visit_call_(call)
            output_shape = [int(s) for s in int8_sinfo.shape]
        elif not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
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
                return tir.call_extern(
                    "int32",
                    extern_name,
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", n),
                    tir.IntImm("int32", d_zp_v),
                    tir.FloatImm("float32", d_scale_v),
                    tir.IntImm("int32", o_zp_v),
                    tir.FloatImm("float32", o_scale_v),
                )

            return te.extern(output_shape, [x_t], fcompute, name="tidl_act_out", dtype="int8")

        int8_result = self.builder_.call_te(te_activation, x_arg, primfunc_name_hint=extern_name)
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
                dq_entries.append((inp, float(s.data.numpy()), int(z.data.numpy())))
            elif "quantize" in op_name and dq_entries:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if len(dq_entries) < 2 or o_scale_val is None:
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)

        # Determine which dq input is excitation [1,C,1,1] and which is feature map.
        (arg0, s0, z0), (arg1, s1, z1) = dq_entries[0], dq_entries[1]
        shape0 = [int(x) for x in arg0.struct_info.shape]
        shape1 = [int(x) for x in arg1.struct_info.shape]

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
                return tir.call_extern(
                    "int32",
                    "c7x_int8_channel_scale_multiply",
                    ins[0].data,
                    ins[1].data,
                    outs[0].data,
                    tir.IntImm("int32", C_v),
                    tir.IntImm("int32", H_W_v),
                    tir.FloatImm("float32", s_exc_v),
                    tir.IntImm("int32", z_exc_v),
                    tir.FloatImm("float32", s_fm_v),
                    tir.IntImm("int32", z_fm_v),
                    tir.FloatImm("float32", s_out_v),
                    tir.IntImm("int32", z_out_v),
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
        return result

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
                d_scale_val = float(s.data.numpy())
                d_zp_val = int(z.data.numpy())
            elif op_name == "relax.permute_dims":
                axes = val.attrs["axes"] if val.attrs else None
                permute_axes = [int(a) for a in axes] if axes is not None else None
            elif op_name == "relax.nn.softmax":
                softmax_axis = int(val.attrs["axis"])
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if x_arg is None or d_scale_val is None or o_scale_val is None:
            return super().visit_call_(call)

        x_sinfo = getattr(x_arg, "struct_info", None)
        if not isinstance(x_sinfo, relax.TensorStructInfo) or not x_sinfo.shape:
            return super().visit_call_(call)
        pre_shape = list(x_sinfo.shape)
        if len(pre_shape) != 4 or any(not isinstance(s, tir.IntImm) for s in pre_shape):
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
                return tir.call_extern(
                    "int32",
                    "c7x_int8_dfl_softmax",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", B_v),
                    tir.IntImm("int32", A_v),
                    tir.IntImm("int32", K_v),
                    tir.IntImm("int32", N_v),
                    tir.IntImm("int32", d_zp_v),
                    tir.FloatImm("float32", d_scale_v),
                    tir.IntImm("int32", o_zp_v),
                    tir.FloatImm("float32", o_scale_v),
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
        return result


# =========================================================================
# Public pass
# =========================================================================


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
                        relax.Call(bval.op, new_args, bval.attrs, bval.sinfo_args)
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
                check = _check_activation
                round1.append((composite_name, pat, annotations, check))

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
