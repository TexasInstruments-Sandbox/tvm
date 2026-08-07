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
"""Fuse QDQ-wrapped concat into a vectorized int8 C kernel call.

Two independent pattern families share this file's registry/lowerer
infrastructure:

1. Channel-axis rescale concat -- the Inception-module pattern that
   dominates GoogleNet and InceptionV3:

     dq(x1, s1, z1)
     dq(x2, s2, z2)   → concat(axis=1) → q(s_out, z_out)
     dq(x3, s3, z3)
     ...

   After PT2E quantization each branch has its own calibrated scale, so
   EliminateQDQTransparent does not remove these QDQ wrappers (scales
   differ). Without this pass, TVM emits a full float32
   dequantize→concat→requantize scalar TIR loop running at ~100–200
   cycles/element. Replacement: call_extern to c7x_int8_concat_rescale,
   which uses SE streaming + Q13 integer fixed-point (~3–5 cycles/element)
   with a memcpy fast path when input and output scales are equal.

2. Last-axis dequantize+sigmoid concat -- the YOLO multi-scale class-score
   glue (detection heads concat per-scale class logits before a bare
   sigmoid, no self-multiply, no trailing quantize):

     dq(reshape(x1,_), s1, z1)
     dq(reshape(x2,_), s2, z2)   → concat(axis=-1) → sigmoid
     dq(reshape(x3,_), s3, z3)
     ...

   Replacement: call_extern to c7x_int8_concat_sigmoid (float32 output).

For both families, arity-specific patterns are registered (arity 2, 3, 4) so
the dequantize (and, for family 2, reshape) ops are inside each composite
function body. Using wildcard() for the Tuple arg would leave those ops
outside the composite (treated as boundary inputs), making scale/zp
extraction impossible without calling-function pre-scan.

Kernel: src/runtime/ti_dsp/kernels/c7x_concat.cpp
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, is_tuple, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import find_composite_span, propagate_span

logger = logging.getLogger(__name__)

_COMPOSITE_PREFIX = "c7x_concat."
_KERNEL_NAME = "c7x_int8_concat_rescale"
_SIGMOID_COMPOSITE_PREFIX = "c7x_concat_sigmoid."
_SIGMOID_KERNEL_NAME = "c7x_int8_concat_sigmoid"
_MAX_INPUTS = 4  # fixed kernel signature; arities above this are skipped


# =========================================================================
# Pattern factory
# =========================================================================


def _make_concat_pattern(n: int):
    """q(concat(Tuple([dq(x1,s1,z1), ..., dq(xn,sn,zn)]), axis=1), so, zo)

    Explicit is_op("relax.dequantize") nodes mean FuseOpsByPattern includes the
    dq ops inside the composite body, where the lowerer can extract scale/zp
    directly via param_to_arg — the same approach as _lower_channel_scale_multiply.
    """
    dq_patterns = []
    annotations = {}
    for i in range(1, n + 1):
        x, s, z = wildcard(), wildcard(), wildcard()
        dq_patterns.append(is_op("relax.dequantize")(x, s, z))
        annotations[f"in{i}"] = x
        annotations[f"s{i}"] = s
        annotations[f"z{i}"] = z

    cat = is_op("relax.concat")(is_tuple(dq_patterns))
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(cat, o_s, o_z)
    annotations["o_scale"] = o_s
    annotations["o_zp"] = o_z
    return q, annotations


def _make_concat_sigmoid_pattern(n: int):
    """sigmoid(concat(Tuple([dq(reshape(x1,_),s1,z1), ...]), axis=-1))

    Same wildcard-loop shape as _make_concat_pattern, but each branch has a
    reshape (int8 NCHW -> flat) between the raw input and the dequantize --
    the YOLO multi-scale class-score glue reshapes per-scale conv outputs to
    [1,C,n_i] before concatenating along the last axis. No trailing
    quantize: the composite root is the bare sigmoid itself.
    """
    dq_patterns = []
    annotations = {}
    for i in range(1, n + 1):
        x, s, z = wildcard(), wildcard(), wildcard()
        reshaped = is_op("relax.reshape")(x, wildcard())
        dq_patterns.append(is_op("relax.dequantize")(reshaped, s, z))
        annotations[f"in{i}"] = x
        annotations[f"s{i}"] = s
        annotations[f"z{i}"] = z

    cat = is_op("relax.concat")(is_tuple(dq_patterns))
    sig = is_op("relax.sigmoid")(cat)
    annotations["concat"] = cat
    return sig, annotations


def _check_dq_inputs(ctx, n: int) -> bool:
    """Shared check: all n dequantize inputs are int8 with constant scale/zp."""
    for i in range(1, n + 1):
        inp = ctx.annotated_expr.get(f"in{i}")
        if inp is None:
            return False
        if isinstance(inp, relax.Constant):
            if inp.data.dtype != "int8":
                return False
        elif hasattr(inp, "struct_info") and hasattr(inp.struct_info, "dtype"):
            if str(inp.struct_info.dtype) != "int8":
                return False
        else:
            return False
        for key in (f"s{i}", f"z{i}"):
            if not isinstance(ctx.annotated_expr.get(key), relax.Constant):
                return False
    return True


def _make_check_concat(n: int):
    """Check callback factory: verify all QDQ params are constants, inputs int8."""

    def _check(ctx) -> bool:
        if not isinstance(ctx.annotated_expr.get("o_scale"), relax.Constant):
            return False
        if not isinstance(ctx.annotated_expr.get("o_zp"), relax.Constant):
            return False
        return _check_dq_inputs(ctx, n)

    return _check


def _make_check_concat_sigmoid(n: int):
    """Check callback factory: int8 const-scale dq inputs, plus a rank-3
    last-axis concat.

    The axis/rank guards live here (not only in the lowerer) so a non-matching
    shape -- e.g. a channel-axis (axis=1) concat feeding a bare sigmoid -- is
    rejected before FuseOpsByPattern wraps it in a composite. Declining only
    in the lowerer would leave an un-lowered Composite function in the module,
    which a subsequent FuseOpsByPattern pass can choke on (see the design
    doc's bug 1).
    """

    def _check(ctx) -> bool:
        if not _check_dq_inputs(ctx, n):
            return False
        cat = ctx.annotated_expr.get("concat")
        if cat is None:
            return False
        sinfo = getattr(cat, "struct_info", None)
        if not isinstance(sinfo, relax.TensorStructInfo) or sinfo.shape is None:
            return False
        rank = len(sinfo.shape)
        if rank != 3:
            return False
        try:
            axis = int(cat.attrs.axis)
        except Exception:
            return False
        # Last-axis concat only (axis=-1 or the equivalent positive index).
        return axis in (-1, rank - 1)

    return _check


# Build the registry: one entry per supported arity, per pattern family.
_PATTERN_REGISTRY = [
    (_COMPOSITE_PREFIX + f"concat{n}", _make_concat_pattern(n), _make_check_concat(n))
    for n in range(2, _MAX_INPUTS + 1)
] + [
    (
        _SIGMOID_COMPOSITE_PREFIX + f"concat{n}",
        _make_concat_sigmoid_pattern(n),
        _make_check_concat_sigmoid(n),
    )
    for n in range(2, _MAX_INPUTS + 1)
]
_COMPOSITE_NAMES = frozenset(name for name, _, _ in _PATTERN_REGISTRY)
_SIGMOID_COMPOSITE_NAMES = frozenset(
    name for name in _COMPOSITE_NAMES if name.startswith(_SIGMOID_COMPOSITE_PREFIX)
)


# =========================================================================
# Lowerer
# =========================================================================


@mutator
class _ConcatLowerer(PyExprMutator):
    """Lower c7x_concat.concat{2,3,4} and c7x_concat_sigmoid.concat{2,3,4}
    composites to c7x_int8_concat_rescale / c7x_int8_concat_sigmoid."""

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
        composite_name = str(func.attrs["Composite"])
        if composite_name not in _COMPOSITE_NAMES:
            return super().visit_call_(call)
        if composite_name in _SIGMOID_COMPOSITE_NAMES:
            return self._lower_concat_sigmoid(call, func)
        return self._lower_concat(call, func)

    def _lower_concat(self, call, func):
        """Lower concat composite to c7x_int8_concat_rescale.

        The composite body contains dq(xi,si,zi) for each input plus the
        concat and output quantize — same structure as channel_scale_multiply.
        Walk bindings, collect dq_entries and o_scale/o_zp, then emit te.extern.
        """
        param_to_arg = dict(zip(func.params, call.args))

        dq_entries = []  # [(int8_input_arg, scale_val, zp_val)]
        o_scale_val = o_zp_val = None
        concat_axis = None

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

            elif "concat" in op_name:
                try:
                    concat_axis = int(val.attrs.axis)
                except Exception:
                    pass

            elif "quantize" in op_name and dq_entries:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if not dq_entries or o_scale_val is None:
            return super().visit_call_(call)

        # Only axis=1 (NCHW channel axis) is supported by the kernel.
        if concat_axis != 1:
            return super().visit_call_(call)

        # Output shape — NCHW expected.
        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)
        output_shape = [int(s) for s in call_sinfo.shape]
        if len(output_shape) != 4:
            return super().visit_call_(call)
        HW_v = output_shape[2] * output_shape[3]

        # Per-slot channel counts from each int8 input's struct_info.
        C_vals = []
        for inp, _, _ in dq_entries:
            if not hasattr(inp, "struct_info") or not hasattr(inp.struct_info, "shape"):
                return super().visit_call_(call)
            shape = [int(s) for s in inp.struct_info.shape]
            C_vals.append(shape[1])

        # Pad to _MAX_INPUTS slots; kernel skips slots with C_i=0.
        dummy = relax.const(np.zeros([1], dtype=np.int8))
        while len(dq_entries) < _MAX_INPUTS:
            dq_entries.append((dummy, 1.0, 0))
            C_vals.append(0)

        (arg0, s0_v, z0_v) = dq_entries[0]
        (arg1, s1_v, z1_v) = dq_entries[1]
        (arg2, s2_v, z2_v) = dq_entries[2]
        (arg3, s3_v, z3_v) = dq_entries[3]
        C0_v, C1_v, C2_v, C3_v = C_vals[0], C_vals[1], C_vals[2], C_vals[3]
        s_out_v = float(o_scale_val)
        z_out_v = int(o_zp_val)

        def te_concat_rescale(t0, t1, t2, t3):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    _KERNEL_NAME,
                    ins[0].data,
                    tir.IntImm("int32", C0_v),
                    tir.FloatImm("float32", s0_v),
                    tir.IntImm("int32", z0_v),
                    ins[1].data,
                    tir.IntImm("int32", C1_v),
                    tir.FloatImm("float32", s1_v),
                    tir.IntImm("int32", z1_v),
                    ins[2].data,
                    tir.IntImm("int32", C2_v),
                    tir.FloatImm("float32", s2_v),
                    tir.IntImm("int32", z2_v),
                    ins[3].data,
                    tir.IntImm("int32", C3_v),
                    tir.FloatImm("float32", s3_v),
                    tir.IntImm("int32", z3_v),
                    outs[0].data,
                    tir.IntImm("int32", HW_v),
                    tir.FloatImm("float32", s_out_v),
                    tir.IntImm("int32", z_out_v),
                )

            return te.extern(
                output_shape,
                [t0, t1, t2, t3],
                fcompute,
                name="concat_rescale_out",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_concat_rescale,
            arg0,
            arg1,
            arg2,
            arg3,
            primfunc_name_hint=_KERNEL_NAME,
        )
        self.count += 1
        n_active = sum(1 for c in C_vals if c > 0)
        logger.debug(
            "Fused %s: %d inputs C=[%d,%d,%d,%d] HW=%d s_out=%.6g",
            _KERNEL_NAME,
            n_active,
            C0_v,
            C1_v,
            C2_v,
            C3_v,
            HW_v,
            s_out_v,
        )
        return propagate_span(result, find_composite_span(func))

    def _lower_concat_sigmoid(self, call, func):
        """Lower concat_sigmoid composite to c7x_int8_concat_sigmoid.

        Each branch is dq(reshape(x,_),s,z) -- reshape sits between the raw
        input and the dequantize, so (unlike _lower_concat's plain
        dq(x,s,z) branches) the raw input tensor is captured at the
        *reshape* binding, where its own operand resolves directly through
        param_to_arg. The per-branch trailing width n_i and channel count
        C_i come from the dequantize binding's own struct_info (unaffected
        by dequantize, so identical to the reshape's target shape). The
        composite ends at a bare sigmoid, so there is no output quantize to
        scan for.
        """
        param_to_arg = dict(zip(func.params, call.args))

        bindings = func.body.blocks[0].bindings
        var_to_val = {b.var: b.value for b in bindings}

        reshape_src = {}  # reshape output Var -> raw input arg

        for binding in bindings:
            val = binding.value
            if not isinstance(val, relax.Call) or not hasattr(val.op, "name"):
                continue
            if "reshape" not in str(val.op.name):
                continue
            reshape_src[binding.var] = param_to_arg.get(val.args[0], val.args[0])

        dq_by_var = {}  # dq output Var -> (raw_input, n_i, C_i, scale_val, zp_val)
        concat_call = None
        concat_axis = None

        for binding in bindings:
            val = binding.value
            if not isinstance(val, relax.Call) or not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)

            if "dequantize" in op_name:
                raw_input = reshape_src.get(val.args[0])
                if raw_input is None:
                    return super().visit_call_(call)
                dq_sinfo = binding.var.struct_info
                if not isinstance(dq_sinfo, relax.TensorStructInfo) or not dq_sinfo.shape:
                    return super().visit_call_(call)
                dq_shape = [int(v) for v in dq_sinfo.shape]
                # [1, C, n_i] only: the kernel carries no batch dimension, so a
                # batch > 1 would silently read one batch and mis-stride the out.
                if len(dq_shape) != 3 or dq_shape[0] != 1:
                    return super().visit_call_(call)
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                dq_by_var[binding.var] = (
                    raw_input,
                    dq_shape[2],
                    dq_shape[1],
                    float(s.data.numpy()),
                    int(z.data.numpy()),
                )

            elif "concat" in op_name:
                concat_call = val
                try:
                    concat_axis = int(val.attrs.axis)
                except Exception:
                    pass

        if not dq_by_var or concat_call is None:
            return super().visit_call_(call)

        # Order branches by the concat's own tuple, not by binding-scan order,
        # so the anchor-axis concatenation order can't silently diverge from
        # the tuple if the bindings are ever reordered upstream.
        tuple_arg = concat_call.args[0]
        if isinstance(tuple_arg, relax.Var):
            tuple_arg = var_to_val.get(tuple_arg, tuple_arg)
        if not isinstance(tuple_arg, relax.Tuple):
            return super().visit_call_(call)
        dq_entries = []  # [(raw_input_arg, n_i, C_i, scale_val, zp_val)]
        for field in tuple_arg.fields:
            entry = dq_by_var.get(field)
            if entry is None:
                return super().visit_call_(call)
            dq_entries.append(entry)

        # Output shape -- [1, C, n0+n1+...] expected; concat must be on the
        # last axis (axis=-1 or the equivalent positive index).
        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)
        output_shape = [int(s) for s in call_sinfo.shape]
        if len(output_shape) != 3 or output_shape[0] != 1:
            return super().visit_call_(call)
        if concat_axis not in (-1, len(output_shape) - 1):
            return super().visit_call_(call)
        C_v = output_shape[1]

        # Every branch must share the same channel count C_v -- see
        # ti_fuse_qdq_c7x_movement.py's FPN lowerer post-review fix (C
        # derived from the wrong shape silently wrote past a buffer) for
        # why this is checked rather than assumed.
        if any(c_i != C_v for (_, _, c_i, _, _) in dq_entries):
            return super().visit_call_(call)
        if sum(n_i for (_, n_i, _, _, _) in dq_entries) != output_shape[-1]:
            return super().visit_call_(call)

        # Pad to _MAX_INPUTS slots; kernel skips slots with n_i=0.
        dummy = relax.const(np.zeros([1], dtype=np.int8))
        while len(dq_entries) < _MAX_INPUTS:
            dq_entries.append((dummy, 0, 0, 1.0, 0))

        (arg0, n0_v, _, s0_v, z0_v) = dq_entries[0]
        (arg1, n1_v, _, s1_v, z1_v) = dq_entries[1]
        (arg2, n2_v, _, s2_v, z2_v) = dq_entries[2]
        (arg3, n3_v, _, s3_v, z3_v) = dq_entries[3]

        def te_concat_sigmoid(t0, t1, t2, t3):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    _SIGMOID_KERNEL_NAME,
                    ins[0].data,
                    tir.IntImm("int32", n0_v),
                    tir.FloatImm("float32", s0_v),
                    tir.IntImm("int32", z0_v),
                    ins[1].data,
                    tir.IntImm("int32", n1_v),
                    tir.FloatImm("float32", s1_v),
                    tir.IntImm("int32", z1_v),
                    ins[2].data,
                    tir.IntImm("int32", n2_v),
                    tir.FloatImm("float32", s2_v),
                    tir.IntImm("int32", z2_v),
                    ins[3].data,
                    tir.IntImm("int32", n3_v),
                    tir.FloatImm("float32", s3_v),
                    tir.IntImm("int32", z3_v),
                    outs[0].data,
                    tir.IntImm("int32", C_v),
                )

            return te.extern(
                output_shape,
                [t0, t1, t2, t3],
                fcompute,
                name="concat_sigmoid_out",
                dtype="float32",
            )

        result = self.builder_.call_te(
            te_concat_sigmoid,
            arg0,
            arg1,
            arg2,
            arg3,
            primfunc_name_hint=_SIGMOID_KERNEL_NAME,
        )
        self.count += 1
        n_active = sum(1 for e in dq_entries if e[1] > 0)
        logger.debug(
            "Fused %s: %d inputs n=[%d,%d,%d,%d] C=%d",
            _SIGMOID_KERNEL_NAME,
            n_active,
            n0_v,
            n1_v,
            n2_v,
            n3_v,
            C_v,
        )
        return propagate_span(result, find_composite_span(func))


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xConcat")
class FuseQDQToC7xConcat:
    """Fuse QDQ-wrapped concat into a vectorized C kernel call.

    Two independent families, both arity 2/3/4:

    - Channel-axis (axis=1) rescale concat, ending in a trailing quantize:
      dq(x1,s1,z1) | ... -> concat(axis=1) -> q(s_out,z_out), lowered to
      c7x_int8_concat_rescale. Intercepts non-transparent concats that
      EliminateQDQTransparent cannot remove (input scales differ from
      output scale).
    - Last-axis (axis=-1) dequantize+sigmoid concat, no trailing quantize:
      dq(reshape(x1,_),s1,z1) | ... -> concat(axis=-1) -> sigmoid, lowered
      to c7x_int8_concat_sigmoid. The YOLO multi-scale class-score glue.

    Must run after EliminateQDQTransparent and before FuseQDQToInt8Conv2D.
    """

    @staticmethod
    def _run(mod: IRModule):
        patterns = [(name, pat, ann, check) for name, (pat, ann), check in _PATTERN_REGISTRY]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)
        lowerer = _ConcatLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()
        if lowerer.count > 0:
            mod = relax.transform.DeadCodeElimination()(mod)
        return mod, lowerer.count

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        mod, n = self._run(mod)
        if n > 0:
            logger.info("FuseQDQToC7xConcat: fused %d concat ops", n)
        return mod
