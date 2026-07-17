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
"""Fuse QDQ-wrapped channel-axis concat into a vectorized int8 C kernel call.

Targets the Inception-module pattern that dominates GoogleNet and InceptionV3:

  dq(x1, s1, z1)
  dq(x2, s2, z2)   → concat(axis=1) → q(s_out, z_out)
  dq(x3, s3, z3)
  ...

After PT2E quantization each branch has its own calibrated scale, so
EliminateQDQTransparent does not remove these QDQ wrappers (scales differ).
Without this pass, TVM emits a full float32 dequantize→concat→requantize
scalar TIR loop running at ~100–200 cycles/element.

Replacement: call_extern to c7x_int8_concat_rescale, which uses SE streaming
+ Q13 integer fixed-point (~3–5 cycles/element) with a memcpy fast path when
input and output scales are equal.

Three separate patterns are registered (arity 2, 3, 4) so the dequantize ops
are inside each composite function body.  Using wildcard() for the Tuple arg
would leave the dq ops outside the composite (treated as boundary inputs),
making scale/zp extraction impossible without calling-function pre-scan.

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

logger = logging.getLogger(__name__)

_COMPOSITE_PREFIX = "c7x_concat."
_KERNEL_NAME = "c7x_int8_concat_rescale"
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


def _make_check_concat(n: int):
    """Check callback factory: verify all QDQ params are constants, inputs int8."""

    def _check(ctx) -> bool:
        if not isinstance(ctx.annotated_expr.get("o_scale"), relax.Constant):
            return False
        if not isinstance(ctx.annotated_expr.get("o_zp"), relax.Constant):
            return False
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

    return _check


# Build the registry: one entry per supported arity.
_PATTERN_REGISTRY = [
    (_COMPOSITE_PREFIX + f"concat{n}", _make_concat_pattern(n), _make_check_concat(n))
    for n in range(2, _MAX_INPUTS + 1)
]
_COMPOSITE_NAMES = frozenset(name for name, _, _ in _PATTERN_REGISTRY)


# =========================================================================
# Lowerer
# =========================================================================


@mutator
class _ConcatLowerer(PyExprMutator):
    """Lower c7x_concat.concat{2,3,4} composites to c7x_int8_concat_rescale."""

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
        if str(func.attrs["Composite"]) not in _COMPOSITE_NAMES:
            return super().visit_call_(call)
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
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xConcat")
class FuseQDQToC7xConcat:
    """Fuse QDQ-wrapped channel-axis concat into c7x_int8_concat_rescale.

    Intercepts non-transparent concats that EliminateQDQTransparent cannot
    remove (input scales differ from output scale).  Handles arity 2, 3, 4.
    Only axis=1 (NCHW channel axis).

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
