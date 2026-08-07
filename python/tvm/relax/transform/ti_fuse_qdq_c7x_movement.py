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
"""Fuse QDQ-wrapped data-movement glue into vectorized int8 C kernel calls.

Targets the resize2d/split/concat/reshape "head-glue" that dominates YOLO
detection heads (see docs/dsp/quantized_model_optimization.md): after
EliminateQDQTransparent, movement ops whose surrounding dequantize/quantize
scales DIFFER (so they aren't eligible for QDQ elimination) fall through to
a scalar float32 TIR loop at ~100-200 cycles/element. This pass intercepts
two confirmed real patterns (found by dumping the compiled yolov8n/yolo26n
Relax IR just after EliminateQDQTransparent -- see the plan doc's Step 2
investigation) and replaces them with call_extern into
c7x_int8_rescale / c7x_int8_fpn_upsample_concat (src/runtime/ti_dsp/kernels/
c7x_rescale.cpp):

  1. dq(x) -> reshape -> q                     [reshape rescale]
     Reshape never reorders elements, so a flat Q13 rescale applies
     unchanged regardless of tensor rank.

  2. dq(x1) -> sigmoid -> multiply -> resize2d(nearest, exact 2x) -> A
     dq(x2) -> B
     concat([A, B], axis=1) -> q                [FPN upsample-concat]
     This is yolo26n's single largest layer profiled
     (fused_dequantize4_resize2d1_concatenate8_quantize20, 89.7M cycles,
     13% of the whole model) and the analogous v8 layer. Branch 1 (which
     gets upsampled) is a SiLU'd feature map (Ultralytics' Conv+SiLU block)
     upsampled 2x (nearest, half_pixel, round -- which collapses exactly to
     2x2 block replication, see c7x_rescale.h). Branch 2 (the skip
     connection) is a *bare* dequantize, not its own SiLU diamond -- despite
     both branches originating from a Conv+SiLU block in the source model,
     branch 2's SiLU output is independently consumed elsewhere in the
     graph too, so FuseQDQToC7xActivation has already matched and lowered
     it via that *other* site by the time this pass runs, leaving a plain
     dequantize (of that already-int8 SiLU result) feeding this concat --
     confirmed by direct inspection of the compiled graphs, not assumed;
     see the pattern's comment below for the full account. Lowered as ONE
     call_extern into c7x_int8_fpn_upsample_concat, which computes branch
     1's SiLU and rescales branch 2, both directly at the composite's
     output scale (no intermediate QDQ roundtrip), writing the
     upsample-replicated branch 1 and plain branch 2 straight into their
     channel ranges of the output buffer. A symmetric "both branches are
     SiLU diamonds" version of this pattern was tried first and reverted:
     it matches fine on a small synthetic graph, but never matches the
     real compiled graphs (that structure doesn't exist there once
     FuseQDQToC7xActivation has run) -- and separately, registering a
     two-independent-SiLU-diamond composite pattern like that crashes an
     unrelated *later* pass's own FuseOpsByPattern call on the real graphs
     ("Variable ... could not be found in any group", from TVM's
     OperatorFusor) even where it happens to match, a hazard this
     single-diamond shape also avoids. See c7x_rescale.h's kernel doc for
     the full note.

Only the exact 2x nearest_neighbor/half_pixel/round case is handled; any
other resize2d config falls through to the generic path unchanged.

Kernels: src/runtime/ti_dsp/kernels/c7x_rescale.cpp
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, is_tuple, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.transform.ti_c7x_composite_inline import inline_declined_composite
from tvm.relax.transform.ti_c7x_const_reachability import ConstReachability

logger = logging.getLogger(__name__)

_COMPOSITE_PREFIX = "c7x_movement."
_RESHAPE_NAME = _COMPOSITE_PREFIX + "reshape"
_RESIZE_SILU_CONCAT_NAME = _COMPOSITE_PREFIX + "resize_silu_concat2"


# =========================================================================
# Pattern 1: dq(x) -> reshape -> q
# =========================================================================


def _make_reshape_pattern():
    x, d_s, d_z = wildcard(), wildcard(), wildcard()
    dq = is_op("relax.dequantize")(x, d_s, d_z)
    reshaped = is_op("relax.reshape")(dq, wildcard())
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(reshaped, o_s, o_z)
    return q, {"x": x, "d_scale": d_s, "d_zp": d_z, "o_scale": o_s, "o_zp": o_z}


def _check_single_input(ctx) -> bool:
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
# Pattern 2: FPN upsample-concat
#   dq(x1) -> sigmoid -> multiply -> resize2d -> A
#   dq(x2) -> B
#   concat([A, B], axis=1) -> q
#
# Branch 2 (the skip branch) is a BARE dequantize, not its own
# sigmoid->multiply SiLU diamond -- confirmed by direct inspection of the
# compiled yolov8n/yolo26n graphs, not assumed. Both real FPN sites' branch
# 2 outputs are consumed in more than one place (the concat here, plus at
# least one other site elsewhere in the graph), so by the time this pass
# runs FuseQDQToC7xActivation has already matched branch 2's SiLU via that
# *other* site (whose pattern requires a direct trailing quantize) and
# lowered it to c7x_int8_silu there, leaving a plain
# dequantize(that already-int8 result) feeding this concat -- see
# _ActivationLowerer._lower_single_input's "is_tuple_out" companion-
# dequantize handling in ti_fuse_qdq_c7x_activation.py, which this is a
# generic consequence of, not a bug. A naively-symmetric "branch 2 is also
# a SiLU diamond" pattern was tried first: it matched fine in isolation on
# a small synthetic graph, but never matches the real compiled graphs (0
# hits) because that structure doesn't exist there after
# FuseQDQToC7xActivation runs -- and, separately, registering a
# two-independent-SiLU-diamond pattern like that crashed TVM's
# FuseOpsByPattern grouping on the real graphs regardless ("Variable ...
# could not be found in any group" from OperatorFusor), which is *also*
# avoided by this single-diamond shape.
# =========================================================================


def _make_resize_silu_concat_pattern():
    x1, s1, z1 = wildcard(), wildcard(), wildcard()
    dq1 = is_op("relax.dequantize")(x1, s1, z1)
    sig1 = is_op("relax.sigmoid")(dq1)
    mul1 = is_op("relax.multiply")(dq1, sig1)
    resized = is_op("relax.image.resize2d")(mul1, wildcard())

    x2, s2, z2 = wildcard(), wildcard(), wildcard()
    dq2 = is_op("relax.dequantize")(x2, s2, z2)

    cat = is_op("relax.concat")(is_tuple([resized, dq2]))
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(cat, o_s, o_z)
    return q, {
        "x1": x1,
        "s1": s1,
        "z1": z1,
        "x2": x2,
        "s2": s2,
        "z2": z2,
        "o_scale": o_s,
        "o_zp": o_z,
    }


def _check_resize_silu_concat(ctx) -> bool:
    for inp_name in ("x1", "x2"):
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
    return True


_PATTERN_REGISTRY = [
    (_RESHAPE_NAME, _make_reshape_pattern, _check_single_input),
    (_RESIZE_SILU_CONCAT_NAME, _make_resize_silu_concat_pattern, _check_resize_silu_concat),
    # No commuted (tuple-field-order-swapped) variant: registering it
    # alongside the canonical pattern crashes TVM's FuseOpsByPattern
    # grouping on real (many-concat) graphs -- confirmed via direct testing
    # against compiled yolov8n/yolo26n ("Variable ... could not be found in
    # any group" from OperatorFusor) -- even though both patterns work fine
    # individually on small synthetic graphs. Not needed for real models
    # either: both confirmed FPN upsample sites always put the resize2d
    # branch first (concat field 0). A reversed-order concat simply falls
    # through to the generic path unfused, which is safe.
]
_COMPOSITE_NAMES = frozenset(name for name, _, _ in _PATTERN_REGISTRY)


# =========================================================================
# Lowerer
# =========================================================================


@mutator
class _MovementLowerer(PyExprMutator):
    """Lower c7x_movement.* composites to call_extern chains."""

    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.count = 0
        self.touched = False
        self._const_reach = ConstReachability(mod)

    def _decline(self, call, func):
        """Undo a declined match: inline the composite's own body back into
        the caller. Also marks the module as touched so the now-orphaned
        composite function is actually deleted by _run()'s
        DeadCodeElimination() -- FuseTIR fuses every Primitive-tagged
        function still in the module regardless of whether anything calls
        it, so an un-deleted orphan is just as dangerous as an un-inlined
        call; see ti_c7x_composite_inline.py."""
        self.touched = True
        return inline_declined_composite(self.builder_, call, func)

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
        if name == _RESHAPE_NAME:
            return self._lower_reshape(call, func)
        return self._lower_resize_silu_concat(call, func)

    # -- Pattern 1: reshape --------------------------------------------

    def _lower_reshape(self, call, func):
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = None
        d_scale_val = d_zp_val = o_scale_val = o_zp_val = None
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
            elif "quantize" in op_name and x_arg is not None:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if x_arg is None or o_scale_val is None:
            return self._decline(call, func)

        # A compile-time-constant x_arg (e.g. Swin's relative-position-bias
        # table: a constant table gathered by a constant index) would later
        # be evaluated eagerly by FoldConstant via a host LLVM JIT, which
        # can't resolve this call_extern's C7x-only symbol and segfaults
        # instead of raising -- see ti_c7x_const_reachability.py. Inline the
        # composite's own body back into the caller rather than leaving the
        # call in place: FuseTIR fuses every Primitive-tagged function still
        # in the module regardless of whether anything calls it, so merely
        # declining isn't enough -- see ti_c7x_composite_inline.py.
        if self._const_reach.is_const(x_arg):
            return self._decline(call, func)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return self._decline(call, func)
        output_shape = [int(s) for s in call_sinfo.shape]
        n_elem = 1
        for s in output_shape:
            n_elem *= s

        d_zp_v = int(d_zp_val)  # type: ignore[arg-type]
        d_scale_v = float(d_scale_val)  # type: ignore[arg-type]
        o_zp_v = int(o_zp_val)  # type: ignore[arg-type]
        o_scale_v = float(o_scale_val)
        n_v = n_elem

        def te_rescale(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_rescale",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", n_v),
                    tir.IntImm("int32", d_zp_v),
                    tir.FloatImm("float32", d_scale_v),
                    tir.IntImm("int32", o_zp_v),
                    tir.FloatImm("float32", o_scale_v),
                )

            return te.extern(
                output_shape, [x_t], fcompute, name="movement_rescale_out", dtype="int8"
            )

        result = self.builder_.call_te(te_rescale, x_arg, primfunc_name_hint="c7x_int8_rescale")
        self.count += 1
        self.touched = True
        logger.debug(
            "Fused c7x_int8_rescale (reshape): n=%d d_zp=%d d_scale=%.6g o_zp=%d o_scale=%.6g",
            n_elem,
            d_zp_v,
            d_scale_v,
            o_zp_v,
            o_scale_v,
        )
        return result

    # -- Pattern 2: FPN upsample-concat ----------------------------------

    def _lower_resize_silu_concat(self, call, func):
        param_to_arg = dict(zip(func.params, call.args))
        bindings_by_var = {}
        for binding in func.body.blocks[0].bindings:
            bindings_by_var[binding.var] = binding.value

        dq_entries = {}  # bound_var -> (arg, scale_val, zp_val)
        resize_call = None
        resize_out_var = None
        concat_fields = None
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
                dq_entries[binding.var] = (inp, float(s.data.numpy()), int(z.data.numpy()))
            elif op_name == "relax.image.resize2d":
                resize_call = val
                resize_out_var = binding.var
            elif op_name == "relax.concat":
                # concat's tuple argument is typically an inline Tuple
                # expression (not a separately-bound Tuple var).
                tup = val.args[0]
                if isinstance(tup, relax.Var):
                    tup = bindings_by_var.get(tup, tup)
                if isinstance(tup, relax.Tuple):
                    concat_fields = list(tup.fields)
            elif "quantize" in op_name and "de" not in op_name:
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                o_scale_val = float(s.data.numpy())
                o_zp_val = int(z.data.numpy())

        if resize_call is None or concat_fields is None or o_scale_val is None:
            return self._decline(call, func)
        if len(dq_entries) != 2:
            return self._decline(call, func)

        # Validate resize2d is the exact 2x nearest/half_pixel/round case
        # this kernel implements -- anything else falls through unchanged.
        rattrs = resize_call.attrs
        if (
            str(rattrs.method) != "nearest_neighbor"
            or str(rattrs.coordinate_transformation_mode) != "half_pixel"
            or str(rattrs.rounding_method) != "round"
            or str(rattrs.layout) != "NCHW"
        ):
            return self._decline(call, func)

        in_shape = resize_call.args[0].struct_info.shape
        out_shape = resize_call.struct_info.shape
        if in_shape is None or out_shape is None or len(in_shape) != 4 or len(out_shape) != 4:
            return self._decline(call, func)
        in_shape = [int(s) for s in in_shape]
        out_shape = [int(s) for s in out_shape]
        if out_shape[0] != in_shape[0] or out_shape[1] != in_shape[1]:
            return self._decline(call, func)
        if out_shape[2] != 2 * in_shape[2] or out_shape[3] != 2 * in_shape[3]:
            return self._decline(call, func)

        # Trace resize2d's input (a multiply/SiLU output var) back to the
        # dequantize that feeds it -- that dequantize's entry is branch 1.
        mul_var = resize_call.args[0]
        mul_call = bindings_by_var.get(mul_var)
        if not isinstance(mul_call, relax.Call) or not isinstance(mul_call.args[0], relax.Var):
            return self._decline(call, func)
        dq1_var = mul_call.args[0]
        if dq1_var not in dq_entries:
            return self._decline(call, func)
        branch1_arg, s1_v, z1_v = dq_entries[dq1_var]
        (dq2_var,) = [v for v in dq_entries if v != dq1_var]
        branch2_arg, s2_v, z2_v = dq_entries[dq2_var]

        # Same all-constant hazard as _lower_reshape above: a
        # compile-time-constant branch would make this call_tir
        # all-constant, and the pipeline's later FoldConstant pass would
        # segfault trying to JIT-fold the unresolvable DSP-only extern
        # symbol. Inline the composite's own body back into the caller
        # rather than leaving the call in place: FuseTIR fuses every
        # Primitive-tagged function still in the module regardless of
        # whether anything calls it, so merely declining isn't enough --
        # see ti_c7x_composite_inline.py.
        if self._const_reach.is_const(branch1_arg) or self._const_reach.is_const(branch2_arg):
            return self._decline(call, func)

        # The kernel always places branch 1 (the resize2d branch) first in
        # the output channel order -- only the canonical tuple order
        # (resize2d branch first) is supported; see _PATTERN_REGISTRY's
        # comment on why no commuted variant is registered.
        if len(concat_fields) != 2 or concat_fields[0] != resize_out_var:
            return self._decline(call, func)

        N, C1, H, W = in_shape
        if N != 1:
            return self._decline(call, func)

        # Branch 2's channel count MUST come from branch 2's own tensor, not
        # from out_shape (the resize2d output): resize2d preserves channels,
        # so out_shape[1] == in_shape[1] == C1 (branch 1's count, which the
        # pass already asserts equal above). Using out_shape[1] here would
        # tell the kernel branch 2 has C1 channels -- reading/writing past
        # both the branch-2 input and the (C1+C2)-channel output whenever the
        # two branches differ in width. Branch 2 is already at the upsampled
        # (2H, 2W) spatial size.
        b2_sinfo = getattr(branch2_arg, "struct_info", None)
        if not isinstance(b2_sinfo, relax.TensorStructInfo) or not b2_sinfo.shape:
            return self._decline(call, func)
        b2_shape = [int(s) for s in b2_sinfo.shape]
        if (
            len(b2_shape) != 4
            or b2_shape[0] != 1
            or b2_shape[2] != 2 * H
            or b2_shape[3] != 2 * W
        ):
            return self._decline(call, func)
        C2 = b2_shape[1]

        call_sinfo = call.struct_info

        # branch 1's SiLU output (pre-resize2d, [1,C1,H,W] float32) may be
        # independently consumed elsewhere in the graph too -- if so,
        # FuseOpsByPattern promotes it to a second tuple field on this
        # composite's call, the same "is_tuple_out" situation
        # _ActivationLowerer._lower_single_input handles for hardswish.
        # Confirmed on both real FPN sites this backs (yolov8n/yolo26n),
        # not assumed -- see the pattern's comment above and
        # c7x_rescale.h's c7x_int8_fpn_upsample_concat_ex doc.
        is_tuple_out = isinstance(call_sinfo, relax.TupleStructInfo)
        if is_tuple_out:
            int8_idx = next(
                (
                    i
                    for i, f in enumerate(call_sinfo.fields)
                    if isinstance(f, relax.TensorStructInfo) and str(f.dtype) == "int8"
                ),
                None,
            )
            if int8_idx is None:
                return self._decline(call, func)
            int8_sinfo = call_sinfo.fields[int8_idx]
            float_idx = 1 - int8_idx if len(call_sinfo.fields) == 2 else None
            if float_idx is None:
                return self._decline(call, func)
            float_sinfo = call_sinfo.fields[float_idx]
            if not isinstance(float_sinfo, relax.TensorStructInfo) or not float_sinfo.shape:
                return self._decline(call, func)
            float_shape = [int(s) for s in float_sinfo.shape]
            # The shared value is branch 1's SiLU output *before* resize2d:
            # must match [1, C1, H, W] exactly, not just be some float32
            # tensor -- otherwise this isn't the situation we know how to
            # reconstruct and we should decline rather than guess.
            if float_shape != [1, C1, H, W]:
                return self._decline(call, func)
            if not int8_sinfo.shape:
                return self._decline(call, func)
            output_shape = [int(s) for s in int8_sinfo.shape]
        elif not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return self._decline(call, func)
        else:
            output_shape = [int(s) for s in call_sinfo.shape]
        if len(output_shape) != 4:
            return self._decline(call, func)
        # The concat output must be exactly C1 + C2 channels; if the branch
        # decomposition doesn't add up, decline rather than emit a kernel
        # call that would over-read/over-write either buffer.
        if output_shape[1] != C1 + C2:
            return self._decline(call, func)

        o_scale_v = float(o_scale_val)
        o_zp_v = int(o_zp_val)  # type: ignore[arg-type]
        C1_v, H_v, W_v = int(C1), int(H), int(W)
        C2_v = int(C2)
        s1_vv, z1_vv, s2_vv, z2_vv = float(s1_v), int(z1_v), float(s2_v), int(z2_v)

        if is_tuple_out:

            def te_fpn_ex(t1, t2):
                def fcompute(ins, outs):
                    return tir.call_extern(
                        "int32",
                        "c7x_int8_fpn_upsample_concat_ex",
                        ins[0].data,
                        tir.IntImm("int32", C1_v),
                        tir.IntImm("int32", H_v),
                        tir.IntImm("int32", W_v),
                        tir.IntImm("int32", z1_vv),
                        tir.FloatImm("float32", s1_vv),
                        ins[1].data,
                        tir.IntImm("int32", C2_v),
                        tir.IntImm("int32", z2_vv),
                        tir.FloatImm("float32", s2_vv),
                        outs[0].data,
                        tir.FloatImm("float32", o_scale_v),
                        tir.IntImm("int32", o_zp_v),
                        outs[1].data,
                    )

                return te.extern(
                    [output_shape, [C1_v, H_v, W_v]],
                    [t1, t2],
                    fcompute,
                    name="movement_fpn_out",
                    dtype=["int8", "float32"],
                )

            fpn_result = self.builder_.emit(
                self.builder_.call_te(
                    te_fpn_ex,
                    branch1_arg,
                    branch2_arg,
                    primfunc_name_hint="c7x_int8_fpn_upsample_concat_ex",
                )
            )
            int8_result = self.builder_.emit(relax.TupleGetItem(fpn_result, 0))
            presize_f32 = self.builder_.emit(relax.TupleGetItem(fpn_result, 1))
            # Branch 1's promoted companion is its pre-upsample SiLU output,
            # a float32 tensor. The _ex kernel emits it directly as the exact
            # float32 SiLU value (its second output), so it can be used as-is
            # after reshaping to the promoted field's [1,C1,H,W] rank -- no
            # quantize->dequantize roundtrip, so the other consumer of that
            # value sees the true SiLU result, not one carrying this
            # composite's output-scale quantization error. (te.extern's
            # second output above is the flat per-buffer shape [C1,H,W],
            # matching the rank convention used elsewhere in this file.)
            float32_result = self.builder_.emit(
                relax.op.reshape(presize_f32, [1, C1_v, H_v, W_v])
            )
            out_fields = [None, None]
            out_fields[int8_idx] = int8_result
            out_fields[float_idx] = float32_result
            result = self.builder_.emit(relax.Tuple(out_fields))
        else:

            def te_fpn(t1, t2):
                def fcompute(ins, outs):
                    return tir.call_extern(
                        "int32",
                        "c7x_int8_fpn_upsample_concat",
                        ins[0].data,
                        tir.IntImm("int32", C1_v),
                        tir.IntImm("int32", H_v),
                        tir.IntImm("int32", W_v),
                        tir.IntImm("int32", z1_vv),
                        tir.FloatImm("float32", s1_vv),
                        ins[1].data,
                        tir.IntImm("int32", C2_v),
                        tir.IntImm("int32", z2_vv),
                        tir.FloatImm("float32", s2_vv),
                        outs[0].data,
                        tir.FloatImm("float32", o_scale_v),
                        tir.IntImm("int32", o_zp_v),
                    )

                return te.extern(
                    output_shape, [t1, t2], fcompute, name="movement_fpn_out", dtype="int8"
                )

            result = self.builder_.call_te(
                te_fpn,
                branch1_arg,
                branch2_arg,
                primfunc_name_hint="c7x_int8_fpn_upsample_concat",
            )

        self.count += 1
        self.touched = True
        logger.debug(
            "Fused FPN upsample-concat: C1=%d C2=%d H=%d W=%d -> H2=%d W2=%d tuple_out=%s",
            C1,
            C2,
            H,
            W,
            2 * H,
            2 * W,
            is_tuple_out,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xMovement")
class FuseQDQToC7xMovement:
    """Fuse QDQ-wrapped data-movement glue into c7x_int8_rescale /
    c7x_int8_fpn_upsample_concat call_extern.

    Handles: dq->reshape->q (any rank), and the SiLU'd FPN
    resize2d(nearest,2x)+concat(axis=1) upsample-skip pattern. Must run
    after EliminateQDQTransparent and before FuseQDQToInt8Conv2D, beside
    the other FuseQDQToC7x* passes.
    """

    @staticmethod
    def _run(mod: IRModule):
        patterns = [(name, factory(), check) for name, factory, check in _PATTERN_REGISTRY]
        patterns = [(name, pat, ann, check) for name, (pat, ann), check in patterns]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)
        lowerer = _MovementLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()
        if lowerer.touched:
            mod = relax.transform.DeadCodeElimination()(mod)
            # Flatten any Var->Var chains left by chaining multiple call_te
            # emissions inside one composite lowering (silu -> resize2d ->
            # silu -> concat_rescale). Without this, a later pass's own
            # FuseOpsByPattern call (e.g. FuseQDQToC7xActivation, if it runs
            # after this pass on an unrelated part of the same graph) can
            # crash with "Variable ... could not be found in any group" --
            # the same false-cycle hazard _inline_tuple_getitems documents
            # in ti_fuse_qdq_c7x_activation.py, confirmed by direct testing
            # against compiled yolov8n/yolo26n.
            mod = relax.transform.CanonicalizeBindings()(mod)
        return mod, lowerer.count

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        mod, n = self._run(mod)
        if n > 0:
            logger.info("FuseQDQToC7xMovement: fused %d movement ops", n)
        return mod
