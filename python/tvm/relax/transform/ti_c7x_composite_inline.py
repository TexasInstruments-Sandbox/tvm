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
"""Shared helper: undo a declined FuseOpsByPattern match.

The C7x QDQ-fusion passes (FuseQDQToC7xMovement, FuseQDQToC7xRelu, etc.) all
call ``relax.transform.FuseOpsByPattern(patterns, bind_constants=False)``.
``bind_constants=False`` maps to ``lift_constant_=true`` in
``src/relax/transform/fuse_ops.cc``, so matched ``relax.Constant`` leaves are
lifted to *parameters* of the Composite(+Primitive)-tagged function and passed
in at the call site; the composite body itself normally holds no
``relax.Constant``. (Verified on the shapes these patterns produce, including a
tensor Constant as the data operand and as a ``concat`` tuple field.)

Every one of these passes' custom lowerers is expected to consume *every*
matched composite call (replacing it with a call_te/call_extern), after which
``DeadCodeElimination`` removes the now-unreferenced composite function. That
invariant matters: ``src/relax/transform/fuse_tir.cc``'s ``TIRFuseMutator``
fuses *every* Primitive-tagged GlobalVar still present in the module,
regardless of whether anything still calls it. So a composite left un-consumed
(or merely inlined-but-not-deleted) makes compilation depend on a function
nothing calls -- it must legalize and fuse cleanly on its own, and FuseTIR
fatals outright on any ``relax.Constant`` that does reach a primitive body
("Relax.Constant is not supported in primitive functions"). A related
already-observed failure mode for surviving composites is scalar scale/zp
arriving as 0-d Buffer params rather than TIR literals -- see the comment at
``fuse_dequantize_matmul.py``'s DeadCodeElimination call.

When a lowerer declines a match (e.g. because an operand turns out to be
compile-time-constant -- see ``ti_c7x_const_reachability.py``), leaving the
composite call in place is therefore not obviously safe.
``inline_declined_composite`` re-emits the composite's own body back into the
caller, restoring exactly the ungrouped ops FuseOpsByPattern started from, so
ordinary LegalizeOps/FuseOps/FuseTIR handle them the normal way and no
orphaned Primitive function is left behind. The caller is still responsible
for making sure ``DeadCodeElimination`` actually runs afterward -- inlining
only rewrites the call site; the now-orphaned composite function must still be
deleted, or FuseTIR will visit it directly regardless of whether anything
still calls it.

Note on scope: this helper is defense-in-depth for that invariant. No model on
this tree has been shown to *require* it -- a declined composite left in place
was measured to compile cleanly through LegalizeOps/FoldConstant/FuseOps/
FuseTIR. The distinct, demonstrated hazard the decline branches themselves
exist for is in ``be39717c39``: lowering an all-constant match makes
FoldConstant host-JIT a C7x-only extern symbol and segfault.

The intermittent "Relax.Constant is not supported in primitive functions" crash
these passes were once suspected of causing came from somewhere else entirely:
a stale parameter index when ``FunctionCreator::CreateFunction`` in
``src/relax/transform/fuse_ops.cc`` spliced several partially-used tuple
parameters. Do not reach for a decline-path change when that error reappears
without first ruling that out.
"""

from tvm import relax


def inline_declined_composite(bb, call, func):
    """Re-emit ``func``'s body inline at the caller's current position.

    Substitutes ``call.args`` for ``func.params`` and re-emits every binding
    from ``func.body.blocks[0].bindings`` into ``bb``'s current block,
    returning the substituted output expression. Used to undo a
    FuseOpsByPattern match a lowerer has decided not to lower.
    """
    subst = dict(zip(func.params, call.args))
    for binding in func.body.blocks[0].bindings:
        new_value = _substitute(binding.value, subst)
        subst[binding.var] = bb.emit(new_value)
    return _substitute(func.body.body, subst)


def _substitute(expr, subst):
    """Rewrite `expr`, replacing any Var found in `subst` with its mapped
    value. Covers exactly the Expr kinds that appear in the flat,
    DFPattern-matched composite bodies these passes produce (mirrors
    ConstReachability.is_const's own Expr-kind coverage in
    ti_c7x_const_reachability.py); anything else passes through unchanged."""
    if isinstance(expr, relax.Var):
        return subst.get(expr, expr)
    if isinstance(expr, relax.Call):
        return relax.Call(
            expr.op,
            [_substitute(a, subst) for a in expr.args],
            expr.attrs,
            expr.sinfo_args,
            span=expr.span,
        )
    if isinstance(expr, relax.Tuple):
        return relax.Tuple([_substitute(f, subst) for f in expr.fields])
    if isinstance(expr, relax.TupleGetItem):
        return relax.TupleGetItem(_substitute(expr.tuple_value, subst), expr.index)
    return expr
