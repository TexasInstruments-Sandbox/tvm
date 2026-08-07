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
With ``bind_constants=False``, the matched scale/zero-point ``relax.Constant``
leaves are embedded directly inside the resulting Composite(+Primitive)-tagged
function body rather than hoisted to call-site parameters.

Every one of these passes' custom lowerers is expected to consume *every*
matched composite call (replacing it with a call_te/call_extern), after which
``DeadCodeElimination`` removes the now-unreferenced composite function. That
invariant matters: ``src/relax/transform/fuse_tir.cc``'s ``TIRFuseMutator``
fuses *every* Primitive-tagged GlobalVar still present in the module,
regardless of whether anything still calls it. A composite left un-consumed
(or merely inlined-but-not-deleted) still gets visited by FuseTIR, which
can't build a TIR PrimFunc from a function with an embedded ``relax.Constant``
-- it raises "Relax.Constant is not supported in primitive functions."

When a lowerer declines a match (e.g. because an operand turns out to be
compile-time-constant -- see ``ti_c7x_const_reachability.py``), it must not
simply leave the composite call in place. ``inline_declined_composite``
re-emits the composite's own body back into the caller, restoring exactly
the ungrouped ops FuseOpsByPattern started from, so ordinary
LegalizeOps/FuseOps/FuseTIR handle them (and their embedded constants) the
normal way. The caller is still responsible for making sure
``DeadCodeElimination`` actually runs afterward -- inlining only rewrites the
call site; the now-orphaned composite function must still be deleted, or
FuseTIR will visit it directly regardless of whether anything still calls it.
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
