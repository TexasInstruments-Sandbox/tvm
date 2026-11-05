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
"""Fold constant-only chains of shape/index ops before TIDL partitioning.

``FoldConstant`` evaluates a call only once it is a ``call_tir``; the high-level ops
(``arange``, ``reshape``, ``broadcast_to``, ``expand_dims``, ...) are left alone.  A detection
head builds its anchor-point table from such ops with no dependence on the model input, so the
TIDL partitioner saw a long chain of live ops and offloaded it: wasted TIDL work for what is
a constant, and the subgraph TIDL's importer then rejects.

``LegalizeOps`` cannot be used for this: it works per op name, so it would also lower the
model's own reshapes and concats, which the TIDL patterns need in their high-level form.
:func:`fold_constant_only_chains` instead calls each op's legalization function on the
constant-only bindings alone, then lets ``FoldConstant`` evaluate the resulting ``call_tir``
calls and ``DeadCodeElimination`` drop what is left.
"""

import logging
from typing import Dict, Optional, Set

import tvm
from tvm import relax, tirx
from tvm.ir import GenericConst, IRModule, is_prim_expr
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

#: Ops that are folded when every operand is itself foldable.  Limited to ops that only move or
#: index data, or do elementwise arithmetic on small tables; reductions and anything stateful are
#: deliberately absent.
FOLDABLE_OPS = frozenset(
    {
        "relax.arange",
        "relax.add",
        "relax.subtract",
        "relax.multiply",
        "relax.astype",
        "relax.reshape",
        "relax.broadcast_to",
        "relax.expand_dims",
        "relax.concat",
        "relax.permute_dims",
        "relax.strided_slice",
    }
)

#: Do not materialise a constant larger than this many elements.
MAX_FOLDED_ELEMENTS = 1 << 20


def _numel(ty) -> Optional[int]:
    """Element count of a tensor type with a fully static shape, else None."""
    shape = getattr(ty, "shape", None)
    if not isinstance(ty, relax.TensorType) or shape is None:
        return None
    n = 1
    for d in shape:
        if not isinstance(d, tirx.IntImm):
            return None
        n *= int(d)
    return n


def _foldable_bindings(func: relax.Function) -> Set[relax.Var]:
    """Bindings of *func* whose value is a constant computed only from constants."""
    if not isinstance(func.body, relax.SeqExpr):
        return set()
    params = set(func.params)
    value_of: Dict[relax.Var, relax.Expr] = {}
    order = []
    for block in func.body.blocks:
        for b in block.bindings:
            if isinstance(b, relax.VarBinding):
                value_of[b.var] = b.value
                order.append(b.var)

    const_leaf: Set[relax.Var] = set()  # bound directly to a constant
    foldable: Set[relax.Var] = set()

    def operand_ok(e) -> bool:
        if isinstance(e, relax.Var):
            return e in foldable or e in const_leaf
        if isinstance(e, relax.Tuple):  # concat takes its operands as one Tuple
            return all(operand_ok(f) for f in e.fields)
        if isinstance(e, relax.ShapeExpr) or is_prim_expr(e):
            return True
        return isinstance(e, GenericConst)

    for v in order:
        val = value_of[v]
        if isinstance(val, GenericConst):
            const_leaf.add(v)
            continue
        if not isinstance(val, relax.Call) or getattr(val.op, "name", None) not in FOLDABLE_OPS:
            continue
        if v in params:
            continue
        n = _numel(v.ty)
        if n is None or n > MAX_FOLDED_ELEMENTS:
            continue
        if all(operand_ok(a) for a in val.args):
            foldable.add(v)
    return foldable


@mutator
class _LegalizeFoldable(PyExprMutator):
    """Lower the selected bindings with the registered per-op legalization function."""

    def __init__(self, mod: IRModule, targets: Set[relax.Var]):
        super().__init__(mod)
        self._targets = targets
        self._current: Optional[relax.Var] = None
        self.lowered = 0

    def visit_var_binding_(self, binding):  # pyright: ignore[reportIncompatibleMethodOverride]
        self._current = binding.var
        super().visit_var_binding_(binding)
        self._current = None

    def visit_call_(self, call):  # pyright: ignore[reportIncompatibleMethodOverride]
        visited = super().visit_call_(call)
        if self._current is None or self._current not in self._targets:
            return visited
        op = getattr(visited, "op", None)
        legalize = op.get_attr("FLegalize") if isinstance(op, tvm.ir.Op) else None
        if legalize is None:
            return visited
        self.lowered += 1
        return legalize(self.builder_, visited)


def fold_constant_only_chains(mod: IRModule) -> IRModule:
    """Evaluate constant-only chains of :data:`FOLDABLE_OPS` and replace them with constants."""
    total = 0
    for gv, func in list(mod.functions_items()):
        if not isinstance(func, relax.Function):
            continue
        if func.attrs and any(k in func.attrs for k in ("Composite", "Codegen", "Primitive")):
            continue
        targets = _foldable_bindings(func)
        if not targets:
            continue
        mutator_ = _LegalizeFoldable(mod, targets)
        new_func = mutator_.visit_expr(func)
        if mutator_.lowered == 0:
            continue
        # The legalization functions add their PrimFuncs to the mutator's builder, not to
        # ``mod``: start from the builder's module (it holds them) and swap in the new function.
        mod = mutator_.builder_.get()
        mod[gv] = new_func
        total += mutator_.lowered
    if total == 0:
        return mod
    logger.info("TIDL prepare: folding %d constant-only ops", total)
    return tvm.transform.Sequential(
        [relax.transform.FoldConstant(), relax.transform.DeadCodeElimination()]
    )(mod)
