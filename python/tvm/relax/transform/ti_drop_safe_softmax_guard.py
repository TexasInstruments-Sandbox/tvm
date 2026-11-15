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
"""Drop PyTorch's safe-softmax guard where no row of scores can be all -inf.

torch.export decomposes scaled_dot_product_attention through
aten._safe_softmax, which from_exported_program imports as

    sm     = softmax(s, axis)
    masked = logical_not(astype(max(astype(logical_not(equal(s, -inf)), int8),
                                    axis, keepdims=True), bool))
    out    = where(masked, full_like(sm, 0.0), sm)

so that a fully masked row (all -inf) gives zeros instead of NaN. Without an
attention mask, s is the Q K^T matmul seen through reshapes and scaling by a
finite constant. A row of it is all -inf only if every dot product in the row
overflows, so out equals sm. This pass replaces such a where with sm, and dead
code elimination then removes the equal / logical_not / astype / max chain:
two fewer TIR functions per attention block. Scores that pass through
anything else, such as the add of an attention mask, keep the guard.
"""

import logging
import math

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)

# Ops that move or broadcast elements without changing their values.
_LAYOUT_OPS = frozenset(
    {
        "relax.reshape",
        "relax.permute_dims",
        "relax.broadcast_to",
        "relax.expand_dims",
        "relax.squeeze",
        "relax.flatten",
    }
)
# Ops whose rows can't be all -inf: dot products of finite operands, and
# dequantized integers.
_FINITE_SOURCES = frozenset({"relax.matmul", "relax.dequantize"})
# The guard's chain from where's condition back to equal(s, -inf).
_GUARD_CHAIN = (
    "relax.logical_not",
    "relax.astype",
    "relax.max",
    "relax.astype",
    "relax.logical_not",
    "relax.equal",
)


def _op_name(expr):
    if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
        return expr.op.name
    return None


def _scalar_const(expr):
    """The value of a one-element constant, else None."""
    if isinstance(expr, tvm.ir.GenericConst):
        data = expr.value.numpy()
        if data.size == 1:
            return float(data.reshape(()))
    return None


class _GuardMatcher:
    """Recognizes the guard in one function, through its dataflow bindings."""

    def __init__(self, func: relax.Function):
        self._var2val = relax.analysis.get_var2val(func)

    def _call(self, expr, op_name):
        """The call bound to `expr` if it is a call to `op_name`, else None."""
        if isinstance(expr, relax.Var):
            expr = self._var2val.get(expr, expr)
        return expr if _op_name(expr) == op_name else None

    def guarded_softmax(self, where: relax.Call):
        """`sm` if `where` is the guard around sm = softmax(s) and no row of s
        can be all -inf, else None."""
        cond, zeros, sm = where.args
        softmax = self._call(sm, "relax.nn.softmax")
        fill = self._call(zeros, "relax.full_like")
        if softmax is None or fill is None or _scalar_const(fill.args[1]) != 0.0:
            return None
        calls, expr = [], cond
        for op_name in _GUARD_CHAIN:
            call = self._call(expr, op_name)
            if call is None:
                return None
            calls.append(call)
            expr = call.args[0]
        _, to_bool, row_any, _, _, is_inf = calls
        s = softmax.args[0]
        if (
            str(to_bool.attrs.dtype) != "bool"
            or not is_inf.args[0].same_as(s)
            or _scalar_const(is_inf.args[1]) != -math.inf
        ):
            return None
        ndim = s.ty.ndim
        axes = row_any.attrs.axis
        if not row_any.attrs.keepdims or axes is None or len(axes) != 1:
            return None
        if int(axes[0]) % ndim != int(softmax.attrs.axis) % ndim:
            return None
        return sm if self._rows_finite(s) else None

    def _rows_finite(self, s):
        """True if s is a matmul or dequantize, seen through layout ops and
        multiplication or division by a finite scalar constant."""
        for _ in range(32):
            if isinstance(s, relax.Var):
                s = self._var2val.get(s, s)
            name = _op_name(s)
            if name in _FINITE_SOURCES:
                return True
            if name in _LAYOUT_OPS:
                s = s.args[0]
                continue
            if name in ("relax.multiply", "relax.divide"):
                a, b = s.args
                k = _scalar_const(b)
                if k is not None and math.isfinite(k) and (name == "relax.multiply" or k != 0):
                    s = a
                    continue
                k = _scalar_const(a)
                if name == "relax.multiply" and k is not None and math.isfinite(k):
                    s = b
                    continue
            return False
        return False


@mutator
class _GuardDropper(PyExprMutator):
    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.matcher = None
        self.count = 0

    def visit_call_(self, call: relax.Call):
        if _op_name(call) == "relax.where":
            sm = self.matcher.guarded_softmax(call)
            if sm is not None:
                self.count += 1
                return self.visit_expr(sm)
        return super().visit_call_(call)


@tvm.transform.module_pass(opt_level=0, name="DropSafeSoftmaxGuard")
class DropSafeSoftmaxGuard:
    """Replace where(all(s == -inf), 0, softmax(s)) with softmax(s) when s is
    an unmasked attention score (see the module docstring)."""

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        dropper = _GuardDropper(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                dropper.matcher = _GuardMatcher(func)
                dropper.builder_.update_func(gv, dropper.visit_expr(func))
        if dropper.count == 0:
            return mod
        logger.info("DropSafeSoftmaxGuard: dropped %d safe-softmax guards", dropper.count)
        return relax.transform.DeadCodeElimination()(dropper.builder_.get())
