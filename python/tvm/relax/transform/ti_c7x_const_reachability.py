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
"""Shared helper: is a Relax value fully determined without runtime input?

The C7x QDQ-fusion passes (FuseDequantizeMatmul, FuseQDQToC7xRelu, etc.) all
run *before* relax.transform.FoldConstant, on purpose -- folding a weight's
dequantize(int8_const, scale_const) too early would expand int8 weights back
to float32 in weights.bin (see fuse_dequantize_matmul.py's module docstring).
Because these passes run first, they see plain Vars, not yet the
relax.Constant nodes FoldConstant would later produce.

That ordering has a failure mode: if an *entire* matched subgraph happens to
depend only on constants -- e.g. Swin V2's continuous-relative-position-bias
MLP, applied to a fixed coordinate buffer rather than the image -- one of
these passes may still wrap it in a call_extern to a C7x-only DSP kernel.
FoldConstant (running later) then tries to evaluate that call_tir eagerly via
a host LLVM JIT to fold it, can't resolve the DSP-only symbol, and segfaults
instead of raising (TVM issue: the JIT failure surfaces only at lazy-resolved
call time, not at the build step fold_constant.cc's GetCachedBuild guards
with a try/except around).

ConstReachability lets each such pass check "is this operand's value fully
determined at compile time" *before* choosing the extern path, so genuinely
input-independent subgraphs fall through to the portable/generic lowering
instead -- which is both crash-free and strictly better (the DSP kernel would
otherwise recompute the same fixed value on every single inference).
"""

from tvm import relax


class ConstReachability:
    """Answers "is this Expr's value determined without any runtime input?"

    Built once per pass invocation from the module's bindings (a single
    linear walk), then queried per matmul/op site with memoized recursion.
    """

    def __init__(self, mod):
        self._bindings = {}
        for _, func in mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            for block in func.body.blocks:
                for binding in block.bindings:
                    self._bindings[binding.var] = binding.value
        self._memo = {}

    def is_const(self, expr) -> bool:
        """Whether `expr` is transitively constant (no dependence on a
        runtime-input Var anywhere in its producer chain)."""
        if expr in self._memo:
            return self._memo[expr]
        # Assume non-const until proven otherwise: guards against infinite
        # recursion if a cycle were ever present (shouldn't happen in a
        # dataflow graph, but this makes that failure mode safe instead of
        # a stack overflow).
        self._memo[expr] = False

        if isinstance(expr, (relax.Constant, relax.ShapeExpr, relax.PrimValue)):
            result = True
        elif isinstance(expr, relax.Var):
            bound = self._bindings.get(expr)
            result = bound is not None and self.is_const(bound)
        elif isinstance(expr, relax.Call):
            result = all(self.is_const(a) for a in expr.args)
        elif isinstance(expr, relax.Tuple):
            result = all(self.is_const(f) for f in expr.fields)
        elif isinstance(expr, relax.TupleGetItem):
            result = self.is_const(expr.tuple_value)
        else:
            result = False

        self._memo[expr] = result
        return result
