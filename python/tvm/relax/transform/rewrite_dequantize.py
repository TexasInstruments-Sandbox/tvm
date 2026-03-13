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
# pylint: disable=invalid-name, unused-argument
"""Rewrite int8 cast+multiply into R.dequantize to prevent constant folding.

When manual per-channel INT8 weight quantization is exported through
torch.export, the pattern in the graph is:

    int8_weight_const -> astype(float32) -> multiply(scale_const)

TVM's constant folding pass will evaluate this eagerly, producing a
float32 constant and defeating the purpose of INT8 storage (the
weights.bin still contains the full float32 result).

This pass rewrites the pattern into:

    R.dequantize(int8_weight_const, scale_const, zero_point=0, axis=0)

which TVM's QDQ infrastructure preserves through compilation, keeping
int8 weights and float32 scales as separate constants in the final
binary.

Run this pass AFTER BindParams and BEFORE FoldConstant / LegalizeOps.
"""

import logging

import numpy as np

from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax import Constant, Expr
from tvm.relax.dpl import is_op, rewrite_call, wildcard

from . import function_pass

logger = logging.getLogger(__name__)


def _make_pattern():
    """Match: multiply(astype(int8_const, float32), scale_const).

    This is the pattern produced by manual per-channel weight-only
    INT8 quantization: weight_int8.float() * scale in PyTorch becomes
    astype(const_int8, "float32") -> multiply(_, const_scale).
    """
    int8_node = wildcard()
    cast_node = is_op("relax.astype")(int8_node)
    scale_node = wildcard()
    pattern = is_op("relax.multiply")(cast_node, scale_node)
    return pattern, int8_node, cast_node, scale_node


def _make_rewriter(int8_node, cast_node, scale_node):
    """Create a rewriter that replaces cast+multiply with dequantize."""

    count = [0]

    def rewriter(expr, matches):
        int8_const = matches[int8_node]
        cast_expr = matches[cast_node]
        scale_const = matches[scale_node]

        # Only rewrite if the source is an int8 constant
        if not isinstance(int8_const, Constant):
            return expr
        arr = int8_const.data.numpy()
        if arr.dtype != np.int8:
            return expr

        # Only rewrite if the scale is a constant
        if not isinstance(scale_const, Constant):
            return expr

        # Only rewrite if the cast target is float32
        if not hasattr(cast_expr, "attrs") or str(cast_expr.attrs.dtype) != "float32":
            return expr

        # Determine the quantization axis.  Per-channel weights have
        # shape [out_channels, ...] with scale shape [out_channels] or
        # [out_channels, 1, ...].  Use axis=0 (output channel dim).
        axis = 0

        # dequantize legalization expects scale/zp to be either scalar
        # or 1-D along the quantization axis.  Squeeze any trailing
        # dimensions (e.g. [out_channels, 1] -> [out_channels]).
        scale_arr = scale_const.data.numpy()
        squeezed_scale = scale_arr.squeeze()
        new_scale = relax.const(squeezed_scale)

        # Zero_point must match scale shape for per-channel quantization
        if squeezed_scale.ndim == 0:
            zp = relax.const(np.int8(0))
        else:
            zp = relax.const(np.zeros(squeezed_scale.shape, dtype="int8"))

        result = relax.op.dequantize(int8_const, new_scale, zp, axis=axis)
        count[0] += 1
        logger.debug(
            "Rewrote cast+multiply -> dequantize (weight shape=%s, scale shape=%s)",
            arr.shape,
            scale_arr.shape,
        )
        return result

    return rewriter, count


@function_pass(opt_level=0)
class RewriteDequantize:
    """Rewrite int8 cast+multiply into R.dequantize to prevent constant folding.

    Matches: Constant(int8) -> astype(float32) -> multiply(Constant(float32))
    Rewrites to: R.dequantize(int8_const, scale_const, zero_point=0, axis=0)

    This preserves int8 weight storage in the compiled binary instead of
    allowing constant folding to expand weights back to float32.

    Run AFTER BindParams, BEFORE FoldConstant / LegalizeOps.
    """

    def transform_function(
        self, func: Expr, mod: IRModule, ctx: PassContext
    ) -> Expr:
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return func

        pattern, int8_node, cast_node, scale_node = _make_pattern()
        rewriter, count = _make_rewriter(int8_node, cast_node, scale_node)
        result = rewrite_call(pattern, rewriter, func)
        if count[0] > 0:
            logger.info("RewriteDequantize: rewrote %d patterns", count[0])
        return result
