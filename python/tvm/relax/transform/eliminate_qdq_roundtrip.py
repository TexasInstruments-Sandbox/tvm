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
"""Eliminate unnecessary QDQ round-trips through int8.

After FuseQDQToInt8Conv2D, fused conv layers output int8 via:

    round(x) -> clip(x, -128, 127) -> astype(x, int8) -> dequantize(x, scale, zp)

This round-trip through int8 introduces rounding error (up to +/-0.5
quantization steps per layer) that compounds through deep networks with
residual connections.

This pass replaces the pattern with a direct float32 rescale:

    (x - float(zp)) * scale

preserving full float32 precision between layers.

The pattern ONLY matches where round-clip-cast-dequant appears in
sequence (i.e. at fused conv outputs feeding into residual adds).
It does NOT match skip-branch dequantize nodes (which have no
round/clip/cast before them).

This pass should run AFTER FuseQDQToInt8Conv2D and BEFORE LegalizeOps.
"""

import logging

import numpy as np

from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax import Expr
from tvm.relax.dpl import is_op, rewrite_call, wildcard

from . import function_pass

logger = logging.getLogger(__name__)


def _is_const_zero(expr: Expr) -> bool:
    """Check if a Relax expression is a constant zero (scalar or vector)."""
    if isinstance(expr, relax.Constant):
        arr = expr.data.numpy()
        return np.all(arr == 0)
    return False


def _make_roundtrip_pattern():
    """Match: dequantize(astype(clip(round(x), lo, hi), int8), scale, zp).

    This is the int8 round-trip produced by the quantization output of
    FuseQDQToInt8Conv2D, immediately followed by a dequantize for the
    next layer's input.
    """
    x = wildcard()
    rounded = is_op("relax.round")(x)
    clipped = is_op("relax.clip")(rounded, wildcard(), wildcard())
    cast_int8 = is_op("relax.astype")(clipped)

    dq_scale = wildcard()
    dq_zp = wildcard()
    pattern = is_op("relax.dequantize")(cast_int8, dq_scale, dq_zp)

    return pattern, x, dq_scale, dq_zp


def _make_roundtrip_rewriter(x_node, scale_node, zp_node):
    """Create a rewriter that replaces the round-trip with direct rescale.

    Given: round(x) -> clip -> int8 -> dequantize(_, scale, zp)
    Emit:  (x - float(zp)) * scale   [or x * scale when zp == 0]
    """

    def rewriter(expr, matches):
        x = matches[x_node]
        scale = matches[scale_node]
        zp = matches[zp_node]

        if _is_const_zero(zp):
            result = relax.op.multiply(x, scale)
        else:
            zp_float = relax.op.astype(zp, "float32")
            result = relax.op.multiply(relax.op.subtract(x, zp_float), scale)

        logger.info("Eliminated QDQ round-trip: round->clip->int8->dequant -> direct rescale")
        return result

    return rewriter


@function_pass(opt_level=0)
class EliminateQDQRoundTrip:
    """Eliminate unnecessary quantize-dequantize round-trips through int8.

    After FuseQDQToInt8Conv2D fuses conv layers into int8 arithmetic,
    the fused output is: round -> clip -> cast(int8).  When this feeds
    directly into a dequantize (e.g. for a residual add in float32),
    the round-trip through int8 introduces rounding error that compounds
    through deep networks.

    This pass rewrites:

        round(x) -> clip(x, -128, 127) -> astype(x, int8) -> dequantize(x, s, zp)

    into:

        (x - float(zp)) * s

    preserving full float32 precision between layers.

    This pass should run AFTER FuseQDQToInt8Conv2D and BEFORE LegalizeOps.
    """

    def transform_function(
        self, func: Expr, mod: IRModule, ctx: PassContext
    ) -> Expr:
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return func

        pattern, x_node, scale_node, zp_node = _make_roundtrip_pattern()
        rewriter = _make_roundtrip_rewriter(x_node, scale_node, zp_node)
        return rewrite_call(pattern, rewriter, func)
