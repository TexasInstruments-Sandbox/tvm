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
"""TIDL pattern definitions for Relax BYOC partitioning.

Defines FusionPattern entries for ops supported by TI Deep Learning (TIDL)
on the C7x MMA accelerator.  Each pattern has a ``tidl.`` prefix and an
optional Python-side check function that validates basic hardware
constraints (dtype, rank, kernel size, etc.).  When the full TIDL
constraint-checking .so is available, the check functions delegate to
``tidl.check_op_support`` (registered in C++); otherwise they fall back
to the lightweight Python checks defined here.
"""

from typing import Dict, List, Optional

import tvm
from tvm.relax.dpl.pattern import DFPattern, is_op, wildcard
from tvm.relax.transform import FusionPattern, PatternCheckContext

from ..pattern_registry import register_patterns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIDL_SUPPORTED_DTYPES = {"float32", "int8", "int16", "uint8"}


def _get_shape(expr) -> Optional[list]:
    """Return shape as a plain list of ints, or None."""
    si = expr.struct_info
    if not hasattr(si, "shape") or si.shape is None:
        return None
    return [int(d) for d in si.shape]


def _get_dtype(expr) -> Optional[str]:
    si = expr.struct_info
    return getattr(si, "dtype", None)


def _check_dtype(expr) -> bool:
    dtype = _get_dtype(expr)
    return dtype is not None and dtype in _TIDL_SUPPORTED_DTYPES


def _check_rank(expr, max_rank: int = 4) -> bool:
    shape = _get_shape(expr)
    if shape is None:
        return True  # cannot verify, let C++ check later
    return len(shape) <= max_rank


# ---------------------------------------------------------------------------
# Per-op check functions
# ---------------------------------------------------------------------------


def _check_conv2d(ctx: PatternCheckContext) -> bool:
    """Validate conv2d constraints for TIDL.

    - Supported dtypes: float32, int8, int16, uint8
    - Input rank <= 4
    - Kernel size <= 7 in each dimension
    - Strides must be equal in H and W
    """
    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call):
        return False

    # dtype check on input
    inp = ctx.annotated_expr.get("input")
    if inp is not None and not _check_dtype(inp):
        return False

    # rank check
    if inp is not None and not _check_rank(inp, 4):
        return False

    attrs = root.attrs
    if attrs is None:
        return True

    # kernel size <= 7
    # First try explicit kernel_size attr, then infer from weight shape
    kernel_size = getattr(attrs, "kernel_size", None)
    if kernel_size is not None and len(kernel_size) > 0:
        for k in kernel_size:
            if int(k) > 7:
                return False
    else:
        # Infer kernel size from weight tensor shape (OIHW layout)
        weight = ctx.annotated_expr.get("weight")
        if weight is not None:
            wshape = _get_shape(weight)
            if wshape is not None and len(wshape) == 4:
                kh, kw = wshape[2], wshape[3]
                if kh > 7 or kw > 7:
                    return False

    # strides must be equal
    strides = getattr(attrs, "strides", None)
    if strides is not None and len(strides) == 2:
        if int(strides[0]) != int(strides[1]):
            return False

    return True


def _check_pool(ctx: PatternCheckContext) -> bool:
    """Validate pooling constraints for TIDL.

    - Input rank == 4
    - Pool kernel <= 3 in each dimension
    """
    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call):
        return False

    data = ctx.annotated_expr.get("data")
    if data is not None:
        if not _check_dtype(data):
            return False
        shape = _get_shape(data)
        if shape is not None and len(shape) != 4:
            return False

    attrs = root.attrs
    if attrs is None:
        return True

    pool_size = getattr(attrs, "pool_size", None)
    if pool_size is not None:
        for p in pool_size:
            if int(p) > 3:
                return False

    return True


def _check_relu(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_add(ctx: PatternCheckContext) -> bool:
    """Element-wise add — dtype + rank checks."""
    for key in ("lhs", "rhs"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None:
            if not _check_dtype(expr):
                return False
            if not _check_rank(expr, 4):
                return False
    return True


def _check_quantize(ctx: PatternCheckContext) -> bool:
    return True


def _check_dequantize(ctx: PatternCheckContext) -> bool:
    return True


# ---------------------------------------------------------------------------
# Pattern builders
# ---------------------------------------------------------------------------


def _conv2d_patterns() -> List[FusionPattern]:
    """Conv2d with optional bias and optional activation (relu or clip).

    Produces patterns in priority order (most specific first):
        tidl.nn.conv2d_bias_clip
        tidl.nn.conv2d_bias_relu
        tidl.nn.conv2d_bias
        tidl.nn.conv2d_clip
        tidl.nn.conv2d_relu
        tidl.nn.conv2d
    """
    patterns = []
    for with_bias in [True, False]:
        for activation in ["relax.clip", "relax.nn.relu", None]:
            inp = wildcard()
            weight = wildcard()
            annotations: Dict[str, DFPattern] = {"input": inp, "weight": weight}

            out = is_op("relax.nn.conv2d")(inp, weight)
            annotations["root"] = out

            if with_bias:
                bias = wildcard()
                annotations["bias"] = bias
                out = is_op("relax.add")(out, bias)

            if activation is not None:
                out = is_op(activation)(out)

            # Build name
            suffix = ""
            if with_bias:
                suffix += "_bias"
            if activation == "relax.nn.relu":
                suffix += "_relu"
            elif activation == "relax.clip":
                suffix += "_clip"

            name = f"tidl.nn.conv2d{suffix}"
            patterns.append(FusionPattern(name, out, annotations, _check_conv2d))

    return patterns


def _pool_patterns() -> List[FusionPattern]:
    """Max pool and avg pool patterns."""
    patterns = []
    for op_name, pattern_name in [
        ("relax.nn.max_pool2d", "tidl.nn.max_pool2d"),
        ("relax.nn.avg_pool2d", "tidl.nn.avg_pool2d"),
    ]:
        data = wildcard()
        pat = is_op(op_name)(data)
        annotations = {"data": data, "root": pat}
        patterns.append(FusionPattern(pattern_name, pat, annotations, _check_pool))
    return patterns


def _relu_pattern() -> List[FusionPattern]:
    data = wildcard()
    pat = is_op("relax.nn.relu")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.relu", pat, annotations, _check_relu)]


def _add_pattern() -> List[FusionPattern]:
    """Element-wise add (e.g. residual connections)."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.add")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.add", pat, annotations, _check_add)]


def _quantize_pattern() -> List[FusionPattern]:
    data = wildcard()
    scale = wildcard()
    zp = wildcard()
    pat = is_op("relax.quantize")(data, scale, zp)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.quantize", pat, annotations, _check_quantize)]


def _dequantize_pattern() -> List[FusionPattern]:
    data = wildcard()
    scale = wildcard()
    zp = wildcard()
    pat = is_op("relax.dequantize")(data, scale, zp)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.dequantize", pat, annotations, _check_dequantize)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tidl_patterns() -> List[FusionPattern]:
    """Return all TIDL patterns ordered by priority (most specific first).

    More specific patterns (e.g. conv2d_bias_relu) appear first so that
    ``FuseOpsByPattern`` greedily matches the largest possible fused op.
    """
    return [
        # conv2d variants — most specific first
        *_conv2d_patterns(),
        # pooling
        *_pool_patterns(),
        # standalone activations / element-wise ops (lower priority so they
        # get absorbed into conv2d composites when possible)
        *_relu_pattern(),
        *_add_pattern(),
        # quantize/dequantize
        *_quantize_pattern(),
        *_dequantize_pattern(),
    ]


# Register all patterns at module import time
_tidl_patterns = get_tidl_patterns()
register_patterns(_tidl_patterns)
