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

    # kernel size 1-5 only: TIDL's allowlisting rejects 6×6 kernels at
    # PostProcessNet time (e.g. YOLOv5's Focus stem conv).  Filter them
    # here before they waste calibration time and require fallback handling.
    kernel_size = getattr(attrs, "kernel_size", None)
    if kernel_size is not None and len(kernel_size) > 0:
        for k in kernel_size:
            k_int = int(k)
            if k_int > 7 or k_int == 6:
                return False
    else:
        # Infer kernel size from weight tensor shape (OIHW layout)
        weight = ctx.annotated_expr.get("weight")
        if weight is not None:
            wshape = _get_shape(weight)
            if wshape is not None and len(wshape) == 4:
                kh, kw = wshape[2], wshape[3]
                if kh > 7 or kw > 7 or kh == 6 or kw == 6:
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
    """Element-wise add — dtype + rank checks.

    TIDL eltwise layers require 4-D (NCHW) tensors.  Reject adds
    on lower-rank tensors such as the FC bias add ``(1, 1000)``
    which would crash in TIDL's algProcess.
    """
    for key in ("lhs", "rhs"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None:
            if not _check_dtype(expr):
                return False
            shape = _get_shape(expr)
            if shape is not None and len(shape) < 4:
                return False
            if not _check_rank(expr, 4):
                return False
    return True


def _check_mean(ctx: PatternCheckContext) -> bool:
    """Validate mean constraints for TIDL.

    - Supported dtypes
    - Input rank == 4 (NCHW)
    - Reduction over spatial axes only: axis in {[2,3], [-2,-1]}
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

    # Validate axis from attrs
    if root.attrs is not None:
        axis = getattr(root.attrs, "axis", None)
        if axis is not None:
            ax_list = [int(a) for a in axis]
            ndim = 4
            # Normalize negative axes
            ax_norm = sorted(a % ndim for a in ax_list)
            # TIDL supports spatial reduction: axes [2,3]
            if ax_norm != [2, 3]:
                return False

    return True


def _check_reshape(ctx: PatternCheckContext) -> bool:
    """Validate reshape constraints for TIDL.

    Lightweight check: dtype only.  Shape validation is done by TIDL import.
    """
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_matmul(ctx: PatternCheckContext) -> bool:
    """Validate matmul constraints for TIDL (InnerProduct layer).

    - Supported dtypes
    - Both inputs must be tensors
    """
    for key in ("data", "weight"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None and not _check_dtype(expr):
            return False
    return True


def _check_matmul_bias(ctx: PatternCheckContext) -> bool:
    """Validate fused matmul+bias constraints for TIDL (InnerProduct with bias).

    - Supported dtypes on all inputs
    """
    for key in ("data", "weight", "bias"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None and not _check_dtype(expr):
            return False
    return True


def _check_multiply(ctx: PatternCheckContext) -> bool:
    """Element-wise multiply — same constraints as add (rank >= 4)."""
    for key in ("lhs", "rhs"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None:
            if not _check_dtype(expr):
                return False
            shape = _get_shape(expr)
            if shape is not None and len(shape) < 4:
                return False
            if not _check_rank(expr, 4):
                return False
    return True


def _check_permute_dims(ctx: PatternCheckContext) -> bool:
    """Validate permute_dims (transpose) constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_softmax(ctx: PatternCheckContext) -> bool:
    """Validate softmax constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_concat(ctx: PatternCheckContext) -> bool:
    """Validate concat constraints for TIDL.

    ``data`` is the Tuple matched by the wildcard — it has TupleStructInfo,
    not a dtype.  Check dtype on the first Tuple field instead.
    """
    data = ctx.annotated_expr.get("data")
    if data is not None:
        si = data.struct_info
        # Tuple input: check dtype of the first field
        if hasattr(si, "fields") and si.fields:
            first_field = si.fields[0]
            dtype = getattr(first_field, "dtype", None)
            if dtype is not None and dtype not in _TIDL_SUPPORTED_DTYPES:
                return False
        elif not _check_dtype(data):
            # Fallback for non-Tuple (shouldn't normally happen)
            return False
    return True


def _check_batch_norm(ctx: PatternCheckContext) -> bool:
    """Validate batch_norm constraints for TIDL.

    - Supported dtypes
    - Input rank == 4 (NCHW)
    """
    data = ctx.annotated_expr.get("data")
    if data is not None:
        if not _check_dtype(data):
            return False
        shape = _get_shape(data)
        if shape is not None and len(shape) != 4:
            return False
    return True


def _check_sigmoid(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_tanh(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_clip(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_leakyrelu(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_prelu(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_eltwise(ctx: PatternCheckContext) -> bool:
    """Shared check for element-wise binary ops (divide, subtract, max, min).

    Same constraints as add/multiply: dtype + rank >= 4.
    """
    for key in ("lhs", "rhs"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None:
            if not _check_dtype(expr):
                return False
            shape = _get_shape(expr)
            if shape is not None and len(shape) < 4:
                return False
            if not _check_rank(expr, 4):
                return False
    return True


def _check_layer_norm(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_squeeze(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_expand_dims(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_strided_slice(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_cast(ctx: PatternCheckContext) -> bool:
    return True


def _check_reduce_sum(ctx: PatternCheckContext) -> bool:
    """Validate sum constraints for TIDL.

    ReduceSum is converted to InnerProduct during OptimizeNet, so it
    can handle any single axis.  Only dtype and single-axis are checked.
    """
    root = ctx.annotated_expr.get("root")
    if root is not None and isinstance(root, tvm.relax.Call):
        attrs = root.attrs
        if attrs is not None:
            axis = getattr(attrs, "axis", None)
            if axis is not None and len(axis) > 1:
                return False  # TIDL only supports single-axis reduction
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_reduce_max_min(ctx: PatternCheckContext) -> bool:
    """Validate reduce_max/reduce_min constraints for TIDL.

    TIDL ReduceLayer only supports single-axis reduction along the HEIGHT
    dimension (axis=2 in 4D NCHW, which maps to TIDL_DIM_HEIGHT=4).
    The TIDL allowlisting constraint rejects all other axes.
    """
    root = ctx.annotated_expr.get("root")
    if root is not None and isinstance(root, tvm.relax.Call):
        attrs = root.attrs
        if attrs is not None:
            axis = getattr(attrs, "axis", None)
            if axis is not None:
                if len(axis) > 1:
                    return False  # TIDL only supports single-axis reduction
                if len(axis) == 1:
                    ax = int(axis[0])
                    data = ctx.annotated_expr.get("data")
                    ndim = 4
                    if data is not None:
                        shape = _get_shape(data)
                        if shape is not None:
                            ndim = len(shape)
                    if ax < 0:
                        ax += ndim
                    # TIDL ReduceLayer only supports HEIGHT (axis 2 in 4D NCHW)
                    if ax != 2:
                        return False
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_argreduce(ctx: PatternCheckContext) -> bool:
    """Validate argmax/argmin constraints for TIDL.

    TIDL ArgOpLayer only supports:
    - axis = channel axis (axis=1 in 4D NCHW → TIDL_DIM_NUMCH=3)
    - keepdims = True
    """
    root = ctx.annotated_expr.get("root")
    if root is not None and isinstance(root, tvm.relax.Call):
        attrs = root.attrs
        if attrs is not None:
            # keepdims must be True
            keepdims = getattr(attrs, "keepdims", None)
            if keepdims is not None and not keepdims:
                return False
            # axis must be channel axis (1 in 4D NCHW)
            axis = getattr(attrs, "axis", None)
            if axis is not None:
                data = ctx.annotated_expr.get("data")
                ndim = 4
                if data is not None:
                    shape = _get_shape(data)
                    if shape is not None:
                        ndim = len(shape)
                ax = int(axis)
                if ax < 0:
                    ax += ndim
                # TIDL ArgOpLayer only supports channel axis (1 in 4D NCHW)
                if ax != 1:
                    return False
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_flatten(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _check_pad(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
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


def _mean_pattern() -> List[FusionPattern]:
    """Mean reduction (global avg pool in ResNet-18)."""
    data = wildcard()
    pat = is_op("relax.mean")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.mean", pat, annotations, _check_mean)]


def _reshape_pattern() -> List[FusionPattern]:
    """Reshape (flatten before FC layer)."""
    data = wildcard()
    pat = is_op("relax.reshape")(data, wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.reshape", pat, annotations, _check_reshape)]


def _matmul_patterns() -> List[FusionPattern]:
    """Matmul with optional bias (FC layer).

    Returns patterns in priority order:
        tidl.matmul_bias  (fused matmul + add)
        tidl.matmul       (standalone matmul)
    """
    patterns = []

    # Fused matmul + bias add (must come first for greedy matching)
    data_mb = wildcard()
    weight_mb = wildcard()
    bias = wildcard()
    matmul_out = is_op("relax.matmul")(data_mb, weight_mb)
    pat_mb = is_op("relax.add")(matmul_out, bias)
    annotations_mb = {"data": data_mb, "weight": weight_mb, "bias": bias, "root": pat_mb}
    patterns.append(FusionPattern("tidl.matmul_bias", pat_mb, annotations_mb, _check_matmul_bias))

    # Standalone matmul (lower priority)
    data_m = wildcard()
    weight_m = wildcard()
    pat_m = is_op("relax.matmul")(data_m, weight_m)
    annotations_m = {"data": data_m, "weight": weight_m, "root": pat_m}
    patterns.append(FusionPattern("tidl.matmul", pat_m, annotations_m, _check_matmul))

    return patterns


def _multiply_pattern() -> List[FusionPattern]:
    """Element-wise multiply (gate * up in transformers)."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.multiply")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.multiply", pat, annotations, _check_multiply)]


def _permute_dims_pattern() -> List[FusionPattern]:
    """Transpose / permute dimensions."""
    data = wildcard()
    pat = is_op("relax.permute_dims")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.permute_dims", pat, annotations, _check_permute_dims)]


def _softmax_pattern() -> List[FusionPattern]:
    """Softmax (attention scores in transformers)."""
    data = wildcard()
    pat = is_op("relax.nn.softmax")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.softmax", pat, annotations, _check_softmax)]


def _concat_pattern() -> List[FusionPattern]:
    """Concatenation (RoPE, multi-head assembly)."""
    # concat takes a Tuple, match it with a wildcard
    data = wildcard()
    pat = is_op("relax.concat")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.concat", pat, annotations, _check_concat)]


def _batch_norm_pattern() -> List[FusionPattern]:
    """Batch normalization (standalone, not folded into conv).

    NOTE: batch_norm returns a Tuple; FuseOpsByPattern does not reliably
    match patterns that terminate with TupleGetItem.  In the normal TIDL
    pipeline ``prepare()`` folds BN into conv2d weights via
    ``FoldBatchnormToConv2D`` so this pattern is only needed for rare
    standalone-BN models.  Currently disabled until FuseOpsByPattern
    TupleGetItem support is resolved.
    """
    # Pattern matching for tuple-returning ops needs framework support.
    # Return empty list for now — BN folding handles the common case.
    return []


def _sigmoid_pattern() -> List[FusionPattern]:
    """Sigmoid activation."""
    data = wildcard()
    pat = is_op("relax.sigmoid")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.sigmoid", pat, annotations, _check_sigmoid)]


def _tanh_pattern() -> List[FusionPattern]:
    """Tanh activation."""
    data = wildcard()
    pat = is_op("relax.tanh")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.tanh", pat, annotations, _check_tanh)]


def _clip_pattern() -> List[FusionPattern]:
    """Standalone clip (e.g. relu6 = clip(x, 0, 6))."""
    data = wildcard()
    pat = is_op("relax.clip")(data, wildcard(), wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.clip", pat, annotations, _check_clip)]


def _leakyrelu_pattern() -> List[FusionPattern]:
    """Leaky ReLU activation."""
    data = wildcard()
    pat = is_op("relax.nn.leakyrelu")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.leakyrelu", pat, annotations, _check_leakyrelu)]


def _prelu_pattern() -> List[FusionPattern]:
    """PReLU activation (parametric ReLU with learnable slope)."""
    data = wildcard()
    alpha = wildcard()
    pat = is_op("relax.nn.prelu")(data, alpha)
    annotations = {"data": data, "alpha": alpha, "root": pat}
    return [FusionPattern("tidl.nn.prelu", pat, annotations, _check_prelu)]


def _divide_pattern() -> List[FusionPattern]:
    """Element-wise divide."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.divide")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.divide", pat, annotations, _check_eltwise)]


def _subtract_pattern() -> List[FusionPattern]:
    """Element-wise subtract."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.subtract")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.subtract", pat, annotations, _check_eltwise)]


def _maximum_pattern() -> List[FusionPattern]:
    """Element-wise maximum."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.maximum")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.maximum", pat, annotations, _check_eltwise)]


def _minimum_pattern() -> List[FusionPattern]:
    """Element-wise minimum."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.minimum")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.minimum", pat, annotations, _check_eltwise)]


def _layer_norm_pattern() -> List[FusionPattern]:
    """Layer normalization (transformers)."""
    data = wildcard()
    gamma = wildcard()
    beta = wildcard()
    pat = is_op("relax.nn.layer_norm")(data, gamma, beta)
    annotations = {"data": data, "gamma": gamma, "beta": beta, "root": pat}
    return [FusionPattern("tidl.nn.layer_norm", pat, annotations, _check_layer_norm)]


def _flatten_pattern() -> List[FusionPattern]:
    """Flatten (before FC layer)."""
    data = wildcard()
    pat = is_op("relax.flatten")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.flatten", pat, annotations, _check_flatten)]


def _squeeze_pattern() -> List[FusionPattern]:
    """Squeeze (remove size-1 dimensions)."""
    data = wildcard()
    pat = is_op("relax.squeeze")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.squeeze", pat, annotations, _check_squeeze)]


def _expand_dims_pattern() -> List[FusionPattern]:
    """Expand dims (insert size-1 dimension)."""
    data = wildcard()
    pat = is_op("relax.expand_dims")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.expand_dims", pat, annotations, _check_expand_dims)]


def _strided_slice_pattern() -> List[FusionPattern]:
    """Strided slice (tensor slicing).

    relax.strided_slice takes 4 args: (data, begin, end, strides)
    where begin/end/strides are Tuples of PrimValues.
    """
    data = wildcard()
    pat = is_op("relax.strided_slice")(data, varg_default_wildcard=True)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.strided_slice", pat, annotations, _check_strided_slice)]


def _cast_pattern() -> List[FusionPattern]:
    """Cast (dtype conversion via astype)."""
    data = wildcard()
    pat = is_op("relax.astype")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.cast", pat, annotations, _check_cast)]


def _pad_pattern() -> List[FusionPattern]:
    """Pad (spatial padding)."""
    data = wildcard()
    pat = is_op("relax.nn.pad")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.pad", pat, annotations, _check_pad)]


def _sum_pattern() -> List[FusionPattern]:
    """Sum reduction."""
    data = wildcard()
    pat = is_op("relax.sum")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.sum", pat, annotations, _check_reduce_sum)]


def _reduce_max_pattern() -> List[FusionPattern]:
    """Max reduction (HEIGHT axis only per TIDL ReduceLayer constraint)."""
    data = wildcard()
    pat = is_op("relax.max")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.reduce_max", pat, annotations, _check_reduce_max_min)]


def _reduce_min_pattern() -> List[FusionPattern]:
    """Min reduction (HEIGHT axis only per TIDL ReduceLayer constraint)."""
    data = wildcard()
    pat = is_op("relax.min")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.reduce_min", pat, annotations, _check_reduce_max_min)]


def _argmax_pattern() -> List[FusionPattern]:
    """Argmax (index of maximum value along an axis)."""
    data = wildcard()
    pat = is_op("relax.argmax")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.argmax", pat, annotations, _check_argreduce)]


def _argmin_pattern() -> List[FusionPattern]:
    """Argmin (index of minimum value along an axis)."""
    data = wildcard()
    pat = is_op("relax.argmin")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.argmin", pat, annotations, _check_argreduce)]


def _check_unary(ctx: PatternCheckContext) -> bool:
    """Dtype check for parameterless unary math ops."""
    data = ctx.annotated_expr.get("data")
    if data is not None and not _check_dtype(data):
        return False
    return True


def _math_unary_pattern(op_name: str, composite_name: str) -> List[FusionPattern]:
    """Helper: create a pattern for a parameterless unary math op."""
    data = wildcard()
    pat = is_op(op_name)(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern(composite_name, pat, annotations, _check_unary)]


def _abs_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.abs", "tidl.abs")


def _sqrt_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.sqrt", "tidl.sqrt")


def _exp_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.exp", "tidl.exp")


def _log_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.log", "tidl.log")


def _erf_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.erf", "tidl.erf")


def _floor_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.floor", "tidl.floor")


def _negative_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.negative", "tidl.negative")


def _sin_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.sin", "tidl.sin")


def _cos_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.cos", "tidl.cos")


def _tan_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.tan", "tidl.tan")


def _sinh_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.sinh", "tidl.sinh")


def _cosh_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.cosh", "tidl.cosh")


def _asin_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.asin", "tidl.asin")


def _acos_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.acos", "tidl.acos")


def _atan_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.atan", "tidl.atan")


def _asinh_pattern() -> List[FusionPattern]:
    return _math_unary_pattern("relax.asinh", "tidl.asinh")


def _power_pattern() -> List[FusionPattern]:
    """Power (x ** scalar_exponent)."""
    data = wildcard()
    exp = wildcard()
    pat = is_op("relax.power")(data, exp)
    annotations = {"data": data, "exp": exp, "root": pat}
    return [FusionPattern("tidl.power", pat, annotations, _check_unary)]


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
    patterns = [
        # conv2d variants — most specific first
        *_conv2d_patterns(),
        # pooling
        *_pool_patterns(),
        # batch normalization (standalone, not folded into conv)
        *_batch_norm_pattern(),
        # mean reduction (global avg pool)
        *_mean_pattern(),
        # standalone activations — lower priority so they get absorbed
        # into conv2d composites when possible
        *_relu_pattern(),
        *_sigmoid_pattern(),
        *_tanh_pattern(),
        *_clip_pattern(),
        *_leakyrelu_pattern(),
        *_prelu_pattern(),
        # matmul (FC layer) — fused matmul_bias before standalone matmul
        *_matmul_patterns(),
        # element-wise ops (rank >= 4 only)
        *_add_pattern(),
        *_multiply_pattern(),
        *_divide_pattern(),
        *_subtract_pattern(),
        *_maximum_pattern(),
        *_minimum_pattern(),
        # shape manipulation
        *_reshape_pattern(),
        *_flatten_pattern(),
        *_squeeze_pattern(),
        *_expand_dims_pattern(),
        *_strided_slice_pattern(),
        *_permute_dims_pattern(),
        *_concat_pattern(),
        # normalization
        *_layer_norm_pattern(),
        # dtype
        *_cast_pattern(),
        *_pad_pattern(),
        # reductions
        *_sum_pattern(),
        *_reduce_max_pattern(),
        *_reduce_min_pattern(),
        *_argmax_pattern(),
        *_argmin_pattern(),
        # math / unary ops
        *_abs_pattern(),
        *_sqrt_pattern(),
        *_power_pattern(),
        *_exp_pattern(),
        *_log_pattern(),
        *_erf_pattern(),
        *_floor_pattern(),
        *_negative_pattern(),
        *_sin_pattern(),
        *_cos_pattern(),
        *_tan_pattern(),
        *_sinh_pattern(),
        *_cosh_pattern(),
        *_asin_pattern(),
        *_acos_pattern(),
        *_atan_pattern(),
        *_asinh_pattern(),
        # quantize/dequantize
        *_quantize_pattern(),
        *_dequantize_pattern(),
    ]

    # Transformer ops
    patterns.extend(
        [
            *_softmax_pattern(),
        ]
    )

    return patterns


# Register all patterns at module import time
_tidl_patterns = get_tidl_patterns()
register_patterns(_tidl_patterns)
