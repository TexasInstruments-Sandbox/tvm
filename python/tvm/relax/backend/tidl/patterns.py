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

# TIDL_NUM_OUT_BUFS (itidl_ti.h): sTIDL_SliceLayerParams_t.slicePoints is a
# fixed [TIDL_NUM_OUT_BUFS + 1] = [17] array.  tidl_parse_relax_split.cpp's
# equal-sections and explicit-indices loops both write slicePoints[0..N]
# with no bound check, so N > 16 is an out-of-bounds write, not a decline.
_TIDL_MAX_SPLIT_OUTPUTS = 16

# TIDL's rank-6 allowlisting ceiling (tidlGetNonSingletonNumDims callers
# across tidlAllowlistingConstraints/*.h), used for early-exit checks that
# mirror an unconditional (not modelType-gated) generic constraint.
_TIDL_MAX_RANK = 6


def _get_shape(expr) -> Optional[list]:
    """Return shape as a plain list of ints, or None.

    None when the shape is unknown or has a symbolic dimension (``int()`` on a
    ``tir.Var`` would raise out of the pattern callback).
    """
    si = expr.struct_info
    if not hasattr(si, "shape") or si.shape is None:
        return None
    try:
        return [int(d) for d in si.shape]
    except (TypeError, ValueError):
        return None


def _get_dtype(expr) -> Optional[str]:
    si = expr.struct_info
    return getattr(si, "dtype", None)


def _check_dtype(expr) -> bool:
    dtype = _get_dtype(expr)
    return dtype is not None and dtype in _TIDL_SUPPORTED_DTYPES


def _check_rank(expr, max_rank: int = 4) -> bool:
    """True if the rank is known and within what TIDL can represent.

    Fails closed on an unknown or symbolic shape.  It used to return True on the
    grounds that the C++ importer would check later; it does not -- 60 of the 68
    parsers in tidlParseRelax/ have no denial path at all, so this callback is
    the only gate.  Declining an op whose rank cannot be established costs a
    fallback to TVM; accepting it risks TIDL mis-parsing the tensor.  TIDL
    serializes concrete dimensions at import, so a symbolic dimension is
    declined even when the rank is known.
    """
    shape = _get_shape(expr)
    if shape is None:
        return False
    return len(shape) <= max_rank


def _get_tuple_len(expr) -> Optional[int]:
    """Length of a literal relax.Tuple, or None if not a literal tuple."""
    if isinstance(expr, tvm.relax.Tuple):
        return len(expr.fields)
    return None


def _get_prim_int(expr) -> Optional[int]:
    """Int value of a literal relax.PrimValue holding an IntImm, or None."""
    if isinstance(expr, tvm.relax.PrimValue) and isinstance(expr.value, tvm.tir.IntImm):
        return int(expr.value.value)
    return None


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
    if inp is None or not _check_dtype(inp):
        return False

    # rank check
    if inp is None or not _check_rank(inp, 4):
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
        if shape is None or len(shape) != 4:
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
    if data is None or not _check_dtype(data):
        return False
    return _check_rank(data, _TIDL_MAX_RANK)


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
        if shape is None or len(shape) != 4:
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
    """Validate reshape constraints for TIDL (ReshapeLayer).

    No mis-parse gap found here, unlike strided_slice/pad/concat: Relax's own
    ConvertNewShapeToExpr resolves every "0" (copy-from-input) and "-1"
    (infer) special value to a concrete dim before the shape ever reaches the
    call args, and both tidl_parse_relax_reshape.cpp and its Relay
    counterpart already decline a remaining non-constant dim.  What TIDL's
    generic ReshapeLayer allowlisting constraint adds -- unconditionally, not
    modelType-gated -- is a rank <= 6 ceiling on the target shape; add the
    same check here to avoid paying for calibration on an op that will be
    declined anyway.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False

    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call) or len(root.args) < 2:
        return False
    target_shape = root.args[1]
    if isinstance(target_shape, tvm.relax.ShapeExpr) and len(target_shape.values) > _TIDL_MAX_RANK:
        return False

    return True


def _check_matmul(ctx: PatternCheckContext) -> bool:
    """Validate matmul constraints for TIDL (InnerProduct layer).

    - Supported dtypes on data and weight
    """
    for key in ("data", "weight"):
        expr = ctx.annotated_expr.get(key)
        if expr is None or not _check_dtype(expr):
            return False
    return True


def _check_matmul_bias(ctx: PatternCheckContext) -> bool:
    """Validate fused matmul+bias constraints for TIDL (InnerProduct with bias).

    - Supported dtypes on all inputs
    """
    for key in ("data", "weight", "bias"):
        expr = ctx.annotated_expr.get(key)
        if expr is None or not _check_dtype(expr):
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
    """Validate permute_dims (transpose) constraints for TIDL (TransposeLayer).

    No mis-parse gap found here either: InferStructInfoPermuteDims already
    requires axes.size() == input ndim (ReportFatal otherwise) and validates
    the permutation itself, and tidl_parse_relax_permute_dims.cpp derives its
    numDim from axes.size() rather than the OOB-prone loop a prior version
    used (see that file's own comment on the SIGFPE bug it replaced). TIDL's
    generic TransposeLayer allowlisting constraint's rank <= 6 ceiling is
    unconditional (not modelType-gated); add the same check here to avoid
    paying for calibration on an op that will be declined anyway. Its
    firmware-version-gated width-position/batch-axis constraints are not
    replicated -- no evidence they apply the same way through Relax's
    metadata as through the ONNX-shape-inference path they were written for.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    if not _check_rank(data, _TIDL_MAX_RANK):
        return False
    return True


def _check_softmax(ctx: PatternCheckContext) -> bool:
    """Validate softmax constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return _check_rank(data, _TIDL_MAX_RANK)


def _check_concat(ctx: PatternCheckContext) -> bool:
    """Validate concat constraints for TIDL (ConcatLayer).

    ``data`` is the Tuple matched by the wildcard — it has TupleStructInfo,
    not a dtype.  Check dtype on the first Tuple field instead.

    ``axis=None`` ("flatten every input, then concatenate") is not the gap it
    looks like: InferStructInfoConcat requires every input to already be 1-D
    whenever axis is unset (ReportFatal otherwise), so by the time this check
    runs, "concat along axis 0" and "flatten-then-concat" are the same
    operation.  tidl_parse_relax_concat.cpp's ``attrs->axis.value_or(0)`` is
    therefore correct for that case, not a mis-parse.

    Decline concat along the batch axis on a >3-D tensor instead: TIDL's
    generic ConcatLayer allowlisting constraint (tidl_constraint_concat.h)
    declines this too, so catching it here just avoids paying for calibration
    first.
    """
    data = ctx.annotated_expr.get("data")
    if data is None:
        return False

    si = data.struct_info
    ndim = None
    if hasattr(si, "fields") and si.fields:
        # Tuple input: check dtype of the first field
        first_field = si.fields[0]
        dtype = getattr(first_field, "dtype", None)
        if dtype is not None and dtype not in _TIDL_SUPPORTED_DTYPES:
            return False
        shape = getattr(first_field, "shape", None)
        if shape is not None:
            ndim = len(shape)
    elif not _check_dtype(data):
        # Fallback for non-Tuple (shouldn't normally happen)
        return False

    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call):
        return False
    attrs = root.attrs
    axis = getattr(attrs, "axis", None) if attrs is not None else None
    axis_val = 0 if axis is None else int(axis)  # None means "the sole 1-D axis"

    if ndim:
        axis_norm = axis_val % ndim
        if axis_norm == 0 and ndim > 3:
            return False  # batch-axis concat on >3D — TIDL declines this

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
        if shape is None or len(shape) != 4:
            return False
    return True


def _check_sigmoid(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_tanh(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_clip(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_leakyrelu(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_prelu(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
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
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_squeeze(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_expand_dims(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_strided_slice(ctx: PatternCheckContext) -> bool:
    """Validate strided_slice constraints for TIDL (SliceLayer).

    tidl_parse_relax_strided_slice.cpp reads only field 0 of the axes/begin/end
    tuples and has no length check, so a multi-axis slice is silently
    mis-parsed (wrong axis/bounds) rather than declined -- this is the Relax
    counterpart of Relay's explicit "TIDL_RT only supports single slice axis"
    guard in tidl_parse_tvm_slice.cpp, which Relax's parser never got.  It also
    only declines a stride > 1; a negative or zero stride is silently
    accepted while the layer's stride field stays hardcoded to 1, i.e. TIDL
    executes a plain forward stride-1 slice regardless of what was requested.
    Both are correctness gaps, not just missing optimizations, so this check
    fails closed on tuple shape and reads the raw args rather than trusting
    the C++ side to catch them.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False

    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call) or len(root.args) < 4:
        return False

    axes, begin, end = root.args[1], root.args[2], root.args[3]
    for tup in (axes, begin, end):
        if _get_tuple_len(tup) != 1:
            return False  # TIDL SliceLayer only supports a single axis

    if len(root.args) >= 5:
        strides = root.args[4]
        if not isinstance(strides, tvm.relax.Tuple) or len(strides.fields) != 1:
            return False
        if _get_prim_int(strides.fields[0]) != 1:
            return False

    return True


def _check_cast(ctx: PatternCheckContext) -> bool:
    """Validate cast (astype) constraints for TIDL.

    A cast whose source or destination dtype TIDL does not represent cannot be
    offloaded, and the C++ parser will not catch it: tidl_parse_relax_cast.cpp
    has no denial path.  Both ends have to be checked -- the destination is the
    whole point of the op, and it is the end the pattern's wildcard hides.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data) or not _check_rank(data):
        return False
    root = ctx.annotated_expr.get("root")
    if root is None or not _check_dtype(root):
        return False
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
    if data is None or not _check_dtype(data):
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
            # Reject global reduce (no axis) — TIDL requires exactly one axis.
            if axis is None or len(axis) == 0:
                return False
            if len(axis) > 1:
                return False  # TIDL only supports single-axis reduction
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
    if data is None or not _check_dtype(data):
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
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_flatten(ctx: PatternCheckContext) -> bool:
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _check_pad(ctx: PatternCheckContext) -> bool:
    """Validate pad constraints for TIDL (PadLayer).

    tidl_parse_relax_pad.cpp only extracts H/W padding from fixed indices
    4-7 of the flat ``pad_width`` array, under the assumption that the input
    is exactly 4-D (``pad_width`` has 8 entries: N,C,H,W before/after pairs).
    If it is not, the ``pw.size() >= 8`` guard silently skips setting
    padT/padB/padL/padR, so a non-4-D pad is accepted and imported as a
    zero-padding no-op instead of being declined.  Worse, unlike its Relay
    counterpart (tidl_parse_tvm_pad.cpp), the Relax parser never inspects the
    N/C (batch/channel) pad amounts at all, so it never sets
    ``padType = TIDL_PadNonWidthHeight`` -- which means the generic
    tidlConstraintPad allowlisting backstop (which only declines on that
    flag) never fires either.  Non-zero N/C padding is therefore silently
    dropped end to end.  ``pad_mode != "constant"`` mirrors what that same
    backstop already declines (``TIDL_PadModeUnsupported``); checking it here
    too just avoids paying for calibration on an op that will fail import.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False

    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call):
        return False
    attrs = root.attrs
    if attrs is None:
        return False

    if getattr(attrs, "pad_mode", None) != "constant":
        return False

    pad_width = getattr(attrs, "pad_width", None)
    if pad_width is None or len(pad_width) != 8:
        return False  # parser assumes exactly 4-D input

    n_before, n_after, c_before, c_after = (int(v) for v in pad_width[:4])
    if n_before or n_after or c_before or c_after:
        return False  # N/C padding is silently dropped by the C++ parser

    return True


def _check_quantize(ctx: PatternCheckContext) -> bool:
    """Validate quantize constraints for TIDL.

    The pattern matches relax.quantize(data, scale, zp) with bare wildcards, and
    the C++ parser in tidlParseRelax/ has no denial path of its own, so anything
    accepted here is offloaded unconditionally.  Check what the parser assumes:
    a supported input and output dtype, and a rank TIDL can represent.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data) or not _check_rank(data):
        return False
    root = ctx.annotated_expr.get("root")
    if root is None or not _check_dtype(root):
        return False
    return True


def _check_dequantize(ctx: PatternCheckContext) -> bool:
    """Validate dequantize constraints for TIDL.

    Mirror of _check_quantize -- see the note there on why this has to be
    strict: the pattern is a bare wildcard and the C++ parser cannot decline.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data) or not _check_rank(data):
        return False
    root = ctx.annotated_expr.get("root")
    if root is None or not _check_dtype(root):
        return False
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


def _check_resize2d(ctx: PatternCheckContext) -> bool:
    """Validate resize2d constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is not None:
        if not _check_dtype(data):
            return False
        shape = _get_shape(data)
        if shape is None or len(shape) != 4:
            return False
    return True


def _check_take(ctx: PatternCheckContext) -> bool:
    """Validate take (gather) constraints for TIDL (GatherLayer).

    tidl_parse_relax_take.cpp treats ``constants[0]`` -- the first
    ``relax.Constant`` found by binding order in the composite body
    (TIDL_relaxFindConstants) -- as the gather *indices* unconditionally.
    When ``data`` is the constant instead (e.g. an embedding table gathered
    by variable token ids -- an extremely common pattern) and it happens to
    bind first, the parser mis-attributes it as indices and corrupts the
    layer's weights/axis interpretation.  Relay's own gather parser declines
    this exact case explicitly ("Data cannot be a constant. Only indices can
    be constant."); Relax's parser has no equivalent guard, so this check
    supplies it.

    Also mirrors two of the generic GatherLayer allowlisting constraint's
    checks (tidl_constraint_gather.h, unconditional -- not modelType-gated):
    data must be >= 2D, and indices must be effectively 1-D (only its last
    dimension may be non-singleton).  A third check in that file --
    "input shape of dimension higher than axis should be 1" -- looked at
    the same time and NOT ported here: on a full read it turned out to
    constrain dims *before* the axis, gated behind
    ``layer->optimized == 0 && axis != TIDL_DIM_WIDTH``, with two further
    exemptions (a reshape-equivalent case and a contiguous-constant-indices
    case treated as a slice) -- an initial misreading of it as "dims after
    axis must be singleton" was caught by test_take's hardware-adjacent
    partition test in test_tidl_layer_offload.py actually offloading
    take(axis=1) on a (1, 8, 16, 16) tensor.  Not porting it rather than
    risk porting it wrong again.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    if isinstance(data, tvm.relax.Constant):
        return False  # C++ parser would mis-attribute this as the indices constant

    shape = _get_shape(data)
    if shape is None or len(shape) < 2:
        return False  # GatherLayer requires >= 2D input

    indices = ctx.annotated_expr.get("indices")
    if indices is not None:
        idx_shape = _get_shape(indices)
        if idx_shape is not None and any(d != 1 for d in idx_shape[:-1]):
            return False  # GatherLayer only supports effectively-1D indices

    return True


def _check_unary(ctx: PatternCheckContext) -> bool:
    """Dtype check for parameterless unary math ops."""
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _resize2d_pattern() -> List[FusionPattern]:
    """Image resize (bilinear/nearest upsampling)."""
    data = wildcard()
    pat = is_op("relax.image.resize2d")(data, varg_default_wildcard=True)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.image.resize2d", pat, annotations, _check_resize2d)]


def _take_pattern() -> List[FusionPattern]:
    """Take / gather along axis."""
    data = wildcard()
    indices = wildcard()
    pat = is_op("relax.take")(data, indices)
    annotations = {"data": data, "indices": indices, "root": pat}
    return [FusionPattern("tidl.take", pat, annotations, _check_take)]


def _check_topk(ctx: PatternCheckContext) -> bool:
    """Validate topk constraints for TIDL (TopKLayer).

    ``sorted`` isn't a real gap: Relax's TopKAttrs has no such field at all
    (tidl_parse_relax_topk.cpp hardcodes ``sorted = 1``, matching the layer's
    only supported mode), and the generic TopKLayer allowlisting constraint's
    "sorted must be 1" check can therefore never fire for Relax.

    The axis constraint is real and unconditional (not modelType-gated,
    verified by reading tidl_constraint_topK.h): TIDL's TopKLayer only
    supports reducing over the width or height axis -- checked here as "one
    of the input's last two dims", which is what tidl_axis normalization
    (``user_axis + (TIDL_DIM_MAX - numDim)``) maps onto TIDL_DIM_WIDTH/
    HEIGHT regardless of the tensor's actual rank.  Declining early avoids
    paying for calibration on an op that will be declined at allowlisting
    anyway -- likely relevant for real models, since a topk over a
    class/box/detection axis (not spatial H/W) is a common pattern this
    would catch.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False

    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call) or root.attrs is None:
        return False
    shape = _get_shape(data)
    if shape is None:
        return False
    ndim = len(shape)
    axis = getattr(root.attrs, "axis", -1)
    axis_norm = int(axis) % ndim
    if axis_norm not in (ndim - 2, ndim - 1):
        return False  # TIDL TopKLayer only supports the width/height axis

    return True


def _topk_pattern() -> List[FusionPattern]:
    """TopK: return top-K values and/or indices along an axis."""
    data = wildcard()
    pat = is_op("relax.topk")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.topk", pat, annotations, _check_topk)]


def _check_split(ctx: PatternCheckContext) -> bool:
    """Validate split constraints for TIDL (SliceLayer, reused).

    tidl_parse_relax_split.cpp maps split onto TIDL_SliceLayer rather than
    TIDL_SplitLayer -- deliberately, since tidlConstraintSplit unconditionally
    declines TIDL_SplitLayer outright ("Not supported").  That dodge also
    means tidlConstraintSlice never runs for split either, and every one of
    its checks is gated to ``modelType == TFLITE/TFLITE_RT`` anyway, so
    nothing in the generic backstop protects Relax's split path.

    Both this parser and its Relay counterpart (tidl_parse_tvm_split.cpp)
    write ``slicePoints[0..N]`` with no bound check, where slicePoints is a
    fixed ``[TIDL_NUM_OUT_BUFS + 1]`` = 17-element array (itidl_ti.h).  A
    split into more than 16 outputs is therefore an out-of-bounds write in
    the C++ importer, not a decline -- on either side, this bug transfers
    unfixed.  Reject it here since nothing else will.
    """
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False

    root = ctx.annotated_expr.get("root")
    if root is None or not isinstance(root, tvm.relax.Call) or root.attrs is None:
        return False
    ios = getattr(root.attrs, "indices_or_sections", None)
    if isinstance(ios, tvm.tir.IntImm):
        num_outputs = int(ios.value)
    elif ios is not None and hasattr(ios, "__len__"):
        num_outputs = len(ios) + 1
    else:
        return False

    if num_outputs <= 0 or num_outputs > _TIDL_MAX_SPLIT_OUTPUTS:
        return False

    return True


def _split_pattern() -> List[FusionPattern]:
    """Split tensor along axis into N equal or variable slices."""
    data = wildcard()
    pat = is_op("relax.split")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.split", pat, annotations, _check_split)]


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
# Composite activation patterns (multi-op, no native Relax op)
# ---------------------------------------------------------------------------


def _elu_pattern() -> List[FusionPattern]:
    """ELU: add(multiply(-alpha, relu(1 - exp(x))), relu(x))."""
    data = wildcard()
    relu_x = is_op("relax.nn.relu")(data)
    exp_x = is_op("relax.exp")(data)
    sub = is_op("relax.subtract")(wildcard(), exp_x)
    relu_neg = is_op("relax.nn.relu")(sub)
    mul = is_op("relax.multiply")(wildcard(), relu_neg)
    pat = is_op("relax.add")(mul, relu_x)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.elu", pat, annotations, _check_unary)]


def _hard_sigmoid_pattern() -> List[FusionPattern]:
    """HardSigmoid: clip(x + 3, 0, 6) / 6."""
    data = wildcard()
    add_3 = is_op("relax.add")(data, wildcard())
    clipped = is_op("relax.clip")(add_3, wildcard(), wildcard())
    pat = is_op("relax.divide")(clipped, wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.hard_sigmoid", pat, annotations, _check_unary)]


def _hard_swish_pattern() -> List[FusionPattern]:
    """HardSwish: x * clip(x + 3, 0, 6) / 6."""
    data = wildcard()
    add_3 = is_op("relax.add")(data, wildcard())
    clipped = is_op("relax.clip")(add_3, wildcard(), wildcard())
    div_6 = is_op("relax.divide")(clipped, wildcard())
    pat = is_op("relax.multiply")(data, div_6)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.hard_swish", pat, annotations, _check_unary)]


def _mish_pattern() -> List[FusionPattern]:
    """Mish: x * tanh(log(1 + exp(x)))."""
    data = wildcard()
    exp_x = is_op("relax.exp")(data)
    add_1 = is_op("relax.add")(wildcard(), exp_x)
    log_sp = is_op("relax.log")(add_1)
    tanh_sp = is_op("relax.tanh")(log_sp)
    pat = is_op("relax.multiply")(data, tanh_sp)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.mish", pat, annotations, _check_unary)]


# ---------------------------------------------------------------------------
# Additional native-op patterns
# ---------------------------------------------------------------------------


def _check_depth_to_space(ctx: PatternCheckContext) -> bool:
    """Validate depth_to_space constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is not None:
        if not _check_dtype(data):
            return False
        shape = _get_shape(data)
        if shape is None or len(shape) != 4:
            return False
    return True


def _depth_to_space_pattern() -> List[FusionPattern]:
    """Depth-to-space (pixel shuffle)."""
    data = wildcard()
    pat = is_op("relax.nn.pixel_shuffle")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.depth_to_space", pat, annotations, _check_depth_to_space)]


def _expand_pattern() -> List[FusionPattern]:
    """Expand (broadcast_to)."""
    data = wildcard()
    pat = is_op("relax.broadcast_to")(data, wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.expand", pat, annotations, _check_unary)]


def _check_instance_norm(ctx: PatternCheckContext) -> bool:
    """Validate instance_norm constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is not None:
        if not _check_dtype(data):
            return False
        shape = _get_shape(data)
        if shape is None or len(shape) != 4:
            return False
    return True


def _instance_norm_pattern() -> List[FusionPattern]:
    """Instance normalization."""
    data = wildcard()
    gamma = wildcard()
    beta = wildcard()
    pat = is_op("relax.nn.instance_norm")(data, gamma, beta)
    annotations = {"data": data, "gamma": gamma, "beta": beta, "root": pat}
    return [FusionPattern("tidl.nn.instance_norm", pat, annotations, _check_instance_norm)]


def _scatter_elements_pattern() -> List[FusionPattern]:
    """Scatter elements (ONNX-style scatter with axis)."""
    data = wildcard()
    indices = wildcard()
    updates = wildcard()
    pat = is_op("relax.scatter_elements")(data, indices, updates)
    annotations = {"data": data, "indices": indices, "updates": updates, "root": pat}
    return [FusionPattern("tidl.scatter_elements", pat, annotations, _check_unary)]


def _check_scatter_nd(ctx: PatternCheckContext) -> bool:
    """Reject scalar inputs for scatter_nd (TIDL limitation)."""
    for key in ("data", "indices", "updates"):
        expr = ctx.annotated_expr.get(key)
        if expr is not None:
            shape = _get_shape(expr)
            if shape is not None and len(shape) == 0:
                return False
    data = ctx.annotated_expr.get("data")
    if data is None or not _check_dtype(data):
        return False
    return True


def _scatter_nd_pattern() -> List[FusionPattern]:
    """Scatter ND."""
    data = wildcard()
    indices = wildcard()
    updates = wildcard()
    pat = is_op("relax.scatter_nd")(data, indices, updates)
    annotations = {"data": data, "indices": indices, "updates": updates, "root": pat}
    return [FusionPattern("tidl.scatter_nd", pat, annotations, _check_scatter_nd)]


def _check_grid_sample(ctx: PatternCheckContext) -> bool:
    """Validate grid_sample constraints for TIDL."""
    data = ctx.annotated_expr.get("data")
    if data is not None:
        if not _check_dtype(data):
            return False
        shape = _get_shape(data)
        if shape is None or len(shape) != 4:
            return False
    return True


def _grid_sample_pattern() -> List[FusionPattern]:
    """Grid sample (image warping)."""
    data = wildcard()
    grid = wildcard()
    pat = is_op("relax.image.grid_sample")(data, grid)
    annotations = {"data": data, "grid": grid, "root": pat}
    return [FusionPattern("tidl.image.grid_sample", pat, annotations, _check_grid_sample)]


def _check_conv2d_transpose(ctx: PatternCheckContext) -> bool:
    """Validate conv2d_transpose constraints for TIDL."""
    inp = ctx.annotated_expr.get("input")
    if inp is not None:
        if not _check_dtype(inp):
            return False
        shape = _get_shape(inp)
        if shape is None or len(shape) != 4:
            return False
    return True


def _conv2d_transpose_patterns() -> List[FusionPattern]:
    """Conv2d transpose with optional bias."""
    patterns = []
    for with_bias in [True, False]:
        inp = wildcard()
        weight = wildcard()
        annotations: Dict[str, DFPattern] = {"input": inp, "weight": weight}
        out = is_op("relax.nn.conv2d_transpose")(inp, weight)
        annotations["root"] = out
        if with_bias:
            bias = wildcard()
            annotations["bias"] = bias
            out = is_op("relax.add")(out, bias)
        suffix = "_bias" if with_bias else ""
        name = f"tidl.nn.conv2d_transpose{suffix}"
        patterns.append(FusionPattern(name, out, annotations, _check_conv2d_transpose))
    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tidl_patterns() -> List[FusionPattern]:
    """Return all TIDL patterns ordered by priority (most specific first).

    More specific patterns (e.g. conv2d_bias_relu) appear first so that
    ``FuseOpsByPattern`` greedily matches the largest possible fused op.
    """
    patterns = [
        # Composite activations FIRST — they require specific multi-op
        # connectivity (e.g. ELU needs relu+exp on same data).  If
        # conv2d_relu matches first, it claims the relu and breaks
        # these patterns.  Safe because each requires a unique op
        # structure (exp+relu for ELU, add+clip+divide for HardSigmoid)
        # that a simple conv2d+relu model does not have.
        *_hard_swish_pattern(),       # contains hard_sigmoid subpattern
        *_hard_sigmoid_pattern(),     # contains add, clip, divide
        *_mish_pattern(),             # contains exp, add, log, tanh, multiply
        *_elu_pattern(),              # contains exp, subtract, relu, multiply, add
        # conv2d variants — most specific first
        *_conv2d_patterns(),
        # conv2d_transpose with optional bias
        *_conv2d_transpose_patterns(),
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
        *_instance_norm_pattern(),
        # dtype
        *_cast_pattern(),
        *_pad_pattern(),
        # reductions
        *_sum_pattern(),
        *_reduce_max_pattern(),
        *_reduce_min_pattern(),
        *_argmax_pattern(),
        *_argmin_pattern(),
        # advanced ops
        *_resize2d_pattern(),
        *_take_pattern(),
        *_topk_pattern(),
        *_split_pattern(),
        *_depth_to_space_pattern(),
        *_expand_pattern(),
        *_scatter_elements_pattern(),
        *_scatter_nd_pattern(),
        *_grid_sample_pattern(),
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
