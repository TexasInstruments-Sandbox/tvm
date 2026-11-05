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

Defines the FusionPattern entries for ops supported by TI Deep Learning (TIDL)
on the C7x MMA accelerator.  Each pattern has a ``tidl.`` prefix and matches
*structure only*: which Relax ops, in which arrangement.

Whether a structural match can actually be offloaded (data types, ranks,
shapes, attributes, constant-vs-variable operands, hardware limits) is decided
in exactly one place: ``TIDL_relaxAllowNode`` in ``tidl_model_import_relax.so``.
Every pattern's check is ``_gate.tidl_gate(name)``, which rebuilds the composite
that fusion would create for the match and asks the library (see ``_gate.py``).
There is deliberately no Python constraint logic here; add new rules to the
Relax parsers in ``c7x-mma-tidl`` (``tidlParseRelax/``), not to this file.
"""

from typing import Dict, List

from tvm.relax.dpl.pattern import DFPattern, is_op, wildcard
from tvm.relax.transform import FusionPattern

from ..pattern_registry import register_patterns
from . import _gate

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
            patterns.append(FusionPattern(name, out, annotations, _gate.tidl_gate(name)))

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
        patterns.append(
            FusionPattern(pattern_name, pat, annotations, _gate.tidl_gate(pattern_name))
        )
    return patterns


def _relu_pattern() -> List[FusionPattern]:
    data = wildcard()
    pat = is_op("relax.nn.relu")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.relu", pat, annotations, _gate.tidl_gate("tidl.nn.relu"))]


def _add_pattern() -> List[FusionPattern]:
    """Element-wise add (e.g. residual connections)."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.add")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.add", pat, annotations, _gate.tidl_gate("tidl.add"))]


def _mean_pattern() -> List[FusionPattern]:
    """Mean reduction (global avg pool in ResNet-18)."""
    data = wildcard()
    pat = is_op("relax.mean")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.mean", pat, annotations, _gate.tidl_gate("tidl.mean"))]


def _reshape_pattern() -> List[FusionPattern]:
    """Reshape (flatten before FC layer)."""
    data = wildcard()
    pat = is_op("relax.reshape")(data, wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.reshape", pat, annotations, _gate.tidl_gate("tidl.reshape"))]


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
    patterns.append(
        FusionPattern(
            "tidl.matmul_bias", pat_mb, annotations_mb, _gate.tidl_gate("tidl.matmul_bias")
        )
    )

    # Standalone matmul (lower priority)
    data_m = wildcard()
    weight_m = wildcard()
    pat_m = is_op("relax.matmul")(data_m, weight_m)
    annotations_m = {"data": data_m, "weight": weight_m, "root": pat_m}
    patterns.append(
        FusionPattern("tidl.matmul", pat_m, annotations_m, _gate.tidl_gate("tidl.matmul"))
    )

    return patterns


def _multiply_pattern() -> List[FusionPattern]:
    """Element-wise multiply (gate * up in transformers)."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.multiply")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.multiply", pat, annotations, _gate.tidl_gate("tidl.multiply"))]


def _permute_dims_pattern() -> List[FusionPattern]:
    """Transpose / permute dimensions."""
    data = wildcard()
    pat = is_op("relax.permute_dims")(data)
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern("tidl.permute_dims", pat, annotations, _gate.tidl_gate("tidl.permute_dims"))
    ]


def _softmax_pattern() -> List[FusionPattern]:
    """Softmax (attention scores in transformers)."""
    data = wildcard()
    pat = is_op("relax.nn.softmax")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.softmax", pat, annotations, _gate.tidl_gate("tidl.nn.softmax"))]


def _concat_pattern() -> List[FusionPattern]:
    """Concatenation (RoPE, multi-head assembly)."""
    # concat takes a Tuple, match it with a wildcard
    data = wildcard()
    pat = is_op("relax.concat")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.concat", pat, annotations, _gate.tidl_gate("tidl.concat"))]


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
    return [FusionPattern("tidl.sigmoid", pat, annotations, _gate.tidl_gate("tidl.sigmoid"))]


def _tanh_pattern() -> List[FusionPattern]:
    """Tanh activation."""
    data = wildcard()
    pat = is_op("relax.tanh")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.tanh", pat, annotations, _gate.tidl_gate("tidl.tanh"))]


def _clip_pattern() -> List[FusionPattern]:
    """Standalone clip (e.g. relu6 = clip(x, 0, 6))."""
    data = wildcard()
    pat = is_op("relax.clip")(data, wildcard(), wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.clip", pat, annotations, _gate.tidl_gate("tidl.clip"))]


def _leakyrelu_pattern() -> List[FusionPattern]:
    """Leaky ReLU activation."""
    data = wildcard()
    pat = is_op("relax.nn.leakyrelu")(data)
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern("tidl.nn.leakyrelu", pat, annotations, _gate.tidl_gate("tidl.nn.leakyrelu"))
    ]


def _prelu_pattern() -> List[FusionPattern]:
    """PReLU activation (parametric ReLU with learnable slope)."""
    data = wildcard()
    alpha = wildcard()
    pat = is_op("relax.nn.prelu")(data, alpha)
    annotations = {"data": data, "alpha": alpha, "root": pat}
    return [FusionPattern("tidl.nn.prelu", pat, annotations, _gate.tidl_gate("tidl.nn.prelu"))]


def _divide_pattern() -> List[FusionPattern]:
    """Element-wise divide."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.divide")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.divide", pat, annotations, _gate.tidl_gate("tidl.divide"))]


def _subtract_pattern() -> List[FusionPattern]:
    """Element-wise subtract."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.subtract")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.subtract", pat, annotations, _gate.tidl_gate("tidl.subtract"))]


def _maximum_pattern() -> List[FusionPattern]:
    """Element-wise maximum."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.maximum")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.maximum", pat, annotations, _gate.tidl_gate("tidl.maximum"))]


def _minimum_pattern() -> List[FusionPattern]:
    """Element-wise minimum."""
    lhs = wildcard()
    rhs = wildcard()
    pat = is_op("relax.minimum")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": pat}
    return [FusionPattern("tidl.minimum", pat, annotations, _gate.tidl_gate("tidl.minimum"))]


def _layer_norm_pattern() -> List[FusionPattern]:
    """Layer normalization (transformers)."""
    data = wildcard()
    gamma = wildcard()
    beta = wildcard()
    pat = is_op("relax.nn.layer_norm")(data, gamma, beta)
    annotations = {"data": data, "gamma": gamma, "beta": beta, "root": pat}
    return [
        FusionPattern("tidl.nn.layer_norm", pat, annotations, _gate.tidl_gate("tidl.nn.layer_norm"))
    ]


def _flatten_pattern() -> List[FusionPattern]:
    """Flatten (before FC layer)."""
    data = wildcard()
    pat = is_op("relax.flatten")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.flatten", pat, annotations, _gate.tidl_gate("tidl.flatten"))]


def _squeeze_pattern() -> List[FusionPattern]:
    """Squeeze (remove size-1 dimensions)."""
    data = wildcard()
    pat = is_op("relax.squeeze")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.squeeze", pat, annotations, _gate.tidl_gate("tidl.squeeze"))]


def _expand_dims_pattern() -> List[FusionPattern]:
    """Expand dims (insert size-1 dimension)."""
    data = wildcard()
    pat = is_op("relax.expand_dims")(data)
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern("tidl.expand_dims", pat, annotations, _gate.tidl_gate("tidl.expand_dims"))
    ]


def _strided_slice_pattern() -> List[FusionPattern]:
    """Strided slice (tensor slicing).

    relax.strided_slice takes 4 args: (data, begin, end, strides)
    where begin/end/strides are Tuples of PrimValues.
    """
    data = wildcard()
    pat = is_op("relax.strided_slice")(data, varg_default_wildcard=True)
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern("tidl.strided_slice", pat, annotations, _gate.tidl_gate("tidl.strided_slice"))
    ]


def _cast_pattern() -> List[FusionPattern]:
    """Cast (dtype conversion via astype)."""
    data = wildcard()
    pat = is_op("relax.astype")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.cast", pat, annotations, _gate.tidl_gate("tidl.cast"))]


def _pad_pattern() -> List[FusionPattern]:
    """Pad (spatial padding)."""
    data = wildcard()
    pat = is_op("relax.nn.pad")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.nn.pad", pat, annotations, _gate.tidl_gate("tidl.nn.pad"))]


def _sum_pattern() -> List[FusionPattern]:
    """Sum reduction."""
    data = wildcard()
    pat = is_op("relax.sum")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.sum", pat, annotations, _gate.tidl_gate("tidl.sum"))]


def _reduce_max_pattern() -> List[FusionPattern]:
    """Max reduction (HEIGHT axis only per TIDL ReduceLayer constraint)."""
    data = wildcard()
    pat = is_op("relax.max")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.reduce_max", pat, annotations, _gate.tidl_gate("tidl.reduce_max"))]


def _reduce_min_pattern() -> List[FusionPattern]:
    """Min reduction (HEIGHT axis only per TIDL ReduceLayer constraint)."""
    data = wildcard()
    pat = is_op("relax.min")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.reduce_min", pat, annotations, _gate.tidl_gate("tidl.reduce_min"))]


def _argmax_pattern() -> List[FusionPattern]:
    """Argmax (index of maximum value along an axis)."""
    data = wildcard()
    pat = is_op("relax.argmax")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.argmax", pat, annotations, _gate.tidl_gate("tidl.argmax"))]


def _argmin_pattern() -> List[FusionPattern]:
    """Argmin (index of minimum value along an axis)."""
    data = wildcard()
    pat = is_op("relax.argmin")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.argmin", pat, annotations, _gate.tidl_gate("tidl.argmin"))]


def _resize2d_pattern() -> List[FusionPattern]:
    """Image resize (bilinear/nearest upsampling)."""
    data = wildcard()
    pat = is_op("relax.image.resize2d")(data, varg_default_wildcard=True)
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern(
            "tidl.image.resize2d", pat, annotations, _gate.tidl_gate("tidl.image.resize2d")
        )
    ]


def _take_pattern() -> List[FusionPattern]:
    """Take / gather along axis."""
    data = wildcard()
    indices = wildcard()
    pat = is_op("relax.take")(data, indices)
    annotations = {"data": data, "indices": indices, "root": pat}
    return [FusionPattern("tidl.take", pat, annotations, _gate.tidl_gate("tidl.take"))]


def _topk_pattern() -> List[FusionPattern]:
    """TopK: return top-K values and/or indices along an axis."""
    data = wildcard()
    pat = is_op("relax.topk")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.topk", pat, annotations, _gate.tidl_gate("tidl.topk"))]


def _split_pattern() -> List[FusionPattern]:
    """Split tensor along axis into N equal or variable slices."""
    data = wildcard()
    pat = is_op("relax.split")(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.split", pat, annotations, _gate.tidl_gate("tidl.split"))]


def _math_unary_pattern(op_name: str, composite_name: str) -> List[FusionPattern]:
    """Helper: create a pattern for a parameterless unary math op."""
    data = wildcard()
    pat = is_op(op_name)(data)
    annotations = {"data": data, "root": pat}
    return [FusionPattern(composite_name, pat, annotations, _gate.tidl_gate(composite_name))]


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
    return [FusionPattern("tidl.power", pat, annotations, _gate.tidl_gate("tidl.power"))]


def _quantize_pattern() -> List[FusionPattern]:
    data = wildcard()
    scale = wildcard()
    zp = wildcard()
    pat = is_op("relax.quantize")(data, scale, zp)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.quantize", pat, annotations, _gate.tidl_gate("tidl.quantize"))]


def _dequantize_pattern() -> List[FusionPattern]:
    data = wildcard()
    scale = wildcard()
    zp = wildcard()
    pat = is_op("relax.dequantize")(data, scale, zp)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.dequantize", pat, annotations, _gate.tidl_gate("tidl.dequantize"))]


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
    return [FusionPattern("tidl.elu", pat, annotations, _gate.tidl_gate("tidl.elu"))]


def _hard_sigmoid_pattern() -> List[FusionPattern]:
    """HardSigmoid: clip(x + 3, 0, 6) / 6."""
    data = wildcard()
    add_3 = is_op("relax.add")(data, wildcard())
    clipped = is_op("relax.clip")(add_3, wildcard(), wildcard())
    pat = is_op("relax.divide")(clipped, wildcard())
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern("tidl.hard_sigmoid", pat, annotations, _gate.tidl_gate("tidl.hard_sigmoid"))
    ]


def _hard_swish_pattern() -> List[FusionPattern]:
    """HardSwish: x * clip(x + 3, 0, 6) / 6."""
    data = wildcard()
    add_3 = is_op("relax.add")(data, wildcard())
    clipped = is_op("relax.clip")(add_3, wildcard(), wildcard())
    div_6 = is_op("relax.divide")(clipped, wildcard())
    pat = is_op("relax.multiply")(data, div_6)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.hard_swish", pat, annotations, _gate.tidl_gate("tidl.hard_swish"))]


def _mish_pattern() -> List[FusionPattern]:
    """Mish: x * tanh(log(1 + exp(x)))."""
    data = wildcard()
    exp_x = is_op("relax.exp")(data)
    add_1 = is_op("relax.add")(wildcard(), exp_x)
    log_sp = is_op("relax.log")(add_1)
    tanh_sp = is_op("relax.tanh")(log_sp)
    pat = is_op("relax.multiply")(data, tanh_sp)
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.mish", pat, annotations, _gate.tidl_gate("tidl.mish"))]


# ---------------------------------------------------------------------------
# Additional native-op patterns
# ---------------------------------------------------------------------------


def _depth_to_space_pattern() -> List[FusionPattern]:
    """Depth-to-space (pixel shuffle)."""
    data = wildcard()
    pat = is_op("relax.nn.pixel_shuffle")(data)
    annotations = {"data": data, "root": pat}
    return [
        FusionPattern(
            "tidl.nn.depth_to_space", pat, annotations, _gate.tidl_gate("tidl.nn.depth_to_space")
        )
    ]


def _expand_pattern() -> List[FusionPattern]:
    """Expand (broadcast_to)."""
    data = wildcard()
    pat = is_op("relax.broadcast_to")(data, wildcard())
    annotations = {"data": data, "root": pat}
    return [FusionPattern("tidl.expand", pat, annotations, _gate.tidl_gate("tidl.expand"))]


def _instance_norm_pattern() -> List[FusionPattern]:
    """Instance normalization."""
    data = wildcard()
    gamma = wildcard()
    beta = wildcard()
    pat = is_op("relax.nn.instance_norm")(data, gamma, beta)
    annotations = {"data": data, "gamma": gamma, "beta": beta, "root": pat}
    return [
        FusionPattern(
            "tidl.nn.instance_norm", pat, annotations, _gate.tidl_gate("tidl.nn.instance_norm")
        )
    ]


def _scatter_elements_pattern() -> List[FusionPattern]:
    """Scatter elements (ONNX-style scatter with axis)."""
    data = wildcard()
    indices = wildcard()
    updates = wildcard()
    pat = is_op("relax.scatter_elements")(data, indices, updates)
    annotations = {"data": data, "indices": indices, "updates": updates, "root": pat}
    return [
        FusionPattern(
            "tidl.scatter_elements", pat, annotations, _gate.tidl_gate("tidl.scatter_elements")
        )
    ]


def _scatter_nd_pattern() -> List[FusionPattern]:
    """Scatter ND."""
    data = wildcard()
    indices = wildcard()
    updates = wildcard()
    pat = is_op("relax.scatter_nd")(data, indices, updates)
    annotations = {"data": data, "indices": indices, "updates": updates, "root": pat}
    return [FusionPattern("tidl.scatter_nd", pat, annotations, _gate.tidl_gate("tidl.scatter_nd"))]


def _grid_sample_pattern() -> List[FusionPattern]:
    """Grid sample (image warping) with a constant grid.

    Relax 0.27 takes the grid as [N, 2, H_out, W_out]; TIDL's GridSample layer reads
    [N, H_out, W_out, 2].  The importer transposes a constant grid into TIDL's layout and
    declines a variable one, which would need a permute in front of the composite, and
    FuseOpsByPattern would then make the grid the composite's first parameter (it orders
    parameters by first use), the wrong order for TIDL.
    """
    data = wildcard()
    grid = wildcard()
    pat = is_op("relax.image.grid_sample")(data, grid)
    annotations = {"data": data, "grid": grid, "root": pat}
    return [
        FusionPattern(
            "tidl.image.grid_sample", pat, annotations, _gate.tidl_gate("tidl.image.grid_sample")
        )
    ]


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
        patterns.append(FusionPattern(name, out, annotations, _gate.tidl_gate(name)))
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
        *_hard_swish_pattern(),  # contains hard_sigmoid subpattern
        *_hard_sigmoid_pattern(),  # contains add, clip, divide
        *_mish_pattern(),  # contains exp, add, log, tanh, multiply
        *_elu_pattern(),  # contains exp, subtract, relu, multiply, add
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
register_patterns(_tidl_patterns)  # pyright: ignore[reportArgumentType]
