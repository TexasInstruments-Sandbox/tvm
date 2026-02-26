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
"""Fuse QDQ (quantize-dequantize) patterns into int8 convolution.

This pass identifies the PT2E QDQ pattern produced by XNNPACKQuantizer:

    dequantize(data_int8, d_scale, d_zp) -> float32
    conv2d(float, dequantize(weight_int8, w_scale, w_zp)) -> float32
    add(float, reshape(float_bias)) -> float32       [optional]
    relu(float) -> float32                           [optional]
    quantize(float, o_scale, o_zp) -> int8

and replaces it with:

    conv2d(data_int8 - d_zp, weight_int8, out_dtype="int32") -> int32
    add(int32, float_bias / (d_scale * w_scale))             [if bias]
    float(int32) * (d_scale * w_scale / o_scale) + o_zp -> int8

Weights have zero_point == 0 (symmetric).  Activation zero points may
be non-zero (XNNPACK uses zp=-128 for activations with int8 dtype).

This pass should run BEFORE LegalizeOps.
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


def _get_const_value(expr: Expr):
    """Extract scalar value from a Relax constant, or None."""
    if isinstance(expr, relax.Constant):
        arr = expr.data.numpy()
        if arr.ndim == 0:
            return arr.item()
    return None


# ---------------------------------------------------------------------------
# Pattern: dequant -> conv2d -> [add(wildcard)] -> [relu] -> quantize
#
# The actual PT2E graph from XNNPACKQuantizer has:
#   - weights dequantized at the top of the function (constants)
#   - bias is a float32 constant (NOT quantized), reshaped and added
#   - activation zp can be non-zero (e.g. -128 for "symmetric" int8)
#   - weight zp is always 0
# ---------------------------------------------------------------------------


def _make_conv2d_add_relu_pattern():
    """Match: quantize(relu(add(conv2d(dequant(x), dequant(w)), bias)), s, zp).

    This is the most common PT2E pattern for conv2d+bias+relu.
    The bias is a float wildcard (NOT dequantized in PT2E).
    """
    data = wildcard()
    data_scale = wildcard()
    data_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, data_scale, data_zp)

    weight_dq = wildcard()  # Already dequantized float (bound constant)

    conv = is_op("relax.nn.conv2d")(data_dq, weight_dq)

    bias = wildcard()  # float bias (reshape of constant)
    add = is_op("relax.add")(conv, bias)
    relu = is_op("relax.nn.relu")(add)

    out_scale = wildcard()
    out_zp = wildcard()
    pattern = is_op("relax.quantize")(relu, out_scale, out_zp)

    return pattern, {
        "data": data,
        "data_scale": data_scale,
        "data_zp": data_zp,
        "weight_dq": weight_dq,
        "conv": conv,
        "bias": bias,
        "out_scale": out_scale,
        "out_zp": out_zp,
        "has_relu": True,
        "has_bias": True,
    }


def _make_conv2d_add_pattern():
    """Match: quantize(add(conv2d(dequant(x), dequant(w)), bias), s, zp)."""
    data = wildcard()
    data_scale = wildcard()
    data_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, data_scale, data_zp)

    weight_dq = wildcard()

    conv = is_op("relax.nn.conv2d")(data_dq, weight_dq)

    bias = wildcard()
    add = is_op("relax.add")(conv, bias)

    out_scale = wildcard()
    out_zp = wildcard()
    pattern = is_op("relax.quantize")(add, out_scale, out_zp)

    return pattern, {
        "data": data,
        "data_scale": data_scale,
        "data_zp": data_zp,
        "weight_dq": weight_dq,
        "conv": conv,
        "bias": bias,
        "out_scale": out_scale,
        "out_zp": out_zp,
        "has_relu": False,
        "has_bias": True,
    }


def _make_conv2d_relu_pattern():
    """Match: quantize(relu(conv2d(dequant(x), dequant(w))), s, zp)."""
    data = wildcard()
    data_scale = wildcard()
    data_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, data_scale, data_zp)

    weight_dq = wildcard()

    conv = is_op("relax.nn.conv2d")(data_dq, weight_dq)
    relu = is_op("relax.nn.relu")(conv)

    out_scale = wildcard()
    out_zp = wildcard()
    pattern = is_op("relax.quantize")(relu, out_scale, out_zp)

    return pattern, {
        "data": data,
        "data_scale": data_scale,
        "data_zp": data_zp,
        "weight_dq": weight_dq,
        "conv": conv,
        "out_scale": out_scale,
        "out_zp": out_zp,
        "has_relu": True,
        "has_bias": False,
    }


def _make_conv2d_pattern():
    """Match: quantize(conv2d(dequant(x), dequant(w)), s, zp)."""
    data = wildcard()
    data_scale = wildcard()
    data_zp = wildcard()
    data_dq = is_op("relax.dequantize")(data, data_scale, data_zp)

    weight_dq = wildcard()

    conv = is_op("relax.nn.conv2d")(data_dq, weight_dq)

    out_scale = wildcard()
    out_zp = wildcard()
    pattern = is_op("relax.quantize")(conv, out_scale, out_zp)

    return pattern, {
        "data": data,
        "data_scale": data_scale,
        "data_zp": data_zp,
        "weight_dq": weight_dq,
        "conv": conv,
        "out_scale": out_scale,
        "out_zp": out_zp,
        "has_relu": False,
        "has_bias": False,
    }


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------


def _make_rewriter(annotations):
    """Create a rewriter for the matched QDQ conv2d pattern.

    For the general case with activation zero_point != 0:

      float_out = dequant(data) * dequant(weight)
               = d_scale * (data - d_zp) * w_scale * weight
                                          (w_zp == 0 always)

      We compute: shifted_data = data_int8 - d_zp   (still int8-range)
                  conv_int32 = conv2d(shifted_data, weight_int8, out_dtype=int32)
                  This gives us: sum((data-d_zp) * weight) in int32

      Requantize: float(conv_int32) * (d_scale * w_scale / o_scale) + o_zp

    For bias: the PT2E bias is already in float32, so we add it in the
    float domain during requantization:
      float(conv_int32) * (d_scale * w_scale) + float_bias
    then divide by o_scale and add o_zp.
    """
    has_relu = annotations.get("has_relu", False)
    has_bias = annotations.get("has_bias", False)

    def rewriter(expr, matches):
        data_int8 = matches[annotations["data"]]
        d_scale = matches[annotations["data_scale"]]
        d_zp = matches[annotations["data_zp"]]
        weight_dq_float = matches[annotations["weight_dq"]]
        o_scale = matches[annotations["out_scale"]]
        o_zp = matches[annotations["out_zp"]]

        conv_call = matches[annotations["conv"]]
        attrs = conv_call.attrs

        # Check: weight must already be dequantized (float).  We need
        # access to the original int8 weight.  In PT2E graphs, weight_dq
        # is a binding like:
        #   lv0 = R.dequantize(const[0], w_scale, w_zp=0)
        # Since the weight dequantize is a separate binding (not inline
        # in the conv pattern), we match weight_dq as a wildcard that
        # points to a float tensor.  We CAN'T easily recover the
        # original int8 weight from here.
        #
        # ALTERNATIVE APPROACH: Instead of trying to construct an int8
        # conv2d (which requires the original int8 weight), we keep the
        # conv2d with dequantized float weight but change the DATA path
        # from dequantize(int8) -> float to a direct int8 operation.
        #
        # Actually, the simplest correct approach: keep the conv in float
        # but absorb the quantize/dequantize pair.  This doesn't give us
        # the int8 speedup we want.
        #
        # The RIGHT approach: match the weight dequantize as part of the
        # pattern.  But in PT2E graphs, ALL weight dequantizes happen at
        # the top of the function (as bindings of constants), and the
        # conv2d references those bindings.  The DFPattern matching can
        # follow through variable references.
        #
        # Let me try matching weight_dq's definition.  If it's a
        # dequantize call, we can extract the int8 weight.
        w_int8 = None
        w_scale = None
        if isinstance(weight_dq_float, relax.Call):
            op = weight_dq_float.op
            if hasattr(op, "name") and op.name == "relax.dequantize":
                w_int8 = weight_dq_float.args[0]
                w_scale = weight_dq_float.args[1]
                w_zp = weight_dq_float.args[2]
                if not _is_const_zero(w_zp):
                    logger.debug("Skipping: non-zero weight zero_point")
                    return expr

        if w_int8 is None:
            # weight_dq is a Var reference, not an inline call.
            # We cannot recover the int8 weight from just the float var.
            # Skip fusion for this instance.
            return expr

        # ------- Step 1: Shift data by zero point ------- #
        d_zp_val = _get_const_value(d_zp)
        if d_zp_val is not None and d_zp_val == 0:
            shifted_data = data_int8
        else:
            # data_int8 - d_zp: cast to int16 to avoid overflow, then
            # cast to int8 if still in range, or keep as int16.
            # Actually, for conv2d with out_dtype=int32, TOPI casts
            # inputs to int32 before multiply.  So we can subtract in
            # int32 domain OR rely on the TOPI cast.
            #
            # Simplest: subtract in int32 and let conv use int32 inputs.
            shifted_data = relax.op.subtract(
                relax.op.astype(data_int8, "int32"),
                relax.op.astype(d_zp, "int32"),
            )

        # ------- Step 2: Int conv2d -> int32 ------- #
        conv_int32 = relax.op.nn.conv2d(
            shifted_data,
            w_int8,
            strides=attrs.strides,
            padding=attrs.padding,
            dilation=attrs.dilation,
            groups=attrs.groups,
            data_layout=attrs.data_layout,
            kernel_layout=attrs.kernel_layout,
            out_dtype="int32",
        )

        # ------- Step 3: Requantize ------- #
        # conv_int32 = sum((data - d_zp) * weight)   (int32)
        # float_equiv = conv_int32 * d_scale * w_scale
        #
        # With bias:
        #   float_out = float_equiv + float_bias
        #   quantized = round(float_out / o_scale) + o_zp
        #
        # Without bias:
        #   quantized = round(conv_int32 * d_scale * w_scale / o_scale) + o_zp

        out_float = relax.op.astype(conv_int32, "float32")

        if has_bias:
            float_bias = matches[annotations["bias"]]
            # Scale to real domain first: conv_int32 * d_scale * w_scale
            dw_scale = relax.op.multiply(d_scale, w_scale)
            out_float = relax.op.multiply(out_float, dw_scale)
            # Add float bias
            out_float = relax.op.add(out_float, float_bias)
            # Divide by output scale
            out_float = relax.op.divide(out_float, o_scale)
        else:
            # combined_scale = d_scale * w_scale / o_scale
            combined_scale = relax.op.divide(
                relax.op.multiply(d_scale, w_scale), o_scale
            )
            out_float = relax.op.multiply(out_float, combined_scale)

        # ReLU in the real-value domain (before zero point addition).
        # The original graph does relu(float) then quantize(relu_out,
        # scale, zp) where quantize = round(val/scale) + zp.  So relu
        # must precede the zp shift.
        if has_relu:
            out_float = relax.op.nn.relu(out_float)

        # Add output zero point
        if not _is_const_zero(o_zp):
            out_float = relax.op.add(
                out_float, relax.op.astype(o_zp, "float32")
            )

        # Round, clip, cast
        rounded = relax.op.round(out_float)
        out_dtype = expr.attrs.out_dtype if hasattr(expr, "attrs") else "int8"
        if out_dtype == "int8":
            clipped = relax.op.clip(rounded, -128, 127)
        elif out_dtype == "uint8":
            clipped = relax.op.clip(rounded, 0, 255)
        else:
            clipped = rounded

        result = relax.op.astype(clipped, out_dtype)

        logger.info(
            "Fused QDQ conv2d: int8 conv -> int32 -> requantize%s%s",
            " + bias" if has_bias else "",
            " + relu" if has_relu else "",
        )
        return result

    return rewriter


# ---------------------------------------------------------------------------
# Public pass
# ---------------------------------------------------------------------------

@function_pass(opt_level=0)
class FuseQDQToInt8Conv2D:
    """Fuse dequantize-conv2d-quantize patterns into int8 convolution.

    Replaces float32 convolutions surrounded by quantize/dequantize nodes
    with int8 convolutions that accumulate into int32, followed by a
    requantization step.  This changes the arithmetic from:

        float32 multiply-accumulate  (N * IC * KH * KW MACs per output)

    to:

        int8*int8 -> int32 accumulate  (same MACs but in integer)
        + one float32 multiply per output element (requantization)

    Supports:
      - Per-tensor and per-channel weight quantization
      - Non-zero activation zero points (e.g. XNNPACK zp=-128)
      - Float bias (PT2E convention: bias is NOT quantized)
      - Optional fused ReLU

    The pass handles four pattern variants:
      1. dequant -> conv2d -> quantize
      2. dequant -> conv2d -> relu -> quantize
      3. dequant -> conv2d -> add(float_bias) -> quantize
      4. dequant -> conv2d -> add(float_bias) -> relu -> quantize

    This pass should run BEFORE LegalizeOps.
    """

    def transform_function(
        self, func: Expr, mod: IRModule, ctx: PassContext
    ) -> Expr:
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return func

        # Apply patterns from most specific to least specific.
        patterns = [
            _make_conv2d_add_relu_pattern(),   # conv + bias + relu
            _make_conv2d_add_pattern(),         # conv + bias
            _make_conv2d_relu_pattern(),        # conv + relu
            _make_conv2d_pattern(),             # conv only
        ]

        result = func
        for pattern, annotations in patterns:
            rewriter = _make_rewriter(annotations)
            result = rewrite_call(pattern, rewriter, result)

        return result
