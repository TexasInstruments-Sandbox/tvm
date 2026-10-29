#!/usr/bin/env python
"""Rejecting-case table and parse-level checks for the Relax eligibility rules.

Every rule in ``tidlParseRelax/tidl_parse_relax_declines.cpp`` (ported from the
former ``patterns.py`` ``_check_*`` functions) and the declines in the
individual parsers has an accepting baseline and at least one rejecting
variant here. Expected results come from TIDL's own constraint text, struct
limits and op semantics, never from a Relay run.

``ACCEPT`` cases also assert the parsed layer (``TIDL_relaxDescribeNode``):
a wrong axis or stride is *accepted* and only visible in the populated
struct, so those values are computed independently from the op's attributes.
"""

import json
import os

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.tidl import _gate, partition_for_tidl

pytestmark = [pytest.mark.quick]

C7X_MMA_TIDL_PATH = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
)
pytestmark += [
    pytest.mark.skipif(not os.path.isfile(RELAX_SO_PATH), reason="relax .so not built"),
]

F32 = "float32"


def _build(inputs, body):
    """Module with ``main(*inputs)`` returning ``body(bb, *vars)``."""
    params = [
        relax.Var(name, relax.TensorType(shape, dtype))
        for name, (shape, dtype) in inputs.items()
    ]
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            out = bb.emit_output(body(bb, *params))
        bb.emit_func_output(out)
    return bb.get()


def _const(shape, dtype=F32, value=1):
    return relax.const(np.full(shape, value, dtype=dtype))


def _x(shape=(1, 8, 16, 16), dtype=F32):
    return {"x": (shape, dtype)}


op = relax.op
nn = relax.op.nn

# (id, pattern, module, expected verdict, reason substring or describe-params dict,
# partition config). Accept cases give the expected ``params`` entries; reject
# cases a reason fragment. ``config`` carries ``tidl_options`` for rules that
# depend on them (tensor_bits, c7x_firmware_version).
CASES = []


def accept(cid, pattern, mod, config=None, **params):
    CASES.append(pytest.param(pattern, mod, True, params, config or {}, id=cid))


def reject(cid, pattern, mod, reason, config=None):
    CASES.append(pytest.param(pattern, mod, False, reason, config or {}, id=cid))


# --- conv2d ---------------------------------------------------------------
def _conv(shape=(1, 8, 16, 16), k=3, dtype=F32, strides=(1, 1), pad=(1, 1)):
    w = _const((16, shape[1], k, k), dtype)
    return _build(_x(shape, dtype), lambda bb, x: nn.conv2d(x, w, strides=strides, padding=pad))


accept(
    "conv2d-3x3",
    "tidl.nn.conv2d",
    _conv(),
    kernelH=3,
    kernelW=3,
    strideH=1,
    strideW=1,
    numGroups=1,
    numInChannels=8,
    numOutChannels=16,
    padT=1,
    padL=1,
)
accept(
    "conv2d-7x7-stride2",
    "tidl.nn.conv2d",
    _conv(k=7, strides=(2, 2), pad=(3, 3)),
    kernelH=7,
    kernelW=7,
    strideH=2,
    strideW=2,
)
reject("conv2d-kernel-6", "tidl.nn.conv2d", _conv(k=6, pad=(2, 2)), "kernel 6x6")
reject("conv2d-kernel-9", "tidl.nn.conv2d", _conv(k=9, pad=(4, 4)), "kernel 9x9")
reject("conv2d-unequal-strides", "tidl.nn.conv2d", _conv(strides=(1, 2)), "strides must be equal")
reject("conv2d-int32", "tidl.nn.conv2d", _conv(dtype="int32"), "data type")


# --- pooling --------------------------------------------------------------
def _pool(fn, size):
    return _build(_x(), lambda bb, x: fn(x, pool_size=size, strides=(2, 2)))


accept(
    "maxpool-2x2",
    "tidl.nn.max_pool2d",
    _pool(nn.max_pool2d, (2, 2)),
    kernelH=2,
    kernelW=2,
    strideH=2,
    strideW=2,
)
reject("maxpool-4x4", "tidl.nn.max_pool2d", _pool(nn.max_pool2d, (4, 4)), "pool size 4 > 3")
reject("avgpool-5x5", "tidl.nn.avg_pool2d", _pool(nn.avg_pool2d, (5, 5)), "pool size 5 > 3")


# --- relu / softmax / permute_dims: rank <= 6, dtype -----------------------
def _permute_reversed(x):
    return op.permute_dims(x, axes=list(range(x.ty.ndim))[::-1])


for name, pattern, fn in [
    ("relu", "tidl.nn.relu", nn.relu),
    ("softmax", "tidl.nn.softmax", lambda x: nn.softmax(x, axis=-1)),
    ("permute", "tidl.permute_dims", _permute_reversed),
]:
    # TIDL restricts >4D permutations that move the width axis, so the permute
    # baseline is 4-D; the others take the largest allowed rank it is wider than.
    accept(
        f"{name}-{'rank4' if name == 'permute' else 'rank5'}",
        pattern,
        _build(
            _x((1, 2, 3, 4) if name == "permute" else (1, 2, 3, 4, 5)),
            lambda bb, x, fn=fn: fn(x),
        ),
    )
    reject(
        f"{name}-rank7",
        pattern,
        _build(_x((1, 1, 2, 3, 4, 5, 6)), lambda bb, x, fn=fn: fn(x)),
        "rank must be known and <= 6",
    )
    reject(
        f"{name}-int32",
        pattern,
        # softmax is float-only in Relax, so its "unsupported dtype" is float16
        _build(
            _x(dtype="float16" if name == "softmax" else "int32"),
            lambda bb, x, fn=fn: fn(x),
        ),
        "data type on the data input",
    )

# TIDL numbers dims in a 6-D layout: HEIGHT=4, WIDTH=5 for the last two NCHW axes.
# A positive and the equivalent negative axis must parse identically (a positive
# axis used to be normalised with a rank read past the end of the shape vector).
for cid, shape, axis, tidl_axis in [
    ("softmax-width", (1, 8, 16, 16), 3, 5),
    ("softmax-width-neg", (1, 8, 16, 16), -1, 5),
    ("softmax-height", (1, 8, 16, 16), 2, 4),
    ("softmax-height-neg", (1, 8, 16, 16), -2, 4),
    ("softmax-width-rank3", (8, 16, 16), 2, 5),
]:
    accept(
        cid,
        "tidl.nn.softmax",
        _build(_x(shape), lambda bb, x, axis=axis: nn.softmax(x, axis=axis)),
        axis=tidl_axis,
    )
reject(
    "softmax-channel",
    "tidl.nn.softmax",
    _build(_x(), lambda bb, x: nn.softmax(x, axis=1)),
    "softmax along width and height",
)

# --- elementwise: dtype + exactly 4-D --------------------------------------
for opname, fn in [
    ("add", op.add),
    ("multiply", op.multiply),
    ("divide", op.divide),
    ("subtract", op.subtract),
    ("maximum", op.maximum),
    ("minimum", op.minimum),
]:
    two = {"a": ((1, 8, 16, 16), F32), "b": ((1, 8, 16, 16), F32)}
    accept(f"{opname}-4d", f"tidl.{opname}", _build(two, lambda bb, a, b, fn=fn: fn(a, b)))
    low = {"a": ((8, 16), F32), "b": ((8, 16), F32)}
    reject(
        f"{opname}-rank2",
        f"tidl.{opname}",
        _build(low, lambda bb, a, b, fn=fn: fn(a, b)),
        "4-D",
    )
    r5 = {"a": ((1, 2, 8, 16, 16), F32), "b": ((1, 2, 8, 16, 16), F32)}
    reject(
        f"{opname}-rank5",
        f"tidl.{opname}",
        _build(r5, lambda bb, a, b, fn=fn: fn(a, b)),
        "4-D",
    )
    ints = {"a": ((1, 8, 16, 16), "int32"), "b": ((1, 8, 16, 16), "int32")}
    reject(
        f"{opname}-int32",
        f"tidl.{opname}",
        _build(ints, lambda bb, a, b, fn=fn: fn(a, b)),
        "operand data type",
    )

# --- mean -----------------------------------------------------------------
accept(
    "mean-spatial", "tidl.mean", _build(_x(), lambda bb, x: op.mean(x, axis=[2, 3], keepdims=True))
)
accept(
    "mean-spatial-neg",
    "tidl.mean",
    _build(_x(), lambda bb, x: op.mean(x, axis=[-2, -1], keepdims=True)),
)
reject(
    "mean-channel-axis",
    "tidl.mean",
    _build(_x(), lambda bb, x: op.mean(x, axis=[1, 2], keepdims=True)),
    "spatial reduction",
)
reject(
    "mean-rank3",
    "tidl.mean",
    _build(_x((8, 16, 16)), lambda bb, x: op.mean(x, axis=[1, 2], keepdims=True)),
    "4-D",
)

# --- reshape --------------------------------------------------------------
accept(
    "reshape-rank4",
    "tidl.reshape",
    _build(_x(), lambda bb, x: op.reshape(x, relax.ShapeExpr([1, 16, 8, 16]))),
)
reject(
    "reshape-target-rank7",
    "tidl.reshape",
    _build(
        _x((1, 2, 2, 2, 2, 2, 2)),
        lambda bb, x: op.reshape(x, relax.ShapeExpr([1, 1, 2, 2, 2, 2, 4])),
    ),
    "<= 6",
)
reject(
    "reshape-int32",
    "tidl.reshape",
    _build(_x(dtype="int32"), lambda bb, x: op.reshape(x, relax.ShapeExpr([1, 16, 8, 16]))),
    "data type",
)

# --- matmul: dtype on every input ------------------------------------------
mm = {"a": ((4, 32), F32)}
accept(
    "matmul-const-weight", "tidl.matmul", _build(mm, lambda bb, a: op.matmul(a, _const((32, 16))))
)
reject(
    "matmul-int32",
    "tidl.matmul",
    _build({"a": ((4, 32), "int32")}, lambda bb, a: op.matmul(a, _const((32, 16), "int32"))),
    "data type",
)

# --- concat ---------------------------------------------------------------
cc = {"a": ((1, 8, 16, 16), F32), "b": ((1, 8, 16, 16), F32)}
accept(
    "concat-channel",
    "tidl.concat",
    _build(cc, lambda bb, a, b: op.concat([a, b], axis=1)),
    axis=3,
)
reject(
    "concat-batch-4d",
    "tidl.concat",
    _build(cc, lambda bb, a, b: op.concat([a, b], axis=0)),
    "batch axis",
)
accept(
    "concat-batch-3d",
    "tidl.concat",
    _build(
        {"a": ((2, 16, 16), F32), "b": ((2, 16, 16), F32)},
        lambda bb, a, b: op.concat([a, b], axis=0),
    ),
)

# --- dtype-only unary ops ---------------------------------------------------
for name, fn in [
    ("sigmoid", op.sigmoid),
    ("tanh", op.tanh),
    ("abs", op.abs),
    ("exp", op.exp),
    ("sqrt", op.sqrt),
    ("negative", op.negative),
    ("floor", op.floor),
    ("erf", op.erf),
    ("log", op.log),
]:
    accept(f"{name}-f32", f"tidl.{name}", _build(_x(), lambda bb, x, fn=fn: fn(x)))
    reject(
        f"{name}-f16",
        f"tidl.{name}",
        _build(_x(dtype="float16"), lambda bb, x, fn=fn: fn(x)),
        "data type on the data input",
    )


# --- strided_slice: single axis, stride 1 -----------------------------------
def _slice(axes, begin, end, strides=None, shape=(1, 8, 16, 16)):
    return _build(_x(shape), lambda bb, x: op.strided_slice(x, axes, begin, end, strides))


accept(
    "slice-single-axis",
    "tidl.strided_slice",
    _slice([2], [4], [12]),
    axis=4,  # TIDL_DIM_HEIGHT
    stride=1,
    slicePoint0=4,
    slicePoint1=12,
)
reject(
    "slice-multi-axis", "tidl.strided_slice", _slice([1, 2], [0, 0], [4, 8]), "single slice axis"
)
reject("slice-stride-2", "tidl.strided_slice", _slice([2], [0], [16], [2]), "stride 1")
reject("slice-negative-stride", "tidl.strided_slice", _slice([2], [15], [0], [-1]), "stride 1")

# --- cast -------------------------------------------------------------------
accept("cast-f32-int8", "tidl.cast", _build(_x(), lambda bb, x: op.astype(x, "int8")))
reject(
    "cast-to-int32",
    "tidl.cast",
    _build(_x(), lambda bb, x: op.astype(x, "int32")),
    "destination data type",
)
reject(
    "cast-rank5",
    "tidl.cast",
    _build(_x((1, 2, 8, 16, 16)), lambda bb, x: op.astype(x, "int8")),
    "rank must be known and <= 4",
)

# --- reductions ---------------------------------------------------------------
accept("sum-single", "tidl.sum", _build(_x(), lambda bb, x: op.sum(x, axis=[1], keepdims=True)))
reject(
    "sum-multi-axis",
    "tidl.sum",
    _build(_x(), lambda bb, x: op.sum(x, axis=[1, 2], keepdims=True)),
    "single-axis",
)
accept(
    "max-height",
    "tidl.reduce_max",
    _build(_x(), lambda bb, x: op.max(x, axis=[2], keepdims=True)),
    axis=4,
)
reject(
    "max-width",
    "tidl.reduce_max",
    _build(_x(), lambda bb, x: op.max(x, axis=[3], keepdims=True)),
    "height axis",
)
reject(
    "max-two-axes",
    "tidl.reduce_max",
    _build(_x(), lambda bb, x: op.max(x, axis=[2, 3], keepdims=True)),
    "exactly one",
)
reject(
    "min-global",
    "tidl.reduce_min",
    _build(_x(), lambda bb, x: op.min(x)),
    "scalar shape",
)
accept(
    "argmax-channel",
    "tidl.argmax",
    _build(_x(), lambda bb, x: op.argmax(x, axis=1, keepdims=True)),
)
reject(
    "argmax-height",
    "tidl.argmax",
    _build(_x(), lambda bb, x: op.argmax(x, axis=2, keepdims=True)),
    "channel axis",
)
reject(
    "argmin-no-keepdims",
    "tidl.argmin",
    _build(_x(), lambda bb, x: op.argmin(x, axis=1, keepdims=False)),
    "keepdims",
)


# --- pad --------------------------------------------------------------------------
def _pad(width, mode="constant", shape=(1, 8, 16, 16)):
    return _build(_x(shape), lambda bb, x: nn.pad(x, width, mode))


accept(
    "pad-hw",
    "tidl.nn.pad",
    _pad([0, 0, 0, 0, 1, 2, 3, 4]),
    padT=1,
    padB=2,
    padL=3,
    padR=4,
)
reject("pad-channel", "tidl.nn.pad", _pad([0, 0, 1, 1, 1, 1, 1, 1]), "batch/channel")
reject("pad-reflect", "tidl.nn.pad", _pad([0, 0, 0, 0, 1, 1, 1, 1], "reflect"), "constant padding")
reject("pad-rank3", "tidl.nn.pad", _pad([0, 0, 1, 1, 1, 1], shape=(8, 16, 16)), "4-D")

# --- take ---------------------------------------------------------------------------
idx = {"i": ((4,), "int32")}
accept(
    "take-variable-indices",
    "tidl.take",
    _build(
        {"x": ((1, 8, 16, 16), F32), "i": ((4,), "int32")},
        lambda bb, x, i: op.take(x, i, axis=3),
    ),
    axis=5,
)
# Constant indices: contiguous ones become a TIDL Slice, others stay a gather.
for cid, indices, dtype, axis in [
    ("take-const-indices-width", [0, 2, 5, 9], "int32", 3),
    ("take-const-indices-width-int64", [0, 2, 5, 9], "int64", 3),
    ("take-const-indices-height-contiguous", [2, 3, 4, 5], "int32", 2),
]:
    accept(
        cid,
        "tidl.take",
        _build(
            _x(),
            lambda bb, x, indices=indices, dtype=dtype, axis=axis: op.take(
                x, relax.const(np.array(indices, dtype)), axis=axis
            ),
        ),
        axis=axis + 2,
    )
reject(
    "take-const-indices-height-gather",
    "tidl.take",
    _build(_x(), lambda bb, x: op.take(x, relax.const(np.array([1, 3, 8, 15], "int32")), axis=2)),
    "Input shape of dimension higher than axis should be 1",
)
reject(
    "take-const-data",
    "tidl.take",
    _build(idx, lambda bb, i: op.take(_const((16, 8)), i, axis=0)),
    "data cannot be a constant",
)
reject(
    "take-rank1-data",
    "tidl.take",
    _build(
        {"x": ((16,), F32)},
        lambda bb, x: op.take(x, relax.const(np.arange(4, dtype="int32")), axis=0),
    ),
    "rank >= 2",
)
reject(
    "take-2d-indices",
    "tidl.take",
    _build(_x(), lambda bb, x: op.take(x, relax.const(np.zeros((2, 3), "int32")), axis=1)),
    "1-D indices",
)

# --- topk -----------------------------------------------------------------------------
accept("topk-last-axis", "tidl.topk", _build(_x(), lambda bb, x: op.topk(x, k=3, axis=-1)), K=3)
reject(
    "topk-channel-axis",
    "tidl.topk",
    _build(_x(), lambda bb, x: op.topk(x, k=3, axis=1)),
    "width/height",
)

# --- split -------------------------------------------------------------------------------
accept(
    "split-16",
    "tidl.split",
    _build(_x((1, 32, 16, 16)), lambda bb, x: op.split(x, 16, axis=1)),
    axis=3,
)
reject(
    "split-17",
    "tidl.split",
    _build(_x((1, 34, 16, 16)), lambda bb, x: op.split(x, 17, axis=1)),
    "outside 1..16",
)


# --- conv2d_transpose ------------------------------------------------------------------------
def _deconv(stride, k=3, pad=(1, 1), output_padding=(0, 0)):
    w = _const((8, 4, k, k))
    return _build(
        _x(),
        lambda bb, x: nn.conv2d_transpose(
            x, w, strides=stride, padding=pad, output_padding=output_padding
        ),
    )


# TIDL deconvolution upsamples by exactly the stride, so the output must be in * stride.
accept("deconv-4x4-s2-p1", "tidl.nn.conv2d_transpose", _deconv((2, 2), k=4))
accept("deconv-3x3-s2-outpad1", "tidl.nn.conv2d_transpose", _deconv((2, 2), output_padding=(1, 1)))
reject(
    "deconv-3x3-s2-size-mismatch",
    "tidl.nn.conv2d_transpose",
    _deconv((2, 2)),
    "times stride",
)
reject("deconv-stride1", "tidl.nn.conv2d_transpose", _deconv((1, 1)), "2x2 stride")


# ---------------------------------------------------------------------------
# Cases that trigger TIDL's own allowlisting constraints (the spec; see
# constraint_coverage.py). Each asserts the constraint text TIDL prints.
# ---------------------------------------------------------------------------
RANK_LE_6 = "Number of non-singleton variable input dimensions must be <= 6"
RANK_LE_4 = "Number of non-singleton variable input dimensions must be <= 4"
R7 = (2, 2, 2, 2, 2, 2, 2)  # 7 non-singleton dims

for _name, _fn in [
    ("abs", op.abs),
    ("sqrt", op.sqrt),
    ("exp", op.exp),
    ("log", op.log),
    ("erf", op.erf),
    ("floor", op.floor),
    ("negative", op.negative),
    ("sin", op.sin),
    ("cos", op.cos),
    ("tan", op.tan),
    ("sinh", op.sinh),
    ("cosh", op.cosh),
    ("asin", op.asin),
    ("acos", op.acos),
    ("atan", op.atan),
    ("asinh", op.asinh),
    ("sigmoid", op.sigmoid),
    ("tanh", op.tanh),
    ("nn.leakyrelu", lambda x: nn.leakyrelu(x, 0.1)),
    ("flatten", op.flatten),
]:
    reject(
        f"{_name}-7d-rank-limit",
        f"tidl.{_name}",
        _build(_x(R7), lambda bb, x, fn=_fn: fn(x)),
        RANK_LE_6
        if _name != "flatten"
        else "Number of non-singleton variable input dimensions must be <=",
    )

reject(
    "clip-min-positive",
    "tidl.clip",
    _build(_x(), lambda bb, x: op.clip(x, 1.0, 6.0)),
    "Only min <= 0 and max > 0 is supported",
)
reject(
    "argmax-rank5",
    "tidl.argmax",
    _build(_x((2, 8, 4, 4, 4)), lambda bb, x: op.argmax(x, axis=2, keepdims=True)),
    RANK_LE_4,
)
reject(
    "reduce-max-rank5",
    "tidl.reduce_max",
    _build(_x((2, 8, 4, 4, 4)), lambda bb, x: op.max(x, axis=[3], keepdims=True)),
    RANK_LE_4,
)
reject(
    "conv2d-3x3-stride3", "tidl.nn.conv2d", _conv(strides=(3, 3)), "Kernel size 3x3 with stride 3"
)
reject(
    "conv2d-depthwise-5x5-stride3",
    "tidl.nn.conv2d",
    _build(
        _x(),
        lambda bb, x: nn.conv2d(x, _const((8, 1, 5, 5)), strides=(3, 3), padding=(2, 2), groups=8),
    ),
    "Depthwise (Fully Grouped) convolution is only supported for",
)
reject(
    "deconv-dilation",
    "tidl.nn.conv2d_transpose",
    _build(
        _x(),
        lambda bb, x: nn.conv2d_transpose(
            x,
            _const((8, 4, 3, 3)),
            strides=(2, 2),
            padding=(2, 2),
            dilation=(2, 2),
            output_padding=(1, 1),  # out = 2 * in, so the size rule passes first
        ),
    ),
    "Only default dilations values (1,1) are supported",
)
reject(
    "matmul-two-variable",
    "tidl.matmul",
    _build(
        {"a": ((4, 32), F32), "b": ((32, 16), F32)},
        lambda bb, a, b: op.matmul(a, b),
    ),
    "no weight constant found",
)
reject(
    "matmul-rank5",
    "tidl.matmul",
    _build(_x((2, 3, 4, 5, 6)), lambda bb, x: op.matmul(x, _const((6, 7)))),
    RANK_LE_4,
)
reject(
    "prelu-variable-slope",
    "tidl.nn.prelu",
    _build(
        {"x": ((1, 8, 16, 16), F32), "slope": ((8,), F32)},
        lambda bb, x, slope: nn.prelu(x, slope, axis=1),
    ),
    "PRelu does not support variable slope",
)
accept(
    "prelu-channel-slope",
    "tidl.nn.prelu",
    _build(_x(), lambda bb, x: nn.prelu(x, _const((8,), value=0.25), axis=1)),
)
reject(
    "pad-nonzero-value",
    "tidl.nn.pad",
    _build(_x(), lambda bb, x: nn.pad(x, [0, 0, 0, 0, 1, 1, 1, 1], "constant", 1.0)),
    "constant_value = 0",
)
reject(
    "power-variable-exponent",
    "tidl.power",
    _build(
        {"x": ((1, 8, 16, 16), F32), "e": ((1, 8, 16, 16), F32)},
        lambda bb, x, e: op.power(x, e),
    ),
    "Power tensor must be a constant tensor",
)
reject(
    "transpose-5d-width-first",
    "tidl.permute_dims",
    _build(_x((1, 2, 3, 4, 5)), lambda bb, x: op.permute_dims(x, [4, 3, 2, 1, 0])),
    "permutations where the width dimension is mapped",
)
reject(
    "expand-dims-to-rank7",
    "tidl.expand_dims",
    _build(_x((1, 2, 3, 4, 5, 6)), lambda bb, x: op.expand_dims(x, 0)),
    "output rank 7 exceeds 6",
)
reject(
    "scatter-elements-channel-axis",
    "tidl.scatter_elements",
    _build(
        {
            "d": ((1, 8, 16, 16), F32),
            "i": ((1, 2, 16, 16), "int32"),
            "u": ((1, 2, 16, 16), F32),
        },
        lambda bb, d, i, u: op.scatter_elements(d, i, u, axis=1),
    ),
    "Only scatter along width axis is supported",
)
accept(
    "scatter-elements-width",
    "tidl.scatter_elements",
    _build(
        {
            "d": ((1, 8, 16, 16), F32),
            "i": ((1, 8, 16, 2), "int32"),
            "u": ((1, 8, 16, 2), F32),
        },
        lambda bb, d, i, u: op.scatter_elements(d, i, u, axis=3),
    ),
    axis=5,
)
reject(
    "scatter-nd-short-indices",
    "tidl.scatter_nd",
    _build(
        {
            "d": ((1, 8, 16, 16), F32),
            "i": ((2, 1), "int32"),
            "u": ((2, 8, 16, 16), F32),
        },
        lambda bb, d, i, u: op.scatter_nd(d, i, u),
    ),
    "indices tensor's last dimension must be at most",
)
reject(
    "resize-cubic",
    "tidl.image.resize2d",
    _build(_x(), lambda bb, x: op.image.resize2d(x, (32, 32), method="cubic")),
    "resize method 'cubic' is unsupported",
)
reject(
    "resize-downscale",
    "tidl.image.resize2d",
    _build(_x(), lambda bb, x: op.image.resize2d(x, (8, 8), method="linear")),
    "Scales < 1 are not supported",
)
reject(
    "grid-sample-bicubic",
    "tidl.image.grid_sample",
    _build(
        {"d": ((1, 8, 16, 16), F32), "g": ((1, 16, 16, 2), F32)},
        lambda bb, d, g: op.image.grid_sample(d, g, method="bicubic"),
    ),
    "Only nearest and linear/bilinear mode is supported",
)
reject(
    "layer-norm-channel-axis",
    "tidl.nn.layer_norm",
    _build(
        _x(),
        lambda bb, x: nn.layer_norm(x, _const((8,)), _const((8,)), axes=[1]),
    ),
    "Only supported across the width axis",
)

# --- conv2d / conv2d_transpose -----------------------------------------------------
BITS16 = {"tidl_options": {"tensor_bits": 16}}
reject(
    "conv2d-width-less-than-pad",
    "tidl.nn.conv2d",
    _build(
        _x((1, 8, 16, 4)), lambda bb, x: nn.conv2d(x, _const((16, 8, 3, 3)), padding=(1, 6, 1, 6))
    ),
    "Input width less than MAX(Pad Left, Pad Right)",
)
# The depthwise pad limit is the MMA width: 32 for 8-bit features on AM62A/AM67A, 16 for 16-bit.
for cid, pad, config, ok in [
    ("conv2d-depthwise-padw-17-8bit", 17, None, True),
    ("conv2d-depthwise-padw-33-8bit", 33, None, False),
    ("conv2d-depthwise-padw-17-16bit", 17, BITS16, False),
]:
    mod = _build(
        _x((1, 8, 16, 80)),
        lambda bb, x, pad=pad: nn.conv2d(
            x, _const((8, 1, 3, 3)), padding=(1, pad, 1, pad), groups=8
        ),
    )
    if ok:
        accept(cid, "tidl.nn.conv2d", mod, config, padL=pad, padR=pad)
    else:
        reject(cid, "tidl.nn.conv2d", mod, "Pad (along width) greater than the MMA width", config)
reject(
    "conv2d-5x5-stride4",
    "tidl.nn.conv2d",
    _build(
        _x((1, 8, 64, 64)),
        lambda bb, x: nn.conv2d(x, _const((16, 8, 5, 5)), strides=(4, 4), padding=(2, 2)),
    ),
    "Stride 4 is only supported with Kernel size 11x11",
)
reject(
    "conv2d-variable-weight",
    "tidl.nn.conv2d",
    _build(
        {"x": ((1, 8, 16, 16), F32), "w": ((16, 8, 3, 3), F32)},
        lambda bb, x, w: nn.conv2d(x, w, padding=(1, 1)),
    ),
    "no weight constant found",
)
reject(
    "deconv-16bit",
    "tidl.nn.conv2d_transpose",
    _deconv((2, 2), k=4),
    "16-bit Deconvolution is not suppported",
    BITS16,
)
reject(
    "deconv-variable-weight",
    "tidl.nn.conv2d_transpose",
    _build(
        {"x": ((1, 8, 16, 16), F32), "w": ((8, 4, 4, 4), F32)},
        lambda bb, x, w: nn.conv2d_transpose(x, w, strides=(2, 2), padding=(1, 1)),
    ),
    "the parser declined",
)

# --- depth_to_space ---------------------------------------------------------------
accept(
    "depth-to-space-4d",
    "tidl.nn.depth_to_space",
    _build(_x((1, 16, 8, 8)), lambda bb, x: nn.pixel_shuffle(x, 2)),
)
reject(
    "depth-to-space-rank5",
    "tidl.nn.depth_to_space",
    _build(_x((1, 2, 16, 8, 8)), lambda bb, x: nn.pixel_shuffle(x, 2)),
    "4-D",
)

# --- elementwise with a constant operand -------------------------------------------------
# A constant with one non-singleton dim on the channel axis must match the variable's channels.
reject(
    "add-channel-vector-broadcasts-channel",
    "tidl.add",
    _build(_x((1, 1, 16, 16)), lambda bb, x: op.add(x, _const((1, 8, 1, 1)))),
    "1D vector dimension should match with channel or width dimension",
)

# --- take (variable indices) ---------------------------------------------------------------
_take_inputs = {"x": ((1, 8, 16, 16), F32), "i": ((4,), "int32")}
reject(
    "take-variable-indices-height",
    "tidl.take",
    _build(_take_inputs, lambda bb, x, i: op.take(x, i, axis=2)),
    "Input shape of dimension higher than axis should be 1",
)
reject(
    "take-output-rank7",
    "tidl.take",
    _build(
        {"x": ((1, 1, 2, 2, 2, 2, 4), F32), "i": ((4,), "int32")},
        lambda bb, x, i: op.take(x, i, axis=6),
    ),
    "Number of output dimensions must be <= 6",
)


# --- grid_sample --------------------------------------------------------------------------
def _grid_sample(grid_shape=(1, 16, 16, 2), data_shape=(1, 8, 16, 16), **kw):
    return _build(
        {"d": (data_shape, F32), "g": (grid_shape, F32)},
        lambda bb, d, g: op.image.grid_sample(d, g, **kw),
    )


accept("grid-sample-bilinear-zeros", "tidl.image.grid_sample", _grid_sample())
reject(
    "grid-sample-3d-grid",
    "tidl.image.grid_sample",
    _grid_sample((1, 16, 16, 3)),
    "Only 2D grid indices are supported",
)
reject(
    "grid-sample-border",
    "tidl.image.grid_sample",
    _grid_sample(padding_mode="border"),
    "Only zero padding mode is supported",
)
reject(
    "grid-sample-nhwc",
    "tidl.image.grid_sample",
    _grid_sample(data_shape=(1, 16, 16, 8), layout="NHWC"),
    "only NCHW layout",
)

# --- matmul -------------------------------------------------------------------------------
for cid, bias, ok in [
    ("matmul-bias-n", (16,), True),
    ("matmul-bias-1xn", (1, 16), True),
    ("matmul-bias-mxn", (4, 16), False),
]:
    mod = _build(mm, lambda bb, a, bias=bias: op.add(op.matmul(a, _const((32, 16))), _const(bias)))
    if ok:
        accept(cid, "tidl.matmul_bias", mod)
    else:
        reject(
            cid, "tidl.matmul_bias", mod, "Bias tensor input should be a vector of [1, N] or [N]"
        )
reject(
    "matmul-1d-weight",
    "tidl.matmul",
    _build(mm, lambda bb, a: op.matmul(a, _const((32,)))),
    "weight must be at least 2D",
)

# --- layer_norm / instance_norm: constant gamma/beta, supported axes ---------------------------
_ln_x = {"x": ((1, 8, 16, 16), F32), "v": ((16,), F32)}
accept(
    "layer-norm-width",
    "tidl.nn.layer_norm",
    _build(_x(), lambda bb, x: nn.layer_norm(x, _const((16,)), _const((16,)), axes=[-1])),
)
reject(
    "layer-norm-variable-gamma",
    "tidl.nn.layer_norm",
    _build(_ln_x, lambda bb, x, v: nn.layer_norm(x, v, _const((16,)), axes=[3])),
    "gamma and beta must be constant",
)
reject(
    "layer-norm-variable-beta",
    "tidl.nn.layer_norm",
    _build(_ln_x, lambda bb, x, v: nn.layer_norm(x, _const((16,)), v, axes=[3])),
    "gamma and beta must be constant",
)
reject(
    "layer-norm-no-center",
    "tidl.nn.layer_norm",
    _build(
        _x(), lambda bb, x: nn.layer_norm(x, _const((16,)), _const((16,)), axes=[3], center=False)
    ),
    "center and scale must both be enabled",
)
reject(
    "layer-norm-height-width",
    "tidl.nn.layer_norm",
    _build(_x(), lambda bb, x: nn.layer_norm(x, _const((16, 16)), _const((16, 16)), axes=[2, 3])),
    "Only supported across the width axis",
)


def _instance_norm(channel_axis=1, axes=(2, 3), gamma=None, channels=8, **kw):
    inputs = _x() if gamma is None else {"x": ((1, 8, 16, 16), F32), "g": ((channels,), F32)}
    return _build(
        inputs,
        lambda bb, x, *g: nn.instance_norm(
            x,
            g[0] if g else _const((channels,)),
            _const((channels,)),
            channel_axis,
            list(axes),
            **kw,
        ),
    )


accept("instance-norm-spatial", "tidl.nn.instance_norm", _instance_norm())
accept("instance-norm-spatial-neg", "tidl.nn.instance_norm", _instance_norm(-3, (-2, -1)))
reject(
    "instance-norm-rank3",
    "tidl.nn.instance_norm",
    _build(_x((1, 8, 16)), lambda bb, x: nn.instance_norm(x, _const((8,)), _const((8,)), 1, [2])),
    "4-D",
)
reject(
    "instance-norm-variable-gamma",
    "tidl.nn.instance_norm",
    _instance_norm(gamma=True),
    "gamma and beta must be constant",
)
reject(
    "instance-norm-width-only",
    "tidl.nn.instance_norm",
    _instance_norm(axes=(3,)),
    "only channel_axis 1 with axes (2, 3)",
)
reject(
    "instance-norm-channel-last",
    "tidl.nn.instance_norm",
    _instance_norm(3, (1, 2), channels=16),
    "only channel_axis 1 with axes (2, 3)",
)
reject(
    "instance-norm-no-scale",
    "tidl.nn.instance_norm",
    _instance_norm(scale=False),
    "center and scale must both be enabled",
)

# --- pad / mean / pooling -----------------------------------------------------------------
reject(
    "pad-rank7",
    "tidl.nn.pad",
    _build(_x((1, 1, 1, 2, 2, 4, 4)), lambda bb, x: nn.pad(x, [0] * 10 + [1, 1, 1, 1])),
    "4-D",
)
# Global average pool is limited to a 1024-element plane.
accept(
    "mean-plane-1024",
    "tidl.mean",
    _build(_x((1, 8, 32, 32)), lambda bb, x: op.mean(x, axis=[2, 3], keepdims=True)),
    kernelH=0,
    kernelW=0,
)
reject(
    "mean-plane-1056",
    "tidl.mean",
    _build(_x((1, 8, 33, 32)), lambda bb, x: op.mean(x, axis=[2, 3], keepdims=True)),
    "GlobalAveragePool does not support plane size > 1024",
)
accept(
    "mean-spatial-no-keepdims",
    "tidl.mean",
    _build(_x(), lambda bb, x: op.mean(x, axis=[2, 3], keepdims=False)),
)
reject(
    "mean-rank5-three-axes",
    "tidl.mean",
    _build(_x((1, 8, 4, 16, 16)), lambda bb, x: op.mean(x, axis=[2, 3, 4], keepdims=True)),
    "4-D",
)
VALIDATED_POOL = "Pooling functionality is validated for 3x3,2x2 and 1x1 kernels"
for cid, fn, size, stride, ok in [
    ("maxpool-3x3-stride1", nn.max_pool2d, 3, 1, True),
    ("avgpool-2x2-stride2", nn.avg_pool2d, 2, 2, True),
    ("maxpool-2x2-stride1", nn.max_pool2d, 2, 1, False),
    ("avgpool-2x2-stride1", nn.avg_pool2d, 2, 1, False),
    ("maxpool-3x3-stride3", nn.max_pool2d, 3, 3, False),
]:
    pattern = "tidl.nn.max_pool2d" if fn is nn.max_pool2d else "tidl.nn.avg_pool2d"
    mod = _build(
        _x((1, 8, 18, 18)),
        lambda bb, x, fn=fn, size=size, stride=stride: fn(
            x, pool_size=(size, size), strides=(stride, stride)
        ),
    )
    if ok:
        accept(cid, pattern, mod, kernelH=size, strideH=stride)
    else:
        reject(cid, pattern, mod, VALIDATED_POOL)

# TIDL skips all constraints for a layer with no variable input, so constant-only
# composites are declined up front.
reject(
    "maxpool-constant-input",
    "tidl.nn.max_pool2d",
    _build(
        _x(),
        lambda bb, x: op.add(
            x, nn.max_pool2d(_const((1, 8, 32, 32)), pool_size=(2, 2), strides=(2, 2))
        ),
    ),
    "needs at least one variable input",
)

# --- power / prelu / reductions --------------------------------------------------------------
accept(
    "power-scalar-exponent",
    "tidl.power",
    _build(_x(), lambda bb, x: op.power(x, _const((), value=2))),
)
reject(
    "power-tensor-exponent",
    "tidl.power",
    _build(_x(), lambda bb, x: op.power(x, _const((1, 8, 1, 1), value=2))),
    "Size of the constant tensor must be 1",
)
reject(
    "power-int8-exponent",
    "tidl.power",
    _build(_x(dtype="int8"), lambda bb, x: op.power(x, _const((), "int8", value=2))),
    "the exponent must be float32",
)
reject(
    "prelu-slope-not-channels",
    "tidl.nn.prelu",
    _build(_x(), lambda bb, x: nn.prelu(x, _const((4,), value=0.25), axis=1)),
    "PRelu slope should be same or broadcast-able to input channel dimension",
)
reject(
    "max-no-keepdims",
    "tidl.reduce_max",
    _build(_x(), lambda bb, x: op.max(x, axis=[2], keepdims=False)),
    "Only keepdims = 1 is supported",
)


# --- reshape / expand with a shape variable -----------------------------------------------------
def _shape_var_module(fn, in_shape):
    s = relax.Var("s", relax.ShapeType(ndim=4))
    x = relax.Var("x", relax.TensorType(in_shape, F32))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, s]):
        with bb.dataflow():
            out = bb.emit_output(fn(x, s))
        bb.emit_func_output(out)
    return bb.get()


reject(
    "reshape-variable-shape",
    "tidl.reshape",
    _shape_var_module(op.reshape, (1, 8, 16, 16)),
    "unknown or non-constant shape",
)
reject(
    "expand-variable-shape",
    "tidl.expand",
    _shape_var_module(op.broadcast_to, (1, 1, 16, 16)),
    "unknown or non-constant shape",
)


# --- resize2d ------------------------------------------------------------------------------------
def _resize(size=(32, 32), shape=(1, 8, 16, 16), **kw):
    return _build(_x(shape), lambda bb, x: op.image.resize2d(x, size, **kw))


NEAREST = {"method": "nearest_neighbor"}
for cid, kw, reason in [
    (
        "resize-linear-asymmetric",
        {"coordinate_transformation_mode": "asymmetric"},
        "'linear' resize mode is not supported when coordinate_transformation_mode is 'asymmetric'",
    ),
    (
        "resize-nearest-asymmetric-round",
        {**NEAREST, "coordinate_transformation_mode": "asymmetric"},
        "Only 'floor' nearest mode is supported when coordinate_transformation_mode is",
    ),
    (
        "resize-nearest-half-pixel-floor",
        {**NEAREST, "coordinate_transformation_mode": "half_pixel", "rounding_method": "floor"},
        "Only 'round_prefer_ceil' nearest mode is supported",
    ),
    ("resize-scale-3", {"size": (48, 48)}, "Only Power of 2 scales are supported"),
    ("resize-unequal-scales", {"size": (32, 64)}, "Width and Height scale should be same"),
    (
        "resize-tf-half-pixel-for-nn",
        {**NEAREST, "coordinate_transformation_mode": "tf_half_pixel_for_nn"},
        "'tf_half_pixel_for_nn' is unsupported",
    ),
    (
        "resize-pytorch-half-pixel-1x1",
        {
            "size": (1, 1),
            "shape": (1, 8, 1, 1),
            "coordinate_transformation_mode": "pytorch_half_pixel",
        },
        "'pytorch_half_pixel' is unsupported",
    ),
    ("resize-nhwc", {"shape": (1, 16, 16, 8), "layout": "NHWC"}, "only NCHW layout"),
]:
    reject(cid, "tidl.image.resize2d", _resize(**kw), reason)
accept(
    "resize-nearest-asymmetric-floor",
    "tidl.image.resize2d",
    _resize(**NEAREST, coordinate_transformation_mode="asymmetric", rounding_method="floor"),
)
accept(
    "resize-nearest-half-pixel-round-prefer-ceil",
    "tidl.image.resize2d",
    _resize(**NEAREST, rounding_method="round_prefer_ceil"),
)
accept("resize-scale-4", "tidl.image.resize2d", _resize((64, 64)))


# --- scatter_elements / scatter_nd ----------------------------------------------------------------
def _scatter_elements(reduction="update", const_data=False):
    inputs = {"i": ((1, 8, 16, 2), "int32"), "u": ((1, 8, 16, 2), F32)}
    if const_data:
        return _build(
            inputs,
            lambda bb, i, u: op.scatter_elements(
                _const((1, 8, 16, 16), value=0), i, u, axis=3, reduction=reduction
            ),
        )
    return _build(
        {"d": ((1, 8, 16, 16), F32), **inputs},
        lambda bb, d, i, u: op.scatter_elements(d, i, u, axis=3, reduction=reduction),
    )


ONLY_NONE = "ScatterElements only supports the 'none' reduction type"
reject("scatter-elements-add", "tidl.scatter_elements", _scatter_elements("add"), ONLY_NONE)
# "mul" used to import as a plain update.
reject("scatter-elements-mul", "tidl.scatter_elements", _scatter_elements("mul"), ONLY_NONE)
reject(
    "scatter-elements-const-data",
    "tidl.scatter_elements",
    _scatter_elements(const_data=True),
    "data type",
)


def _scatter_nd(reduction="update", indices=(16, 3), updates=(16, 16)):
    return _build(
        {"d": ((1, 8, 16, 16), F32), "i": (indices, "int64"), "u": (updates, F32)},
        lambda bb, d, i, u: op.scatter_nd(d, i, u, reduction=reduction),
    )


accept(
    "scatter-nd-add", "tidl.scatter_nd", _scatter_nd("add"), reduction=0
)  # TIDL_ScatterElementsAdd
reject(
    "scatter-nd-mul",
    "tidl.scatter_nd",
    _scatter_nd("mul"),
    "ScatterND does not support the 'mul' reduction type",
)
reject(
    "scatter-nd-add-16bit",
    "tidl.scatter_nd",
    _scatter_nd("add"),
    "ScatterND does not support the 'add' reduction type in 16bit precision",
    BITS16,
)
reject(
    "scatter-nd-indices-rank5",
    "tidl.scatter_nd",
    _scatter_nd(indices=(2, 2, 2, 2, 3), updates=(2, 2, 2, 2, 16)),
    "number of non-singleton indices dimensions must be <= 4",
)
reject(
    "scatter-nd-add-updates-2-channels",
    "tidl.scatter_nd",
    _scatter_nd("add", indices=(2, 16, 3), updates=(2, 16, 16)),
    "Updates tensor should not have more than 1 channel",
)
accept(
    "scatter-nd-add-updates-1-channel",
    "tidl.scatter_nd",
    _scatter_nd("add", indices=(1, 16, 3), updates=(1, 16, 16)),
)

# --- strided_slice / split on the 6-D batch axis ------------------------------------------------
# TIDL's batch axis is dim 0 of its 6-D layout, i.e. axis 0 of a rank-6 tensor.
R6 = (2, 1, 2, 8, 16, 16)
reject(
    "slice-rank6-batch-axis",
    "tidl.strided_slice",
    _build(_x(R6), lambda bb, x: op.strided_slice(x, [0], [0], [1])),
    "Only supports non-batch dimension",
)
reject(
    "split-rank6-batch-axis",
    "tidl.split",
    _build(_x(R6), lambda bb, x: op.split(x, 2, axis=0)),
    "Only supports non-batch dimension",
)


# --- c7x firmware-version gates (Relay's parsers gate the same ops at the same versions) ----------
def _fw(version):
    return {"tidl_options": {"advanced_options:c7x_firmware_version": version}}


for pattern, mod, minimum, older in [
    ("tidl.log", _build(_x(), lambda bb, x: op.log(x)), "10_00_04_00", "10_00_03_00"),
    ("tidl.asin", _build(_x(), lambda bb, x: op.asin(x)), "10_00_04_00", "10_00_03_00"),
    ("tidl.asinh", _build(_x(), lambda bb, x: op.asinh(x)), "10_00_04_00", "10_00_03_00"),
    ("tidl.nn.instance_norm", _instance_norm(), "10_00_04_00", "10_00_03_00"),
    ("tidl.tan", _build(_x(), lambda bb, x: op.tan(x)), "10_01_06_00", "10_01_05_00"),
    ("tidl.cos", _build(_x(), lambda bb, x: op.cos(x)), "10_01_06_00", "10_01_05_00"),
    ("tidl.sinh", _build(_x(), lambda bb, x: op.sinh(x)), "10_01_06_00", "10_01_05_00"),
    ("tidl.cosh", _build(_x(), lambda bb, x: op.cosh(x)), "10_01_06_00", "10_01_05_00"),
    ("tidl.atan", _build(_x(), lambda bb, x: op.atan(x)), "10_01_06_00", "10_01_05_00"),
    ("tidl.negative", _build(_x(), lambda bb, x: op.negative(x)), "10_01_06_00", "10_01_05_00"),
]:
    name = pattern.rsplit(".", 1)[1]
    accept(f"{name}-firmware-{minimum}", pattern, mod, _fw(minimum))
    reject(f"{name}-firmware-{older}", pattern, mod, "current c7x firmware version", _fw(older))
# Gates in TIDL's own constraints rather than in a parser.
_sub = _build(
    {"a": ((1, 8, 16, 16), F32), "b": ((1, 8, 16, 16), F32)}, lambda bb, a, b: op.subtract(a, b)
)
accept("subtract-two-variables-firmware-10_01_06_00", "tidl.subtract", _sub, _fw("10_01_06_00"))
reject(
    "subtract-two-variables-firmware-10_01_05_00",
    "tidl.subtract",
    _sub,
    "Both inputs as variable are supported in Sub only if firmware version >= 10_01_06_00",
    _fw("10_01_05_00"),
)
for cid, shape, axes, reason in [
    (
        "transpose-6d-batch",
        (2, 1, 2, 3, 4, 5),
        [1, 0, 2, 3, 4, 5],
        "Transpose over batch dimension is only supported in firmware version >= 10_01_04_00",
    ),
    (
        "transpose-5d-width-to-height",
        (2, 2, 3, 4, 5),
        [0, 1, 2, 4, 3],
        "For firmware version < 10_01_04_00, only permutes are suported when number of",
    ),
]:
    mod = _build(_x(shape), lambda bb, x, axes=axes: op.permute_dims(x, axes))
    accept(f"{cid}-firmware-10_01_04_00", "tidl.permute_dims", mod, _fw("10_01_04_00"))
    reject(f"{cid}-firmware-10_01_03_00", "tidl.permute_dims", mod, reason, _fw("10_01_03_00"))
# Relay's argmax/argmin set selectLastIndex = 1 for firmware before 11_00_07_00.
for version, last in [("11_00_06_00", 1), ("11_00_07_00", 0)]:
    accept(
        f"argmax-firmware-{version}",
        "tidl.argmax",
        _build(_x(), lambda bb, x: op.argmax(x, axis=1, keepdims=True)),
        _fw(version),
        selectLastIndex=last,
    )


@pytest.fixture(scope="module", autouse=True)
def _library():
    _gate.load_relax_so(tidl_tools_path=os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools"))


def _run(mod, pattern, config=None):
    with _gate.recording() as records:
        partition_for_tidl(mod, dict(config or {}))
    matches = [r for r in records if r.pattern == pattern]
    assert matches, (
        f"{pattern} did not match (patterns tried: {sorted({r.pattern for r in records})})"
    )
    return matches[0]


@pytest.mark.parametrize("pattern,mod,verdict,expect,config", CASES)
def test_eligibility(pattern, mod, verdict, expect, config):
    res = _run(mod, pattern, config)
    if verdict:
        assert res.verdict, f"expected accept, library said: {res.reason}"
        d = json.loads(str(tvm.get_global_func("TIDL_relaxDescribeNode")(res.call)))
        for key, value in expect.items():
            assert d["params"].get(key) == value, (key, d["params"])
    else:
        assert not res.verdict, "expected the gate to decline"
        assert expect in res.reason, res.reason
