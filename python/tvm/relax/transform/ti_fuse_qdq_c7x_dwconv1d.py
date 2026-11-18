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
"""Lower int8 depthwise 1-D convolutions to the c7x_int8_dwconv1d kernel.

Runs right after FuseQDQToInt8Conv2D and matches what it emits for a
symmetric QDQ conv (no bias, no ReLU, zero points 0):

    astype(clip(round(astype(conv2d(x_i8, w_i8, out_dtype="int32"), "float32")
                       * scale), -128, 127), "int8")

When the conv is depthwise (groups == C_in == C_out, channel multiplier 1),
NCHW on a [1, C, 1, T] tensor with a 1xK kernel, stride 1 or 2 along T, the
chain becomes

    call_extern c7x_int8_dwconv1d(x, w[C, K], scale[C], y, C, T_in, T_out, K,
                                  stride, dilation, pad_left)

Left as it is, the chain lowers to generated C that keeps a padded copy of the
input and int32 and float intermediates in DDR and makes a pass over them for
each requantize step; the kernel keeps its accumulators in vector registers.
Matching after FuseQDQToInt8Conv2D rather than on the QDQ graph leaves every
legality decision (zero points, per-channel input scales) to that pass, and
the kernel computes exactly the arithmetic its chain does, so outputs don't
change.

The scale is the constant expression FuseQDQToInt8Conv2D builds,
(d_scale * w_scale) / o_scale with reshapes; it is evaluated here in float32,
as constant folding would evaluate it.

Kernel: src/runtime/ti_dsp/kernels/c7x_dwconv1d.cpp
"""

import logging

import numpy as np

import tvm
from tvm import relax, te, tirx
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import find_composite_span, propagate_span
from .ti_const_utils import ti_tensor_const

logger = logging.getLogger(__name__)

_COMPOSITE = "c7x_dwconv1d.int8"

# Must match c7x_dwconv1d.h.
_MAX_LINE = 16384
_MAX_TAPS = 1024
_BLOCK = 32


def _line_bytes(t_out: int, k: int, stride: int, dilation: int) -> int:
    """Line buffer one channel needs; line_bytes in c7x_dwconv1d.cpp."""
    t_blk = -(-t_out // _BLOCK) * _BLOCK
    span = (k - 1) * dilation
    return t_blk + span if stride == 1 else 2 * (t_blk + span // 2 + 1)


def _pattern():
    x, w, scale = wildcard(), wildcard(), wildcard()
    conv = is_op("relax.nn.conv2d")(x, w)
    as_float = is_op("relax.astype")(conv)
    scaled = is_op("relax.multiply")(as_float, scale)
    rounded = is_op("relax.round")(scaled)
    clipped = is_op("relax.clip")(rounded, varg_default_wildcard=True)
    out = is_op("relax.astype")(clipped)
    return out, {"x": x, "w": w, "scale": scale, "conv": conv, "clip": clipped, "out": out}


def _const_value(expr, var2val):
    """Evaluate a constant expression of reshape/multiply/divide in float32, or None."""
    if isinstance(expr, relax.Var):
        if expr not in var2val:
            return None
        return _const_value(var2val[expr], var2val)
    if isinstance(expr, tvm.ir.GenericConst):
        return expr.value.numpy()
    if not isinstance(expr, relax.Call) or not hasattr(expr.op, "name"):
        return None
    args = [_const_value(a, var2val) for a in expr.args[:2]]
    name = expr.op.name
    if name == "relax.reshape" and args[0] is not None:
        return args[0].reshape([int(d) for d in expr.ty.shape])
    if name in ("relax.multiply", "relax.divide") and all(a is not None for a in args):
        a, b = (np.asarray(v, dtype=np.float32) for v in args)
        return a * b if name == "relax.multiply" else a / b
    return None


def _shape(expr):
    ty = expr.ty
    if not isinstance(ty, relax.TensorType) or ty.shape is None:
        return None
    try:
        return [int(d) for d in ty.shape]
    except (TypeError, ValueError):
        return None


class _Matcher:
    """Pattern check, plus the per-call parameters the lowering needs."""

    def __init__(self, mod: IRModule):
        self.var2val = {}
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.var2val.update(relax.analysis.get_var2val(func))

    def check(self, ctx) -> bool:
        ann = ctx.annotated_expr
        conv, clip, out = ann["conv"], ann["clip"], ann["out"]
        attrs = conv.attrs
        if (
            str(attrs.data_layout) != "NCHW"
            or str(attrs.kernel_layout) != "OIHW"
            or str(attrs.out_dtype) != "int32"
            or str(out.attrs.dtype) != "int8"
        ):
            return False
        x_shape, w_shape, y_shape = _shape(ann["x"]), _shape(ann["w"]), _shape(conv)
        if x_shape is None or w_shape is None or y_shape is None:
            return False
        if str(ann["x"].ty.dtype) != "int8" or str(ann["w"].ty.dtype) != "int8":
            return False
        n, c, h, t_in = x_shape
        if n != 1 or h != 1 or int(attrs.groups) != c or w_shape != [c, 1, 1, w_shape[3]]:
            return False
        if y_shape[:3] != [1, c, 1]:
            return False
        k, t_out = w_shape[3], y_shape[3]
        strides = [int(s) for s in attrs.strides]
        dilation = [int(d) for d in attrs.dilation]
        padding = [int(p) for p in attrs.padding]  # top, left, bottom, right
        if strides[0] != 1 or dilation[0] != 1 or padding[0] != 0 or padding[2] != 0:
            return False
        if strides[1] not in (1, 2) or k > _MAX_TAPS:
            return False
        if _line_bytes(t_out, k, strides[1], dilation[1]) > _MAX_LINE:
            return False
        if [int(getattr(v, "value", -1)) for v in clip.args[1:3]] != [-128, 127]:
            return False
        if _const_value(ann["w"], self.var2val) is None:
            return False
        return self.channel_scale(ann["scale"], c) is not None

    def channel_scale(self, expr, c: int):
        """The requantize multiplier as a [C] float32 array, or None."""
        scale = _const_value(expr, self.var2val)
        if scale is None or scale.dtype != np.float32 or scale.size not in (1, c):
            return None
        if scale.size == c and scale.shape not in ([c], (c,), (1, c, 1, 1)):
            return None
        return np.ascontiguousarray(np.broadcast_to(scale.reshape(-1), [c]))


@mutator
class _Lowerer(PyExprMutator):
    """Lower c7x_dwconv1d composites to call_extern."""

    def __init__(self, mod: IRModule, matcher: _Matcher):
        super().__init__(mod)
        self.matcher = matcher
        self.count = 0

    def visit_call_(self, call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)
        func = self.builder_.get()[call.op]
        if not isinstance(func, relax.Function) or "Composite" not in (func.attrs or {}):
            return super().visit_call_(call)
        if str(func.attrs["Composite"]) != _COMPOSITE:
            return super().visit_call_(call)

        param_to_arg = dict(zip(func.params, call.args))

        def arg(expr):
            for p, a in param_to_arg.items():
                if p.same_as(expr):
                    return a
            return expr

        conv = mult = None
        for binding in func.body.blocks[0].bindings:
            val = binding.value
            if isinstance(val, relax.Call) and hasattr(val.op, "name"):
                if val.op.name == "relax.nn.conv2d":
                    conv = val
                elif val.op.name == "relax.multiply":
                    mult = val
        assert conv is not None and mult is not None, "pattern guarantees both"
        x, w, scale_expr = arg(conv.args[0]), arg(conv.args[1]), arg(mult.args[1])
        weight = _const_value(w, self.matcher.var2val)
        assert weight is not None, "check() required a constant weight"
        _, c, _, t_in = _shape(x)
        scale = self.matcher.channel_scale(scale_expr, c)
        k = int(weight.shape[3])
        t_out = _shape(call)[3]
        attrs = conv.attrs
        stride, dilation, pad_left = (
            int(attrs.strides[1]),
            int(attrs.dilation[1]),
            int(attrs.padding[1]),
        )
        w_const = ti_tensor_const(np.ascontiguousarray(weight.reshape(c, k).astype(np.int8)))
        s_const = ti_tensor_const(scale.astype(np.float32))

        dims = (c, t_in, t_out, k, stride, dilation, pad_left)

        def te_dwconv1d(x_t, w_t, s_t):
            def fcompute(ins, outs):
                return tirx.call_extern(
                    "int32",
                    "c7x_int8_dwconv1d",
                    ins[0].data,
                    ins[1].data,
                    ins[2].data,
                    outs[0].data,
                    *(tirx.IntImm("int32", v) for v in dims),
                )

            return te.extern(
                [1, c, 1, t_out], [x_t, w_t, s_t], fcompute, name="c7x_dwconv1d_out", dtype="int8"
            )

        result = self.builder_.call_te(
            te_dwconv1d, x, w_const, s_const, primfunc_name_hint="c7x_int8_dwconv1d"
        )
        self.count += 1
        logger.debug(
            "c7x_int8_dwconv1d: C=%d T_in=%d T_out=%d K=%d stride=%d dilation=%d pad=%d",
            c,
            t_in,
            t_out,
            k,
            stride,
            dilation,
            pad_left,
        )
        return propagate_span(result, find_composite_span(func))


@tvm.transform.module_pass(opt_level=0, name="FuseQDQToC7xDwConv1d")
class FuseQDQToC7xDwConv1d:
    """Lower FuseQDQToInt8Conv2D's int8 1xK depthwise convs to c7x_int8_dwconv1d.

    Must run after FuseQDQToInt8Conv2D and before LegalizeOps.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        matcher = _Matcher(mod)
        pattern, annotations = _pattern()
        mod = relax.transform.FuseOpsByPattern(
            [(_COMPOSITE, pattern, annotations, matcher.check)], bind_constants=False
        )(mod)

        lowerer = _Lowerer(mod, matcher)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function) and "Composite" not in (func.attrs or {}):
                lowerer.builder_.update_func(gv, lowerer.visit_expr(func))
        mod = lowerer.builder_.get()
        if lowerer.count:
            logger.info("FuseQDQToC7xDwConv1d: lowered %d depthwise convs", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)
        return mod
