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
"""Fold torchvision's per-channel input normalize (transform_input) into
the input float32->int8 quantize.

Why this pass is needed
------------------------
torchvision's Inception3/GoogLeNet share the same _transform_input, applied
to the raw model input whenever pretrained weights are loaded:

    x_ch0 = unsqueeze(x[:, 0], 1) * a0 + b0
    x_ch1 = unsqueeze(x[:, 1], 1) * a1 + b1
    x_ch2 = unsqueeze(x[:, 2], 1) * a2 + b2
    x = concat((x_ch0, x_ch1, x_ch2), 1)

torch.export/PT2E traces this as take->expand_dims->multiply->add (x3, one
per channel) -> concat(axis=1) -> quantize(scale, zp), all float32, directly
on the model's raw input Var.  FuseInputQuantize (ti_fuse_input_quantize.py)
does not match this: its quantize's operand is the model input Var itself,
while here the quantize's operand is the concat's output (a Call).  So the
whole chain falls through to LegalizeOps's generic scalar lowering -- one
slow ~157-cycles/element scalar TIR PrimFunc (42.2M cycles / 15.6% of
InceptionV3, see docs/dsp/quantized_model_optimization.md Step 16).

This is a compile-time fold, not a literal-replay kernel.  The affine
y_c = a_c*x_c + b_c feeds directly into q = round(y*inv_scale + zp) -- the
exact formula c7x_int8_quantize already implements.  Substituting:

    q_c = round(x_c*(a_c*inv_scale) + (b_c*inv_scale + zp))

the same multiply-add-round-clamp shape, just with per-channel constants
inv_scale_c = a_c*inv_scale and offset_c = b_c*inv_scale + zp.  So the
whole take/expand_dims/multiply/add/concat/quantize chain is replaced by a
single call_extern to c7x_int8_quantize_rgb operating directly on the raw
pixel input -- no intermediate float32 tensor or concat is ever
materialized, and the result is bit-exact (rounding still happens exactly
once, at the end, per element).

Pattern matched:
  take(x,0,axis=1)->expand_dims->multiply(a0)->add(b0) \
  take(x,1,axis=1)->expand_dims->multiply(a1)->add(b1)  -> concat(axis=1) -> quantize(scale,zp)
  take(x,2,axis=1)->expand_dims->multiply(a2)->add(b2) /
  -> call_extern c7x_int8_quantize_rgb(in, out, N, HW,
                                        inv_scale0, offset0,
                                        inv_scale1, offset1,
                                        inv_scale2, offset2)

Only per-tensor quantize is matched (scalar scale/zp constant), same guard
as FuseInputQuantize.  This is not model-specific: it matches the IR shape,
not a model name, so it also folds GoogLeNet's identical transform_input.

Kernel: src/runtime/ti_dsp/kernels/c7x_quantize.cpp (c7x_int8_quantize_rgb)
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, is_tuple, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_fuse_input_quantize import is_per_tensor_scalar_constant

logger = logging.getLogger(__name__)

_COMPOSITE_NAME = "c7x.input_normalize_quantize"
_NUM_CHANNELS = 3


# =========================================================================
# Pattern definition
# =========================================================================


def _make_input_normalize_pattern():
    """3x [take(x,c,axis=1) -> expand_dims -> multiply(a_c) -> add(b_c)]
    -> concat(axis=1) -> quantize(scale, zp)

    Only x (shared across all 3 branches -- reusing the same wildcard object
    forces the matcher to require the identical expression) and the
    trailing quantize's scale/zp are annotated for the check function.
    Each branch's take index and multiply/add constants are extracted in
    the lowerer by scanning the composite body directly, the same
    technique _MaxPoolLowerer._lower uses for max_pool2d's attrs and
    _lower_channel_scale_multiply uses for dequantize's scale/zp args --
    op attrs (take's axis) and args that aren't simple pattern leaves are
    not resolvable through FuseOpsByPattern's annotation mechanism.
    """
    x = wildcard()
    branches = []
    for _ in range(_NUM_CHANNELS):
        taken = is_op("relax.take")(x, wildcard())
        expanded = is_op("relax.expand_dims")(taken)
        mul = is_op("relax.multiply")(expanded, wildcard())
        added = is_op("relax.add")(mul, wildcard())
        branches.append(added)
    cat = is_op("relax.concat")(is_tuple(branches))
    o_s, o_z = wildcard(), wildcard()
    q = is_op("relax.quantize")(cat, o_s, o_z)
    return q, {"x": x, "o_scale": o_s, "o_zp": o_z}


def _check_input_normalize(ctx) -> bool:
    """Require x to be the raw float32 model input (relax.Var, NCHW, C=3)
    and the trailing quantize to be per-tensor with constant scale/zp.
    """
    x = ctx.annotated_expr["x"]
    if not isinstance(x, relax.Var):
        return False
    if not (hasattr(x, "struct_info") and hasattr(x.struct_info, "dtype")):
        return False
    if str(x.struct_info.dtype) != "float32":
        return False

    x_shape = getattr(x.struct_info, "shape", None)
    if x_shape is None:
        return False
    x_shape = [int(s) for s in x_shape]
    if len(x_shape) != 4 or x_shape[1] != _NUM_CHANNELS:
        return False

    o_scale = ctx.annotated_expr["o_scale"]
    o_zp = ctx.annotated_expr["o_zp"]
    if not is_per_tensor_scalar_constant(o_scale) or not isinstance(o_zp, relax.Constant):
        return False

    return True


# =========================================================================
# Composite lowering
# =========================================================================


@mutator
class _InputNormalizeLowerer(PyExprMutator):
    """Lower c7x.input_normalize_quantize composites to call_extern
    c7x_int8_quantize_rgb."""

    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.count = 0

    def visit_call_(self, call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)

        func = self.builder_.get()[call.op]
        if not isinstance(func, relax.Function):
            return super().visit_call_(call)
        if "Composite" not in func.attrs:
            return super().visit_call_(call)
        if str(func.attrs["Composite"]) != _COMPOSITE_NAME:
            return super().visit_call_(call)

        return self._lower(call, func)

    def _lower(self, call, func):
        param_to_arg = dict(zip(func.params, call.args))

        x_arg = None
        # var -> channel index, threaded through take->expand_dims->multiply->add
        var_channel = {}
        scale_by_channel = {}
        offset_by_channel = {}
        o_scale_val = o_zp_val = None

        for binding in func.body.blocks[0].bindings:
            val = binding.value
            var = binding.var
            if not isinstance(val, relax.Call) or not hasattr(val.op, "name"):
                continue
            op_name = str(val.op.name)

            if op_name == "relax.take":
                if val.attrs is None or int(val.attrs.axis) != 1:
                    continue
                idx_arg = param_to_arg.get(val.args[1], val.args[1])
                if not isinstance(idx_arg, relax.Constant):
                    continue
                channel = int(idx_arg.data.numpy())
                var_channel[var] = channel
                x_param = val.args[0]
                x_arg = param_to_arg.get(x_param, x_param)
            elif op_name == "relax.expand_dims":
                src = val.args[0]
                if src in var_channel:
                    var_channel[var] = var_channel[src]
            elif op_name == "relax.multiply":
                src = val.args[0]
                other = param_to_arg.get(val.args[1], val.args[1])
                if src in var_channel and isinstance(other, relax.Constant):
                    channel = var_channel[src]
                    var_channel[var] = channel
                    scale_by_channel[channel] = float(other.data.numpy())
            elif op_name == "relax.add":
                # add's result feeds concat directly (not scanned further),
                # so no need to propagate var_channel past this point.
                src = val.args[0]
                other = param_to_arg.get(val.args[1], val.args[1])
                if src in var_channel and isinstance(other, relax.Constant):
                    offset_by_channel[var_channel[src]] = float(other.data.numpy())
            elif op_name == "relax.quantize":
                s = param_to_arg.get(val.args[1], val.args[1])
                z = param_to_arg.get(val.args[2], val.args[2])
                if isinstance(s, relax.Constant) and isinstance(z, relax.Constant):
                    o_scale_val = float(s.data.numpy())
                    o_zp_val = int(z.data.numpy())

        expected_channels = set(range(_NUM_CHANNELS))
        if (
            x_arg is None
            or o_scale_val is None
            or o_scale_val == 0.0
            or set(scale_by_channel) != expected_channels
            or set(offset_by_channel) != expected_channels
        ):
            return super().visit_call_(call)

        call_sinfo = call.struct_info
        if not isinstance(call_sinfo, relax.TensorStructInfo) or not call_sinfo.shape:
            return super().visit_call_(call)
        out_shape = [int(s) for s in call_sinfo.shape]
        if len(out_shape) != 4 or out_shape[1] != _NUM_CHANNELS:
            return super().visit_call_(call)

        inv_scale = 1.0 / o_scale_val
        params = []  # [(inv_scale_c, offset_c), ...] for c in 0..2
        for c in range(_NUM_CHANNELS):
            a_c = scale_by_channel[c]
            b_c = offset_by_channel[c]
            params.append((a_c * inv_scale, b_c * inv_scale + o_zp_val))

        _N, _, _H, _W = out_shape
        _HW = _H * _W
        (_is0, _off0), (_is1, _off1), (_is2, _off2) = params

        def te_input_normalize_quantize(x_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "c7x_int8_quantize_rgb",
                    ins[0].data,
                    outs[0].data,
                    tir.IntImm("int32", _N),
                    tir.IntImm("int32", _HW),
                    tir.FloatImm("float32", _is0),
                    tir.FloatImm("float32", _off0),
                    tir.FloatImm("float32", _is1),
                    tir.FloatImm("float32", _off1),
                    tir.FloatImm("float32", _is2),
                    tir.FloatImm("float32", _off2),
                )

            return te.extern(
                out_shape,
                [x_t],
                fcompute,
                name="input_normalize_quantize_out",
                dtype="int8",
            )

        result = self.builder_.call_te(
            te_input_normalize_quantize,
            x_arg,
            primfunc_name_hint="c7x_int8_quantize_rgb",
        )
        self.count += 1
        logger.debug(
            "Fused c7x_int8_quantize_rgb: N=%d HW=%d params=%s",
            _N,
            _HW,
            params,
        )
        return result


# =========================================================================
# Public pass
# =========================================================================


@tvm.transform.module_pass(opt_level=0, name="FuseInputNormalizeQuantize")
class FuseInputNormalizeQuantize:
    """Fold transform_input's per-channel normalize into the input quantize.

    Must run before LegalizeOps (which lowers R.quantize to a scalar TIR
    loop) and before EliminateQDQTransparent, same constraint as
    FuseInputQuantize.  Registered before FuseInputQuantize since this
    pattern is the more specific one; the two are mutually exclusive
    (quantize(concat(...)) vs. quantize(Var) directly) so ordering between
    them doesn't affect correctness.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        pat, annotations = _make_input_normalize_pattern()
        patterns = [(_COMPOSITE_NAME, pat, annotations, _check_input_normalize)]

        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        lowerer = _InputNormalizeLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("FuseInputNormalizeQuantize: fused %d input normalize ops", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
