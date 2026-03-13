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
"""Fuse dequantize + permute_dims + matmul into a single TIR kernel.

This pass targets INT8 weight-only quantization where the graph after
RewriteDequantize + BindParams contains:

    R.dequantize(w_int8_const, scale_const, zp=0, axis=0)
      -> R.permute_dims(w_float)
      -> R.matmul(activation, w_T)
      [+ R.add(bias)]

Without this pass, FoldConstant evaluates dequantize(const, const, const)
at compile time, expanding int8 weights back to float32 in weights.bin.

The fused kernel reads int8 weights directly and computes:

    output[..., n] = sum_k(act[..., k] * float(w_int8[n, k])) * scale[n]

Scale is factored out of the reduction (one float mul per output element
rather than per MAC).  Since the fused op has a non-constant activation
input, FoldConstant naturally skips it.

This pass should run BEFORE LegalizeOps.
"""

import logging

import numpy as np

import tvm
from tvm import IRModule, relax, te
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _check_pattern(ctx):
    """Validate guards common to both bias and no-bias patterns."""
    # w_int8 must be int8
    w = ctx.annotated_expr["w_int8"]
    if isinstance(w, relax.Constant):
        if w.data.dtype != "int8":
            return False
    elif hasattr(w, "struct_info") and hasattr(w.struct_info, "dtype"):
        if str(w.struct_info.dtype) != "int8":
            return False
    else:
        return False

    # w_zp must be constant zero
    z = ctx.annotated_expr["w_zp"]
    if isinstance(z, relax.Constant):
        if not np.all(z.data.numpy() == 0):
            return False
    else:
        return False

    # activation must NOT be constant
    a = ctx.annotated_expr["act"]
    if isinstance(a, relax.Constant):
        return False

    # permute_dims must be a standard transpose of the last two dims
    perm = ctx.annotated_expr["w_perm"]
    if isinstance(perm, relax.Call):
        src = perm.args[0]
        ndim = src.struct_info.ndim
        if ndim < 2:
            return False
        if perm.attrs.axes is not None:
            expected = list(range(ndim))
            expected[-1], expected[-2] = expected[-2], expected[-1]
            if list(perm.attrs.axes) != expected:
                return False
        elif ndim != 2:
            # axes=None reverses all dims; only valid for 2-D
            return False

    return True


# ---------------------------------------------------------------------------
# DPL patterns
# ---------------------------------------------------------------------------


def _pattern_with_bias():
    """Match: add(matmul(act, permute_dims(dequantize(w, s, z))), bias)."""
    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)
    w_perm = is_op("relax.permute_dims")(w_dq)
    act = wildcard()
    mm = is_op("relax.matmul")(act, w_perm)
    bias = wildcard()
    add_out = is_op("relax.add")(mm, bias)

    annotations = {
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "w_perm": w_perm,
        "act": act,
        "mm": mm,
        "bias": bias,
        "add_out": add_out,
    }
    return add_out, annotations, _check_pattern


def _pattern_no_bias():
    """Match: matmul(act, permute_dims(dequantize(w, s, z)))."""
    w_int8 = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    w_dq = is_op("relax.dequantize")(w_int8, w_scale, w_zp)
    w_perm = is_op("relax.permute_dims")(w_dq)
    act = wildcard()
    mm = is_op("relax.matmul")(act, w_perm)

    annotations = {
        "w_int8": w_int8,
        "w_scale": w_scale,
        "w_zp": w_zp,
        "w_perm": w_perm,
        "act": act,
        "mm": mm,
    }
    return mm, annotations, _check_pattern


# ---------------------------------------------------------------------------
# TE compute
# ---------------------------------------------------------------------------


def _te_dequantize_matmul(activation, w_int8, scale):
    """Fused dequantize + transpose + matmul as a TE compute.

    Parameters
    ----------
    activation : te.Tensor, shape [*, K], float32
    w_int8     : te.Tensor, shape [N, K], int8
    scale      : te.Tensor, shape [N] or scalar, float32

    Returns
    -------
    te.Tensor, shape [*, N], float32

    Computes
    --------
    output[..., n] = sum_k(act[..., k] * float(w_int8[n, k])) * scale[n]

    Scale is factored out of the reduction (one float mul per output
    element rather than per MAC).
    """
    K = w_int8.shape[1]
    N = w_int8.shape[0]

    out_shape = [activation.shape[i] for i in range(len(activation.shape) - 1)] + [N]
    k = te.reduce_axis((0, K), name="k")

    # Stage 1: int8->float32 matmul accumulation
    def acc_compute(*indices):
        batch = indices[:-1]
        n = indices[-1]
        return te.sum(
            activation[(*batch, k)].astype("float32")
            * w_int8[n, k].astype("float32"),
            axis=k,
        )

    acc = te.compute(out_shape, acc_compute, name="dequantize_matmul_acc")

    # Stage 2: multiply by per-channel scale (factored out of reduction)
    def scale_compute(*indices):
        n = indices[-1]
        if len(scale.shape) == 0:
            return acc(*indices) * scale[()]
        return acc(*indices) * scale[n]

    return te.compute(out_shape, scale_compute, name="dequantize_matmul")


# ---------------------------------------------------------------------------
# Mutator: replace composite functions with fused call_te
# ---------------------------------------------------------------------------


@mutator
class _DequantizeMatmulFuser(PyExprMutator):
    """Replace composite dequantize_matmul functions with fused TIR."""

    def __init__(self, mod):
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

        name = str(func.attrs["Composite"])
        if name == "dequantize_matmul_bias_fuse":
            return self._rewrite(call, func, has_bias=True)
        if name == "dequantize_matmul_fuse":
            return self._rewrite(call, func, has_bias=False)

        return super().visit_call_(call)

    def _rewrite(self, call, func, has_bias):
        param_to_arg = dict(zip(func.params, call.args))
        roles = self._extract_roles(func, has_bias)

        if "act" not in roles or "w_int8" not in roles or "w_scale" not in roles:
            logger.warning("Could not identify roles in composite function")
            return super().visit_call_(call)

        act = param_to_arg[roles["act"]]
        w_int8 = param_to_arg[roles["w_int8"]]
        w_scale = param_to_arg[roles["w_scale"]]

        self.count += 1
        result = self.builder_.call_te(
            _te_dequantize_matmul,
            act,
            w_int8,
            w_scale,
            primfunc_name_hint="dequantize_matmul",
        )

        if has_bias and "bias" in roles:
            bias = param_to_arg[roles["bias"]]
            result = relax.op.add(result, bias)

        logger.info(
            "Fused dequantize_matmul #%d%s",
            self.count,
            " (bias)" if has_bias else "",
        )
        return result

    @staticmethod
    def _extract_roles(func, has_bias):
        """Walk composite function body to map params to roles."""
        roles = {}
        for block in func.body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    continue
                val = binding.value
                if not isinstance(val, relax.Call):
                    continue
                if not hasattr(val.op, "name"):
                    continue

                op_name = val.op.name
                if op_name == "relax.dequantize":
                    roles["w_int8"] = val.args[0]
                    roles["w_scale"] = val.args[1]
                elif op_name == "relax.matmul":
                    roles["act"] = val.args[0]
                elif has_bias and op_name == "relax.add":
                    roles["bias"] = val.args[1]
        return roles


# ---------------------------------------------------------------------------
# Public pass
# ---------------------------------------------------------------------------


@tvm.transform.module_pass(opt_level=0, name="FuseDequantizeMatmul")
class FuseDequantizeMatmul:  # pylint: disable=too-few-public-methods
    """Fuse dequantize + permute_dims + matmul into a single TIR kernel.

    Replaces the pattern::

        R.dequantize(w_int8, scale, zp=0) -> R.permute_dims -> R.matmul(act, _)
        [+ R.add(_, bias)]

    with a fused TIR kernel that reads int8 weights directly, preventing
    FoldConstant from expanding them to float32.

    Supports:
      - Per-channel and per-tensor scale
      - Optional float bias (kept as a separate R.add)
      - Arbitrary batch dimensions on the activation

    This pass should run BEFORE LegalizeOps.
    """

    def transform_module(
        self, mod: IRModule, _ctx: tvm.transform.PassContext
    ) -> IRModule:
        # Phase 1: pattern-match and wrap into composite functions
        mod = relax.transform.FuseOpsByPattern(
            [
                ("dequantize_matmul_bias_fuse", *_pattern_with_bias()),
                ("dequantize_matmul_fuse", *_pattern_no_bias()),
            ],
            bind_constants=False,
        )(mod)

        # Phase 2: replace composite functions with fused call_te
        fuser = _DequantizeMatmulFuser(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = fuser.visit_expr(func)
                fuser.builder_.update_func(gv, func)
        mod = fuser.builder_.get()

        if fuser.count > 0:
            logger.info(
                "FuseDequantizeMatmul: fused %d patterns", fuser.count
            )

        return mod
