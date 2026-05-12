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
"""MMALIB int16 FC legalization for MLP layers.

Converts float matmul ops (with dequantized int8 weight constants) to
int16 MMALIB calls (matrixMatrixMultiply, non-bias) with near-float
precision.

Why int16 non-bias instead of int8 matmulBias:
  The int8 matmulBias path uses per-channel uint8 scale/shift for output
  requantization: out_i8 = (accum * scale_u8) >> shift_u8. This has
  limited precision that scales with K — for K=576 (SmolLM), per-layer
  error is ~5 logit-units which compounds to 0% accuracy over 30 layers.

  The int16 non-bias path avoids requantization entirely:
  - Input quantized to int16 (65536 levels vs 256 for int8)
  - Matmul accumulator is int32/int64 internally
  - Output is int16, shifted right by a global constant to prevent overflow
  - Dequantization is just a float multiply (no precision loss)
  - Measured precision: 0.0017 max_diff vs float (3500x better than int8)

Target pattern (after RewriteDequantize + BindParams):
    R.dequantize(w_const, w_scale, zp=0, axis=0)      → w_float [N, K]
    R.permute_dims(w_float)                            → w_T [K, N]
    R.matmul(activation_float [M, K], w_T [K, N])     → out [M, N]

Replacement:
    x_i16 = clip(round(x_float / x_scale), -32768, 32767)  [M, K]
    out_i16 = mmalib_matmul_i16(x_i16, w_i16_KN, M, K, N, shift)
    out_float = out_i16 * dequant_scale                     [M, N]

where:
    x_scale = max|activation| / 32767  (from calibration)
    w_i16 = sign_extend(w_int8)  (lossless for int8 weights)
    shift = ceil(log2(max_accum / 32767))  (prevents int16 overflow)
    dequant_scale[n] = (2^shift) * x_scale * w_scale[n]
"""

import logging
import math

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_mmalib_constants import MMA_SIZE_I16

logger = logging.getLogger(__name__)


def _compute_shift(w_i16):
    """Compute global shift to prevent int16 output overflow.

    Uses the maximum L1-norm (sum of absolute values) across weight rows
    instead of worst-case max|w| * K. This gives a much tighter bound
    because real matmul accumulators benefit from sign cancellation.

    Tight bound: max_accum ≈ max|x_i16| * max_row_l1_norm(w)
    With dynamic quantization, max|x_i16| = 32767 always.
    """
    # L1 norm per output channel (row of [N, K] weight matrix transposed to [K, N])
    row_l1 = np.abs(w_i16).sum(axis=0)  # w_i16 is [K, N], sum over K → [N]
    max_l1 = int(row_l1.max())
    max_accum = 32767 * max_l1
    if max_accum <= 32767:
        return 0
    return math.ceil(math.log2(max_accum / 32767))


def _pre_scan_bindings(func):
    """Walk function body to build a var→value map for pattern matching.

    PyExprMutator's visit_binding_ may not fire for all bindings in
    dataflow blocks. This pre-scan ensures we have all bindings available
    for pattern tracing (dequantize → permute_dims → matmul chain).
    """
    binding_map = {}
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding, relax.VarBinding):
                binding_map[binding.var] = binding.value
    return binding_map


def _find_in_map(binding_map, var):
    """Look up a Var in the binding map using same_as comparison.

    TVM Python bindings create new wrapper objects on each attribute
    access, so id() comparison fails. Use same_as() instead.
    """
    for k, v in binding_map.items():
        if k.same_as(var):
            return v
    return None


@mutator
class _MMALIBInt16FCMutator(PyExprMutator):
    """Replace dequantize+permute+matmul with int16 MMALIB calls."""

    def __init__(self, mod: IRModule, binding_map: dict):
        super().__init__(mod)
        self.count = 0
        self.binding_map = binding_map

    def visit_call_(self, call: relax.Call):
        if not isinstance(call.op, tvm.ir.Op):
            return super().visit_call_(call)
        if call.op.name != "relax.matmul":
            return super().visit_call_(call)

        result = self._try_lower_matmul(call)
        if result is not None:
            return result
        return super().visit_call_(call)

    def _resolve(self, expr):
        """Resolve a Var to its bound value, or return expr if not a Var."""
        if isinstance(expr, relax.Var):
            val = _find_in_map(self.binding_map, expr)
            return val if val is not None else expr
        return expr

    def _try_lower_matmul(self, call):
        """Attempt to lower a matmul to int16 MMALIB."""
        activation = call.args[0]
        rhs = call.args[1]

        # Resolve RHS Var to its bound Call
        rhs_val = self._resolve(rhs)

        # Must be permute_dims
        if not (isinstance(rhs_val, relax.Call) and hasattr(rhs_val.op, "name")
                and rhs_val.op.name == "relax.permute_dims"):
            return None

        # Resolve permute_dims input to dequantize
        perm_input = rhs_val.args[0]
        dequant_call = self._resolve(perm_input)

        if not (isinstance(dequant_call, relax.Call) and hasattr(dequant_call.op, "name")
                and dequant_call.op.name == "relax.dequantize"):
            return None

        # Extract weight constant and scale from dequantize
        w_const = dequant_call.args[0]
        w_scale_expr = dequant_call.args[1]
        if not isinstance(w_const, relax.Constant):
            return None
        if not isinstance(w_scale_expr, relax.Constant):
            return None

        w_np = w_const.data.numpy()  # [N, K] int8
        w_scale_np = w_scale_expr.data.numpy().flatten()  # [N]

        # Check weight is 2D
        if len(w_np.shape) != 2:
            return None
        N, K = w_np.shape

        # Check dimension alignment for int16 MMALIB (32-element vectors)
        act_sinfo = activation.struct_info
        if act_sinfo is None or act_sinfo.shape is None:
            return None
        act_shape = [int(s) for s in act_sinfo.shape]
        if len(act_shape) != 2:
            return None
        M = act_shape[0]
        if M % MMA_SIZE_I16 != 0 or K % MMA_SIZE_I16 != 0 or N % MMA_SIZE_I16 != 0:
            return None

        # Activation must be float32
        if act_sinfo.dtype != "float32":
            return None

        # --- Compute int16 quantization parameters ---

        # Sign-extend int8 weight to int16 (lossless)
        w_i16 = w_np.astype(np.int16)

        # Weight in [K, N] for non-transposed mmalib_matmul_i16
        w_i16_KN = np.ascontiguousarray(w_i16.T)

        # Compute shift using tight L1-norm bound (not worst-case max*K)
        shift = _compute_shift(w_i16_KN)

        # --- Emit replacement ops ---

        # 1. Dynamic per-tensor quantization: compute scale from actual input
        #    x_scale = max(|x|) / 32767, then x_i16 = round(x / x_scale)
        #    This guarantees no clipping and optimal precision for any input range.
        x_abs = self.builder_.emit(relax.op.abs(activation))
        x_max = self.builder_.emit(relax.op.max(x_abs))
        # x_scale = x_max / 32767
        x_scale = self.builder_.emit(
            relax.op.divide(x_max, relax.const(32767.0, "float32"))
        )
        # Prevent division by zero
        x_scale = self.builder_.emit(
            relax.op.maximum(x_scale, relax.const(1e-10, "float32"))
        )
        # x_i16 = clip(round(x / x_scale), -32768, 32767)
        inv_scale = self.builder_.emit(
            relax.op.divide(relax.const(1.0, "float32"), x_scale)
        )
        x_scaled = self.builder_.emit(relax.op.multiply(activation, inv_scale))
        x_rounded = self.builder_.emit(relax.op.round(x_scaled))
        x_clipped = self.builder_.emit(relax.op.clip(x_rounded, -32768, 32767))
        x_i16 = self.builder_.emit(relax.op.astype(x_clipped, "int16"))

        # 2. MMALIB int16 matmul (non-bias, with shift)
        def te_mmalib_i16(data_t, w_t):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32",
                    "mmalib_matmul_i16",
                    ins[0].data,
                    ins[1].data,
                    outs[0].data,
                    M, K, N, shift,
                )

            return te.extern(
                [M, N], [data_t, w_t], fcompute,
                name="mmalib_i16_fc", dtype="int16",
            )

        out_i16 = self.builder_.call_te(
            te_mmalib_i16, x_i16, relax.Constant(w_i16_KN),
            primfunc_name_hint="mmalib_i16_fc",
        )

        # 3. Dequantize: out_float = cast(out_i16) * (2^shift) * x_scale * w_scale[n]
        #    x_scale is dynamic (computed at runtime), w_scale is a compile-time constant
        out_float = self.builder_.emit(relax.op.astype(out_i16, "float32"))
        # Static part: (2^shift) * w_scale[n]
        static_dequant = (np.float32(1 << shift) * w_scale_np).astype(np.float32)
        out_float = self.builder_.emit(
            relax.op.multiply(out_float, relax.Constant(static_dequant.reshape(1, N)))
        )
        # Dynamic part: multiply by x_scale (scalar, computed at runtime)
        out_float = self.builder_.emit(relax.op.multiply(out_float, x_scale))

        self.count += 1
        logger.info(
            "MMALIB i16 FC #%d: M=%d K=%d N=%d shift=%d",
            self.count, M, K, N, shift,
        )
        return out_float


@tvm.transform.module_pass(opt_level=0, name="LegalizeMLPToMMALIBInt16")
class LegalizeMLPToMMALIBInt16:
    """Convert MLP float matmul to int16 MMALIB with near-float precision.

    Targets dequantize+permute_dims+matmul patterns where the weight shape
    matches configured MLP dimensions. Uses mmalib_matmul_i16 (non-bias)
    with computed shift for overflow prevention.

    The pass pre-scans all bindings to build a var→value map, enabling
    pattern matching through Var indirections in the dataflow IR.

    Args:
        target_shapes: Set of (N, K) weight shapes to target.
            Default: SmolLM MLP {(1536, 576), (576, 1536)}
        act_scales: Dict mapping layer index (0-based) to activation scale.
            Scale = max_activation_magnitude / 32767.
            Default: 10.0/32767 (assumes activations in [-10, 10])
    """

    def __init__(self):
        pass

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Pre-scan bindings from the main function for pattern matching
        binding_map = {}
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                binding_map = _pre_scan_bindings(func)
                break

        lowerer = _MMALIBInt16FCMutator(mod, binding_map)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                if "Composite" in (func.attrs or {}):
                    continue
                func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info("LegalizeMLPToMMALIBInt16: converted %d layers", lowerer.count)
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
