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
"""MMALIB int16 legalization and shared helpers for TI C7x MMA.

Provides:
  - MMALIBLegalize: custom legalization for int16 matmul/conv2d via
    LegalizeOps(customize_legalize_map=...).
  - _check_conv2d_mmalib_constraints: shared eligibility check (used by
    both this module and ti_mmalib_qdq_fusion.py).
  - _validate_and_fold_bias: validates the weight-scale/rescale range and
    folds the float bias into accumulator scale. The single shared entry
    point behind all 6 int8/int16 conv2d/dwconv2d/FC QDQ lowerers
    (ti_mmalib_qdq_fusion.py, ti_mmalib_qdq_dwconv.py, ti_mmalib_qdq_fc.py,
    ti_mmalib_qdq_i16_conv.py, ti_mmalib_qdq_i16_dwconv.py).

Data layout: NCHW throughout.
MMALIB's conv kernel (convolveBias_row) operates on planar channel-first
data — each input channel is a contiguous H*W block. This matches NCHW.
When -mmalib=1 is set, the pipeline skips ConvertLayoutNHWC so that all
ops (conv, relu, add, pool) stay in NCHW. Layout conversion happens at
network I/O boundaries only.
"""

import logging
from typing import Union

import numpy as np

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext

from .legalize_ops.linear_algebra import _matmul
from .legalize_ops.nn import _nn_conv2d
from .ti_c7x_span_utils import propagate_span

logger = logging.getLogger(__name__)

# =======================================================================
# Int16 matmul legalization
# =======================================================================


def _is_mmalib_eligible(call: relax.Call) -> bool:
    """Check if a matmul call can be handled by MMALIB."""
    lhs_sinfo = call.args[0].struct_info
    rhs_sinfo = call.args[1].struct_info

    if not isinstance(lhs_sinfo, relax.TensorStructInfo):
        return False
    if not isinstance(rhs_sinfo, relax.TensorStructInfo):
        return False

    if lhs_sinfo.dtype != "int16" or rhs_sinfo.dtype != "int16":
        return False

    lhs_shape = lhs_sinfo.shape
    rhs_shape = rhs_sinfo.shape
    if lhs_shape is None or rhs_shape is None:
        return False

    if lhs_sinfo.ndim != 2 or rhs_sinfo.ndim != 2:
        return False

    from .ti_mmalib_constants import MMA_SIZE_I16

    mma_size_i16 = MMA_SIZE_I16
    for s in lhs_shape:
        if not isinstance(s, tir.IntImm):
            return False
        if int(s) % mma_size_i16 != 0:
            return False
    for s in rhs_shape:
        if not isinstance(s, tir.IntImm):
            return False
        if int(s) % mma_size_i16 != 0:
            return False

    return True


def _mmalib_matmul_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    """Legalize R.matmul to MMALIB extern call when eligible."""
    if not _is_mmalib_eligible(call):
        return _matmul(bb, call)

    lhs_sinfo = call.args[0].struct_info
    rhs_sinfo = call.args[1].struct_info
    M = int(lhs_sinfo.shape[0])
    K = int(lhs_sinfo.shape[1])
    N = int(rhs_sinfo.shape[1])

    def te_mmalib_matmul(a: te.Tensor, b: te.Tensor) -> te.Tensor:
        def fcompute(ins, outs):
            return _call_extern_checked(
                "int32",
                "mmalib_matmul_i16",
                ins[0].data,
                ins[1].data,
                outs[0].data,
                M,
                K,
                N,
                0,
            )

        return te.extern(
            (M, N),
            [a, b],
            fcompute,
            name="mmalib_matmul",
            dtype="int16",
        )

    return propagate_span(
        bb.call_te(
            te_mmalib_matmul,
            call.args[0],
            call.args[1],
            primfunc_name_hint="mmalib_matmul",
            primfunc_attrs={"c7x_offload_backend": "mmalib"},
        ),
        call,
    )


# =======================================================================
# Int16 conv2d legalization
# =======================================================================


def _check_conv2d_mmalib_constraints(
    attrs, data_sinfo, kernel_sinfo, allow_groups: bool = False
) -> bool:
    """Shared MMALIB conv2d eligibility check for both int8 and int16.

    MMALIB constraints:
      - dilation must be 1x1
      - groups must be 1, unless allow_groups=True (see below)
      - strides must be symmetric (strideX == strideY)
      - N must be 1
      - all shapes must be static
      - row-kernel geometry: when stride>1, padding must be either zero or
        exact "SAME"-derived padding; when stride==1, an unpadded
        ("VALID") kernel wider than 1x1 is rejected unless the output size
        already absorbs the padding-agnostic column count MMALIB computes
        internally (see _check_conv2d_row_kernel_geometry below) — both
        real MMALIB_CNN_convolveBias_row_ixX_ixX_oxX hardware constraints,
        not a TVM-side choice

    allow_groups: when True, permits groups>1 for genuinely grouped
    (partial-channel) convolution — e.g. ResNeXt101's cardinality=32
    bottleneck convs — routed to a per-group call_extern loop by the
    caller (see ti_mmalib_qdq_fusion.py). True
    depthwise (groups == C_in) is excluded here: that's
    ti_mmalib_qdq_dwconv.py's job, not this path's. Defaults to False so
    every other existing caller (int16 QDQ conv, int16 plain-legalize)
    keeps rejecting groups>1 unchanged.
    """
    if list(attrs.dilation) != [1, 1]:
        return False

    strides = [int(s) for s in attrs.strides]
    if strides[0] != strides[1]:
        return False

    if data_sinfo.shape is None or kernel_sinfo.shape is None:
        return False
    for s in data_sinfo.shape:
        if not isinstance(s, tir.IntImm):
            return False
    for s in kernel_sinfo.shape:
        if not isinstance(s, tir.IntImm):
            return False

    data_layout = tir.layout(attrs.data_layout)
    kernel_layout = tir.layout(attrs.kernel_layout)

    if attrs.groups != 1:
        if not allow_groups or attrs.groups <= 0:
            return False
        c_in = int(data_sinfo.shape[data_layout.index_of("C")])
        c_out = int(kernel_sinfo.shape[kernel_layout.index_of("O")])
        if attrs.groups == c_in:
            return False  # true depthwise — ti_mmalib_qdq_dwconv.py's job
        if c_in % attrs.groups != 0 or c_out % attrs.groups != 0:
            return False

    if int(data_sinfo.shape[data_layout.index_of("N")]) != 1:
        return False

    KH = int(kernel_sinfo.shape[kernel_layout.index_of("H")])
    KW = int(kernel_sinfo.shape[kernel_layout.index_of("W")])
    H_in = int(data_sinfo.shape[data_layout.index_of("H")])
    W_in = int(data_sinfo.shape[data_layout.index_of("W")])
    if not _check_conv2d_row_kernel_geometry(attrs, KH, KW, H_in, W_in, strides[0]):
        return False

    return True


def _check_conv2d_row_kernel_geometry(
    attrs, KH: int, KW: int, H_in: int, W_in: int, stride: int
) -> bool:
    """MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_init_checkParams's two
    stride-dependent geometry rules (dilation is always 1x1 here per the
    caller's own check).

    stride>1: requires either padTop==0 or padBottom is either 0 or the
    exact "SAME"-derived value (KH-1)//2. A 1x1 kernel with stride>1
    additionally requires padLeft==padRight. AlexNet's conv1 (11x11,
    stride 4, pad=2/2 symmetric) satisfies neither the general nor
    1x1-specific rule -- exactly the geometry that produced
    MMALIB_ERR_INVALID_DIMENSION (status 4) on real c7x_dload hardware.

    stride==1: the vendor's own validColsOut is padding-agnostic
    (``H_in*W_in - W_in*(KH-1) - (KW-1)``) while dst_addr->stride_y
    (H_out*W_out) is computed *with* padding, so an unpadded ("VALID")
    conv wider than 1x1 trips ``validColsOut > dst_addr->stride_y``
    whenever H_in>KH (algebraically, the difference is
    ``(KW-1)*(H_in-KH)``, positive for any real shrinking VALID conv) --
    "SAME" padding or a 1x1 kernel always keep the difference <=0.
    Confirmed against the vendor source and reproduced in isolation on
    real c7x_dload hardware for Inception_v3's second stem conv
    (32,149,149,32, 3x3, stride=1, pad=0), the exact geometry that
    aborted the DSP before this check existed.
    """
    padding = [int(p) for p in attrs.padding]
    if len(padding) == 2:
        pad_top, pad_left = padding[0], padding[1]
        pad_bottom, pad_right = padding[0], padding[1]
    else:
        pad_top, pad_left, pad_bottom, pad_right = padding

    if stride > 1:
        if not (pad_top == 0 or pad_bottom in (0, (KH - 1) // 2)):
            logger.info(
                "MMALIB conv2d decline: stride=%d requires padTop==0 or padBottom "
                "in (0, (KH-1)//2=%d); got padTop=%d, padBottom=%d (KH=%d) -- "
                "falling back to scalar path",
                stride,
                (KH - 1) // 2,
                pad_top,
                pad_bottom,
                KH,
            )
            return False

        if KH == 1 and KW == 1 and pad_left != pad_right:
            logger.info(
                "MMALIB conv2d decline: 1x1 kernel with stride=%d requires "
                "padLeft==padRight; got padLeft=%d, padRight=%d -- falling back "
                "to scalar path",
                stride,
                pad_left,
                pad_right,
            )
            return False
    else:
        valid_cols_out = H_in * W_in - W_in * (KH - 1) - (KW - 1)
        H_out = H_in + pad_top + pad_bottom - KH + 1
        W_out = W_in + pad_left + pad_right - KW + 1
        if valid_cols_out > H_out * W_out:
            logger.info(
                "MMALIB conv2d decline: stride=1 unpadded/under-padded %dx%d "
                "kernel on H_in=%d,W_in=%d gives validColsOut=%d > "
                "H_out*W_out=%d (H_out=%d,W_out=%d) -- falling back to "
                "scalar path",
                KH,
                KW,
                H_in,
                W_in,
                valid_cols_out,
                H_out * W_out,
                H_out,
                W_out,
            )
            return False

    return True


def _is_conv2d_mmalib_eligible(call: relax.Call) -> bool:
    """Check if a conv2d call can be handled by MMALIB (int16 path)."""
    data_sinfo = call.args[0].struct_info
    kernel_sinfo = call.args[1].struct_info

    if not isinstance(data_sinfo, relax.TensorStructInfo):
        return False
    if not isinstance(kernel_sinfo, relax.TensorStructInfo):
        return False
    if data_sinfo.dtype != "int16" or kernel_sinfo.dtype != "int16":
        return False
    if data_sinfo.ndim != 4 or kernel_sinfo.ndim != 4:
        return False

    return _check_conv2d_mmalib_constraints(call.attrs, data_sinfo, kernel_sinfo)


def _mmalib_conv2d_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    """Legalize R.nn.conv2d to MMALIB extern call when eligible (int16)."""
    if not _is_conv2d_mmalib_eligible(call):
        return _nn_conv2d(bb, call)

    attrs = call.attrs
    data_sinfo = call.args[0].struct_info
    kernel_sinfo = call.args[1].struct_info

    data_layout = tir.layout(attrs.data_layout)
    kernel_layout = tir.layout(attrs.kernel_layout)

    C_in = int(data_sinfo.shape[data_layout.index_of("C")])
    H_in = int(data_sinfo.shape[data_layout.index_of("H")])
    W_in = int(data_sinfo.shape[data_layout.index_of("W")])
    C_out = int(kernel_sinfo.shape[kernel_layout.index_of("O")])
    KH = int(kernel_sinfo.shape[kernel_layout.index_of("H")])
    KW = int(kernel_sinfo.shape[kernel_layout.index_of("W")])

    strides = [int(s) for s in attrs.strides]
    stride_h, stride_w = strides[0], strides[1]

    padding = [int(p) for p in attrs.padding]
    if len(padding) == 2:
        pad_top, pad_left = padding[0], padding[1]
        pad_bottom, pad_right = padding[0], padding[1]
    else:
        pad_top, pad_left, pad_bottom, pad_right = padding

    H_out = (H_in + pad_top + pad_bottom - KH) // stride_h + 1
    W_out = (W_in + pad_left + pad_right - KW) // stride_w + 1

    # Identity params: zero bias, scale=1, shift=0 for all channels.
    # The QDQ fusion pass (FuseMMALIBQDQConv2dI16) will pass real per-channel
    # values; this legalize path handles plain float32 conv2d with no requant.
    bias_const = relax.Constant(np.zeros(C_out, dtype=np.int64))
    scale_const = relax.Constant(np.ones(C_out, dtype=np.uint8))
    shift_const = relax.Constant(np.zeros(C_out, dtype=np.uint8))

    def te_mmalib_conv2d(
        data: te.Tensor,
        weight: te.Tensor,
        bias_t: te.Tensor,
        scale_t: te.Tensor,
        shift_t: te.Tensor,
    ) -> te.Tensor:
        def fcompute(ins, outs):
            return _call_extern_checked(
                "int32",
                "mmalib_conv2d_i16",
                ins[0].data,  # input
                ins[1].data,  # kernel
                ins[2].data,  # bias  (int64[C_out])
                ins[3].data,  # scale (uint8[C_out])
                ins[4].data,  # shift (uint8[C_out])
                outs[0].data,  # output
                C_in,
                H_in,
                W_in,
                C_out,
                KH,
                KW,
                stride_h,
                stride_w,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
            )

        return te.extern(
            (1, C_out, H_out, W_out),
            [data, weight, bias_t, scale_t, shift_t],
            fcompute,
            name="mmalib_conv2d",
            dtype="int16",
        )

    return propagate_span(
        bb.call_te(
            te_mmalib_conv2d,
            call.args[0],
            call.args[1],
            bias_const,
            scale_const,
            shift_const,
            primfunc_name_hint="mmalib_conv2d",
            primfunc_attrs={"c7x_offload_backend": "mmalib"},
        ),
        call,
    )


# =======================================================================
# Shared helpers (used by ti_mmalib_qdq_fusion.py)
# =======================================================================


def _float_to_scale_shift(rescale: np.ndarray):
    """Convert per-channel float rescale to (int8 scale, uint8 shift).

    Finds (s, sh) per channel such that s * 2^(-sh) ≈ rescale[ch],
    where s is in [1, 127] (signed int8 positive range) and sh is in [0, 31].

    MMALIB's matrixMatrixMultiplyBias expects signed int8 scale values
    (confirmed by MMALIB test case 8 which declares scale as int8_t).

    Raises ValueError if any channel's rescale is outside the representable
    range [2^-31, 127].  The previous behavior silently emitted scale=0/
    shift=0 for non-positive values (all-zero output) or scale=1/shift=0 for
    tiny values (~1.0 rescale), yielding silently wrong results.
    """
    min_rescale = 2.0**-31
    max_rescale = 127.0

    n_channels = rescale.shape[0]
    scale_out = np.zeros(n_channels, dtype=np.uint8)
    shift_out = np.zeros(n_channels, dtype=np.uint8)

    for ch in range(n_channels):
        r = float(rescale[ch])
        if not (min_rescale <= r <= max_rescale):
            raise ValueError(
                f"rescale[{ch}]={r} is outside the representable range "
                f"[{min_rescale}, {max_rescale}]"
            )

        best_err = float("inf")
        best_s, best_sh = 1, 0
        for sh in range(32):
            s_float = r * (1 << sh)
            s_int = int(round(s_float))
            if s_int < 1:
                continue
            if s_int > 127:
                break
            actual = s_int / (1 << sh)
            err = abs(actual - r) / r
            if err < best_err:
                best_err = err
                best_s = s_int
                best_sh = sh

        scale_out[ch] = best_s
        shift_out[ch] = best_sh

    return scale_out, shift_out


def _scale_shift_or_none(rescale: np.ndarray):
    """Like _float_to_scale_shift, but returns (None, None) instead of raising."""
    try:
        return _float_to_scale_shift(rescale)
    except ValueError:
        return None, None


def _validate_and_fold_bias(
    w_scale_np: np.ndarray,
    n_channels: int,
    bias_np,
    d_scale_val: float,
    o_scale_val: float,
    zp_correction: Union[int, np.ndarray] = 0,
    o_zp_val: int = 0,
    bias_dtype: type = np.int32,
    op_name: str = "MMALIB op",
):
    """Validate the weight-scale size and rescale range, then fold the
    float bias into accumulator scale.

    Order matters: both checks below must run *before* the bias fold. A
    collapsed weight scale (e.g. float32 eps for a constant tensor) makes
    dw_scale tiny, so bias/dw_scale can overflow int32 -- validating first
    avoids emitting a RuntimeWarning and then discarding a garbage bias.
    ``bias_dtype=np.int32`` (the int8 MMALIB kernels' accumulator) is the
    only width where this overflow is a real risk, so the range check
    below applies only then; ``np.int64`` (the int16 kernels') has ample
    headroom for any bias/dw_scale ratio this pass produces.

    Returns (scale_u8, shift_u8, bias_folded), or None to signal that the
    caller should decline the fusion.
    """
    if w_scale_np.size != n_channels:
        logger.warning("Per-tensor weight scale is not supported for %s; declining", op_name)
        return None
    dw_scale = d_scale_val * w_scale_np[:n_channels]
    combined_rescale = dw_scale / o_scale_val
    scale_u8, shift_u8 = _scale_shift_or_none(combined_rescale)
    if scale_u8 is None:
        logger.warning("Rescale out of range for %s; declining", op_name)
        return None

    if bias_np is not None:
        bias_accum_f = np.round(bias_np[:n_channels] / dw_scale)
        if bias_dtype == np.int32 and (
            not np.all(np.isfinite(bias_accum_f))
            or np.any(np.abs(bias_accum_f) > np.iinfo(np.int32).max)
        ):
            logger.warning("Bias fold overflows int32 accumulator for %s; declining", op_name)
            return None
        bias_accum = bias_accum_f.astype(bias_dtype)
    else:
        bias_accum = np.zeros(n_channels, dtype=bias_dtype)

    bias_folded = (bias_accum + zp_correction).astype(bias_dtype)

    if o_zp_val != 0:
        bias_folded = (bias_folded + np.round(o_zp_val / combined_rescale)).astype(bias_dtype)

    return scale_u8, shift_u8, bias_folded


def _call_extern_checked(dtype: str, op_name: str, *args):
    """Emit ``call_extern(dtype, op_name, ...)`` and fail loudly on non-zero status.

    The offload kernels (MMALIB/TIDL/SDPA/residual-add) return an int32
    status (0 == success).  Previously that status was discarded, so an OOM
    or init failure silently produced stale/partial output.  This wraps the
    call in a ``let`` + ``if`` that reports the error through the firmware's
    exported ``tvm_dsp_report_error`` and then aborts the enclosing PrimFunc
    via ``tvm_throw_last_error`` -- the same builtin lower_l2sram_alloc.py
    uses for its own tvm_l2_alloc null-check, confirmed (by inspecting
    generated code) to compile to a plain early ``return -1;`` here, not
    the packed-call "set the return value" path ``tir.ret`` would take.
    Without this, the wrapper function fell through to its own
    unconditional ``return 0`` on the very next line: it reported the
    error but still told the caller the call succeeded, so execution
    went on to consume whatever stale or partial output the failed kernel
    left. The generated driver (lib0.c) already checks and propagates a
    non-zero return from this wrapper the same way it does for the
    L2-alloc failure paths (``if (__call_ret != 0) return __call_ret;``),
    so aborting here is sufficient to turn the failure into a real,
    surfaced ModelError instead of a log line next to garbage output.
    """
    call = tir.call_extern(dtype, op_name, *args)
    status = tir.Var("status", "int64")
    return tir.LetStmt(
        status,
        tir.Cast("int64", call),
        tir.IfThenElse(
            status != 0,
            tir.SeqStmt(
                [
                    tir.Evaluate(
                        tir.call_extern(
                            "int32",
                            "tvm_dsp_report_error",
                            tir.StringImm(op_name),
                            tir.Cast("int32", status),
                        )
                    ),
                    tir.Evaluate(tir.tvm_throw_last_error()),
                ]
            ),
            None,
        ),
    )


# Shape-only ops that PT2E/ATen's conv-bias decomposition may interpose
# between a bias `relax.Constant` leaf and the `relax.add` that broadcasts
# it against the conv/matmul output (e.g. reshape (C,) -> (1, C, 1, 1)).
_CONSTANT_FOLDABLE_SHAPE_OPS = (
    "relax.reshape",
    "relax.expand_dims",
    "relax.squeeze",
    "relax.astype",
)


def _resolve_constant_tensor(expr, lookup=None):
    """Resolve `expr` to a numpy array if it is a compile-time constant.

    Returns the literal array for a bare `relax.Constant`. Also unwraps
    chains of shape-only ops (reshape/expand_dims/squeeze/astype) applied
    to a `relax.Constant` leaf -- e.g. PT2E's conv-bias decomposition
    represents `add(conv_out, bias)` as `add(conv_out, reshape(bias_const,
    (1, C, 1, 1)))` rather than passing the bias constant directly, so a
    bare `isinstance(x, relax.Constant)` check misses it and the bias is
    silently dropped (folded as all-zero) by every MMALIB QDQ lowering
    pass. Returns None if `expr` is not a compile-time constant.

    `lookup`: optional `Var -> Optional[Expr]` callable (e.g. a
    `PyExprMutator.lookup_binding` bound method) used to dereference a
    `relax.Var`/`DataflowVar` to its bound value. Needed when `expr` comes
    from a composite call's call-site argument (as in each pass's `_lower`)
    rather than from `PatternCheckContext.annotated_expr` (which is already
    fully dereferenced during pattern matching).
    """
    if isinstance(expr, relax.Constant):
        return expr.data.numpy()
    if isinstance(expr, relax.Var):
        if lookup is None:
            return None
        bound = lookup(expr)
        if bound is None:
            return None
        return _resolve_constant_tensor(bound, lookup)
    if (
        isinstance(expr, relax.Call)
        and hasattr(expr.op, "name")
        and expr.op.name in _CONSTANT_FOLDABLE_SHAPE_OPS
        and expr.struct_info is not None
        and expr.struct_info.shape is not None
    ):
        inner = _resolve_constant_tensor(expr.args[0], lookup)
        if inner is None:
            return None
        if expr.op.name == "relax.astype":
            return inner.astype(str(expr.struct_info.dtype))
        out_shape = [int(s) for s in expr.struct_info.shape]
        return inner.reshape(out_shape)
    return None


# =======================================================================
# Public passes
# =======================================================================


@tvm.transform.module_pass(opt_level=0, name="MMALIBLegalize")
class MMALIBLegalize:
    """Wraps LegalizeOps with MMALIB custom legalization for int16 ops."""

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        custom_map = {
            "relax.matmul": _mmalib_matmul_legalize,
            "relax.nn.conv2d": _mmalib_conv2d_legalize,
        }
        return relax.transform.LegalizeOps(customize_legalize_map=custom_map)(mod)
