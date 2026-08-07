"""Fuse GQA expand + attention_bias into c7x_sdpa_decode extern for decode (seq_q=1).

Matches the pattern produced by StaticCache attention at seq_len=1:

    expand_dims(K_scatter) → broadcast_to → reshape → permute_dims → K_t
    expand_dims(V_scatter) → broadcast_to → reshape → permute_dims → V_t
    permute_dims(Q_rope) → Q_t
    attention_bias(Q_t, K_t, V_t, mask) → attn_output

Phase 1: FuseOpsByPattern groups expand+broadcast+reshape+permute+attention_bias
          into a composite function (K/V scatter outputs become parameters).
Phase 2: PyExprMutator lowers the composite to call_extern("c7x_sdpa_decode").

Must run AFTER: RewriteDequantize, _add_kv_scatter_outputs
Must run BEFORE: LegalizeOps, FuseOps (i.e. before compile_for_dsp)
"""

import logging

import tvm
from tvm import relax, te, tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator

from .ti_c7x_span_utils import find_composite_span, propagate_span

logger = logging.getLogger(__name__)

COMPOSITE_NAME = "c7x.sdpa_decode"


def _sdpa_decode_pattern():
    """DPL pattern: expand+broadcast+reshape+permute for K and V, then attention_bias."""
    # K chain: scatter_output → expand_dims → broadcast_to → reshape → permute_dims
    k_input = wildcard()
    k_expand = is_op("relax.expand_dims")(k_input)
    k_broadcast = is_op("relax.broadcast_to")(k_expand, wildcard())
    k_reshape = is_op("relax.reshape")(k_broadcast, wildcard())
    k_permute = is_op("relax.permute_dims")(k_reshape)

    # V chain: same structure
    v_input = wildcard()
    v_expand = is_op("relax.expand_dims")(v_input)
    v_broadcast = is_op("relax.broadcast_to")(v_expand, wildcard())
    v_reshape = is_op("relax.reshape")(v_broadcast, wildcard())
    v_permute = is_op("relax.permute_dims")(v_reshape)

    # Q: just permute_dims
    q_input = wildcard()
    q_permute = is_op("relax.permute_dims")(q_input)

    # Mask
    mask = wildcard()

    # attention_bias(Q_t, K_t, V_t, mask)
    attn = is_op("relax.nn.attention_bias")(q_permute, k_permute, v_permute, mask)

    annotations = {
        "k_input": k_input,
        "v_input": v_input,
        "q_input": q_input,
        "mask": mask,
        "attn": attn,
    }
    return attn, annotations, _check_sdpa_decode


def _check_sdpa_decode(ctx) -> bool:
    """Constraint: verify this is a decode attention (seq_q=1) with GQA."""
    k_input = ctx.annotated_expr.get("k_input")
    q_input = ctx.annotated_expr.get("q_input")
    if k_input is None or q_input is None:
        return False

    # K input shape should be [1, kv_heads, cache_len, head_dim]
    k_sinfo = k_input.struct_info
    if not isinstance(k_sinfo, relax.TensorStructInfo) or k_sinfo.shape is None:
        return False
    try:
        k_shape = [int(s) for s in k_sinfo.shape]
    except (TypeError, ValueError):
        return False
    if len(k_shape) != 4 or k_shape[0] != 1:
        return False

    # Q input shape should be [1, num_q_heads, 1, head_dim] — seq_q=1
    q_sinfo = q_input.struct_info
    if not isinstance(q_sinfo, relax.TensorStructInfo) or q_sinfo.shape is None:
        return False
    try:
        q_shape = [int(s) for s in q_sinfo.shape]
    except (TypeError, ValueError):
        return False
    if len(q_shape) != 4 or q_shape[0] != 1 or q_shape[2] != 1:
        return False

    # GQA: num_q_heads must be a multiple of num_kv_heads
    num_q_heads = q_shape[1]
    num_kv_heads = k_shape[1]
    if num_q_heads % num_kv_heads != 0:
        return False

    return True


@mutator
class _SDPADecodeLowerer(PyExprMutator):
    """Lower c7x.sdpa_decode composite functions to c7x_sdpa_decode extern."""

    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod
        self.count = 0

    def visit_call_(self, call: relax.Call):
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)

        func = self.mod[call.op]
        if not isinstance(func, relax.Function):
            return super().visit_call_(call)
        composite = func.attrs.get("Composite", "") if func.attrs else ""
        if composite != COMPOSITE_NAME:
            return super().visit_call_(call)

        # The composite function has params that map to call.args:
        # params correspond to: k_input, v_input, q_input, mask (in the order
        # FuseOpsByPattern places them — matched by the pattern wildcards)
        # We need to identify which param is which by tracing the composite body.
        param_to_arg = dict(zip(func.params, call.args))

        # Walk composite body to find the annotated inputs
        # The composite body structure:
        #   param_k → expand_dims → broadcast_to → reshape → permute_dims → k_t
        #   param_v → expand_dims → broadcast_to → reshape → permute_dims → v_t
        #   param_q → permute_dims → q_t
        #   param_mask → mask
        #   attention_bias(q_t, k_t, v_t, mask) → output
        k_arg = None
        v_arg = None
        q_arg = None
        mask_arg = None
        attn_call = None

        var_to_val = {}
        for block in func.body.blocks:
            for b in block.bindings:
                if isinstance(b, relax.VarBinding):
                    var_to_val[b.var] = b.value

        # Find attention_bias call in composite
        for block in func.body.blocks:
            for b in block.bindings:
                if not isinstance(b, relax.VarBinding):
                    continue
                val = b.value
                if isinstance(val, relax.Call) and "attention_bias" in str(val.op):
                    attn_call = val
                    break
            if attn_call:
                break

        if attn_call is None:
            return super().visit_call_(call)

        # Trace Q (arg[0] of attention_bias) back through permute_dims to param
        def _trace_to_param(var):
            cur = var
            while cur in var_to_val:
                val = var_to_val[cur]
                if isinstance(val, relax.Call) and len(val.args) > 0:
                    arg0 = val.args[0]
                    if isinstance(arg0, relax.Var):
                        cur = arg0
                    else:
                        break
                else:
                    break
            return param_to_arg.get(cur)

        q_t = attn_call.args[0]
        k_t = attn_call.args[1]
        v_t = attn_call.args[2]
        mask_t = attn_call.args[3]

        q_arg = _trace_to_param(q_t)
        k_arg = _trace_to_param(k_t)
        v_arg = _trace_to_param(v_t)
        mask_arg = _trace_to_param(mask_t)

        if any(x is None for x in [q_arg, k_arg, v_arg, mask_arg]):
            logger.warning("SDPA lowering: could not trace all inputs")
            return super().visit_call_(call)

        # Extract dimensions from input shapes
        # k_arg shape: [1, kv_heads, cache_len, head_dim]
        # q_arg shape: [1, num_q_heads, 1, head_dim]
        # mask_arg shape: [1, 1, 1, cache_len]
        k_shape = [int(s) for s in k_arg.struct_info.shape]
        q_shape = [int(s) for s in q_arg.struct_info.shape]

        num_kv_heads = k_shape[1]
        max_cache_len = k_shape[2]
        head_dim = k_shape[3]
        num_q_heads = q_shape[1]

        bb = self.builder_

        # Reshape inputs for the kernel
        q_sq = bb.emit(relax.op.reshape(q_arg, relax.ShapeExpr([num_q_heads, head_dim])))
        k_sq = bb.emit(relax.op.reshape(k_arg, relax.ShapeExpr([num_kv_heads, max_cache_len, head_dim])))
        v_sq = bb.emit(relax.op.reshape(v_arg, relax.ShapeExpr([num_kv_heads, max_cache_len, head_dim])))
        m_sq = bb.emit(relax.op.reshape(mask_arg, relax.ShapeExpr([max_cache_len])))

        nqh, nkvh, hd, mcl = num_q_heads, num_kv_heads, head_dim, max_cache_len

        def _te_sdpa(qt, kt, vt, mt, _nqh=nqh, _nkvh=nkvh, _hd=hd, _mcl=mcl):
            def fcompute(ins, outs):
                return tir.call_extern(
                    "int32", "c7x_sdpa_decode",
                    ins[0].data, ins[1].data, ins[2].data,
                    ins[3].data, outs[0].data,
                    _nqh, _nkvh, _hd, _mcl,
                )
            return te.extern(
                [_nqh, _hd], [qt, kt, vt, mt],
                fcompute, name="sdpa_decode", dtype="float32",
            )

        sdpa_out = propagate_span(
            bb.call_te(
                _te_sdpa, q_sq, k_sq, v_sq, m_sq,
                primfunc_name_hint="sdpa_decode",
            ),
            find_composite_span(func),
        )

        # Reshape to original attention_bias output shape
        out_sinfo = call.struct_info
        out_shape = [int(s) for s in out_sinfo.shape]
        result = bb.emit(relax.op.reshape(sdpa_out, relax.ShapeExpr(out_shape)))

        self.count += 1
        return result


@tvm.transform.module_pass(opt_level=0, name="FuseSDPADecode")
class FuseSDPADecode:
    """Fuse GQA attention pattern into c7x_sdpa_decode extern call.

    Only effective for decode models (seq_q=1). Prefill models (seq_q>1)
    are not matched because the constraint checks seq_q==1.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Phase 1: Pattern match — group expand+broadcast+reshape+permute+attention_bias
        patterns = [
            (COMPOSITE_NAME, *_sdpa_decode_pattern()),
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: Lower composites to call_extern
        lowerer = _SDPADecodeLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                new_func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, new_func)
        mod = lowerer.builder_.get()

        if lowerer.count > 0:
            logger.info(f"FuseSDPADecode: lowered {lowerer.count} attention layers")
            mod = relax.transform.DeadCodeElimination()(mod)

        return mod
