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
"""Inject L2 DMA prefetch for MMALIB extern PrimFuncs.

MMALIB conv2d/matmul PrimFuncs are single call_extern statements with
no loops.  This TIR pass emits blocking DMA transfers to copy input
(and weights when they fit) from DDR into L2 SRAM before the MMALIB
call, so the MMA coprocessor operates from fast scratchpad memory.

The pass runs after StorageRewrite and before LowerL2SramAlloc in the
C7x TIR pipeline.  It emits:
  - AllocBuffer(scope="global.l2sram") for L2 buffers
  - call_extern("tvm_dsp_dma_copy") for async DMA transfers
  - call_extern("tvm_dsp_dma_wait") to block until complete
  - Modified MMALIB call_extern with L2 buffer pointers

LowerL2SramAlloc converts the AllocBuffer statements to tvm_l2_alloc at runtime.
"""

import logging

import tvm_ffi

import tvm
from tvm import tirx

from .ti_mmalib_constants import MMA_SIZE_I8
from .ti_mmalib_legalize import _call_extern_checked

logger = logging.getLogger(__name__)

_MMALIB_CONV2D_I8 = "mmalib_conv2d_i8"
_MMALIB_CONV2D_I8_GROUPED_LOOP = "mmalib_conv2d_i8_grouped_loop"
_MMALIB_CONV2D_I16 = "mmalib_conv2d_i16"
_MMALIB_DWCONV2D_I8 = "mmalib_depthwise_conv2d_i8"
_MMALIB_DWCONV2D_I16 = "mmalib_depthwise_conv2d_i16"
_MMALIB_FC_I8 = "mmalib_matmul_bias_i8"
_MMALIB_FC_I16 = "mmalib_matmul_bias_i16"
_SUPPORTED = {
    _MMALIB_CONV2D_I8,
    _MMALIB_CONV2D_I8_GROUPED_LOOP,
    _MMALIB_CONV2D_I16,
    _MMALIB_DWCONV2D_I8,
    _MMALIB_DWCONV2D_I16,
    _MMALIB_FC_I8,
    _MMALIB_FC_I16,
}


def _find_mmalib_call(body):
    """Find a call_extern to an MMALIB function in the statement tree.

    Matches both a bare ``Evaluate(call_extern(...))`` and the status-checked
    form emitted by ``_call_extern_checked`` (``Bind(status, Cast(...,
    call_extern(...)))`` followed by an ``IfThenElse``) -- MMALIB legalization wraps its
    calls in the latter to report kernel failures instead of discarding the
    status, so a matcher that only recognized the bare form would silently
    stop finding any MMALIB call to inject DMA for.

    Returns (Call node, extern name, checked) or (None, None, None).
    `checked` is True when the call was found in the status-checked form,
    meaning any rewrite of its args must go back through
    `_call_extern_checked` rather than a bare `Evaluate` to preserve the
    error check.
    """
    call_node = None
    extern_name = None
    checked = False
    op_call_extern = tvm.ir.Op.get("tirx.call_extern")

    def _visit(node):
        nonlocal call_node, extern_name, checked
        if call_node is not None:
            return
        if isinstance(node, tirx.Evaluate):
            call = node.value
        elif isinstance(node, tirx.Bind):
            call = node.value.value if isinstance(node.value, tirx.Cast) else node.value
        else:
            return
        if (
            isinstance(call, tvm.ir.Call)
            and call.op.same_as(op_call_extern)
            and len(call.args) > 0
            and isinstance(call.args[0], tvm.ir.StringImm)
            and call.args[0].value in _SUPPORTED
        ):
            call_node = call
            extern_name = call.args[0].value
            checked = isinstance(node, tirx.Bind)

    tvm_ffi.structural_walk(body, [(tirx.Evaluate, _visit), (tirx.Bind, _visit)], order="post")
    return call_node, extern_name, checked


def _extract_dims_conv2d_i8(call_args):
    """Extract dimensions from mmalib_conv2d_i8 call_extern args.

    Args: (name, input, kernel, bias, scale, shift, output,
           C_in, H_in, W_in, C_out, KH, KW,
           stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right)
    Indices are 0-based after the return type in call_extern.
    In the Call node: args[0]=name_str, args[1]=input, ..., args[7]=C_in, ...

    `groups` defaults to 1: the weight tensor is [C_out, C_in, KH, KW] for
    this (ungrouped) kernel. See _extract_dims_conv2d_i8_grouped_loop for
    the grouped variant, where the weight tensor is [C_out, C_in/groups,
    KH, KW] instead and `groups` must be read from the call to size the
    weight DMA correctly.
    """
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": int(call_args[7].value),
        "h_in": int(call_args[8].value),
        "w_in": int(call_args[9].value),
        "c_out": int(call_args[10].value),
        "kh": int(call_args[11].value),
        "kw": int(call_args[12].value),
        "groups": 1,
    }


def _extract_dims_conv2d_i8_grouped_loop(call_args):
    """Extract dimensions from mmalib_conv2d_i8_grouped_loop call_extern args.

    args[0..18] are identical to mmalib_conv2d_i8's (C_in/C_out are the
    full, ungrouped channel counts -- the *input* DDR region this pass
    prefetches into L2 is the same full C_in*H_in*W_in buffer the grouped
    C++ loop internally slices per group). The one extra arg, `groups` at
    args[19], is read here because it changes the *weight* tensor's real
    size: PyTorch's native grouped-conv weight layout is
    [C_out, C_in/groups, KH, KW], not [C_out, C_in, KH, KW] -- using the
    full C_in to size the weight DMA would over-read past the actual
    weight buffer.
    """
    dims = _extract_dims_conv2d_i8(call_args)
    dims["groups"] = int(call_args[19].value)
    return dims


def _extract_dims_conv2d_i16(call_args):
    """Extract dimensions from mmalib_conv2d_i16 call_extern args.

    Args: (name, input, kernel, bias, scale, shift, output,
           C_in, H_in, W_in, C_out, KH, KW, stride_h, stride_w, ...)

    Note: bias/scale/shift were added in Phase 2b to match the i8 interface.
    The dimension indices now match mmalib_conv2d_i8 exactly.
    """
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": int(call_args[7].value),
        "h_in": int(call_args[8].value),
        "w_in": int(call_args[9].value),
        "c_out": int(call_args[10].value),
        "kh": int(call_args[11].value),
        "kw": int(call_args[12].value),
    }


def _extract_dims_dwconv2d_i8(call_args):
    """Extract dimensions from mmalib_depthwise_conv2d_i8 call_extern args.

    Args: (name, input, reordered_weights, bias, scale, shift, output,
           channels, H_in, W_in, KH, KW, stride_h, stride_w,
           pad_top, pad_bottom, pad_left, pad_right, num_groups)
    """
    channels = int(call_args[7].value)
    h_in = int(call_args[8].value)
    w_in = int(call_args[9].value)
    kh = int(call_args[10].value)
    kw = int(call_args[11].value)
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": channels,
        "h_in": h_in,
        "w_in": w_in,
        "c_out": channels,
        "kh": kh,
        "kw": kw,
        # True depthwise: weight tensor is [channels, 1, KH, KW], not
        # [channels, channels, KH, KW]. `c_in` above is deliberately the
        # full channel count (correct for input_bytes = c_in*h_in*w_in),
        # so `groups=channels` is needed here to make weight_bytes's
        # `c_in // groups` resolve to 1 instead of over-reading the
        # weight buffer by a factor of `channels`.
        "groups": channels,
    }


def _extract_dims_fc_i8(call_args):
    """Extract dimensions from mmalib_matmul_bias_i8 call_extern args.

    Args: (name, input, weights, bias, scale, shift, output, M, K, N)
    """
    m = int(call_args[7].value)
    k = int(call_args[8].value)
    n = int(call_args[9].value)
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": k,
        "h_in": m,
        "w_in": 1,
        "c_out": n,
        "kh": 1,
        "kw": 1,
    }


def _extract_dims_dwconv2d_i16(call_args):
    """Extract dimensions from mmalib_depthwise_conv2d_i16 call_extern args.

    Same argument layout as mmalib_depthwise_conv2d_i8 (same signature),
    so the geometry mapping is identical — only elem_bytes differs (2 vs 1).

    Args: (name, input, weights, bias, scale, shift, output,
           channels, H_in, W_in, KH, KW, stride_h, stride_w,
           pad_top, pad_bottom, pad_left, pad_right, num_groups)
    """
    channels = int(call_args[7].value)
    h_in = int(call_args[8].value)
    w_in = int(call_args[9].value)
    kh = int(call_args[10].value)
    kw = int(call_args[11].value)
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": channels,
        "h_in": h_in,
        "w_in": w_in,
        "c_out": channels,
        "kh": kh,
        "kw": kw,
        # See the matching comment in _extract_dims_dwconv2d_i8 -- true
        # depthwise weight tensor is [channels, 1, KH, KW].
        "groups": channels,
    }


def _extract_dims_fc_i16(call_args):
    """Extract dimensions from mmalib_matmul_bias_i16 call_extern args.

    Args: (name, input, weights, bias, scale, shift, output, M, K, N)

    Same arg layout as mmalib_matmul_bias_i8 but elem_bytes=2 for int16.
    The geometry mapping (K→c_in, M→h_in, N→c_out) treats the FC as a
    1×1 spatial convolution for the purpose of DMA budget calculation.
    """
    m = int(call_args[7].value)
    k = int(call_args[8].value)
    n = int(call_args[9].value)
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": k,
        "h_in": m,
        "w_in": 1,
        "c_out": n,
        "kh": 1,
        "kw": 1,
    }


def _l2_buffer(name, nbytes):
    """Create an int8 ``global.l2sram`` buffer variable of ``nbytes`` bytes."""
    return tirx.decl_buffer([int(nbytes)], "int8", name=name, scope="global.l2sram")


def _collect_decl_buffers(body):
    """Return every ``DeclBuffer`` statement in ``body``, in source order.

    After FlattenBuffer a PrimFunc's parameters are aliased through
    ``DeclBuffer(x_1, data=x.data)`` statements, and the MMALIB call refers to
    the aliases.  The pass rebuilds the function body from scratch, so the
    declarations must be carried over or the aliases become free variables.
    """
    decls = []
    tvm_ffi.structural_walk(body, (tirx.DeclBuffer, lambda node: decls.append(node)), order="pre")
    return decls


def _with_l2_allocs(allocs, body, decls=()):
    """Prepend ``AllocBuffer`` statements for ``allocs`` (in order) to ``body``.

    AllocBuffer has no body: the buffer is in scope for the statements that
    follow it in the enclosing SeqStmt.  LowerL2SramAlloc collects the leading
    run of ``global.l2sram`` AllocBuffer statements of the function body, so
    the allocations must come first and in this order (guard outermost).  The
    original ``decls`` follow the allocations and precede ``body``.
    """
    return tirx.SeqStmt([*(tirx.AllocBuffer(buf) for buf in allocs), *decls, body])


def _inject_dma(func, l2_budget):
    """Transform a MMALIB PrimFunc to prefetch data into L2 via DMA."""
    call_node, extern_name, checked = _find_mmalib_call(func.body)
    if call_node is None:
        return func

    if extern_name == _MMALIB_CONV2D_I8:
        dims = _extract_dims_conv2d_i8(call_node.args)
        elem_bytes = 1
    elif extern_name == _MMALIB_CONV2D_I8_GROUPED_LOOP:
        dims = _extract_dims_conv2d_i8_grouped_loop(call_node.args)
        elem_bytes = 1
    elif extern_name == _MMALIB_CONV2D_I16:
        dims = _extract_dims_conv2d_i16(call_node.args)
        elem_bytes = 2
    elif extern_name == _MMALIB_DWCONV2D_I8:
        dims = _extract_dims_dwconv2d_i8(call_node.args)
        elem_bytes = 1
    elif extern_name == _MMALIB_FC_I8:
        dims = _extract_dims_fc_i8(call_node.args)
        elem_bytes = 1
    elif extern_name == _MMALIB_DWCONV2D_I16:
        dims = _extract_dims_dwconv2d_i16(call_node.args)
        elem_bytes = 2
    elif extern_name == _MMALIB_FC_I16:
        dims = _extract_dims_fc_i16(call_node.args)
        elem_bytes = 2
    else:
        return func

    c_in = dims["c_in"]
    h_in = dims["h_in"]
    w_in = dims["w_in"]
    c_out = dims["c_out"]
    kh = dims["kh"]
    kw = dims["kw"]
    groups = dims.get("groups", 1)

    input_bytes = c_in * h_in * w_in * elem_bytes
    # Weight tensor is [C_out, C_in/groups, KH, KW]; groups=1 for every
    # kernel except mmalib_conv2d_i8_grouped_loop, where using the full
    # C_in here would over-read past the actual weight buffer.
    weight_bytes = c_out * (c_in // groups) * kh * kw * elem_bytes

    if input_bytes > l2_budget:
        logger.debug(
            "MMALIB input %d KB exceeds L2 %d KB, skipping",
            input_bytes // 1024,
            l2_budget // 1024,
        )
        return func

    cache_weight = (input_bytes + weight_bytes) <= l2_budget

    # OC-tiling: when weights don't fit whole but input fits and we can tile
    # by output channel.  Only for int8 conv2d (not i16, dwconv, or FC):
    # mmalib_conv2d_i8_sliced hardcodes int8 element sizes for output (1 byte)
    # and int32 bias (4 bytes); i16 uses different sizes and needs its own
    # sliced wrapper before OC-tiling can be enabled for it.
    oc_tile = None
    weight_tile_bytes = None
    if not cache_weight and extern_name == _MMALIB_CONV2D_I8:
        weight_per_oc = c_in * kh * kw * elem_bytes
        avail = l2_budget - input_bytes
        if avail > 0 and weight_per_oc > 0:
            raw_tile = avail // weight_per_oc
            raw_tile = (raw_tile // MMA_SIZE_I8) * MMA_SIZE_I8
            if raw_tile >= MMA_SIZE_I8:
                oc_tile = min(raw_tile, c_out)
                weight_tile_bytes = oc_tile * weight_per_oc

    queue_id = tirx.const(0, "int32")
    bypass = tirx.const(0, "int32")
    zero_inflight = tirx.const(0, "int32")
    stmts = []

    # Guard: prevents SE backward prefetch from underflowing L2 start.
    # For conv2d kernels (i8, i16, and grouped_loop), pad_top is at args[15]
    # in all three (grouped_loop's args[0..18] match mmalib_conv2d_i8's
    # exactly, with an extra `groups` at args[19] that doesn't shift this):
    #   (name, input, kernel, bias, scale, shift, output, C_in, H_in, W_in,
    #    C_out, KH, KW, stride_h, stride_w, pad_top, ...)
    if extern_name in (_MMALIB_CONV2D_I8, _MMALIB_CONV2D_I8_GROUPED_LOOP, _MMALIB_CONV2D_I16):
        # args[15] is pad_top for i8, i16, and grouped_loop conv2d.  Guard
        # against symbolic values (e.g. from dynamic-padding graphs) that
        # have no .value attribute — fall back to the safe 128-byte default.
        arg15 = call_node.args[15]
        pad_top = int(arg15.value) if isinstance(arg15, tirx.IntImm) else 0
    else:
        pad_top = 0
    guard_bytes = pad_top * w_in * elem_bytes if pad_top > 0 else 128
    l2_guard_var = _l2_buffer("l2_guard", guard_bytes)

    l2_vars = {}  # arg_idx -> (l2_var, nbytes)

    # -------------------------------------------------------------------------
    # OC-tiled path: DMA input once, loop over OC tiles loading one weight
    # tile per iteration and calling mmalib_conv2d_i8_sliced.
    # -------------------------------------------------------------------------
    if oc_tile is not None:
        l2_input_var = _l2_buffer("l2_input", input_bytes)
        l2_weight_var = _l2_buffer("l2_weight_tile", weight_tile_bytes)

        # DMA full input once (stays in L2 for all OC tiles)
        stmts.append(
            tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    "tvm_dsp_dma_copy",
                    queue_id,
                    tirx.buffer_data(l2_input_var),
                    call_node.args[dims["input_arg_idx"]],
                    tirx.const(input_bytes, "int32"),
                    bypass,
                )
            )
        )
        stmts.append(
            tirx.Evaluate(tirx.call_extern("int32", "tvm_dsp_dma_wait", queue_id, zero_inflight))
        )

        # Build For loop: one weight-tile DMA + sliced call per iteration
        oc_idx = tirx.Var("oc_chunk", "int32")
        n_tiles = (c_out + oc_tile - 1) // oc_tile
        oc_start = tirx.Mul(oc_idx, tirx.const(oc_tile, "int32"))
        tile_c_out = tirx.Min(
            tirx.const(oc_tile, "int32"), tirx.Sub(tirx.const(c_out, "int32"), oc_start)
        )
        actual_dma = tirx.Mul(tile_c_out, tirx.const(c_in * kh * kw * elem_bytes, "int32"))

        # weight DDR pointer + oc_idx * weight_tile_bytes via tvm_ptr_add
        # (direct handle arithmetic is not valid in TVM TIR; use a C helper).
        wt_base = call_node.args[dims["weight_arg_idx"]]
        wt_byte_off = tirx.Mul(oc_idx, tirx.const(weight_tile_bytes, "int32"))
        wt_tile_src = tirx.call_extern("handle", "tvm_ptr_add", wt_base, wt_byte_off)

        # mmalib_conv2d_i8_sliced args: matches mmalib_conv2d_i8 minus C_out,
        # plus oc_start at the end.  Bias/scale/shift/output stay as DDR ptrs;
        # the wrapper advances them by oc_start internally.
        sliced_args = [
            tvm.ir.StringImm("mmalib_conv2d_i8_sliced"),
            tirx.buffer_data(l2_input_var),  # input  (L2)
            tirx.buffer_data(l2_weight_var),  # kernel (L2 tile)
            call_node.args[3],  # bias   (DDR full)
            call_node.args[4],  # scale  (DDR full)
            call_node.args[5],  # shift  (DDR full)
            call_node.args[6],  # output (DDR full)
            call_node.args[7],  # C_in
            call_node.args[8],  # H_in
            call_node.args[9],  # W_in
            tile_c_out,  # C_out_tile
            call_node.args[11],  # KH
            call_node.args[12],  # KW
            call_node.args[13],  # stride_h
            call_node.args[14],  # stride_w
            call_node.args[15],  # pad_top
            call_node.args[16],  # pad_bottom
            call_node.args[17],  # pad_left
            call_node.args[18],  # pad_right
            oc_start,  # oc_start
        ]
        tile_body = tirx.SeqStmt(
            [
                tirx.Evaluate(
                    tirx.call_extern(
                        "int32",
                        "tvm_dsp_dma_copy",
                        queue_id,
                        tirx.buffer_data(l2_weight_var),
                        wt_tile_src,
                        actual_dma,
                        bypass,
                    )
                ),
                tirx.Evaluate(
                    tirx.call_extern("int32", "tvm_dsp_dma_wait", queue_id, zero_inflight)
                ),
                tirx.Evaluate(tvm.ir.Call(call_node.op, sliced_args, ret_ty=call_node.ty)),
            ]
        )
        stmts.append(tirx.For(oc_idx, 0, n_tiles, tirx.ForKind.SERIAL, tile_body))

        # Guard outermost (first), then input, then the weight tile.
        body = _with_l2_allocs(
            [l2_guard_var, l2_input_var, l2_weight_var],
            tirx.SeqStmt(stmts),
            _collect_decl_buffers(func.body),
        )
        logger.info(
            "InjectMMALIBDMA OC-tiled: %s [input=%dKB, oc_tile=%d, n_tiles=%d, weight_tile=%dKB]",
            extern_name,
            input_bytes // 1024,
            oc_tile,
            n_tiles,
            weight_tile_bytes // 1024,
        )
        return func.with_body(body)
    # -------------------------------------------------------------------------

    l2_input_var = _l2_buffer("l2_input", input_bytes)
    l2_vars[dims["input_arg_idx"]] = (l2_input_var, input_bytes)
    dma_copy = tirx.call_extern(
        "int32",
        "tvm_dsp_dma_copy",
        queue_id,
        l2_input_var,
        call_node.args[dims["input_arg_idx"]],
        tirx.const(input_bytes, "int32"),
        bypass,
    )
    stmts.append(tirx.Evaluate(dma_copy))

    if cache_weight:
        l2_weight_var = _l2_buffer("l2_weight", weight_bytes)
        l2_vars[dims["weight_arg_idx"]] = (l2_weight_var, weight_bytes)
        dma_copy = tirx.call_extern(
            "int32",
            "tvm_dsp_dma_copy",
            queue_id,
            tirx.buffer_data(l2_weight_var),
            call_node.args[dims["weight_arg_idx"]],
            tirx.const(weight_bytes, "int32"),
            bypass,
        )
        stmts.append(tirx.Evaluate(dma_copy))

    # Wait for all DMAs to complete
    dma_wait = tirx.call_extern("int32", "tvm_dsp_dma_wait", queue_id, zero_inflight)
    stmts.append(tirx.Evaluate(dma_wait))

    # Modified MMALIB call with L2 pointers. Preserve the status-check
    # wrapper (_call_extern_checked) if the original call had one, or its
    # error reporting is silently lost.
    new_args = list(call_node.args)
    for arg_idx, (l2_var, _) in l2_vars.items():
        new_args[arg_idx] = tirx.buffer_data(l2_var)
    if checked:
        stmts.append(_call_extern_checked(call_node.ty, extern_name, *new_args[1:]))
    else:
        stmts.append(tirx.Evaluate(tvm.ir.Call(call_node.op, new_args, ret_ty=call_node.ty)))

    # Build body: the L2 allocations first, then the DMA + call sequence.
    # The guard allocation comes first -- it bumps the L2 pointer past the danger
    # zone so the input buffer doesn't start at the very beginning of L2.  The
    # remaining buffers follow in ascending argument-index order.  This is the same
    # allocation order the bump allocator saw with the old nested Allocate chain,
    # which was wrapped from the highest index outward (lowest index outermost).
    ordered = [l2_vars[i][0] for i in sorted(l2_vars.keys())]
    body = _with_l2_allocs(
        [l2_guard_var, *ordered], tirx.SeqStmt(stmts), _collect_decl_buffers(func.body)
    )

    cached = [f"input={input_bytes // 1024}KB"]
    if cache_weight:
        cached.append(f"weight={weight_bytes // 1024}KB")
    logger.info("InjectMMALIBDMA: %s [%s]", extern_name, ", ".join(cached))

    return func.with_body(body)


def InjectMMALIBDMA(l2_budget=393216):
    """Create a TIR pass that injects blocking DMA for MMALIB PrimFuncs.

    Scans each PrimFunc for mmalib_conv2d_i8/i8_grouped_loop/i16 call_extern
    nodes. When found, wraps the call with L2 allocations and DMA copy/wait
    so MMALIB operates from L2 SRAM instead of DDR.

    Parameters
    ----------
    l2_budget : int
        L2 SRAM budget in bytes. Default 393216 (384 KB).

    Returns
    -------
    fpass : tvm.transform.Pass
    """

    @tvm.tirx.transform.prim_func_pass(opt_level=0, name="InjectMMALIBDMA")
    def _pass(func, mod, ctx):  # noqa: ARG001
        return _inject_dma(func, l2_budget)

    return _pass
