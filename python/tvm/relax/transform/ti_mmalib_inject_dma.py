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
  - Allocate(scope="global.l2sram") for L2 buffers
  - call_extern("tvm_dsp_dma_copy") for async DMA transfers
  - call_extern("tvm_dsp_dma_wait") to block until complete
  - Modified MMALIB call_extern with L2 buffer pointers

LowerL2SramAlloc converts the Allocate to tvm_l2_alloc at runtime.
"""

import logging

import tvm
from tvm import tir
from tvm.ir import PointerType, PrimType

logger = logging.getLogger(__name__)

_MMALIB_CONV2D_I8 = "mmalib_conv2d_i8"
_MMALIB_CONV2D_I16 = "mmalib_conv2d_i16"
_SUPPORTED = {_MMALIB_CONV2D_I8, _MMALIB_CONV2D_I16}


def _find_mmalib_call(body):
    """Find a call_extern to an MMALIB function in the statement tree.

    Returns (Evaluate stmt, Call node, extern name) or (None, None, None).
    """
    result = [None, None, None]
    op_call_extern = tvm.ir.Op.get("tir.call_extern")

    def _visit(node):
        if result[0] is not None:
            return
        if isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir.Call) and call.op.same_as(op_call_extern):
                if len(call.args) > 0 and isinstance(call.args[0], tir.StringImm):
                    name = call.args[0].value
                    if name in _SUPPORTED:
                        result[0] = node
                        result[1] = call
                        result[2] = name

    tir.stmt_functor.post_order_visit(body, _visit)
    return result[0], result[1], result[2]


def _extract_dims_conv2d_i8(call_args):
    """Extract dimensions from mmalib_conv2d_i8 call_extern args.

    Args: (name, input, kernel, bias, scale, shift, output,
           C_in, H_in, W_in, C_out, KH, KW,
           stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right)
    Indices are 0-based after the return type in call_extern.
    In the Call node: args[0]=name_str, args[1]=input, ..., args[7]=C_in, ...
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


def _extract_dims_conv2d_i16(call_args):
    """Extract dimensions from mmalib_conv2d_i16 call_extern args.

    Args: (name, input, kernel, output,
           C_in, H_in, W_in, C_out, KH, KW, ...)
    """
    return {
        "input_arg_idx": 1,
        "weight_arg_idx": 2,
        "c_in": int(call_args[4].value),
        "h_in": int(call_args[5].value),
        "w_in": int(call_args[6].value),
        "c_out": int(call_args[7].value),
        "kh": int(call_args[8].value),
        "kw": int(call_args[9].value),
    }


def _inject_dma(func, l2_budget):
    """Transform a MMALIB PrimFunc to prefetch data into L2 via DMA."""
    _, call_node, extern_name = _find_mmalib_call(func.body)
    if call_node is None:
        return func

    if extern_name == _MMALIB_CONV2D_I8:
        dims = _extract_dims_conv2d_i8(call_node.args)
        elem_bytes = 1
    elif extern_name == _MMALIB_CONV2D_I16:
        dims = _extract_dims_conv2d_i16(call_node.args)
        elem_bytes = 2
    else:
        return func

    c_in = dims["c_in"]
    h_in = dims["h_in"]
    w_in = dims["w_in"]
    c_out = dims["c_out"]
    kh = dims["kh"]
    kw = dims["kw"]

    input_bytes = c_in * h_in * w_in * elem_bytes
    weight_bytes = c_out * c_in * kh * kw * elem_bytes

    if input_bytes > l2_budget:
        logger.debug(
            "MMALIB input %d KB exceeds L2 %d KB, skipping",
            input_bytes // 1024,
            l2_budget // 1024,
        )
        return func

    cache_weight = (input_bytes + weight_bytes) <= l2_budget

    queue_id = tir.const(0, "int32")
    bypass = tir.const(0, "int32")
    zero_inflight = tir.const(0, "int32")
    stmts = []

    # Guard: prevents SE backward prefetch from underflowing L2 start
    pad_top = int(call_node.args[13].value) if extern_name == _MMALIB_CONV2D_I8 else 0
    guard_bytes = pad_top * w_in * elem_bytes if pad_top > 0 else 128
    l2_guard_var = tir.Var("l2_guard", PointerType(PrimType("int8"), "global.l2sram"))

    l2_vars = {}  # arg_idx -> (l2_var, nbytes)

    l2_input_var = tir.Var("l2_input", PointerType(PrimType("int8"), "global.l2sram"))
    l2_vars[dims["input_arg_idx"]] = (l2_input_var, input_bytes)
    dma_copy = tir.call_extern(
        "int32",
        "tvm_dsp_dma_copy",
        queue_id,
        l2_input_var,
        call_node.args[dims["input_arg_idx"]],
        tir.const(input_bytes, "int32"),
        bypass,
    )
    stmts.append(tir.Evaluate(dma_copy))

    if cache_weight:
        l2_weight_var = tir.Var("l2_weight", PointerType(PrimType("int8"), "global.l2sram"))
        l2_vars[dims["weight_arg_idx"]] = (l2_weight_var, weight_bytes)
        dma_copy = tir.call_extern(
            "int32",
            "tvm_dsp_dma_copy",
            queue_id,
            l2_weight_var,
            call_node.args[dims["weight_arg_idx"]],
            tir.const(weight_bytes, "int32"),
            bypass,
        )
        stmts.append(tir.Evaluate(dma_copy))

    # Wait for all DMAs to complete
    dma_wait = tir.call_extern("int32", "tvm_dsp_dma_wait", queue_id, zero_inflight)
    stmts.append(tir.Evaluate(dma_wait))

    # Modified MMALIB call with L2 pointers
    new_args = list(call_node.args)
    for arg_idx, (l2_var, _) in l2_vars.items():
        new_args[arg_idx] = l2_var
    new_call = tir.Call(call_node.dtype, call_node.op, new_args)
    stmts.append(tir.Evaluate(new_call))

    # Build body: SeqStmt wrapped in Allocate nodes
    body = tir.SeqStmt(stmts)

    # Wrap with Allocate (innermost last, guard outermost)
    for arg_idx in sorted(l2_vars.keys(), reverse=True):
        l2_var, nbytes = l2_vars[arg_idx]
        body = tir.Allocate(
            l2_var,
            "int8",
            [tir.const(nbytes, "int64")],
            tir.const(1, "bool"),
            body,
        )
    # Guard allocation outermost — bumps L2 pointer past the danger zone
    # so the input buffer doesn't start at the very beginning of L2.
    body = tir.Allocate(
        l2_guard_var,
        "int8",
        [tir.const(guard_bytes, "int64")],
        tir.const(1, "bool"),
        body,
    )

    cached = [f"input={input_bytes // 1024}KB"]
    if cache_weight:
        cached.append(f"weight={weight_bytes // 1024}KB")
    logger.info("InjectMMALIBDMA: %s [%s]", extern_name, ", ".join(cached))

    return func.with_body(body)


def InjectMMALIBDMA(l2_budget=393216):
    """Create a TIR pass that injects blocking DMA for MMALIB PrimFuncs.

    Scans each PrimFunc for mmalib_conv2d_i8/i16 call_extern nodes.
    When found, wraps the call with L2 allocations and DMA copy/wait
    so MMALIB operates from L2 SRAM instead of DDR.

    Parameters
    ----------
    l2_budget : int
        L2 SRAM budget in bytes. Default 393216 (384 KB).

    Returns
    -------
    fpass : tvm.transform.Pass
    """

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="InjectMMALIBDMA")
    def _pass(func, mod, ctx):  # noqa: ARG001
        return _inject_dma(func, l2_budget)

    return _pass
