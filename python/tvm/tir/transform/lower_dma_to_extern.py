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

"""Lower DMA builtins to call_extern for C7x DSP targets.

After LowerAsyncDMA converts async copy loops into builtin::dma_copy()
and builtin::dma_wait() calls, this pass converts them to call_extern()
calls targeting the C7x DMA runtime functions:

  - tir.dma_copy(qid, dst, src, size, bypass) ->
    call_extern("tvm_dsp_dma_copy", qid, dst, src, size, bypass)

  - tir.dma_wait(qid, inflight) ->
    call_extern("tvm_dsp_dma_wait", qid, inflight)

This must run AFTER LowerAsyncDMA and BEFORE LowerTVMBuiltin (which
would otherwise lower these to device API packed calls).

The c_static codegen handles call_extern natively via PrintCallExtern.
"""

import tvm
from tvm import tir
from tvm.ir import Op


def LowerDMAToExtern():
    """Create a pass that lowers DMA builtins to call_extern.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """
    op_dma_copy = Op.get("tir.dma_copy")
    op_dma_wait = Op.get("tir.dma_wait")
    op_dma_start_group = Op.get("tir.dma_start_group")
    op_dma_end_group = Op.get("tir.dma_end_group")

    def _postorder(op):
        if isinstance(op, tir.Evaluate):
            call = op.value
            if isinstance(call, tir.Call):
                if call.op.same_as(op_dma_copy):
                    new_call = tir.call_extern("int32", "tvm_dsp_dma_copy", *call.args)
                    return tir.Evaluate(new_call)
                if call.op.same_as(op_dma_wait):
                    new_call = tir.call_extern("int32", "tvm_dsp_dma_wait", *call.args)
                    return tir.Evaluate(new_call)
                # DMA grouping is a no-op for C7x: individual copies
                # are tracked by queue_id and inflight count.
                if call.op.same_as(op_dma_start_group):
                    return tir.Evaluate(tir.const(0, "int32"))
                if call.op.same_as(op_dma_end_group):
                    return tir.Evaluate(tir.const(0, "int32"))
        return None

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="LowerDMAToExtern")
    def _pass(func, mod, ctx):
        new_body = tir.stmt_functor.ir_transform(func.body, None, _postorder, ["tir.Evaluate"])
        if new_body.same_as(func.body):
            return func
        return func.with_body(new_body)

    return _pass
