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

"""Lower global.l2sram allocations to inline L2 bump allocator calls.

After the DMA tiling schedule creates cache_read buffers with
``global.l2sram`` scope and StorageRewrite keeps them separate,
this pass converts those Allocate nodes into call_extern calls
to ``tvm_l2_alloc`` -- an inline bump allocator emitted in the
generated lib0.c by the C static codegen.

    Allocate(buf, float32, [N], scope="global.l2sram", body)
    -->
    LetStmt(buf, call_extern("handle", "tvm_l2_alloc", N*4), body)

No explicit free is needed; ``tvm_l2_reset()`` is prepended to each
PrimFunc that uses L2, giving every kernel the full L2 pool.  The
DSP wrapper also calls ``tvm_l2_reset()`` at inference start as a
safety net.

Must run after StorageRewrite and before LowerTVMBuiltin (which
would otherwise convert them to TVMBackendAllocWorkspace calls).
"""

import tvm
from tvm import tir
from tvm.ir import PointerType


def LowerL2SramAlloc():
    """Create a pass that lowers global.l2sram allocations to tvm_l2_alloc.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass.
    """

    def _get_scope(var):
        """Extract storage scope from a buffer Var's type annotation."""
        ptr_type = var.type_annotation
        if isinstance(ptr_type, PointerType):
            return ptr_type.storage_scope
        return ""

    def _nbytes(extents, dtype):
        """Compute allocation size in bytes as a PrimExpr."""
        elem_bytes = (dtype.bits * dtype.lanes + 7) // 8
        nbytes = tir.const(elem_bytes, "int32")
        for e in extents:
            nbytes = nbytes * e
        return nbytes

    def _preorder(op):
        if isinstance(op, tir.Allocate):
            if _get_scope(op.buffer_var) == "global.l2sram":
                # Recursively process body for nested L2 allocations
                new_body = tir.stmt_functor.ir_transform(
                    op.body, _preorder, None, ["tir.Allocate"]
                )
                total = _nbytes(op.extents, op.dtype)
                alloc_call = tir.call_extern("handle", "tvm_l2_alloc", total)
                return tir.LetStmt(op.buffer_var, alloc_call, new_body)
        return None

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="LowerL2SramAlloc")
    def _pass(func, mod, ctx):
        new_body = tir.stmt_functor.ir_transform(
            func.body, _preorder, None, ["tir.Allocate"]
        )
        if new_body.same_as(func.body):
            return func
        # Prepend tvm_l2_reset() so this kernel gets the full L2 pool.
        # Each kernel's L2 buffers are scratch that don't outlive the call.
        reset_call = tir.Evaluate(tir.call_extern("int32", "tvm_l2_reset"))
        new_body = tir.SeqStmt([reset_call, new_body])
        return func.with_body(new_body)

    return _pass
