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
``global.l2sram`` scope and InjectSoftwarePipeline hoists their
Allocate nodes to function scope, this pass converts those Allocate
nodes into call_extern calls to ``tvm_l2_alloc``.

For a function with a single tiled conv layer:

    Allocate(buf, float32, [N], scope="global.l2sram", body)
    -->
    tvm_l2_reset()
    LetStmt(buf, call_extern("handle", "tvm_l2_alloc", N*4), body)

For functions with multiple sequential tiled layers (e.g. a fused
YOLO backbone), InjectSoftwarePipeline hoists ALL layers' double-buffer
allocations to function scope.  The static total can exceed the 1.25 MB
L2 SRAM budget even though the layers execute sequentially and reuse
the same physical memory.

This pass addresses that by inserting additional ``tvm_l2_reset()``
calls at the boundaries between sequential allocation groups so each
group receives the full L2 budget:

    # Static: sum of all layers' L2 > 1.25 MB, but sequential
    Alloc(v0_l1, scope=l2, Alloc(v1_l1, scope=l2,
        Alloc(v0_l2, scope=l2, Alloc(v1_l2, scope=l2,
            SeqStmt([layer1_code, layer2_code])))))
    -->
    tvm_l2_reset()
    LetStmt(v0_l1, l2_alloc(s0),    # offset 0
     LetStmt(v1_l1, l2_alloc(s1),   # offset s0
      SeqStmt([
        layer1_code,
        tvm_l2_reset(),             # <-- reset: reuse L2 for layer 2
        LetStmt(v0_l2, l2_alloc(s2),# offset 0 again
         LetStmt(v1_l2, l2_alloc(s3),
          layer2_code))])))

Group detection: a group boundary is inserted when:
  (a) the running sum of the current group's allocations would exceed
      the hardware L2 budget, AND
  (b) all vars in the current group have their last use at or before
      some statement index (they are "dead" past that point).

Condition (b) ensures correctness: we never reset L2 while a buffer
from the current group is still live.
"""

import logging

import tvm
from tvm import tir
from tvm.ir import PointerType

logger = logging.getLogger(__name__)

# AM67A (J722S) C7x L2 SRAM hardware limit: 1.25 MB
_L2_SRAM_BYTES = 1280 * 1024


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
            return str(ptr_type.storage_scope)
        return ""

    def _nbytes_expr(extents, dtype):
        """Compute allocation size in bytes as a PrimExpr."""
        elem_bytes = (dtype.bits * dtype.lanes + 7) // 8
        nbytes = tir.const(elem_bytes, "int32")
        for e in extents:
            nbytes = nbytes * e
        return nbytes

    def _nbytes_static(extents, dtype):
        """Compute allocation size in bytes as a Python int, or None if dynamic."""
        elem_bytes = (dtype.bits * dtype.lanes + 7) // 8
        total = elem_bytes
        for e in extents:
            if isinstance(e, tir.IntImm):
                total *= int(e)
            else:
                return None
        return total

    def _vars_in_stmt(stmt):
        """Return the set of Var objects referenced anywhere in stmt."""
        found = set()

        def _visit(node):
            if isinstance(node, tir.Var):
                found.add(node)

        tir.stmt_functor.post_order_visit(stmt, _visit)
        return found

    def _collect_l2_chain(stmt):
        """Walk the outermost nested Allocate(global.l2sram) chain.

        Returns (alloc_list, innermost_body) where alloc_list is a list of
        (buffer_var, extents, dtype) tuples in outer-to-inner order and
        innermost_body is the stmt inside the innermost Allocate.
        """
        alloc_list = []
        while isinstance(stmt, tir.Allocate):
            if _get_scope(stmt.buffer_var) != "global.l2sram":
                break
            alloc_list.append((stmt.buffer_var, stmt.extents, stmt.dtype))
            stmt = stmt.body
        return alloc_list, stmt

    def _build_let_chain(alloc_list, body):
        """Wrap body in nested LetStmts for each (var, extents, dtype).

        Each binding is followed by a null-check mirroring
        lower_tvm_builtin.cc's AllocateNode handling: if tvm_l2_alloc's DDR
        fallback (TVMBackendAllocWorkspace -> tvm_dsp_alloc) also fails,
        throw cleanly instead of dereferencing NULL (a memory exception the
        firmware can't recover from -- see docs/dsp/oom_reporting_design.md
        "tvm_l2_alloc null-check (Gap A)").
        """
        for var, extents, dtype in reversed(alloc_list):
            nbytes = _nbytes_expr(extents, dtype)
            alloc_call = tir.call_extern("handle", "tvm_l2_alloc", nbytes)
            null_check = tir.IfThenElse(
                tir.isnullptr(var), tir.Evaluate(tir.tvm_throw_last_error()), None
            )
            body = tir.LetStmt(var, alloc_call, tir.SeqStmt([null_check, body]))
        return body

    def _lower_func(func):
        """Lower global.l2sram allocations in a single PrimFunc."""
        alloc_list, inner_body = _collect_l2_chain(func.body)

        if not alloc_list:
            return func

        reset_stmt = tir.Evaluate(tir.call_extern("int32", "tvm_l2_reset"))

        # Compute total static size to decide if grouping is needed.
        total_bytes = 0
        all_static = True
        for _, extents, dtype in alloc_list:
            n = _nbytes_static(extents, dtype)
            if n is None:
                all_static = False
                break
            total_bytes += n

        if not all_static or total_bytes <= _L2_SRAM_BYTES:
            # Single group: everything fits in one shot.
            new_body = tir.SeqStmt([reset_stmt, _build_let_chain(alloc_list, inner_body)])
            logger.debug(
                "LowerL2SramAlloc: single group, total=%d bytes",
                total_bytes if all_static else -1,
            )
            return func.with_body(new_body)

        # Multi-group path: static total exceeds L2 budget.
        # Partition alloc_list into sequential groups.  A new group starts when
        # (a) adding the next allocation would push the current group over budget
        # AND (b) all current-group vars are dead after some statement boundary.
        logger.info(
            "LowerL2SramAlloc: %d allocations totalling %d bytes exceed L2 "
            "budget (%d bytes); scanning for sequential group boundaries",
            len(alloc_list),
            total_bytes,
            _L2_SRAM_BYTES,
        )

        # Flatten body to a statement list for scanning.
        if isinstance(inner_body, tir.SeqStmt):
            body_stmts = list(inner_body.seq)
        else:
            body_stmts = [inner_body]

        # Precompute var sets per statement.
        stmt_var_sets = [_vars_in_stmt(s) for s in body_stmts]

        def _last_use_idx(var):
            for i in range(len(body_stmts) - 1, -1, -1):
                if var in stmt_var_sets[i]:
                    return i
            return -1

        # Greedy grouping: accumulate allocations until budget is exceeded,
        # then split at the last-use boundary of the current group.
        groups = []  # list of (alloc_sublist, exclusive_end_stmt_idx)
        cur_group = []
        cur_bytes = 0

        for var, extents, dtype in alloc_list:
            n = _nbytes_static(extents, dtype) or 0
            if cur_bytes + n > _L2_SRAM_BYTES and cur_group:
                max_last = max(_last_use_idx(v) for v, _, _ in cur_group)
                if max_last >= 0:
                    groups.append((list(cur_group), max_last + 1))
                    logger.info(
                        "LowerL2SramAlloc: group boundary after stmt %d "
                        "(group %d bytes, next alloc %d bytes)",
                        max_last,
                        cur_bytes,
                        n,
                    )
                    cur_group = []
                    cur_bytes = 0
                # else: can't split (vars not found) — keep accumulating
            cur_group.append((var, extents, dtype))
            cur_bytes += n

        groups.append((cur_group, len(body_stmts)))

        if len(groups) == 1:
            # No useful split found — fall back to single-group with warning.
            logger.warning(
                "LowerL2SramAlloc: could not split %d-byte allocation into groups "
                "(budget %d bytes). Runtime may overflow L2 bump allocator.",
                total_bytes,
                _L2_SRAM_BYTES,
            )
            new_body = tir.SeqStmt([reset_stmt, _build_let_chain(alloc_list, inner_body)])
            return func.with_body(new_body)

        # Reconstruct body: build from innermost group outward.
        # Each group gets its slice of body_stmts, wrapped in LetStmts for its vars.
        # Between groups, tvm_l2_reset() is inserted.
        stmt_start = 0
        group_data = []
        for group_allocs, split_end in groups:
            group_data.append((group_allocs, body_stmts[stmt_start:split_end]))
            stmt_start = split_end

        # Build combined body from last group to first.
        combined = None
        for group_allocs, slice_stmts in reversed(group_data):
            if combined is not None:
                # Append reset + remainder to this group's slice.
                slice_stmts = list(slice_stmts) + [reset_stmt, combined]
            if len(slice_stmts) == 1:
                body_part = slice_stmts[0]
            elif slice_stmts:
                body_part = tir.SeqStmt(slice_stmts)
            else:
                body_part = tir.Evaluate(0)
            combined = _build_let_chain(group_allocs, body_part)

        new_body = tir.SeqStmt([reset_stmt, combined])

        group_sizes = [
            sum(_nbytes_static(e, d) or 0 for _, e, d in ga) for ga, _ in group_data
        ]
        logger.info(
            "LowerL2SramAlloc: split into %d groups, peak %d bytes "
            "(budget %d bytes); sizes: %s",
            len(groups),
            max(group_sizes),
            _L2_SRAM_BYTES,
            group_sizes,
        )
        return func.with_body(new_body)

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="LowerL2SramAlloc")
    def _pass(func, mod, ctx):  # pylint: disable=unused-argument
        return _lower_func(func)

    return _pass
