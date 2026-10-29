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
AllocBuffer statements to function scope, this pass converts those
AllocBuffer statements into call_extern calls to ``tvm_l2_alloc``.

AllocBuffer is a flat statement (it has no body); the buffer is in scope
for the statements that follow it in the enclosing SeqStmt.  The pass
therefore works on the leading run of ``global.l2sram`` AllocBuffer
statements of the function body.

For a function with a single tiled conv layer:

    AllocBuffer(buf, scope="global.l2sram"); body...
    -->
    tvm_l2_reset()
    DeclBuffer(buf, call_extern("handle", "tvm_l2_alloc", N*4)); body...

For functions with multiple sequential tiled layers (e.g. a fused
YOLO backbone), InjectSoftwarePipeline hoists ALL layers' double-buffer
allocations to function scope.  The static total can exceed the 1.25 MB
L2 SRAM budget even though the layers execute sequentially and reuse
the same physical memory.

This pass addresses that by inserting additional ``tvm_l2_reset()``
calls at the boundaries between sequential allocation groups so each
group receives the full L2 budget:

    # Static: sum of all layers' L2 > 1.25 MB, but sequential
    AllocBuffer(v0_l1, l2); AllocBuffer(v1_l1, l2);
    AllocBuffer(v0_l2, l2); AllocBuffer(v1_l2, l2);
    layer1_code; layer2_code
    -->
    tvm_l2_reset()
    DeclBuffer(v0_l1, l2_alloc(s0))    # offset 0
    DeclBuffer(v1_l1, l2_alloc(s1))    # offset s0
    layer1_code
    tvm_l2_reset()                     # <-- reset: reuse L2 for layer 2
    DeclBuffer(v0_l2, l2_alloc(s2))    # offset 0 again
    DeclBuffer(v1_l2, l2_alloc(s3))
    layer2_code

Group detection: a group boundary is inserted when:
  (a) the running sum of the current group's allocations would exceed
      the hardware L2 budget, AND
  (b) all vars in the current group have their last use at or before
      some statement index (they are "dead" past that point).

Condition (b) ensures correctness: we never reset L2 while a buffer
from the current group is still live.
"""

import logging

import tvm_ffi

import tvm
from tvm import tirx

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

    def _get_scope(buf):
        """Extract the storage scope of a buffer variable."""
        return str(buf.scope())

    def _elem_bytes(dtype):
        """Bytes per element, counting vector lanes."""
        return (dtype.bits * dtype.lanes + 7) // 8

    def _nbytes_expr(extents, dtype):
        """Compute allocation size in bytes as a PrimExpr."""
        nbytes = tirx.const(_elem_bytes(dtype), "int32")
        for e in extents:
            nbytes = nbytes * e
        return nbytes

    def _nbytes_static(extents, dtype):
        """Compute allocation size in bytes as a Python int, or None if dynamic."""
        total = _elem_bytes(dtype)
        for e in extents:
            if isinstance(e, tirx.IntImm):
                total *= int(e)
            else:
                return None
        return total

    def _vars_in_stmt(stmt):
        """Return the set of Var objects referenced anywhere in stmt."""
        found = set()

        def _visit(node):
            found.add(node)

        tvm_ffi.structural_walk(stmt, (tirx.Var, _visit), order="post")
        return found

    def _collect_l2_chain(stmt):
        """Collect the global.l2sram AllocBuffer statements of the function's allocation prefix.

        AllocBuffer has no body: the allocated buffers scope over the statements
        that follow in the enclosing SeqStmt.  The 0.23 pass walked the outermost
        nested Allocate chain, which included every allocation regardless of
        scope, so the L2 buffers were found even when an ordinary allocation
        (e.g. a padded temp) was declared first.  The flat equivalent is the
        leading run of AllocBuffer statements of ANY scope: the global.l2sram
        ones are collected, and the others are kept, in order, ahead of the
        remaining body.

        Returns (alloc_list, remaining_body) where alloc_list is a list of
        (buffer, extents, dtype) tuples in program order and remaining_body is
        the statement (or SeqStmt) with the non-L2 leading allocations kept and
        the L2 ones removed.
        """
        stmts = list(stmt.seq) if isinstance(stmt, tirx.SeqStmt) else [stmt]
        alloc_list = []
        kept = []
        idx = 0
        while idx < len(stmts) and isinstance(stmts[idx], tirx.AllocBuffer):
            buf = stmts[idx].buffer
            if _get_scope(buf) == "global.l2sram":
                alloc_list.append((buf, buf.shape, buf.dtype))
            else:
                kept.append(stmts[idx])
            idx += 1
        rest = kept + stmts[idx:]
        if not rest:
            return alloc_list, tirx.Evaluate(0)
        return alloc_list, rest[0] if len(rest) == 1 else tirx.SeqStmt(rest)

    def _build_decl_chain(alloc_list, body):
        """Bind each (buffer, extents, dtype) to a tvm_l2_alloc result, then run body.

        Returns a flat SeqStmt: for every allocation a DeclBuffer binding the
        buffer to ``tvm_l2_alloc(nbytes)``, followed by a null check, and
        finally ``body``.  The null check mirrors lower_tvm_builtin.cc's
        AllocBuffer handling: if tvm_l2_alloc's DDR fallback
        (TVMBackendAllocWorkspace -> tvm_dsp_alloc) also fails, throw cleanly
        instead of dereferencing NULL (a memory exception the firmware can't
        recover from).
        """
        stmts = []
        for buf, extents, dtype in alloc_list:
            nbytes = _nbytes_expr(extents, dtype)
            alloc_call = tirx.call_extern("handle", "tvm_l2_alloc", nbytes)
            stmts.append(tirx.DeclBuffer(buf, data=alloc_call))
            stmts.append(
                tirx.IfThenElse(
                    tirx.isnullptr(tirx.buffer_data(buf)),
                    tirx.Evaluate(tirx.tvm_throw_last_error()),
                    None,
                )
            )
        stmts.append(body)
        return tirx.SeqStmt(stmts)

    def _lower_func(func):
        """Lower global.l2sram allocations in a single PrimFunc."""
        alloc_list, inner_body = _collect_l2_chain(func.body)

        if not alloc_list:
            return func

        reset_stmt = tirx.Evaluate(tirx.call_extern("int32", "tvm_l2_reset"))

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
            new_body = tirx.SeqStmt([reset_stmt, _build_decl_chain(alloc_list, inner_body)])
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
        if isinstance(inner_body, tirx.SeqStmt):
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
            new_body = tirx.SeqStmt([reset_stmt, _build_decl_chain(alloc_list, inner_body)])
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
                slice_stmts = [*slice_stmts, reset_stmt, combined]
            if len(slice_stmts) == 1:
                body_part = slice_stmts[0]
            elif slice_stmts:
                body_part = tirx.SeqStmt(slice_stmts)
            else:
                body_part = tirx.Evaluate(0)
            combined = _build_decl_chain(group_allocs, body_part)

        new_body = tirx.SeqStmt([reset_stmt, combined])

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

    @tvm.tirx.transform.prim_func_pass(opt_level=0, name="LowerL2SramAlloc")
    def _pass(func, mod, ctx):  # pylint: disable=unused-argument
        return _lower_func(func)

    return _pass
