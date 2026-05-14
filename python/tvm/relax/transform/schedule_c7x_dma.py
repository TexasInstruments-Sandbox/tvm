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
"""Schedule C7x DMA tiling for fused conv2d PrimFuncs.

This pass applies cache_read with ``global.l2sram`` scope and software
pipeline annotations to fused conv2d PrimFuncs.  It runs after FuseTIR
and produces the TIR that InjectSoftwarePipeline -> LowerAsyncDMA ->
LowerDMAToExtern can lower and execute.

Two tiling strategies are supported:

* **NHWC H-tiling** (preferred): splits the output height (H) loop so
  that each tile's input activation strip (double-buffered) fits in L2.
  Works with fused quantized kernels because per-channel post-conv ops
  are independent of H.  Applied to ``conv2d_nhwc`` blocks.

* **NCHW OC-tiling** (legacy): splits the output-channel (OC) loop so
  that each tile's input + weight working set (double-buffered) fits in
  L2.  Only safe for standalone conv2d (no fused post-conv blocks).
  Applied to ``conv2d_nchw`` blocks.
"""

import logging

import tvm
from tvm import tir
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext

logger = logging.getLogger(__name__)


def _compute_oc_tile(ic, ih_pad, iw_pad, oc, kh, kw, l2_budget, elem_bytes=1):
    """Compute the largest power-of-2 oc_tile that fits double-buffered in L2.

    Per tile iteration (full-spatial oc-tiling):
      input_tile  = IC * IH_pad * IW_pad * elem_bytes   (constant, full input)
      weight_tile = oc_tile * IC * KH * KW * elem_bytes

    Double-buffered: 2 * (input_tile + weight_tile) <= l2_budget
    """
    input_tile = ic * ih_pad * iw_pad * elem_bytes
    # Solve: 2 * (input_tile + oc_tile * ic * kh * kw * elem_bytes) <= l2_budget
    weight_per_oc = ic * kh * kw * elem_bytes
    if weight_per_oc == 0:
        return oc

    # max_oc_tile = (l2_budget / 2 - input_tile) / weight_per_oc
    half_budget = l2_budget // 2
    if half_budget <= input_tile:
        # Input alone exceeds half the budget; skip tiling
        return oc

    max_oc_tile = (half_budget - input_tile) // weight_per_oc
    if max_oc_tile <= 0:
        return oc

    # Largest power-of-2 <= max_oc_tile, capped at OC
    tile = 1
    while tile * 2 <= max_oc_tile and tile * 2 <= oc:
        tile *= 2
    return min(tile, oc)


def _compute_h_tile(ic, iw_pad, kh, oh, l2_budget, elem_bytes=1):
    """Compute the largest h_tile that divides OH for NHWC H-tiling.

    Per tile iteration the input activation strip is:
      input_tile = (h_tile + kh - 1) * iw_pad * ic * elem_bytes

    Double-buffered: 2 * input_tile <= l2_budget

    The tile must evenly divide OH to avoid boundary conditionals
    in the copy loop, which would prevent IdentifyMemCpy from
    recognizing the pattern for DMA lowering.
    """
    strip_per_row = iw_pad * ic * elem_bytes
    if strip_per_row == 0:
        return oh

    half_budget = l2_budget // 2
    # max rows of input that fit: h_tile + kh - 1
    max_input_rows = half_budget // strip_per_row
    max_h_tile = max_input_rows - (kh - 1)
    if max_h_tile <= 0:
        return oh

    # Find the largest factor of OH that fits in L2 budget.
    # Iterate up to sqrt(oh) and collect both small and large factors.
    best = 1
    f = 1
    while f * f <= oh:
        if oh % f == 0:
            if f <= max_h_tile:
                best = max(best, f)
            complement = oh // f
            if complement <= max_h_tile:
                best = max(best, complement)
        f += 1

    if best <= 1:
        return oh  # No useful tiling — skip
    return best


def _schedule_conv2d_nhwc(func, l2_budget):
    """Apply H-tiling DMA schedule to a PrimFunc containing conv2d_nhwc.

    Splits the output height loop so each tile's input activation strip
    (double-buffered) fits in L2 SRAM.  Works with fused quantized
    kernels since per-channel post-conv ops are independent of H.

    Returns the scheduled PrimFunc, or the original if not applicable.
    """
    sch = tir.Schedule(func)

    # --- Detect conv2d_nhwc block ---
    try:
        root_block = sch.get_block("root")
    except Exception:
        return func  # No root block (e.g. TIDL extern stub)
    all_blocks = sch.get_child_blocks(root_block)
    block_names = [sch.get(b).name_hint for b in all_blocks]
    if "conv2d_nhwc" not in block_names:
        return func
    conv_block = sch.get_block("conv2d_nhwc")

    # --- Reorder ff (output channel) innermost ---
    # In HWIO layout, weight[ry, rx, rc, ff] is contiguous on ff.
    # Moving ff innermost makes weight access stride-1 and enables
    # SIMD vectorization of the accumulation loop.
    # Before: [nn, yy, xx, ff, ry, rx, rc]
    # After:  [nn, yy, xx, ry, rx, rc, ff]
    loops = sch.get_loops(conv_block)
    if len(loops) != 7:
        logger.debug(
            "Unexpected loop count %d for conv2d_nhwc reorder, skipping",
            len(loops),
        )
        return func
    sch.reorder(loops[4], loops[5], loops[6], loops[3])

    # --- Extract dimensions from block reads/writes ---
    try:
        block_stmt = sch.get(conv_block)
        # reads[0] = pad_temp: [N, IH_pad, IW_pad, IC]
        # reads[1] = weight (HWIO): [KH, KW, IC, OC]
        # writes[0] = output: [N, OH, OW, OC]
        input_shape = block_stmt.reads[0].buffer.shape
        weight_shape = block_stmt.reads[1].buffer.shape
        output_shape = block_stmt.writes[0].buffer.shape

        ic = int(input_shape[3])
        iw_pad = int(input_shape[2])
        kh = int(weight_shape[0])
        kw = int(weight_shape[1])
        oc = int(weight_shape[3])
        oh = int(output_shape[1])
    except (IndexError, TypeError, ValueError) as e:
        logger.debug("Cannot extract conv2d_nhwc dimensions: %s", e)
        return func

    # --- Compute h_tile ---
    input_dtype = block_stmt.reads[0].buffer.dtype
    elem_bytes = (input_dtype.bits * input_dtype.lanes + 7) // 8
    h_tile = _compute_h_tile(ic, iw_pad, kh, oh, l2_budget, elem_bytes)
    if h_tile >= oh:
        # --- Force DMA H-tiling even when input fits in L2 ---
        #
        # The J722S L2 SRAM is a scratchpad (not a hardware cache).
        # Without explicit DMA, the conv2d compute loops read input
        # data directly from slow DDR.  The DMA H-tiling pipeline
        # (stage 0 = async DMA prefetch of next tile into L2,
        # stage 1 = compute on current tile in L2) ensures the
        # hot data is in fast L2 SRAM before the compute starts.
        #
        # With the current scalar inner loop, the benefit is modest
        # (~0.5%) because the .D unit instruction count — not memory
        # latency — is the bottleneck.  However, once the ff inner
        # loop is vectorized with C7x SIMD (64 int8 elements per
        # vector load), memory bandwidth becomes the limiting factor
        # and L2 prefetch will be critical for performance.
        #
        # Strategy: find h_tile that gives >= 4 tiles (enough for the
        # SW pipeline to reach steady state), falling back to >= 2
        # tiles.  h_tile must evenly divide OH so the copy loop has
        # no boundary conditionals (required for DMA pattern matching).
        # Skip only if OH is prime (no proper factors).
        best = 0
        # Pass 1: prefer >= 4 tiles for good pipeline efficiency
        f = 1
        while f * f <= oh:
            if oh % f == 0:
                if f < oh and oh // f >= 4:
                    best = max(best, f)
                c = oh // f
                if c < oh and oh // c >= 4:
                    best = max(best, c)
            f += 1
        # Pass 2: fall back to >= 2 tiles if 4+ not possible
        if best <= 0:
            f = 1
            while f * f <= oh:
                if oh % f == 0:
                    if f < oh:
                        best = max(best, f)
                    c = oh // f
                    if c < oh:
                        best = max(best, c)
                f += 1
        if best <= 1:
            # OH is prime (e.g. 7) — no proper factors, can't tile.
            # Apply decompose_reduction only (no DMA).
            loops = sch.get_loops(conv_block)
            sch.decompose_reduction(conv_block, loops[3])
            logger.debug(
                "conv2d_nhwc OH=%d is prime, skipping DMA tiling",
                oh,
            )
            return sch.mod["main"]
        h_tile = best

    logger.info(
        "DMA H-tiling conv2d_nhwc: OH=%d, h_tile=%d, L2 budget=%d",
        oh,
        h_tile,
        l2_budget,
    )

    # --- Apply DMA tiling ---
    # Re-fetch loops after reorder: [nn, yy, xx, ry, rx, rc, ff]
    loops = sch.get_loops(conv_block)
    if len(loops) < 4:
        logger.debug("Unexpected loop count %d, skipping", len(loops))
        return sch.mod["main"]

    # Split output height loop: yy -> h_outer, h_inner
    h_outer, h_inner = sch.split(loops[1], factors=[None, h_tile])

    # Cache input into L2 SRAM (per h_outer tile)
    cache_input = sch.cache_read(conv_block, 0, "global.l2sram")
    sch.compute_at(cache_input, h_outer)

    # Fuse cache copy loops for LowerAsyncDMA pattern matching.
    all_cache_loops = sch.get_loops(cache_input)
    copy_loops = all_cache_loops[2:]  # skip nn and h_outer
    if len(copy_loops) >= 2:
        sch.fuse(*copy_loops)

    # Cache weights into L2 SRAM (once per batch, invariant across
    # H-tiles).  Weights in HWIO layout are read repeatedly for each
    # tile — caching them in L2 avoids DDR latency on every access.
    # Only cache if weights + double-buffered input strip fit in L2.
    weight_dtype = block_stmt.reads[1].buffer.dtype
    weight_elem = (weight_dtype.bits * weight_dtype.lanes + 7) // 8
    weight_bytes = kh * kw * ic * oc * weight_elem
    input_strip = (h_tile + kh - 1) * iw_pad * ic * elem_bytes
    # Use 95% budget to account for StorageRewrite alignment/merging overhead
    if weight_bytes + 2 * input_strip <= int(l2_budget * 0.95):
        cache_weight = sch.cache_read(conv_block, 1, "global.l2sram")
        # Compute at nn (outermost) — weights are loaded once, reused
        # by all H-tiles.
        nn_loop = sch.get_loops(conv_block)[0]
        sch.compute_at(cache_weight, nn_loop)
        # Fuse weight copy loops for DMA pattern matching
        w_cache_loops = sch.get_loops(cache_weight)
        w_copy_loops = w_cache_loops[1:]  # skip nn
        if len(w_copy_loops) >= 2:
            sch.fuse(*w_copy_loops)
        logger.info(
            "  Weight L2 cache: %d KB (fits with input strip %d KB)",
            weight_bytes // 1024,
            input_strip // 1024,
        )
    else:
        logger.debug(
            "  Weights too large for L2: %d KB + input %d KB > %d KB",
            weight_bytes // 1024,
            2 * input_strip // 1024,
            l2_budget // 1024,
        )

    # Decompose reduction after DMA tiling is set up.
    # The update block's loops are now:
    #   [nn, h_outer, h_inner, xx, ry, rx, rc, ff]
    # Insert init before ry (the first reduction loop).
    update_loops = sch.get_loops(conv_block)
    for i, lp in enumerate(update_loops):
        loop_var = sch.get(lp)
        # Find the first reduction-axis loop by checking the extent
        # against kh (kernel height).  After H-split the layout is:
        # nn(0), h_outer(1), h_inner(2), xx(3), ry(4), rx(5), rc(6), ff(7)
        if i >= 4:  # ry is at index 4 after split
            sch.decompose_reduction(conv_block, lp)
            break

    # Software pipeline: stage 0 = DMA (async), stage 1 = compute
    sch.annotate(h_outer, "software_pipeline_stage", [0, 1])
    sch.annotate(h_outer, "software_pipeline_order", [0, 1])
    sch.annotate(h_outer, "software_pipeline_async_stages", [0])

    return sch.mod["main"]


def _schedule_conv2d(func, l2_budget):
    """Apply DMA tiling schedule to a PrimFunc containing conv2d_nchw.

    Returns the scheduled PrimFunc, or the original if scheduling is
    not applicable.
    """
    sch = tir.Schedule(func)

    # --- 1a. Detect conv2d block ---
    try:
        root_block = sch.get_block("root")
    except Exception:
        return func  # No root block (e.g. TIDL extern stub)
    all_blocks = sch.get_child_blocks(root_block)
    block_names = [sch.get(b).name_hint for b in all_blocks]
    if "conv2d_nchw" not in block_names:
        return func
    conv_block = sch.get_block("conv2d_nchw")

    # --- 1b. Extract dimensions from block reads/writes ---
    try:
        block_stmt = sch.get(conv_block)
        # block.reads[0] = input (pad_temp): [1, IC, IH_pad, IW_pad]
        # block.reads[1] = weight:           [OC, IC, KH, KW]
        # block.writes[0] = output:          [1, OC, OH, OW]
        input_shape = block_stmt.reads[0].buffer.shape
        weight_shape = block_stmt.reads[1].buffer.shape

        ic = int(input_shape[1])
        ih_pad = int(input_shape[2])
        iw_pad = int(input_shape[3])
        oc = int(weight_shape[0])
        kh = int(weight_shape[2])
        kw = int(weight_shape[3])
    except (IndexError, TypeError, ValueError) as e:
        logger.debug("Cannot extract conv2d dimensions: %s", e)
        return func

    # --- 1b2. Skip fused quantized kernels ---
    # QDQ-fused conv2d has int8 weights and many post-conv blocks
    # (requantize, bias, relu, clip, cast).  OC-tiling breaks the
    # fused post-conv operations that index by output channel.
    # Only tile standalone conv2d (pad_temp + conv2d_nchw only).
    # block_names already computed in step 1a.
    if len(block_names) > 2:
        logger.debug(
            "Skipping DMA tiling for fused kernel with %d blocks: %s",
            len(block_names),
            block_names,
        )
        return func

    # --- 1c. Compute oc_tile ---
    input_dtype = block_stmt.reads[0].buffer.dtype
    elem_bytes = (input_dtype.bits * input_dtype.lanes + 7) // 8
    oc_tile = _compute_oc_tile(ic, ih_pad, iw_pad, oc, kh, kw, l2_budget, elem_bytes)
    if oc_tile >= oc:
        logger.debug(
            "conv2d OC=%d fits in L2 budget (%d bytes), skipping DMA tiling",
            oc,
            l2_budget,
        )
        return func

    logger.info(
        "DMA tiling conv2d: OC=%d, oc_tile=%d, L2 budget=%d",
        oc,
        oc_tile,
        l2_budget,
    )

    # --- 1d. Apply scheduling sequence ---
    loops = sch.get_loops(conv_block)  # [nn, ff, yy, xx, rc, ry, rx]
    if len(loops) < 4:
        logger.debug("Unexpected loop count %d, skipping", len(loops))
        return func

    # Split output-channel loop: ff -> oc_outer, oc_inner
    oc_outer, oc_inner = sch.split(loops[1], factors=[None, oc_tile])

    # Cache reads into L2 SRAM
    cache_input = sch.cache_read(conv_block, 0, "global.l2sram")
    sch.compute_at(cache_input, oc_outer)

    cache_weight = sch.cache_read(conv_block, 1, "global.l2sram")
    sch.compute_at(cache_weight, oc_outer)

    # Fuse cache copy loops for LowerAsyncDMA pattern matching.
    # After compute_at(cache, oc_outer), get_loops returns all loops
    # from outermost to innermost: [nn, oc_outer, copy_ax0, copy_ax1, ...]
    # The cache block's own copy loops start at index 2 (after nn and
    # oc_outer).  We fuse them into a single flat copy loop.
    for cache_blk in [cache_input, cache_weight]:
        all_loops = sch.get_loops(cache_blk)
        copy_loops = all_loops[2:]  # skip nn and oc_outer
        if len(copy_loops) >= 2:
            sch.fuse(*copy_loops)

    # Software pipeline: stage 0 = DMA (async), stage 1 = compute
    sch.annotate(oc_outer, "software_pipeline_stage", [0, 0, 1])
    sch.annotate(oc_outer, "software_pipeline_order", [0, 1, 2])
    sch.annotate(oc_outer, "software_pipeline_async_stages", [0])

    return sch.mod["main"]


def _schedule_dequantize_matmul(func, l2_budget):
    """Apply N-tiling DMA schedule to a PrimFunc containing dequantize_matmul_acc.

    Tiles the N (output channel) loop and prefetches weight tiles from
    DDR into L2 SRAM with double-buffering. Also reorders the inner
    loops to K-outer → N → K-inner for better vectorization potential.

    Returns the scheduled PrimFunc, or the original if not applicable.
    """
    sch = tir.Schedule(func)

    try:
        root_block = sch.get_block("root")
    except Exception:
        return func
    all_blocks = sch.get_child_blocks(root_block)
    block_names = [sch.get(b).name_hint for b in all_blocks]
    if "dequantize_matmul_acc" not in block_names:
        return func
    acc_block = sch.get_block("dequantize_matmul_acc")

    # Extract dimensions from block
    try:
        block_stmt = sch.get(acc_block)
        # reads[0] = activation: [M, K] float32
        # reads[1] = weight: [N, K] int8
        # writes[0] = acc: [M, N] float32
        act_shape = block_stmt.reads[0].buffer.shape
        w_shape = block_stmt.reads[1].buffer.shape
        M = int(act_shape[0])
        K = int(act_shape[1])
        N = int(w_shape[0])
    except (IndexError, TypeError, ValueError) as e:
        logger.debug("Cannot extract dequantize_matmul dimensions: %s", e)
        return func

    # Current loop order: [i0(M), i1(N), k(K)]
    loops = sch.get_loops(acc_block)
    if len(loops) != 3:
        logger.debug(
            "Unexpected loop count %d for dequantize_matmul, skipping",
            len(loops),
        )
        return func

    m_loop, n_loop, k_loop = loops

    # Only cache weight in L2 if the entire matrix fits.
    # SW-pipelined N-tiling for larger weights is deferred until
    # IdentifyMemCpy recognition of the fused copy pattern is fixed.
    weight_bytes = N * K  # int8
    if weight_bytes > int(l2_budget * 0.75):
        logger.debug(
            "dequantize_matmul: weight %d KB > L2 budget, skipping",
            weight_bytes // 1024,
        )
        return func

    n_tile = N  # entire weight fits
    n_tiles = 1
    if n_tiles < 2:
        # Weight fits in L2 — cache it once before the M loop (no SW pipeline)
        cache_weight = sch.cache_read(acc_block, 1, "global.l2sram")
        # Don't compute_at — leave at root so the copy runs once before M loop
        w_cache_loops = sch.get_loops(cache_weight)
        if len(w_cache_loops) >= 2:
            sch.fuse(*w_cache_loops)

        sch.decompose_reduction(acc_block, sch.get_loops(acc_block)[2])

        logger.info(
            "DMA dequantize_matmul: M=%d K=%d N=%d, weight cached in L2 (%d KB)",
            M, K, N, N * K // 1024,
        )
        return sch.mod["main"]

    # Split N loop: n_outer, n_inner
    n_outer, n_inner = sch.split(n_loop, factors=[None, n_tile])

    # Cache weight into L2 SRAM (per n_outer tile)
    cache_weight = sch.cache_read(acc_block, 1, "global.l2sram")
    sch.compute_at(cache_weight, n_outer)

    # Fuse weight cache copy loops for LowerAsyncDMA pattern matching
    w_cache_loops = sch.get_loops(cache_weight)
    w_copy_loops = w_cache_loops[2:]  # skip m_loop and n_outer
    if len(w_copy_loops) >= 2:
        sch.fuse(*w_copy_loops)

    # Decompose reduction: split init from update for the k-loop
    update_loops = sch.get_loops(acc_block)
    for i, lp in enumerate(update_loops):
        loop_var = sch.get(lp)
        if hasattr(loop_var, 'extent') and int(loop_var.extent) == K:
            sch.decompose_reduction(acc_block, lp)
            break

    # Software pipeline: stage 0 = DMA weight tile, stage 1 = compute
    sch.annotate(n_outer, "software_pipeline_stage", [0, 1])
    sch.annotate(n_outer, "software_pipeline_order", [0, 1])
    sch.annotate(n_outer, "software_pipeline_async_stages", [0])

    logger.info(
        "DMA N-tiling dequantize_matmul: M=%d K=%d N=%d, n_tile=%d (%d tiles), "
        "tile=%d KB",
        M, K, N, n_tile, n_tiles, n_tile * K // 1024,
    )
    return sch.mod["main"]


@tvm.transform.module_pass(opt_level=0, name="ScheduleC7xDMATiling")
class ScheduleC7xDMATiling:
    """Apply DMA tiling to conv2d PrimFuncs for C7x L2 SRAM.

    This module pass iterates over all PrimFuncs in the IRModule.  For
    each PrimFunc containing a ``conv2d_nhwc`` or ``conv2d_nchw`` block
    it inserts ``cache_read`` into ``global.l2sram`` scope and annotates
    the outer loop for software pipelining.

    Two strategies are tried in order:

    1. **NHWC H-tiling**: splits the output height loop (for
       ``conv2d_nhwc`` blocks).
    2. **NCHW OC-tiling**: splits the output-channel loop (for
       ``conv2d_nchw`` blocks).

    The resulting TIR can be lowered by InjectSoftwarePipeline ->
    LowerAsyncDMA -> LowerDMAToExtern.

    Parameters
    ----------
    l2_budget : int
        L2 SRAM budget in bytes for double-buffered tiling.
        Default 393216 (384 KB).
    """

    def __init__(self, l2_budget=393216):
        self.l2_budget = l2_budget

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        new_funcs = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, tir.PrimFunc):
                # Try NHWC H-tiling first, then NCHW OC-tiling, then matmul
                new_func = _schedule_conv2d_nhwc(func, self.l2_budget)
                if new_func is func:
                    new_func = _schedule_conv2d(func, self.l2_budget)
                if new_func is func:
                    new_func = _schedule_dequantize_matmul(func, self.l2_budget)
                if new_func is not func:
                    new_funcs[gvar] = new_func

        if not new_funcs:
            return mod

        # Build updated module
        updated = IRModule(mod.functions)
        for gvar, func in new_funcs.items():
            updated[gvar] = func
        # Preserve module attributes
        if mod.attrs:
            updated = updated.with_attrs(mod.attrs)
        return updated
