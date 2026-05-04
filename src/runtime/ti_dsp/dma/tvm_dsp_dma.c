/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm_dsp_dma.c
 * \brief TVM DSP DMA - C7x target implementation using EDMA
 *
 * Async DMA transfers via TI DmaUtilsAutoInc3d (DRU direct TR mode).
 * Uses standalone UDMA driver -- no SysConfig DMA channel allocation
 * needed.
 *
 * Transfer pattern follows TIDL single-channel approach:
 *   dma_copy:  prepareTr -> configure -> trigger  (submit + start)
 *   dma_wait:  wait -> deconfigure                (drain + reset)
 *
 * The deconfigure after each wait resets the DRU channel state,
 * ensuring reliable operation across consecutive transfers.
 */

#include "tvm_dsp_dma.h"
#include "dsp_platform.h"

#include <string.h>

/* Debug trace — DebugP_log writes to remoteproc trace buffer,
 * readable via /sys/kernel/debug/remoteproc/remoteproc0/trace0
 * even when the firmware hangs. */
#ifdef TVM_DMA_DEBUG
#include <kernel/dpl/DebugP.h>
#define DMA_TRACE(...) DebugP_log(__VA_ARGS__)
#else
#define DMA_TRACE(...) ((void)0)
#endif
#include <stdint.h>

#include <dmautils_autoincrement_3d.h>
#include <udma.h>
#include <kernel/dpl/CacheP.h>

/* Maximum DMA channels (queue IDs) supported */
#define MAX_DMA_CHANNELS 2

/* Static buffers for DmaUtils context and TR memory.
 * These must survive pool resets between module load/unload cycles.
 * Sizes are generous upper bounds; the actual sizes are queried at
 * runtime via DmaUtilsAutoInc3d_getContextSize/getTrMemReq. */
#define DMA_CONTEXT_BUF_SIZE   4096
#define DMA_TR_BUF_SIZE         512
static uint8_t g_dma_context_buf[DMA_CONTEXT_BUF_SIZE]
    __attribute__((aligned(128)));
static uint8_t g_dma_tr_buf[MAX_DMA_CHANNELS][DMA_TR_BUF_SIZE]
    __attribute__((aligned(128)));

/* UDMA instance for the C7x local DRU.
 * J722S: C7x_1 DRU = instance 5 (from MCU+ SDK udma_soc.h).
 * The standalone udma.h bundled with dmautils only defines MAIN_0/1,
 * so we provide the C7x-specific ID here. */
#define TVM_DSP_UDMA_DRU_INST_ID  (5U)

/* ------------------------------------------------------------------ */
/* Virtual-to-physical address translation for DRU                    */
/* ------------------------------------------------------------------ */

/*
 * The DRU accesses memory via the system bus using physical addresses.
 * The C7x MMU maps some regions with non-identity translations:
 *   Region 9 (Cached):  vAddr 0xC0000000  -> pAddr 0x900000000 (512 MB)
 *   Region 12 (NC):     vAddr 0x100000000 -> pAddr 0x880000000 (32 MB)
 *   Region 13 (Cached): vAddr 0x102000000 -> pAddr 0x882000000 (224 MB)
 * Other regions (L2 SRAM, DDR_C7x_1, IPC) use identity mapping.
 */
static uint64_t virt_to_phys(const void *vaddr) {
    uint64_t va = (uint64_t)(uintptr_t)vaddr;

    /* Region 9: DDR shared memory / host staging buffer (512 MB) */
    if (va >= 0xC0000000ULL && va < 0xE0000000ULL) {
        return va - 0xC0000000ULL + 0x900000000ULL;
    }
    /* Region 13: DDR cacheable heap (0x102000000 - 0x110000000) */
    if (va >= 0x102000000ULL && va < 0x110000000ULL) {
        return va - 0x102000000ULL + 0x882000000ULL;
    }
    /* Region 12: DDR non-cacheable heap (0x100000000 - 0x102000000) */
    if (va >= 0x100000000ULL && va < 0x102000000ULL) {
        return va - 0x100000000ULL + 0x880000000ULL;
    }
    /* All other regions: identity mapping */
    return va;
}

/* ------------------------------------------------------------------ */
/* Module state                                                       */
/* ------------------------------------------------------------------ */
static struct Udma_DrvObj g_udma_drv;
static uint8_t *g_dma_context;
static uint8_t *g_tr_mem[MAX_DMA_CHANNELS];
static int g_num_channels;
static int g_initialized;

/* Per-channel state */
static int g_pending[MAX_DMA_CHANNELS];
static void *g_dst[MAX_DMA_CHANNELS];
static uint32_t g_dst_size[MAX_DMA_CHANNELS];

/* UDMA virt-to-phys callback for convertTrVirtToPhyAddr.
 * This writes 64-bit physical addresses into the TR struct,
 * avoiding the 32-bit pointer truncation in ioPointers. */
static uint64_t udma_virt_to_phys_cb(const void *virtAddr,
                                     uint32_t chNum, void *appData) {
    (void)chNum;
    (void)appData;
    return virt_to_phys(virtAddr);
}

/* ------------------------------------------------------------------ */
/* Init / Deinit                                                      */
/* ------------------------------------------------------------------ */

int tvm_dsp_dma_init(int num_channels) {
    int32_t ret;
    int ch;
    Udma_InitPrms initPrms;
    DmaUtilsAutoInc3d_InitParam dmaInitParams;
    DmaUtilsAutoInc3d_ChannelInitParam chParams[MAX_DMA_CHANNELS];

    if (num_channels <= 0 || num_channels > MAX_DMA_CHANNELS) {
        return -1;
    }
    g_num_channels = num_channels;

    /* NOTE: CLEC event routing for DRU (events 128-143 -> C7x 32-47)
     * must be programmed by the firmware BEFORE calling tvm_dsp_dma_init().
     * The firmware does this in compute_service.c before the first DMA use. */

    /* Init standalone UDMA driver (DRU direct TR mode).
     * J722S C7x uses local DRU instances (ID 5/6), not MAIN_0 (ID 0).
     * The standalone udma.h doesn't define C7x-specific IDs, so we
     * provide them here (values from SDK udma_soc.h). */
    UdmaInitPrms_init(TVM_DSP_UDMA_DRU_INST_ID, &initPrms);
    /* Register our virt-to-phys callback for convertTrVirtToPhyAddr.
     * The default (identity) doesn't handle the DDR heap MMU mapping. */
    initPrms.virtToPhyFxn = &udma_virt_to_phys_cb;
    ret = Udma_init(&g_udma_drv, &initPrms);
    if (ret != 0) {
        return ret;
    }

    /* Use static buffers for DmaUtils context and TR memory so they
     * survive tvm_dsp_reset_pools() between module load/unload cycles. */
    {
        int32_t ctx_size = DmaUtilsAutoInc3d_getContextSize(num_channels);
        if (ctx_size > (int32_t)sizeof(g_dma_context_buf)) {
            Udma_deinit(&g_udma_drv);
            return -2;
        }
        g_dma_context = g_dma_context_buf;
    }
    for (ch = 0; ch < num_channels; ch++) {
        int32_t tr_size = DmaUtilsAutoInc3d_getTrMemReq(1);
        if (tr_size > (int32_t)sizeof(g_dma_tr_buf[ch])) {
            Udma_deinit(&g_udma_drv);
            return -3;
        }
        g_tr_mem[ch] = g_dma_tr_buf[ch];
    }

    /* Init DmaUtilsAutoInc3d */
    memset(&dmaInitParams, 0, sizeof(dmaInitParams));
    dmaInitParams.contextSize = DmaUtilsAutoInc3d_getContextSize(num_channels);
    dmaInitParams.numChannels = num_channels;
    dmaInitParams.traceLogLevel = 0;
    dmaInitParams.udmaDrvHandle = &g_udma_drv;
    dmaInitParams.DmaUtilsVprintf = NULL;

    for (ch = 0; ch < num_channels; ch++) {
        chParams[ch].dmaQueNo = 0;
        chParams[ch].druOwner = DMAUTILSAUTOINC3D_DRUOWNER_DIRECT_TR;
    }

    ret = DmaUtilsAutoInc3d_init(g_dma_context, &dmaInitParams, chParams);
    if (ret != 0) {
        Udma_deinit(&g_udma_drv);
        return ret;
    }

    memset(g_pending, 0, sizeof(g_pending));
    g_initialized = 1;
    return 0;
}

void tvm_dsp_dma_deinit(void) {
    int ch;
    if (!g_initialized) return;

    for (ch = 0; ch < g_num_channels; ch++) {
        if (g_pending[ch]) {
            DmaUtilsAutoInc3d_wait(g_dma_context, ch);
        }
        DmaUtilsAutoInc3d_deconfigure(g_dma_context, ch,
                                      g_tr_mem[ch], 1);
    }
    DmaUtilsAutoInc3d_deinit(g_dma_context);
    Udma_deinit(&g_udma_drv);
    g_initialized = 0;
}

/* ------------------------------------------------------------------ */
/* Async copy: prepareTr -> configure -> trigger                      */
/* ------------------------------------------------------------------ */

int tvm_dsp_dma_copy(int queue_id, void *dst, const void *src,
                     int size, int bypass_cache) {
    DmaUtilsAutoInc3d_TransferProp xferProp;
    DmaUtilsAutoInc3d_TrPrepareParam trPrep;
    uint16_t block_size;
    uint16_t num_blocks;

    (void)bypass_cache;

    if (!g_initialized) {
        DMA_TRACE("[DMA] ERROR: not initialized\r\n");
        return -1;  /* firmware must call tvm_dsp_dma_init() first */
    }

    DMA_TRACE("[DMA] copy: dst=%p src=%p size=%d q=%d\r\n", dst, src, size, queue_id);
    DMA_TRACE("[DMA]   dst_phys=0x%llx src_phys=0x%llx\r\n",
              virt_to_phys(dst), virt_to_phys(src));

    if (queue_id < 0 || queue_id >= g_num_channels) {
        DMA_TRACE("[DMA] ERROR: bad queue_id %d\r\n", queue_id);
        return -1;
    }

    /* Drain any pending transfer on this channel before reuse. */
    if (g_pending[queue_id]) {
        DMA_TRACE("[DMA]   draining pending on q=%d\r\n", queue_id);
        DmaUtilsAutoInc3d_wait(g_dma_context, queue_id);
        CacheP_inv(g_dst[queue_id], g_dst_size[queue_id], CacheP_TYPE_ALLD);
        DmaUtilsAutoInc3d_deconfigure(g_dma_context, queue_id,
                                      g_tr_mem[queue_id], 1);
        g_pending[queue_id] = 0;
        DMA_TRACE("[DMA]   drained\r\n");
    }

    /* 2D decomposition: split size into block_size * num_blocks.
     * icnt0 is uint16_t (max 65535), so large transfers need splitting.
     * Find largest pow2 block_size <= 0x8000 that evenly divides size. */
    block_size = 0x8000;  /* 32768 */
    while (block_size > 1 && (size % block_size) != 0) {
        block_size >>= 1;
    }
    num_blocks = (uint16_t)(size / block_size);

    /* Fill transfer properties -- contiguous 1D transfer as 2D grid. */
    memset(&xferProp, 0, sizeof(xferProp));
    xferProp.syncType = DMAUTILSAUTOINC3D_SYNC_4D;
    xferProp.dmaDfmt  = DMAUTILSAUTOINC3D_DFMT_NONE;

    xferProp.transferDim.sicnt0 = block_size;
    xferProp.transferDim.sicnt1 = num_blocks;
    xferProp.transferDim.sicnt2 = 1;
    xferProp.transferDim.sicnt3 = 1;
    xferProp.transferDim.sdim1  = (int32_t)block_size;
    xferProp.transferDim.sdim2  = 0;
    xferProp.transferDim.sdim3  = 0;

    xferProp.transferDim.dicnt0 = block_size;
    xferProp.transferDim.dicnt1 = num_blocks;
    xferProp.transferDim.dicnt2 = 1;
    xferProp.transferDim.dicnt3 = 1;
    xferProp.transferDim.ddim1  = (int32_t)block_size;
    xferProp.transferDim.ddim2  = 0;
    xferProp.transferDim.ddim3  = 0;

    xferProp.ioPointers.srcPtr = (uint8_t *)src;
    xferProp.ioPointers.dstPtr = (uint8_t *)dst;

    /* Prepare TR descriptor with virtual addresses. */
    trPrep.trMem     = g_tr_mem[queue_id];
    trPrep.trMemSize = DmaUtilsAutoInc3d_getTrMemReq(1);
    trPrep.numTRs    = 1;
    trPrep.channelId = queue_id;

    DMA_TRACE("[DMA]   prepareTr (blk=%u x %u)\r\n", block_size, num_blocks);
    DmaUtilsAutoInc3d_prepareTr(&trPrep, &xferProp);

    /* Convert virtual addresses in TR to 64-bit physical addresses
     * via the registered udma_virt_to_phys_cb callback. */
    DmaUtilsAutoInc3d_convertTrVirtToPhyAddr(
        g_dma_context, &trPrep,
        DMAUTILSAUTOINC3D_ADDRCONVERTMASK_SRCADDR |
        DMAUTILSAUTOINC3D_ADDRCONVERTMASK_DSTADDR);
    DMA_TRACE("[DMA]   convertAddr done\r\n");

    /* Write-back source data from cache to physical memory so DRU
     * reads current values via the system bus. */
    DMA_TRACE("[DMA]   CacheP_wb src=%p size=%d\r\n", src, size);
    CacheP_wb((void *)src, (uint32_t)size, CacheP_TYPE_ALLD);
    DMA_TRACE("[DMA]   CacheP_wb done\r\n");

    /* Configure channel and trigger the DMA transfer. */
    DMA_TRACE("[DMA]   configure+trigger\r\n");
    DmaUtilsAutoInc3d_configure(g_dma_context, queue_id,
                                g_tr_mem[queue_id], 1);
    DmaUtilsAutoInc3d_trigger(g_dma_context, queue_id);
    DMA_TRACE("[DMA]   triggered\r\n");

    /* Record destination for post-wait cache invalidation. */
    g_pending[queue_id]  = 1;
    g_dst[queue_id]      = dst;
    g_dst_size[queue_id] = (uint32_t)size;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Wait: wait for completion -> deconfigure (reset channel)           */
/* ------------------------------------------------------------------ */

int tvm_dsp_dma_wait(int queue_id, int max_inflight) {
    (void)max_inflight;

    if (!g_initialized || queue_id < 0 || queue_id >= g_num_channels) {
        return -1;
    }

    if (!g_pending[queue_id]) {
        DMA_TRACE("[DMA] wait: nothing pending on q=%d\r\n", queue_id);
        return 0;
    }

    /* Wait for the DRU transfer to complete. */
    DMA_TRACE("[DMA] wait: waiting on q=%d dst=%p size=%u\r\n",
              queue_id, g_dst[queue_id], g_dst_size[queue_id]);
    DmaUtilsAutoInc3d_wait(g_dma_context, queue_id);
    DMA_TRACE("[DMA] wait: completed\r\n");

    /* Invalidate destination cache lines so CPU reads the fresh
     * DRU-written data from DDR instead of stale cached values.
     * Safe because TVMBackendAllocWorkspace aligns to 128 bytes
     * (C7x cache line) and sizes are rounded to 128-byte multiples,
     * so no adjacent allocations share cache lines. */
    CacheP_inv(g_dst[queue_id], g_dst_size[queue_id], CacheP_TYPE_ALLD);

    /* Reset channel state for next transfer. */
    DmaUtilsAutoInc3d_deconfigure(g_dma_context, queue_id,
                                  g_tr_mem[queue_id], 1);
    g_pending[queue_id] = 0;

    return 0;
}

/* ------------------------------------------------------------------ */
/* UDMA handle accessor (for TIDL)                                    */
/* ------------------------------------------------------------------ */

void* tvm_dsp_dma_get_udma_handle(void) {
    if (!g_initialized) {
        return NULL;
    }
    return (void*)&g_udma_drv;
}
