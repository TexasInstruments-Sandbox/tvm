/*
 * TIDL support functions for C7x firmware.
 *
 * Provides firmware-side resources for TIDL-enabled DLOAD modules:
 *
 * - TIDL_VISION_FXNS: IALG function table (from linked tidl_algo.lib)
 * - appMemAlloc/appMemFree: DDR heap allocation for TIDL IALG memory records
 * - appUdmaGetObj: shared UDMA driver handle
 * - g_l1/l2/l3_mem_addr/size: memory pool addresses for IALG allocator
 * - TVM_lockInterrupts/unlockInterrupts: interrupt control for TIDL
 * - TVM_cacheWbInv: cache writeback+invalidate
 *
 * These are exported via the DLOAD symbol table (dsp_syms[] in dyn_loader.c).
 *
 * Architecture: TIDL algo libraries (tidl_algo.lib, tidl_obj_algo.lib,
 * MMALIB) are linked into the FIRMWARE.  The model DLOAD module only
 * links tidl_api.lib (thin IALG wrapper).  This follows the neo-tvm
 * architecture and avoids symbol resolution issues during cross-compilation.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * UDMA driver handle
 * -------------------------------------------------------------------------- */
extern void* tvm_dsp_dma_get_udma_handle(void);

void* appUdmaGetObj(void) {
    return tvm_dsp_dma_get_udma_handle();
}

/* Keep old name as alias for backward compatibility */
void* getUDMADrvObjPtr(void) {
    return tvm_dsp_dma_get_udma_handle();
}

/* --------------------------------------------------------------------------
 * DDR heap allocation (for TIDL IALG memory records)
 *
 * neo-tvm uses appMemAlloc/appMemFree from PSDK OSAL.  We use the TVM
 * DDR heap (128 MB at 0x108000000) via tvm_dsp_alloc/tvm_dsp_free.
 * The RTS heap (-heap 0x20000) is only 128 KB — far too small for
 * TIDL network copies and memory records (multi-MB).
 * -------------------------------------------------------------------------- */

/* tvm_dsp_alloc: pool=1 is TVM_DSP_MEM_MAIN (DDR heap, 128 MB) */
extern void* tvm_dsp_alloc(size_t size, size_t alignment, int pool);
extern void  tvm_dsp_free(void* ptr);

#define APP_MEM_HEAP_DDR            (0u)
#define APP_MEM_HEAP_DDR_SCRATCH    (4u)

void* appMemAlloc(uint32_t heap_id, uint32_t size, uint32_t align) {
    (void)heap_id;
    if (align < 1) align = 1;
    /* pool=1 is TVM_DSP_MEM_MAIN (DDR heap, 128 MB) */
    void* ptr = tvm_dsp_alloc(size, align, 1);
    if (ptr != NULL) {
        memset(ptr, 0, size);
    }
    return ptr;
}

int32_t appMemFree(uint32_t heap_id, void* ptr, uint32_t size) {
    (void)heap_id;
    (void)size;
    if (ptr != NULL) {
        tvm_dsp_free(ptr);
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * Memory pool globals (for TIDL IALG allocator)
 *
 * Names match neo-tvm convention: g_l1_mem_addr, g_l2_mem_addr, etc.
 * -------------------------------------------------------------------------- */
void*    g_l1_mem_addr = NULL;
uint32_t g_l1_mem_size = 0;
void*    g_l2_mem_addr = NULL;
uint32_t g_l2_mem_size = 0;
void*    g_l3_mem_addr = NULL;
uint32_t g_l3_mem_size = 0;

/* --------------------------------------------------------------------------
 * Interrupt control (called by TIDL createParams for thread safety)
 * -------------------------------------------------------------------------- */

int32_t TVM_lockInterrupts(void) {
    /* On single-core FreeRTOS C7x, TIDL runs in a single task context.
     * No preemption needed — return dummy state. */
    return 0;
}

void TVM_unlockInterrupts(int32_t prev_state) {
    (void)prev_state;
}

/* --------------------------------------------------------------------------
 * Cache control
 * -------------------------------------------------------------------------- */
extern void CacheP_wbInv(void*, uint32_t, uint32_t);

void TVM_cacheWbInv(void) {
    /* Writeback+invalidate all data cache.
     * CacheP_wbInv with NULL ptr and 0 size flushes everything
     * on MCU+ SDK. Use large size as fallback. */
    CacheP_wbInv(NULL, 0xFFFFFFFFu, 0 /* CacheP_TYPE_ALL */);
}

/* Properly-typed cache writeback for TIDL IVISION interface.
 * Signature: int32_t (*)(void *addr, uint32_t size)
 * TIDL calls this to flush specific memory regions after CPU writes
 * that DMA will later read. Required because our DDR heap is cached. */
int32_t TVM_cacheWbInvRegion(void *addr, uint32_t size) {
    CacheP_wbInv(addr, size, 0 /* CacheP_TYPE_ALL */);
    return 0;
}

/* --------------------------------------------------------------------------
 * Trace output (remoteproc trace buffer)
 *
 * DebugP_log writes to the remoteproc trace buffer, readable from A53
 * even if the DSP is hung.  DLOAD modules call dsp_trace_msg() to log.
 * -------------------------------------------------------------------------- */
#include <kernel/dpl/DebugP.h>

void dsp_trace_msg(const char *msg) {
    DebugP_log("[DLOAD] %s\r\n", msg);
}

/* --------------------------------------------------------------------------
 * Initialization
 * -------------------------------------------------------------------------- */
void tidl_support_init(void) {
    /* L1D scratch — 16 KB auxiliary L2-as-L1 region.
     * J722S: L2RAM_C7x_1_AUX_AS_L1 @ 0x7F03C000, 16 KB */
    g_l1_mem_addr = (void*)0x7F03C000;
    g_l1_mem_size = 16 * 1024;

    /* L2 scratch — full 1.25 MB L2 SRAM (shared with TVM bump allocator).
     * J722S: L2RAM_C7x_1_MAIN @ 0x7E000000, 1.25 MB */
    g_l2_mem_addr = (void*)0x7E000000;
    g_l2_mem_size = 0x140000;  /* 1.25 MB */

    /* L3/MSMC — 240 KB auxiliary L2 region.
     * J722S: L2RAM_C7x_1_AUX @ 0x7F000000, 240 KB */
    g_l3_mem_addr = (void*)0x7F000000;
    g_l3_mem_size = 240 * 1024;
}
