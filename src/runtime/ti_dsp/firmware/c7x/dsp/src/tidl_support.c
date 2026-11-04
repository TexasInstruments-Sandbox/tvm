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
#if defined(TVM_DSP_DEVICE_AM62DX) || defined(TVM_DSP_DEVICE_J722S)
#include <kernel/dpl/CacheP.h>
#include <kernel/dpl/HwiP.h>
#endif

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
#if defined(TVM_DSP_DEVICE_AM62DX) || defined(TVM_DSP_DEVICE_J722S)
    return (int32_t)HwiP_disable();
#else
    /* On single-core FreeRTOS C7x, TIDL runs in a single task context.
     * No preemption needed — return dummy state. */
    return 0;
#endif
}

void TVM_unlockInterrupts(int32_t prev_state) {
#if defined(TVM_DSP_DEVICE_AM62DX) || defined(TVM_DSP_DEVICE_J722S)
    HwiP_restore((uintptr_t)(uint32_t)prev_state);
#else
    (void)prev_state;
#endif
}

/* --------------------------------------------------------------------------
 * Cache control
 * -------------------------------------------------------------------------- */
void TVM_cacheWbInv(void) {
#if defined(TVM_DSP_DEVICE_AM62DX)
    /* AM62D: CacheP_wbInv(NULL, 0xFFFFFFFF) drives the Streaming Engine
     * through the full 4 GB address range, and the AM62D MMU has a gap at
     * 0x70000000 (between CONFIG_MMU_REGION3 and CLEC) that causes a uTLB
     * fault.  Use the hardware global WB+INV command instead. */
    CacheP_wbInvAll(CacheP_TYPE_L1D);
#else
    /* Writeback+invalidate all data cache.
     * CacheP_wbInv with NULL ptr and 0 size flushes everything
     * on MCU+ SDK. Use large size as fallback. */
    CacheP_wbInv(NULL, 0xFFFFFFFFu, 0 /* CacheP_TYPE_ALL */);
#endif
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

#if defined(TVM_DSP_DEVICE_AM62DX)
    /* AM62D follows the vision_apps AM62A convention (matches
     * c7x-mma-tidl's am62a_config.cfg: L2MEMSIZE_KB=224, MSMCSIZE_KB=1024):
     *
     *  L2 scratch: L2RAM_C7x_1_AUX  @ 0x7F000000, 240 KB
     *  L3 scratch: L2RAM_C7x_1_MAIN @ 0x7E000000, 1 MB (extended L2 as L3)
     *
     * The TVM runtime L2 heap (all of AUX, see the linker command file) and
     * the generated-code L2 scratch alias the AUX bank.  The generated-code
     * scratch lives for one kernel only; the runtime heap holds tensors that
     * can be live across a TIDL call, so compute_service.c disables it while a
     * TIDL module is loaded. */
    g_l2_mem_addr = (void*)0x7F000000;
    g_l2_mem_size = 0x3C000;   /* 240 KB */

    g_l3_mem_addr = (void*)0x7E000000;
    g_l3_mem_size = 0x100000;  /* 1 MB */
#else
    /* J722S follows vision_apps (platform/j722s/rtos/common/app_cfg_c7x_1.h:
     * L2 = AUX, L3 = MAIN) and c7x-mma-tidl's j722s_config.cfg
     * (L2MEMSIZE_KB=224).  With the banks swapped (L2 = MAIN, L3 = AUX) a
     * large YOLO subgraph was wrong on the device even with nothing else in
     * AUX; the cause inside TIDL is not known.
     *
     *  L2 scratch: L2RAM_C7x_1_AUX  @ 0x7F000000, 224 KB
     *  L3 scratch: L2RAM_C7x_1_MAIN @ 0x7E000000, 1.25 MB
     *
     * Both banks are shared with TVM the same way as on AM62D above: MAIN with
     * the per-kernel generated-code scratch, AUX with the runtime heap that
     * compute_service.c disables for TIDL modules. */
    g_l2_mem_addr = (void*)0x7F000000;
    g_l2_mem_size = 224 * 1024;

    g_l3_mem_addr = (void*)0x7E000000;
    g_l3_mem_size = 0x140000;  /* 1.25 MB */
#endif
}
