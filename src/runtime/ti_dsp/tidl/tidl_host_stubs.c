/*
 * Host-emulation stubs for TIDL firmware-provided symbols.
 *
 * Compiled only when HOST_EMULATION is defined (c7x_host build with
 * USE_TIDL=ON).  This is the host-emulation equivalent of
 * dynmod/c7x_dynmod/tidl_firmware_syms.c (DLOAD), providing the same
 * symbol set without __declspec(dllexport) or TI-specific syntax.
 *
 * Symbols provided:
 *   appMemAlloc / appMemFree   -- malloc/free backed DDR allocator
 *   TVM_cacheWbInvRegion       -- no-op (CPU cache is coherent)
 *   TVM_lockInterrupts         -- no-op (single-threaded host test)
 *   TVM_unlockInterrupts       -- no-op
 *   g_tidl_l{1,2,3}_mem_{addr,size} -- zero (TIDL allocates from DDR)
 *   DmaUtilsAutoInc3d_*        -- no-ops (DMA not used on host)
 */

#ifdef HOST_EMULATION

#include <stdlib.h>
#include <stdint.h>

/* -------------------------------------------------------------------
 * appMem* — TIDL memory allocation API (app_utils layer).
 * ------------------------------------------------------------------- */
void *appMemAlloc(uint32_t heap_id, uint32_t size, uint32_t align) {
    (void)heap_id;
    if (align <= sizeof(void *) || align == 0) {
        return malloc((size_t)size);
    }
    size_t a = (size_t)(align < sizeof(void *) ? sizeof(void *) : align);
    void *p = NULL;
    posix_memalign(&p, a, (size_t)size);
    return p;
}

int32_t appMemFree(uint32_t heap_id, void *ptr, uint32_t size) {
    (void)heap_id;
    (void)size;
    free(ptr);
    return 0;
}

/* -------------------------------------------------------------------
 * Cache writeback/invalidate — no-op on host.
 * ------------------------------------------------------------------- */
int32_t TVM_cacheWbInvRegion(void *addr, uint32_t size) {
    (void)addr;
    (void)size;
    return 0;
}

/* -------------------------------------------------------------------
 * Lock / unlock — no-op (single-threaded host test).
 * ------------------------------------------------------------------- */
void TVM_lockInterrupts(void) {}
void TVM_unlockInterrupts(void) {}

/* -------------------------------------------------------------------
 * TIDL L1/L2/L3 scratch pointers — all zero; TIDL will allocate
 * from DDR (appMemAlloc) instead of pre-allocated SRAM.
 * ------------------------------------------------------------------- */
uint8_t  *g_tidl_l1_mem_addr = 0;
uint32_t  g_tidl_l1_mem_size = 0;
uint8_t  *g_tidl_l2_mem_addr = 0;
uint32_t  g_tidl_l2_mem_size = 0;
uint8_t  *g_tidl_l3_mem_addr = 0;
uint32_t  g_tidl_l3_mem_size = 0;

/* -------------------------------------------------------------------
 * DmaUtils stubs — no-ops; TIDL uses DMA for L2 prefetch on DSP but
 * falls back to memcpy when DMA is unavailable.
 * ------------------------------------------------------------------- */
void DmaUtilsAutoInc3d_configure(void)                          {}
void DmaUtilsAutoInc3d_convertTrVirtToPhyAddr(void)             {}
void DmaUtilsAutoInc3d_deconfigure(void)                        {}
void DmaUtilsAutoInc3d_deinit(void)                             {}
void DmaUtilsAutoInc3d_getContextSize(void)                     {}
void DmaUtilsAutoInc3d_getTrMemReq(void)                        {}
void DmaUtilsAutoInc3d_init(void)                               {}
void DmaUtilsAutoInc3d_prepareTr(void)                          {}
void DmaUtilsAutoInc3d_prepareTrWithPhysicalAddress(void)       {}
void DmaUtilsAutoInc3d_trigger(void)                            {}
void DmaUtilsAutoInc3d_wait(void)                               {}

#endif /* HOST_EMULATION */
