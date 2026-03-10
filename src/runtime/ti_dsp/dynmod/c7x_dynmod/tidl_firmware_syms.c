/*
 * TIDL firmware symbol stubs for DLOAD link resolution.
 *
 * These provide definitions at static link time for symbols that TIDL
 * algo libraries reference.  At runtime, DLOAD patches these with the
 * real firmware addresses.
 */
#include <stdint.h>

/* TIDL support (from firmware tidl_support.c) */
__declspec(dllexport) void* getUDMADrvObjPtr() { return (void*)0; }
__declspec(dllexport) uint8_t* g_tidl_l1_mem_addr = 0;
__declspec(dllexport) uint32_t g_tidl_l1_mem_size = 0;
__declspec(dllexport) uint8_t* g_tidl_l2_mem_addr = 0;
__declspec(dllexport) uint32_t g_tidl_l2_mem_size = 0;
__declspec(dllexport) uint8_t* g_tidl_l3_mem_addr = 0;
__declspec(dllexport) uint32_t g_tidl_l3_mem_size = 0;
__declspec(dllexport) void TVM_lockInterrupts() {}
__declspec(dllexport) void TVM_unlockInterrupts() {}

/* DmaUtils (from firmware dmautils_standalone library) */
__declspec(dllexport) void DmaUtilsAutoInc3d_configure() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_convertTrVirtToPhyAddr() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_deconfigure() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_deinit() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_getContextSize() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_getTrMemReq() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_init() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_prepareTr() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_prepareTrWithPhysicalAddress() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_trigger() {}
__declspec(dllexport) void DmaUtilsAutoInc3d_wait() {}
