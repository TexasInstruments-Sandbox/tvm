/*
 * Pseudo-firmware symbol declarations for DLOAD dynamic linking.
 *
 * This file is compiled into dsp_syms.out, which provides symbol definitions
 * at link time for TVM-generated C7x modules. The real symbol resolution
 * happens at dynamic load time via DLOAD against the actual c7x_compute
 * firmware's export table (dyn_loader.c).
 *
 * The symbol list here MUST match the firmware's dsp_syms_names[] array
 * in c7x_compute/dsp/src/dyn_loader.c.
 */

#include <stdint.h>

/* ========================================================================
 * Standard C library
 * ======================================================================== */
__declspec(dllexport) void memcpy() {}
__declspec(dllexport) void memset() {}
__declspec(dllexport) void memcmp() {}
__declspec(dllexport) void memmove() {}
__declspec(dllexport) void printf() {}
__declspec(dllexport) void puts() {}
__declspec(dllexport) void vprintf() {}
__declspec(dllexport) void snprintf() {}
__declspec(dllexport) void vsnprintf() {}
__declspec(dllexport) void fputs() {}
__declspec(dllexport) void fflush() {}
__declspec(dllexport) void malloc() {}
__declspec(dllexport) void free() {}
__declspec(dllexport) void calloc() {}
__declspec(dllexport) void abort() {}

/* ========================================================================
 * TVM C backend API
 * ======================================================================== */
__declspec(dllexport) void TVMBackendAllocWorkspace() {}
__declspec(dllexport) void TVMBackendFreeWorkspace() {}
__declspec(dllexport) void TVMBackendGetFuncFromEnv() {}
__declspec(dllexport) void TVMBackendGetFuncFromGlobalRegistry() {}
__declspec(dllexport) void TVMBackendRegisterSystemLibSymbol() {}
__declspec(dllexport) void TVMBackendParallelLaunch() {}
__declspec(dllexport) void TVMBackendParallelBarrier() {}

/* ========================================================================
 * TVM VM builtins (direct)
 * ======================================================================== */
__declspec(dllexport) void TVMDSPBuiltinAllocStorage() {}
__declspec(dllexport) void TVMDSPBuiltinAllocTensor() {}
__declspec(dllexport) void TVMDSPBuiltinAllocShapeHeap() {}
__declspec(dllexport) void TVMDSPBuiltinMakeShape() {}
__declspec(dllexport) void TVMDSPBuiltinMatchShape() {}
__declspec(dllexport) void TVMDSPBuiltinCheckTensorInfo() {}
__declspec(dllexport) void TVMDSPBuiltinCheckShapeInfo() {}
__declspec(dllexport) void TVMDSPBuiltinNullValue() {}

/* ========================================================================
 * TVM VM builtins (packed wrappers)
 * ======================================================================== */
__declspec(dllexport) void TVMDSPBuiltinAllocStoragePacked() {}
__declspec(dllexport) void TVMDSPBuiltinAllocTensorPacked() {}
__declspec(dllexport) void TVMDSPBuiltinAllocShapeHeapPacked() {}
__declspec(dllexport) void TVMDSPBuiltinMakeShapePacked() {}
__declspec(dllexport) void TVMDSPBuiltinMatchShapePacked() {}
__declspec(dllexport) void TVMDSPBuiltinCheckTensorInfoPacked() {}
__declspec(dllexport) void TVMDSPBuiltinNullValuePacked() {}
__declspec(dllexport) void TVMDSPBuiltinCopyPacked() {}
__declspec(dllexport) void TVMDSPBuiltinMakeTuplePacked() {}
__declspec(dllexport) void TVMDSPBuiltinReshapePacked() {}

/* ========================================================================
 * TVM VM builtins (direct C++ API variants - no packed marshalling)
 * ======================================================================== */
__declspec(dllexport) void TVMDSPBuiltinReshapeDirect() {}
__declspec(dllexport) void TVMDSPBuiltinMakeTupleDirect() {}

/* ========================================================================
 * TVM register file management
 * ======================================================================== */
__declspec(dllexport) void TVMDSPRegFileInit() {}
__declspec(dllexport) void TVMDSPRegSetAny() {}
__declspec(dllexport) void TVMDSPRegFileCleanup() {}

/* ========================================================================
 * TVM FFI object management
 * ======================================================================== */
__declspec(dllexport) void TVMFFIObjectIncRef() {}
__declspec(dllexport) void TVMFFIObjectDecRef() {}
__declspec(dllexport) void TVMFFIObjectFree() {}
__declspec(dllexport) void TVMFFIAnyMove() {}
__declspec(dllexport) void TVMFFIAnyCopy() {}
__declspec(dllexport) void TVMFFIAnyClear() {}

/* ========================================================================
 * TVM VM storage (direct allocation)
 * ======================================================================== */
__declspec(dllexport) void TVMDSPStorageAlloc() {}
__declspec(dllexport) void TVMDSPStorageAllocNDArray() {}

/* ========================================================================
 * TVM L2 SRAM bump allocator (getter functions)
 * ======================================================================== */
__declspec(dllexport) void tvm_dsp_get_l2_base() {}
__declspec(dllexport) void tvm_dsp_get_l2_size() {}

/* ========================================================================
 * TVM platform
 * ======================================================================== */
__declspec(dllexport) void tvm_dsp_alloc() {}
__declspec(dllexport) void tvm_dsp_free() {}
__declspec(dllexport) void tvm_dsp_cache_writeback() {}
__declspec(dllexport) void tvm_dsp_cache_invalidate() {}
__declspec(dllexport) void tvm_dsp_cache_writeback_invalidate() {}
__declspec(dllexport) void tvm_dsp_log() {}
__declspec(dllexport) void tvm_dsp_get_free_memory() {}

/* ========================================================================
 * TVM constants
 * ======================================================================== */
__declspec(dllexport) void TVMDSPGetConstant() {}
__declspec(dllexport) void TVMDSPGetAllConstants() {}

/* ========================================================================
 * TVM DMA runtime
 * ======================================================================== */
__declspec(dllexport) void tvm_dsp_dma_copy() {}
__declspec(dllexport) void tvm_dsp_dma_wait() {}
__declspec(dllexport) void tvm_int8_residual_add_relu() {}
__declspec(dllexport) void tvm_int16_residual_add_relu() {}
__declspec(dllexport) void tvm_dequantize_vecmatmul() {}
__declspec(dllexport) void tvm_sdpa_decode() {}
__declspec(dllexport) int32_t tidl_int8_gelu() { return 0; }
__declspec(dllexport) int32_t tidl_int8_silu() { return 0; }
__declspec(dllexport) int32_t tidl_int8_hardsigmoid() { return 0; }
__declspec(dllexport) int32_t tidl_int8_hardswish() { return 0; }
__declspec(dllexport) int32_t tidl_int8_global_avg_pool() { return 0; }
__declspec(dllexport) int32_t tidl_int8_avg_pool() { return 0; }
__declspec(dllexport) int32_t tidl_int8_layer_norm() { return 0; }

/* ========================================================================
 * MCU+ SDK cache operations
 * ======================================================================== */
__declspec(dllexport) void CacheP_inv() {}
__declspec(dllexport) void CacheP_wb() {}
__declspec(dllexport) void CacheP_wbInv() {}
__declspec(dllexport) void DebugP_log() {}

/* ========================================================================
 * TIDL support (following neo-tvm dsp_syms.c convention)
 * Firmware provides TIDL algo libs; model module links tidl_api.lib.
 * ======================================================================== */
__declspec(dllexport) void* TIDL_VISION_FXNS;
__declspec(dllexport) void* g_l1_mem_addr;
__declspec(dllexport) void* g_l2_mem_addr;
__declspec(dllexport) void* g_l3_mem_addr;
__declspec(dllexport) uint32_t g_l1_mem_size;
__declspec(dllexport) uint32_t g_l2_mem_size;
__declspec(dllexport) uint32_t g_l3_mem_size;

__declspec(dllexport) void appMemAlloc() {}
__declspec(dllexport) void appMemFree() {}
__declspec(dllexport) void* appUdmaGetObj() { return (void*)0; }
__declspec(dllexport) void* getUDMADrvObjPtr() { return (void*)0; }

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

__declspec(dllexport) void TVM_lockInterrupts() {}
__declspec(dllexport) void TVM_unlockInterrupts() {}
__declspec(dllexport) void TVM_cacheWbInv() {}
__declspec(dllexport) int TVM_cacheWbInvRegion(void *addr, unsigned int size) { return 0; }
__declspec(dllexport) void dsp_trace_msg(const char *msg) {}

/* ========================================================================
 * MMALIB wrappers (int8/int16 matmul and conv2d)
 * ======================================================================== */
__declspec(dllexport) int32_t mmalib_conv2d_i8() { return 0; }
__declspec(dllexport) int32_t mmalib_conv2d_i16() { return 0; }
__declspec(dllexport) int32_t mmalib_matmul_i8() { return 0; }
__declspec(dllexport) int32_t mmalib_matmul_i16() { return 0; }
__declspec(dllexport) int32_t mmalib_depthwise_conv2d_i8() { return 0; }
__declspec(dllexport) int32_t mmalib_depthwise_conv2d_i16() { return 0; }
__declspec(dllexport) int32_t mmalib_matmul_bias_i8() { return 0; }
__declspec(dllexport) int32_t mmalib_matmul_bias_i16() { return 0; }
