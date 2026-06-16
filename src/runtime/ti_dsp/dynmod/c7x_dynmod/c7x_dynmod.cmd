/*
 * Linker script for building TVM-generated C7x code as a DLOAD-compatible
 * relocatable module.
 *
 * Two-stage link approach (same pattern as neo-tvm TIDL deployment):
 *   Stage 1: dsp_syms.c → dsp_syms.out (pseudo-firmware with exported symbols)
 *   Stage 2: lib0.c + dsp_syms.out → lib0.out (this linker script)
 *
 * The resulting lib0.out is a relocatable ELF with dynamic relocations.
 * At runtime, DLOAD resolves imported symbols against the real firmware
 * and patches all relocations (R_C7X_ABS64, R_C7X_MVK32_*, R_C7X_PCR_*, etc.)
 */

--ram_model
--display_error_number
--diag_suppress=10290
--diag_suppress=10291
--priority

/* Link against pseudo-firmware for symbol definitions at link time */
-ldsp_syms.out

--dynamic=lib
--relocatable
--no_entry_point
--warn_sections
-x
-heap 0x0


MEMORY
{
    DDR:    o = 0x80000000 l = 0x19000000
}

SECTIONS
{
    /* Embedded pseudo-firmware symbol table — extracted by DLOAD at load time */
    .dsp_syms_out: type = COPY

    .gnu.offload_funcs > DDR
    .gnu.offload_vars  > DDR

    GROUP
    {
        .rodata:
        /* SE guard: 4 KB padding before weights. The Streaming Engine's
         * prefetch buffer (32x64B = 2KB) may issue speculative reads
         * near buffer boundaries. Without padding, weights start at
         * staging+0x40 — only one cache line from the mapped region
         * edge at 0xC0000000. */
        .se_guard: load = 0x80000000, fill = 0x00 { . += 0x1000; }
        .rodata.weights:    /* Embedded model weights */
        .neardata:
        .bss:
    } > DDR

    /* DMA tiling: L2 SRAM buffers for cache_read tiles.
     * Placed in DDR for DLOAD modules (no direct L2 access at link time).
     * The firmware allocates these from L2 at load time via relocation. */
    .bss:l2mem  > DDR

    .text       > DDR
    .cinit      > DDR
    .const      > DDR
    .data       > DDR
    .switch     > DDR
    .far        > DDR
    .fardata    > DDR
    .plt        > DDR
    .sysmem     > DDR
}

/* ========================================================================
 * Import symbols from C7x firmware (resolved by DLOAD at load time)
 * Must match dsp_syms_names[] in c7x_compute/dsp/src/dyn_loader.c
 * ======================================================================== */

/* Standard C library */
--import=memcpy
--import=memset
--import=memcmp
--import=memmove
--import=printf
--import=puts
--import=vprintf
--import=snprintf
--import=vsnprintf
--import=fputs
--import=fflush
--import=malloc
--import=free
--import=calloc
--import=abort

/* TVM C backend API */
--import=TVMBackendAllocWorkspace
--import=TVMBackendFreeWorkspace
--import=TVMBackendGetFuncFromEnv
--import=TVMBackendGetFuncFromGlobalRegistry
--import=TVMBackendRegisterSystemLibSymbol
--import=TVMBackendParallelLaunch
--import=TVMBackendParallelBarrier

/* TVM VM builtins (direct) */
--import=TVMDSPBuiltinAllocStorage
--import=TVMDSPBuiltinAllocTensor
--import=TVMDSPBuiltinAllocShapeHeap
--import=TVMDSPBuiltinMakeShape
--import=TVMDSPBuiltinMatchShape
--import=TVMDSPBuiltinCheckTensorInfo
--import=TVMDSPBuiltinCheckShapeInfo
--import=TVMDSPBuiltinNullValue

/* TVM VM builtins (packed wrappers) */
--import=TVMDSPBuiltinAllocStoragePacked
--import=TVMDSPBuiltinAllocTensorPacked
--import=TVMDSPBuiltinAllocShapeHeapPacked
--import=TVMDSPBuiltinMakeShapePacked
--import=TVMDSPBuiltinMatchShapePacked
--import=TVMDSPBuiltinCheckTensorInfoPacked
--import=TVMDSPBuiltinNullValuePacked
--import=TVMDSPBuiltinCopyPacked
--import=TVMDSPBuiltinMakeTuplePacked
--import=TVMDSPBuiltinReshapePacked

/* TVM VM builtins (direct C++ API variants) */
--import=TVMDSPBuiltinReshapeDirect
--import=TVMDSPBuiltinMakeTupleDirect

/* TVM register file management */
--import=TVMDSPRegFileInit
--import=TVMDSPRegSetAny
--import=TVMDSPRegFileCleanup

/* TVM FFI object management */
--import=TVMFFIObjectIncRef
--import=TVMFFIObjectDecRef
--import=TVMFFIObjectFree
--import=TVMFFIAnyMove
--import=TVMFFIAnyCopy
--import=TVMFFIAnyClear

/* TVM VM storage (direct allocation) */
--import=TVMDSPStorageAlloc
--import=TVMDSPStorageAllocNDArray

/* TVM L2 SRAM bump allocator (getter functions) */
--import=tvm_dsp_get_l2_base
--import=tvm_dsp_get_l2_size

/* TVM DMA runtime */
--import=tvm_dsp_dma_copy
--import=tvm_dsp_dma_wait
/* TVM kernels */
--import=tvm_int8_residual_add_relu
--import=tvm_int16_residual_add_relu
--import=tvm_dequantize_vecmatmul
--import=tvm_sdpa_decode

/* TVM platform */
--import=tvm_dsp_alloc
--import=tvm_dsp_free
--import=tvm_dsp_cache_writeback
--import=tvm_dsp_cache_invalidate
--import=tvm_dsp_cache_writeback_invalidate
--import=tvm_dsp_log
--import=tvm_dsp_get_free_memory

/* TVM constants */
--import=TVMDSPGetConstant
--import=TVMDSPGetAllConstants

/* MCU+ SDK */
--import=CacheP_inv
--import=CacheP_wb
--import=CacheP_wbInv
--import=DebugP_log

/* TIDL support (firmware provides TIDL algo libs + shared resources) */
--import=TIDL_VISION_FXNS
--import=g_l1_mem_addr
--import=g_l2_mem_addr
--import=g_l3_mem_addr
--import=g_l1_mem_size
--import=g_l2_mem_size
--import=g_l3_mem_size
--import=appMemAlloc
--import=appMemFree
--import=appUdmaGetObj
--import=getUDMADrvObjPtr
--import=DmaUtilsAutoInc3d_configure
--import=DmaUtilsAutoInc3d_convertTrVirtToPhyAddr
--import=DmaUtilsAutoInc3d_deconfigure
--import=DmaUtilsAutoInc3d_deinit
--import=DmaUtilsAutoInc3d_getContextSize
--import=DmaUtilsAutoInc3d_getTrMemReq
--import=DmaUtilsAutoInc3d_init
--import=DmaUtilsAutoInc3d_prepareTr
--import=DmaUtilsAutoInc3d_prepareTrWithPhysicalAddress
--import=DmaUtilsAutoInc3d_trigger
--import=DmaUtilsAutoInc3d_wait
--import=TVM_lockInterrupts
--import=TVM_unlockInterrupts
--import=TVM_cacheWbInv
--import=TVM_cacheWbInvRegion
--import=dsp_trace_msg
--import=memalign

/* MMALIB wrappers */
--import=mmalib_conv2d_i8
--import=mmalib_conv2d_i16
--import=mmalib_matmul_i8
--import=mmalib_matmul_i16
--import=mmalib_depthwise_conv2d_i8
--import=mmalib_depthwise_conv2d_i16
--import=mmalib_matmul_bias_i8
--import=mmalib_matmul_bias_i16

/* ========================================================================
 * Prevent RTS library from pulling in standard C implementations.
 * These symbols are resolved at load time from the firmware, not from
 * the compiler's runtime support library.
 * ======================================================================== */
--symbol_map=printf=__dummy_printf
--symbol_map=puts=__dummy_puts
--symbol_map=vprintf=__dummy_vprintf
--symbol_map=snprintf=__dummy_snprintf
--symbol_map=vsnprintf=__dummy_vsnprintf
--symbol_map=fputs=__dummy_fputs
--symbol_map=fflush=__dummy_fflush
--symbol_map=memcpy=__dummy_memcpy
--symbol_map=memset=__dummy_memset
--symbol_map=memcmp=__dummy_memcmp
--symbol_map=memmove=__dummy_memmove
--symbol_map=malloc=__dummy_malloc
--symbol_map=free=__dummy_free
--symbol_map=calloc=__dummy_calloc
--symbol_map=abort=__dummy_abort

/* Redirect fprintf to firmware's printf (stdout only) */
--symbol_map=fprintf=printf

/* Retain optional symbols that the firmware queries by name at runtime.
 * If the symbol doesn't exist (e.g., profiling not compiled in), the
 * linker silently ignores the --retain directive. */
--retain=TVMPrintLayerProfile
