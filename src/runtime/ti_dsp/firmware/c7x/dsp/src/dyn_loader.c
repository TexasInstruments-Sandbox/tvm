/*
 * C7x Compute Service - Dynamic ELF Loader
 *
 * Integrates TI's DLOAD library to load dynamically-linked .out files
 * on the C7x DSP at runtime. Adapted from the TIDL/TVM OpenVX node's
 * dsp_load.c for use in a FreeRTOS-based compute service.
 *
 * Key differences from the original dsp_load.c:
 * - Uses tvm_dsp_alloc/tvm_dsp_free for segment memory (DDR pool at 0x108000000)
 * - Uses CacheP_wbInv instead of appMemCacheWbInv
 * - Exports TVM DSP runtime symbols (not TIDL symbols)
 * - Uses DebugP_log instead of printf for firmware logging
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/CacheP.h>
#include <dlpack/dlpack.h>

/* TVM DSP Runtime - memory allocator for DLOAD segments */
#include "platform/dsp_platform.h"

#include "dyn_loader.h"
#include "dload/DLOAD_API/dload_api.h"
#include "dload/DLOAD/elf32.h"

/*
 * =============================================================================
 * Configuration
 * =============================================================================
 */

#define DYN_LOAD_MEM_ALIGN      128
#define MAX_PTR_SIZE_MAP         1024
#define MAX_LOADED_MODULES       4

/*
 * =============================================================================
 * Memory Tracking
 * =============================================================================
 */

typedef struct {
    void *ptrs[MAX_PTR_SIZE_MAP];
    int   sizes[MAX_PTR_SIZE_MAP];
    int   count;
} AllocTracker_t;

static AllocTracker_t *g_alloc_tracker = NULL;

/*
 * =============================================================================
 * Client Handle (per-module state for DLOAD)
 * =============================================================================
 */

/*
 * Symbol table entry: keeps name and address together to prevent mismatch.
 */
typedef struct {
    const char *name;
    void       *addr;
} DspSymEntry;

#define SYM(fn)           { #fn, (void *)(fn) }
#define SYM_ALIAS(nm, fn) { nm,  (void *)(fn) }

typedef struct {
    DLOAD_HANDLE    dload_handle;
    int             file_handle;
    const char     *file_data;
    uint32_t        file_size;      /* ELF data size for bounds checking */
    uint32_t        assigned_handle; /* Handle returned to caller */
    int             dsp_syms_size;
    const DspSymEntry *dsp_syms;
    AllocTracker_t *alloc_tracker;
    uint32_t        text_size;
    uint32_t        data_size;
    uint32_t        inplace_end;    /* End of in-place rodata in input buf */
    uintptr_t       input_buf_start; /* Start of input buffer (from file_data) */
    uintptr_t       input_buf_end;   /* End of input buffer (file_data + file_size) */
} DynLoaderClient_t;

/*
 * =============================================================================
 * Module Table
 * =============================================================================
 */

static DynLoaderClient_t *g_modules[MAX_LOADED_MODULES];
static uint32_t g_next_handle = 1;
static int g_initialized = 0;

/*
 * =============================================================================
 * Error Handling
 * =============================================================================
 */

static jmp_buf g_jmpbuf;

static void dyn_loader_error_exit(int code)
{
    longjmp(g_jmpbuf, -1);
}

/* Required by DLOAD */
int debugging_on = 0;
int profiling_on = 0;

/*
 * =============================================================================
 * Exported Symbol Table
 *
 * These are the firmware symbols that dynamically loaded modules can
 * reference. DLOAD resolves undefined symbols in the loaded .out
 * against this table via DLIF_load_dependent / DLOAD_update_symbol.
 * =============================================================================
 */

/* TVM DSP Runtime - C backend API */
extern void *TVMBackendAllocWorkspace(int, int, uint64_t, int, int);
extern int   TVMBackendFreeWorkspace(int, int, void *);
extern int   TVMBackendGetFuncFromEnv(void *, const char *, void *);
extern int   TVMBackendGetFuncFromGlobalRegistry(const char *, void *);
extern int   TVMBackendRegisterSystemLibSymbol(const char *, void *);
extern int   TVMBackendParallelLaunch(int (*)(int, void *, void *), void *, int);
extern int   TVMBackendParallelBarrier(int, void *);

/* TVM DSP Runtime - DMA */
extern int   tvm_dsp_dma_copy(int, void *, const void *, int, int);
extern int   tvm_dsp_dma_wait(int, int);
/* TVM DSP Runtime - Kernels */
extern int   c7x_int8_residual_add_relu(const void *, const void *,
                                        const void *, void *, int, int);
extern int   c7x_int16_residual_add_relu(const void *, const void *,
                                         const void *, void *, int, int);
extern int   c7x_dequantize_vecmatmul(const void *, const void *,
                                      const void *, void *,
                                      int32_t, int32_t, int32_t);
/* C7x-native activation kernels (no TIDL library calls) */
extern int32_t c7x_int8_gelu(const void *, void *, int32_t,
                               int32_t, float, int32_t, float);
extern int32_t c7x_int8_silu(const void *, void *, int32_t,
                               int32_t, float, int32_t, float);
extern int32_t c7x_int8_hardsigmoid(const void *, void *, int32_t,
                                     int32_t, float, int32_t, float);
extern int32_t c7x_int8_hardswish(const void *, void *, int32_t,
                                   int32_t, float, int32_t, float);
extern int32_t c7x_int8_channel_scale_multiply(
                                   const void *, const void *, void *,
                                   int32_t, int32_t,
                                   float, int32_t, float, int32_t,
                                   float, int32_t);
/* C7x-native average-pool kernels (no TIDL library calls) */
extern int32_t c7x_int8_global_avg_pool(const void *, void *,
                                        int32_t, int32_t, int32_t, int32_t,
                                        int32_t, float, int32_t, float);
extern int32_t c7x_int8_avg_pool(const void *, void *,
                                 int32_t, int32_t, int32_t, int32_t,
                                 int32_t, int32_t, int32_t, int32_t,
                                 int32_t, int32_t, int32_t, int32_t,
                                 int32_t, float, int32_t, float);
/* C7x-native max-pool and relu — transparent ops, no float conversion at runtime */
extern int32_t c7x_int8_max_pool(const void *, void *,
                                 int32_t, int32_t, int32_t, int32_t,
                                 int32_t, int32_t, int32_t, int32_t,
                                 int32_t, int32_t, int32_t, int32_t);
#ifdef USE_TIDL_RUNTIME
/* TIDL-backed max pool — wraps TIDL_spatialMaxPool_ixX_oxX_init/exec */
extern int32_t c7x_int8_max_pool_tidl(const void *, void *,
                                      int32_t, int32_t, int32_t, int32_t,
                                      int32_t, int32_t, int32_t, int32_t,
                                      int32_t, int32_t, int32_t, int32_t);
#endif
extern int32_t c7x_int8_relu(const void *, void *, int32_t, int32_t);
extern int32_t c7x_int8_clamp(const void *, void *, int32_t, int32_t, int32_t);
extern int32_t c7x_int8_requantize_clamp(const void *, void *, int32_t, float, int32_t, int32_t);
extern int32_t c7x_int8_quantize(const float *, int8_t *, int32_t, float, int32_t);
extern int32_t c7x_int8_quantize_rgb(const void *, void *, int32_t, int32_t,
                                      float, float, float, float, float, float);
#ifdef USE_TI_MMALIB
extern int32_t mmalib_conv2d_i8_sliced(void *, void *, void *, void *, void *, void *,
                                        int32_t, int32_t, int32_t, int32_t,
                                        int32_t, int32_t, int32_t, int32_t,
                                        int32_t, int32_t, int32_t, int32_t, int32_t);
#endif
/* C7x-native concat with per-input rescaling */
extern int32_t c7x_int8_concat_rescale(
                                   const void *, int32_t, float, int32_t,
                                   const void *, int32_t, float, int32_t,
                                   const void *, int32_t, float, int32_t,
                                   const void *, int32_t, float, int32_t,
                                   void *, int32_t, float, int32_t);
/* C7x-native normalization kernels (no TIDL library calls) */
extern int32_t c7x_int8_layer_norm(const void *, const void *,
                                    const void *, void *,
                                    int32_t, int32_t, float,
                                    int32_t, float, int32_t, float);
extern int   c7x_sdpa_decode(const void *, const void *,
                             const void *, const void *, void *,
                             int32_t, int32_t, int32_t, int32_t);

/* TVM DSP Runtime - VM builtins */
extern void *TVMDSPBuiltinAllocStorage(int64_t, int32_t, DLDataType);
extern void *TVMDSPBuiltinAllocTensor(void *, int64_t, const int64_t *, int32_t, DLDataType);
extern void *TVMDSPBuiltinAllocShapeHeap(int64_t);
extern void *TVMDSPBuiltinMakeShape(void *, int32_t, const int32_t *, const int64_t *);
extern int   TVMDSPBuiltinMatchShape(const void *, void *, int32_t, const int32_t *, const int64_t *);
extern int   TVMDSPBuiltinCheckTensorInfo(const void *, int32_t, DLDataType);
extern int   TVMDSPBuiltinCheckShapeInfo(const void *, int32_t);
extern void  TVMDSPBuiltinNullValue(void *);

/* TVM DSP Runtime - VM builtins (packed function wrappers) */
extern int TVMDSPBuiltinAllocStoragePacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinAllocTensorPacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinAllocShapeHeapPacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinMakeShapePacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinMatchShapePacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinCheckTensorInfoPacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinNullValuePacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinCopyPacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinMakeTuplePacked(const void *, int32_t, void *);
extern int TVMDSPBuiltinReshapePacked(const void *, int32_t, void *);

/* TVM DSP Runtime - VM builtins (direct C++ API variants) */
extern void *TVMDSPBuiltinReshapeDirect(void *, const int64_t *, int32_t);
extern void *TVMDSPBuiltinMakeTupleDirect(void *, int32_t);

/* TVM DSP Runtime - VM storage (direct allocation) */
extern void *TVMDSPStorageAlloc(size_t, DLDevice, DLDataType);
extern void *TVMDSPStorageAllocNDArray(void *, int64_t, const int64_t *, int32_t, DLDataType);

/* TVM DSP Runtime - Register file management */
extern void TVMDSPRegFileInit(void *, int32_t);
extern void TVMDSPRegSetAny(int32_t, const void *);
extern int  TVMDSPRegFileCleanup(void);

/* TVM DSP Runtime - FFI object management */
extern void TVMFFIObjectIncRef(void *);
extern void TVMFFIObjectDecRef(void *);
extern int  TVMFFIObjectFree(void *);
extern void TVMFFIAnyMove(void *, void *);
extern void TVMFFIAnyCopy(const void *, void *);
extern void TVMFFIAnyClear(void *);

/* TVM DSP Runtime - L2 SRAM getters for inline bump allocator */
extern uint8_t *tvm_dsp_get_l2_base(void);
extern uint32_t tvm_dsp_get_l2_size(void);

/* TVM DSP Runtime - Platform (provided by platform/dsp_platform.h) */

/* TVM DSP Runtime - Constants C API */
extern void *TVMDSPGetConstant(int);
extern void *TVMDSPGetAllConstants(int *);

/* Shared memory printf (fast path for loaded module's printf calls) */
extern int shm_printf(const char *, ...);

/* MCU+ SDK */
extern void CacheP_inv(void *, uint32_t, uint32_t);
extern void CacheP_wb(void *, uint32_t, uint32_t);
extern void CacheP_wbInv(void *, uint32_t, uint32_t);

/* TIDL support (from tidl_support.c and linked tidl_algo.lib) */
extern void* getUDMADrvObjPtr(void);
extern void* appUdmaGetObj(void);
extern void* appMemAlloc(uint32_t, uint32_t, uint32_t);
extern int32_t appMemFree(uint32_t, void*, uint32_t);
extern void* g_l1_mem_addr;
extern uint32_t g_l1_mem_size;
extern void* g_l2_mem_addr;
extern uint32_t g_l2_mem_size;
extern void* g_l3_mem_addr;
extern uint32_t g_l3_mem_size;
extern int32_t TVM_lockInterrupts(void);
extern void TVM_unlockInterrupts(int32_t);
extern void TVM_cacheWbInv(void);
extern int32_t TVM_cacheWbInvRegion(void *addr, uint32_t size);
extern void dsp_trace_msg(const char *msg);
#ifdef USE_TIDL_RUNTIME
/* TIDL_VISION_FXNS: IALG function table from tidl_algo.lib.
 * Must be exported so DLOAD modules can call into TIDL. */
extern char TIDL_VISION_FXNS[];
#endif

#ifdef USE_TI_MMALIB
/* MMALIB wrappers: compiled into firmware, exported to DLOAD modules. */
#include "mmalib_wrappers.h"
#endif

/*
 * DebugP_log is a macro in MCU+ SDK, so we can't take its address.
 * Provide a callable wrapper for dynamically loaded modules.
 */
static void debugp_log_wrapper(char *format, ...)
{
    char buf[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    _DebugP_logZone(DebugP_LOG_ZONE_ALWAYS_ON, "%s", buf);
}

static const DspSymEntry dsp_syms[] = {
    /* Standard C library */
    SYM(memcpy),
    SYM(memset),
    SYM(memcmp),
    SYM(memmove),
    SYM_ALIAS("printf", shm_printf),
    SYM(puts),
    SYM(vprintf),
    SYM(snprintf),
    SYM(vsnprintf),
    SYM(fputs),
    SYM(fflush),
    SYM(malloc),
    SYM(free),
    SYM(calloc),
    SYM(abort),

    /* TVM C backend API */
    SYM(TVMBackendAllocWorkspace),
    SYM(TVMBackendFreeWorkspace),
    SYM(TVMBackendGetFuncFromEnv),
    SYM(TVMBackendGetFuncFromGlobalRegistry),
    SYM(TVMBackendRegisterSystemLibSymbol),
    SYM(TVMBackendParallelLaunch),
    SYM(TVMBackendParallelBarrier),

    /* TVM VM builtins (direct) */
    SYM(TVMDSPBuiltinAllocStorage),
    SYM(TVMDSPBuiltinAllocTensor),
    SYM(TVMDSPBuiltinAllocShapeHeap),
    SYM(TVMDSPBuiltinMakeShape),
    SYM(TVMDSPBuiltinMatchShape),
    SYM(TVMDSPBuiltinCheckTensorInfo),
    SYM(TVMDSPBuiltinCheckShapeInfo),
    SYM(TVMDSPBuiltinNullValue),

    /* TVM VM builtins (packed wrappers) */
    SYM(TVMDSPBuiltinAllocStoragePacked),
    SYM(TVMDSPBuiltinAllocTensorPacked),
    SYM(TVMDSPBuiltinAllocShapeHeapPacked),
    SYM(TVMDSPBuiltinMakeShapePacked),
    SYM(TVMDSPBuiltinMatchShapePacked),
    SYM(TVMDSPBuiltinCheckTensorInfoPacked),
    SYM(TVMDSPBuiltinNullValuePacked),
    SYM(TVMDSPBuiltinCopyPacked),
    SYM(TVMDSPBuiltinMakeTuplePacked),
    SYM(TVMDSPBuiltinReshapePacked),

    /* TVM VM builtins (direct C++ API variants) */
    SYM(TVMDSPBuiltinReshapeDirect),
    SYM(TVMDSPBuiltinMakeTupleDirect),

    /* TVM VM storage (direct allocation) */
    SYM(TVMDSPStorageAlloc),
    SYM(TVMDSPStorageAllocNDArray),

    /* TVM register file management */
    SYM(TVMDSPRegFileInit),
    SYM(TVMDSPRegSetAny),
    SYM(TVMDSPRegFileCleanup),

    /* TVM FFI object management */
    SYM(TVMFFIObjectIncRef),
    SYM(TVMFFIObjectDecRef),
    SYM(TVMFFIObjectFree),
    SYM(TVMFFIAnyMove),
    SYM(TVMFFIAnyCopy),
    SYM(TVMFFIAnyClear),

    /* TVM DMA runtime */
    SYM(tvm_dsp_dma_copy),
    SYM(tvm_dsp_dma_wait),
    /* TVM kernels */
    SYM(c7x_int8_residual_add_relu),
    SYM(c7x_int16_residual_add_relu),
    SYM(c7x_dequantize_vecmatmul),
    SYM(c7x_sdpa_decode),
    /* C7x-native activation kernels (no TIDL library calls) */
    SYM(c7x_int8_gelu),
    SYM(c7x_int8_silu),
    SYM(c7x_int8_hardsigmoid),
    SYM(c7x_int8_hardswish),
    SYM(c7x_int8_channel_scale_multiply),
    /* C7x-native average-pool kernels (no TIDL library calls) */
    SYM(c7x_int8_global_avg_pool),
    SYM(c7x_int8_avg_pool),
    /* C7x-native max-pool and relu — transparent ops, no float conversion at runtime */
    SYM(c7x_int8_max_pool),
#ifdef USE_TIDL_RUNTIME
    SYM(c7x_int8_max_pool_tidl),
#endif
    SYM(c7x_int8_relu),
    SYM(c7x_int8_clamp),
    SYM(c7x_int8_requantize_clamp),
    SYM(c7x_int8_quantize),
    SYM(c7x_int8_quantize_rgb),
    /* C7x-native normalization kernels (no TIDL library calls) */
    SYM(c7x_int8_layer_norm),
    SYM(c7x_int8_concat_rescale),

    /* TVM L2 SRAM bump allocator (getter functions) */
    SYM(tvm_dsp_get_l2_base),
    SYM(tvm_dsp_get_l2_size),

    /* TVM platform */
    SYM(tvm_dsp_alloc),
    SYM(tvm_dsp_free),
    SYM(tvm_dsp_cache_writeback),
    SYM(tvm_dsp_cache_invalidate),
    SYM(tvm_dsp_cache_writeback_invalidate),
    SYM(tvm_dsp_log),
    SYM(tvm_dsp_get_free_memory),

    /* TVM constants */
    SYM(TVMDSPGetConstant),
    SYM(TVMDSPGetAllConstants),

    /* MCU+ SDK cache operations */
    SYM(CacheP_inv),
    SYM(CacheP_wb),
    SYM(CacheP_wbInv),
    SYM_ALIAS("DebugP_log", debugp_log_wrapper),

    /* TIDL support (firmware provides TIDL algo libs + shared resources).
     * DLOAD modules link tidl_api.lib (thin wrapper) and resolve these
     * symbols from the firmware at load time.
     * Note: SYM_ALIAS with & for variables — SYM() reads the value
     * which isn't a compile-time constant for the TI compiler. */
    SYM(getUDMADrvObjPtr),
    SYM(appUdmaGetObj),
    SYM(appMemAlloc),
    SYM(appMemFree),
    SYM_ALIAS("g_l1_mem_addr", &g_l1_mem_addr),
    SYM_ALIAS("g_l1_mem_size", &g_l1_mem_size),
    SYM_ALIAS("g_l2_mem_addr", &g_l2_mem_addr),
    SYM_ALIAS("g_l2_mem_size", &g_l2_mem_size),
    SYM_ALIAS("g_l3_mem_addr", &g_l3_mem_addr),
    SYM_ALIAS("g_l3_mem_size", &g_l3_mem_size),
    SYM(TVM_lockInterrupts),
    SYM(TVM_unlockInterrupts),
    SYM(TVM_cacheWbInv),
    SYM(TVM_cacheWbInvRegion),
    SYM(dsp_trace_msg),
#ifdef USE_TIDL_RUNTIME
    SYM(TIDL_VISION_FXNS),
#endif

#ifdef USE_TI_MMALIB
    /* MMALIB wrappers (int8/int16 matmul and conv2d) */
    SYM(mmalib_conv2d_i8),
    SYM(mmalib_conv2d_i8_sliced),
    SYM(mmalib_conv2d_i8_grouped_loop),
    SYM(mmalib_conv2d_i16),
    SYM(mmalib_matmul_i8),
    SYM(mmalib_matmul_i16),
    SYM(mmalib_depthwise_conv2d_i8),
    SYM(mmalib_depthwise_conv2d_i16),
    SYM(mmalib_matmul_bias_i8),
    SYM(mmalib_matmul_bias_i16),
#endif
};

#define NUM_DSP_SYMS (sizeof(dsp_syms) / sizeof(dsp_syms[0]))


/*
 * =============================================================================
 * Memory Allocation Helpers
 * =============================================================================
 */

static void *tracked_alloc(size_t size)
{
    void *ptr;

    if (g_alloc_tracker == NULL) {
        g_alloc_tracker = (AllocTracker_t *)malloc(sizeof(AllocTracker_t));
        if (g_alloc_tracker == NULL) return NULL;
        memset(g_alloc_tracker, 0, sizeof(AllocTracker_t));
    }

    /* Allocate from TVM DDR pool (128 MB cacheable region at 0x108000000) */
    ptr = tvm_dsp_alloc(size, DYN_LOAD_MEM_ALIGN, TVM_DSP_MEM_MAIN);
    if (ptr == NULL) return NULL;

    if (g_alloc_tracker->count < MAX_PTR_SIZE_MAP) {
        g_alloc_tracker->ptrs[g_alloc_tracker->count] = ptr;
        g_alloc_tracker->sizes[g_alloc_tracker->count] = (int)size;
        g_alloc_tracker->count++;
    }

    return ptr;
}

static void tracked_free(void *ptr)
{
    if (g_alloc_tracker == NULL || ptr == NULL) return;

    int i;
    for (i = 0; i < g_alloc_tracker->count; i++) {
        if (g_alloc_tracker->ptrs[i] == ptr) {
            /* Swap with last */
            g_alloc_tracker->ptrs[i] = g_alloc_tracker->ptrs[g_alloc_tracker->count - 1];
            g_alloc_tracker->sizes[i] = g_alloc_tracker->sizes[g_alloc_tracker->count - 1];
            g_alloc_tracker->count--;
            tvm_dsp_free(ptr);
            return;
        }
    }
    /* Not found in tracker - free anyway */
    tvm_dsp_free(ptr);
}

static void tracked_free_all(AllocTracker_t *tracker)
{
    if (tracker == NULL) return;

    int i;
    for (i = 0; i < tracker->count; i++) {
        if (tracker->ptrs[i] != NULL) {
            tvm_dsp_free(tracker->ptrs[i]);
        }
    }
    /* Tracker itself is allocated with malloc (small bookkeeping struct) */
    free(tracker);
}

/*
 * =============================================================================
 * DLIF Callback Implementations
 * (Required by DLOAD - these provide the platform-specific glue)
 * =============================================================================
 */

void DLIF_exit(int code)
{
    dyn_loader_error_exit(code);
}

void DLIF_warning(LOADER_WARNING_TYPE wtype, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    DebugP_log("[DLOAD] WARNING: %s\r\n", buf);
}

void DLIF_error(LOADER_ERROR_TYPE etype, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    DebugP_log("[DLOAD] ERROR: %s\r\n", buf);
    DLIF_exit(-1);
}

void DLIF_trace(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    DebugP_log("[DLOAD] %s", buf);
}

int DLIF_fseek(LOADER_FILE_DESC *stream, int32_t offset, int origin)
{
    int8_t *new_pos;

    switch (origin) {
    case SEEK_SET:
        new_pos = stream->orig + offset;
        break;
    case SEEK_CUR:
        new_pos = stream->cur + offset;
        break;
    case SEEK_END:
        new_pos = stream->orig + stream->size;
        break;
    default:
        DLIF_exit(-1);
        return -1;
    }

    /* Bounds check: ensure new position is within [orig, orig+size] */
    if (new_pos < stream->orig || new_pos > stream->orig + stream->size) {
        DLIF_error(DLET_FILE, "DLIF_fseek: position out of bounds\n");
        return -1;
    }

    stream->cur = new_pos;
    stream->read_size = (uint32_t)(new_pos - stream->orig);
    return 0;
}

size_t DLIF_fread(void *ptr, size_t size, size_t nmemb,
                  LOADER_FILE_DESC *stream)
{
    size_t total = size * nmemb;
    size_t remaining = (size_t)(stream->orig + stream->size - stream->cur);
    if (total > remaining) {
        DLIF_error(DLET_FILE, "DLIF_fread: read of %u bytes exceeds "
                   "remaining %u bytes in ELF data\n",
                   (unsigned)total, (unsigned)remaining);
        total = remaining;
        nmemb = (size > 0) ? (total / size) : 0;
    }
    memcpy(ptr, stream->cur, total);
    stream->cur += total;
    stream->read_size += total;
    return nmemb;
}

int32_t DLIF_ftell(LOADER_FILE_DESC *stream)
{
    return (int32_t)(stream->cur - stream->orig);
}

int32_t DLIF_fclose(LOADER_FILE_DESC *fd)
{
    return 0;
}

void *DLIF_malloc(size_t size)
{
    void *ptr = tracked_alloc(size);
    if (ptr == NULL) {
        DLIF_error(DLET_MEMORY, "DLIF_malloc() failed for size %d\n", (int)size);
    }
    return ptr;
}

void DLIF_free(void *ptr)
{
    tracked_free(ptr);
}

BOOL DLIF_read(void *client_handle,
               void *ptr, size_t size, size_t nmemb, TARGET_ADDRESS src)
{
    DLIF_error(DLET_MISC, "DLIF_read should not be called\n");
    return FALSE;
}

BOOL DLIF_memcpy(void *client_handle, void *to, void *from, size_t size)
{
    return (memcpy(to, from, size) != NULL) ? TRUE : FALSE;
}

int32_t DLIF_execute(void *client_handle, TARGET_ADDRESS exec_addr)
{
    DLIF_error(DLET_MISC, "DLIF_execute should not be called\n");
    return -1;
}

BOOL DLIF_register_dsbt_index_request(DLOAD_HANDLE handle,
                                      const char *requestor_name,
                                      int32_t requestor_file_handle,
                                      int32_t requested_dsbt_index)
{
    DLIF_error(DLET_MISC, "DLIF_register_dsbt_index_request should not be called\n");
    return FALSE;
}

void DLIF_assign_dsbt_indices(void)
{
    DLIF_error(DLET_MISC, "DLIF_assign_dsbt_indices should not be called\n");
}

int32_t DLIF_get_dsbt_index(int32_t file_handle)
{
    DLIF_error(DLET_MISC, "DLIF_get_dsbt_index should not be called\n");
    return DSBT_INDEX_INVALID;
}

BOOL DLIF_update_all_dsbts(void)
{
    DLIF_error(DLET_MISC, "DLIF_update_all_dsbts should not be called\n");
    return FALSE;
}

/*
 * is_rodata_segment - Check if a segment is read-only data that can
 * be mapped in-place from the input buffer (no DDR pool copy needed).
 *
 * Read-only, non-executable segments contain weights and TIDL artifacts
 * (generated by bin_to_asm.py as pure .byte directives with no symbol
 * references).  No relocations target their content, so leaving them
 * in the input buffer is safe.  DLOAD's relocation pass computes
 * addr_offset = target_address - p_vaddr, which works correctly
 * regardless of whether target_address is in the DDR pool or input buf.
 */
static inline int is_rodata_segment(struct DLOAD_MEMORY_REQUEST *req)
{
    return !(req->flags & DLOAD_SF_executable) &&
           !(req->flags & DLOAD_SF_writable);
}

/*
 * DLIF_allocate - Allocate target memory for a loaded segment.
 *
 * Read-only data segments (weights, TIDL artifacts) are mapped in-place
 * from the shared DDR input buffer to avoid a redundant copy to the
 * DDR pool.  Code and writable data segments are allocated normally.
 */
BOOL DLIF_allocate(void *client_handle, struct DLOAD_MEMORY_REQUEST *req)
{
    struct DLOAD_MEMORY_SEGMENT *obj_desc = req->segment;
    DynLoaderClient_t *client = (DynLoaderClient_t *)client_handle;

    /* Read-only data: map in-place from input buffer */
    if (is_rodata_segment(req)) {
        void *inplace = (void *)(req->fp->orig + req->offset);
        obj_desc->target_address = (TARGET_ADDRESS)inplace;
        client->data_size += obj_desc->memsz_in_bytes;
        uint32_t seg_end = (uint32_t)(req->offset + obj_desc->memsz_in_bytes);
        if (seg_end > client->inplace_end)
            client->inplace_end = seg_end;
        DebugP_log("[DLOAD] In-place rodata: %u bytes at %p "
                   "(ELF offset 0x%x)\r\n",
                   obj_desc->memsz_in_bytes, inplace, req->offset);
        return TRUE;
    }

    /* Normal path: allocate from DDR pool */
    void *addr = DLIF_malloc(obj_desc->memsz_in_bytes);
    if (addr == NULL) return FALSE;

    if (req->flags & DLOAD_SF_executable) {
        client->text_size += obj_desc->memsz_in_bytes;
    } else {
        client->data_size += obj_desc->memsz_in_bytes;
    }

    obj_desc->target_address = (TARGET_ADDRESS)addr;
    return TRUE;
}

BOOL DLIF_release(void *client_handle, struct DLOAD_MEMORY_SEGMENT *ptr)
{
    DynLoaderClient_t *client = (DynLoaderClient_t *)client_handle;
    TARGET_ADDRESS addr = ptr->target_address;
    /* In-place rodata: address is within the input buffer range
     * passed to dyn_loader_load() -- do not free it. */
    if ((uintptr_t)addr >= client->input_buf_start &&
        (uintptr_t)addr < client->input_buf_end)
        return TRUE;
    DLIF_free((void *)addr);
    return TRUE;
}

/*
 * DLIF_copy - Copy segment data from ELF to allocated memory.
 * Since we're running on C7x loading for C7x, target == host.
 *
 * In-place rodata segments skip the copy -- data is already at
 * the target_address (pointing into the input buffer).
 */
BOOL DLIF_copy(void *client_handle, struct DLOAD_MEMORY_REQUEST *req)
{
    struct DLOAD_MEMORY_SEGMENT *obj_desc = req->segment;

    /* In-place rodata: data already at target_address, skip copy */
    if (is_rodata_segment(req)) {
        req->host_address = (void *)obj_desc->target_address;
        return TRUE;
    }

    LOADER_FILE_DESC *f = req->fp;
    void *buf = NULL;
    int result = 1;

    if (obj_desc->objsz_in_bytes) {
        buf = (void *)obj_desc->target_address;
        if (buf == NULL) {
            DLIF_error(DLET_MEMORY, "DLIF_copy allocation failure\n");
            return FALSE;
        }

        DLIF_fseek(f, req->offset, SEEK_SET);
        result = DLIF_fread(buf, obj_desc->objsz_in_bytes, 1, f);
        if (result != 1) {
            DLIF_error(DLET_FILE, "DLIF_fread failed\n");
            return FALSE;
        }
    }

    req->host_address = buf;
    return TRUE;
}

/*
 * DLIF_write - Finalize segment: writeback + invalidate cache
 * so that instruction fetch sees the relocated code.
 *
 * In-place rodata: invalidate only (host already did DMA_BUF_SYNC
 * writeback after staging the ELF).
 */
BOOL DLIF_write(void *client_handle, struct DLOAD_MEMORY_REQUEST *req)
{
    struct DLOAD_MEMORY_SEGMENT *obj_desc = req->segment;

    if (is_rodata_segment(req)) {
        /* Invalidate DSP cache to see host-written data */
        CacheP_inv((void *)obj_desc->target_address,
                   obj_desc->memsz_in_bytes, CacheP_TYPE_ALL);
        return TRUE;
    }

    if (req->host_address) {
        req->host_address = NULL;
    }

    /* Cache WbInv ensures data cache -> DDR -> program cache coherence */
    CacheP_wbInv((void *)obj_desc->target_address,
                 obj_desc->memsz_in_bytes, CacheP_TYPE_ALL);

    return TRUE;
}

/*
 * DLIF_load_dependent - Load dependent .out (dsp_syms.out).
 *
 * The dynamically loaded .out has "dsp_syms.out" as a DT_NEEDED dependency.
 * This embedded section contains dummy symbol addresses from build time.
 * We load those symbols, then update them with real firmware addresses.
 */
int DLIF_load_dependent(void *client_handle, const char *so_name)
{
    int file_handle = 0;

    if (strcmp(so_name, "dsp_syms.out") == 0) {
        DynLoaderClient_t *client = (DynLoaderClient_t *)client_handle;
        DLOAD_HANDLE dload_handle = client->dload_handle;

        /* Find the .dsp_syms_out section in the loaded ELF */
        const char *data = client->file_data;
        struct Elf64_Ehdr *ehdr = (struct Elf64_Ehdr *)data;
        struct Elf64_Shdr *shdr = (struct Elf64_Shdr *)(data + ehdr->e_shoff +
                                      ehdr->e_shstrndx * ehdr->e_shentsize);
        const char *names_start = data + shdr->sh_offset;

        int i;
        for (i = 0; i < ehdr->e_shnum; i++) {
            shdr = (struct Elf64_Shdr *)(data + ehdr->e_shoff +
                                         i * ehdr->e_shentsize);
            if (strncmp(".dsp_syms_out", (names_start + shdr->sh_name), 13) == 0) {
                break;
            }
        }

        if (i >= ehdr->e_shnum) {
            DLIF_error(DLET_FILE, "Could not find .dsp_syms_out section\n");
            return 0;
        }

        /* Load symbols from the embedded dsp_syms.out */
        LOADER_FILE_DESC f;
        f.binary = (int8_t *)(data + shdr->sh_offset);
        f.cur = f.orig = f.binary;
        f.length = f.size = shdr->sh_size;
        f.read_size = 0;

        file_handle = DLOAD_load_symbols(dload_handle, &f);

        /* Update dummy addresses with real firmware symbol addresses */
        for (i = 0; i < client->dsp_syms_size; i++) {
            DLOAD_update_symbol(dload_handle, file_handle,
                                client->dsp_syms[i].name,
                                (TARGET_ADDRESS)client->dsp_syms[i].addr);
        }
    } else {
        DebugP_log("[DLOAD] Unknown dependent: %s\r\n", so_name);
    }

    return file_handle;
}

void DLIF_unload_dependent(void *client_handle, uint32_t file_handle)
{
    DynLoaderClient_t *client = (DynLoaderClient_t *)client_handle;
    DLOAD_HANDLE dload_handle = client->dload_handle;
    DLOAD_unload(dload_handle, file_handle);
}

/*
 * =============================================================================
 * Internal Helpers
 * =============================================================================
 */

static DynLoaderClient_t *find_module_by_handle(uint32_t handle,
                                                 int *slot_out)
{
    int slot;
    for (slot = 0; slot < MAX_LOADED_MODULES; slot++) {
        if (g_modules[slot] != NULL &&
            g_modules[slot]->assigned_handle == handle) {
            if (slot_out) *slot_out = slot;
            return g_modules[slot];
        }
    }
    return NULL;
}

/*
 * =============================================================================
 * Public API
 * =============================================================================
 */

int32_t dyn_loader_init(void)
{
    if (g_initialized) return 0;

    memset(g_modules, 0, sizeof(g_modules));
    g_next_handle = 1;
    g_initialized = 1;

    DebugP_log("[DLOAD] Dynamic loader initialized, %d exported symbols\r\n",
               (int)NUM_DSP_SYMS);
    return 0;
}

int32_t dyn_loader_load(uint64_t elf_addr, uint32_t elf_size,
                        uint32_t *handle_out)
{
    DynLoaderClient_t *client;
    int slot;

    if (!g_initialized) return -1;
    if (handle_out == NULL) return -2;

    /* Find free module slot */
    for (slot = 0; slot < MAX_LOADED_MODULES; slot++) {
        if (g_modules[slot] == NULL) break;
    }
    if (slot >= MAX_LOADED_MODULES) {
        DebugP_log("[DLOAD] No free module slots\r\n");
        return -3;
    }

    /* Invalidate cache on ELF data to ensure we read from DDR */
    CacheP_inv((void *)(uintptr_t)elf_addr, elf_size, CacheP_TYPE_ALL);

    client = (DynLoaderClient_t *)malloc(sizeof(DynLoaderClient_t));
    if (client == NULL) {
        DebugP_log("[DLOAD] Failed to allocate client\r\n");
        return -4;
    }
    memset(client, 0, sizeof(DynLoaderClient_t));

    g_alloc_tracker = NULL;

    if (!setjmp(g_jmpbuf)) {
        client->dload_handle = DLOAD_create(client);
        client->file_data = (const char *)(uintptr_t)elf_addr;
        client->file_size = elf_size;
        client->input_buf_start = (uintptr_t)elf_addr;
        client->input_buf_end = (uintptr_t)elf_addr + elf_size;
        client->dsp_syms_size = (int)NUM_DSP_SYMS;
        client->dsp_syms = dsp_syms;

        LOADER_FILE_DESC f;
        f.binary = (int8_t *)(uintptr_t)elf_addr;
        f.cur = f.orig = f.binary;
        f.length = f.size = elf_size;
        f.read_size = 0;

        client->file_handle = DLOAD_load(client->dload_handle, &f);
    } else {
        /* longjmp from DLIF_error */
        client->file_handle = 0;
    }

    client->alloc_tracker = g_alloc_tracker;
    g_alloc_tracker = NULL;

    if (client->file_handle == 0) {
        DebugP_log("[DLOAD] Load failed\r\n");
        tracked_free_all(client->alloc_tracker);
        free(client);
        return -5;
    }

    uint32_t handle = g_next_handle++;
    client->assigned_handle = handle;
    g_modules[slot] = client;
    *handle_out = handle;

    DebugP_log("[DLOAD] Loaded module handle=%u (text=%u data=%u)\r\n",
               *handle_out, client->text_size, client->data_size);
    return 0;
}

int32_t dyn_loader_query_symbol(uint32_t handle, const char *name,
                                uint64_t *addr_out)
{
    DynLoaderClient_t *client;
    TARGET_ADDRESS sym_val = 0;
    BOOL found;

    if (!g_initialized || handle == 0 || addr_out == NULL) return -1;

    client = find_module_by_handle(handle, NULL);
    if (client == NULL) {
        DebugP_log("[DLOAD] Module not found for handle=%u\r\n", handle);
        return -3;
    }

    found = DLOAD_query_symbol(client->dload_handle,
                               client->file_handle, name, &sym_val);
    if (found) {
        *addr_out = (uint64_t)sym_val;
        return 0;
    }

    DebugP_log("[DLOAD] Symbol not found in module handle=%u: %s\r\n",
               handle, name);
    return -2;
}

int32_t dyn_loader_unload(uint32_t handle)
{
    DynLoaderClient_t *client;
    int slot;

    if (!g_initialized || handle == 0) return -1;

    client = find_module_by_handle(handle, &slot);
    if (client == NULL) {
        DebugP_log("[DLOAD] Module not found for handle=%u\r\n", handle);
        return -2;
    }

    /* Restore alloc tracker for DLOAD_unload's DLIF_free calls */
    g_alloc_tracker = client->alloc_tracker;

    DLOAD_unload(client->dload_handle, client->file_handle);
    DLOAD_destroy(client->dload_handle);

    tracked_free_all(g_alloc_tracker);
    g_alloc_tracker = NULL;

    free(client);
    g_modules[slot] = NULL;

    DebugP_log("[DLOAD] Unloaded module handle=%u\r\n", handle);
    return 0;
}

void dyn_loader_deinit(void)
{
    int slot;

    for (slot = 0; slot < MAX_LOADED_MODULES; slot++) {
        if (g_modules[slot] != NULL) {
            dyn_loader_unload(g_modules[slot]->assigned_handle);
        }
    }

    g_initialized = 0;
    DebugP_log("[DLOAD] Dynamic loader deinitialized\r\n");
}
