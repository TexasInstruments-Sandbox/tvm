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
 * \file c7x_platform.c
 * \brief TVM DSP Runtime - C7x Platform Implementation
 *
 * This file implements the platform abstraction layer for C7x DSP cores.
 * It manages dual memory pools (L2 SRAM and DDR) and provides cache
 * operations and cycle counting for performance measurement.
 */

/* TVM_DSP_TARGET_C7X must be defined by the build system */
#include "../dsp_platform.h"
#include "../dsp_memory.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __TI_COMPILER_VERSION__
#include <c7x.h>
#endif

/*
 * C7x Security Mode (CXM) definitions
 * CXM is bits [2:0] of TSR (Task State Register)
 * Values match TI SDK's Hwi_TSR_CXM enum
 */
#define CXM_GUEST_USER              0
#define CXM_GUEST_SUPERVISOR        1
#define CXM_ROOT_USER               2
#define CXM_ROOT_SUPERVISOR         3
#define CXM_SECURE_USER             4
#define CXM_SECURE_SUPERVISOR       5

#ifdef __TI_COMPILER_VERSION__
/*!
 * \brief Get current execution mode (CXM field from TSR)
 * \return CXM value (0-7)
 *
 * Implemented in c7x_cxm.asm
 */
extern uint32_t tvm_dsp_get_cxm(void);

/*!
 * \brief Get human-readable name for CXM mode
 */
static const char* tvm_dsp_cxm_name(uint32_t cxm) {
  switch (cxm) {
    case CXM_GUEST_USER:        return "GuestUser";
    case CXM_GUEST_SUPERVISOR:  return "GuestSupervisor";
    case CXM_ROOT_USER:         return "RootUser";
    case CXM_ROOT_SUPERVISOR:   return "RootSupervisor";
    case CXM_SECURE_USER:       return "SecureUser";
    case CXM_SECURE_SUPERVISOR: return "SecureSupervisor";
    default:                    return "Reserved";
  }
}
#endif /* __TI_COMPILER_VERSION__ */

/* Global memory pools */
static TVMDSPMemoryPoolDesc g_fast_pool; /* L2 pool */
static TVMDSPMemoryPoolDesc g_main_pool; /* DDR pool */

/* Pool base addresses and sizes - populated from linker symbols or fallback */
static uint8_t* g_l2_heap_base;
static uint8_t* g_ddr_heap_base;
static size_t g_l2_heap_size;
static size_t g_ddr_heap_size;

/*
 * L2 SRAM base and size for the inline bump allocator in
 * TVM-generated code.  Getter functions are exported to DLOAD
 * modules (function calls resolve reliably; data symbols don't).
 * For c7x_host, the linker resolves them directly.
 */
uint8_t* g_tvm_l2_base;
uint32_t g_tvm_l2_size;

uint8_t* tvm_dsp_get_l2_base(void) { return g_tvm_l2_base; }
uint32_t tvm_dsp_get_l2_size(void) { return g_tvm_l2_size; }

#ifndef __TI_COMPILER_VERSION__
/* Fallback storage for non-TI compilers (host emulation) */
static uint8_t g_l2_heap_storage[TVM_DSP_L2_SIZE_FALLBACK];
/* DDR pool is dynamically allocated to avoid large BSS for big models */
static uint8_t* g_ddr_heap_storage = NULL;
#endif

/* Platform initialization state */
static int g_platform_initialized = 0;

int tvm_dsp_platform_init(void) {
  if (g_platform_initialized) {
    return 0; /* Already initialized */
  }

#ifdef __TI_COMPILER_VERSION__
  /*
   * NOTE: MMU initialization is the application's responsibility.
   * The application must configure the MMU before calling any TVM runtime
   * functions. This allows the application to control the memory map via
   * its linker command file, which defines the memory regions.
   *
   * See tests/ti-dsp-runtime/dsp-cpp/j722s/mmu.c for a reference implementation.
   */

  /* Use linker-defined symbols for heap bounds */
  g_l2_heap_base = (uint8_t*)__TVM_DSP_L2_HEAP_START;
  g_l2_heap_size = (size_t)(__TVM_DSP_L2_HEAP_END - __TVM_DSP_L2_HEAP_START);
  g_ddr_heap_base = (uint8_t*)__TVM_DSP_DDR_HEAP_START;
  g_ddr_heap_size = (size_t)(__TVM_DSP_DDR_HEAP_END - __TVM_DSP_DDR_HEAP_START);
#else
  /* Fallback for host emulation */
  g_l2_heap_base = g_l2_heap_storage;
  g_l2_heap_size = sizeof(g_l2_heap_storage);
  if (!g_ddr_heap_storage) {
    g_ddr_heap_storage = (uint8_t*)malloc(TVM_DSP_DDR_SIZE_FALLBACK);
    if (!g_ddr_heap_storage) return -1;
  }
  g_ddr_heap_base = g_ddr_heap_storage;
  g_ddr_heap_size = TVM_DSP_DDR_SIZE_FALLBACK;
#endif

  /* Export L2 base/size for the inline bump allocator in generated code.
   * On real hardware, the bump allocator uses the full L2SRAM MAIN region
   * (1.25 MB at 0x7E000000 on J722S), separate from the runtime pool.
   * For host emulation, allocate a matching buffer from the heap. */
#if defined(__TI_COMPILER_VERSION__) && !defined(C7X_HOST_EMULATION)
  g_tvm_l2_base = (uint8_t*)0x7E000000;
  g_tvm_l2_size = 0x140000;  /* 1.25 MB per J722S TRM */
#else
  {
    static uint8_t* l2_bump_storage = NULL;
    if (!l2_bump_storage) {
      l2_bump_storage = (uint8_t*)malloc(0x140000);
    }
    g_tvm_l2_base = l2_bump_storage;
    g_tvm_l2_size = l2_bump_storage ? 0x140000 : 0;
  }
#endif

  /* Initialize L2 (fast) memory pool */
  int ret = tvm_dsp_memory_pool_init(&g_fast_pool, g_l2_heap_base, g_l2_heap_size);
  if (ret != 0) {
    return ret;
  }

  /* Initialize DDR (main) memory pool */
  ret = tvm_dsp_memory_pool_init(&g_main_pool, g_ddr_heap_base, g_ddr_heap_size);
  if (ret != 0) {
    return ret;
  }

  /* Reset cycle counter */
  tvm_dsp_cycle_counter_reset();

  g_platform_initialized = 1;

#ifdef __TI_COMPILER_VERSION__
  /* Print security mode diagnostic */
  uint32_t cxm = tvm_dsp_get_cxm();
  tvm_dsp_log("C7x security mode: CXM=%d (%s)\n", cxm, tvm_dsp_cxm_name(cxm));
#endif

  tvm_dsp_log("TVM DSP Runtime initialized on %s (%s)\n", TVM_DSP_TARGET_NAME, TVM_DSP_DEVICE_NAME);
  tvm_dsp_log("  L2 pool: 0x%08x - 0x%08x (%u KB)\n",
              (unsigned int)(uintptr_t)g_l2_heap_base,
              (unsigned int)((uintptr_t)g_l2_heap_base + g_l2_heap_size),
              (unsigned int)(g_l2_heap_size / 1024));
  tvm_dsp_log("  DDR pool: 0x%08x - 0x%08x (%u MB)\n",
              (unsigned int)(uintptr_t)g_ddr_heap_base,
              (unsigned int)((uintptr_t)g_ddr_heap_base + g_ddr_heap_size),
              (unsigned int)(g_ddr_heap_size / (1024 * 1024)));

  return 0;
}

void tvm_dsp_platform_shutdown(void) {
  if (!g_platform_initialized) {
    return;
  }

  /* Log final statistics */
  TVMDSPMemoryStats stats;
  tvm_dsp_get_memory_stats(TVM_DSP_MEM_FAST, &stats);
  tvm_dsp_log("L2 pool stats: %u allocs, %u frees, peak %u bytes\n",
              stats.alloc_count, stats.free_count, (unsigned int)stats.peak_used);

  tvm_dsp_get_memory_stats(TVM_DSP_MEM_MAIN, &stats);
  tvm_dsp_log("DDR pool stats: %u allocs, %u frees, peak %u bytes\n",
              stats.alloc_count, stats.free_count, (unsigned int)stats.peak_used);

  /* Reset pools */
  tvm_dsp_memory_pool_reset(&g_fast_pool);
  tvm_dsp_memory_pool_reset(&g_main_pool);

  g_platform_initialized = 0;
}

void* tvm_dsp_alloc(size_t size, size_t alignment, TVMDSPMemoryPool pool) {
  if (!g_platform_initialized) {
    tvm_dsp_log("ERROR: Platform not initialized\n");
    return NULL;
  }

  TVMDSPMemoryPoolDesc* pool_desc;
  switch (pool) {
    case TVM_DSP_MEM_FAST:
      pool_desc = &g_fast_pool;
      break;
    case TVM_DSP_MEM_MAIN:
      pool_desc = &g_main_pool;
      break;
    default:
      tvm_dsp_log("ERROR: Invalid memory pool %d\n", pool);
      return NULL;
  }

  void* result = tvm_dsp_memory_pool_alloc(pool_desc, size, alignment);
  if (result == NULL && size > 0) {
    /* Suppress L2 OOM messages: falling back from L2 to DDR is normal for
     * models with KV caches larger than the 128 KB L2 pool (e.g. 256-token
     * cache).  Printing one message per failed allocation floods the 64 KB
     * shared printf buffer and hangs cg_main_dsp() on subsequent printf calls.
     * DDR OOM is unexpected and still reported. */
    if (pool != TVM_DSP_MEM_FAST) {
      tvm_dsp_log("ERROR: OOM in DDR pool: requested %u bytes, "
                  "free %u / %u bytes\n",
                  (unsigned)size,
                  (unsigned)tvm_dsp_memory_pool_free_space(pool_desc),
                  (unsigned)pool_desc->size);
    }
  }
  return result;
}

void tvm_dsp_free(void* ptr) {
  if (ptr == NULL) {
    return;
  }

  if (!g_platform_initialized) {
    return;
  }

  /* Determine which pool the pointer belongs to */
  if (tvm_dsp_memory_pool_contains(&g_fast_pool, ptr)) {
    tvm_dsp_memory_pool_free(&g_fast_pool, ptr);
  } else if (tvm_dsp_memory_pool_contains(&g_main_pool, ptr)) {
    tvm_dsp_memory_pool_free(&g_main_pool, ptr);
  } else {
    tvm_dsp_log("WARNING: Free of unknown pointer 0x%08x\n", (unsigned int)(uintptr_t)ptr);
  }
}

void tvm_dsp_reset_pools(void) {
  if (!g_platform_initialized) {
    return;
  }
  tvm_dsp_memory_pool_reset(&g_fast_pool);
  tvm_dsp_memory_pool_reset(&g_main_pool);
}

/* Watermarks for reclaiming per-inference pool memory.
 * Saved after DYN_LOAD so DLOAD code and DYN_LOAD constants (below
 * the watermark) are preserved when the pool is restored.
 * Per-inference allocations (above the watermark) are reclaimed. */
static void* g_infer_wm_main = NULL;
static void* g_infer_wm_fast = NULL;

void tvm_dsp_save_infer_watermark(void) {
  if (!g_platform_initialized) return;
  g_infer_wm_main = g_main_pool.bump_ptr;
  g_infer_wm_fast = g_fast_pool.bump_ptr;
}

void tvm_dsp_restore_infer_watermark(void) {
  if (!g_platform_initialized) return;
  /* L2 (fast) pool: DLOAD code is in DDR not L2, so restoring to the
   * saved post-DLOAD watermark (which is essentially pool base) is safe. */
  if (g_infer_wm_fast != NULL) {
    g_fast_pool.bump_ptr  = g_infer_wm_fast;
    g_fast_pool.allocated = 0;
    g_fast_pool.num_allocs = 0;
    g_fast_pool.num_frees  = 0;
    g_fast_pool.free_list  = NULL;
  }
  /* DDR pool: restore to watermark above DLOAD code, not to pool base. */
  if (g_infer_wm_main != NULL) {
    g_main_pool.bump_ptr  = g_infer_wm_main;
    g_main_pool.allocated = 0;
    g_main_pool.num_allocs = 0;
    g_main_pool.num_frees  = 0;
    g_main_pool.free_list  = NULL;
  }
}

size_t tvm_dsp_get_free_memory(TVMDSPMemoryPool pool) {
  if (!g_platform_initialized) {
    return 0;
  }

  switch (pool) {
    case TVM_DSP_MEM_FAST:
      return tvm_dsp_memory_pool_free_space(&g_fast_pool);
    case TVM_DSP_MEM_MAIN:
      return tvm_dsp_memory_pool_free_space(&g_main_pool);
    default:
      return 0;
  }
}

void tvm_dsp_get_memory_stats(TVMDSPMemoryPool pool, TVMDSPMemoryStats* stats) {
  if (stats == NULL) {
    return;
  }

  TVMDSPMemoryPoolDesc* pool_desc = NULL;
  switch (pool) {
    case TVM_DSP_MEM_FAST:
      pool_desc = &g_fast_pool;
      break;
    case TVM_DSP_MEM_MAIN:
      pool_desc = &g_main_pool;
      break;
    default:
      memset(stats, 0, sizeof(*stats));
      return;
  }

  stats->total_size = pool_desc->size;
  stats->used_size = pool_desc->allocated;
  stats->peak_used = pool_desc->peak;
  stats->alloc_count = pool_desc->num_allocs;
  stats->free_count = pool_desc->num_frees;
}

void tvm_dsp_cycle_counter_reset(void) {
#ifdef __TI_COMPILER_VERSION__
  /*
   * On C7x, the timestamp counter (TSC) is a 64-bit free-running counter
   * that starts at power-on. Unlike C66x, we cannot reset it by writing
   * to TSCL. Instead, we just note that the counter is always running.
   *
   * For relative measurements, callers should use tvm_dsp_cycle_counter_get()
   * to capture start/end times and compute the difference.
   */
  /* No reset possible on C7x - counter is always running */
#endif
  /* No-op on host - cycle counter not available */
}

uint64_t tvm_dsp_cycle_counter_get(void) {
#ifdef __TI_COMPILER_VERSION__
  /*
   * On C7x, use the __TSC intrinsic to read the 64-bit timestamp counter.
   * This provides atomic access to the full 64-bit value.
   */
  return __TSC;
#else
  return 0;
#endif
}

void tvm_dsp_cache_writeback(void* addr, size_t size) {
#ifdef __TI_COMPILER_VERSION__
  /*
   * C7x cache writeback
   *
   * On C7x, we can use the __cache_wb() intrinsic or
   * control ECR registers directly. The intrinsic is preferred
   * when available.
   *
   * For block operations, use cache line-aligned addresses.
   */
  if (addr != NULL && size > 0) {
    /* Align to cache line boundary */
    uintptr_t start = (uintptr_t)addr & ~(TVM_DSP_CACHE_LINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + TVM_DSP_CACHE_LINE_SIZE - 1) &
                    ~(TVM_DSP_CACHE_LINE_SIZE - 1);

    /* Use cache writeback intrinsic if available */
    /* Note: C7x compiler may provide __cache_wb() or similar */
    /* For now, we rely on the cache being write-through or
     * the application handling cache coherency explicitly */
    (void)start;
    (void)end;
  }
#else
  (void)addr;
  (void)size;
#endif
}

void tvm_dsp_cache_invalidate(void* addr, size_t size) {
#ifdef __TI_COMPILER_VERSION__
  /*
   * C7x cache invalidate
   *
   * Similar to writeback, we align to cache lines and use
   * intrinsics when available.
   */
  if (addr != NULL && size > 0) {
    uintptr_t start = (uintptr_t)addr & ~(TVM_DSP_CACHE_LINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + TVM_DSP_CACHE_LINE_SIZE - 1) &
                    ~(TVM_DSP_CACHE_LINE_SIZE - 1);
    (void)start;
    (void)end;
  }
#else
  (void)addr;
  (void)size;
#endif
}

void tvm_dsp_cache_writeback_invalidate(void* addr, size_t size) {
#ifdef __TI_COMPILER_VERSION__
  /*
   * C7x cache writeback + invalidate
   */
  if (addr != NULL && size > 0) {
    uintptr_t start = (uintptr_t)addr & ~(TVM_DSP_CACHE_LINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + TVM_DSP_CACHE_LINE_SIZE - 1) &
                    ~(TVM_DSP_CACHE_LINE_SIZE - 1);
    (void)start;
    (void)end;
  }
#else
  (void)addr;
  (void)size;
#endif
}

void tvm_dsp_log(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
#ifdef __TI_COMPILER_VERSION__
  /* On C7x, use CIO printf (requires host-side CIO server via JTAG) */
  vprintf(fmt, args);
#else
  vprintf(fmt, args);
#endif
  va_end(args);
}

int tvm_dsp_runtime_is_initialized(void) {
  return g_platform_initialized;
}
