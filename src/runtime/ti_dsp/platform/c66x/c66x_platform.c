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
 * \file c66x_platform.c
 * \brief TVM DSP Runtime - C66x Platform Implementation
 */

/* TVM_DSP_TARGET_C66X must be defined by the build system */
#include "../dsp_platform.h"
#include "../dsp_memory.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef __TI_COMPILER_VERSION__
#include <c6x.h>
#endif

/* Global memory pools */
static TVMDSPMemoryPoolDesc g_fast_pool; /* L2 pool */
static TVMDSPMemoryPoolDesc g_main_pool; /* L3 pool */

/* Pre-allocated heap regions (placed by linker script) */
#ifdef __TI_COMPILER_VERSION__
/* These symbols are defined in the linker script */
extern char __TVM_DSP_L2_HEAP_START[];
extern char __TVM_DSP_L2_HEAP_END[];
extern char __TVM_DSP_L3_HEAP_START[];
extern char __TVM_DSP_L3_HEAP_END[];

/* Alternatively, use fixed addresses from config */
static uint8_t* g_l2_heap_base = (uint8_t*)TVM_DSP_L2_BASE;
static uint8_t* g_l3_heap_base = (uint8_t*)TVM_DSP_L3_BASE;
#else
/* Fallback for non-TI compilers (shouldn't be used) */
static uint8_t g_l2_heap_storage[TVM_DSP_L2_SIZE];
static uint8_t g_l3_heap_storage[TVM_DSP_L3_SIZE];
static uint8_t* g_l2_heap_base = g_l2_heap_storage;
static uint8_t* g_l3_heap_base = g_l3_heap_storage;
#endif

/* Platform initialization state */
static int g_platform_initialized = 0;

int tvm_dsp_platform_init(void) {
  if (g_platform_initialized) {
    return 0; /* Already initialized */
  }

  /* Initialize L2 (fast) memory pool */
  int ret = tvm_dsp_memory_pool_init(&g_fast_pool, g_l2_heap_base, TVM_DSP_L2_SIZE);
  if (ret != 0) {
    return ret;
  }

  /* Initialize L3 (main) memory pool */
  ret = tvm_dsp_memory_pool_init(&g_main_pool, g_l3_heap_base, TVM_DSP_L3_SIZE);
  if (ret != 0) {
    return ret;
  }

  /* Reset cycle counter */
  tvm_dsp_cycle_counter_reset();

  g_platform_initialized = 1;
  tvm_dsp_log("TVM DSP Runtime initialized on %s (%s)\n", TVM_DSP_TARGET_NAME, TVM_DSP_DEVICE_NAME);
  tvm_dsp_log("  L2 pool: 0x%08x - 0x%08x (%u KB)\n", (unsigned int)(uintptr_t)g_l2_heap_base,
              (unsigned int)((uintptr_t)g_l2_heap_base + TVM_DSP_L2_SIZE),
              (unsigned int)(TVM_DSP_L2_SIZE / 1024));
  tvm_dsp_log("  L3 pool: 0x%08x - 0x%08x (%u KB)\n", (unsigned int)(uintptr_t)g_l3_heap_base,
              (unsigned int)((uintptr_t)g_l3_heap_base + TVM_DSP_L3_SIZE),
              (unsigned int)(TVM_DSP_L3_SIZE / 1024));

  return 0;
}

void tvm_dsp_platform_shutdown(void) {
  if (!g_platform_initialized) {
    return;
  }

  /* Log final statistics */
  TVMDSPMemoryStats stats;
  tvm_dsp_get_memory_stats(TVM_DSP_MEM_FAST, &stats);
  tvm_dsp_log("L2 pool stats: %u allocs, %u frees, peak %u bytes\n", stats.alloc_count,
              stats.free_count, (unsigned int)stats.peak_used);

  tvm_dsp_get_memory_stats(TVM_DSP_MEM_MAIN, &stats);
  tvm_dsp_log("L3 pool stats: %u allocs, %u frees, peak %u bytes\n", stats.alloc_count,
              stats.free_count, (unsigned int)stats.peak_used);

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
    const char* pool_name = (pool == TVM_DSP_MEM_FAST) ? "L2" : "L3";
    tvm_dsp_log("ERROR: OOM in %s pool: requested %u bytes, "
                "free %u / %u bytes\n",
                pool_name, (unsigned)size,
                (unsigned)tvm_dsp_memory_pool_free_space(pool_desc),
                (unsigned)pool_desc->size);
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

/* C66x: no persistent DLOAD session, so watermarks are always pool base. */
void tvm_dsp_save_infer_watermark(void) { }
void tvm_dsp_restore_infer_watermark(void) {
  if (!g_platform_initialized) return;
  tvm_dsp_memory_pool_reset(&g_fast_pool);
  tvm_dsp_memory_pool_reset(&g_main_pool);
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
  /* On C66x, use TSCL/TSCH registers */
  TSCL = 0; /* Writing to TSCL resets both TSCL and TSCH */
#endif
  /* No-op on host - cycle counter not available */
}

uint64_t tvm_dsp_cycle_counter_get(void) {
#ifdef __TI_COMPILER_VERSION__
  /* Read 64-bit timestamp counter */
  uint32_t low = TSCL;
  uint32_t high = TSCH;
  return ((uint64_t)high << 32) | low;
#else
  return 0;
#endif
}

void tvm_dsp_cache_writeback(void* addr, size_t size) {
#ifdef __TI_COMPILER_VERSION__
  /* C66x cache writeback using CSL or direct MAR manipulation */
  /* For now, use the compiler intrinsic if available */
  /* TODO: Implement proper cache operations */
  (void)addr;
  (void)size;
#else
  (void)addr;
  (void)size;
#endif
}

void tvm_dsp_cache_invalidate(void* addr, size_t size) {
#ifdef __TI_COMPILER_VERSION__
  /* C66x cache invalidate */
  (void)addr;
  (void)size;
#else
  (void)addr;
  (void)size;
#endif
}

void tvm_dsp_cache_writeback_invalidate(void* addr, size_t size) {
#ifdef __TI_COMPILER_VERSION__
  /* C66x cache writeback + invalidate */
  (void)addr;
  (void)size;
#else
  (void)addr;
  (void)size;
#endif
}

void tvm_dsp_log(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
#ifdef __TI_COMPILER_VERSION__
  /* On C66x, use CIO printf (requires host-side CIO server) */
  vprintf(fmt, args);
#else
  vprintf(fmt, args);
#endif
  va_end(args);
}

int tvm_dsp_runtime_is_initialized(void) { return g_platform_initialized; }
