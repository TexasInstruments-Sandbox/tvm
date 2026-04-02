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
 * \file host_platform.c
 * \brief TVM DSP Runtime - Host (PC) Emulation Platform Implementation
 */

#include "../dsp_memory.h"
#include "../dsp_platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Global memory pools */
static TVMDSPMemoryPoolDesc g_fast_pool; /* Emulated L2 pool */
static TVMDSPMemoryPoolDesc g_main_pool; /* Emulated L3 pool */

/* Heap storage (allocated via malloc at init) */
static void* g_fast_heap = NULL;
static void* g_main_heap = NULL;

/* Platform initialization state */
static int g_platform_initialized = 0;

/* High resolution timer base */
static struct timespec g_timer_base;

int tvm_dsp_platform_init(void) {
  if (g_platform_initialized) {
    return 0; /* Already initialized */
  }

  /* Allocate emulated L2 heap */
  g_fast_heap = malloc(TVM_DSP_L2_SIZE);
  if (g_fast_heap == NULL) {
    fprintf(stderr, "ERROR: Failed to allocate emulated L2 heap (%zu bytes)\n",
            (size_t)TVM_DSP_L2_SIZE);
    return -1;
  }

  /* Allocate emulated L3 heap */
  g_main_heap = malloc(TVM_DSP_L3_SIZE);
  if (g_main_heap == NULL) {
    fprintf(stderr, "ERROR: Failed to allocate emulated L3 heap (%zu bytes)\n",
            (size_t)TVM_DSP_L3_SIZE);
    free(g_fast_heap);
    g_fast_heap = NULL;
    return -1;
  }

  /* Initialize memory pools */
  int ret = tvm_dsp_memory_pool_init(&g_fast_pool, g_fast_heap, TVM_DSP_L2_SIZE);
  if (ret != 0) {
    free(g_fast_heap);
    free(g_main_heap);
    g_fast_heap = NULL;
    g_main_heap = NULL;
    return ret;
  }

  ret = tvm_dsp_memory_pool_init(&g_main_pool, g_main_heap, TVM_DSP_L3_SIZE);
  if (ret != 0) {
    free(g_fast_heap);
    free(g_main_heap);
    g_fast_heap = NULL;
    g_main_heap = NULL;
    return ret;
  }

  /* Initialize timer base */
  tvm_dsp_cycle_counter_reset();

  g_platform_initialized = 1;
  tvm_dsp_log("TVM DSP Runtime initialized on %s (%s)\n", TVM_DSP_TARGET_NAME, TVM_DSP_DEVICE_NAME);
  tvm_dsp_log("  Fast pool: %zu KB\n", (size_t)(TVM_DSP_L2_SIZE / 1024));
  tvm_dsp_log("  Main pool: %zu KB\n", (size_t)(TVM_DSP_L3_SIZE / 1024));

  return 0;
}

void tvm_dsp_platform_shutdown(void) {
  if (!g_platform_initialized) {
    return;
  }

  /* Log final statistics */
  TVMDSPMemoryStats stats;
  tvm_dsp_get_memory_stats(TVM_DSP_MEM_FAST, &stats);
  tvm_dsp_log("Fast pool stats: %u allocs, %u frees, peak %zu bytes\n", stats.alloc_count,
              stats.free_count, stats.peak_used);

  tvm_dsp_get_memory_stats(TVM_DSP_MEM_MAIN, &stats);
  tvm_dsp_log("Main pool stats: %u allocs, %u frees, peak %zu bytes\n", stats.alloc_count,
              stats.free_count, stats.peak_used);

  /* Reset pools and free backing memory */
  tvm_dsp_memory_pool_reset(&g_fast_pool);
  tvm_dsp_memory_pool_reset(&g_main_pool);

  if (g_fast_heap) {
    free(g_fast_heap);
    g_fast_heap = NULL;
  }
  if (g_main_heap) {
    free(g_main_heap);
    g_main_heap = NULL;
  }

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
    const char* pool_name = (pool == TVM_DSP_MEM_FAST) ? "L2" : "Main";
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
    tvm_dsp_log("WARNING: Free of unknown pointer %p\n", ptr);
  }
}

void tvm_dsp_reset_pools(void) {
  if (!g_platform_initialized) {
    return;
  }
  tvm_dsp_memory_pool_reset(&g_fast_pool);
  tvm_dsp_memory_pool_reset(&g_main_pool);
}

void tvm_dsp_save_infer_watermark(void) { }
void tvm_dsp_restore_infer_watermark(void) { }

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
#if defined(__APPLE__)
  clock_gettime(CLOCK_MONOTONIC, &g_timer_base);
#elif defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
  clock_gettime(CLOCK_MONOTONIC, &g_timer_base);
#else
  g_timer_base.tv_sec = 0;
  g_timer_base.tv_nsec = 0;
#endif
}

uint64_t tvm_dsp_cycle_counter_get(void) {
#if defined(__APPLE__) || (defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0)
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  /* Calculate elapsed nanoseconds */
  int64_t sec_diff = now.tv_sec - g_timer_base.tv_sec;
  int64_t nsec_diff = now.tv_nsec - g_timer_base.tv_nsec;
  uint64_t elapsed_ns = (uint64_t)(sec_diff * 1000000000LL + nsec_diff);

  /* Convert to simulated cycles (1 cycle = 1ns at 1GHz) */
  return elapsed_ns;
#else
  return 0;
#endif
}

void tvm_dsp_cache_writeback(void* addr, size_t size) {
  /* No-op on host */
  (void)addr;
  (void)size;
}

void tvm_dsp_cache_invalidate(void* addr, size_t size) {
  /* No-op on host */
  (void)addr;
  (void)size;
}

void tvm_dsp_cache_writeback_invalidate(void* addr, size_t size) {
  /* No-op on host */
  (void)addr;
  (void)size;
}

void tvm_dsp_log(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
}

int tvm_dsp_runtime_is_initialized(void) { return g_platform_initialized; }
