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
 * \file dsp_platform.h
 * \brief TVM DSP Runtime - Platform Abstraction Layer
 *
 * This header defines the generic interface that all DSP platforms
 * (C66x, C7x, host emulation) must implement.
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_DSP_PLATFORM_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_DSP_PLATFORM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Target identification - set by build system */
#if defined(TVM_DSP_TARGET_C66X)
#include "c66x/c66x_platform.h"
#elif defined(TVM_DSP_TARGET_C7X)
#include "c7x/c7x_platform.h"
#elif defined(TVM_DSP_TARGET_HOST)
#include "host/host_platform.h"
#else
#error \
    "No TVM DSP target defined. Set TVM_DSP_TARGET_C66X, TVM_DSP_TARGET_C7X, or TVM_DSP_TARGET_HOST"
#endif

/*!
 * \brief Memory pool types for DSP platforms.
 *
 * DSP platforms typically have a tiered memory hierarchy:
 * - Fast local memory (L2 SRAM on C66x/C7x) for hot data
 * - Main memory (L3/DDR) for larger, less frequently accessed data
 */
typedef enum {
  TVM_DSP_MEM_FAST = 0, /*!< Fast local memory (L2 on C66x/C7x) */
  TVM_DSP_MEM_MAIN = 1, /*!< Main memory (L3/DDR) */
  TVM_DSP_MEM_COUNT
} TVMDSPMemoryPool;

/*!
 * \brief Memory statistics for a pool.
 */
typedef struct {
  size_t total_size;    /*!< Total pool size in bytes */
  size_t used_size;     /*!< Currently allocated bytes */
  size_t peak_used;     /*!< Peak allocated bytes */
  uint32_t alloc_count; /*!< Number of allocations */
  uint32_t free_count;  /*!< Number of frees */
} TVMDSPMemoryStats;

/*!
 * \brief Initialize the DSP platform.
 *
 * This function must be called once at startup before any other
 * DSP runtime functions. It initializes memory pools, cache
 * configuration, and other platform-specific resources.
 *
 * \return 0 on success, non-zero error code on failure.
 */
int tvm_dsp_platform_init(void);

/*!
 * \brief Shutdown the DSP platform.
 *
 * Releases all resources allocated by the platform. After calling
 * this function, no other DSP runtime functions should be called
 * until tvm_dsp_platform_init() is called again.
 */
void tvm_dsp_platform_shutdown(void);

/*!
 * \brief Allocate memory from a specific pool.
 *
 * \param size Number of bytes to allocate.
 * \param alignment Required alignment (must be power of 2).
 * \param pool Memory pool to allocate from.
 * \return Pointer to allocated memory, or NULL on failure.
 */
void* tvm_dsp_alloc(size_t size, size_t alignment, TVMDSPMemoryPool pool);

/*!
 * \brief Free previously allocated memory.
 *
 * The implementation determines which pool the memory belongs to
 * based on the pointer address.
 *
 * \param ptr Pointer to free (NULL is safe).
 */
void tvm_dsp_free(void* ptr);

/*!
 * \brief Reset all memory pools, reclaiming all allocated memory.
 *
 * This must only be called when no pool memory is in use (e.g. after
 * a complete module unload cycle).  It eliminates free-list
 * fragmentation so that subsequent loads can use the full pool.
 */
void tvm_dsp_reset_pools(void);

/*!
 * \brief Get free memory available in a pool.
 *
 * \param pool Memory pool to query.
 * \return Number of free bytes available.
 */
size_t tvm_dsp_get_free_memory(TVMDSPMemoryPool pool);

/*!
 * \brief Get detailed memory statistics for a pool.
 *
 * \param pool Memory pool to query.
 * \param stats Output structure for statistics.
 */
void tvm_dsp_get_memory_stats(TVMDSPMemoryPool pool, TVMDSPMemoryStats* stats);

/*!
 * \brief Reset the cycle counter.
 *
 * On DSP targets, this resets the hardware cycle counter.
 * On host, this captures a reference timestamp.
 */
void tvm_dsp_cycle_counter_reset(void);

/*!
 * \brief Get the current cycle count.
 *
 * \return Cycles elapsed since last reset (or simulated cycles on host).
 */
uint64_t tvm_dsp_cycle_counter_get(void);

/*!
 * \brief Write back cache lines containing the given address range.
 *
 * Ensures data is written from cache to main memory.
 * No-op on host emulation.
 *
 * \param addr Start address.
 * \param size Size in bytes.
 */
void tvm_dsp_cache_writeback(void* addr, size_t size);

/*!
 * \brief Invalidate cache lines containing the given address range.
 *
 * Ensures subsequent reads fetch from main memory.
 * No-op on host emulation.
 *
 * \param addr Start address.
 * \param size Size in bytes.
 */
void tvm_dsp_cache_invalidate(void* addr, size_t size);

/*!
 * \brief Write back and invalidate cache lines.
 *
 * Combination of writeback and invalidate operations.
 * No-op on host emulation.
 *
 * \param addr Start address.
 * \param size Size in bytes.
 */
void tvm_dsp_cache_writeback_invalidate(void* addr, size_t size);

/*!
 * \brief Log a message (printf-style).
 *
 * On DSP targets, this typically outputs to UART or CIO.
 * On host, this outputs to stdout.
 *
 * \param fmt Format string (printf-style).
 * \param ... Format arguments.
 */
void tvm_dsp_log(const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_DSP_PLATFORM_H_ */
