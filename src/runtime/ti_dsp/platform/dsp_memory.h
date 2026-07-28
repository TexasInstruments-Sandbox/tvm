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
 * \file dsp_memory.h
 * \brief TVM DSP Runtime - Memory Allocator Interface
 *
 * This header defines the internal memory allocator interface used by
 * platform-specific implementations. Applications should use the
 * tvm_dsp_alloc/tvm_dsp_free functions from dsp_platform.h instead.
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_DSP_MEMORY_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_DSP_MEMORY_H_

#include <stddef.h>
#include <stdint.h>

/* Note: This header is platform-agnostic. Platform-specific code should
 * include dsp_platform.h separately when needed. */

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Memory pool descriptor.
 *
 * Each pool manages a contiguous region of memory using a simple
 * bump allocator with free list for deallocation.
 */
typedef struct TVMDSPMemoryPoolDesc {
  void* base;           /*!< Base address of pool */
  size_t size;          /*!< Total size in bytes */
  size_t allocated;     /*!< Currently allocated bytes */
  size_t peak;          /*!< Peak allocation (high water mark) */
  uint32_t num_allocs;  /*!< Allocation count */
  uint32_t num_frees;   /*!< Free count */
  void* free_list;      /*!< Head of free list (internal) */
  void* bump_ptr;       /*!< Current bump pointer (internal) */
} TVMDSPMemoryPoolDesc;

/*!
 * \brief Initialize a memory pool.
 *
 * \param pool Pool descriptor to initialize.
 * \param base Base address of memory region.
 * \param size Size of memory region in bytes.
 * \return 0 on success, non-zero on error.
 */
int tvm_dsp_memory_pool_init(TVMDSPMemoryPoolDesc* pool, void* base, size_t size);

/*!
 * \brief Reset a memory pool to initial state.
 *
 * Frees all allocations and resets statistics. The pool retains
 * its base address and size.
 *
 * \param pool Pool to reset.
 */
void tvm_dsp_memory_pool_reset(TVMDSPMemoryPoolDesc* pool);

/*!
 * \brief Allocate from a specific pool descriptor.
 *
 * \param pool Pool to allocate from.
 * \param size Number of bytes to allocate.
 * \param alignment Required alignment (must be power of 2).
 * \return Pointer to allocated memory, or NULL on failure.
 */
void* tvm_dsp_memory_pool_alloc(TVMDSPMemoryPoolDesc* pool, size_t size, size_t alignment);

/*!
 * \brief Free memory to its pool.
 *
 * \param pool Pool the memory belongs to.
 * \param ptr Pointer to free.
 */
void tvm_dsp_memory_pool_free(TVMDSPMemoryPoolDesc* pool, void* ptr);

/*!
 * \brief Get free space in a pool.
 *
 * \param pool Pool to query.
 * \return Approximate free bytes available.
 */
size_t tvm_dsp_memory_pool_free_space(TVMDSPMemoryPoolDesc* pool);

/*!
 * \brief Check if a pointer belongs to a pool.
 *
 * \param pool Pool to check.
 * \param ptr Pointer to test.
 * \return Non-zero if ptr is within pool's address range.
 */
int tvm_dsp_memory_pool_contains(TVMDSPMemoryPoolDesc* pool, void* ptr);

/*!
 * \brief Internal: record a MAIN-pool allocation failure.
 *
 * Called only from each platform's tvm_dsp_alloc(), inside its existing
 * DDR-OOM log branch. Not part of the public dsp_platform.h API -- do not
 * call from firmware code (use tvm_dsp_oom_take() instead).
 *
 * \param requested Bytes requested by the failing allocation.
 * \param free_at_fail Pool free bytes at the moment of failure.
 * \param pool_size Total pool bytes.
 */
void tvm_dsp_oom_record(size_t requested, size_t free_at_fail, size_t pool_size);

/*!
 * \brief Memory allocation header (internal).
 *
 * Prepended to each allocation for tracking.
 */
typedef struct TVMDSPAllocHeader {
  uint32_t magic;       /*!< Magic number for validation */
  uint32_t size;        /*!< Allocation size (excluding header) */
  struct TVMDSPAllocHeader* next; /*!< Next in free list (when freed) */
} TVMDSPAllocHeader;

/*! Magic number for allocation validation */
#define TVM_DSP_ALLOC_MAGIC 0x54564D41 /* "TVMA" */

#ifdef __cplusplus
}
#endif

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_DSP_MEMORY_H_ */
