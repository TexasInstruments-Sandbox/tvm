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
 * \file memory_pool.c
 * \brief TVM DSP Runtime - Generic Memory Pool Implementation
 *
 * This implements a simple bump-pointer allocator with free list.
 * The allocator is platform-agnostic and used by all DSP targets
 * (C66x, C7x, host emulation).
 *
 * Design goals:
 * - Fast allocation (O(1) bump pointer)
 * - Minimal fragmentation (large allocations from bump, small reuse free list)
 * - No external dependencies (no malloc/free)
 *
 * The allocator uses a header-per-allocation approach for tracking
 * allocation sizes, enabling proper free list management.
 */

#include "../dsp_memory.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Minimum allocation size (includes header) */
#define MIN_ALLOC_SIZE (sizeof(TVMDSPAllocHeader) + 16)

/* Align value up to alignment (alignment must be power of 2) */
static inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

/* Align pointer up to alignment */
static inline void* ptr_align_up(void* ptr, size_t alignment) {
  uintptr_t addr = (uintptr_t)ptr;
  return (void*)align_up(addr, alignment);
}

int tvm_dsp_memory_pool_init(TVMDSPMemoryPoolDesc* pool, void* base, size_t size) {
  if (pool == NULL || base == NULL || size < MIN_ALLOC_SIZE) {
    return -1;
  }

  pool->base = base;
  pool->size = size;
  pool->allocated = 0;
  pool->peak = 0;
  pool->num_allocs = 0;
  pool->num_frees = 0;
  pool->free_list = NULL;
  pool->bump_ptr = base;

  /* No memset needed - bump allocator doesn't require zeroed memory.
   * Individual allocations can be zeroed on demand if needed. */

  return 0;
}

void tvm_dsp_memory_pool_reset(TVMDSPMemoryPoolDesc* pool) {
  if (pool == NULL) {
    return;
  }

  pool->allocated = 0;
  pool->num_allocs = 0;
  pool->num_frees = 0;
  pool->free_list = NULL;
  pool->bump_ptr = pool->base;
  /* Note: peak is preserved for statistics */
}

void* tvm_dsp_memory_pool_alloc(TVMDSPMemoryPoolDesc* pool, size_t size, size_t alignment) {
  if (pool == NULL || size == 0) {
    return NULL;
  }

  /* Enforce minimum alignment */
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }

  /* Round up alignment to power of 2 */
  if (alignment & (alignment - 1)) {
    /* Not a power of 2, round up */
    size_t a = 1;
    while (a < alignment) a <<= 1;
    alignment = a;
  }

  /* Calculate total allocation size:
   * [padding for alignment] [header] [user data]
   * The header must be placed such that user data starts at aligned address
   */
  size_t header_size = sizeof(TVMDSPAllocHeader);
  size_t total_size = header_size + align_up(size, alignment);

  /* First, try to find a suitable block in the free list */
  TVMDSPAllocHeader** prev_ptr = (TVMDSPAllocHeader**)&pool->free_list;
  TVMDSPAllocHeader* curr = (TVMDSPAllocHeader*)pool->free_list;

  while (curr != NULL) {
    /* Check if this block is large enough */
    size_t block_size = curr->size + header_size;
    if (block_size >= total_size) {
      /* Check alignment */
      void* user_ptr = (void*)((uint8_t*)curr + header_size);
      void* aligned_ptr = ptr_align_up(user_ptr, alignment);
      size_t padding = (uint8_t*)aligned_ptr - (uint8_t*)user_ptr;

      if (block_size >= total_size + padding) {
        /* Remove from free list and return */
        *prev_ptr = curr->next;

        /* Reinitialize header */
        curr->magic = TVM_DSP_ALLOC_MAGIC;
        curr->next = NULL;

        pool->allocated += block_size;
        if (pool->allocated > pool->peak) {
          pool->peak = pool->allocated;
        }
        pool->num_allocs++;

        /* Return aligned pointer */
        return aligned_ptr;
      }
    }
    prev_ptr = &curr->next;
    curr = curr->next;
  }

  /* No suitable block in free list, use bump allocator */
  uint8_t* pool_end = (uint8_t*)pool->base + pool->size;
  uint8_t* bump = (uint8_t*)pool->bump_ptr;

  /* Calculate aligned position for header + user data */
  uint8_t* header_pos = bump;
  uint8_t* user_pos = header_pos + header_size;
  uint8_t* aligned_user_pos = (uint8_t*)ptr_align_up(user_pos, alignment);

  /* Adjust header position to maintain alignment */
  if (aligned_user_pos != user_pos) {
    header_pos = aligned_user_pos - header_size;
  }

  uint8_t* alloc_end = aligned_user_pos + align_up(size, alignment);

  /* Check if we have enough space */
  if (alloc_end > pool_end) {
    return NULL; /* Out of memory */
  }

  /* Initialize header */
  TVMDSPAllocHeader* header = (TVMDSPAllocHeader*)header_pos;
  header->magic = TVM_DSP_ALLOC_MAGIC;
  header->size = (uint32_t)(alloc_end - aligned_user_pos);
  header->next = NULL;

  /* Update bump pointer */
  pool->bump_ptr = alloc_end;

  /* Update statistics */
  size_t alloc_size = alloc_end - bump;
  pool->allocated += alloc_size;
  if (pool->allocated > pool->peak) {
    pool->peak = pool->allocated;
  }
  pool->num_allocs++;

  return aligned_user_pos;
}

void tvm_dsp_memory_pool_free(TVMDSPMemoryPoolDesc* pool, void* ptr) {
  if (pool == NULL || ptr == NULL) {
    return;
  }

  /* Get header */
  TVMDSPAllocHeader* header = (TVMDSPAllocHeader*)((uint8_t*)ptr - sizeof(TVMDSPAllocHeader));

  /* Validate magic number */
  if (header->magic != TVM_DSP_ALLOC_MAGIC) {
    return; /* Invalid free - silently ignore */
  }

  /* Mark as freed (change magic) */
  header->magic = 0;

  /* Add to free list (LIFO) */
  header->next = (TVMDSPAllocHeader*)pool->free_list;
  pool->free_list = header;

  /* Update statistics */
  size_t block_size = header->size + sizeof(TVMDSPAllocHeader);
  if (pool->allocated >= block_size) {
    pool->allocated -= block_size;
  }
  pool->num_frees++;

  /* Auto-reset: if every allocation has been freed, reclaim the
   * entire pool.  Uses alloc/free counts because the byte-level
   * 'allocated' field can accumulate rounding from alignment
   * padding in the bump allocator. */
  if (pool->num_frees == pool->num_allocs) {
    pool->free_list = NULL;
    pool->bump_ptr = pool->base;
    pool->allocated = 0;
  }
}

size_t tvm_dsp_memory_pool_free_space(TVMDSPMemoryPoolDesc* pool) {
  if (pool == NULL) {
    return 0;
  }

  /* Calculate remaining bump space */
  uint8_t* pool_end = (uint8_t*)pool->base + pool->size;
  uint8_t* bump = (uint8_t*)pool->bump_ptr;
  size_t bump_free = (bump < pool_end) ? (pool_end - bump) : 0;

  /* Add up free list space */
  size_t free_list_total = 0;
  TVMDSPAllocHeader* curr = (TVMDSPAllocHeader*)pool->free_list;
  while (curr != NULL) {
    free_list_total += curr->size + sizeof(TVMDSPAllocHeader);
    curr = curr->next;
  }

  return bump_free + free_list_total;
}

int tvm_dsp_memory_pool_contains(TVMDSPMemoryPoolDesc* pool, void* ptr) {
  if (pool == NULL || ptr == NULL) {
    return 0;
  }

  uintptr_t addr = (uintptr_t)ptr;
  uintptr_t base = (uintptr_t)pool->base;
  uintptr_t end = base + pool->size;

  return (addr >= base && addr < end);
}
