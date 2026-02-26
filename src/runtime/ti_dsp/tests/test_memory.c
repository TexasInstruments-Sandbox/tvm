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
 * \file test_memory.c
 * \brief TVM DSP Runtime - Memory allocator tests
 */

#include "../platform/dsp_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test helper macros */
#define TEST(name) \
  do {             \
    printf("TEST: %s... ", #name);
#define TEST_END(result)                      \
  if (result) {                               \
    printf("PASSED\n");                       \
  } else {                                    \
    printf("FAILED\n");                       \
    test_failed++;                            \
  }                                           \
  test_count++;                               \
  }                                           \
  while (0)

static int test_count = 0;
static int test_failed = 0;

/* Test platform initialization */
static void test_platform_init(void) {
  TEST(platform_init);
  int ret = tvm_dsp_platform_init();
  TEST_END(ret == 0);
}

/* Test basic allocation */
static void test_basic_alloc(void) {
  TEST(basic_alloc);
  void* ptr = tvm_dsp_alloc(1024, 64, TVM_DSP_MEM_FAST);
  int result = (ptr != NULL);
  if (ptr) {
    /* Write to allocated memory */
    memset(ptr, 0xAB, 1024);
    tvm_dsp_free(ptr);
  }
  TEST_END(result);
}

/* Test aligned allocation */
static void test_aligned_alloc(void) {
  TEST(aligned_alloc);
  void* ptr = tvm_dsp_alloc(256, 128, TVM_DSP_MEM_FAST);
  int result = (ptr != NULL && ((uintptr_t)ptr % 128) == 0);
  if (ptr) {
    tvm_dsp_free(ptr);
  }
  TEST_END(result);
}

/* Test multiple allocations */
static void test_multiple_alloc(void) {
  TEST(multiple_alloc);
  void* ptrs[10];
  int i;
  int result = 1;

  for (i = 0; i < 10; i++) {
    ptrs[i] = tvm_dsp_alloc(256, 64, TVM_DSP_MEM_FAST);
    if (ptrs[i] == NULL) {
      result = 0;
      break;
    }
    memset(ptrs[i], i, 256);
  }

  /* Free in reverse order */
  for (i = 9; i >= 0; i--) {
    if (ptrs[i]) {
      tvm_dsp_free(ptrs[i]);
    }
  }

  TEST_END(result);
}

/* Test allocation from both pools */
static void test_both_pools(void) {
  TEST(both_pools);
  void* fast_ptr = tvm_dsp_alloc(1024, 64, TVM_DSP_MEM_FAST);
  void* main_ptr = tvm_dsp_alloc(4096, 64, TVM_DSP_MEM_MAIN);

  int result = (fast_ptr != NULL && main_ptr != NULL);

  /* Pointers should be different (different pools) */
  if (result) {
    result = (fast_ptr != main_ptr);
  }

  if (fast_ptr) tvm_dsp_free(fast_ptr);
  if (main_ptr) tvm_dsp_free(main_ptr);

  TEST_END(result);
}

/* Test memory statistics */
static void test_memory_stats(void) {
  TEST(memory_stats);
  TVMDSPMemoryStats stats_before, stats_after;

  tvm_dsp_get_memory_stats(TVM_DSP_MEM_FAST, &stats_before);

  void* ptr = tvm_dsp_alloc(1024, 64, TVM_DSP_MEM_FAST);

  tvm_dsp_get_memory_stats(TVM_DSP_MEM_FAST, &stats_after);

  int result = (ptr != NULL);
  if (result) {
    /* Allocation count should increase */
    result = (stats_after.alloc_count > stats_before.alloc_count);
  }
  if (result) {
    /* Used size should increase */
    result = (stats_after.used_size > stats_before.used_size);
  }

  if (ptr) tvm_dsp_free(ptr);

  TEST_END(result);
}

/* Test free memory query */
static void test_free_memory(void) {
  TEST(free_memory);
  size_t free_before = tvm_dsp_get_free_memory(TVM_DSP_MEM_FAST);

  void* ptr = tvm_dsp_alloc(4096, 64, TVM_DSP_MEM_FAST);

  size_t free_after = tvm_dsp_get_free_memory(TVM_DSP_MEM_FAST);

  int result = (ptr != NULL);
  if (result) {
    /* Free memory should decrease after allocation */
    result = (free_after < free_before);
  }

  if (ptr) tvm_dsp_free(ptr);

  TEST_END(result);
}

/* Test allocation reuse after free */
static void test_reuse_after_free(void) {
  TEST(reuse_after_free);
  void* ptr1 = tvm_dsp_alloc(512, 64, TVM_DSP_MEM_FAST);
  tvm_dsp_free(ptr1);

  /* Second allocation of same size should reuse freed block */
  void* ptr2 = tvm_dsp_alloc(512, 64, TVM_DSP_MEM_FAST);

  /* Note: ptr2 might be same as ptr1 (reuse) or different (from bump) */
  int result = (ptr1 != NULL && ptr2 != NULL);

  if (ptr2) tvm_dsp_free(ptr2);

  TEST_END(result);
}

/* Test large allocation */
static void test_large_alloc(void) {
  TEST(large_alloc);
  /* Query available memory and allocate 75% of it */
  size_t available = tvm_dsp_get_free_memory(TVM_DSP_MEM_MAIN);
  size_t alloc_size = (available * 3) / 4; /* 75% of available */
  if (alloc_size < 1024) alloc_size = 1024; /* minimum 1KB */

  void* ptr = tvm_dsp_alloc(alloc_size, 64, TVM_DSP_MEM_MAIN);
  int result = (ptr != NULL);

  if (ptr) {
    /* Write pattern to verify access */
    memset(ptr, 0x55, alloc_size);
    tvm_dsp_free(ptr);
  }

  TEST_END(result);
}

/* Test null free (should not crash) */
static void test_null_free(void) {
  TEST(null_free);
  tvm_dsp_free(NULL); /* Should not crash */
  TEST_END(1);
}

/* Test cycle counter */
static void test_cycle_counter(void) {
  TEST(cycle_counter);
  tvm_dsp_cycle_counter_reset();
  uint64_t start = tvm_dsp_cycle_counter_get();

  /* Do some work */
  volatile int sum = 0;
  for (int i = 0; i < 10000; i++) {
    sum += i;
  }
  (void)sum;

  uint64_t end = tvm_dsp_cycle_counter_get();

  /* End should be >= start (time moved forward) */
  int result = (end >= start);

  TEST_END(result);
}

/* Test platform shutdown */
static void test_platform_shutdown(void) {
  TEST(platform_shutdown);
  tvm_dsp_platform_shutdown();
  TEST_END(1);
}

int main(void) {
  printf("=== TVM DSP Runtime Memory Tests ===\n");
  printf("Target: %s\n\n", TVM_DSP_TARGET_NAME);

  /* Run tests in order */
  test_platform_init();
  test_basic_alloc();
  test_aligned_alloc();
  test_multiple_alloc();
  test_both_pools();
  test_memory_stats();
  test_free_memory();
  test_reuse_after_free();
  test_large_alloc();
  test_null_free();
  test_cycle_counter();
  test_platform_shutdown();

  /* Summary */
  printf("\n=== Test Summary ===\n");
  printf("Total: %d, Passed: %d, Failed: %d\n", test_count, test_count - test_failed, test_failed);

  return (test_failed > 0) ? 1 : 0;
}
