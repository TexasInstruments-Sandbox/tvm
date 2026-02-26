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
 * \file tests/test_span.cpp
 * \brief Tests for the Span non-owning array view
 *
 * Tests verify that:
 * - Default construction creates empty span
 * - Construction from pointer+size works
 * - Construction from C array deduces size
 * - Element access via operator[] works
 * - Iterators enable range-based for loops
 * - Subspan operations work correctly
 * - Const correctness is maintained
 */

#include <cstdio>
#include <cstring>

/* Include the span header */
#include "span.h"

/* Include platform for initialization */
extern "C" {
#include "dsp_platform.h"
}

/* Test framework macros */
static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name)                          \
  do {                                      \
    printf("Test: %s... ", #name);          \
    g_tests_run++;                          \
    if (test_##name()) {                    \
      printf("PASSED\n");                   \
      g_tests_passed++;                     \
    } else {                                \
      printf("FAILED\n");                   \
    }                                       \
  } while (0)

/*
 * =============================================================================
 * TEST CASES
 * =============================================================================
 */

/*!
 * \brief Test 1: Default construction creates empty span
 */
static int test_default_construction() {
  tvm::dsp::Span<int> span;

  if (span.data() != nullptr) return 0;
  if (span.size() != 0) return 0;
  if (!span.empty()) return 0;

  /* Iteration over empty span should work (zero iterations) */
  int count = 0;
  for (int x : span) {
    (void)x;
    count++;
  }
  if (count != 0) return 0;

  return 1;
}

/*!
 * \brief Test 2: Construction from pointer and size
 */
static int test_pointer_size_construction() {
  int64_t data[5] = {10, 20, 30, 40, 50};

  tvm::dsp::Span<int64_t> span(data, 5);

  if (span.data() != data) return 0;
  if (span.size() != 5) return 0;
  if (span.empty()) return 0;

  /* Check all elements */
  if (span[0] != 10) return 0;
  if (span[1] != 20) return 0;
  if (span[2] != 30) return 0;
  if (span[3] != 40) return 0;
  if (span[4] != 50) return 0;

  return 1;
}

/*!
 * \brief Test 3: Construction from C array with size deduction
 */
static int test_array_construction() {
  int64_t dims[4] = {1, 3, 224, 224};

  /* Size should be automatically deduced as 4 */
  tvm::dsp::Span<int64_t> span(dims);

  if (span.size() != 4) return 0;
  if (span[0] != 1) return 0;
  if (span[1] != 3) return 0;
  if (span[2] != 224) return 0;
  if (span[3] != 224) return 0;

  return 1;
}

/*!
 * \brief Test 4: Element modification via operator[]
 */
static int test_element_modification() {
  int data[3] = {1, 2, 3};
  tvm::dsp::Span<int> span(data);

  /* Modify through span */
  span[0] = 100;
  span[1] = 200;
  span[2] = 300;

  /* Verify original array was modified */
  if (data[0] != 100) return 0;
  if (data[1] != 200) return 0;
  if (data[2] != 300) return 0;

  return 1;
}

/*!
 * \brief Test 5: front() and back() accessors
 */
static int test_front_back() {
  int data[4] = {1, 2, 3, 4};
  tvm::dsp::Span<int> span(data);

  if (span.front() != 1) return 0;
  if (span.back() != 4) return 0;

  /* Modify via front/back */
  span.front() = 10;
  span.back() = 40;

  if (data[0] != 10) return 0;
  if (data[3] != 40) return 0;

  return 1;
}

/*!
 * \brief Test 6: Range-based for loop iteration
 */
static int test_range_based_for() {
  int data[5] = {1, 2, 3, 4, 5};
  tvm::dsp::Span<int> span(data);

  /* Sum all elements using range-based for */
  int sum = 0;
  for (int x : span) {
    sum += x;
  }

  if (sum != 15) return 0;

  /* Count elements */
  int count = 0;
  for (int x : span) {
    (void)x;
    count++;
  }
  if (count != 5) return 0;

  return 1;
}

/*!
 * \brief Test 7: Subspan first()
 */
static int test_subspan_first() {
  int data[5] = {0, 1, 2, 3, 4};
  tvm::dsp::Span<int> span(data);

  auto first_three = span.first(3);

  if (first_three.size() != 3) return 0;
  if (first_three[0] != 0) return 0;
  if (first_three[1] != 1) return 0;
  if (first_three[2] != 2) return 0;

  /* first(0) should give empty span */
  auto empty = span.first(0);
  if (!empty.empty()) return 0;

  return 1;
}

/*!
 * \brief Test 8: Subspan last()
 */
static int test_subspan_last() {
  int data[5] = {0, 1, 2, 3, 4};
  tvm::dsp::Span<int> span(data);

  auto last_two = span.last(2);

  if (last_two.size() != 2) return 0;
  if (last_two[0] != 3) return 0;
  if (last_two[1] != 4) return 0;

  /* last(0) should give empty span */
  auto empty = span.last(0);
  if (!empty.empty()) return 0;

  return 1;
}

/*!
 * \brief Test 9: Subspan subspan()
 */
static int test_subspan_subspan() {
  int data[5] = {0, 1, 2, 3, 4};
  tvm::dsp::Span<int> span(data);

  /* Middle portion */
  auto mid = span.subspan(1, 3);
  if (mid.size() != 3) return 0;
  if (mid[0] != 1) return 0;
  if (mid[1] != 2) return 0;
  if (mid[2] != 3) return 0;

  /* From offset to end (no count specified) */
  auto rest = span.subspan(2);
  if (rest.size() != 3) return 0;
  if (rest[0] != 2) return 0;
  if (rest[1] != 3) return 0;
  if (rest[2] != 4) return 0;

  return 1;
}

/*!
 * \brief Test 10: Const span (read-only)
 */
static int test_const_span() {
  const int data[3] = {10, 20, 30};

  /* Create const span */
  tvm::dsp::Span<const int> span(data, 3);

  if (span.size() != 3) return 0;
  if (span[0] != 10) return 0;
  if (span[1] != 20) return 0;
  if (span[2] != 30) return 0;

  /* Iteration should work */
  int sum = 0;
  for (int x : span) {
    sum += x;
  }
  if (sum != 60) return 0;

  return 1;
}

/*!
 * \brief Test 11: MakeSpan helper function
 */
static int test_make_span() {
  int data[4] = {5, 10, 15, 20};

  /* From pointer + size */
  auto span1 = tvm::dsp::MakeSpan(data, 4);
  if (span1.size() != 4) return 0;
  if (span1[0] != 5) return 0;

  /* From array (size deduced) */
  auto span2 = tvm::dsp::MakeSpan(data);
  if (span2.size() != 4) return 0;

  return 1;
}

/*!
 * \brief Test 12: MakeConstSpan helper function
 */
static int test_make_const_span() {
  int data[3] = {1, 2, 3};

  /* MakeConstSpan creates read-only span */
  auto span = tvm::dsp::MakeConstSpan(data);

  if (span.size() != 3) return 0;
  if (span[0] != 1) return 0;

  /* Verify it's actually const (this is a compile-time check,
     but we can at least verify the values are accessible) */
  int sum = 0;
  for (int x : span) {
    sum += x;
  }
  if (sum != 6) return 0;

  return 1;
}

/*!
 * \brief Test 13: size_bytes()
 */
static int test_size_bytes() {
  int32_t data[4] = {1, 2, 3, 4};
  tvm::dsp::Span<int32_t> span(data);

  /* 4 elements * 4 bytes = 16 bytes */
  if (span.size_bytes() != 16) return 0;

  /* Different type */
  int64_t data64[3] = {1, 2, 3};
  tvm::dsp::Span<int64_t> span64(data64);

  /* 3 elements * 8 bytes = 24 bytes */
  if (span64.size_bytes() != 24) return 0;

  return 1;
}

/*!
 * \brief Test 14: Span as function parameter
 *
 * This tests the typical use case: passing spans to functions.
 */
static int64_t sum_shape(tvm::dsp::Span<const int64_t> shape) {
  int64_t sum = 0;
  for (int64_t dim : shape) {
    sum += dim;
  }
  return sum;
}

static int test_as_function_parameter() {
  int64_t dims[4] = {1, 3, 224, 224};

  /* Pass array directly - size deduced */
  int64_t sum1 = sum_shape(dims);
  if (sum1 != 452) return 0;

  /* Pass explicit span */
  tvm::dsp::Span<int64_t> span(dims, 4);
  int64_t sum2 = sum_shape(span);
  if (sum2 != 452) return 0;

  /* Pass subspan */
  int64_t sum3 = sum_shape(span.first(2));
  if (sum3 != 4) return 0;  /* 1 + 3 */

  return 1;
}

/*!
 * \brief Test 15: Span with DSP allocated memory
 */
static int test_with_dsp_memory() {
  /* Allocate memory from DSP pool */
  int64_t* data = static_cast<int64_t*>(
      tvm_dsp_alloc(4 * sizeof(int64_t), 8, TVM_DSP_MEM_MAIN));

  if (!data) {
    printf("(skipped - alloc failed) ");
    return 1;
  }

  /* Initialize */
  data[0] = 100;
  data[1] = 200;
  data[2] = 300;
  data[3] = 400;

  /* Create span over allocated memory */
  auto span = tvm::dsp::MakeSpan(data, 4);

  /* Verify access */
  int64_t sum = 0;
  for (int64_t x : span) {
    sum += x;
  }

  /* Clean up */
  tvm_dsp_free(data);

  return sum == 1000;
}

/*
 * =============================================================================
 * MAIN
 * =============================================================================
 */

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  /* Initialize platform for memory tests */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("=== Span Test Suite ===\n");
  printf("Testing non-owning array view\n\n");

  TEST(default_construction);
  TEST(pointer_size_construction);
  TEST(array_construction);
  TEST(element_modification);
  TEST(front_back);
  TEST(range_based_for);
  TEST(subspan_first);
  TEST(subspan_last);
  TEST(subspan_subspan);
  TEST(const_span);
  TEST(make_span);
  TEST(make_const_span);
  TEST(size_bytes);
  TEST(as_function_parameter);
  TEST(with_dsp_memory);

  printf("\n=== Results: %d/%d tests passed ===\n", g_tests_passed, g_tests_run);

  return (g_tests_passed == g_tests_run) ? 0 : 1;
}
