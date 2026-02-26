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
 * \file tests/test_fixed_vector.cpp
 * \brief Tests for the FixedVector fixed-capacity container
 *
 * Tests verify that:
 * - Default construction creates empty vector
 * - push_back adds elements correctly
 * - push_back returns false when full
 * - Element access works correctly
 * - Iterators enable range-based for loops
 * - Initializer list construction works
 * - Copy construction and assignment work
 * - AsSpan conversion works
 */

#include <cstdio>
#include <cstring>

/* Include the fixed_vector header */
#include "fixed_vector.h"

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
 * \brief Test 1: Default construction creates empty vector
 */
static int test_default_construction() {
  tvm::dsp::FixedVector<int, 8> vec;

  if (vec.size() != 0) return 0;
  if (vec.capacity() != 8) return 0;
  if (!vec.empty()) return 0;
  if (vec.full()) return 0;

  /* Iteration over empty vector should work (zero iterations) */
  int count = 0;
  for (int x : vec) {
    (void)x;
    count++;
  }
  if (count != 0) return 0;

  return 1;
}

/*!
 * \brief Test 2: push_back adds elements correctly
 */
static int test_push_back() {
  tvm::dsp::FixedVector<int, 4> vec;

  /* Add elements one by one */
  if (!vec.push_back(10)) return 0;
  if (!vec.push_back(20)) return 0;
  if (!vec.push_back(30)) return 0;

  /* Check size and contents */
  if (vec.size() != 3) return 0;
  if (vec[0] != 10) return 0;
  if (vec[1] != 20) return 0;
  if (vec[2] != 30) return 0;

  return 1;
}

/*!
 * \brief Test 3: push_back returns false when vector is full
 */
static int test_push_back_full() {
  tvm::dsp::FixedVector<int, 3> vec;

  /* Fill to capacity */
  if (!vec.push_back(1)) return 0;
  if (!vec.push_back(2)) return 0;
  if (!vec.push_back(3)) return 0;

  /* Vector should be full */
  if (!vec.full()) return 0;
  if (vec.size() != 3) return 0;

  /* Next push_back should fail */
  if (vec.push_back(4)) return 0;  /* Should return false */

  /* Size should still be 3 */
  if (vec.size() != 3) return 0;

  return 1;
}

/*!
 * \brief Test 4: Element modification via operator[]
 */
static int test_element_modification() {
  tvm::dsp::FixedVector<int, 4> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  /* Modify elements */
  vec[0] = 100;
  vec[1] = 200;
  vec[2] = 300;

  if (vec[0] != 100) return 0;
  if (vec[1] != 200) return 0;
  if (vec[2] != 300) return 0;

  return 1;
}

/*!
 * \brief Test 5: front() and back() accessors
 */
static int test_front_back() {
  tvm::dsp::FixedVector<int, 4> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);
  vec.push_back(4);

  if (vec.front() != 1) return 0;
  if (vec.back() != 4) return 0;

  /* Modify via front/back */
  vec.front() = 10;
  vec.back() = 40;

  if (vec[0] != 10) return 0;
  if (vec[3] != 40) return 0;

  return 1;
}

/*!
 * \brief Test 6: Range-based for loop iteration
 */
static int test_range_based_for() {
  tvm::dsp::FixedVector<int, 8> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);
  vec.push_back(4);
  vec.push_back(5);

  /* Sum all elements using range-based for */
  int sum = 0;
  for (int x : vec) {
    sum += x;
  }

  if (sum != 15) return 0;

  /* Count elements */
  int count = 0;
  for (int x : vec) {
    (void)x;
    count++;
  }
  if (count != 5) return 0;

  return 1;
}

/*!
 * \brief Test 7: Initializer list construction
 */
static int test_initializer_list() {
  tvm::dsp::FixedVector<int, 8> vec = {10, 20, 30, 40};

  if (vec.size() != 4) return 0;
  if (vec.capacity() != 8) return 0;
  if (vec[0] != 10) return 0;
  if (vec[1] != 20) return 0;
  if (vec[2] != 30) return 0;
  if (vec[3] != 40) return 0;

  return 1;
}

/*!
 * \brief Test 8: Initializer list truncation (more items than capacity)
 */
static int test_initializer_list_truncation() {
  /* Capacity is only 3, but we provide 5 elements */
  tvm::dsp::FixedVector<int, 3> vec = {1, 2, 3, 4, 5};

  /* Only first 3 should be stored */
  if (vec.size() != 3) return 0;
  if (vec[0] != 1) return 0;
  if (vec[1] != 2) return 0;
  if (vec[2] != 3) return 0;

  return 1;
}

/*!
 * \brief Test 9: pop_back removes elements
 */
static int test_pop_back() {
  tvm::dsp::FixedVector<int, 4> vec = {1, 2, 3, 4};

  if (vec.size() != 4) return 0;

  vec.pop_back();
  if (vec.size() != 3) return 0;
  if (vec.back() != 3) return 0;

  vec.pop_back();
  vec.pop_back();
  if (vec.size() != 1) return 0;
  if (vec.back() != 1) return 0;

  vec.pop_back();
  if (vec.size() != 0) return 0;
  if (!vec.empty()) return 0;

  return 1;
}

/*!
 * \brief Test 10: clear() removes all elements
 */
static int test_clear() {
  tvm::dsp::FixedVector<int, 8> vec = {1, 2, 3, 4, 5};

  if (vec.size() != 5) return 0;

  vec.clear();

  if (vec.size() != 0) return 0;
  if (!vec.empty()) return 0;
  if (vec.capacity() != 8) return 0;  /* Capacity unchanged */

  /* Should be able to add elements again */
  vec.push_back(100);
  if (vec.size() != 1) return 0;
  if (vec[0] != 100) return 0;

  return 1;
}

/*!
 * \brief Test 11: resize() changes size
 */
static int test_resize() {
  tvm::dsp::FixedVector<int, 8> vec = {1, 2, 3};

  /* Grow */
  if (!vec.resize(5)) return 0;
  if (vec.size() != 5) return 0;
  if (vec[0] != 1) return 0;
  if (vec[1] != 2) return 0;
  if (vec[2] != 3) return 0;
  /* New elements should be zero-initialized */
  if (vec[3] != 0) return 0;
  if (vec[4] != 0) return 0;

  /* Shrink */
  if (!vec.resize(2)) return 0;
  if (vec.size() != 2) return 0;
  if (vec[0] != 1) return 0;
  if (vec[1] != 2) return 0;

  /* Resize beyond capacity should fail */
  if (vec.resize(100)) return 0;  /* Should return false */
  if (vec.size() != 2) return 0;  /* Size unchanged */

  return 1;
}

/*!
 * \brief Test 12: resize() with fill value
 */
static int test_resize_with_value() {
  tvm::dsp::FixedVector<int, 8> vec = {1, 2};

  /* Grow with fill value */
  if (!vec.resize(5, 99)) return 0;
  if (vec.size() != 5) return 0;
  if (vec[0] != 1) return 0;
  if (vec[1] != 2) return 0;
  if (vec[2] != 99) return 0;
  if (vec[3] != 99) return 0;
  if (vec[4] != 99) return 0;

  return 1;
}

/*!
 * \brief Test 13: Copy construction
 */
static int test_copy_construction() {
  tvm::dsp::FixedVector<int, 4> vec1 = {10, 20, 30};

  /* Copy construct */
  tvm::dsp::FixedVector<int, 4> vec2(vec1);

  if (vec2.size() != 3) return 0;
  if (vec2[0] != 10) return 0;
  if (vec2[1] != 20) return 0;
  if (vec2[2] != 30) return 0;

  /* Modify vec2 - vec1 should be unaffected */
  vec2[0] = 100;
  if (vec1[0] != 10) return 0;

  return 1;
}

/*!
 * \brief Test 14: Copy assignment
 */
static int test_copy_assignment() {
  tvm::dsp::FixedVector<int, 4> vec1 = {1, 2, 3};
  tvm::dsp::FixedVector<int, 4> vec2 = {100, 200};

  /* Assign */
  vec2 = vec1;

  if (vec2.size() != 3) return 0;
  if (vec2[0] != 1) return 0;
  if (vec2[1] != 2) return 0;
  if (vec2[2] != 3) return 0;

  /* Self-assignment should work */
  vec1 = vec1;
  if (vec1.size() != 3) return 0;
  if (vec1[0] != 1) return 0;

  return 1;
}

/*!
 * \brief Test 15: AsSpan() conversion
 */
static int test_as_span() {
  tvm::dsp::FixedVector<int, 8> vec = {5, 10, 15, 20};

  /* Get mutable span */
  tvm::dsp::Span<int> span = vec.AsSpan();

  if (span.size() != 4) return 0;
  if (span[0] != 5) return 0;
  if (span[3] != 20) return 0;

  /* Modify through span */
  span[0] = 50;
  if (vec[0] != 50) return 0;  /* Original changed */

  return 1;
}

/*!
 * \brief Test 16: AsSpan() const conversion
 */
static int test_as_const_span() {
  const tvm::dsp::FixedVector<int, 8> vec = {1, 2, 3};

  /* Get const span from const vector */
  tvm::dsp::Span<const int> span = vec.AsSpan();

  if (span.size() != 3) return 0;
  if (span[0] != 1) return 0;

  /* Sum through span */
  int sum = 0;
  for (int x : span) {
    sum += x;
  }
  if (sum != 6) return 0;

  return 1;
}

/*!
 * \brief Test 17: data() returns pointer to underlying array
 */
static int test_data() {
  tvm::dsp::FixedVector<int, 4> vec = {1, 2, 3};

  int* ptr = vec.data();
  if (ptr[0] != 1) return 0;
  if (ptr[1] != 2) return 0;
  if (ptr[2] != 3) return 0;

  /* Modify through pointer */
  ptr[0] = 100;
  if (vec[0] != 100) return 0;

  return 1;
}

/*!
 * \brief Test 18: Works with int64_t (typical DSP shape type)
 */
static int test_int64_type() {
  tvm::dsp::FixedVector<int64_t, 8> shape;
  shape.push_back(1);
  shape.push_back(3);
  shape.push_back(224);
  shape.push_back(224);

  if (shape.size() != 4) return 0;

  /* Calculate total elements */
  int64_t total = 1;
  for (int64_t dim : shape) {
    total *= dim;
  }

  if (total != 1 * 3 * 224 * 224) return 0;

  return 1;
}

/*!
 * \brief Test 19: Works with float type
 */
static int test_float_type() {
  tvm::dsp::FixedVector<float, 4> vec = {1.5f, 2.5f, 3.5f};

  float sum = 0.0f;
  for (float f : vec) {
    sum += f;
  }

  /* 1.5 + 2.5 + 3.5 = 7.5 */
  if (sum < 7.4f || sum > 7.6f) return 0;

  return 1;
}

/*!
 * \brief Test 20: Works with pointers
 */
static int test_pointer_type() {
  int a = 10, b = 20, c = 30;

  tvm::dsp::FixedVector<int*, 4> vec;
  vec.push_back(&a);
  vec.push_back(&b);
  vec.push_back(&c);

  if (vec.size() != 3) return 0;
  if (*vec[0] != 10) return 0;
  if (*vec[1] != 20) return 0;
  if (*vec[2] != 30) return 0;

  /* Modify through pointers */
  *vec[0] = 100;
  if (a != 100) return 0;

  return 1;
}

/*
 * =============================================================================
 * MAIN
 * =============================================================================
 */

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  /* Initialize platform */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("=== FixedVector Test Suite ===\n");
  printf("Testing fixed-capacity container with no heap allocation\n\n");

  TEST(default_construction);
  TEST(push_back);
  TEST(push_back_full);
  TEST(element_modification);
  TEST(front_back);
  TEST(range_based_for);
  TEST(initializer_list);
  TEST(initializer_list_truncation);
  TEST(pop_back);
  TEST(clear);
  TEST(resize);
  TEST(resize_with_value);
  TEST(copy_construction);
  TEST(copy_assignment);
  TEST(as_span);
  TEST(as_const_span);
  TEST(data);
  TEST(int64_type);
  TEST(float_type);
  TEST(pointer_type);

  printf("\n=== Results: %d/%d tests passed ===\n", g_tests_passed, g_tests_run);

  return (g_tests_passed == g_tests_run) ? 0 : 1;
}
