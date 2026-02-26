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
 * \file tests/test_result.cpp
 * \brief Tests for the Result error handling type
 *
 * Tests verify that:
 * - Ok() creates success results
 * - Err() creates error results
 * - IsOk()/IsErr() correctly identify state
 * - Value()/Error() access the correct data
 * - ValueOr()/ErrorOr() provide defaults
 * - Result<void, E> works correctly
 * - Copy construction and assignment work
 * - Boolean conversion works
 */

#include <cstdio>
#include <cstring>

/* Include the result header */
#include "result.h"

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

/* Custom error type for testing */
struct TestError {
  int code;
  const char* message;
};

/*
 * =============================================================================
 * TEST CASES
 * =============================================================================
 */

/*!
 * \brief Test 1: Ok() creates a success result
 */
static int test_ok_creation() {
  tvm::dsp::Result<int, TestError> result = tvm::dsp::Ok(42);

  if (!result.IsOk()) return 0;
  if (result.IsErr()) return 0;
  if (result.Value() != 42) return 0;

  return 1;
}

/*!
 * \brief Test 2: Err() creates an error result
 */
static int test_err_creation() {
  TestError err = {-1, "test error"};
  tvm::dsp::Result<int, TestError> result = tvm::dsp::Err(err);

  if (result.IsOk()) return 0;
  if (!result.IsErr()) return 0;
  if (result.Error().code != -1) return 0;
  if (strcmp(result.Error().message, "test error") != 0) return 0;

  return 1;
}

/*!
 * \brief Test 3: Value() returns the success value
 */
static int test_value_access() {
  tvm::dsp::Result<int64_t, int> result = tvm::dsp::Ok<int64_t>(12345678901234LL);

  if (result.Value() != 12345678901234LL) return 0;

  /* Modify through reference */
  result.Value() = 99;
  if (result.Value() != 99) return 0;

  return 1;
}

/*!
 * \brief Test 4: Error() returns the error value
 */
static int test_error_access() {
  TestError err = {42, "error message"};
  tvm::dsp::Result<int, TestError> result = tvm::dsp::Err(err);

  if (result.Error().code != 42) return 0;
  if (strcmp(result.Error().message, "error message") != 0) return 0;

  /* Modify through reference */
  result.Error().code = 100;
  if (result.Error().code != 100) return 0;

  return 1;
}

/*!
 * \brief Test 5: ValueOr() returns value on success
 */
static int test_value_or_success() {
  tvm::dsp::Result<int, int> result = tvm::dsp::Ok(42);

  int value = result.ValueOr(-1);
  if (value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 6: ValueOr() returns default on error
 */
static int test_value_or_error() {
  tvm::dsp::Result<int, int> result = tvm::dsp::Err(999);

  int value = result.ValueOr(-1);
  if (value != -1) return 0;

  return 1;
}

/*!
 * \brief Test 7: ErrorOr() returns error on failure
 */
static int test_error_or_failure() {
  tvm::dsp::Result<int, int> result = tvm::dsp::Err(42);

  int err = result.ErrorOr(-1);
  if (err != 42) return 0;

  return 1;
}

/*!
 * \brief Test 8: ErrorOr() returns default on success
 */
static int test_error_or_success() {
  tvm::dsp::Result<int, int> result = tvm::dsp::Ok(100);

  int err = result.ErrorOr(-1);
  if (err != -1) return 0;

  return 1;
}

/*!
 * \brief Test 9: Boolean conversion (success = true)
 */
static int test_bool_conversion_success() {
  tvm::dsp::Result<int, int> result = tvm::dsp::Ok(42);

  if (!result) return 0;  /* Should be true */
  if (!static_cast<bool>(result)) return 0;

  return 1;
}

/*!
 * \brief Test 10: Boolean conversion (error = false)
 */
static int test_bool_conversion_error() {
  tvm::dsp::Result<int, int> result = tvm::dsp::Err(-1);

  if (result) return 0;  /* Should be false */
  if (static_cast<bool>(result)) return 0;

  return 1;
}

/*!
 * \brief Test 11: Copy construction (success)
 */
static int test_copy_construction_success() {
  tvm::dsp::Result<int, int> r1 = tvm::dsp::Ok(42);
  tvm::dsp::Result<int, int> r2(r1);

  if (!r2.IsOk()) return 0;
  if (r2.Value() != 42) return 0;

  /* Modify r2 - r1 should be unchanged */
  r2.Value() = 100;
  if (r1.Value() != 42) return 0;

  return 1;
}

/*!
 * \brief Test 12: Copy construction (error)
 */
static int test_copy_construction_error() {
  tvm::dsp::Result<int, int> r1 = tvm::dsp::Err(-1);
  tvm::dsp::Result<int, int> r2(r1);

  if (!r2.IsErr()) return 0;
  if (r2.Error() != -1) return 0;

  return 1;
}

/*!
 * \brief Test 13: Copy assignment
 */
static int test_copy_assignment() {
  tvm::dsp::Result<int, int> r1 = tvm::dsp::Ok(42);
  tvm::dsp::Result<int, int> r2 = tvm::dsp::Err(-1);

  /* Assign success to error */
  r2 = r1;
  if (!r2.IsOk()) return 0;
  if (r2.Value() != 42) return 0;

  /* Self-assignment */
  r1 = r1;
  if (!r1.IsOk()) return 0;
  if (r1.Value() != 42) return 0;

  return 1;
}

/*!
 * \brief Test 14: Result<void, E> success
 */
static int test_void_result_success() {
  tvm::dsp::Result<void, int> result = tvm::dsp::Ok();

  if (!result.IsOk()) return 0;
  if (result.IsErr()) return 0;
  if (!result) return 0;

  return 1;
}

/*!
 * \brief Test 15: Result<void, E> error
 */
static int test_void_result_error() {
  tvm::dsp::Result<void, int> result = tvm::dsp::Err(42);

  if (result.IsOk()) return 0;
  if (!result.IsErr()) return 0;
  if (result.Error() != 42) return 0;
  if (result) return 0;

  return 1;
}

/*!
 * \brief Test 16: Result<void, E> copy
 */
static int test_void_result_copy() {
  tvm::dsp::Result<void, int> r1 = tvm::dsp::Err(42);
  tvm::dsp::Result<void, int> r2(r1);

  if (!r2.IsErr()) return 0;
  if (r2.Error() != 42) return 0;

  tvm::dsp::Result<void, int> r3 = tvm::dsp::Ok();
  r3 = r1;
  if (!r3.IsErr()) return 0;
  if (r3.Error() != 42) return 0;

  return 1;
}

/*!
 * \brief Test 17: Using ErrorCode enum
 */
static int test_error_code_enum() {
  using tvm::dsp::ErrorCode;

  tvm::dsp::Result<int, ErrorCode> result = tvm::dsp::Err(ErrorCode::kInvalidArgument);

  if (!result.IsErr()) return 0;
  if (result.Error() != ErrorCode::kInvalidArgument) return 0;

  return 1;
}

/*!
 * \brief Test 18: Using Error struct
 */
static int test_error_struct() {
  using tvm::dsp::Error;
  using tvm::dsp::ErrorCode;

  tvm::dsp::ErrorResult<int> result =
      tvm::dsp::Err(Error{ErrorCode::kOutOfMemory, "allocation failed"});

  if (!result.IsErr()) return 0;
  if (result.Error().code != ErrorCode::kOutOfMemory) return 0;
  if (strcmp(result.Error().message, "allocation failed") != 0) return 0;

  return 1;
}

/*!
 * \brief Test 19: Result in function return
 */
static tvm::dsp::Result<int, TestError> divide(int a, int b) {
  if (b == 0) {
    return tvm::dsp::Err(TestError{-1, "division by zero"});
  }
  return tvm::dsp::Ok(a / b);
}

static int test_function_return() {
  auto r1 = divide(10, 2);
  if (!r1.IsOk()) return 0;
  if (r1.Value() != 5) return 0;

  auto r2 = divide(10, 0);
  if (!r2.IsErr()) return 0;
  if (strcmp(r2.Error().message, "division by zero") != 0) return 0;

  return 1;
}

/*!
 * \brief Test 20: Error propagation pattern
 */
static tvm::dsp::Result<int, TestError> step1() {
  return tvm::dsp::Ok(10);
}

static tvm::dsp::Result<int, TestError> step2_fail() {
  return tvm::dsp::Err(TestError{-2, "step2 failed"});
}

static tvm::dsp::Result<int, TestError> chain_success() {
  auto r1 = step1();
  if (r1.IsErr()) return tvm::dsp::Err(r1.Error());

  /* Use result from step1 */
  return tvm::dsp::Ok(r1.Value() * 2);
}

static tvm::dsp::Result<int, TestError> chain_failure() {
  auto r1 = step1();
  if (r1.IsErr()) return tvm::dsp::Err(r1.Error());

  auto r2 = step2_fail();
  if (r2.IsErr()) return tvm::dsp::Err(r2.Error());

  return tvm::dsp::Ok(r1.Value() + r2.Value());
}

static int test_error_propagation() {
  auto r1 = chain_success();
  if (!r1.IsOk()) return 0;
  if (r1.Value() != 20) return 0;

  auto r2 = chain_failure();
  if (!r2.IsErr()) return 0;
  if (strcmp(r2.Error().message, "step2 failed") != 0) return 0;

  return 1;
}

/*!
 * \brief Test 21: Result with pointer value
 */
static int test_pointer_value() {
  int x = 42;
  tvm::dsp::Result<int*, int> result = tvm::dsp::Ok(&x);

  if (!result.IsOk()) return 0;
  if (*result.Value() != 42) return 0;

  /* Modify through pointer */
  *result.Value() = 100;
  if (x != 100) return 0;

  return 1;
}

/*!
 * \brief Test 22: Result with float value
 */
static int test_float_value() {
  tvm::dsp::Result<float, int> result = tvm::dsp::Ok(3.14f);

  if (!result.IsOk()) return 0;
  float val = result.Value();
  if (val < 3.13f || val > 3.15f) return 0;

  return 1;
}

/*!
 * \brief Test 23: Result with struct value
 */
struct Point {
  int x;
  int y;
};

static int test_struct_value() {
  Point p = {10, 20};
  tvm::dsp::Result<Point, int> result = tvm::dsp::Ok(p);

  if (!result.IsOk()) return 0;
  if (result.Value().x != 10) return 0;
  if (result.Value().y != 20) return 0;

  /* Modify through reference */
  result.Value().x = 100;
  if (result.Value().x != 100) return 0;

  return 1;
}

/*!
 * \brief Test 24: CodeResult alias
 */
static int test_code_result_alias() {
  using tvm::dsp::CodeResult;
  using tvm::dsp::ErrorCode;

  CodeResult<int> r1 = tvm::dsp::Ok(42);
  if (!r1.IsOk()) return 0;
  if (r1.Value() != 42) return 0;

  CodeResult<int> r2 = tvm::dsp::Err(ErrorCode::kNotFound);
  if (!r2.IsErr()) return 0;
  if (r2.Error() != ErrorCode::kNotFound) return 0;

  return 1;
}

/*!
 * \brief Test 25: Const Result access
 */
static int test_const_access() {
  const tvm::dsp::Result<int, int> r1 = tvm::dsp::Ok(42);
  const tvm::dsp::Result<int, int> r2 = tvm::dsp::Err(-1);

  /* Const value access */
  if (r1.Value() != 42) return 0;
  if (r1.ValueOr(-1) != 42) return 0;

  /* Const error access */
  if (r2.Error() != -1) return 0;
  if (r2.ErrorOr(0) != -1) return 0;

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

  printf("=== Result Test Suite ===\n");
  printf("Testing error handling without exceptions\n\n");

  TEST(ok_creation);
  TEST(err_creation);
  TEST(value_access);
  TEST(error_access);
  TEST(value_or_success);
  TEST(value_or_error);
  TEST(error_or_failure);
  TEST(error_or_success);
  TEST(bool_conversion_success);
  TEST(bool_conversion_error);
  TEST(copy_construction_success);
  TEST(copy_construction_error);
  TEST(copy_assignment);
  TEST(void_result_success);
  TEST(void_result_error);
  TEST(void_result_copy);
  TEST(error_code_enum);
  TEST(error_struct);
  TEST(function_return);
  TEST(error_propagation);
  TEST(pointer_value);
  TEST(float_value);
  TEST(struct_value);
  TEST(code_result_alias);
  TEST(const_access);

  printf("\n=== Results: %d/%d tests passed ===\n", g_tests_passed, g_tests_run);

  return (g_tests_passed == g_tests_run) ? 0 : 1;
}
