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
 * \file tests/test_scope_guard.cpp
 * \brief Tests for the ScopeGuard RAII utility
 *
 * Tests verify that:
 * - Cleanup runs on normal scope exit
 * - Cleanup runs on early return
 * - Dismiss() prevents cleanup
 * - Multiple guards execute in reverse order
 * - Move semantics work correctly
 */

#include <cstdio>

/* Include the scope guard header */
#include "scope_guard.h"

/* Include platform for DSP memory allocation tests */
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
 * \brief Test 1: Basic scope guard executes cleanup on scope exit
 *
 * This test verifies the fundamental behavior: when a scope guard goes
 * out of scope, its cleanup function is called.
 */
static int test_basic_cleanup() {
  int cleanup_called = 0;

  {
    /* Create a scope guard that sets cleanup_called to 1 */
    TVM_DSP_SCOPE_EXIT(cleanup_called = 1);

    /* At this point, cleanup hasn't run yet */
    if (cleanup_called != 0) return 0;
  }
  /* Guard went out of scope, cleanup should have run */

  return cleanup_called == 1;
}

/*!
 * \brief Test 2: Cleanup runs on early return
 *
 * This is the key use case: cleanup runs even if we return early.
 * We simulate this with a nested scope and a "goto end" pattern.
 */
static int test_early_return_cleanup() {
  int cleanup_called = 0;
  int early_exit = 1;  /* Simulate an error condition */

  {
    TVM_DSP_SCOPE_EXIT(cleanup_called = 1);

    if (early_exit) {
      /* Simulate early return - guard should still run cleanup */
      goto end_of_scope;
    }
    /* This code is skipped */
    cleanup_called = 999;
  end_of_scope:;
    /* Guard destructor runs here as we exit the scope */
  }

  return cleanup_called == 1;
}

/*!
 * \brief Test 3: Dismiss() prevents cleanup execution
 *
 * When you call Dismiss(), the cleanup should NOT run on destruction.
 * This is useful when ownership is successfully transferred.
 */
static int test_dismiss() {
  int cleanup_called = 0;

  {
    auto guard = tvm::dsp::MakeScopeGuard([&]() { cleanup_called = 1; });

    /* Verify guard is active */
    if (!guard.IsActive()) return 0;

    /* Dismiss the guard */
    guard.Dismiss();

    /* Verify guard is no longer active */
    if (guard.IsActive()) return 0;
  }
  /* Guard went out of scope but was dismissed */

  /* Cleanup should NOT have been called */
  return cleanup_called == 0;
}

/*!
 * \brief Test 4: Multiple guards execute in reverse order (LIFO)
 *
 * C++ destroys local variables in reverse order of construction.
 * So if we create guard1 then guard2, guard2's cleanup runs first.
 */
static int test_reverse_order() {
  int order[3] = {0, 0, 0};
  int index = 0;

  {
    TVM_DSP_SCOPE_EXIT(order[index++] = 1);  /* First created, last to run */
    TVM_DSP_SCOPE_EXIT(order[index++] = 2);  /* Second created */
    TVM_DSP_SCOPE_EXIT(order[index++] = 3);  /* Last created, first to run */
  }

  /* Check order: 3 should be first, then 2, then 1 */
  return order[0] == 3 && order[1] == 2 && order[2] == 1;
}

/*!
 * \brief Test 5: Move semantics - moved-from guard doesn't run cleanup
 *
 * When we move a guard, the source guard should be deactivated.
 * Only the destination guard should run cleanup.
 */
static int test_move_semantics() {
  int cleanup_count = 0;

  {
    auto guard1 = tvm::dsp::MakeScopeGuard([&]() { cleanup_count++; });

    /* Move guard1 to guard2 */
    auto guard2 = std::move(guard1);

    /* guard1 should be inactive after move */
    if (guard1.IsActive()) return 0;

    /* guard2 should be active */
    if (!guard2.IsActive()) return 0;
  }

  /* Cleanup should have run exactly once (from guard2) */
  return cleanup_count == 1;
}

/*!
 * \brief Test 6: Guard works with platform memory allocation
 *
 * Real-world usage test: allocate memory from DSP pool and ensure
 * it gets freed even on "error" paths.
 */
static int test_with_dsp_memory() {
  void* ptr = nullptr;
  int freed = 0;

  /* We can't easily track if tvm_dsp_free was called, so we use
   * a tracking variable instead */
  {
    /* Allocate from main memory pool */
    ptr = tvm_dsp_alloc(64, 4, TVM_DSP_MEM_MAIN);
    if (ptr == nullptr) {
      printf("(skipped - alloc failed) ");
      return 1;  /* Skip test if allocation fails */
    }

    /* Create guard to track that cleanup would run */
    TVM_DSP_SCOPE_EXIT({
      tvm_dsp_free(ptr);
      freed = 1;
    });

    /* Write something to verify memory is valid */
    static_cast<int*>(ptr)[0] = 42;
  }

  /* Verify cleanup ran (memory was freed) */
  return freed == 1;
}

/*!
 * \brief Test 7: Nested scopes with guards
 *
 * Guards in inner scopes should run when inner scope exits,
 * not when outer scope exits.
 */
static int test_nested_scopes() {
  int outer_cleanup = 0;
  int inner_cleanup = 0;
  int checkpoint = 0;

  {
    TVM_DSP_SCOPE_EXIT(outer_cleanup = checkpoint);
    checkpoint = 1;

    {
      TVM_DSP_SCOPE_EXIT(inner_cleanup = checkpoint);
      checkpoint = 2;
    }
    /* Inner scope ended - inner guard should have run */

    /* Verify inner cleanup ran with checkpoint=2 */
    if (inner_cleanup != 2) return 0;

    checkpoint = 3;
  }
  /* Outer scope ended - outer guard should have run */

  /* Outer cleanup should have run with checkpoint=3 */
  return outer_cleanup == 3;
}

/*!
 * \brief Test 8: Guard with captured reference modification
 *
 * The lambda captures by reference, so it can modify local variables.
 */
static int test_capture_modification() {
  int value = 10;

  {
    TVM_DSP_SCOPE_EXIT(value *= 2);  /* Should set value to 20 */
  }

  return value == 20;
}

/*!
 * \brief Test 9: Guard with multiple statements in cleanup
 *
 * The cleanup code can contain multiple statements.
 */
static int test_multi_statement_cleanup() {
  int a = 0;
  int b = 0;
  int c = 0;

  {
    TVM_DSP_SCOPE_EXIT({
      a = 1;
      b = 2;
      c = 3;
    });
  }

  return a == 1 && b == 2 && c == 3;
}

/*!
 * \brief Test 10: IsActive() correctly reports state
 *
 * Verify that IsActive() returns the correct value before and after Dismiss().
 */
static int test_is_active() {
  auto guard = tvm::dsp::MakeScopeGuard([]() {});

  /* Should be active after creation */
  if (!guard.IsActive()) return 0;

  guard.Dismiss();

  /* Should be inactive after dismiss */
  if (guard.IsActive()) return 0;

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

  /* Initialize platform for memory tests */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("=== ScopeGuard Test Suite ===\n");
  printf("Testing RAII scope guard for automatic resource cleanup\n\n");

  TEST(basic_cleanup);
  TEST(early_return_cleanup);
  TEST(dismiss);
  TEST(reverse_order);
  TEST(move_semantics);
  TEST(with_dsp_memory);
  TEST(nested_scopes);
  TEST(capture_modification);
  TEST(multi_statement_cleanup);
  TEST(is_active);

  printf("\n=== Results: %d/%d tests passed ===\n", g_tests_passed, g_tests_run);

  return (g_tests_passed == g_tests_run) ? 0 : 1;
}
