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
 * \file tests/test_typed_handle.cpp
 * \brief Tests for the TypedHandle type-safe pointer wrapper
 *
 * Tests verify that:
 * - Default construction creates null handle
 * - Construction from pointer works
 * - Null checking works
 * - Pointer access via get(), ->, * works
 * - Casting between handle types works
 * - FFIHandle type checking works
 * - Comparison operators work
 */

#include <cstdio>
#include <cstring>

/* Include the typed_handle header */
#include "typed_handle.h"
#include "ffi_types.h"

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

/* Test struct for handle testing */
struct TestObject {
  int value;
  const char* name;
};

/* Derived test struct */
struct DerivedObject {
  TestObject base;
  float extra;
};

/*
 * =============================================================================
 * TEST CASES
 * =============================================================================
 */

/*!
 * \brief Test 1: Default construction creates null handle
 */
static int test_default_construction() {
  tvm::dsp::TypedHandle<TestObject> handle;

  if (!handle.IsNull()) return 0;
  if (handle.IsValid()) return 0;
  if (handle.get() != nullptr) return 0;
  if (handle) return 0;  /* Should be false */

  return 1;
}

/*!
 * \brief Test 2: Construction from nullptr
 */
static int test_nullptr_construction() {
  tvm::dsp::TypedHandle<TestObject> handle(nullptr);

  if (!handle.IsNull()) return 0;
  if (handle != nullptr) return 0;

  return 1;
}

/*!
 * \brief Test 3: Construction from pointer
 */
static int test_pointer_construction() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> handle(&obj);

  if (handle.IsNull()) return 0;
  if (!handle.IsValid()) return 0;
  if (handle.get() != &obj) return 0;
  if (!handle) return 0;  /* Should be true */

  return 1;
}

/*!
 * \brief Test 4: Arrow operator for member access
 */
static int test_arrow_operator() {
  TestObject obj = {42, "hello"};
  tvm::dsp::TypedHandle<TestObject> handle(&obj);

  if (handle->value != 42) return 0;
  if (strcmp(handle->name, "hello") != 0) return 0;

  /* Modify through handle */
  handle->value = 100;
  if (obj.value != 100) return 0;

  return 1;
}

/*!
 * \brief Test 5: Dereference operator
 */
static int test_dereference_operator() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> handle(&obj);

  TestObject& ref = *handle;
  if (ref.value != 42) return 0;

  /* Modify through reference */
  ref.value = 200;
  if (obj.value != 200) return 0;

  return 1;
}

/*!
 * \brief Test 6: Copy construction
 */
static int test_copy_construction() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> h1(&obj);
  tvm::dsp::TypedHandle<TestObject> h2(h1);

  /* Both should point to same object */
  if (h1.get() != h2.get()) return 0;
  if (h2->value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 7: Copy assignment
 */
static int test_copy_assignment() {
  TestObject obj1 = {42, "first"};
  TestObject obj2 = {99, "second"};

  tvm::dsp::TypedHandle<TestObject> h1(&obj1);
  tvm::dsp::TypedHandle<TestObject> h2(&obj2);

  h2 = h1;
  if (h2.get() != &obj1) return 0;
  if (h2->value != 42) return 0;

  /* Self assignment */
  h1 = h1;
  if (h1.get() != &obj1) return 0;

  return 1;
}

/*!
 * \brief Test 8: Nullptr assignment
 */
static int test_nullptr_assignment() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> handle(&obj);

  handle = nullptr;
  if (!handle.IsNull()) return 0;

  return 1;
}

/*!
 * \brief Test 9: FromRaw factory method
 */
static int test_from_raw() {
  TestObject obj = {42, "test"};
  void* raw = &obj;

  auto handle = tvm::dsp::TypedHandle<TestObject>::FromRaw(raw);
  if (handle.get() != &obj) return 0;
  if (handle->value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 10: Cast to another type
 */
static int test_cast() {
  DerivedObject derived;
  derived.base.value = 42;
  derived.base.name = "derived";
  derived.extra = 3.14f;

  tvm::dsp::TypedHandle<DerivedObject> derived_handle(&derived);

  /* Cast to base type */
  auto base_handle = derived_handle.Cast<TestObject>();
  if (base_handle->value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 11: ToVoid conversion
 */
static int test_to_void() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> handle(&obj);

  auto void_handle = handle.ToVoid();
  if (void_handle.get() != &obj) return 0;

  /* Cast back */
  auto back = void_handle.Cast<TestObject>();
  if (back->value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 12: Equality comparison
 */
static int test_equality() {
  TestObject obj1 = {42, "first"};
  TestObject obj2 = {99, "second"};

  tvm::dsp::TypedHandle<TestObject> h1(&obj1);
  tvm::dsp::TypedHandle<TestObject> h2(&obj1);  /* Same as h1 */
  tvm::dsp::TypedHandle<TestObject> h3(&obj2);
  tvm::dsp::TypedHandle<TestObject> h_null;

  if (!(h1 == h2)) return 0;
  if (h1 != h2) return 0;
  if (h1 == h3) return 0;
  if (!(h1 != h3)) return 0;
  if (!(h_null == nullptr)) return 0;
  if (h1 == nullptr) return 0;

  return 1;
}

/*!
 * \brief Test 13: Release
 */
static int test_release() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> handle(&obj);

  TestObject* released = handle.Release();
  if (released != &obj) return 0;
  if (!handle.IsNull()) return 0;

  return 1;
}

/*!
 * \brief Test 14: Reset
 */
static int test_reset() {
  TestObject obj1 = {42, "first"};
  TestObject obj2 = {99, "second"};

  tvm::dsp::TypedHandle<TestObject> handle(&obj1);
  handle.Reset(&obj2);
  if (handle.get() != &obj2) return 0;

  handle.Reset();  /* Reset to null */
  if (!handle.IsNull()) return 0;

  return 1;
}

/*!
 * \brief Test 15: MakeHandle helper
 */
static int test_make_handle() {
  TestObject obj = {42, "test"};

  auto handle = tvm::dsp::MakeHandle(&obj);
  if (handle->value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 16: NullHandle helper
 */
static int test_null_handle() {
  auto handle = tvm::dsp::NullHandle<TestObject>();
  if (!handle.IsNull()) return 0;

  return 1;
}

/*!
 * \brief Test 17: CheckHandle helper
 */
static int test_check_handle() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<TestObject> valid_handle(&obj);
  tvm::dsp::TypedHandle<TestObject> null_handle;

  tvm::dsp::HandleError error;

  /* Valid handle */
  if (!tvm::dsp::CheckHandle(valid_handle, &error)) return 0;

  /* Null handle */
  if (tvm::dsp::CheckHandle(null_handle, &error)) return 0;
  if (error != tvm::dsp::HandleError::kNullHandle) return 0;

  return 1;
}

/*!
 * \brief Test 18: TypedHandle<void> specialization
 */
static int test_void_handle() {
  TestObject obj = {42, "test"};
  tvm::dsp::TypedHandle<void> handle(&obj);

  if (handle.IsNull()) return 0;
  if (handle.get() != &obj) return 0;

  /* Cast to typed */
  auto typed = handle.Cast<TestObject>();
  if (typed->value != 42) return 0;

  return 1;
}

/*!
 * \brief Test 19: FFIHandle basic operations
 */
static int test_ffi_handle_basic() {
  /* Create a mock FFI object on stack */
  struct {
    TVMFFIObject header;
    int data;
  } mock_obj;

  mock_obj.header.type_index = kTVMFFITensor;
  mock_obj.header.ref_counter = 1;
  mock_obj.header.deleter = nullptr;
  mock_obj.data = 42;

  tvm::dsp::FFIHandle<TVMFFIObject> handle(&mock_obj.header);

  if (handle.IsNull()) return 0;
  if (handle.GetTypeIndex() != kTVMFFITensor) return 0;
  if (!handle.HasType(kTVMFFITensor)) return 0;
  if (handle.HasType(kTVMFFIShape)) return 0;

  return 1;
}

/*!
 * \brief Test 20: FFIHandle FromRawChecked success
 */
static int test_ffi_handle_checked_success() {
  struct {
    TVMFFIObject header;
    int data;
  } mock_obj;

  mock_obj.header.type_index = kTVMFFITensor;
  mock_obj.header.ref_counter = 1;
  mock_obj.header.deleter = nullptr;
  mock_obj.data = 42;

  tvm::dsp::FFIHandle<TVMFFIObject> handle;
  auto error = tvm::dsp::FFIHandle<TVMFFIObject>::FromRawChecked(
      &mock_obj, kTVMFFITensor, &handle);

  /* Success returns 0 cast to HandleError */
  if (static_cast<int>(error) != 0) return 0;
  if (handle.GetTypeIndex() != kTVMFFITensor) return 0;

  return 1;
}

/*!
 * \brief Test 21: FFIHandle FromRawChecked type mismatch
 */
static int test_ffi_handle_checked_type_mismatch() {
  struct {
    TVMFFIObject header;
    int data;
  } mock_obj;

  mock_obj.header.type_index = kTVMFFITensor;
  mock_obj.header.ref_counter = 1;
  mock_obj.header.deleter = nullptr;
  mock_obj.data = 42;

  /* Try to cast as Shape when it's actually NDArray */
  tvm::dsp::FFIHandle<TVMFFIObject> handle;
  auto error = tvm::dsp::FFIHandle<TVMFFIObject>::FromRawChecked(
      &mock_obj, kTVMFFIShape, &handle);

  if (error != tvm::dsp::HandleError::kTypeMismatch) return 0;

  return 1;
}

/*!
 * \brief Test 22: FFIHandle FromRawChecked null
 */
static int test_ffi_handle_checked_null() {
  tvm::dsp::FFIHandle<TVMFFIObject> handle;
  auto error = tvm::dsp::FFIHandle<TVMFFIObject>::FromRawChecked(
      nullptr, kTVMFFITensor, &handle);

  if (error != tvm::dsp::HandleError::kNullHandle) return 0;

  return 1;
}

/*!
 * \brief Test 23: Const correctness
 */
static int test_const_correctness() {
  const TestObject obj = {42, "const"};
  tvm::dsp::TypedHandle<const TestObject> handle(&obj);

  if (handle->value != 42) return 0;
  if (strcmp(handle->name, "const") != 0) return 0;

  return 1;
}

/*!
 * \brief Test 24: Handle with DSP allocated memory
 */
static int test_with_dsp_memory() {
  /* Allocate from DSP pool */
  TestObject* ptr = static_cast<TestObject*>(
      tvm_dsp_alloc(sizeof(TestObject), 8, TVM_DSP_MEM_MAIN));

  if (!ptr) {
    printf("(skipped - alloc failed) ");
    return 1;
  }

  ptr->value = 42;
  ptr->name = "allocated";

  tvm::dsp::TypedHandle<TestObject> handle(ptr);
  if (handle->value != 42) {
    tvm_dsp_free(ptr);
    return 0;
  }

  tvm_dsp_free(ptr);
  return 1;
}

/*!
 * \brief Test 25: Multiple handles to same object
 */
static int test_multiple_handles() {
  TestObject obj = {42, "shared"};

  tvm::dsp::TypedHandle<TestObject> h1(&obj);
  tvm::dsp::TypedHandle<TestObject> h2(&obj);
  tvm::dsp::TypedHandle<TestObject> h3(&obj);

  /* All should point to same object */
  if (h1.get() != h2.get()) return 0;
  if (h2.get() != h3.get()) return 0;

  /* Modify through one handle */
  h1->value = 100;

  /* All should see the change */
  if (h2->value != 100) return 0;
  if (h3->value != 100) return 0;

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

  printf("=== TypedHandle Test Suite ===\n");
  printf("Testing type-safe pointer wrapper\n\n");

  TEST(default_construction);
  TEST(nullptr_construction);
  TEST(pointer_construction);
  TEST(arrow_operator);
  TEST(dereference_operator);
  TEST(copy_construction);
  TEST(copy_assignment);
  TEST(nullptr_assignment);
  TEST(from_raw);
  TEST(cast);
  TEST(to_void);
  TEST(equality);
  TEST(release);
  TEST(reset);
  TEST(make_handle);
  TEST(null_handle);
  TEST(check_handle);
  TEST(void_handle);
  TEST(ffi_handle_basic);
  TEST(ffi_handle_checked_success);
  TEST(ffi_handle_checked_type_mismatch);
  TEST(ffi_handle_checked_null);
  TEST(const_correctness);
  TEST(with_dsp_memory);
  TEST(multiple_handles);

  printf("\n=== Results: %d/%d tests passed ===\n", g_tests_passed, g_tests_run);

  return (g_tests_passed == g_tests_run) ? 0 : 1;
}
