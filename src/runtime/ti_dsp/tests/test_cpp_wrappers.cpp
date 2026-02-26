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
 * \file tests/test_cpp_wrappers.cpp
 * \brief Tests for Phase 6 C++ compatibility wrappers
 *
 * Tests the C++ header-only wrappers that provide compatibility
 * with TVM-generated code.
 */

#include <cstdio>
#include <cstring>

/* Include the C++ wrappers */
#include "c_runtime_api.h"

/* For testing, also include the C runtime directly */
extern "C" {
#include "dsp_platform.h"
#include "vm_builtins.h"
#include "registry.h"
}

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("Test: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASSED\n"); \
        tests_passed++; \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

/* Test 1: ObjectRef construction and destruction */
static int test_object_ref_lifecycle() {
    /* ObjectRef with null data */
    tvm::runtime::ObjectRef ref1;
    if (ref1.defined()) return 0;
    if (ref1 != nullptr) return 0;

    /* Test nullptr assignment */
    ref1 = nullptr;
    if (ref1.defined()) return 0;

    return 1;
}

/* Test 2: Any type construction */
static int test_any_construction() {
    /* Default Any is None */
    tvm::ffi::Any a1;
    if (!a1.IsNone()) return 0;
    if (a1.type_index() != kTVMFFINone) return 0;

    /* Integer construction */
    tvm::ffi::Any a2(static_cast<int64_t>(42));
    if (a2.type_index() != kTVMFFIInt) return 0;
    if (a2.AsInt() != 42) return 0;

    /* Float construction */
    tvm::ffi::Any a3(3.14159);
    if (a3.type_index() != kTVMFFIFloat) return 0;
    /* Check approximate equality */
    double diff = a3.AsFloat() - 3.14159;
    if (diff < -0.0001 || diff > 0.0001) return 0;

    /* Nullptr construction */
    tvm::ffi::Any a4(nullptr);
    if (!a4.IsNone()) return 0;

    return 1;
}

/* Test 3: Any copy semantics */
static int test_any_copy() {
    tvm::ffi::Any original(static_cast<int64_t>(100));
    tvm::ffi::Any copy = original;

    if (copy.AsInt() != 100) return 0;
    if (copy.type_index() != kTVMFFIInt) return 0;

    /* Original should be unchanged */
    if (original.AsInt() != 100) return 0;

    return 1;
}

/* Test 4: Any move semantics */
static int test_any_move() {
    tvm::ffi::Any source(static_cast<int64_t>(200));
    tvm::ffi::Any dest = std::move(source);

    /* Dest should have the value */
    if (dest.AsInt() != 200) return 0;

    /* Source should be None after move */
    if (!source.IsNone()) return 0;

    return 1;
}

/* Test 5: Any assignment operators */
static int test_any_assignment() {
    tvm::ffi::Any a;

    /* Assignment from copy */
    tvm::ffi::Any b(static_cast<int64_t>(50));
    a = b;
    if (a.AsInt() != 50) return 0;

    /* Assignment from move */
    tvm::ffi::Any c(static_cast<int64_t>(75));
    a = std::move(c);
    if (a.AsInt() != 75) return 0;
    if (!c.IsNone()) return 0;

    /* Nullptr assignment */
    a = nullptr;
    if (!a.IsNone()) return 0;

    return 1;
}

/* Test 6: AnyUnsafe MoveTVMFFIAnyToAny */
static int test_any_unsafe_move() {
    TVMFFIAny raw;
    raw.type_index = kTVMFFIInt;
    raw.small_len = 0;
    raw.v_int64 = 12345;

    tvm::ffi::Any result = tvm::ffi::details::AnyUnsafe::MoveTVMFFIAnyToAny(std::move(raw));

    if (result.type_index() != kTVMFFIInt) return 0;
    if (result.AsInt() != 12345) return 0;

    /* Raw should be cleared */
    if (raw.type_index != kTVMFFINone) return 0;

    return 1;
}

/* Test 7: NDArray wrapper (empty array) */
static int test_ndarray_wrapper() {
    /* Create an empty NDArray */
    tvm::runtime::NDArray arr;
    if (arr.defined()) return 0;

    /* Test Size() on null array */
    if (arr.Size() != 0) return 0;
    if (arr.DataSize() != 0) return 0;
    if (arr.ndim() != 0) return 0;

    return 1;
}

/* Test 8: NDArray::Empty static method */
static int test_ndarray_empty() {
    int64_t shape[] = {2, 3};
    DLDataType dtype = {kDLFloat, 32, 1};  /* float32 */
    DLDevice device = {kDLCPU, 0};

    tvm::runtime::NDArray arr = tvm::runtime::NDArray::Empty(shape, 2, dtype, device);

    if (!arr.defined()) return 0;
    if (arr.ndim() != 2) return 0;
    if (arr.shape()[0] != 2) return 0;
    if (arr.shape()[1] != 3) return 0;
    if (arr.Size() != 6) return 0;
    if (arr.dtype().code != kDLFloat) return 0;
    if (arr.dtype().bits != 32) return 0;
    if (arr.data() == nullptr) return 0;

    return 1;
}

/* Test 9: PackedFunc wrapper (null function) */
static int test_packed_func_wrapper() {
    tvm::runtime::PackedFunc func;
    if (func.defined()) return 0;
    if (func != nullptr) return 0;

    return 1;
}

/* Test 10: GetGlobalFunc */
static int test_get_global_func() {
    /* First register builtins */
    TVMDSPRegisterVMBuiltins();

    /* Look up a registered function */
    tvm::runtime::PackedFunc func = tvm::runtime::GetGlobalFunc("vm.builtin.alloc_storage");

    if (!func.defined()) return 0;

    return 1;
}

/* Test 11: Type aliases */
static int test_type_aliases() {
    tvm::runtime::Device dev = {kDLCPU, 0};
    if (dev.device_type != kDLCPU) return 0;

    DLDataType dtype = {kDLFloat, 32, 1};
    if (dtype.code != kDLFloat) return 0;

    return 1;
}

/* Test 12: C API functions via C++ */
static int test_c_api_functions() {
    /* Test TVMBackendGetFuncFromGlobalRegistry */
    void* handle = nullptr;
    int ret = TVMBackendGetFuncFromGlobalRegistry("vm.builtin.alloc_storage", &handle);
    if (ret != 0) return 0;
    if (handle == nullptr) return 0;

    return 1;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    /* Initialize platform */
    if (tvm_dsp_platform_init() != 0) {
        printf("FATAL: Platform initialization failed\n");
        return 1;
    }

    printf("=== Phase 6: C++ Wrappers Test Suite ===\n");
    printf("Testing C++ compatibility layer for TVM DSP Runtime\n\n");

    TEST(object_ref_lifecycle);
    TEST(any_construction);
    TEST(any_copy);
    TEST(any_move);
    TEST(any_assignment);
    TEST(any_unsafe_move);
    TEST(ndarray_wrapper);
    TEST(ndarray_empty);
    TEST(packed_func_wrapper);
    TEST(get_global_func);
    TEST(type_aliases);
    TEST(c_api_functions);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
