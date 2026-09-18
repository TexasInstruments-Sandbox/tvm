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
 * \file tests/test_storage.cpp
 * \brief Tests for TVMDSPStorageAllocNDArray bounds/overflow checks
 *
 * Regression coverage for the numel * elem_bytes overflow guard: the
 * product was originally computed in uint64_t and then compared against
 * SIZE_MAX, so a product that itself overflowed 64 bits wrapped back
 * around to a small value *before* the comparison ran, silently
 * defeating the check. A shape/dtype combination whose true byte size is
 * an exact multiple of 2^64 (e.g. 2^62 float32 elements = 2^64 bytes)
 * reproduces this: the wrapped size compares as "fits", and the
 * subsequent buffer-size check also passes because the wrapped value is
 * smaller than any real buffer.
 */

#include <cstdint>
#include <cstdio>

#include "storage.h"

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
    }                                        \
  } while (0)

namespace {

/*! \brief A stack-allocated storage backed by a small local buffer. */
struct TestStorage {
  uint8_t bytes[64];
  TVMDSPStorage storage;

  explicit TestStorage(size_t size) {
    storage.type_index = TVM_DSP_STORAGE_TYPE_INDEX;
    storage.ref_counter = 1;
    storage.deleter = nullptr;
    storage.buffer.data = bytes;
    storage.buffer.size = size;
    storage.buffer.device = DLDevice{kDLCPU, 0};
    storage.buffer.pool = 0;
  }
};

DLDataType MakeDType(uint8_t code, uint8_t bits, uint16_t lanes) {
  DLDataType dtype;
  dtype.code = code;
  dtype.bits = bits;
  dtype.lanes = lanes;
  return dtype;
}

}  // namespace

/*!
 * \brief Test 1: numel * elem_bytes wrapping past SIZE_MAX must be rejected.
 *
 * 2^62 float32 elements is exactly 2^64 bytes, which is 0 mod 2^64. Before
 * the fix, the overflow check computed this wrapped product and compared
 * it (as 0) to SIZE_MAX, so it never fired, and the request was accepted.
 */
static int test_overflow_wrap_rejected() {
  TestStorage ts(64);
  int64_t shape[1] = {static_cast<int64_t>(1) << 62};
  DLDataType dtype = MakeDType(kDLFloat, 32, 1);

  TVMDSPNDArray* arr = TVMDSPStorageAllocNDArray(&ts.storage, 0, shape, 1, dtype);
  if (arr != nullptr) {
    TVMDSPNDArrayDecRef(arr);
    return 0; /* should have been rejected */
  }
  return 1;
}

/*!
 * \brief Test 2: a huge (but non-wrapping) request is still rejected by the
 * ordinary buffer-size check, not silently accepted.
 */
static int test_large_non_wrapping_still_rejected() {
  TestStorage ts(64);
  int64_t shape[1] = {1LL << 40}; /* 2^40 * 4 bytes = 2^42 bytes, no wrap */
  DLDataType dtype = MakeDType(kDLFloat, 32, 1);

  TVMDSPNDArray* arr = TVMDSPStorageAllocNDArray(&ts.storage, 0, shape, 1, dtype);
  if (arr != nullptr) {
    TVMDSPNDArrayDecRef(arr);
    return 0; /* far larger than the 64-byte buffer; must be rejected */
  }
  return 1;
}

/*!
 * \brief Test 3: an ordinary in-bounds allocation must still succeed --
 * the overflow guard must not reject legitimate requests.
 */
static int test_normal_allocation_succeeds() {
  TestStorage ts(64);
  int64_t shape[1] = {4};
  DLDataType dtype = MakeDType(kDLFloat, 32, 1); /* 4 elems * 4 bytes = 16 */

  TVMDSPNDArray* arr = TVMDSPStorageAllocNDArray(&ts.storage, 0, shape, 1, dtype);
  if (arr == nullptr) return 0;
  int ok = (arr->data == ts.bytes) && (arr->ndim == 1);
  TVMDSPNDArrayDecRef(arr);
  return ok ? 1 : 0;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("=== Storage AllocNDArray Test Suite ===\n");
  printf("Testing numel * elem_bytes overflow guard\n\n");

  TEST(overflow_wrap_rejected);
  TEST(large_non_wrapping_still_rejected);
  TEST(normal_allocation_succeeds);

  printf("\n=== Results: %d/%d tests passed ===\n", g_tests_passed, g_tests_run);

  return (g_tests_passed == g_tests_run) ? 0 : 1;
}
