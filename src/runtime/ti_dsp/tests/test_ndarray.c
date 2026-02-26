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
 * \file test_ndarray.c
 * \brief Test suite for TVM DSP Runtime NDArray and Shape containers
 */

#include "../container/ndarray.h"
#include "../container/shape.h"
#include "../platform/dsp_platform.h"
#include "../ffi/ffi_types.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) \
  do { printf("  Testing %s... ", name); test_count++; } while(0)

#define PASS() \
  do { printf("PASS\n"); pass_count++; } while(0)

#define FAIL(msg) \
  do { printf("FAIL: %s\n", msg); return -1; } while(0)

/* ---------------------------------------------------------------------------
 * Test: NDArray Memory Layout
 * ---------------------------------------------------------------------------*/
static int test_ndarray_layout(void) {
  TEST("NDArray memory layout");

  /* Verify that 'data' field is at offset 16 (sizeof(TVMFFIObject)) */
  size_t data_offset = offsetof(TVMDSPNDArray, data);
  size_t expected_offset = sizeof(TVMFFIObject);

  if (data_offset != expected_offset) {
    printf("FAIL: data at offset %zu, expected %zu\n", data_offset, expected_offset);
    return -1;
  }

  /* Verify TVMFFIObject size is 16 bytes */
  if (sizeof(TVMFFIObject) != 16) {
    printf("FAIL: sizeof(TVMFFIObject) = %zu, expected 16\n", sizeof(TVMFFIObject));
    return -1;
  }

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: NDArray Allocation
 * ---------------------------------------------------------------------------*/
static int test_ndarray_alloc(void) {
  TEST("NDArray allocation");

  int64_t shape[] = {2, 3, 4};
  DLDataType dtype = {kDLFloat, 32, 1};  /* float32 */
  DLDevice device = {kDLCPU, 0};

  TVMDSPNDArray* arr = TVMDSPNDArrayAlloc(shape, 3, dtype, device);
  if (arr == NULL) FAIL("allocation returned NULL");

  /* Check header */
  if (arr->type_index != kTVMFFITensor) FAIL("type_index wrong");
  if (arr->ref_counter != 1) FAIL("ref_counter wrong");
  if (arr->deleter == NULL) FAIL("deleter is NULL");

  /* Check DLTensor fields */
  if (arr->data == NULL) FAIL("data is NULL");
  if (arr->ndim != 3) FAIL("ndim wrong");
  if (arr->dtype.code != kDLFloat) FAIL("dtype.code wrong");
  if (arr->dtype.bits != 32) FAIL("dtype.bits wrong");
  if (arr->shape == NULL) FAIL("shape is NULL");
  if (arr->shape[0] != 2) FAIL("shape[0] wrong");
  if (arr->shape[1] != 3) FAIL("shape[1] wrong");
  if (arr->shape[2] != 4) FAIL("shape[2] wrong");
  if (arr->strides != NULL) FAIL("strides should be NULL (contiguous)");
  if (arr->byte_offset != 0) FAIL("byte_offset wrong");

  /* Check computed values */
  if (TVMDSPNDArrayNumElements(arr) != 24) FAIL("NumElements wrong");
  if (TVMDSPNDArrayDataSize(arr) != 96) FAIL("DataSize wrong");  /* 24 * 4 bytes */
  if (!TVMDSPNDArrayIsContiguous(arr)) FAIL("should be contiguous");

  TVMDSPNDArrayDecRef(arr);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: NDArray from Data
 * ---------------------------------------------------------------------------*/
static int test_ndarray_from_data(void) {
  TEST("NDArray from data");

  float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t shape[] = {2, 3};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};

  TVMDSPNDArray* arr = TVMDSPNDArrayFromData(data, shape, 2, dtype, device);
  if (arr == NULL) FAIL("FromData returned NULL");

  /* Check that data pointer points to our array */
  if (arr->data != data) FAIL("data pointer wrong");

  /* Check that flags indicate non-ownership */
  if (arr->flags & TVM_DSP_NDARRAY_OWNS_DATA) FAIL("should not own data");

  /* Verify data is accessible */
  float* arr_data = (float*)arr->data;
  if (arr_data[0] != 1.0f) FAIL("data[0] wrong");
  if (arr_data[5] != 6.0f) FAIL("data[5] wrong");

  TVMDSPNDArrayDecRef(arr);
  /* Original data should still be valid after decref */

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: NDArray DLTensor Accessor
 * ---------------------------------------------------------------------------*/
static int test_ndarray_dltensor(void) {
  TEST("NDArray DLTensor accessor");

  int64_t shape[] = {4, 4};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};

  TVMDSPNDArray* arr = TVMDSPNDArrayAlloc(shape, 2, dtype, device);
  if (arr == NULL) FAIL("allocation failed");

  /* Get DLTensor pointer using our accessor */
  DLTensor* tensor = TVMDSPNDArrayGetDLTensor(arr);

  /* Verify it points to the correct location */
  if ((char*)tensor != (char*)arr + sizeof(TVMFFIObject)) {
    FAIL("DLTensor pointer offset wrong");
  }

  /* Verify DLTensor fields match */
  if (tensor->data != arr->data) FAIL("tensor->data mismatch");
  if (tensor->ndim != arr->ndim) FAIL("tensor->ndim mismatch");
  if (tensor->dtype.code != arr->dtype.code) FAIL("tensor->dtype.code mismatch");

  /* Also verify using the TVM API function signature */
  /* TVMFFINDArrayGetDLTensorPtr from c_api.h does: (DLTensor*)((char*)obj + sizeof(TVMFFIObject)) */
  DLTensor* tensor2 = (DLTensor*)((char*)arr + sizeof(TVMFFIObject));
  if (tensor != tensor2) FAIL("accessor doesn't match c_api.h formula");

  TVMDSPNDArrayDecRef(arr);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: NDArray Copy Operations
 * ---------------------------------------------------------------------------*/
static int test_ndarray_copy(void) {
  TEST("NDArray copy operations");

  int64_t shape[] = {2, 3};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};

  /* Create source array */
  TVMDSPNDArray* src = TVMDSPNDArrayAlloc(shape, 2, dtype, device);
  if (src == NULL) FAIL("src allocation failed");

  /* Fill with test data */
  float* src_data = (float*)src->data;
  int i;
  for (i = 0; i < 6; i++) {
    src_data[i] = (float)(i + 1);
  }

  /* Create destination array */
  TVMDSPNDArray* dst = TVMDSPNDArrayAlloc(shape, 2, dtype, device);
  if (dst == NULL) FAIL("dst allocation failed");

  /* Copy data */
  if (TVMDSPNDArrayCopy(src, dst) != 0) FAIL("copy failed");

  /* Verify copy */
  float* dst_data = (float*)dst->data;
  for (i = 0; i < 6; i++) {
    if (dst_data[i] != src_data[i]) {
      printf("FAIL: dst[%d] = %f, expected %f\n", i, dst_data[i], src_data[i]);
      return -1;
    }
  }

  TVMDSPNDArrayDecRef(src);
  TVMDSPNDArrayDecRef(dst);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: NDArray Reference Counting
 * ---------------------------------------------------------------------------*/
static int test_ndarray_refcount(void) {
  TEST("NDArray reference counting");

  int64_t shape[] = {4};
  DLDataType dtype = {kDLInt, 32, 1};
  DLDevice device = {kDLCPU, 0};

  TVMDSPNDArray* arr = TVMDSPNDArrayAlloc(shape, 1, dtype, device);
  if (arr == NULL) FAIL("allocation failed");

  if (arr->ref_counter != 1) FAIL("initial refcount wrong");

  TVMDSPNDArrayIncRef(arr);
  if (arr->ref_counter != 2) FAIL("refcount after inc wrong");

  TVMDSPNDArrayDecRef(arr);
  if (arr->ref_counter != 1) FAIL("refcount after dec wrong");

  /* Final decref should free */
  TVMDSPNDArrayDecRef(arr);
  /* Can't check arr after this - it's freed */

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Shape Creation
 * ---------------------------------------------------------------------------*/
static int test_shape_create(void) {
  TEST("Shape creation");

  int64_t dims[] = {2, 3, 4, 5};
  TVMDSPShape* shape = TVMDSPShapeCreate(dims, 4);
  if (shape == NULL) FAIL("creation failed");

  /* Check header */
  if (shape->type_index != kTVMFFIShape) FAIL("type_index wrong");
  if (shape->ref_counter != 1) FAIL("ref_counter wrong");

  /* Check cell */
  if (shape->size != 4) FAIL("size wrong");
  if (shape->data == NULL) FAIL("data is NULL");

  /* Check values */
  if (TVMDSPShapeAt(shape, 0) != 2) FAIL("dim[0] wrong");
  if (TVMDSPShapeAt(shape, 1) != 3) FAIL("dim[1] wrong");
  if (TVMDSPShapeAt(shape, 2) != 4) FAIL("dim[2] wrong");
  if (TVMDSPShapeAt(shape, 3) != 5) FAIL("dim[3] wrong");

  /* Check product */
  if (TVMDSPShapeProduct(shape) != 120) FAIL("product wrong");

  TVMDSPShapeDecRef(shape);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Shape Cell Accessor
 * ---------------------------------------------------------------------------*/
static int test_shape_cell(void) {
  TEST("Shape cell accessor");

  int64_t dims[] = {10, 20};
  TVMDSPShape* shape = TVMDSPShapeCreate(dims, 2);
  if (shape == NULL) FAIL("creation failed");

  /* Get cell pointer */
  TVMFFIShapeCell* cell = TVMDSPShapeGetCell(shape);

  /* Verify it points to the correct location */
  if ((char*)cell != (char*)shape + sizeof(TVMFFIObject)) {
    FAIL("cell pointer offset wrong");
  }

  /* Verify cell fields */
  if (cell->data != shape->data) FAIL("cell->data mismatch");
  if (cell->size != shape->size) FAIL("cell->size mismatch");

  TVMDSPShapeDecRef(shape);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Empty Shape (Scalar)
 * ---------------------------------------------------------------------------*/
static int test_shape_empty(void) {
  TEST("Empty shape (scalar)");

  TVMDSPShape* shape = TVMDSPShapeCreateEmpty();
  if (shape == NULL) FAIL("creation failed");

  if (shape->size != 0) FAIL("size should be 0");
  if (TVMDSPShapeProduct(shape) != 1) FAIL("scalar product should be 1");

  TVMDSPShapeDecRef(shape);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Backend API - Workspace
 * ---------------------------------------------------------------------------*/
/* External declarations for backend API functions */
extern void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                      int dtype_code_hint, int dtype_bits_hint);
extern int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

static int test_backend_workspace(void) {
  TEST("Backend workspace allocation");

  /* Allocate workspace */
  void* workspace = TVMBackendAllocWorkspace(1, 0, 1024, 2, 32);
  if (workspace == NULL) FAIL("workspace allocation failed");

  /* Write to workspace to verify it's usable */
  memset(workspace, 0xAB, 1024);

  /* Free workspace */
  if (TVMBackendFreeWorkspace(1, 0, workspace) != 0) FAIL("workspace free failed");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Backend API - Parallel Launch (single-threaded)
 * ---------------------------------------------------------------------------*/
extern int TVMBackendParallelLaunch(int (*flambda)(int, void*, void*), void* cdata, int num_task);

static int g_parallel_sum = 0;

static int parallel_lambda(int task_id, void* penv, void* cdata) {
  (void)penv;
  int* base = (int*)cdata;
  g_parallel_sum += *base + task_id;
  return 0;
}

static int test_backend_parallel(void) {
  TEST("Backend parallel launch");

  g_parallel_sum = 0;
  int base = 10;

  /* Launch with 4 tasks */
  if (TVMBackendParallelLaunch(parallel_lambda, &base, 4) != 0) {
    FAIL("parallel launch failed");
  }

  /* Should compute: (10+0) + (10+1) + (10+2) + (10+3) = 46 */
  if (g_parallel_sum != 46) {
    printf("FAIL: parallel sum = %d, expected 46\n", g_parallel_sum);
    return -1;
  }

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Main Test Runner
 * ---------------------------------------------------------------------------*/
int main(void) {
  printf("\n=== TVM DSP Runtime: NDArray/Shape Tests ===\n\n");

  /* Initialize platform */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("Running tests:\n");

  /* NDArray tests */
  if (test_ndarray_layout() != 0) goto fail;
  if (test_ndarray_alloc() != 0) goto fail;
  if (test_ndarray_from_data() != 0) goto fail;
  if (test_ndarray_dltensor() != 0) goto fail;
  if (test_ndarray_copy() != 0) goto fail;
  if (test_ndarray_refcount() != 0) goto fail;

  /* Shape tests */
  if (test_shape_create() != 0) goto fail;
  if (test_shape_cell() != 0) goto fail;
  if (test_shape_empty() != 0) goto fail;

  /* Backend API tests */
  if (test_backend_workspace() != 0) goto fail;
  if (test_backend_parallel() != 0) goto fail;

  printf("\n=== Results: %d/%d tests passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 0;

fail:
  printf("\n=== TESTS FAILED: %d/%d passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 1;
}
