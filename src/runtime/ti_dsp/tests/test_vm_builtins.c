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
 * \file test_vm_builtins.c
 * \brief Test suite for TVM DSP Runtime VM builtins (Phase 4)
 */

#include "../vm/vm_builtins.h"
#include "../vm/storage.h"
#include "../container/ndarray.h"
#include "../container/shape.h"
#include "../platform/dsp_platform.h"
#include "../ffi/ffi_types.h"
#include <stdio.h>
#include <string.h>

/* External function declarations */
extern int TVMBackendGetFuncFromGlobalRegistry(const char* func_name, TVMFFIObjectHandle* out);

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) \
  do { printf("  Testing %s... ", name); test_count++; } while(0)

#define PASS() \
  do { printf("PASS\n"); pass_count++; } while(0)

#define FAIL(msg) \
  do { printf("FAIL: %s\n", msg); return -1; } while(0)

/* ---------------------------------------------------------------------------
 * Test: Storage Allocation
 * ---------------------------------------------------------------------------*/
static int test_storage_alloc(void) {
  TVMDSPStorage* storage;
  DLDataType dtype;

  TEST("Storage allocation");

  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;

  storage = TVMDSPBuiltinAllocStorage(1024, 0, dtype);
  if (storage == NULL) FAIL("storage allocation returned NULL");

  /* Verify storage fields */
  if (storage->type_index != TVM_DSP_STORAGE_TYPE_INDEX) FAIL("type_index wrong");
  if (storage->ref_counter != 1) FAIL("ref_counter wrong");
  if (storage->buffer.data == NULL) FAIL("buffer.data is NULL");
  if (storage->buffer.size != 1024) FAIL("buffer.size wrong");

  TVMDSPStorageDecRef(storage);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Tensor Allocation from Storage
 * ---------------------------------------------------------------------------*/
static int test_tensor_from_storage(void) {
  TVMDSPStorage* storage;
  TVMDSPNDArray* tensor;
  DLDataType dtype;
  int64_t shape[] = {4, 8};

  TEST("Tensor allocation from storage");

  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;

  /* Allocate storage for 32 floats = 128 bytes */
  storage = TVMDSPBuiltinAllocStorage(128, 0, dtype);
  if (storage == NULL) FAIL("storage allocation failed");

  /* Allocate tensor from storage */
  tensor = TVMDSPBuiltinAllocTensor(storage, 0, shape, 2, dtype);
  if (tensor == NULL) FAIL("tensor allocation failed");

  /* Verify tensor */
  if (tensor->ndim != 2) FAIL("ndim wrong");
  if (tensor->shape[0] != 4) FAIL("shape[0] wrong");
  if (tensor->shape[1] != 8) FAIL("shape[1] wrong");
  if (tensor->dtype.code != kDLFloat) FAIL("dtype.code wrong");
  if (tensor->dtype.bits != 32) FAIL("dtype.bits wrong");

  /* Tensor data should point into storage */
  if (tensor->data != storage->buffer.data) FAIL("tensor data mismatch");

  TVMDSPNDArrayDecRef(tensor);
  TVMDSPStorageDecRef(storage);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Tensor at Offset
 * ---------------------------------------------------------------------------*/
static int test_tensor_offset(void) {
  TVMDSPStorage* storage;
  TVMDSPNDArray* tensor1;
  TVMDSPNDArray* tensor2;
  DLDataType dtype;
  int64_t shape1[] = {4};
  int64_t shape2[] = {4};

  TEST("Tensor allocation at offset");

  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;

  /* Allocate storage for 8 floats = 32 bytes */
  storage = TVMDSPBuiltinAllocStorage(32, 0, dtype);
  if (storage == NULL) FAIL("storage allocation failed");

  /* Allocate first tensor at offset 0 */
  tensor1 = TVMDSPBuiltinAllocTensor(storage, 0, shape1, 1, dtype);
  if (tensor1 == NULL) FAIL("tensor1 allocation failed");

  /* Allocate second tensor at offset 16 (4 floats) */
  tensor2 = TVMDSPBuiltinAllocTensor(storage, 16, shape2, 1, dtype);
  if (tensor2 == NULL) FAIL("tensor2 allocation failed");

  /* Verify tensors are at different offsets */
  if (tensor1->data == tensor2->data) FAIL("tensors have same data ptr");
  if ((char*)tensor2->data - (char*)tensor1->data != 16) {
    FAIL("tensor2 offset wrong");
  }

  TVMDSPNDArrayDecRef(tensor1);
  TVMDSPNDArrayDecRef(tensor2);
  TVMDSPStorageDecRef(storage);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Shape Heap Allocation
 * ---------------------------------------------------------------------------*/
static int test_shape_heap(void) {
  TVMDSPNDArray* heap;

  TEST("Shape heap allocation");

  heap = TVMDSPBuiltinAllocShapeHeap(16);
  if (heap == NULL) FAIL("heap allocation failed");

  /* Verify heap */
  if (heap->ndim != 1) FAIL("heap ndim wrong");
  if (heap->shape[0] != 16) FAIL("heap size wrong");
  if (heap->dtype.code != kDLInt) FAIL("heap dtype wrong");
  if (heap->dtype.bits != 64) FAIL("heap bits wrong");

  TVMDSPNDArrayDecRef(heap);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Make Shape with Immediate Values
 * ---------------------------------------------------------------------------*/
static int test_make_shape_imm(void) {
  TVMDSPShape* shape;
  int32_t codes[] = {kMakeShapeUseImm, kMakeShapeUseImm, kMakeShapeUseImm};
  int64_t values[] = {2, 3, 4};

  TEST("Make shape with immediate values");

  shape = TVMDSPBuiltinMakeShape(NULL, 3, codes, values);
  if (shape == NULL) FAIL("make_shape failed");

  /* Verify shape */
  if (shape->size != 3) FAIL("size wrong");
  if (TVMDSPShapeAt(shape, 0) != 2) FAIL("dim[0] wrong");
  if (TVMDSPShapeAt(shape, 1) != 3) FAIL("dim[1] wrong");
  if (TVMDSPShapeAt(shape, 2) != 4) FAIL("dim[2] wrong");

  TVMDSPShapeDecRef(shape);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Make Shape with Heap Values
 * ---------------------------------------------------------------------------*/
static int test_make_shape_heap(void) {
  TVMDSPNDArray* heap;
  TVMDSPShape* shape;
  int32_t codes[] = {kMakeShapeLoadShape, kMakeShapeUseImm, kMakeShapeLoadShape};
  int64_t values[] = {0, 5, 1};  /* heap[0], 5, heap[1] */
  int64_t* heap_data;

  TEST("Make shape with heap values");

  /* Create heap and populate */
  heap = TVMDSPBuiltinAllocShapeHeap(4);
  if (heap == NULL) FAIL("heap allocation failed");

  heap_data = (int64_t*)heap->data;
  heap_data[0] = 10;
  heap_data[1] = 20;

  /* Create shape using heap */
  shape = TVMDSPBuiltinMakeShape(heap, 3, codes, values);
  if (shape == NULL) FAIL("make_shape failed");

  /* Verify shape: [heap[0], 5, heap[1]] = [10, 5, 20] */
  if (shape->size != 3) FAIL("size wrong");
  if (TVMDSPShapeAt(shape, 0) != 10) FAIL("dim[0] wrong");
  if (TVMDSPShapeAt(shape, 1) != 5) FAIL("dim[1] wrong");
  if (TVMDSPShapeAt(shape, 2) != 20) FAIL("dim[2] wrong");

  TVMDSPShapeDecRef(shape);
  TVMDSPNDArrayDecRef(heap);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Match Shape with Immediate Assert
 * ---------------------------------------------------------------------------*/
static int test_match_shape_imm(void) {
  TVMDSPNDArray* tensor;
  TVMFFIAny input;
  int64_t shape[] = {2, 3, 4};
  int32_t codes[] = {kMatchShapeAssertEqualToImm, kMatchShapeAssertEqualToImm, kMatchShapeAssertEqualToImm};
  int64_t values[] = {2, 3, 4};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};
  int result;

  TEST("Match shape with immediate assert");

  tensor = TVMDSPNDArrayAlloc(shape, 3, dtype, device);
  if (tensor == NULL) FAIL("tensor allocation failed");

  input.type_index = kTVMFFITensor;
  input.v_obj = (TVMFFIObject*)tensor;

  /* Should succeed - shape matches */
  result = TVMDSPBuiltinMatchShape(&input, NULL, 3, codes, values);
  if (result != 0) FAIL("match_shape should succeed");

  /* Change expected value - should fail */
  values[1] = 5;
  result = TVMDSPBuiltinMatchShape(&input, NULL, 3, codes, values);
  if (result == 0) FAIL("match_shape should fail on mismatch");

  TVMDSPNDArrayDecRef(tensor);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Match Shape Store to Heap
 * ---------------------------------------------------------------------------*/
static int test_match_shape_store(void) {
  TVMDSPNDArray* tensor;
  TVMDSPNDArray* heap;
  TVMFFIAny input;
  int64_t shape[] = {10, 20};
  int32_t codes[] = {kMatchShapeStoreToHeap, kMatchShapeStoreToHeap};
  int64_t values[] = {0, 1};  /* Store to heap[0], heap[1] */
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};
  int64_t* heap_data;
  int result;

  TEST("Match shape store to heap");

  tensor = TVMDSPNDArrayAlloc(shape, 2, dtype, device);
  if (tensor == NULL) FAIL("tensor allocation failed");

  heap = TVMDSPBuiltinAllocShapeHeap(4);
  if (heap == NULL) FAIL("heap allocation failed");

  input.type_index = kTVMFFITensor;
  input.v_obj = (TVMFFIObject*)tensor;

  result = TVMDSPBuiltinMatchShape(&input, heap, 2, codes, values);
  if (result != 0) FAIL("match_shape store failed");

  /* Verify heap values */
  heap_data = (int64_t*)heap->data;
  if (heap_data[0] != 10) FAIL("heap[0] wrong");
  if (heap_data[1] != 20) FAIL("heap[1] wrong");

  TVMDSPNDArrayDecRef(tensor);
  TVMDSPNDArrayDecRef(heap);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Check Tensor Info
 * ---------------------------------------------------------------------------*/
static int test_check_tensor_info(void) {
  TVMDSPNDArray* tensor;
  int64_t shape[] = {4, 8};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDataType wrong_dtype = {kDLInt, 64, 1};
  DLDevice device = {kDLCPU, 0};
  int result;

  TEST("Check tensor info");

  tensor = TVMDSPNDArrayAlloc(shape, 2, dtype, device);
  if (tensor == NULL) FAIL("tensor allocation failed");

  /* Check correct info */
  result = TVMDSPBuiltinCheckTensorInfo(tensor, 2, dtype);
  if (result != 0) FAIL("check should pass for correct info");

  /* Check with any ndim */
  result = TVMDSPBuiltinCheckTensorInfo(tensor, -1, dtype);
  if (result != 0) FAIL("check should pass with any ndim");

  /* Check wrong ndim */
  result = TVMDSPBuiltinCheckTensorInfo(tensor, 3, dtype);
  if (result == 0) FAIL("check should fail for wrong ndim");

  /* Check wrong dtype */
  result = TVMDSPBuiltinCheckTensorInfo(tensor, 2, wrong_dtype);
  if (result == 0) FAIL("check should fail for wrong dtype");

  TVMDSPNDArrayDecRef(tensor);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Null Value
 * ---------------------------------------------------------------------------*/
static int test_null_value(void) {
  TVMFFIAny out;

  TEST("Null value");

  out.type_index = 123;
  out.v_obj = (TVMFFIObject*)1;

  TVMDSPBuiltinNullValue(&out);

  if (out.type_index != kTVMFFINone) FAIL("type_index should be None");
  if (out.v_obj != NULL) FAIL("v_obj should be NULL");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Builtin Registration
 * ---------------------------------------------------------------------------*/
static int test_builtin_registration(void) {
  TVMFFIObjectHandle func;
  int result;

  TEST("VM builtin registration");

  /* Look up alloc_storage */
  result = TVMBackendGetFuncFromGlobalRegistry("vm.builtin.alloc_storage", &func);
  if (result != 0) FAIL("alloc_storage not found");
  if (func == NULL) FAIL("alloc_storage func is NULL");

  /* Look up alloc_tensor */
  result = TVMBackendGetFuncFromGlobalRegistry("vm.builtin.alloc_tensor", &func);
  if (result != 0) FAIL("alloc_tensor not found");
  if (func == NULL) FAIL("alloc_tensor func is NULL");

  /* Look up make_shape */
  result = TVMBackendGetFuncFromGlobalRegistry("vm.builtin.make_shape", &func);
  if (result != 0) FAIL("make_shape not found");
  if (func == NULL) FAIL("make_shape func is NULL");

  /* Look up match_shape */
  result = TVMBackendGetFuncFromGlobalRegistry("vm.builtin.match_shape", &func);
  if (result != 0) FAIL("match_shape not found");
  if (func == NULL) FAIL("match_shape func is NULL");

  /* Look up null_value */
  result = TVMBackendGetFuncFromGlobalRegistry("vm.builtin.null_value", &func);
  if (result != 0) FAIL("null_value not found");
  if (func == NULL) FAIL("null_value func is NULL");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Storage Reference Counting
 * ---------------------------------------------------------------------------*/
static int test_storage_refcount(void) {
  TVMDSPStorage* storage;
  DLDataType dtype = {kDLFloat, 32, 1};

  TEST("Storage reference counting");

  storage = TVMDSPBuiltinAllocStorage(64, 0, dtype);
  if (storage == NULL) FAIL("storage allocation failed");

  if (storage->ref_counter != 1) FAIL("initial refcount wrong");

  TVMDSPStorageIncRef(storage);
  if (storage->ref_counter != 2) FAIL("refcount after inc wrong");

  TVMDSPStorageDecRef(storage);
  if (storage->ref_counter != 1) FAIL("refcount after dec wrong");

  /* Final decref should free */
  TVMDSPStorageDecRef(storage);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Reshape Direct (Phase 3 FFI bypass)
 * ---------------------------------------------------------------------------*/
static int test_reshape_direct(void) {
  TVMDSPStorage* storage;
  TVMDSPNDArray* tensor;
  TVMDSPNDArray* reshaped;
  DLDataType dtype = {kDLFloat, 32, 1};
  int64_t shape[] = {2, 3, 4};
  int64_t new_shape[] = {6, 4};
  float* data;
  float* reshaped_data;

  TEST("Reshape direct");

  /* Allocate storage for 24 floats = 96 bytes */
  storage = TVMDSPBuiltinAllocStorage(96, 0, dtype);
  if (storage == NULL) FAIL("storage allocation failed");

  /* Allocate tensor with shape [2, 3, 4] */
  tensor = TVMDSPBuiltinAllocTensor(storage, 0, shape, 3, dtype);
  if (tensor == NULL) FAIL("tensor allocation failed");

  /* Fill with test data */
  data = (float*)tensor->data;
  for (int i = 0; i < 24; i++) {
    data[i] = (float)i;
  }

  /* Reshape to [6, 4] */
  reshaped = TVMDSPBuiltinReshapeDirect(tensor, new_shape, 2);
  if (reshaped == NULL) FAIL("reshape failed");

  /* Verify new shape */
  if (reshaped->ndim != 2) FAIL("reshaped ndim wrong");
  if (reshaped->shape[0] != 6) FAIL("reshaped shape[0] wrong");
  if (reshaped->shape[1] != 4) FAIL("reshaped shape[1] wrong");

  /* Verify data is shared (same pointer) */
  if (reshaped->data != tensor->data) FAIL("reshaped data not shared");

  /* Verify data is accessible through reshaped view */
  reshaped_data = (float*)reshaped->data;
  if (reshaped_data[0] != 0.0f) FAIL("reshaped data[0] wrong");
  if (reshaped_data[23] != 23.0f) FAIL("reshaped data[23] wrong");

  /* Verify dtype preserved */
  if (reshaped->dtype.code != kDLFloat) FAIL("reshaped dtype.code wrong");
  if (reshaped->dtype.bits != 32) FAIL("reshaped dtype.bits wrong");

  TVMDSPNDArrayDecRef(reshaped);
  TVMDSPNDArrayDecRef(tensor);
  TVMDSPStorageDecRef(storage);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Make Tuple Direct (Phase 3 FFI bypass)
 * ---------------------------------------------------------------------------*/
static int test_make_tuple_direct(void) {
  TVMDSPStorage* storage;
  TVMDSPNDArray* tensor1;
  TVMDSPNDArray* tensor2;
  TVMDSPNDArray* result;
  TVMFFIAny values[2];
  DLDataType dtype = {kDLFloat, 32, 1};
  int64_t shape[] = {4};

  TEST("Make tuple direct");

  /* Allocate storage */
  storage = TVMDSPBuiltinAllocStorage(32, 0, dtype);
  if (storage == NULL) FAIL("storage allocation failed");

  /* Allocate two tensors */
  tensor1 = TVMDSPBuiltinAllocTensor(storage, 0, shape, 1, dtype);
  if (tensor1 == NULL) FAIL("tensor1 allocation failed");

  tensor2 = TVMDSPBuiltinAllocTensor(storage, 16, shape, 1, dtype);
  if (tensor2 == NULL) FAIL("tensor2 allocation failed");

  /* Setup values array */
  values[0].type_index = kTVMFFITensor;
  values[0].v_obj = (TVMFFIObject*)tensor1;
  values[1].type_index = kTVMFFITensor;
  values[1].v_obj = (TVMFFIObject*)tensor2;

  /* Save initial refcount */
  int initial_refcount = tensor1->ref_counter;

  /* Make tuple - should return first element with IncRef */
  result = TVMDSPBuiltinMakeTupleDirect(values, 2);
  if (result == NULL) FAIL("make_tuple failed");

  /* Verify it returns first tensor */
  if (result != tensor1) FAIL("make_tuple should return first tensor");

  /* Verify refcount was incremented */
  if (tensor1->ref_counter != initial_refcount + 1) FAIL("refcount not incremented");

  /* Test with single element */
  TVMDSPNDArrayDecRef(result);  /* Balance the IncRef from make_tuple */

  result = TVMDSPBuiltinMakeTupleDirect(values, 1);
  if (result != tensor1) FAIL("single element tuple wrong");

  TVMDSPNDArrayDecRef(result);  /* Balance the IncRef */
  TVMDSPNDArrayDecRef(tensor1);
  TVMDSPNDArrayDecRef(tensor2);
  TVMDSPStorageDecRef(storage);

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Make Tuple Direct with empty/non-NDArray values
 * ---------------------------------------------------------------------------*/
static int test_make_tuple_direct_edge_cases(void) {
  TVMDSPNDArray* result;
  TVMFFIAny values[2];

  TEST("Make tuple direct edge cases");

  /* Test with empty array */
  result = TVMDSPBuiltinMakeTupleDirect(values, 0);
  if (result != NULL) FAIL("empty tuple should return NULL");

  /* Test with non-NDArray first element */
  values[0].type_index = kTVMFFIInt;
  values[0].v_int64 = 42;
  result = TVMDSPBuiltinMakeTupleDirect(values, 1);
  if (result != NULL) FAIL("non-NDArray tuple should return NULL");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Main Test Runner
 * ---------------------------------------------------------------------------*/
int main(void) {
  printf("\n=== TVM DSP Runtime: VM Builtins Tests (Phase 4) ===\n\n");

  /* Initialize platform */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("Running tests:\n");

  /* Storage tests */
  if (test_storage_alloc() != 0) goto fail;
  if (test_tensor_from_storage() != 0) goto fail;
  if (test_tensor_offset() != 0) goto fail;
  if (test_storage_refcount() != 0) goto fail;

  /* Shape heap tests */
  if (test_shape_heap() != 0) goto fail;

  /* Make shape tests */
  if (test_make_shape_imm() != 0) goto fail;
  if (test_make_shape_heap() != 0) goto fail;

  /* Match shape tests */
  if (test_match_shape_imm() != 0) goto fail;
  if (test_match_shape_store() != 0) goto fail;

  /* Check info tests */
  if (test_check_tensor_info() != 0) goto fail;

  /* Null value test */
  if (test_null_value() != 0) goto fail;

  /* Registration test */
  if (test_builtin_registration() != 0) goto fail;

  /* Phase 3 direct API tests */
  if (test_reshape_direct() != 0) goto fail;
  if (test_make_tuple_direct() != 0) goto fail;
  if (test_make_tuple_direct_edge_cases() != 0) goto fail;

  printf("\n=== Results: %d/%d tests passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 0;

fail:
  printf("\n=== TESTS FAILED: %d/%d passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 1;
}
