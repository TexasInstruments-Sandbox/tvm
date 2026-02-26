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
 * \file test_ffi_types.c
 * \brief Test suite for TVM DSP Runtime FFI types
 */

#include "../ffi/ffi_types.h"
#include "../ffi/object.h"
#include "../platform/dsp_platform.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) \
  do { printf("  Testing %s... ", name); test_count++; } while(0)

#define PASS() \
  do { printf("PASS\n"); pass_count++; } while(0)

#define FAIL(msg) \
  do { printf("FAIL: %s\n", msg); return -1; } while(0)

/*---------------------------------------------------------------------------
 * Test: Type Index Constants
 *---------------------------------------------------------------------------*/
static int test_type_indices(void) {
  TEST("type index values");

  /* Verify type indices match TVM's c_api.h */
  if (kTVMFFINone != 0) FAIL("kTVMFFINone != 0");
  if (kTVMFFIInt != 1) FAIL("kTVMFFIInt != 1");
  if (kTVMFFIBool != 2) FAIL("kTVMFFIBool != 2");
  if (kTVMFFIFloat != 3) FAIL("kTVMFFIFloat != 3");
  if (kTVMFFIOpaquePtr != 4) FAIL("kTVMFFIOpaquePtr != 4");
  if (kTVMFFIDataType != 5) FAIL("kTVMFFIDataType != 5");
  if (kTVMFFIDevice != 6) FAIL("kTVMFFIDevice != 6");
  if (kTVMFFIDLTensorPtr != 7) FAIL("kTVMFFIDLTensorPtr != 7");
  if (kTVMFFIRawStr != 8) FAIL("kTVMFFIRawStr != 8");

  /* Static object types */
  if (kTVMFFIStaticObjectBegin != 64) FAIL("kTVMFFIStaticObjectBegin != 64");
  if (kTVMFFIObject != 64) FAIL("kTVMFFIObject != 64");
  if (kTVMFFITensor != 70) FAIL("kTVMFFITensor != 70");
  if (kTVMFFIShape != 69) FAIL("kTVMFFIShape != 69");
  if (kTVMFFIFunction != 68) FAIL("kTVMFFIFunction != 68");

  /* Dynamic object range */
  if (kTVMFFIDynObjectBegin != 128) FAIL("kTVMFFIDynObjectBegin != 128");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Structure Size
 *---------------------------------------------------------------------------*/
static int test_any_size(void) {
  TEST("TVMFFIAny size");

  /* TVMFFIAny should be 16 bytes: 4 + 4 + 8 */
  if (sizeof(TVMFFIAny) != 16) {
    printf("FAIL: sizeof(TVMFFIAny) = %zu, expected 16\n", sizeof(TVMFFIAny));
    return -1;
  }

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIObject Structure Size
 *---------------------------------------------------------------------------*/
static int test_object_size(void) {
  TEST("TVMFFIObject size");

  /* TVMFFIObject should be 16 bytes: 4 + 4 + 8 */
  if (sizeof(TVMFFIObject) != 16) {
    printf("FAIL: sizeof(TVMFFIObject) = %zu, expected 16\n", sizeof(TVMFFIObject));
    return -1;
  }

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny None Value
 *---------------------------------------------------------------------------*/
static int test_any_none(void) {
  TEST("TVMFFIAny None");

  TVMFFIAny any;
  TVMFFIAnySetNone(&any);

  if (any.type_index != kTVMFFINone) FAIL("type_index not kTVMFFINone");
  if (!TVMFFIAnyIsNone(&any)) FAIL("IsNone returned false");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Integer Value
 *---------------------------------------------------------------------------*/
static int test_any_int(void) {
  TEST("TVMFFIAny Int");

  TVMFFIAny any;
  TVMFFIAnySetInt(&any, 42);

  if (any.type_index != kTVMFFIInt) FAIL("type_index not kTVMFFIInt");
  if (!TVMFFIAnyIsInt(&any)) FAIL("IsInt returned false");
  if (TVMFFIAnyGetInt(&any) != 42) FAIL("value != 42");

  /* Test negative value */
  TVMFFIAnySetInt(&any, -12345);
  if (TVMFFIAnyGetInt(&any) != -12345) FAIL("negative value failed");

  /* Test large value */
  TVMFFIAnySetInt(&any, 0x7FFFFFFFFFFFFFFFLL);
  if (TVMFFIAnyGetInt(&any) != 0x7FFFFFFFFFFFFFFFLL) FAIL("large value failed");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Float Value
 *---------------------------------------------------------------------------*/
static int test_any_float(void) {
  TEST("TVMFFIAny Float");

  TVMFFIAny any;
  TVMFFIAnySetFloat(&any, 3.14159);

  if (any.type_index != kTVMFFIFloat) FAIL("type_index not kTVMFFIFloat");
  if (!TVMFFIAnyIsFloat(&any)) FAIL("IsFloat returned false");
  if (fabs(TVMFFIAnyGetFloat(&any) - 3.14159) > 1e-10) FAIL("value != 3.14159");

  /* Test negative float */
  TVMFFIAnySetFloat(&any, -1.0e-10);
  if (fabs(TVMFFIAnyGetFloat(&any) - (-1.0e-10)) > 1e-20) FAIL("negative float failed");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Boolean Value
 *---------------------------------------------------------------------------*/
static int test_any_bool(void) {
  TEST("TVMFFIAny Bool");

  TVMFFIAny any;
  TVMFFIAnySetBool(&any, 1);

  if (any.type_index != kTVMFFIBool) FAIL("type_index not kTVMFFIBool");
  if (TVMFFIAnyGetInt(&any) != 1) FAIL("true value != 1");

  TVMFFIAnySetBool(&any, 0);
  if (TVMFFIAnyGetInt(&any) != 0) FAIL("false value != 0");

  /* Non-zero should become 1 */
  TVMFFIAnySetBool(&any, 42);
  if (TVMFFIAnyGetInt(&any) != 1) FAIL("non-zero should be 1");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Pointer Value
 *---------------------------------------------------------------------------*/
static int test_any_ptr(void) {
  TEST("TVMFFIAny Ptr");

  int dummy = 123;
  TVMFFIAny any;
  TVMFFIAnySetPtr(&any, &dummy);

  if (any.type_index != kTVMFFIOpaquePtr) FAIL("type_index not kTVMFFIOpaquePtr");
  if (TVMFFIAnyGetPtr(&any) != &dummy) FAIL("pointer mismatch");

  /* NULL pointer */
  TVMFFIAnySetPtr(&any, NULL);
  if (TVMFFIAnyGetPtr(&any) != NULL) FAIL("NULL pointer failed");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny DLDataType Value
 *---------------------------------------------------------------------------*/
static int test_any_dtype(void) {
  TEST("TVMFFIAny DataType");

  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;

  TVMFFIAny any;
  TVMFFIAnySetDataType(&any, dtype);

  if (any.type_index != kTVMFFIDataType) FAIL("type_index not kTVMFFIDataType");

  DLDataType result = TVMFFIAnyGetDataType(&any);
  if (result.code != kDLFloat) FAIL("dtype.code mismatch");
  if (result.bits != 32) FAIL("dtype.bits mismatch");
  if (result.lanes != 1) FAIL("dtype.lanes mismatch");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny DLDevice Value
 *---------------------------------------------------------------------------*/
static int test_any_device(void) {
  TEST("TVMFFIAny Device");

  DLDevice device;
  device.device_type = kDLCPU;
  device.device_id = 0;

  TVMFFIAny any;
  TVMFFIAnySetDevice(&any, device);

  if (any.type_index != kTVMFFIDevice) FAIL("type_index not kTVMFFIDevice");

  DLDevice result = TVMFFIAnyGetDevice(&any);
  if (result.device_type != kDLCPU) FAIL("device_type mismatch");
  if (result.device_id != 0) FAIL("device_id mismatch");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Move Operation
 *---------------------------------------------------------------------------*/
static int test_any_move(void) {
  TEST("TVMFFIAny Move");

  TVMFFIAny src, dst;
  TVMFFIAnySetInt(&src, 999);

  TVMFFIAnyMove(&src, &dst);

  /* Destination should have the value */
  if (dst.type_index != kTVMFFIInt) FAIL("dst type_index wrong");
  if (TVMFFIAnyGetInt(&dst) != 999) FAIL("dst value wrong");

  /* Source should be None */
  if (src.type_index != kTVMFFINone) FAIL("src not cleared");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: TVMFFIAny Copy Operation
 *---------------------------------------------------------------------------*/
static int test_any_copy(void) {
  TEST("TVMFFIAny Copy");

  TVMFFIAny src, dst;
  TVMFFIAnySetFloat(&src, 2.718);

  TVMFFIAnyCopy(&src, &dst);

  /* Both should have the value */
  if (src.type_index != kTVMFFIFloat) FAIL("src type_index changed");
  if (dst.type_index != kTVMFFIFloat) FAIL("dst type_index wrong");
  if (fabs(TVMFFIAnyGetFloat(&src) - 2.718) > 1e-10) FAIL("src value changed");
  if (fabs(TVMFFIAnyGetFloat(&dst) - 2.718) > 1e-10) FAIL("dst value wrong");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: Type Name Lookup
 *---------------------------------------------------------------------------*/
static int test_type_names(void) {
  TEST("Type names");

  if (strcmp(TVMDSPGetTypeName(kTVMFFINone), "None") != 0)
    FAIL("None name wrong");
  if (strcmp(TVMDSPGetTypeName(kTVMFFIInt), "Int") != 0)
    FAIL("Int name wrong");
  if (strcmp(TVMDSPGetTypeName(kTVMFFITensor), "Tensor") != 0)
    FAIL("Tensor name wrong");
  if (strcmp(TVMDSPGetTypeName(kTVMFFIShape), "Shape") != 0)
    FAIL("Shape name wrong");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: Object Reference Counting
 *---------------------------------------------------------------------------*/

/* Test object with custom deleter */
static int test_object_deleted = 0;

static void test_object_deleter(TVMFFIObject* obj) {
  test_object_deleted = 1;
  TVMDSPObjectFree(obj);
}

static int test_object_refcount(void) {
  TEST("Object ref counting");

  test_object_deleted = 0;

  /* Allocate a test object */
  TVMFFIObject* obj = TVMDSPObjectAlloc(sizeof(TVMFFIObject),
                                         kTVMFFIObject,
                                         test_object_deleter);
  if (obj == NULL) FAIL("allocation failed");

  /* Initial ref count should be 1 */
  if (obj->ref_counter != 1) FAIL("initial ref_counter != 1");

  /* Increment ref count */
  TVMFFIObjectIncRef(obj);
  if (obj->ref_counter != 2) FAIL("ref_counter after inc != 2");

  /* Decrement ref count */
  TVMFFIObjectDecRef(obj);
  if (obj->ref_counter != 1) FAIL("ref_counter after dec != 1");
  if (test_object_deleted) FAIL("object deleted too early");

  /* Final decrement should delete */
  TVMFFIObjectDecRef(obj);
  if (!test_object_deleted) FAIL("object not deleted");

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Test: Object Accessor Functions
 *---------------------------------------------------------------------------*/
static int test_object_accessors(void) {
  TEST("Object accessors");

  /* Allocate a mock NDArray-like object */
  size_t obj_size = sizeof(TVMFFIObject) + sizeof(DLTensor);
  TVMFFIObject* obj = TVMDSPObjectAlloc(obj_size, kTVMFFITensor,
                                         TVMDSPDefaultObjectDeleter);
  if (obj == NULL) FAIL("allocation failed");

  /* Test type index accessor */
  if (TVMFFIObjectGetTypeIndex(obj) != kTVMFFITensor)
    FAIL("GetTypeIndex wrong");

  /* Test DLTensor pointer accessor */
  DLTensor* tensor = TVMFFINDArrayGetDLTensorPtr(obj);
  if (tensor != (DLTensor*)((char*)obj + sizeof(TVMFFIObject)))
    FAIL("GetDLTensorPtr wrong");

  /* Test device creation helper */
  DLDevice dev = TVMFFIDLDeviceFromIntPair(kDLCPU, 0);
  if (dev.device_type != kDLCPU) FAIL("device_type wrong");
  if (dev.device_id != 0) FAIL("device_id wrong");

  TVMFFIObjectDecRef(obj);

  PASS();
  return 0;
}

/*---------------------------------------------------------------------------
 * Main Test Runner
 *---------------------------------------------------------------------------*/
int main(void) {
  printf("\n=== TVM DSP Runtime: FFI Types Tests ===\n\n");

  /* Initialize platform */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  printf("Running tests:\n");

  /* Run all tests */
  if (test_type_indices() != 0) goto fail;
  if (test_any_size() != 0) goto fail;
  if (test_object_size() != 0) goto fail;
  if (test_any_none() != 0) goto fail;
  if (test_any_int() != 0) goto fail;
  if (test_any_float() != 0) goto fail;
  if (test_any_bool() != 0) goto fail;
  if (test_any_ptr() != 0) goto fail;
  if (test_any_dtype() != 0) goto fail;
  if (test_any_device() != 0) goto fail;
  if (test_any_move() != 0) goto fail;
  if (test_any_copy() != 0) goto fail;
  if (test_type_names() != 0) goto fail;
  if (test_object_refcount() != 0) goto fail;
  if (test_object_accessors() != 0) goto fail;

  printf("\n=== Results: %d/%d tests passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 0;

fail:
  printf("\n=== TESTS FAILED: %d/%d passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 1;
}
