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
 * \file test_registry.c
 * \brief Test suite for TVM DSP Runtime Function Registry (Phase 5)
 */

#include "../registry/registry.h"
#include "../platform/dsp_platform.h"
#include "../ffi/ffi_types.h"
#include <stdio.h>
#include <string.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) \
  do { printf("  Testing %s... ", name); test_count++; } while(0)

#define PASS() \
  do { printf("PASS\n"); pass_count++; } while(0)

#define FAIL(msg) \
  do { printf("FAIL: %s\n", msg); return -1; } while(0)

/* ---------------------------------------------------------------------------
 * Test packed functions
 * ---------------------------------------------------------------------------*/

/* Simple test function that adds two integers */
static int test_add_packed(const TVMFFIAny* args, int32_t num_args,
                           TVMFFIAny* ret) {
  if (num_args != 2) {
    return -1;
  }
  if (args[0].type_index != kTVMFFIInt || args[1].type_index != kTVMFFIInt) {
    return -1;
  }

  ret->type_index = kTVMFFIInt;
  ret->zero_padding = 0;
  ret->v_int64 = args[0].v_int64 + args[1].v_int64;

  return 0;
}

/* Test function that doubles an integer */
static int test_double_packed(const TVMFFIAny* args, int32_t num_args,
                              TVMFFIAny* ret) {
  if (num_args != 1) {
    return -1;
  }
  if (args[0].type_index != kTVMFFIInt) {
    return -1;
  }

  ret->type_index = kTVMFFIInt;
  ret->zero_padding = 0;
  ret->v_int64 = args[0].v_int64 * 2;

  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Registry Registration
 * ---------------------------------------------------------------------------*/
static int test_registry_register(void) {
  int ret;

  TEST("Registry registration");

  ret = TVMRegistryRegister("test.add", test_add_packed);
  if (ret != 0) FAIL("failed to register test.add");

  ret = TVMRegistryRegister("test.double", test_double_packed);
  if (ret != 0) FAIL("failed to register test.double");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Registry Lookup
 * ---------------------------------------------------------------------------*/
static int test_registry_lookup(void) {
  TVMFFIObjectHandle func;

  TEST("Registry lookup");

  func = TVMRegistryLookup("test.add");
  if (func == NULL) FAIL("test.add not found");

  func = TVMRegistryLookup("test.double");
  if (func == NULL) FAIL("test.double not found");

  func = TVMRegistryLookup("nonexistent.function");
  if (func != NULL) FAIL("nonexistent function should return NULL");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: TVMFFIFunctionCall
 * ---------------------------------------------------------------------------*/
static int test_function_call(void) {
  TVMFFIObjectHandle func;
  TVMFFIAny args[2];
  TVMFFIAny result;
  int ret;

  TEST("TVMFFIFunctionCall");

  /* Look up test.add */
  func = TVMRegistryLookup("test.add");
  if (func == NULL) FAIL("test.add not found");

  /* Set up arguments */
  args[0].type_index = kTVMFFIInt;
  args[0].zero_padding = 0;
  args[0].v_int64 = 10;

  args[1].type_index = kTVMFFIInt;
  args[1].zero_padding = 0;
  args[1].v_int64 = 25;

  result.type_index = kTVMFFINone;
  result.zero_padding = 0;
  result.v_int64 = 0;

  /* Call function */
  ret = TVMFFIFunctionCall(func, args, 2, &result);
  if (ret != 0) FAIL("function call returned error");

  if (result.type_index != kTVMFFIInt) FAIL("result type wrong");
  if (result.v_int64 != 35) FAIL("result value wrong (expected 35)");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: TVMBackendAnyListSetPackedArg
 *
 * API: TVMBackendAnyListSetPackedArg(anylist, index, args, arg_offset)
 *      Copies anylist[index] to args[arg_offset]
 *      anylist = source (e.g., register file)
 *      args = destination (packed args for function call)
 * ---------------------------------------------------------------------------*/
static int test_anylist_set_arg(void) {
  TVMFFIAny regfile[3];  /* Register file (source) */
  TVMFFIAny packed_args[3];  /* Packed arguments (destination) */
  int ret;

  TEST("TVMBackendAnyListSetPackedArg");

  /* Set up register file values */
  regfile[0].type_index = kTVMFFIInt;
  regfile[0].zero_padding = 0;
  regfile[0].v_int64 = 42;

  regfile[1].type_index = kTVMFFIFloat;
  regfile[1].zero_padding = 0;
  regfile[1].v_float64 = 3.14159;

  /* Clear packed args */
  memset(packed_args, 0, sizeof(packed_args));

  /* Copy regfile[0] to packed_args[1] */
  ret = TVMBackendAnyListSetPackedArg(regfile, 0, packed_args, 1);
  if (ret != 0) FAIL("set arg failed");

  if (packed_args[1].type_index != kTVMFFIInt) FAIL("packed_args type wrong");
  if (packed_args[1].v_int64 != 42) FAIL("packed_args value wrong");

  /* Copy regfile[1] to packed_args[2] */
  ret = TVMBackendAnyListSetPackedArg(regfile, 1, packed_args, 2);
  if (ret != 0) FAIL("set arg failed");

  if (packed_args[2].type_index != kTVMFFIFloat) FAIL("packed_args type wrong");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: TVMBackendAnyListMoveFromPackedReturn
 * ---------------------------------------------------------------------------*/
static int test_anylist_move_return(void) {
  TVMFFIAny source[3];
  TVMFFIAny target[3];
  int ret;

  TEST("TVMBackendAnyListMoveFromPackedReturn");

  /* Set up source with result at index 2 */
  source[2].type_index = kTVMFFIInt;
  source[2].zero_padding = 0;
  source[2].v_int64 = 999;

  /* Clear target */
  memset(target, 0, sizeof(target));

  /* Move source[2] to target[0] */
  ret = TVMBackendAnyListMoveFromPackedReturn(target, 0, source, 2);
  if (ret != 0) FAIL("move failed");

  if (target[0].type_index != kTVMFFIInt) FAIL("target type wrong");
  if (target[0].v_int64 != 999) FAIL("target value wrong");

  /* Source should be cleared (moved) */
  if (source[2].type_index != kTVMFFINone) FAIL("source should be cleared");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: TVMBackendAnyListResetItem
 * ---------------------------------------------------------------------------*/
static int test_anylist_reset(void) {
  TVMFFIAny list[3];
  int ret;

  TEST("TVMBackendAnyListResetItem");

  /* Set up list with values */
  list[0].type_index = kTVMFFIInt;
  list[0].zero_padding = 0;
  list[0].v_int64 = 123;

  list[1].type_index = kTVMFFIFloat;
  list[1].zero_padding = 0;
  list[1].v_float64 = 4.56;

  /* Reset item 1 */
  ret = TVMBackendAnyListResetItem(list, 1);
  if (ret != 0) FAIL("reset failed");

  if (list[1].type_index != kTVMFFINone) FAIL("reset item should be None");

  /* Item 0 should be unchanged */
  if (list[0].type_index != kTVMFFIInt) FAIL("item 0 changed unexpectedly");
  if (list[0].v_int64 != 123) FAIL("item 0 value changed");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: End-to-end function call pattern
 * ---------------------------------------------------------------------------*/
static int test_e2e_function_call(void) {
  TVMFFIObjectHandle func;
  TVMFFIAny args[3];  /* 2 args + 1 result slot */
  int ret;

  TEST("End-to-end function call pattern");

  /* Look up test.add */
  func = TVMRegistryLookup("test.add");
  if (func == NULL) FAIL("test.add not found");

  /* Prepare argument list like generated code does */
  args[0].type_index = kTVMFFIInt;
  args[0].zero_padding = 0;
  args[0].v_int64 = 100;

  args[1].type_index = kTVMFFIInt;
  args[1].zero_padding = 0;
  args[1].v_int64 = 200;

  args[2].type_index = kTVMFFINone;
  args[2].zero_padding = 0;
  args[2].v_int64 = 0;

  /* Call function: TVMFFIFunctionCall(func, args, 2, &args[2]) */
  ret = TVMFFIFunctionCall(func, args, 2, &args[2]);
  if (ret != 0) FAIL("function call failed");

  /* Check result at args[2] */
  if (args[2].type_index != kTVMFFIInt) FAIL("result type wrong");
  if (args[2].v_int64 != 300) FAIL("result wrong (expected 300)");

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Test: Packed function type checking
 * ---------------------------------------------------------------------------*/
static int test_packed_func_type_check(void) {
  TVMFFIObjectHandle func;
  TVMDSPPackedFunc* pfunc;

  TEST("PackedFunc type check");

  func = TVMRegistryLookup("test.add");
  if (func == NULL) FAIL("test.add not found");

  pfunc = (TVMDSPPackedFunc*)func;

  if (pfunc->type_index != TVM_DSP_PACKED_FUNC_TYPE_INDEX) {
    FAIL("PackedFunc type index wrong");
  }

  if (TVMDSPIsPackedFunc(func) != 1) {
    FAIL("TVMDSPIsPackedFunc should return 1");
  }

  if (TVMDSPIsPackedFunc(NULL) != 0) {
    FAIL("TVMDSPIsPackedFunc(NULL) should return 0");
  }

  PASS();
  return 0;
}

/* ---------------------------------------------------------------------------
 * Main Test Runner
 * ---------------------------------------------------------------------------*/
int main(void) {
  printf("\n=== TVM DSP Runtime: Function Registry Tests (Phase 5) ===\n\n");

  /* Initialize platform */
  if (tvm_dsp_platform_init() != 0) {
    printf("FATAL: Platform initialization failed\n");
    return 1;
  }

  /* Initialize registry */
  TVMRegistryInit();

  printf("Running tests:\n");

  /* Registration tests */
  if (test_registry_register() != 0) goto fail;
  if (test_registry_lookup() != 0) goto fail;

  /* Function call tests */
  if (test_function_call() != 0) goto fail;
  if (test_packed_func_type_check() != 0) goto fail;

  /* AnyList helper tests */
  if (test_anylist_set_arg() != 0) goto fail;
  if (test_anylist_move_return() != 0) goto fail;
  if (test_anylist_reset() != 0) goto fail;

  /* End-to-end test */
  if (test_e2e_function_call() != 0) goto fail;

  printf("\n=== Results: %d/%d tests passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 0;

fail:
  printf("\n=== TESTS FAILED: %d/%d passed ===\n\n", pass_count, test_count);
  tvm_dsp_platform_shutdown();
  return 1;
}
