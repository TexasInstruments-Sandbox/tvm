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
 * \file tests/test_constants.c
 * \brief Test the constants parser with real weights.bin data
 */

#include "../constants/constants.h"
#include "../constants/constants_c_api.h"
#include "../ffi/ffi_types.h"
#include "../platform/dsp_platform.h"
#include "../platform/dsp_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Test configuration */
#ifndef TEST_WEIGHTS_PATH
#define TEST_WEIGHTS_PATH "weights.bin"
#endif

/*---------------------------------------------------------------------------
 * Test Utilities
 *---------------------------------------------------------------------------*/

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
  tests_run++; \
  if (!(cond)) { \
    printf("FAIL: %s (line %d)\n", msg, __LINE__); \
    return -1; \
  } \
  tests_passed++; \
} while(0)

/* Load file into malloc'd buffer */
static void* load_file(const char* path, size_t* size_out) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    printf("ERROR: Cannot open file: %s\n", path);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (size <= 0) {
    fclose(f);
    return NULL;
  }

  void* data = malloc(size);
  if (!data) {
    fclose(f);
    return NULL;
  }

  size_t read = fread(data, 1, size, f);
  fclose(f);

  if (read != (size_t)size) {
    free(data);
    return NULL;
  }

  *size_out = (size_t)size;
  return data;
}

/* Get type name from type index */
static const char* get_type_name(int type_index) {
  switch (type_index) {
    case kTVMFFINone:       return "None";
    case kTVMFFIInt:        return "Int";
    case kTVMFFIBool:       return "Bool";
    case kTVMFFIFloat:      return "Float";
    case kTVMFFIDataType:   return "DataType";
    case kTVMFFIStr:        return "String";
    case kTVMFFIShape:      return "Shape";
    case kTVMFFITensor:    return "NDArray";
    default:                return "Unknown";
  }
}

/*---------------------------------------------------------------------------
 * Test Cases
 *---------------------------------------------------------------------------*/

static int test_parse_weights(const char* weights_path) {
  size_t size;
  void* data = load_file(weights_path, &size);
  TEST_ASSERT(data != NULL, "Failed to load weights file");

  printf("Loaded weights file: %s (%zu bytes)\n", weights_path, size);

  /* Initialize constants system */
  TVMDSPConstantsInit();

  /* Parse weights */
  int count = TVMDSPConstantsParse(data, size);
  printf("Parse result: %d\n", count);

  if (count < 0) {
    printf("Parse error: %s\n", TVMDSPConstantsErrorString(count));
    free(data);
    return -1;
  }

  TEST_ASSERT(count > 0, "Expected at least one constant");
  printf("Successfully parsed %d constants\n", count);

  /* Get constants array */
  int retrieved_count = 0;
  TVMFFIAny* constants = TVMDSPConstantsGet(&retrieved_count);
  TEST_ASSERT(constants != NULL, "TVMDSPConstantsGet returned NULL");
  TEST_ASSERT(retrieved_count == count, "Count mismatch");

  /* Print statistics */
  int num_ndarrays = 0;
  int num_strings = 0;
  int num_shapes = 0;
  int num_datatypes = 0;
  int num_ints = 0;
  int num_floats = 0;
  int num_other = 0;

  for (int i = 0; i < count; i++) {
    switch (constants[i].type_index) {
      case kTVMFFITensor:  num_ndarrays++; break;
      case kTVMFFIStr:      num_strings++; break;
      case kTVMFFIShape:    num_shapes++; break;
      case kTVMFFIDataType: num_datatypes++; break;
      case kTVMFFIInt:      num_ints++; break;
      case kTVMFFIFloat:    num_floats++; break;
      default:              num_other++; break;
    }
  }

  printf("\nConstant types:\n");
  printf("  NDArrays:   %d\n", num_ndarrays);
  printf("  Strings:    %d\n", num_strings);
  printf("  Shapes:     %d\n", num_shapes);
  printf("  DataTypes:  %d\n", num_datatypes);
  printf("  Ints:       %d\n", num_ints);
  printf("  Floats:     %d\n", num_floats);
  printf("  Other:      %d\n", num_other);

  /* Print first few constants in detail */
  printf("\nFirst 10 constants:\n");
  for (int i = 0; i < count && i < 10; i++) {
    TVMFFIAny* c = &constants[i];
    printf("  [%d] type=%s (%d)", i, get_type_name(c->type_index), c->type_index);

    switch (c->type_index) {
      case kTVMFFITensor: {
        TVMDSPNDArray* arr = (TVMDSPNDArray*)c->v_obj;
        /* Access DLTensor fields directly (embedded in TVMDSPNDArray) */
        printf(": ndim=%d, dtype=%d.%d.%d, shape=[",
               arr->ndim, arr->dtype.code, arr->dtype.bits, arr->dtype.lanes);
        for (int d = 0; d < arr->ndim; d++) {
          printf("%lld%s", (long long)arr->shape[d], d < arr->ndim-1 ? "," : "");
        }
        printf("]");
        break;
      }
      case kTVMFFIStr: {
        TVMDSPString* str = (TVMDSPString*)c->v_obj;
        printf(": \"%.*s\"", (int)(str->size > 40 ? 40 : str->size), str->data);
        if (str->size > 40) printf("...");
        break;
      }
      case kTVMFFIShape: {
        TVMDSPShape* shape = (TVMDSPShape*)c->v_obj;
        printf(": [");
        for (size_t d = 0; d < shape->size; d++) {
          printf("%lld%s", (long long)shape->data[d], d < shape->size-1 ? "," : "");
        }
        printf("]");
        break;
      }
      case kTVMFFIDataType: {
        DLDataType* dtype = (DLDataType*)&c->v_int64;
        printf(": code=%d, bits=%d, lanes=%d", dtype->code, dtype->bits, dtype->lanes);
        break;
      }
      case kTVMFFIInt:
        printf(": %lld", (long long)c->v_int64);
        break;
      case kTVMFFIFloat:
        printf(": %f", c->v_float64);
        break;
      default:
        break;
    }
    printf("\n");
  }

  /* Test getting constant by index */
  TVMFFIAny* c0 = TVMDSPConstantGetByIndex(0);
  TEST_ASSERT(c0 != NULL, "TVMDSPConstantGetByIndex(0) returned NULL");
  TEST_ASSERT(c0 == &constants[0], "TVMDSPConstantGetByIndex returned wrong pointer");

  TVMFFIAny* c_invalid = TVMDSPConstantGetByIndex(count);
  TEST_ASSERT(c_invalid == NULL, "TVMDSPConstantGetByIndex(count) should return NULL");

  /* Note: data must remain valid while constants are in use */
  /* We'd normally keep this around for the lifetime of the app */
  free(data);

  return 0;
}

static int test_empty_input(void) {
  TVMDSPConstantsInit();

  int result = TVMDSPConstantsParse(NULL, 0);
  TEST_ASSERT(result == TVM_DSP_CONST_ERR_NULL_INPUT, "Expected NULL input error");

  /* Empty but non-null - either NULL_INPUT or BUFFER_END is acceptable */
  uint8_t empty[1] = {0};
  result = TVMDSPConstantsParse(empty, 0);
  TEST_ASSERT(result == TVM_DSP_CONST_ERR_NULL_INPUT ||
              result == TVM_DSP_CONST_ERR_BUFFER_END,
              "Expected NULL input or buffer end error for size=0");

  return 0;
}

static int test_truncated_input(void) {
  TVMDSPConstantsInit();

  /* Just a count, no constants */
  uint64_t header[] = {10};  /* Says 10 constants but no data */
  int result = TVMDSPConstantsParse(header, sizeof(header));
  TEST_ASSERT(result < 0, "Expected error for truncated input");

  return 0;
}

/*---------------------------------------------------------------------------
 * Main
 *---------------------------------------------------------------------------*/

int main(int argc, char** argv) {
  const char* weights_path = TEST_WEIGHTS_PATH;

  if (argc > 1) {
    weights_path = argv[1];
  }

  printf("=== TVM DSP Constants Parser Test ===\n\n");

  /* Initialize platform (required for tvm_dsp_alloc) */
  int ret = tvm_dsp_platform_init();
  if (ret != 0) {
    printf("ERROR: Failed to initialize platform: %d\n", ret);
    return 1;
  }

  printf("Test 1: Parse weights file\n");
  if (test_parse_weights(weights_path) != 0) {
    printf("FAILED\n\n");
  } else {
    printf("PASSED\n\n");
  }

  printf("Test 2: Empty input handling\n");
  if (test_empty_input() != 0) {
    printf("FAILED\n\n");
  } else {
    printf("PASSED\n\n");
  }

  printf("Test 3: Truncated input handling\n");
  if (test_truncated_input() != 0) {
    printf("FAILED\n\n");
  } else {
    printf("PASSED\n\n");
  }

  printf("=== Summary: %d/%d tests passed ===\n", tests_passed, tests_run);

  return (tests_passed == tests_run) ? 0 : 1;
}
