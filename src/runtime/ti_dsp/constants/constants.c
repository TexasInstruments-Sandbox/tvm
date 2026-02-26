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
 * \file constants/constants.c
 * \brief Constants loader implementation for TVM DSP runtime
 *
 * This module parses TVM's weights.bin format and creates TVMFFIAny
 * constants with zero-copy access to embedded weight data.
 *
 * Memory allocation:
 * - Constant metadata (NDArray objects, shape arrays) is allocated from L3
 * - Tensor DATA is NOT copied - it points directly into embedded weights
 * - All allocations are sized dynamically based on actual model needs
 */

#include "constants.h"
#include "stream.h"
#include "../platform/dsp_platform.h"
#include <string.h>

/*---------------------------------------------------------------------------
 * Maximum Limits (from constants.h, which imports core/config.h)
 *
 * TVM_DSP_CONST_MAX_SANITY - Max constants per model (sanity check)
 * TVM_DSP_PARSE_MAX_NDIM   - Max dimensions for parsing
 *---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
 * Dynamically Allocated Pools
 *
 * These are allocated from L3 memory based on actual model needs.
 * Pointers are NULL until initialization.
 *---------------------------------------------------------------------------*/

/* NDArray object pool */
static TVMDSPNDArray* g_ndarray_pool = NULL;
static int g_ndarray_capacity = 0;
static int g_ndarray_count = 0;

/* Shape data pool (contiguous int64_t array for all shapes) */
static int64_t* g_shape_pool = NULL;
static int g_shape_capacity = 0;
static int g_shape_used = 0;

/* String data pool (for string constants) */
static char* g_string_pool = NULL;
static int g_string_capacity = 0;
static int g_string_used = 0;

/* Shape object pool (for Shape constants, not NDArray shapes) */
static TVMDSPShape* g_shape_obj_pool = NULL;
static int g_shape_obj_capacity = 0;
static int g_shape_obj_count = 0;

/* String object pool (for String constants) */
static TVMDSPString* g_string_obj_pool = NULL;
static int g_string_obj_capacity = 0;
static int g_string_obj_count = 0;

/* Output constants array */
static TVMFFIAny* g_constants = NULL;
static int g_num_constants = 0;
static int g_initialized = 0;

/*
 * Aligned buffer tracking (for C66x unaligned data copies)
 * When embedded weight data is unaligned, we allocate aligned buffers.
 * These must be freed separately from the pools.
 */
#define TVM_DSP_MAX_ALIGNED_BUFFERS 256
static void* g_aligned_buffers[TVM_DSP_MAX_ALIGNED_BUFFERS];
static int g_aligned_buffer_count = 0;

/*---------------------------------------------------------------------------
 * Pool Allocators (from dynamically allocated pools)
 *---------------------------------------------------------------------------*/

static int64_t* alloc_shape(int ndim) {
  int64_t* ptr;
  if (g_shape_used + ndim > g_shape_capacity) {
    tvm_dsp_log("ERROR: Shape pool overflow (%d + %d > %d)\n",
                g_shape_used, ndim, g_shape_capacity);
    return NULL;
  }
  ptr = &g_shape_pool[g_shape_used];
  g_shape_used += ndim;
  return ptr;
}

static TVMDSPNDArray* alloc_ndarray(void) {
  if (g_ndarray_count >= g_ndarray_capacity) {
    tvm_dsp_log("ERROR: NDArray pool overflow (%d >= %d)\n",
                g_ndarray_count, g_ndarray_capacity);
    return NULL;
  }
  return &g_ndarray_pool[g_ndarray_count++];
}

static char* alloc_string(size_t len) {
  char* ptr;
  /* +1 for null terminator */
  if (g_string_used + (int)(len + 1) > g_string_capacity) {
    tvm_dsp_log("ERROR: String pool overflow\n");
    return NULL;
  }
  ptr = &g_string_pool[g_string_used];
  g_string_used += (int)(len + 1);
  return ptr;
}

static TVMDSPShape* alloc_shape_obj(void) {
  if (g_shape_obj_count >= g_shape_obj_capacity) {
    tvm_dsp_log("ERROR: Shape object pool overflow\n");
    return NULL;
  }
  return &g_shape_obj_pool[g_shape_obj_count++];
}

static TVMDSPString* alloc_string_obj(void) {
  if (g_string_obj_count >= g_string_obj_capacity) {
    tvm_dsp_log("ERROR: String object pool overflow\n");
    return NULL;
  }
  return &g_string_obj_pool[g_string_obj_count++];
}

/*---------------------------------------------------------------------------
 * Pre-scan: Count resources needed for allocation
 *
 * This scans the binary once to determine how much memory to allocate.
 *---------------------------------------------------------------------------*/

typedef struct {
  int num_constants;      /* Total number of constants */
  int num_ndarrays;       /* Number of NDArray constants */
  int total_shape_elems;  /* Total int64_t elements for all shapes */
  int num_shape_objs;     /* Number of Shape constants (not NDArray shapes) */
  int num_string_objs;    /* Number of String constants */
  int total_string_bytes; /* Total bytes for all strings */
} TVMDSPConstantsScan;

static int scan_ndarray(TVMDSPStream* stream, TVMDSPConstantsScan* scan) {
  uint64_t magic, reserved;
  int32_t device_type, device_id, ndim;
  DLDataType dtype;
  int64_t data_size;

  /* Read and skip header */
  if (TVMDSPStreamReadU64(stream, &magic) != 0) return -1;
  if (magic != TVM_NDARRAY_MAGIC) return TVM_DSP_CONST_ERR_INVALID_MAGIC;
  if (TVMDSPStreamReadU64(stream, &reserved) != 0) return -1;
  if (TVMDSPStreamReadI32(stream, &device_type) != 0) return -1;
  if (TVMDSPStreamReadI32(stream, &device_id) != 0) return -1;
  if (TVMDSPStreamReadI32(stream, &ndim) != 0) return -1;
  if (TVMDSPStreamRead(stream, &dtype, sizeof(DLDataType)) != 0) return -1;

  /* Skip shape data */
  if (TVMDSPStreamSkip(stream, ndim * sizeof(int64_t)) != 0) return -1;

  /* Read and skip tensor data */
  if (TVMDSPStreamReadI64(stream, &data_size) != 0) return -1;
  if (TVMDSPStreamSkip(stream, (size_t)data_size) != 0) return -1;

  /* Update counts */
  scan->num_ndarrays++;
  scan->total_shape_elems += ndim;

  return 0;
}

static int scan_shape(TVMDSPStream* stream, TVMDSPConstantsScan* scan) {
  uint64_t num_dims;
  if (TVMDSPStreamReadU64(stream, &num_dims) != 0) return -1;
  if (TVMDSPStreamSkip(stream, num_dims * sizeof(int64_t)) != 0) return -1;

  scan->num_shape_objs++;
  scan->total_shape_elems += (int)num_dims;
  return 0;
}

static int scan_string(TVMDSPStream* stream, TVMDSPConstantsScan* scan) {
  uint64_t length;
  if (TVMDSPStreamReadU64(stream, &length) != 0) return -1;
  if (TVMDSPStreamSkip(stream, (size_t)length) != 0) return -1;

  scan->num_string_objs++;
  scan->total_string_bytes += (int)(length + 1);  /* +1 for null terminator */
  return 0;
}

static int prescan_constants(const void* data, size_t size, TVMDSPConstantsScan* scan) {
  TVMDSPStream stream;
  uint64_t num_constants;
  size_t i;
  int ret;

  memset(scan, 0, sizeof(*scan));
  TVMDSPStreamInit(&stream, data, size);

  /* Read constant count */
  if (TVMDSPStreamReadU64(&stream, &num_constants) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  if (num_constants > TVM_DSP_CONST_MAX_SANITY) {
    tvm_dsp_log("ERROR: Too many constants (%llu > %d)\n",
                (unsigned long long)num_constants, TVM_DSP_CONST_MAX_SANITY);
    return TVM_DSP_CONST_ERR_TOO_MANY;
  }

  scan->num_constants = (int)num_constants;

  /* Scan each constant to count resources */
  for (i = 0; i < num_constants; i++) {
    int32_t type_index;
    size_t pos_before = TVMDSPStreamPosition(&stream);

#if defined(TVM_DSP_ALIGNED_WEIGHTS)
    /*
     * TVM's SaveConstantSectionToFileAligned only adds padding before NDArray
     * entries. Detect padding by peeking - if we see zero byte at non-aligned
     * position, it's likely padding, so align first before reading type_index.
     *
     * Valid type_index values are: 0 (None), 1 (Int), 2 (Bool), 3 (Float),
     * 5 (DataType), 65 (Str), 71 (Shape), 72 (NDArray).
     * A zero byte followed by non-zero is almost certainly padding.
     */
    if ((pos_before % 4) != 0) {
      const uint8_t* peek = (const uint8_t*)TVMDSPStreamPeek(&stream, 1);
      if (peek && *peek == 0) {
        /* Zero byte at non-aligned position = padding, align first */
        TVMDSPStreamAlign(&stream, 4);
      }
    }
#endif

    if (TVMDSPStreamReadI32(&stream, &type_index) != 0) {
      return TVM_DSP_CONST_ERR_BUFFER_END;
    }

    switch (type_index) {
      case kTVMFFITensor:
        ret = scan_ndarray(&stream, scan);
        break;
      case kTVMFFIShape:
        ret = scan_shape(&stream, scan);
        break;
      case kTVMFFIStr:
        ret = scan_string(&stream, scan);
        break;
      case kTVMFFIInt:
        if (TVMDSPStreamSkip(&stream, sizeof(int64_t)) != 0) ret = -1;
        else ret = 0;
        break;
      case kTVMFFIFloat:
        if (TVMDSPStreamSkip(&stream, sizeof(double)) != 0) ret = -1;
        else ret = 0;
        break;
      case kTVMFFIDataType:
        if (TVMDSPStreamSkip(&stream, sizeof(DLDataType)) != 0) ret = -1;
        else ret = 0;
        break;
      default:
        tvm_dsp_log("ERROR: Unknown constant type %d at offset %zu during prescan\n",
                    type_index, pos_before);
        return TVM_DSP_CONST_ERR_UNKNOWN_TYPE;
    }

    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

/*---------------------------------------------------------------------------
 * Memory Allocation from L3
 *---------------------------------------------------------------------------*/

static int allocate_pools(const TVMDSPConstantsScan* scan) {
  size_t total_alloc = 0;

  /* Free any existing pools first */
  if (g_constants) {
    tvm_dsp_free(g_constants);
    g_constants = NULL;
  }
  if (g_ndarray_pool) {
    tvm_dsp_free(g_ndarray_pool);
    g_ndarray_pool = NULL;
  }
  if (g_shape_pool) {
    tvm_dsp_free(g_shape_pool);
    g_shape_pool = NULL;
  }
  if (g_string_pool) {
    tvm_dsp_free(g_string_pool);
    g_string_pool = NULL;
  }
  if (g_shape_obj_pool) {
    tvm_dsp_free(g_shape_obj_pool);
    g_shape_obj_pool = NULL;
  }
  if (g_string_obj_pool) {
    tvm_dsp_free(g_string_obj_pool);
    g_string_obj_pool = NULL;
  }

  /* Reset counters */
  g_ndarray_count = 0;
  g_shape_used = 0;
  g_string_used = 0;
  g_shape_obj_count = 0;
  g_string_obj_count = 0;
  g_num_constants = 0;

  /* Allocate constants array */
  if (scan->num_constants > 0) {
    size_t size = scan->num_constants * sizeof(TVMFFIAny);
    g_constants = (TVMFFIAny*)tvm_dsp_alloc(size, 8, TVM_DSP_MEM_MAIN);
    if (!g_constants) {
      tvm_dsp_log("ERROR: Failed to allocate constants array (%zu bytes)\n", size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    memset(g_constants, 0, size);
    total_alloc += size;
  }

  /* Allocate NDArray pool */
  if (scan->num_ndarrays > 0) {
    size_t size = scan->num_ndarrays * sizeof(TVMDSPNDArray);
    g_ndarray_pool = (TVMDSPNDArray*)tvm_dsp_alloc(size, 8, TVM_DSP_MEM_MAIN);
    if (!g_ndarray_pool) {
      tvm_dsp_log("ERROR: Failed to allocate NDArray pool (%zu bytes)\n", size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    g_ndarray_capacity = scan->num_ndarrays;
    total_alloc += size;
  }

  /* Allocate shape pool */
  if (scan->total_shape_elems > 0) {
    size_t size = scan->total_shape_elems * sizeof(int64_t);
    g_shape_pool = (int64_t*)tvm_dsp_alloc(size, 8, TVM_DSP_MEM_MAIN);
    if (!g_shape_pool) {
      tvm_dsp_log("ERROR: Failed to allocate shape pool (%zu bytes)\n", size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    g_shape_capacity = scan->total_shape_elems;
    total_alloc += size;
  }

  /* Allocate string pool */
  if (scan->total_string_bytes > 0) {
    size_t size = scan->total_string_bytes;
    g_string_pool = (char*)tvm_dsp_alloc(size, 1, TVM_DSP_MEM_MAIN);
    if (!g_string_pool) {
      tvm_dsp_log("ERROR: Failed to allocate string pool (%zu bytes)\n", size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    g_string_capacity = scan->total_string_bytes;
    total_alloc += size;
  }

  /* Allocate Shape object pool */
  if (scan->num_shape_objs > 0) {
    size_t size = scan->num_shape_objs * sizeof(TVMDSPShape);
    g_shape_obj_pool = (TVMDSPShape*)tvm_dsp_alloc(size, 8, TVM_DSP_MEM_MAIN);
    if (!g_shape_obj_pool) {
      tvm_dsp_log("ERROR: Failed to allocate Shape object pool (%zu bytes)\n", size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    g_shape_obj_capacity = scan->num_shape_objs;
    total_alloc += size;
  }

  /* Allocate String object pool */
  if (scan->num_string_objs > 0) {
    size_t size = scan->num_string_objs * sizeof(TVMDSPString);
    g_string_obj_pool = (TVMDSPString*)tvm_dsp_alloc(size, 8, TVM_DSP_MEM_MAIN);
    if (!g_string_obj_pool) {
      tvm_dsp_log("ERROR: Failed to allocate String object pool (%zu bytes)\n", size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    g_string_obj_capacity = scan->num_string_objs;
    total_alloc += size;
  }

  tvm_dsp_log("INFO: Allocated %zu bytes in L3 for constant metadata\n", total_alloc);

  return 0;
}

/*---------------------------------------------------------------------------
 * Constant Parsers (second pass - actual parsing)
 *---------------------------------------------------------------------------*/

static int parse_ndarray(TVMDSPStream* stream, TVMFFIAny* out) {
  uint64_t magic, reserved;
  int32_t device_type, device_id, ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t data_size;
  const void* data_ptr;
  TVMDSPNDArray* arr;
  int i;

  /* Read and verify magic number */
  if (TVMDSPStreamReadU64(stream, &magic) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }
  if (magic != TVM_NDARRAY_MAGIC) {
    tvm_dsp_log("ERROR: Invalid NDArray magic (got 0x%llx)\n",
                (unsigned long long)magic);
    return TVM_DSP_CONST_ERR_INVALID_MAGIC;
  }

  /* Read reserved field */
  if (TVMDSPStreamReadU64(stream, &reserved) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  /* Read device info */
  if (TVMDSPStreamReadI32(stream, &device_type) != 0 ||
      TVMDSPStreamReadI32(stream, &device_id) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  /* Read ndim */
  if (TVMDSPStreamReadI32(stream, &ndim) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }
  if (ndim < 0 || ndim > TVM_DSP_PARSE_MAX_NDIM) {
    tvm_dsp_log("ERROR: Invalid ndim=%d\n", ndim);
    return TVM_DSP_CONST_ERR_TOO_MANY;
  }

  /* Read dtype */
  if (TVMDSPStreamRead(stream, &dtype, sizeof(DLDataType)) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  /* Allocate and read shape */
  shape = alloc_shape(ndim);
  if (shape == NULL && ndim > 0) {
    return TVM_DSP_CONST_ERR_SHAPE_FULL;
  }
  for (i = 0; i < ndim; i++) {
    if (TVMDSPStreamReadI64(stream, &shape[i]) != 0) {
      return TVM_DSP_CONST_ERR_BUFFER_END;
    }
  }

  /* Read data size */
  if (TVMDSPStreamReadI64(stream, &data_size) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  /* ZERO-COPY: Get pointer to data in embedded buffer */
  data_ptr = TVMDSPStreamPeek(stream, (size_t)data_size);
  if (data_ptr == NULL) {
    tvm_dsp_log("ERROR: Insufficient data for NDArray (%lld bytes)\n",
                (long long)data_size);
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }
  TVMDSPStreamSkip(stream, (size_t)data_size);

  /* Allocate NDArray from pool */
  arr = alloc_ndarray();
  if (arr == NULL) {
    return TVM_DSP_CONST_ERR_NDARRAY_FULL;
  }

  /* Initialize NDArray object header */
  arr->type_index = kTVMFFITensor;
  arr->ref_counter = 1;  /* Constant - never freed */
  arr->deleter = NULL;

  /*
   * Handle data pointer alignment.
   * On C66x, float/int access requires 4-byte alignment. If the embedded
   * data is unaligned, we must copy to an aligned buffer. This sacrifices
   * zero-copy but ensures correct data access.
   */
#ifdef TVM_DSP_TARGET_C66X
  if (((uintptr_t)data_ptr % 4) != 0) {
    /* Unaligned: allocate aligned buffer and copy */
    void* aligned_data = tvm_dsp_alloc((size_t)data_size, 4, TVM_DSP_MEM_MAIN);
    if (aligned_data == NULL) {
      tvm_dsp_log("ERROR: Failed to allocate aligned buffer for NDArray (%lld bytes)\n",
                  (long long)data_size);
      return TVM_DSP_CONST_ERR_ALLOC_FAIL;
    }
    memcpy(aligned_data, data_ptr, (size_t)data_size);
    arr->data = aligned_data;
    /* Track aligned buffer for cleanup */
    if (g_aligned_buffer_count < TVM_DSP_MAX_ALIGNED_BUFFERS) {
      g_aligned_buffers[g_aligned_buffer_count++] = aligned_data;
    }
    tvm_dsp_log("INFO: Copied unaligned NDArray data to aligned buffer (%lld bytes)\n",
                (long long)data_size);
  } else {
    /* Already aligned: zero-copy */
    arr->data = (void*)data_ptr;
  }
#else
  /* Host: zero-copy always works */
  arr->data = (void*)data_ptr;
#endif

  arr->device.device_type = kDLCPU;
  arr->device.device_id = 0;
  arr->ndim = ndim;
  arr->dtype = dtype;
  arr->shape = shape;
  arr->strides = NULL;
  arr->byte_offset = 0;

  /* Set output as NDArray */
  TVMFFIAnySetNDArray(out, arr);

  return TVM_DSP_CONST_SUCCESS;
}

static int parse_shape(TVMDSPStream* stream, TVMFFIAny* out) {
  uint64_t num_dims;
  int64_t* shape_data;
  TVMDSPShape* shape_obj;
  size_t i;

  if (TVMDSPStreamReadU64(stream, &num_dims) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  shape_data = alloc_shape((int)num_dims);
  if (shape_data == NULL && num_dims > 0) {
    return TVM_DSP_CONST_ERR_SHAPE_FULL;
  }

  for (i = 0; i < num_dims; i++) {
    if (TVMDSPStreamReadI64(stream, &shape_data[i]) != 0) {
      return TVM_DSP_CONST_ERR_BUFFER_END;
    }
  }

  shape_obj = alloc_shape_obj();
  if (shape_obj == NULL) {
    return TVM_DSP_CONST_ERR_TOO_MANY;
  }

  shape_obj->type_index = kTVMFFIShape;
  shape_obj->ref_counter = 1;
  shape_obj->deleter = NULL;
  shape_obj->data = shape_data;
  shape_obj->size = (size_t)num_dims;

  TVMFFIAnySetObject(out, (TVMFFIObject*)shape_obj, kTVMFFIShape);
  return TVM_DSP_CONST_SUCCESS;
}

static int parse_string(TVMDSPStream* stream, TVMFFIAny* out) {
  uint64_t length;
  char* str_data;
  TVMDSPString* str_obj;
  size_t i;

  if (TVMDSPStreamReadU64(stream, &length) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  str_data = alloc_string((size_t)length);
  if (str_data == NULL) {
    return TVM_DSP_CONST_ERR_STRING_FULL;
  }

  for (i = 0; i < length; i++) {
    if (TVMDSPStreamRead(stream, &str_data[i], 1) != 0) {
      return TVM_DSP_CONST_ERR_BUFFER_END;
    }
  }
  str_data[length] = '\0';

  str_obj = alloc_string_obj();
  if (str_obj == NULL) {
    return TVM_DSP_CONST_ERR_TOO_MANY;
  }

  str_obj->type_index = kTVMFFIStr;
  str_obj->ref_counter = 1;
  str_obj->deleter = NULL;
  str_obj->data = str_data;
  str_obj->size = (size_t)length;

  TVMFFIAnySetObject(out, (TVMFFIObject*)str_obj, kTVMFFIStr);
  return TVM_DSP_CONST_SUCCESS;
}

static int parse_int(TVMDSPStream* stream, TVMFFIAny* out) {
  int64_t value;
  if (TVMDSPStreamReadI64(stream, &value) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }
  TVMFFIAnySetInt(out, value);
  return TVM_DSP_CONST_SUCCESS;
}

static int parse_float(TVMDSPStream* stream, TVMFFIAny* out) {
  double value;
  if (TVMDSPStreamReadF64(stream, &value) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }
  TVMFFIAnySetFloat(out, value);
  return TVM_DSP_CONST_SUCCESS;
}

static int parse_dtype(TVMDSPStream* stream, TVMFFIAny* out) {
  DLDataType dtype;
  if (TVMDSPStreamRead(stream, &dtype, sizeof(DLDataType)) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }
  TVMFFIAnySetDataType(out, dtype);
  return TVM_DSP_CONST_SUCCESS;
}

/*---------------------------------------------------------------------------
 * Public API Implementation
 *---------------------------------------------------------------------------*/

void TVMDSPConstantsInit(void) {
  /* Free any existing pools */
  if (g_constants) {
    tvm_dsp_free(g_constants);
    g_constants = NULL;
  }
  if (g_ndarray_pool) {
    tvm_dsp_free(g_ndarray_pool);
    g_ndarray_pool = NULL;
  }
  if (g_shape_pool) {
    tvm_dsp_free(g_shape_pool);
    g_shape_pool = NULL;
  }
  if (g_string_pool) {
    tvm_dsp_free(g_string_pool);
    g_string_pool = NULL;
  }
  if (g_shape_obj_pool) {
    tvm_dsp_free(g_shape_obj_pool);
    g_shape_obj_pool = NULL;
  }
  if (g_string_obj_pool) {
    tvm_dsp_free(g_string_obj_pool);
    g_string_obj_pool = NULL;
  }

  /* Reset all state */
  g_ndarray_capacity = 0;
  g_ndarray_count = 0;
  g_shape_capacity = 0;
  g_shape_used = 0;
  g_string_capacity = 0;
  g_string_used = 0;
  g_shape_obj_capacity = 0;
  g_shape_obj_count = 0;
  g_string_obj_capacity = 0;
  g_string_obj_count = 0;
  g_num_constants = 0;

  g_initialized = 1;
}

int TVMDSPConstantsParse(const void* data, size_t size) {
  TVMDSPConstantsScan scan;
  TVMDSPStream stream;
  uint64_t num_constants;
  size_t i;
  int ret;

  if (data == NULL) {
    return TVM_DSP_CONST_ERR_NULL_INPUT;
  }

  /* Phase 1: Pre-scan to count resources */
  tvm_dsp_log("INFO: Pre-scanning weights.bin (%zu bytes)...\n", size);
  ret = prescan_constants(data, size, &scan);
  if (ret != 0) {
    tvm_dsp_log("ERROR: Pre-scan failed: %s\n", TVMDSPConstantsErrorString(ret));
    return ret;
  }

  tvm_dsp_log("INFO: Found %d constants (%d NDArrays, %d shapes, %d strings)\n",
              scan.num_constants, scan.num_ndarrays,
              scan.num_shape_objs, scan.num_string_objs);
  tvm_dsp_log("INFO: Need %d shape elements, %d string bytes\n",
              scan.total_shape_elems, scan.total_string_bytes);

  /* Phase 2: Allocate pools from L3 memory */
  ret = allocate_pools(&scan);
  if (ret != 0) {
    return ret;
  }

  /* Phase 3: Parse constants (second pass) */
  TVMDSPStreamInit(&stream, data, size);

  /* Skip constant count (already read in prescan) */
  if (TVMDSPStreamReadU64(&stream, &num_constants) != 0) {
    return TVM_DSP_CONST_ERR_BUFFER_END;
  }

  for (i = 0; i < num_constants; i++) {
    int32_t type_index;
    size_t pos_before = TVMDSPStreamPosition(&stream);

#if defined(TVM_DSP_ALIGNED_WEIGHTS)
    /*
     * TVM's SaveConstantSectionToFileAligned only adds padding before NDArray
     * entries. Detect padding by peeking - if we see zero byte at non-aligned
     * position, it's likely padding, so align first before reading type_index.
     */
    if ((pos_before % 4) != 0) {
      const uint8_t* peek = (const uint8_t*)TVMDSPStreamPeek(&stream, 1);
      if (peek && *peek == 0) {
        /* Zero byte at non-aligned position = padding, align first */
        TVMDSPStreamAlign(&stream, 4);
      }
    }
#endif

    if (TVMDSPStreamReadI32(&stream, &type_index) != 0) {
      tvm_dsp_log("ERROR: Failed to read type index for constant %zu\n", i);
      return TVM_DSP_CONST_ERR_BUFFER_END;
    }

    TVMFFIAnySetNone(&g_constants[i]);

    switch (type_index) {
      case kTVMFFITensor:
        ret = parse_ndarray(&stream, &g_constants[i]);
        break;
      case kTVMFFIShape:
        ret = parse_shape(&stream, &g_constants[i]);
        break;
      case kTVMFFIStr:
        ret = parse_string(&stream, &g_constants[i]);
        break;
      case kTVMFFIInt:
        ret = parse_int(&stream, &g_constants[i]);
        break;
      case kTVMFFIFloat:
        ret = parse_float(&stream, &g_constants[i]);
        break;
      case kTVMFFIDataType:
        ret = parse_dtype(&stream, &g_constants[i]);
        break;
      default:
        tvm_dsp_log("ERROR: Unknown constant type %d at index %zu\n",
                    type_index, i);
        return TVM_DSP_CONST_ERR_UNKNOWN_TYPE;
    }

    if (ret != TVM_DSP_CONST_SUCCESS) {
      tvm_dsp_log("ERROR: Failed to parse constant %zu (type=%d, err=%d)\n",
                  i, type_index, ret);
      return ret;
    }
  }

  g_num_constants = (int)num_constants;
  g_initialized = 1;

  tvm_dsp_log("INFO: Successfully parsed %d constants\n", g_num_constants);

  return g_num_constants;
}

TVMFFIAny* TVMDSPConstantsGet(int* count) {
  if (!g_initialized) {
    if (count) *count = 0;
    return NULL;
  }
  if (count) {
    *count = g_num_constants;
  }
  return g_constants;
}

TVMFFIAny* TVMDSPConstantGetByIndex(int index) {
  if (!g_initialized || index < 0 || index >= g_num_constants) {
    return NULL;
  }
  return &g_constants[index];
}

int TVMDSPConstantsCount(void) {
  return g_initialized ? g_num_constants : 0;
}

const char* TVMDSPConstantsErrorString(int err) {
  switch (err) {
    case TVM_DSP_CONST_SUCCESS:
      return "Success";
    case TVM_DSP_CONST_ERR_NULL_INPUT:
      return "Null input pointer";
    case TVM_DSP_CONST_ERR_INVALID_MAGIC:
      return "Invalid NDArray magic number";
    case TVM_DSP_CONST_ERR_TOO_MANY:
      return "Too many constants or dimensions";
    case TVM_DSP_CONST_ERR_SHAPE_FULL:
      return "Shape pool exhausted";
    case TVM_DSP_CONST_ERR_STRING_FULL:
      return "String pool exhausted";
    case TVM_DSP_CONST_ERR_UNKNOWN_TYPE:
      return "Unknown constant type";
    case TVM_DSP_CONST_ERR_BUFFER_END:
      return "Unexpected end of buffer";
    case TVM_DSP_CONST_ERR_NDARRAY_FULL:
      return "NDArray pool exhausted";
    case TVM_DSP_CONST_ERR_NOT_INIT:
      return "Constants not initialized";
    case TVM_DSP_CONST_ERR_ALLOC_FAIL:
      return "Memory allocation failed";
    default:
      return "Unknown error";
  }
}

void TVMDSPConstantsCleanup(void) {
  int i;

  /* Free aligned buffers allocated for unaligned C66x data */
  for (i = 0; i < g_aligned_buffer_count; i++) {
    if (g_aligned_buffers[i] != NULL) {
      tvm_dsp_free(g_aligned_buffers[i]);
      g_aligned_buffers[i] = NULL;
    }
  }
  g_aligned_buffer_count = 0;

  /* Free all memory pools */
  if (g_constants) {
    tvm_dsp_free(g_constants);
    g_constants = NULL;
  }
  if (g_ndarray_pool) {
    tvm_dsp_free(g_ndarray_pool);
    g_ndarray_pool = NULL;
  }
  if (g_shape_pool) {
    tvm_dsp_free(g_shape_pool);
    g_shape_pool = NULL;
  }
  if (g_string_pool) {
    tvm_dsp_free(g_string_pool);
    g_string_pool = NULL;
  }
  if (g_shape_obj_pool) {
    tvm_dsp_free(g_shape_obj_pool);
    g_shape_obj_pool = NULL;
  }
  if (g_string_obj_pool) {
    tvm_dsp_free(g_string_obj_pool);
    g_string_obj_pool = NULL;
  }

  /* Reset all counters and capacities */
  g_ndarray_capacity = 0;
  g_ndarray_count = 0;
  g_shape_capacity = 0;
  g_shape_used = 0;
  g_string_capacity = 0;
  g_string_used = 0;
  g_shape_obj_capacity = 0;
  g_shape_obj_count = 0;
  g_string_obj_capacity = 0;
  g_string_obj_count = 0;
  g_num_constants = 0;

  /* Mark as uninitialized */
  g_initialized = 0;
}
