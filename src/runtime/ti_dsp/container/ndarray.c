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
 * \file container/ndarray.c
 * \brief NDArray implementation for TVM DSP Runtime
 */

#include "ndarray.h"
#include "../platform/dsp_platform.h"
#include "../platform/dsp_memory.h"
#include <string.h>
#include <stddef.h>

/*
 * Verify layout at compile time (where possible) and runtime.
 *
 * The critical requirement is that DLTensor fields start at offset
 * sizeof(TVMFFIObject) = 16 bytes from the object start.
 */

/* Helper: compute data size from shape and dtype */
static size_t compute_data_size(const int64_t* shape, int32_t ndim, DLDataType dtype) {
  int64_t num = 1;
  int32_t i;
  for (i = 0; i < ndim; i++) {
    num *= shape[i];
  }
  /* Handle uint1 stored as uint8 */
  if (dtype.code == kDLUInt && dtype.bits == 1 && dtype.lanes == 1) {
    return (size_t)num;
  }
  return (size_t)((num * dtype.bits * dtype.lanes + 7) / 8);
}

/*---------------------------------------------------------------------------
 * NDArray Deleter
 *---------------------------------------------------------------------------*/
void TVMDSPNDArrayDeleter(TVMFFIObject* obj) {
  TVMDSPNDArray* arr = (TVMDSPNDArray*)obj;

  /* Free data buffer if we own it */
  if (arr->flags & TVM_DSP_NDARRAY_OWNS_DATA) {
    if (arr->data != NULL) {
      tvm_dsp_free(arr->data);
      arr->data = NULL;
    }
  }

  /* Free the NDArray object itself */
  tvm_dsp_free(arr);
}

/*---------------------------------------------------------------------------
 * NDArray Creation
 *---------------------------------------------------------------------------*/

TVMDSPNDArray* TVMDSPNDArrayAlloc(const int64_t* shape, int32_t ndim,
                                   DLDataType dtype, DLDevice device) {
  TVMDSPNDArray* arr;
  size_t data_size;
  void* data_ptr;
  int32_t i;

  /* Validate inputs */
  if (ndim < 0 || ndim > TVM_DSP_NDARRAY_MAX_NDIM) {
    return NULL;
  }

  /*
   * Memory allocation strategy for DSP:
   * - NDArray objects and data buffers go to L3 (main memory)
   * - L2 (fast memory) is reserved for TVMBackendAllocWorkspace
   *   which is used for hot inner loop computation buffers
   */

  /* Allocate NDArray object from L3 */
  arr = (TVMDSPNDArray*)tvm_dsp_alloc(sizeof(TVMDSPNDArray),
                                       TVM_DSP_CACHE_LINE_SIZE,
                                       TVM_DSP_MEM_MAIN);
  if (arr == NULL) {
    return NULL;
  }

  /* Initialize object header */
  arr->type_index = kTVMFFITensor;
  arr->ref_counter = 1;
  arr->deleter = TVMDSPNDArrayDeleter;

  /* Copy shape to inline storage */
  for (i = 0; i < ndim; i++) {
    arr->shape_storage[i] = shape[i];
  }

  /* Initialize DLTensor fields */
  arr->ndim = ndim;
  arr->dtype = dtype;
  arr->device = device;
  arr->shape = arr->shape_storage;
  arr->strides = NULL;  /* Contiguous */
  arr->byte_offset = 0;
  arr->flags = TVM_DSP_NDARRAY_OWNS_DATA;
  arr->_pad = 0;
  arr->storage_ref = NULL;

  /* Allocate data buffer from L3 */
  data_size = compute_data_size(shape, ndim, dtype);
  if (data_size > 0) {
    data_ptr = tvm_dsp_alloc(data_size, TVM_DSP_CACHE_LINE_SIZE, TVM_DSP_MEM_MAIN);
    if (data_ptr == NULL) {
      /* Failed to allocate data, free the object */
      tvm_dsp_free(arr);
      return NULL;
    }
    /* Zero-initialize data */
    memset(data_ptr, 0, data_size);
    arr->data = data_ptr;
  } else {
    arr->data = NULL;
  }

  return arr;
}

TVMDSPNDArray* TVMDSPNDArrayFromData(void* data, const int64_t* shape,
                                      int32_t ndim, DLDataType dtype,
                                      DLDevice device) {
  TVMDSPNDArray* arr;
  int32_t i;

  /* Validate inputs */
  if (ndim < 0 || ndim > TVM_DSP_NDARRAY_MAX_NDIM) {
    return NULL;
  }

  /* Allocate NDArray object from L3 (L2 reserved for workspace) */
  arr = (TVMDSPNDArray*)tvm_dsp_alloc(sizeof(TVMDSPNDArray),
                                       TVM_DSP_CACHE_LINE_SIZE,
                                       TVM_DSP_MEM_MAIN);
  if (arr == NULL) {
    return NULL;
  }

  /* Initialize object header */
  arr->type_index = kTVMFFITensor;
  arr->ref_counter = 1;
  arr->deleter = TVMDSPNDArrayDeleter;

  /* Copy shape to inline storage */
  for (i = 0; i < ndim; i++) {
    arr->shape_storage[i] = shape[i];
  }

  /* Initialize DLTensor fields - data NOT owned */
  arr->data = data;
  arr->ndim = ndim;
  arr->dtype = dtype;
  arr->device = device;
  arr->shape = arr->shape_storage;
  arr->strides = NULL;
  arr->byte_offset = 0;
  arr->flags = 0;  /* Does NOT own data */
  arr->_pad = 0;
  arr->storage_ref = NULL;

  return arr;
}

/*---------------------------------------------------------------------------
 * NDArray Copy Operations
 *---------------------------------------------------------------------------*/

int TVMDSPNDArrayCopyFromBytes(const void* src, size_t nbytes, TVMDSPNDArray* dst) {
  size_t dst_size;

  if (src == NULL || dst == NULL) {
    return -1;
  }

  dst_size = TVMDSPNDArrayDataSize(dst);
  if (nbytes != dst_size) {
    return -1;  /* Size mismatch */
  }

  if (dst->data == NULL) {
    return -1;
  }

  memcpy(dst->data, src, nbytes);
  return 0;
}

int TVMDSPNDArrayCopyToBytes(const TVMDSPNDArray* src, void* dst, size_t nbytes) {
  size_t src_size;

  if (src == NULL || dst == NULL) {
    return -1;
  }

  src_size = TVMDSPNDArrayDataSize(src);
  if (nbytes < src_size) {
    return -1;  /* Buffer too small */
  }

  if (src->data == NULL) {
    return -1;
  }

  memcpy(dst, src->data, src_size);
  return 0;
}

int TVMDSPNDArrayCopy(const TVMDSPNDArray* src, TVMDSPNDArray* dst) {
  size_t src_size, dst_size;
  int32_t i;

  if (src == NULL || dst == NULL) {
    return -1;
  }

  /* Verify shape matches */
  if (src->ndim != dst->ndim) {
    return -1;
  }
  for (i = 0; i < src->ndim; i++) {
    if (src->shape[i] != dst->shape[i]) {
      return -1;
    }
  }

  /* Verify dtype matches */
  if (src->dtype.code != dst->dtype.code ||
      src->dtype.bits != dst->dtype.bits ||
      src->dtype.lanes != dst->dtype.lanes) {
    return -1;
  }

  src_size = TVMDSPNDArrayDataSize(src);
  dst_size = TVMDSPNDArrayDataSize(dst);
  if (src_size != dst_size) {
    return -1;
  }

  if (src->data == NULL || dst->data == NULL) {
    return (src_size == 0) ? 0 : -1;
  }

  memcpy(dst->data, src->data, src_size);
  return 0;
}

/*---------------------------------------------------------------------------
 * Layout Verification (called during init)
 *---------------------------------------------------------------------------*/

int TVMDSPNDArrayVerifyLayout(void) {
  /*
   * Verify that the 'data' field is at offset sizeof(TVMFFIObject) = 16.
   * This is critical for TVMFFINDArrayGetDLTensorPtr compatibility.
   */
  size_t data_offset = offsetof(TVMDSPNDArray, data);
  size_t expected_offset = sizeof(TVMFFIObject);

  if (data_offset != expected_offset) {
    /* Layout mismatch - this is a fatal error */
    tvm_dsp_log("ERROR: NDArray layout error: data at offset %zu, expected %zu\n",
                data_offset, expected_offset);
    return -1;
  }

  return 0;
}
