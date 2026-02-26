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
 * \file vm/storage.cpp
 * \brief Storage implementation using C++14 utilities
 *
 * This is a C++ implementation of the storage module using:
 * - ScopeGuard for automatic cleanup on error paths
 * - TypedHandle for type-safe storage/NDArray access
 *
 * The C API (extern "C") is preserved for compatibility.
 */

#include "storage.h"

/* C headers - no extern "C" wrapper needed (they have their own guards) */
#include "../core/config.h"
#include "../platform/dsp_platform.h"
#include "../platform/dsp_memory.h"

#include "../cpp/scope_guard.h"
#include "../cpp/typed_handle.h"

#include <cstring>

/*
 * =============================================================================
 * C API IMPLEMENTATION
 * =============================================================================
 */

extern "C" {

/* Forward declaration for storage view deleter */
void TVMDSPNDArrayStorageViewDeleter(TVMFFIObject* obj);

/*---------------------------------------------------------------------------
 * Storage Deleter
 *---------------------------------------------------------------------------*/
void TVMDSPStorageDeleter(TVMFFIObject* obj) {
  auto storage = tvm::dsp::TypedHandle<TVMDSPStorage>::FromRaw(obj);

  /* Free the buffer data */
  if (storage->buffer.data != nullptr) {
    tvm_dsp_free(storage->buffer.data);
    storage->buffer.data = nullptr;
  }

  /* Free the storage object itself */
  tvm_dsp_free(storage.get());
}

/*---------------------------------------------------------------------------
 * Storage Creation
 *---------------------------------------------------------------------------*/

TVMDSPStorage* TVMDSPStorageAlloc(size_t size, DLDevice device,
                                   DLDataType dtype_hint) {
  (void)dtype_hint;  /* Currently unused, could be used for alignment hints */

  /*
   * Memory allocation strategy for DSP:
   * - Storage objects go to L3 (small metadata)
   * - Storage DATA <=32KB goes to L2 first (fast compute), fallback to L3
   * - Storage DATA >32KB goes directly to L3
   *
   * With proper reference counting (fixed memory leak), intermediate
   * Storage is freed after use, keeping L2 available for new allocations.
   */

  /* Allocate storage object from L3 */
  void* storage_mem = tvm_dsp_alloc(sizeof(TVMDSPStorage),
                                     TVM_DSP_CACHE_LINE_SIZE,
                                     TVM_DSP_MEM_MAIN);
  if (storage_mem == nullptr) {
    return nullptr;
  }

  /* Use ScopeGuard to ensure storage is freed on error */
  auto storage_guard = tvm::dsp::MakeScopeGuard([storage_mem]() {
    tvm_dsp_free(storage_mem);
  });

  auto storage = tvm::dsp::TypedHandle<TVMDSPStorage>::FromRaw(storage_mem);

  /* Initialize object header */
  storage->type_index = TVM_DSP_STORAGE_TYPE_INDEX;
  storage->ref_counter = 1;
  storage->deleter = TVMDSPStorageDeleter;

  /* Allocate data buffer - L2 for small, L3 for large */
  void* data_ptr = nullptr;
  TVMDSPMemoryPool pool;

  if (size <= TVM_DSP_L2_ALLOC_THRESHOLD) {
    pool = TVM_DSP_MEM_FAST;
    data_ptr = tvm_dsp_alloc(size, TVM_DSP_CACHE_LINE_SIZE, pool);
    if (data_ptr == nullptr) {
      pool = TVM_DSP_MEM_MAIN;
      data_ptr = tvm_dsp_alloc(size, TVM_DSP_CACHE_LINE_SIZE, pool);
    }
  } else {
    pool = TVM_DSP_MEM_MAIN;
    data_ptr = tvm_dsp_alloc(size, TVM_DSP_CACHE_LINE_SIZE, pool);
  }

  if (data_ptr == nullptr && size > 0) {
    /* Failed to allocate data buffer - storage_guard will free storage */
    return nullptr;
  }

  /* Initialize buffer */
  storage->buffer.data = data_ptr;
  storage->buffer.size = size;
  storage->buffer.device = device;
  storage->buffer.pool = pool;

  /* Success - dismiss the guard so storage isn't freed */
  storage_guard.Dismiss();

  return storage.get();
}

/*---------------------------------------------------------------------------
 * NDArray Allocation from Storage
 *---------------------------------------------------------------------------*/

TVMDSPNDArray* TVMDSPStorageAllocNDArray(TVMDSPStorage* storage, int64_t offset,
                                          const int64_t* shape, int32_t ndim,
                                          DLDataType dtype) {
  if (storage == nullptr) {
    return nullptr;
  }

  /* Validate offset */
  if (offset < 0 || static_cast<size_t>(offset) > storage->buffer.size) {
    return nullptr;
  }

  /* Allocate NDArray object from L3 (L2 reserved for workspace) */
  void* arr_mem = tvm_dsp_alloc(sizeof(TVMDSPNDArray),
                                 TVM_DSP_CACHE_LINE_SIZE,
                                 TVM_DSP_MEM_MAIN);
  if (arr_mem == nullptr) {
    return nullptr;
  }

  auto arr = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(arr_mem);

  /* Initialize object header */
  arr->type_index = kTVMFFITensor;
  arr->ref_counter = 1;
  arr->deleter = TVMDSPNDArrayStorageViewDeleter;

  /* Copy shape to inline storage */
  for (int32_t i = 0; i < ndim && i < TVM_DSP_NDARRAY_MAX_NDIM; i++) {
    arr->shape_storage[i] = shape[i];
  }

  /* Calculate data pointer with offset */
  void* data_ptr = static_cast<char*>(storage->buffer.data) + offset;

  /* Initialize DLTensor fields */
  arr->data = data_ptr;
  arr->ndim = ndim;
  arr->dtype = dtype;
  arr->device = storage->buffer.device;
  arr->shape = arr->shape_storage;
  arr->strides = nullptr;  /* Contiguous */
  arr->byte_offset = 0;
  arr->flags = 0;  /* Does NOT own data - storage owns it */
  arr->_pad = 0;

  /* Increment storage ref count to keep it alive while NDArray exists */
  TVMDSPStorageIncRef(storage);

  /* Store reference to storage for cleanup in deleter */
  arr->storage_ref = storage;

  return arr.get();
}

/*---------------------------------------------------------------------------
 * NDArray Storage View Deleter
 *
 * This deleter is used for NDArrays created from storage. It does NOT
 * free the data buffer (owned by storage) but releases the reference
 * to the storage so it can be freed when all views are gone.
 *---------------------------------------------------------------------------*/
void TVMDSPNDArrayStorageViewDeleter(TVMFFIObject* obj) {
  auto arr = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(obj);
  auto storage = tvm::dsp::TypedHandle<TVMDSPStorage>(
      static_cast<TVMDSPStorage*>(arr->storage_ref));

  /* Data is owned by storage, not by this NDArray - don't free it */

  /* Decrement storage reference count (may free storage) */
  if (storage.IsValid()) {
    TVMDSPStorageDecRef(storage.get());
  }

  /* Free the NDArray object itself */
  tvm_dsp_free(arr.get());
}

}  /* extern "C" */
