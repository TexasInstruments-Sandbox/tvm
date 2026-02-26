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
 * \file vm/storage.h
 * \brief Storage container for TVM DSP Runtime
 *
 * Storage represents a raw memory buffer that can be used to allocate
 * multiple NDArray tensors. This matches TVM's vm::Storage object.
 *
 * TVM's Storage layout:
 *   class StorageObj : public Object {
 *     Buffer buffer;
 *     Allocator* allocator;
 *   }
 *
 * On DSP, we don't need dynamic allocators - we use the fixed L2/L3 pools.
 */

#ifndef TVM_DSP_RUNTIME_VM_STORAGE_H_
#define TVM_DSP_RUNTIME_VM_STORAGE_H_

#include <dlpack/dlpack.h>
#include "../ffi/ffi_types.h"
#include "../container/ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Storage type index - in the dynamic object range (>=128).
 * Must not collide with any kTVMFFI* static type index.
 * Upstream TVM assigns this dynamically at runtime; we use a
 * fixed value since the DSP runtime has no type registry. */
#define TVM_DSP_STORAGE_TYPE_INDEX 128

/*!
 * \brief Buffer structure representing raw allocated memory
 */
typedef struct TVMDSPBuffer {
  void* data;          /* Pointer to allocated memory */
  size_t size;         /* Size in bytes */
  DLDevice device;     /* Device where memory resides */
  int pool;            /* Memory pool (TVM_DSP_MEM_FAST or TVM_DSP_MEM_MAIN) */
} TVMDSPBuffer;

/*!
 * \brief Storage object that holds a memory buffer
 *
 * Storage is used by vm.builtin.alloc_storage to allocate raw memory,
 * which can then be used by vm.builtin.alloc_tensor to create NDArrays.
 */
typedef struct TVMDSPStorage {
  /* === TVMFFIObject Header (16 bytes) === */
  int32_t type_index;    /* Storage type index */
  int32_t ref_counter;   /* Reference count */
  union {
    void (*deleter)(TVMFFIObject* self);
    int64_t _align;
  };

  /* === Storage Fields === */
  TVMDSPBuffer buffer;   /* The underlying memory buffer */
} TVMDSPStorage;

/* ---------------------------------------------------------------------------
 * Storage Creation and Destruction
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Allocate a new storage buffer
 * \param size Size in bytes to allocate
 * \param device Target device
 * \param dtype_hint Data type hint (used to determine alignment)
 * \return Allocated storage, or NULL on failure
 *
 * This is called by vm.builtin.alloc_storage.
 */
TVMDSPStorage* TVMDSPStorageAlloc(size_t size, DLDevice device, DLDataType dtype_hint);

/*!
 * \brief Create an NDArray from storage at a given offset
 * \param storage The storage to use
 * \param offset Byte offset into the storage buffer
 * \param shape Shape of the tensor
 * \param ndim Number of dimensions
 * \param dtype Data type of the tensor
 * \return Created NDArray, or NULL on failure
 *
 * This is called by vm.builtin.alloc_tensor.
 * The NDArray shares the storage's data buffer (with offset).
 */
TVMDSPNDArray* TVMDSPStorageAllocNDArray(TVMDSPStorage* storage, int64_t offset,
                                          const int64_t* shape, int32_t ndim,
                                          DLDataType dtype);

/* ---------------------------------------------------------------------------
 * Storage Accessors
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Get the data pointer from storage
 */
static inline void* TVMDSPStorageGetData(TVMDSPStorage* storage) {
  return storage ? storage->buffer.data : NULL;
}

/*!
 * \brief Get the buffer size
 */
static inline size_t TVMDSPStorageGetSize(TVMDSPStorage* storage) {
  return storage ? storage->buffer.size : 0;
}

/* ---------------------------------------------------------------------------
 * Reference Counting
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Increment storage reference count
 */
static inline void TVMDSPStorageIncRef(TVMDSPStorage* storage) {
  if (storage) storage->ref_counter++;
}

/*!
 * \brief Decrement storage reference count (may free)
 */
static inline void TVMDSPStorageDecRef(TVMDSPStorage* storage) {
  if (storage) {
    storage->ref_counter--;
    if (storage->ref_counter == 0 && storage->deleter) {
      storage->deleter((TVMFFIObject*)storage);
    }
  }
}

/* ---------------------------------------------------------------------------
 * Type Checking
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Check if an FFI object is a Storage
 */
static inline int TVMDSPIsStorage(const TVMFFIObject* obj) {
  return obj && obj->type_index == TVM_DSP_STORAGE_TYPE_INDEX;
}

/*!
 * \brief Cast TVMFFIAny to Storage (with type check)
 */
static inline TVMDSPStorage* TVMDSPAnyAsStorage(const TVMFFIAny* any) {
  if (any && any->type_index == TVM_DSP_STORAGE_TYPE_INDEX) {
    return (TVMDSPStorage*)any->v_obj;
  }
  return NULL;
}

/*!
 * \brief Set TVMFFIAny to hold a Storage reference
 */
static inline void TVMDSPAnySetStorage(TVMFFIAny* any, TVMDSPStorage* storage) {
  any->type_index = TVM_DSP_STORAGE_TYPE_INDEX;
  any->small_len = 0;
  any->v_obj = (TVMFFIObject*)storage;
  TVMDSPStorageIncRef(storage);
}

/* ---------------------------------------------------------------------------
 * Internal: Default deleter
 * ---------------------------------------------------------------------------*/

void TVMDSPStorageDeleter(TVMFFIObject* obj);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_DSP_RUNTIME_VM_STORAGE_H_ */
