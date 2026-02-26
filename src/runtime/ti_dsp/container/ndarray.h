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
 * \file container/ndarray.h
 * \brief NDArray container for TVM DSP Runtime
 *
 * This provides an ABI-compatible NDArray implementation for bare-metal DSP.
 * The memory layout matches TVM's ffi::NDArrayObj exactly.
 *
 * CRITICAL: TVM-generated code calls TVMFFINDArrayGetDLTensorPtr() which
 * expects the DLTensor to be at offset sizeof(TVMFFIObject) from the object.
 *
 * TVM's NDArrayObj C++ layout (multiple inheritance):
 *   class NDArrayObj : public Object, public DLTensor { ... }
 *
 * Our C layout must match:
 *   - TVMFFIObject header (16 bytes)
 *   - DLTensor fields (directly embedded, not pointer)
 *   - shape_data_ backing storage
 */

#ifndef TVM_DSP_RUNTIME_CONTAINER_NDARRAY_H_
#define TVM_DSP_RUNTIME_CONTAINER_NDARRAY_H_

#include <dlpack/dlpack.h>
#include <stddef.h>  /* offsetof for static_assert */
#include "../ffi/ffi_types.h"
#include "../core/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum dimensions for inline shape storage (from core/config.h) */
#define TVM_DSP_NDARRAY_MAX_NDIM TVM_DSP_MAX_NDIM

/*!
 * \brief NDArray object structure matching TVM's NDArrayObj layout
 *
 * Layout (must match TVMFFINDArrayGetDLTensorPtr expectation):
 *   Offset 0:   TVMFFIObject header
 *   Offset 16:  DLTensor fields (data, device, ndim, dtype, shape, strides, byte_offset)
 *   After DLTensor: shape backing storage (Optional<Shape> in C++)
 *
 * Note: We embed DLTensor fields directly (not as a struct member) to match
 * the C++ multiple inheritance layout where DLTensor is a base class.
 */
typedef struct TVMDSPNDArray {
  /* === TVMFFIObject Header (16 bytes) === */
  int32_t type_index;    /* Must be kTVMFFITensor (70) */
  int32_t ref_counter;   /* Reference count */
  union {
    void (*deleter)(TVMFFIObject* self);
    int64_t _align;      /* Ensure 8-byte alignment */
  };

  /* === DLTensor Fields (embedded, not nested struct) === */
  /* These must start at exactly offset 16 (sizeof(TVMFFIObject)) */
  void* data;            /* Pointer to tensor data */
  DLDevice device;       /* Device info */
  int32_t ndim;          /* Number of dimensions */
  DLDataType dtype;      /* Data type */
  int64_t* shape;        /* Shape array pointer */
  int64_t* strides;      /* Strides array (NULL if contiguous) */
  uint64_t byte_offset;  /* Byte offset into data */

  /* === Shape Backing Storage === */
  /* This corresponds to Optional<Shape> shape_data_ in TVM */
  int64_t shape_storage[TVM_DSP_NDARRAY_MAX_NDIM];
  int64_t strides_storage[TVM_DSP_NDARRAY_MAX_NDIM];

  /* Internal flags */
  uint32_t flags;
  uint32_t _pad;

  /* Reference to backing storage (for storage views created via TVMDSPStorageAllocNDArray) */
  void* storage_ref;
} TVMDSPNDArray;

/* Flags for NDArray */
#define TVM_DSP_NDARRAY_OWNS_DATA  0x01  /* NDArray owns data buffer */

/* ---------------------------------------------------------------------------
 * Compile-time layout verification
 * ---------------------------------------------------------------------------*/

/*
 * Verify that DLTensor fields start at correct offset.
 * TVMFFINDArrayGetDLTensorPtr does: (char*)obj + sizeof(TVMFFIObject)
 * which expects 'data' field to be at offset 16.
 */
#if !defined(__TI_COMPILER_VERSION__)
/* Standard C11 static_assert for host builds */
#define TVM_DSP_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
/* TI compiler may not support _Static_assert, use compile-time array trick */
#define TVM_DSP_STATIC_ASSERT(cond, msg) \
  typedef char static_assertion_##__LINE__[(cond) ? 1 : -1]
#endif

/* Verify 'data' field is at offset 16 (sizeof(TVMFFIObject)) */
/* TVMFFINDArrayGetDLTensorPtr expects DLTensor at this offset */
#ifdef __cplusplus
static_assert(offsetof(TVMDSPNDArray, data) == sizeof(TVMFFIObject),
              "DLTensor fields must start at offset sizeof(TVMFFIObject)");
static_assert(sizeof(TVMFFIObject) == 16,
              "TVMFFIObject must be exactly 16 bytes");
#endif

/* ---------------------------------------------------------------------------
 * NDArray Creation Functions (DSP-specific allocation)
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Allocate an empty NDArray with given shape and dtype
 * \param shape Array of dimension sizes
 * \param ndim Number of dimensions
 * \param dtype Data type
 * \param device Target device
 * \return Allocated NDArray, or NULL on failure
 *
 * The returned NDArray owns its data buffer and will free it on destruction.
 */
TVMDSPNDArray* TVMDSPNDArrayAlloc(const int64_t* shape, int32_t ndim,
                                   DLDataType dtype, DLDevice device);

/*!
 * \brief Create an NDArray that wraps existing data (no copy, no ownership)
 * \param data Pointer to existing data buffer
 * \param shape Array of dimension sizes
 * \param ndim Number of dimensions
 * \param dtype Data type
 * \param device Target device
 * \return Created NDArray, or NULL on failure
 *
 * The returned NDArray does NOT own the data buffer.
 * Caller must ensure data remains valid for NDArray lifetime.
 */
TVMDSPNDArray* TVMDSPNDArrayFromData(void* data, const int64_t* shape,
                                      int32_t ndim, DLDataType dtype,
                                      DLDevice device);

/* ---------------------------------------------------------------------------
 * NDArray Accessors (TVM-compatible)
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Get DLTensor pointer from NDArray (TVM API compatible)
 *
 * This function matches TVMFFINDArrayGetDLTensorPtr from tvm/ffi/c_api.h
 * It returns a pointer to offset sizeof(TVMFFIObject) from the object.
 */
static inline DLTensor* TVMDSPNDArrayGetDLTensor(TVMDSPNDArray* arr) {
  /* Return pointer to 'data' field, which is the start of DLTensor */
  return (DLTensor*)(&arr->data);
}

/*!
 * \brief Get number of elements in NDArray
 */
static inline int64_t TVMDSPNDArrayNumElements(const TVMDSPNDArray* arr) {
  int64_t num = 1;
  int32_t i;
  for (i = 0; i < arr->ndim; i++) {
    num *= arr->shape[i];
  }
  return num;
}

/*!
 * \brief Get data size in bytes
 */
static inline size_t TVMDSPNDArrayDataSize(const TVMDSPNDArray* arr) {
  int64_t num = TVMDSPNDArrayNumElements(arr);
  /* Handle uint1 stored as uint8 */
  if (arr->dtype.code == kDLUInt && arr->dtype.bits == 1 && arr->dtype.lanes == 1) {
    return (size_t)num;
  }
  return (size_t)((num * arr->dtype.bits * arr->dtype.lanes + 7) / 8);
}

/*!
 * \brief Check if NDArray is contiguous
 */
static inline int TVMDSPNDArrayIsContiguous(const TVMDSPNDArray* arr) {
  int64_t expected_stride;
  int32_t i;

  if (arr->strides == NULL) return 1;

  expected_stride = 1;
  for (i = arr->ndim - 1; i >= 0; i--) {
    if (arr->shape[i] == 1) continue;
    if (arr->strides[i] != expected_stride) return 0;
    expected_stride *= arr->shape[i];
  }
  return 1;
}

/* ---------------------------------------------------------------------------
 * NDArray Operations
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Copy data from byte buffer to NDArray
 */
int TVMDSPNDArrayCopyFromBytes(const void* src, size_t nbytes, TVMDSPNDArray* dst);

/*!
 * \brief Copy data from NDArray to byte buffer
 */
int TVMDSPNDArrayCopyToBytes(const TVMDSPNDArray* src, void* dst, size_t nbytes);

/*!
 * \brief Copy data between NDArrays
 */
int TVMDSPNDArrayCopy(const TVMDSPNDArray* src, TVMDSPNDArray* dst);

/* ---------------------------------------------------------------------------
 * Reference Counting
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Increment reference count
 */
static inline void TVMDSPNDArrayIncRef(TVMDSPNDArray* arr) {
  if (arr) arr->ref_counter++;
}

/*!
 * \brief Decrement reference count (may free)
 */
static inline void TVMDSPNDArrayDecRef(TVMDSPNDArray* arr) {
  if (arr) {
    arr->ref_counter--;
    if (arr->ref_counter == 0 && arr->deleter) {
      arr->deleter((TVMFFIObject*)arr);
    }
  }
}

/* ---------------------------------------------------------------------------
 * Type Checking
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Check if an FFI object is an NDArray
 */
static inline int TVMDSPIsNDArray(const TVMFFIObject* obj) {
  return obj && obj->type_index == kTVMFFITensor;
}

/*!
 * \brief Cast TVMFFIAny to NDArray (with type check)
 */
static inline TVMDSPNDArray* TVMDSPAnyAsNDArray(const TVMFFIAny* any) {
  if (any && any->type_index == kTVMFFITensor) {
    return (TVMDSPNDArray*)any->v_obj;
  }
  return NULL;
}

/*!
 * \brief Set TVMFFIAny to hold an NDArray reference
 */
static inline void TVMDSPAnySetNDArray(TVMFFIAny* any, TVMDSPNDArray* arr) {
  any->type_index = kTVMFFITensor;
  any->small_len = 0;
  any->v_obj = (TVMFFIObject*)arr;
  TVMDSPNDArrayIncRef(arr);
}

/* ---------------------------------------------------------------------------
 * Internal: Default deleter for NDArray
 * ---------------------------------------------------------------------------*/

void TVMDSPNDArrayDeleter(TVMFFIObject* obj);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_DSP_RUNTIME_CONTAINER_NDARRAY_H_ */
