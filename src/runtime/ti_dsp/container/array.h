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
 * \file container/array.h
 * \brief Array container for TVM DSP Runtime (multi-output tuple support)
 *
 * This provides a simple array container for holding multiple TVMFFIAny values,
 * primarily used for multi-output models that return tuples of NDArrays.
 *
 * Design constraints:
 * - Static inline storage (no heap allocation for elements)
 * - Maximum 8 elements (sufficient for most multi-output models)
 * - Reference counting for contained objects
 * - ABI-compatible with TVM's FFI object conventions
 */

#ifndef TVM_DSP_RUNTIME_CONTAINER_ARRAY_H_
#define TVM_DSP_RUNTIME_CONTAINER_ARRAY_H_

#include "../ffi/ffi_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Maximum number of elements in a DSP array container.
 *
 * This limit is enforced at compile time by c_static backend.
 * Models with more outputs will fail compilation with a clear error.
 */
/* Raised from 8 to 128 to support KV cache models that return
 * (logits + 60 KV scatter outputs) = 61 total tuple elements. */
#define TVM_DSP_ARRAY_MAX_ELEMENTS 128

/*!
 * \brief Array object structure for holding multiple TVMFFIAny values.
 *
 * Layout:
 *   Offset 0:   TVMFFIObject header (16 bytes)
 *   Offset 16:  size (4 bytes) + padding (4 bytes)
 *   Offset 24:  elements array (8 elements * 16 bytes = 128 bytes)
 *
 * Total size: 152 bytes (fits in 3 cache lines on C66x)
 *
 * Usage:
 *   TVMDSPArray is created by vm.builtin.make_tuple and holds the
 *   output NDArrays from a multi-output model. Each element is a
 *   TVMFFIAny that typically contains an NDArray pointer.
 */
typedef struct TVMDSPArray {
  /* === TVMFFIObject Header (16 bytes) === */
  int32_t type_index;    /*!< Must be kTVMFFIArray (71) */
  int32_t ref_counter;   /*!< Reference count */
  union {
    void (*deleter)(TVMFFIObject* self);  /*!< Cleanup function */
    int64_t _align;      /*!< Ensure 8-byte alignment */
  };

  /* === Array Data === */
  int32_t size;          /*!< Number of elements in array */
  int32_t _padding;      /*!< Padding for 8-byte alignment */

  /*!
   * \brief Inline storage for array elements.
   *
   * Each element is a TVMFFIAny (16 bytes) that can hold:
   * - NDArray pointer (type_index = kTVMFFITensor)
   * - Scalar values
   * - Other object pointers
   *
   * Reference counting: When the array is deleted, the deleter
   * must decrement ref_counter on all contained objects.
   */
  TVMFFIAny elements[TVM_DSP_ARRAY_MAX_ELEMENTS];
} TVMDSPArray;

/*!
 * \brief Deleter function for TVMDSPArray.
 *
 * This function decrements reference counts on all contained objects
 * and frees the array memory. Called when ref_counter reaches 0.
 *
 * \param self Pointer to the array object to delete.
 */
void TVMDSPArrayDeleter(TVMFFIObject* self);

/*!
 * \brief Get element from array by index.
 *
 * \param arr Pointer to array object.
 * \param index Element index (0-based).
 * \return Pointer to element, or NULL if index out of bounds.
 */
static inline TVMFFIAny* TVMDSPArrayGetItem(TVMDSPArray* arr, int32_t index) {
  if (arr == NULL || index < 0 || index >= arr->size) {
    return NULL;
  }
  return &arr->elements[index];
}

/*!
 * \brief Get array size (number of elements).
 *
 * \param arr Pointer to array object.
 * \return Number of elements, or 0 if arr is NULL.
 */
static inline int32_t TVMDSPArraySize(TVMDSPArray* arr) {
  return (arr != NULL) ? arr->size : 0;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TVM_DSP_RUNTIME_CONTAINER_ARRAY_H_ */
