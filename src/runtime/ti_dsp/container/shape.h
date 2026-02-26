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
 * \file container/shape.h
 * \brief Shape container for TVM DSP Runtime
 *
 * This provides an ABI-compatible Shape implementation for bare-metal DSP.
 * The memory layout matches TVM's ffi::ShapeObj.
 *
 * TVM's ShapeObj layout:
 *   class ShapeObj : public Object, public TVMFFIShapeCell
 *
 * where TVMFFIShapeCell is:
 *   typedef struct { const int64_t* data; size_t size; } TVMFFIShapeCell;
 *
 * So the layout is:
 *   - TVMFFIObject header (16 bytes)
 *   - TVMFFIShapeCell (data pointer + size)
 *   - Inline data storage follows the object
 */

#ifndef TVM_DSP_RUNTIME_CONTAINER_SHAPE_H_
#define TVM_DSP_RUNTIME_CONTAINER_SHAPE_H_

#include <stddef.h>
#include "../ffi/ffi_types.h"
#include "../core/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum dimensions for static allocation (from core/config.h) */
#define TVM_DSP_SHAPE_MAX_NDIM TVM_DSP_MAX_NDIM

/*!
 * \brief Shape object structure matching TVM's ShapeObj layout
 *
 * Layout:
 *   Offset 0:   TVMFFIObject header (16 bytes)
 *   Offset 16:  TVMFFIShapeCell (data pointer + size)
 *   After that: Inline data storage
 */
typedef struct TVMDSPShape {
  /* === TVMFFIObject Header (16 bytes) === */
  int32_t type_index;    /* Must be kTVMFFIShape (69) */
  int32_t ref_counter;   /* Reference count */
  union {
    void (*deleter)(TVMFFIObject* self);
    int64_t _align;      /* Ensure 8-byte alignment */
  };

  /* === TVMFFIShapeCell === */
  const int64_t* data;   /* Pointer to shape data */
  size_t size;           /* Number of dimensions */

  /* === Inline Data Storage === */
  /* TVM stores data inline after the object, we use fixed storage */
  int64_t shape_data[TVM_DSP_SHAPE_MAX_NDIM];
} TVMDSPShape;

/* ---------------------------------------------------------------------------
 * Shape Creation Functions
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Create a Shape from an array of dimension sizes
 * \param dims Array of dimension sizes
 * \param ndim Number of dimensions
 * \return Allocated Shape object, or NULL on failure
 */
TVMDSPShape* TVMDSPShapeCreate(const int64_t* dims, size_t ndim);

/*!
 * \brief Create an empty shape (0 dimensions, scalar)
 * \return Allocated Shape object, or NULL on failure
 */
TVMDSPShape* TVMDSPShapeCreateEmpty(void);

/* ---------------------------------------------------------------------------
 * Shape Accessors
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Get the shape cell pointer (TVM API compatible)
 *
 * This matches TVMFFIShapeGetCellPtr from tvm/ffi/c_api.h
 */
static inline TVMFFIShapeCell* TVMDSPShapeGetCell(TVMDSPShape* shape) {
  return (TVMFFIShapeCell*)(&shape->data);
}

/*!
 * \brief Get the number of dimensions
 */
static inline size_t TVMDSPShapeSize(const TVMDSPShape* shape) {
  return shape->size;
}

/*!
 * \brief Get the i-th dimension size
 */
static inline int64_t TVMDSPShapeAt(const TVMDSPShape* shape, size_t idx) {
  return shape->data[idx];
}

/*!
 * \brief Compute the product of all dimensions (total elements)
 */
static inline int64_t TVMDSPShapeProduct(const TVMDSPShape* shape) {
  int64_t product = 1;
  size_t i;
  for (i = 0; i < shape->size; i++) {
    product *= shape->data[i];
  }
  return product;
}

/* ---------------------------------------------------------------------------
 * Reference Counting
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Increment reference count
 */
static inline void TVMDSPShapeIncRef(TVMDSPShape* shape) {
  if (shape) shape->ref_counter++;
}

/*!
 * \brief Decrement reference count (may free)
 */
static inline void TVMDSPShapeDecRef(TVMDSPShape* shape) {
  if (shape) {
    shape->ref_counter--;
    if (shape->ref_counter == 0 && shape->deleter) {
      shape->deleter((TVMFFIObject*)shape);
    }
  }
}

/* ---------------------------------------------------------------------------
 * Type Checking
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Check if an FFI object is a Shape
 */
static inline int TVMDSPIsShape(const TVMFFIObject* obj) {
  return obj && obj->type_index == kTVMFFIShape;
}

/*!
 * \brief Cast TVMFFIAny to Shape (with type check)
 */
static inline TVMDSPShape* TVMDSPAnyAsShape(const TVMFFIAny* any) {
  if (any && any->type_index == kTVMFFIShape) {
    return (TVMDSPShape*)any->v_obj;
  }
  return NULL;
}

/* ---------------------------------------------------------------------------
 * Internal: Default deleter
 * ---------------------------------------------------------------------------*/

void TVMDSPShapeDeleter(TVMFFIObject* obj);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_DSP_RUNTIME_CONTAINER_SHAPE_H_ */
