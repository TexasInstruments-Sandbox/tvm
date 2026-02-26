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
 * \file object.h
 * \brief TVM DSP Runtime - Object creation and management
 *
 * This header provides macros and functions for creating and managing
 * TVM objects in the DSP runtime environment.
 */
#ifndef TVM_RUNTIME_TI_DSP_FFI_OBJECT_H_
#define TVM_RUNTIME_TI_DSP_FFI_OBJECT_H_

#include "ffi_types.h"
#include "../platform/dsp_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------------------------------
 * Object Allocation
 *---------------------------------------------------------------------------*/

/*!
 * \brief Allocate an object from the DSP memory pool.
 *
 * \param size Size of the object in bytes.
 * \param type_index Type index of the object.
 * \param deleter Deleter function to call when ref count reaches zero.
 * \return Pointer to allocated object, or NULL on failure.
 *
 * The object header (TVMFFIObject) is initialized with:
 *   - type_index set to the provided value
 *   - ref_counter set to 1 (initial reference)
 *   - deleter set to the provided function
 */
TVMFFIObject* TVMDSPObjectAlloc(size_t size, int32_t type_index,
                                 void (*deleter)(TVMFFIObject*));

/*!
 * \brief Free an object back to the DSP memory pool.
 *
 * This is typically called by the object's deleter function.
 *
 * \param obj Object to free.
 */
void TVMDSPObjectFree(TVMFFIObject* obj);

/*---------------------------------------------------------------------------
 * Object Initialization Macros
 *---------------------------------------------------------------------------*/

/*!
 * \brief Initialize the header of an object.
 *
 * Use this macro in object constructors to set up the TVMFFIObject header.
 *
 * \param obj Pointer to the object.
 * \param type_idx Type index.
 * \param del Deleter function.
 */
#define TVM_DSP_OBJECT_INIT_HEADER(obj, type_idx, del) \
  do {                                                  \
    (obj)->type_index = (type_idx);                    \
    (obj)->ref_counter = 1;                            \
    (obj)->deleter = (del);                            \
  } while (0)

/*---------------------------------------------------------------------------
 * Default Deleters for Common Object Types
 *---------------------------------------------------------------------------*/

/*!
 * \brief Default deleter that just frees the object memory.
 *
 * Use this for simple objects that don't need special cleanup.
 */
void TVMDSPDefaultObjectDeleter(TVMFFIObject* obj);

/*---------------------------------------------------------------------------
 * Object Type Helpers
 *---------------------------------------------------------------------------*/

/*!
 * \brief Check if an object is of a specific type.
 *
 * \param obj Object to check.
 * \param type_index Expected type index.
 * \return Non-zero if object is of the specified type.
 */
static inline int TVMDSPObjectIsType(const TVMFFIObject* obj, int32_t type_index) {
  return obj != NULL && obj->type_index == type_index;
}

/*!
 * \brief Check if an object is an NDArray.
 */
static inline int TVMDSPObjectIsNDArray(const TVMFFIObject* obj) {
  return TVMDSPObjectIsType(obj, kTVMFFITensor);
}

/*!
 * \brief Check if an object is a Shape.
 */
static inline int TVMDSPObjectIsShape(const TVMFFIObject* obj) {
  return TVMDSPObjectIsType(obj, kTVMFFIShape);
}

/*!
 * \brief Check if an object is a Function.
 */
static inline int TVMDSPObjectIsFunction(const TVMFFIObject* obj) {
  return TVMDSPObjectIsType(obj, kTVMFFIFunction);
}

/*!
 * \brief Check if an object is a String.
 */
static inline int TVMDSPObjectIsString(const TVMFFIObject* obj) {
  return TVMDSPObjectIsType(obj, kTVMFFIStr);
}

/*---------------------------------------------------------------------------
 * Object Safe Cast
 *---------------------------------------------------------------------------*/

/*!
 * \brief Safely cast an object to a specific type.
 *
 * \param obj Object to cast.
 * \param type_index Expected type index.
 * \return Pointer to object if type matches, NULL otherwise.
 */
static inline void* TVMDSPObjectCast(TVMFFIObject* obj, int32_t type_index) {
  if (TVMDSPObjectIsType(obj, type_index)) {
    return obj;
  }
  return NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* TVM_RUNTIME_TI_DSP_FFI_OBJECT_H_ */
