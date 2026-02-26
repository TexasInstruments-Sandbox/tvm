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
 * \file registry/packed_func.h
 * \brief Packed function type and object definitions for TVM DSP Runtime
 *
 * A PackedFunc is a callable object that accepts TVMFFIAny arguments and
 * returns a TVMFFIAny result. This is TVM's universal calling convention
 * for interop between generated code and runtime functions.
 */

#ifndef TVM_DSP_RUNTIME_PACKED_FUNC_H_
#define TVM_DSP_RUNTIME_PACKED_FUNC_H_

#include "../ffi/ffi_types.h"
#include "../ffi/object.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Type index for PackedFunc - must match TVM's ffi type system */
#define TVM_DSP_PACKED_FUNC_TYPE_INDEX 73

/*!
 * \brief Raw packed function signature
 *
 * This is the function type that all packed functions implement.
 *
 * \param args Array of input arguments
 * \param num_args Number of input arguments
 * \param ret Output return value
 * \return 0 on success, non-zero on error
 */
typedef int (*TVMDSPPackedFuncRaw)(const TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* ret);

/*!
 * \brief PackedFunc object structure
 *
 * Wraps a raw function pointer in a TVM object so it can be passed around
 * and managed by the runtime. This is stored in the global registry.
 */
typedef struct TVMDSPPackedFunc {
  /* Object header - must match TVMFFIObject layout */
  int32_t type_index;
  int32_t ref_counter;
  union {
    void (*deleter)(struct TVMDSPPackedFunc* self);
    int64_t __ensure_align;
  };

  /* Function pointer */
  TVMDSPPackedFuncRaw func;

  /* Optional name for debugging */
  const char* name;
} TVMDSPPackedFunc;

/*!
 * \brief Create a new PackedFunc object
 *
 * Creates a statically allocated PackedFunc object. The object is not
 * reference counted since it's expected to live for the program lifetime.
 *
 * \param func The raw function pointer
 * \param name Function name (for debugging)
 * \return Pointer to PackedFunc object
 */
TVMDSPPackedFunc* TVMDSPPackedFuncCreate(TVMDSPPackedFuncRaw func,
                                         const char* name);

/*!
 * \brief Call a PackedFunc object
 *
 * Invokes the underlying function with the given arguments.
 *
 * \param pfunc The PackedFunc object
 * \param args Input arguments
 * \param num_args Number of arguments
 * \param ret Output return value
 * \return 0 on success, non-zero on error
 */
static inline int TVMDSPPackedFuncCall(TVMDSPPackedFunc* pfunc,
                                       const TVMFFIAny* args,
                                       int32_t num_args,
                                       TVMFFIAny* ret) {
  if (pfunc == NULL || pfunc->func == NULL) {
    return -1;
  }
  return pfunc->func(args, num_args, ret);
}

/*!
 * \brief Check if an object is a PackedFunc
 *
 * \param obj Object to check
 * \return 1 if obj is a PackedFunc, 0 otherwise
 */
static inline int TVMDSPIsPackedFunc(TVMFFIObjectHandle obj) {
  if (obj == NULL) {
    return 0;
  }
  TVMFFIObject* o = (TVMFFIObject*)obj;
  return o->type_index == TVM_DSP_PACKED_FUNC_TYPE_INDEX;
}

#ifdef __cplusplus
}
#endif

#endif  /* TVM_DSP_RUNTIME_PACKED_FUNC_H_ */
