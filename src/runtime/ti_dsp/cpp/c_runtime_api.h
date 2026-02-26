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
 * \file cpp/c_runtime_api.h
 * \brief C Runtime API compatibility header for TVM DSP Runtime
 *
 * This header provides compatibility with TVM's c_runtime_api.h.
 * Generated code includes <tvm/runtime/c_runtime_api.h> which
 * this header provides for DSP targets.
 */

#ifndef TVM_DSP_RUNTIME_CPP_C_RUNTIME_API_H_
#define TVM_DSP_RUNTIME_CPP_C_RUNTIME_API_H_

/* Include the C FFI types (which includes dlpack) */
#ifdef __cplusplus
extern "C" {
#endif
#include "../ffi/ffi_types.h"
#ifdef __cplusplus
}
#endif

/* For C++ code, also include C++ wrappers */
#ifdef __cplusplus
#include "object_ref.h"
#include "ndarray.h"
#include "any.h"
#include "packed_func.h"
#endif

/*
 * C API function declarations
 *
 * These match the signatures expected by TVM-generated code.
 */
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Get function from global registry
 * \param name Function name
 * \param out Output handle
 * \return 0 on success, non-zero on error
 */
int TVMBackendGetFuncFromGlobalRegistry(const char* name, void** out);

/*!
 * \brief Allocate workspace memory
 * \param device_type Device type
 * \param device_id Device ID
 * \param nbytes Number of bytes
 * \param dtype_code_hint Type code hint
 * \param dtype_bits_hint Bit width hint
 * \return Allocated pointer, or NULL on failure
 */
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                               int dtype_code_hint, int dtype_bits_hint);

/*!
 * \brief Free workspace memory
 * \param ptr Pointer to free
 * \return 0 on success
 */
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

/*!
 * \brief Report error for generated code
 * \param err Error return value
 * \return 0 on success
 */
int TVMBackendReportError(int err);

/*!
 * \brief Call a packed function
 * \param func Function handle
 * \param args Argument array
 * \param num_args Number of arguments
 * \param result Result output
 * \return 0 on success, non-zero on error
 */
int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args,
                       int32_t num_args, TVMFFIAny* result);

/*!
 * \brief Set packed argument in any list
 * \param anylist The any list
 * \param index Index in anylist
 * \param args Packed arguments array
 * \param arg_offset Offset in args array
 * \return 0 on success
 */
int TVMBackendAnyListSetPackedArg(void* anylist, int index,
                                  TVMFFIAny* args, int arg_offset);

/*!
 * \brief Move return value from packed args to any list
 * \param anylist The any list
 * \param index Index in anylist
 * \param args Packed arguments array
 * \param ret_offset Offset in args array for return value
 * \return 0 on success
 */
int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index,
                                          TVMFFIAny* args, int ret_offset);

/*!
 * \brief Reset item in any list
 * \param anylist The any list
 * \param index Index to reset
 * \return 0 on success
 */
int TVMBackendAnyListResetItem(void* anylist, int index);

#ifdef __cplusplus
}  /* extern "C" */
#endif

/*
 * TVM FFI Safe Call Macro
 *
 * This macro is used by generated code to wrap function calls
 * with error handling.
 */
#ifndef TVM_FFI_SAFE_CALL
#define TVM_FFI_SAFE_CALL(func) \
  do { \
    int __ret = (func); \
    if (__ret != 0) { \
      return __ret; \
    } \
  } while (0)
#endif

/*
 * Additional type aliases for compatibility
 * Note: DataType is defined as a class in cpp/ndarray.h
 */
#ifdef __cplusplus
namespace tvm {
namespace runtime {

/* Type alias for DLDevice */
using Device = DLDevice;

}  // namespace runtime
}  // namespace tvm
#endif

#endif  /* TVM_DSP_RUNTIME_CPP_C_RUNTIME_API_H_ */
