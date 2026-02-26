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
 * \file registry/registry.h
 * \brief Function registry and FFI call interface for TVM DSP Runtime
 *
 * This provides the core FFI functions that TVM-generated code calls:
 *   - TVMFFIFunctionCall: Call a packed function
 *   - TVMBackendAnyListSetPackedArg: Set argument in list
 *   - TVMBackendAnyListMoveFromPackedReturn: Move return value from list
 *   - TVMBackendAnyListResetItem: Reset list item
 */

#ifndef TVM_DSP_RUNTIME_REGISTRY_H_
#define TVM_DSP_RUNTIME_REGISTRY_H_

#include "../ffi/ffi_types.h"
#include "packed_func.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * FFI Function Call Interface
 *
 * These functions implement TVM's FFI calling convention, used by generated
 * code to call packed functions retrieved from the global registry.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Call a packed function
 *
 * This is the main entry point for calling packed functions from generated
 * code. The function handle is retrieved via TVMBackendGetFuncFromGlobalRegistry.
 *
 * \param func Function handle (TVMDSPPackedFunc*)
 * \param args Array of input arguments
 * \param num_args Number of input arguments
 * \param result Output result (type_index must be set to kTVMFFINone before call)
 * \return 0 on success, non-zero on error
 */
int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                       TVMFFIAny* result);

/*!
 * \brief Set a packed argument from another TVMFFIAny array
 *
 * Copies an argument from the source args array to the target anylist.
 * Used by generated code to build up argument lists for packed function calls.
 *
 * \param anylist Target argument list (TVMFFIAny*)
 * \param index Index in target list to set
 * \param args Source argument array
 * \param arg_offset Offset into source array
 * \return 0 on success, non-zero on error
 */
int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMFFIAny* args,
                                  int arg_offset);

/*!
 * \brief Move return value from packed call to register
 *
 * Moves the return value from the result position in args array to the
 * target anylist. Handles reference counting correctly.
 *
 * \param anylist Target list to receive result (TVMFFIAny*)
 * \param index Index in target list
 * \param args Source argument array (result is at ret_offset)
 * \param ret_offset Offset of result in source array
 * \return 0 on success, non-zero on error
 */
int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index,
                                           TVMFFIAny* args, int ret_offset);

/*!
 * \brief Reset an item in anylist to None
 *
 * Decrements reference count if item holds an object, then sets to None.
 *
 * \param anylist Target list (TVMFFIAny*)
 * \param index Index to reset
 * \return 0 on success
 */
int TVMBackendAnyListResetItem(void* anylist, int index);

/* ---------------------------------------------------------------------------
 * Registry Management
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Initialize the function registry
 *
 * Sets up the global function registry and registers built-in functions.
 * Call this once during platform initialization.
 */
void TVMRegistryInit(void);

/*!
 * \brief Register a packed function with a name
 *
 * Adds a function to the global registry. The function can then be
 * retrieved via TVMBackendGetFuncFromGlobalRegistry.
 *
 * \param name Function name (e.g., "vm.builtin.alloc_storage")
 * \param func Packed function pointer
 * \return 0 on success, -1 if registry is full
 */
int TVMRegistryRegister(const char* name, TVMDSPPackedFuncRaw func);

/*!
 * \brief Lookup function by name
 *
 * Finds a registered function by its name.
 *
 * \param name Function name
 * \return Function handle, or NULL if not found
 */
TVMFFIObjectHandle TVMRegistryLookup(const char* name);

/* ---------------------------------------------------------------------------
 * PackedFunc Pool Management
 *
 * For DSP, we use a static pool of PackedFunc objects since we don't have
 * dynamic allocation for function objects.
 * ---------------------------------------------------------------------------*/

#ifndef TVM_DSP_MAX_PACKED_FUNCS
#define TVM_DSP_MAX_PACKED_FUNCS 64
#endif

/*!
 * \brief Allocate a PackedFunc from the static pool
 *
 * \param func Raw function pointer
 * \param name Function name (for debugging)
 * \return PackedFunc object, or NULL if pool exhausted
 */
TVMDSPPackedFunc* TVMRegistryAllocPackedFunc(TVMDSPPackedFuncRaw func,
                                              const char* name);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_DSP_RUNTIME_REGISTRY_H_ */
