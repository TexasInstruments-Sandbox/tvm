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
 * \file vm/vm_builtins.h
 * \brief VM builtin functions for TVM DSP Runtime
 *
 * This implements the vm.builtin.* functions that TVM-generated code calls.
 * These are the core functions needed to run Relax VM bytecode on DSP.
 *
 * Key builtins:
 *   - vm.builtin.alloc_storage: Allocate raw memory buffer
 *   - vm.builtin.alloc_tensor: Create NDArray from storage at offset
 *   - vm.builtin.make_shape: Construct shape from heap/immediate values
 *   - vm.builtin.match_shape: Validate shape dimensions
 *   - vm.builtin.alloc_shape_heap: Allocate heap for symbolic shapes
 *   - vm.builtin.check_tensor_info: Validate tensor dtype/ndim
 *   - vm.builtin.null_value: Return null for killing objects
 */

#ifndef TVM_DSP_RUNTIME_VM_BUILTINS_H_
#define TVM_DSP_RUNTIME_VM_BUILTINS_H_

#include <dlpack/dlpack.h>
#include "../ffi/ffi_types.h"
#include "../container/ndarray.h"
#include "../container/shape.h"
#include "storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Shape Code Enums - Must match TVM's tvm/runtime/vm/builtin.h
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Op code for vm.builtin.match_shape
 */
typedef enum {
  kMatchShapeAssertEqualToImm = 0,  /* assert input_shape[i] == reg */
  kMatchShapeStoreToHeap = 1,       /* shape_heap[reg] = input_shape[i] */
  kMatchShapeNoOp = 2,              /* skip */
  kMatchShapeAssertEqualToLoad = 3, /* assert input_shape[i] == shape_heap[reg] */
} TVMDSPMatchShapeCode;

/*!
 * \brief Op code for vm.builtin.make_shape
 */
typedef enum {
  kMakeShapeUseImm = 0,             /* Use reg as immediate value */
  kMakeShapeLoadShape = 1,          /* Load from shape_heap[reg] */
} TVMDSPMakeShapeCode;

/* ---------------------------------------------------------------------------
 * VM Builtin Functions
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Allocate storage (vm.builtin.alloc_storage)
 *
 * Allocates a raw memory buffer that can be used by alloc_tensor.
 *
 * \param size Size in bytes to allocate
 * \param device_index Device index (ignored on single-device DSP)
 * \param dtype_hint Data type hint (used for alignment)
 * \return Allocated storage object, or NULL on failure
 */
TVMDSPStorage* TVMDSPBuiltinAllocStorage(int64_t size, int32_t device_index,
                                          DLDataType dtype_hint);

/*!
 * \brief Allocate tensor from storage (vm.builtin.alloc_tensor)
 *
 * Creates an NDArray that uses memory from the given storage at offset.
 *
 * \param storage The storage to allocate from
 * \param offset Byte offset into storage
 * \param shape Shape of the tensor
 * \param ndim Number of dimensions
 * \param dtype Data type of the tensor
 * \return Allocated NDArray, or NULL on failure
 */
TVMDSPNDArray* TVMDSPBuiltinAllocTensor(TVMDSPStorage* storage, int64_t offset,
                                         const int64_t* shape, int32_t ndim,
                                         DLDataType dtype);

/*!
 * \brief Allocate shape heap (vm.builtin.alloc_shape_heap)
 *
 * Allocates an NDArray to store symbolic shape values during execution.
 *
 * \param size Number of shape slots to allocate
 * \return Allocated NDArray of int64_t values, or NULL on failure
 */
TVMDSPNDArray* TVMDSPBuiltinAllocShapeHeap(int64_t size);

/*!
 * \brief Make shape from codes and values (vm.builtin.make_shape)
 *
 * Constructs a shape object from immediate values or heap lookups.
 *
 * \param heap Shape heap (NDArray of int64_t), can be NULL if all UseImm
 * \param ndim Number of dimensions
 * \param codes Array of MakeShapeCode values
 * \param values Array of values (immediate or heap indices)
 * \return Allocated shape, or NULL on failure
 */
TVMDSPShape* TVMDSPBuiltinMakeShape(TVMDSPNDArray* heap, int32_t ndim,
                                     const int32_t* codes, const int64_t* values);

/*!
 * \brief Match shape against expected pattern (vm.builtin.match_shape)
 *
 * Validates shape dimensions and optionally stores to heap.
 *
 * \param input Input tensor or shape
 * \param heap Shape heap for storing/loading values
 * \param ndim Number of dimensions to match
 * \param codes Array of MatchShapeCode values
 * \param values Array of values (immediate or heap indices)
 * \return 0 on success, -1 on mismatch
 */
int TVMDSPBuiltinMatchShape(const TVMFFIAny* input, TVMDSPNDArray* heap,
                            int32_t ndim, const int32_t* codes,
                            const int64_t* values);

/*!
 * \brief Check tensor info (vm.builtin.check_tensor_info)
 *
 * Validates that a tensor has expected dtype and ndim.
 *
 * \param tensor The tensor to check
 * \param ndim Expected ndim (-1 for any)
 * \param dtype Expected dtype (void for any)
 * \return 0 on success, -1 on type error
 */
int TVMDSPBuiltinCheckTensorInfo(const TVMDSPNDArray* tensor, int32_t ndim,
                                  DLDataType dtype);

/*!
 * \brief Check shape info (vm.builtin.check_shape_info)
 *
 * Validates that a shape has expected size.
 *
 * \param shape The shape to check
 * \param ndim Expected size (-1 for any)
 * \return 0 on success, -1 on type error
 */
int TVMDSPBuiltinCheckShapeInfo(const TVMDSPShape* shape, int32_t ndim);

/*!
 * \brief Get null value (vm.builtin.null_value)
 *
 * Returns a null object reference for killing registers.
 *
 * \param out Output TVMFFIAny to set to null
 */
void TVMDSPBuiltinNullValue(TVMFFIAny* out);

/* ---------------------------------------------------------------------------
 * Registration Functions
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Register all VM builtins in the function registry
 *
 * Call this during initialization to make builtins available via
 * TVMBackendGetFuncFromGlobalRegistry.
 *
 * \return 0 on success, -1 on failure
 */
int TVMDSPRegisterVMBuiltins(void);

/* ---------------------------------------------------------------------------
 * Register File Management with Automatic Cleanup
 *
 * These functions provide proper reference counting when register values
 * are overwritten, ensuring intermediate objects are freed immediately.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Initialize register file for use with automatic cleanup
 *
 * Must be called before using TVMDSPRegSetAny.
 *
 * \param reg_file Pointer to the register file array
 * \param size Number of registers
 */
void TVMDSPRegFileInit(TVMFFIAny* reg_file, int32_t size);

/*!
 * \brief Set a register value with automatic cleanup of previous value
 *
 * If the register previously held an object, its ref count is decremented,
 * potentially freeing it. The new value is stored in the register.
 *
 * \param reg_idx Register index
 * \param value New value to store
 */
void TVMDSPRegSetAny(int32_t reg_idx, const TVMFFIAny* value);

/*!
 * \brief Cleanup all objects in the register file
 *
 * Call after inference to free all remaining objects in registers.
 *
 * \return Number of objects freed
 */
int TVMDSPRegFileCleanup(void);

/* ---------------------------------------------------------------------------
 * Packed Function Wrappers
 *
 * These match the calling convention of TVM's packed functions,
 * taking TVMFFIAny* arrays for arguments and return value.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Packed wrapper for alloc_storage
 * Args: [size (int64), device_idx (int32), dtype (DLDataType)]
 * Returns: Storage object
 */
int TVMDSPBuiltinAllocStoragePacked(const TVMFFIAny* args, int32_t num_args,
                                     TVMFFIAny* ret);

/*!
 * \brief Packed wrapper for alloc_tensor
 * Args: [storage, offset (int64), shape (Shape), dtype (DLDataType)]
 * Returns: NDArray
 */
int TVMDSPBuiltinAllocTensorPacked(const TVMFFIAny* args, int32_t num_args,
                                    TVMFFIAny* ret);

/*!
 * \brief Packed wrapper for alloc_shape_heap
 * Args: [ctx_ptr (ignored), size (int64)]
 * Returns: NDArray (int64 heap)
 */
int TVMDSPBuiltinAllocShapeHeapPacked(const TVMFFIAny* args, int32_t num_args,
                                       TVMFFIAny* ret);

/*!
 * \brief Packed wrapper for make_shape
 * Args: [heap, ndim, code0, val0, code1, val1, ...]
 * Returns: Shape
 */
int TVMDSPBuiltinMakeShapePacked(const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* ret);

/*!
 * \brief Packed wrapper for match_shape
 * Args: [input, heap, ndim, code0, val0, ..., err_ctx]
 * Returns: void (asserts on mismatch)
 */
int TVMDSPBuiltinMatchShapePacked(const TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* ret);

/*!
 * \brief Packed wrapper for check_tensor_info
 * Args: [tensor, ndim, dtype (optional), err_ctx]
 * Returns: void (asserts on mismatch)
 */
int TVMDSPBuiltinCheckTensorInfoPacked(const TVMFFIAny* args, int32_t num_args,
                                        TVMFFIAny* ret);

/*!
 * \brief Packed wrapper for null_value
 * Args: []
 * Returns: null
 */
int TVMDSPBuiltinNullValuePacked(const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* ret);

/* ---------------------------------------------------------------------------
 * Direct Functions (no packed wrapper overhead)
 *
 * These functions bypass the packed argument marshalling for use in
 * generated code with direct C++ calls.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Direct reshape without packed wrapper validation
 * \param arr Source NDArray
 * \param shape_data New shape data array
 * \param shape_size New shape size (ndim)
 * \return New NDArray view with new shape, or NULL on failure
 */
TVMDSPNDArray* TVMDSPBuiltinReshapeDirect(TVMDSPNDArray* arr,
                                           const int64_t* shape_data,
                                           int32_t shape_size);

/*!
 * \brief Direct make_tuple without packed wrapper
 * \param values Array of TVMFFIAny values
 * \param num_values Number of values
 * \return First NDArray from values, or NULL
 */
TVMDSPNDArray* TVMDSPBuiltinMakeTupleDirect(TVMFFIAny* values, int32_t num_values);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_DSP_RUNTIME_VM_BUILTINS_H_ */
