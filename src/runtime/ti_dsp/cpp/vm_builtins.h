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
 * \file cpp/vm_builtins.h
 * \brief Direct VM builtin functions for C66x DSP
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * This header provides direct C++ functions that replace FFI-dispatched VM
 * builtin calls. These functions call the underlying DSP runtime directly
 * without the overhead of:
 *   - TVMBackendAnyListSetPackedArg (arg packing)
 *   - TVMFFIFunctionCall (dispatch)
 *   - Packed wrapper validation
 *   - TVMBackendAnyListMoveFromPackedReturn (result extraction)
 *
 * =============================================================================
 * PERFORMANCE
 * =============================================================================
 *
 * FFI-based call (~280 cycles):
 *   TVMBackendAnyListSetPackedArg(r, 2, stack, 0);    // 20 cycles
 *   SetFFIAnyInt(&stack[1], 0);                       // 10 cycles
 *   TVMBackendAnyListSetPackedArg(c, 5, stack, 2);    // 20 cycles
 *   TVMBackendAnyListSetPackedArg(c, 6, stack, 3);    // 20 cycles
 *   SetFFIAnyNone(&stack[4]);                         // 10 cycles
 *   TVMFFIFunctionCall(builtin, stack, 4, &result);   // 100 cycles
 *   TVMBackendAnyListMoveFromPackedReturn(...);       // 50 cycles
 *   + packed wrapper validation                       // 50 cycles
 *
 * Direct call (~150 cycles):
 *   AllocTensor(storage, 0, shape, dtype);            // 150 cycles (actual work)
 *
 * =============================================================================
 * USAGE
 * =============================================================================
 *
 * In generated code:
 *
 *   #include "tvm/dsp/vm_builtins.h"
 *   using namespace tvm::dsp::vm;
 *
 *   // Instead of FFI dispatch:
 *   r.SetStorage(2, AllocStorage(1024, dtype));
 *   r.SetNDArray(3, AllocTensor(r.GetStorage(2), 0, c.GetShape(5), c.GetDType(6)));
 *   r.SetNDArray(4, Reshape(r.GetNDArray(3), c.GetShape(7)));
 */

#ifndef TVM_RUNTIME_TI_DSP_CPP_VM_BUILTINS_H_
#define TVM_RUNTIME_TI_DSP_CPP_VM_BUILTINS_H_

#include "../ffi/ffi_types.h"
#include "../container/ndarray.h"
#include "../container/shape.h"
#include "../vm/storage.h"

/* Forward declarations for direct functions in vm_builtins.cpp */
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Direct reshape without packed wrapper validation
 * \param arr Source NDArray
 * \param shape_data New shape data
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
 * \return Tuple object, or NULL on failure
 *
 * Note: For single-output models, this is often unused.
 * Full tuple support requires dynamic allocation.
 */
TVMDSPNDArray* TVMDSPBuiltinMakeTupleDirect(TVMFFIAny* values, int32_t num_values);

/*!
 * \brief Packed make_tuple for multi-element tuples
 * \param args Array of TVMFFIAny arguments
 * \param num_args Number of arguments
 * \param ret Return value (output TVMDSPArray)
 * \return 0 on success, -1 on failure
 *
 * Creates a TVMDSPArray container holding all elements.
 * Used for multi-output models (e.g., ResNet skip connections).
 */
int TVMDSPBuiltinMakeTuplePacked(const TVMFFIAny* args, int32_t num_args, TVMFFIAny* ret);

#ifdef __cplusplus
}
#endif

namespace tvm {
namespace dsp {
namespace vm {

/*-----------------------------------------------------------------------------
 * Memory Allocation Builtins
 *-----------------------------------------------------------------------------*/

/*!
 * \brief Allocate storage buffer
 * \param size Size in bytes
 * \param dtype_hint Data type hint for alignment
 * \return New storage object (ref_count=1), or nullptr on failure
 *
 * Direct replacement for vm.builtin.alloc_storage FFI call.
 * Eliminates ~250 cycles of FFI overhead per call.
 */
inline TVMDSPStorage* AllocStorage(int64_t size, DLDataType dtype_hint) {
  DLDevice device;
  device.device_type = kDLCPU;
  device.device_id = 0;
  return TVMDSPStorageAlloc(static_cast<size_t>(size), device, dtype_hint);
}

/*!
 * \brief Allocate tensor view on storage
 * \param storage Storage buffer to use
 * \param offset Byte offset into storage
 * \param shape Shape object
 * \param dtype Data type
 * \return New NDArray view (ref_count=1), or nullptr on failure
 *
 * Direct replacement for vm.builtin.alloc_tensor FFI call.
 * Eliminates ~130 cycles of FFI overhead per call.
 */
inline TVMDSPNDArray* AllocTensor(TVMDSPStorage* storage, int64_t offset,
                                   TVMDSPShape* shape, DLDataType dtype) {
  return TVMDSPStorageAllocNDArray(storage, offset, shape->data,
                                    static_cast<int32_t>(shape->size), dtype);
}

/*!
 * \brief Allocate shape heap for dynamic shapes
 * \param size Number of int64 elements
 * \return NDArray with shape [size] and dtype int64, or nullptr on failure
 *
 * Direct replacement for vm.builtin.alloc_shape_heap FFI call.
 * The shape heap is used to store dynamically computed shape values.
 */
inline TVMDSPNDArray* AllocShapeHeap(int64_t size) {
  int64_t shape[1] = {size};
  DLDataType dtype;
  DLDevice device;

  dtype.code = kDLInt;
  dtype.bits = 64;
  dtype.lanes = 1;
  device.device_type = kDLCPU;
  device.device_id = 0;

  return TVMDSPNDArrayAlloc(shape, 1, dtype, device);
}

/*-----------------------------------------------------------------------------
 * Tensor View Builtins
 *-----------------------------------------------------------------------------*/

/*!
 * \brief Create reshaped view of tensor
 * \param tensor Source tensor
 * \param new_shape New shape object
 * \return New NDArray view with new shape (ref_count=1), or nullptr on failure
 *
 * Direct replacement for vm.builtin.reshape FFI call.
 * The returned NDArray shares data with the original tensor.
 * Eliminates ~150 cycles of FFI overhead per call.
 */
inline TVMDSPNDArray* Reshape(TVMDSPNDArray* tensor, TVMDSPShape* new_shape) {
  return TVMDSPBuiltinReshapeDirect(tensor, new_shape->data,
                                     static_cast<int32_t>(new_shape->size));
}

/*-----------------------------------------------------------------------------
 * Shape Builtins
 *-----------------------------------------------------------------------------*/

/*!
 * \brief Create shape from immediate values
 * \param shape_data Array of dimension values
 * \param ndim Number of dimensions
 * \return New shape object (ref_count=1), or nullptr on failure
 *
 * Simplified direct creation without the code/value protocol used by make_shape.
 * Use when shape values are known directly (not loaded from heap).
 */
inline TVMDSPShape* CreateShape(const int64_t* shape_data, int32_t ndim) {
  return TVMDSPShapeCreate(shape_data, ndim);
}

/*!
 * \brief Create shape from code/value protocol (heap-based lookups)
 * \param heap Shape heap NDArray containing runtime dimension values
 * \param ndim Number of dimensions
 * \param codes Array of codes: 0=UseImm, 1=LoadFromHeap
 * \param values Array of values: immediate value or heap index
 * \return New shape object (ref_count=1), or nullptr on failure
 *
 * Direct replacement for vm.builtin.make_shape FFI call.
 * Supports both immediate values and heap-loaded dynamic dimensions.
 */
inline TVMDSPShape* MakeShape(TVMDSPNDArray* heap, int32_t ndim,
                               const int32_t* codes, const int64_t* values) {
  return TVMDSPBuiltinMakeShape(heap, ndim, codes, values);
}

/*-----------------------------------------------------------------------------
 * Tuple Builtins
 *-----------------------------------------------------------------------------*/

/*!
 * \brief Create tuple from values (single-element optimization)
 * \param values Array of TVMFFIAny values
 * \param num_values Number of values
 * \return Tuple-like object (simplified as first NDArray for single-output)
 *
 * Direct replacement for vm.builtin.make_tuple FFI call.
 * Note: Full tuple support requires dynamic allocation. For most DSP models
 * with single output, this returns the first value.
 */
inline TVMDSPNDArray* MakeTuple(TVMFFIAny* values, int32_t num_values) {
  return TVMDSPBuiltinMakeTupleDirect(values, num_values);
}

/*!
 * \brief Create multi-element tuple array
 * \param values Array of TVMFFIAny values
 * \param num_values Number of values
 * \return TVMDSPArray* containing all elements, or nullptr on failure
 *
 * Direct replacement for vm.builtin.make_tuple FFI call with multiple elements.
 * Allocates a TVMDSPArray container and copies all elements with proper
 * reference counting. Used for multi-output models (e.g., ResNet skip connections).
 */
inline TVMDSPArray* MakeTupleArray(TVMFFIAny* values, int32_t num_values) {
  TVMFFIAny result;
  if (TVMDSPBuiltinMakeTuplePacked(values, num_values, &result) != 0) {
    return nullptr;
  }
  return reinterpret_cast<TVMDSPArray*>(result.v_obj);
}

/*-----------------------------------------------------------------------------
 * Reference Management Builtins
 *-----------------------------------------------------------------------------*/

/*!
 * \brief Copy reference (identity function with IncRef)
 * \param arr NDArray to copy reference of
 * \return Same NDArray with incremented ref_count
 *
 * Direct replacement for vm.builtin.copy FFI call.
 * Simply increments the reference count and returns the same pointer.
 */
inline TVMDSPNDArray* Copy(TVMDSPNDArray* arr) {
  if (arr != nullptr) {
    arr->ref_counter++;
  }
  return arr;
}

/*-----------------------------------------------------------------------------
 * Validation Builtins (typically skipped with skip_runtime_checks)
 *-----------------------------------------------------------------------------*/

/*!
 * \brief Check tensor info matches expected
 * \param tensor Tensor to check
 * \param ndim Expected number of dimensions (-1 to skip)
 * \param dtype Expected data type (code=kDLOpaqueHandle to skip)
 * \return 0 on success, -1 on mismatch
 *
 * Direct replacement for vm.builtin.check_tensor_info FFI call.
 * Note: This is typically skipped entirely when skip_runtime_checks is enabled
 * since TVM's compile-time type inference already validates shapes.
 */
inline int CheckTensorInfo(TVMDSPNDArray* tensor, int32_t ndim, DLDataType dtype) {
  if (tensor == nullptr) {
    return -1;
  }
  if (ndim != -1 && tensor->ndim != ndim) {
    return -1;
  }
  if (dtype.code != kDLOpaqueHandle) {
    if (tensor->dtype.code != dtype.code ||
        tensor->dtype.bits != dtype.bits ||
        tensor->dtype.lanes != dtype.lanes) {
      return -1;
    }
  }
  return 0;
}

/*!
 * \brief Get shape of tensor
 * \param tensor Source tensor
 * \return New shape object with tensor's shape, or nullptr on failure
 *
 * Direct replacement for vm.builtin.shape_of FFI call.
 */
inline TVMDSPShape* ShapeOf(TVMDSPNDArray* tensor) {
  if (tensor == nullptr) {
    return nullptr;
  }
  return TVMDSPShapeCreate(tensor->shape, tensor->ndim);
}

}  // namespace vm
}  // namespace dsp
}  // namespace tvm

#endif  // TVM_RUNTIME_TI_DSP_CPP_VM_BUILTINS_H_
