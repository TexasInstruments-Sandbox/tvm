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
 * \file vm/vm_builtins.cpp
 * \brief VM builtin function implementations using C++14 utilities
 *
 * This is a C++ implementation of the VM builtins using:
 * - ScopeGuard for automatic cleanup on error paths
 * - TypedHandle for type-safe handle access
 * - Span for safer array iteration
 *
 * The C API (extern "C") is preserved for compatibility.
 */

#include "vm_builtins.h"

extern "C" {
#include "../platform/dsp_platform.h"
#include "../platform/dsp_memory.h"
#include "../container/array.h"
}

#include "../registry/registry.h"
#include "../cpp/scope_guard.h"
#include "../cpp/typed_handle.h"
#include "../cpp/span.h"

#include <cstring>

namespace {

/*
 * =============================================================================
 * INTERNAL STATE
 * =============================================================================
 */

/* Register file pointer and size */
TVMFFIAny* g_reg_file = nullptr;
int32_t g_reg_file_size = 0;

/*
 * =============================================================================
 * INTERNAL HELPER FUNCTIONS
 * =============================================================================
 */

/*!
 * \brief Deleter for reshape view NDArrays
 *
 * Reshape creates a view that shares data with the source array.
 * This deleter just frees the NDArray struct itself - it does NOT
 * free the data buffer (owned by the source array).
 */
void ReshapeViewDeleter(TVMFFIObject* obj) {
  /* Just free the NDArray struct - data is shared with source */
  tvm_dsp_free(obj);
}

}  // anonymous namespace

/*
 * =============================================================================
 * C API IMPLEMENTATION
 * =============================================================================
 */

extern "C" {

/*---------------------------------------------------------------------------
 * Register File Management with Automatic Cleanup
 *---------------------------------------------------------------------------*/

void TVMDSPRegFileInit(TVMFFIAny* reg_file, int32_t size) {
  g_reg_file = reg_file;
  g_reg_file_size = size;
}

void TVMDSPRegSetAny(int32_t reg_idx, const TVMFFIAny* value) {
  if (g_reg_file == nullptr || reg_idx < 0 || reg_idx >= g_reg_file_size) {
    return;
  }

  /* Decrement ref count of previous value (may free it) */
  TVMFFIAnyDecRef(&g_reg_file[reg_idx]);

  /* Store new value */
  g_reg_file[reg_idx] = *value;

  /* Increment ref count of new value if it's an object */
  if (value->type_index >= kTVMFFIStaticObjectBegin && value->v_obj != nullptr) {
    static_cast<TVMFFIObject*>(value->v_obj)->ref_counter++;
  }
}

int TVMDSPRegFileCleanup(void) {
  int freed = 0;

  if (g_reg_file == nullptr) return 0;

  for (int32_t i = 0; i < g_reg_file_size; i++) {
    if (g_reg_file[i].type_index >= kTVMFFIStaticObjectBegin &&
        g_reg_file[i].v_obj != nullptr) {
      auto obj = tvm::dsp::TypedHandle<TVMFFIObject>::FromRaw(g_reg_file[i].v_obj);
      /* Force-free by decrementing ref_counter until zero */
      while (obj->ref_counter > 0) {
        obj->ref_counter--;
        if (obj->ref_counter == 0 && obj->deleter != nullptr) {
          obj->deleter(obj.get());
          freed++;
        }
      }
      g_reg_file[i].v_obj = nullptr;
      g_reg_file[i].type_index = kTVMFFINone;
    }
  }

  return freed;
}

/*---------------------------------------------------------------------------
 * Core Builtin Implementations
 *---------------------------------------------------------------------------*/

TVMDSPStorage* TVMDSPBuiltinAllocStorage(int64_t size, int32_t device_index,
                                          DLDataType dtype_hint) {
  DLDevice device;
  (void)device_index;  /* Single device on DSP */

  /* Use CPU device type for DSP (could use a custom type) */
  device.device_type = kDLCPU;
  device.device_id = 0;

  return TVMDSPStorageAlloc(static_cast<size_t>(size), device, dtype_hint);
}

TVMDSPNDArray* TVMDSPBuiltinAllocTensor(TVMDSPStorage* storage, int64_t offset,
                                         const int64_t* shape, int32_t ndim,
                                         DLDataType dtype) {
  return TVMDSPStorageAllocNDArray(storage, offset, shape, ndim, dtype);
}

TVMDSPNDArray* TVMDSPBuiltinAllocShapeHeap(int64_t size) {
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

TVMDSPShape* TVMDSPBuiltinMakeShape(TVMDSPNDArray* heap, int32_t ndim,
                                     const int32_t* codes, const int64_t* values) {
  int64_t shape_data[TVM_DSP_SHAPE_MAX_NDIM];
  int64_t* heap_data = nullptr;

  if (ndim > TVM_DSP_SHAPE_MAX_NDIM) {
    return nullptr;
  }

  if (heap != nullptr) {
    heap_data = static_cast<int64_t*>(heap->data);
  }

  for (int32_t i = 0; i < ndim; i++) {
    auto code = static_cast<TVMDSPMakeShapeCode>(codes[i]);
    int64_t val = values[i];

    if (code == kMakeShapeUseImm) {
      shape_data[i] = val;
    } else if (code == kMakeShapeLoadShape) {
      if (heap_data == nullptr) {
        tvm_dsp_log("ERROR: make_shape requires heap for LoadShape\n");
        return nullptr;
      }
      shape_data[i] = heap_data[val];
    } else {
      tvm_dsp_log("ERROR: unknown make_shape code %d\n", code);
      return nullptr;
    }
  }

  return TVMDSPShapeCreate(shape_data, ndim);
}

int TVMDSPBuiltinMatchShape(const TVMFFIAny* input, TVMDSPNDArray* heap,
                            int32_t ndim, const int32_t* codes,
                            const int64_t* values) {
  int64_t* heap_data = nullptr;
  const int64_t* input_shape = nullptr;
  int32_t input_ndim = 0;

  /* Extract shape from input - could be NDArray or Shape */
  if (input->type_index == kTVMFFITensor) {
    auto arr = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(input->v_obj);
    input_shape = arr->shape;
    input_ndim = arr->ndim;
  } else if (input->type_index == kTVMFFIShape) {
    auto shp = tvm::dsp::TypedHandle<TVMDSPShape>::FromRaw(input->v_obj);
    input_shape = shp->data;
    input_ndim = static_cast<int32_t>(shp->size);
  } else {
    tvm_dsp_log("ERROR: match_shape input must be NDArray or Shape\n");
    return -1;
  }

  /* Validate ndim */
  if (input_ndim != ndim) {
    tvm_dsp_log("ERROR: match_shape ndim mismatch: %d != %d\n", input_ndim, ndim);
    return -1;
  }

  if (heap != nullptr) {
    heap_data = static_cast<int64_t*>(heap->data);
  }

  for (int32_t i = 0; i < ndim; i++) {
    auto code = static_cast<TVMDSPMatchShapeCode>(codes[i]);
    int64_t val = values[i];

    switch (code) {
      case kMatchShapeAssertEqualToImm:
        if (input_shape[i] != val) {
          tvm_dsp_log("ERROR: shape[%d] = %lld, expected %lld\n",
                      i, static_cast<long long>(input_shape[i]),
                      static_cast<long long>(val));
          return -1;
        }
        break;

      case kMatchShapeStoreToHeap:
        if (heap_data == nullptr) {
          tvm_dsp_log("ERROR: match_shape StoreToHeap requires heap\n");
          return -1;
        }
        heap_data[val] = input_shape[i];
        break;

      case kMatchShapeNoOp:
        /* Skip */
        break;

      case kMatchShapeAssertEqualToLoad:
        if (heap_data == nullptr) {
          tvm_dsp_log("ERROR: match_shape AssertEqualToLoad requires heap\n");
          return -1;
        }
        if (input_shape[i] != heap_data[val]) {
          tvm_dsp_log("ERROR: shape[%d] = %lld, heap[%lld] = %lld\n",
                      i, static_cast<long long>(input_shape[i]),
                      static_cast<long long>(val),
                      static_cast<long long>(heap_data[val]));
          return -1;
        }
        break;

      default:
        tvm_dsp_log("ERROR: unknown match_shape code %d\n", code);
        return -1;
    }
  }

  return 0;
}

int TVMDSPBuiltinCheckTensorInfo(const TVMDSPNDArray* tensor, int32_t ndim,
                                  DLDataType dtype) {
  if (tensor == nullptr) {
    tvm_dsp_log("ERROR: check_tensor_info: tensor is NULL\n");
    return -1;
  }

  /* Check ndim if specified */
  if (ndim != -1 && tensor->ndim != ndim) {
    tvm_dsp_log("ERROR: tensor ndim %d != expected %d\n", tensor->ndim, ndim);
    return -1;
  }

  /* Check dtype if not void */
  if (dtype.code != kDLOpaqueHandle) {
    if (tensor->dtype.code != dtype.code ||
        tensor->dtype.bits != dtype.bits ||
        tensor->dtype.lanes != dtype.lanes) {
      tvm_dsp_log("ERROR: tensor dtype mismatch\n");
      return -1;
    }
  }

  return 0;
}

int TVMDSPBuiltinCheckShapeInfo(const TVMDSPShape* shape, int32_t ndim) {
  if (shape == nullptr) {
    tvm_dsp_log("ERROR: check_shape_info: shape is NULL\n");
    return -1;
  }

  if (ndim != -1 && static_cast<int32_t>(shape->size) != ndim) {
    tvm_dsp_log("ERROR: shape size %zu != expected %d\n", shape->size, ndim);
    return -1;
  }

  return 0;
}

void TVMDSPBuiltinNullValue(TVMFFIAny* out) {
  out->type_index = kTVMFFINone;
  out->small_len = 0;
  out->v_obj = nullptr;
}

/*---------------------------------------------------------------------------
 * Packed Function Wrappers
 *---------------------------------------------------------------------------*/

int TVMDSPBuiltinAllocStoragePacked(const TVMFFIAny* args, int32_t num_args,
                                     TVMFFIAny* ret) {
  TVMDSPStorage* storage;
  int64_t size;
  int32_t device_index;
  DLDataType dtype_hint;

  /* Expected args: [ctx_ptr (ignored), shape, device_index, dtype] */
  /* For DSP, we simplify: [size, device_index, dtype] */
  if (num_args < 3) {
    tvm_dsp_log("ERROR: alloc_storage requires 3+ args, got %d\n", num_args);
    return -1;
  }

  /* First arg could be ctx_ptr (ignored) or size directly */
  if (args[0].type_index == kTVMFFIInt) {
    size = args[0].v_int64;
    device_index = static_cast<int32_t>(args[1].v_int64);
    dtype_hint = *reinterpret_cast<const DLDataType*>(&args[2].v_int64);
  } else if (num_args >= 4) {
    /* First arg is ctx_ptr, skip it */
    if (args[1].type_index == kTVMFFIShape) {
      auto shape = tvm::dsp::TypedHandle<TVMDSPShape>::FromRaw(args[1].v_obj);
      int64_t nelems = TVMDSPShapeProduct(shape.get());
      DLDataType dt = *reinterpret_cast<const DLDataType*>(&args[3].v_int64);
      size = nelems * (dt.bits * dt.lanes + 7) / 8;
    } else {
      size = args[1].v_int64;
    }
    device_index = static_cast<int32_t>(args[2].v_int64);
    dtype_hint = *reinterpret_cast<const DLDataType*>(&args[3].v_int64);
  } else {
    tvm_dsp_log("ERROR: alloc_storage invalid args\n");
    return -1;
  }

  storage = TVMDSPBuiltinAllocStorage(size, device_index, dtype_hint);
  if (storage == nullptr) {
    return -1;
  }

  ret->type_index = TVM_DSP_STORAGE_TYPE_INDEX;
  ret->small_len = 0;
  ret->v_obj = reinterpret_cast<TVMFFIObject*>(storage);
  return 0;
}

int TVMDSPBuiltinAllocTensorPacked(const TVMFFIAny* args, int32_t num_args,
                                    TVMFFIAny* ret) {
  /* Expected args: [storage, offset, shape, dtype] */
  if (num_args < 4) {
    tvm_dsp_log("ERROR: alloc_tensor requires 4 args, got %d\n", num_args);
    return -1;
  }

  TVMDSPStorage* storage = TVMDSPAnyAsStorage(&args[0]);
  if (storage == nullptr) {
    tvm_dsp_log("ERROR: alloc_tensor: first arg is not storage\n");
    return -1;
  }

  int64_t offset = args[1].v_int64;

  if (args[2].type_index != kTVMFFIShape) {
    tvm_dsp_log("ERROR: alloc_tensor: third arg is not shape (type=%d)\n",
                args[2].type_index);
    return -1;
  }
  auto shape = tvm::dsp::TypedHandle<TVMDSPShape>::FromRaw(args[2].v_obj);

  DLDataType dtype = *reinterpret_cast<const DLDataType*>(&args[3].v_int64);

  TVMDSPNDArray* arr = TVMDSPBuiltinAllocTensor(
      storage, offset, shape->data, static_cast<int32_t>(shape->size), dtype);
  if (arr == nullptr) {
    return -1;
  }

  ret->type_index = kTVMFFITensor;
  ret->small_len = 0;
  ret->v_obj = reinterpret_cast<TVMFFIObject*>(arr);
  return 0;
}

int TVMDSPBuiltinAllocShapeHeapPacked(const TVMFFIAny* args, int32_t num_args,
                                       TVMFFIAny* ret) {
  /* Expected args: [ctx_ptr (ignored), size] */
  if (num_args < 2) {
    tvm_dsp_log("ERROR: alloc_shape_heap requires 2 args, got %d\n", num_args);
    return -1;
  }

  int64_t size = args[1].v_int64;

  TVMDSPNDArray* heap = TVMDSPBuiltinAllocShapeHeap(size);
  if (heap == nullptr) {
    return -1;
  }

  ret->type_index = kTVMFFITensor;
  ret->small_len = 0;
  ret->v_obj = reinterpret_cast<TVMFFIObject*>(heap);
  return 0;
}

int TVMDSPBuiltinMakeShapePacked(const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* ret) {
  TVMDSPNDArray* heap = nullptr;
  int32_t codes[TVM_DSP_SHAPE_MAX_NDIM];
  int64_t values[TVM_DSP_SHAPE_MAX_NDIM];

  /* Expected args: [heap, ndim, code0, val0, code1, val1, ...] */
  if (num_args < 2) {
    tvm_dsp_log("ERROR: make_shape requires 2+ args, got %d\n", num_args);
    return -1;
  }

  /* First arg is heap (can be null) */
  if (args[0].type_index == kTVMFFITensor) {
    heap = reinterpret_cast<TVMDSPNDArray*>(args[0].v_obj);
  }

  int64_t ndim = args[1].v_int64;

  if (ndim > TVM_DSP_SHAPE_MAX_NDIM) {
    tvm_dsp_log("ERROR: make_shape ndim %lld exceeds max\n",
                static_cast<long long>(ndim));
    return -1;
  }

  /* Check we have enough args for code/value pairs */
  if (num_args < 2 + ndim * 2) {
    tvm_dsp_log("ERROR: make_shape needs %lld code/val pairs\n",
                static_cast<long long>(ndim));
    return -1;
  }

  /* Extract codes and values */
  for (int64_t i = 0; i < ndim; i++) {
    codes[i] = static_cast<int32_t>(args[2 + i * 2].v_int64);
    values[i] = args[2 + i * 2 + 1].v_int64;
  }

  TVMDSPShape* shape = TVMDSPBuiltinMakeShape(heap, static_cast<int32_t>(ndim),
                                               codes, values);
  if (shape == nullptr) {
    return -1;
  }

  ret->type_index = kTVMFFIShape;
  ret->small_len = 0;
  ret->v_obj = reinterpret_cast<TVMFFIObject*>(shape);
  return 0;
}

int TVMDSPBuiltinMatchShapePacked(const TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* ret) {
  TVMDSPNDArray* heap = nullptr;
  int32_t codes[TVM_DSP_SHAPE_MAX_NDIM];
  int64_t values[TVM_DSP_SHAPE_MAX_NDIM];

  /* Expected args: [input, heap, ndim, code0, val0, ..., err_ctx] */
  if (num_args < 3) {
    tvm_dsp_log("ERROR: match_shape requires 3+ args, got %d\n", num_args);
    return -1;
  }

  /* Second arg is heap (can be null) */
  if (args[1].type_index == kTVMFFITensor) {
    heap = reinterpret_cast<TVMDSPNDArray*>(args[1].v_obj);
  }

  int64_t ndim = args[2].v_int64;

  if (ndim > TVM_DSP_SHAPE_MAX_NDIM) {
    tvm_dsp_log("ERROR: match_shape ndim %lld exceeds max\n",
                static_cast<long long>(ndim));
    return -1;
  }

  /* Extract codes and values */
  for (int64_t i = 0; i < ndim; i++) {
    codes[i] = static_cast<int32_t>(args[3 + i * 2].v_int64);
    values[i] = args[3 + i * 2 + 1].v_int64;
  }

  int result = TVMDSPBuiltinMatchShape(&args[0], heap, static_cast<int32_t>(ndim),
                                        codes, values);

  /* Return void on success */
  ret->type_index = kTVMFFINone;
  ret->small_len = 0;
  ret->v_obj = nullptr;

  return result;
}

int TVMDSPBuiltinCheckTensorInfoPacked(const TVMFFIAny* args, int32_t num_args,
                                        TVMFFIAny* ret) {
  /* Expected args: [tensor, ndim, dtype (optional), err_ctx] */
  if (num_args < 2) {
    tvm_dsp_log("ERROR: check_tensor_info requires 2+ args\n");
    return -1;
  }

  if (args[0].type_index != kTVMFFITensor) {
    tvm_dsp_log("ERROR: check_tensor_info: first arg is not NDArray (type=%d)\n",
                args[0].type_index);
    return -1;
  }
  auto tensor = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(args[0].v_obj);

  int32_t ndim = static_cast<int32_t>(args[1].v_int64);

  DLDataType dtype;
  /* dtype is optional */
  if (num_args >= 3 && args[2].type_index == kTVMFFIDataType) {
    dtype = *reinterpret_cast<const DLDataType*>(&args[2].v_int64);
  } else {
    /* Use void dtype (any type accepted) */
    dtype.code = kDLOpaqueHandle;
    dtype.bits = 0;
    dtype.lanes = 0;
  }

  int result = TVMDSPBuiltinCheckTensorInfo(tensor.get(), ndim, dtype);

  ret->type_index = kTVMFFINone;
  ret->small_len = 0;
  ret->v_obj = nullptr;

  return result;
}

int TVMDSPBuiltinNullValuePacked(const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* ret) {
  (void)args;
  (void)num_args;

  TVMDSPBuiltinNullValue(ret);
  return 0;
}

/*---------------------------------------------------------------------------
 * Additional Builtins for Generated Code Compatibility
 *---------------------------------------------------------------------------*/

int TVMDSPBuiltinCopyPacked(const TVMFFIAny* args, int32_t num_args,
                             TVMFFIAny* ret) {
  if (num_args < 1) {
    tvm_dsp_log("ERROR: copy requires 1 arg\n");
    return -1;
  }

  /* Just return the input - copy is identity in TVM */
  *ret = args[0];

  /* Increment ref count if this is an object */
  if (TVMFFIAnyIsObject(ret) && ret->v_obj) {
    static_cast<TVMFFIObject*>(ret->v_obj)->ref_counter++;
  }

  return 0;
}

int TVMDSPBuiltinMakeTuplePacked(const TVMFFIAny* args, int32_t num_args,
                                  TVMFFIAny* ret) {
  /* Validate element count */
  if (num_args > TVM_DSP_ARRAY_MAX_ELEMENTS) {
    tvm_dsp_log("ERROR: make_tuple max %d elements, got %d\n",
                TVM_DSP_ARRAY_MAX_ELEMENTS, num_args);
    return -1;
  }

  /* Handle empty tuple */
  if (num_args == 0) {
    TVMFFIAnySetNone(ret);
    return 0;
  }

  /* Allocate array container from L3 memory */
  void* mem = tvm_dsp_alloc(sizeof(TVMDSPArray), TVM_DSP_CACHE_LINE_SIZE,
                            TVM_DSP_MEM_MAIN);
  if (mem == nullptr) {
    tvm_dsp_log("ERROR: make_tuple failed to allocate TVMDSPArray\n");
    return -1;
  }

  /* Initialize array object header */
  TVMDSPArray* arr = static_cast<TVMDSPArray*>(mem);
  arr->type_index = kTVMFFIArray;
  arr->ref_counter = 1;
  arr->deleter = TVMDSPArrayDeleter;
  arr->size = num_args;
  arr->_padding = 0;

  /* Copy all elements with reference count increment */
  for (int32_t i = 0; i < num_args; i++) {
    arr->elements[i] = args[i];
    /* Increment ref count on contained objects */
    if (TVMFFIAnyIsObject(&args[i]) && args[i].v_obj != nullptr) {
      static_cast<TVMFFIObject*>(args[i].v_obj)->ref_counter++;
    }
  }

  /* Return the array object */
  ret->type_index = kTVMFFIArray;
  ret->v_obj = reinterpret_cast<TVMFFIObject*>(arr);
  return 0;
}

int TVMDSPBuiltinReshapePacked(const TVMFFIAny* args, int32_t num_args,
                                TVMFFIAny* ret) {
  if (num_args < 2) {
    tvm_dsp_log("ERROR: reshape requires 2 args\n");
    return -1;
  }

  auto arr = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(args[0].v_obj);
  auto shape = tvm::dsp::TypedHandle<TVMDSPShape>::FromRaw(args[1].v_obj);

  if (arr.IsNull()) {
    tvm_dsp_log("ERROR: reshape got NULL NDArray\n");
    return -1;
  }

  if (shape.IsNull()) {
    tvm_dsp_log("ERROR: reshape got NULL Shape\n");
    return -1;
  }

  int32_t new_ndim = static_cast<int32_t>(shape->size);

  /* Create new NDArray with same data but different shape */
  /* This is a view - shares data with original */
  /* Allocate from L3 (L2 reserved for TVMBackendAllocWorkspace) */
  size_t alloc_size = sizeof(TVMDSPNDArray) + new_ndim * sizeof(int64_t);
  void* result_mem = tvm_dsp_alloc(alloc_size, TVM_DSP_CACHE_LINE_SIZE,
                                    TVM_DSP_MEM_MAIN);
  if (result_mem == nullptr) {
    return -1;
  }

  auto result = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(result_mem);

  /* Set TVMFFIObject header fields */
  result->type_index = kTVMFFITensor;
  result->ref_counter = 1;
  result->deleter = ReshapeViewDeleter;  /* Frees struct only, not data */

  /* Copy DLTensor fields */
  result->data = arr->data;
  result->device = arr->device;
  result->dtype = arr->dtype;
  result->ndim = new_ndim;
  result->byte_offset = arr->byte_offset;
  result->strides = nullptr;

  /* Copy new shape - stored inline after the struct */
  result->shape = reinterpret_cast<int64_t*>(result.get() + 1);
  for (size_t i = 0; i < shape->size; i++) {
    result->shape[i] = shape->data[i];
  }

  ret->type_index = kTVMFFITensor;
  ret->small_len = 0;
  ret->v_obj = reinterpret_cast<TVMFFIObject*>(result.get());

  return 0;
}

/*---------------------------------------------------------------------------
 * Registration
 *---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
 * Direct Functions (no packed wrapper overhead)
 *
 * These functions are called directly from generated code, bypassing the
 * packed argument marshalling. They provide the same functionality as the
 * Packed versions but without TVMFFIAny extraction/validation overhead.
 *---------------------------------------------------------------------------*/

TVMDSPNDArray* TVMDSPBuiltinReshapeDirect(TVMDSPNDArray* arr,
                                           const int64_t* shape_data,
                                           int32_t shape_size) {
  if (arr == nullptr || shape_data == nullptr) {
    tvm_dsp_log("ERROR: reshape got NULL input\n");
    return nullptr;
  }

  /* Create new NDArray with same data but different shape */
  /* This is a view - shares data with original */
  /* Allocate from L3 (L2 reserved for TVMBackendAllocWorkspace) */
  size_t alloc_size = sizeof(TVMDSPNDArray) + shape_size * sizeof(int64_t);
  void* result_mem = tvm_dsp_alloc(alloc_size, TVM_DSP_CACHE_LINE_SIZE,
                                    TVM_DSP_MEM_MAIN);
  if (result_mem == nullptr) {
    return nullptr;
  }

  auto result = tvm::dsp::TypedHandle<TVMDSPNDArray>::FromRaw(result_mem);

  /* Set TVMFFIObject header fields */
  result->type_index = kTVMFFITensor;
  result->ref_counter = 1;
  result->deleter = ReshapeViewDeleter;  /* Frees struct only, not data */

  /* Copy DLTensor fields */
  result->data = arr->data;
  result->device = arr->device;
  result->dtype = arr->dtype;
  result->ndim = shape_size;
  result->byte_offset = arr->byte_offset;
  result->strides = nullptr;

  /* Copy new shape - stored inline after the struct */
  result->shape = reinterpret_cast<int64_t*>(result.get() + 1);
  for (int32_t i = 0; i < shape_size; i++) {
    result->shape[i] = shape_data[i];
  }

  return result.get();
}

TVMDSPNDArray* TVMDSPBuiltinMakeTupleDirect(TVMFFIAny* values, int32_t num_values) {
  /* For simplicity, just return the first NDArray if present */
  /* Full tuple support would require dynamic allocation */
  if (num_values > 0 && values[0].type_index == kTVMFFITensor) {
    TVMDSPNDArray* arr = reinterpret_cast<TVMDSPNDArray*>(values[0].v_obj);
    if (arr != nullptr) {
      arr->ref_counter++;
    }
    return arr;
  }
  return nullptr;
}

int TVMDSPRegisterVMBuiltins(void) {
  int ret = 0;

  ret |= TVMRegistryRegister("vm.builtin.alloc_storage",
                             TVMDSPBuiltinAllocStoragePacked);
  ret |= TVMRegistryRegister("vm.builtin.alloc_tensor",
                             TVMDSPBuiltinAllocTensorPacked);
  ret |= TVMRegistryRegister("vm.builtin.alloc_shape_heap",
                             TVMDSPBuiltinAllocShapeHeapPacked);
  ret |= TVMRegistryRegister("vm.builtin.make_shape",
                             TVMDSPBuiltinMakeShapePacked);
  ret |= TVMRegistryRegister("vm.builtin.match_shape",
                             TVMDSPBuiltinMatchShapePacked);
  ret |= TVMRegistryRegister("vm.builtin.check_tensor_info",
                             TVMDSPBuiltinCheckTensorInfoPacked);
  ret |= TVMRegistryRegister("vm.builtin.null_value",
                             TVMDSPBuiltinNullValuePacked);
  ret |= TVMRegistryRegister("vm.builtin.copy",
                             TVMDSPBuiltinCopyPacked);
  ret |= TVMRegistryRegister("vm.builtin.make_tuple",
                             TVMDSPBuiltinMakeTuplePacked);
  ret |= TVMRegistryRegister("vm.builtin.reshape",
                             TVMDSPBuiltinReshapePacked);

  return ret;
}

}  /* extern "C" */
