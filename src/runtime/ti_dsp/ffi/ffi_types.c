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
 * \file ffi_types.c
 * \brief TVM DSP Runtime - FFI type implementation
 *
 * This file provides the implementation of FFI operations for the DSP runtime.
 * Reference counting is simplified for single-threaded DSP execution (no atomics).
 */

#include "ffi_types.h"
#include "../core/logging.h"

/*---------------------------------------------------------------------------
 * Type Name Table
 *---------------------------------------------------------------------------*/

/*!
 * \brief Get human-readable type name from type index.
 *
 * This is primarily used for debugging and error messages.
 */
const char* TVMDSPGetTypeName(int32_t type_index) {
  switch (type_index) {
    case kTVMFFINone:
      return "None";
    case kTVMFFIInt:
      return "Int";
    case kTVMFFIBool:
      return "Bool";
    case kTVMFFIFloat:
      return "Float";
    case kTVMFFIOpaquePtr:
      return "OpaquePtr";
    case kTVMFFIDataType:
      return "DataType";
    case kTVMFFIDevice:
      return "Device";
    case kTVMFFIDLTensorPtr:
      return "DLTensorPtr";
    case kTVMFFIRawStr:
      return "RawStr";
    case kTVMFFIByteArrayPtr:
      return "ByteArrayPtr";
    case kTVMFFIObjectRValueRef:
      return "ObjectRValueRef";
    case kTVMFFIObject:
      return "Object";
    case kTVMFFIStr:
      return "String";
    case kTVMFFIBytes:
      return "Bytes";
    case kTVMFFIError:
      return "Error";
    case kTVMFFIFunction:
      return "Function";
    case kTVMFFIArray:
      return "Array";
    case kTVMFFIMap:
      return "Map";
    case kTVMFFIShape:
      return "Shape";
    case kTVMFFITensor:
      return "Tensor";
    case kTVMFFIModule:
      return "Module";
    default:
      if (type_index >= kTVMFFIDynObjectBegin) {
        return "DynamicObject";
      }
      return "Unknown";
  }
}

/*---------------------------------------------------------------------------
 * Object Reference Counting
 *
 * Note: Single-threaded DSP - no atomic operations needed.
 *---------------------------------------------------------------------------*/

void TVMFFIObjectIncRef(TVMFFIObject* obj) {
  if (obj != NULL) {
    obj->ref_counter++;
  }
}

void TVMFFIObjectDecRef(TVMFFIObject* obj) {
  if (obj != NULL) {
    obj->ref_counter--;
    if (obj->ref_counter == 0) {
      if (obj->deleter != NULL) {
        obj->deleter(obj);
      } else {
        /* Default: log warning - object should have deleter */
        TVM_DSP_LOGW("Object at %p has no deleter (type=%d)\n",
                     (void*)obj, obj->type_index);
      }
    }
  }
}

int TVMFFIObjectFree(TVMFFIObjectHandle handle) {
  TVMFFIObjectDecRef((TVMFFIObject*)handle);
  return 0;
}

/*---------------------------------------------------------------------------
 * Any Value Operations
 *---------------------------------------------------------------------------*/

void TVMFFIAnyMove(TVMFFIAny* src, TVMFFIAny* dst) {
  /* Copy the raw bytes */
  dst->type_index = src->type_index;
  dst->small_len = src->small_len;
  dst->v_int64 = src->v_int64;

  /* Clear source (no ref count change since we're moving) */
  src->type_index = kTVMFFINone;
  src->small_len = 0;
  src->v_int64 = 0;
}

void TVMFFIAnyCopy(const TVMFFIAny* src, TVMFFIAny* dst) {
  /* Copy the raw bytes */
  dst->type_index = src->type_index;
  dst->small_len = src->small_len;
  dst->v_int64 = src->v_int64;

  /* If it's an object type, increment reference count */
  if (src->type_index >= kTVMFFIStaticObjectBegin) {
    TVMFFIObjectIncRef(src->v_obj);
  }
}

void TVMFFIAnyClear(TVMFFIAny* any) {
  /* If it's an object type, decrement reference count */
  if (any->type_index >= kTVMFFIStaticObjectBegin) {
    TVMFFIObjectDecRef(any->v_obj);
  }

  /* Reset to None */
  any->type_index = kTVMFFINone;
  any->small_len = 0;
  any->v_int64 = 0;
}
