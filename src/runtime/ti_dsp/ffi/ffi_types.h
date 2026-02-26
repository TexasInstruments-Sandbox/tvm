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
 * \file ffi_types.h
 * \brief TVM DSP Runtime - FFI type definitions
 *
 * This header provides FFI type definitions compatible with TVM's
 * ffi/c_api.h. The types use the SAME names (TVMFFIAny, TVMFFIObject, etc.)
 * to ensure drop-in compatibility with TVM-generated code.
 *
 * When building for DSP targets, this header is included instead of
 * the full TVM headers. The definitions are ABI-compatible but simplified
 * for embedded use (no exceptions, no dynamic type registration, etc.).
 */
#ifndef TVM_RUNTIME_TI_DSP_FFI_FFI_TYPES_H_
#define TVM_RUNTIME_TI_DSP_FFI_FFI_TYPES_H_

/* Use DLPack types directly */
#include <dlpack/dlpack.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------------------------------
 * Type Indices - Must match TVM's ffi/c_api.h exactly
 *
 * These are defined in the same enum format as TVM for compatibility.
 *---------------------------------------------------------------------------*/

#ifdef __cplusplus
enum TVMFFITypeIndex : int32_t {
#else
typedef enum {
#endif
  /* POD and special types: [0, 64) */
  kTVMFFIAny = -1,           /*!< Root type (never in Any::type_index) */
  kTVMFFINone = 0,           /*!< None/nullptr value */
  kTVMFFIInt = 1,            /*!< POD int value */
  kTVMFFIBool = 2,           /*!< POD bool value */
  kTVMFFIFloat = 3,          /*!< POD float value */
  kTVMFFIOpaquePtr = 4,      /*!< Opaque pointer */
  kTVMFFIDataType = 5,       /*!< DLDataType */
  kTVMFFIDevice = 6,         /*!< DLDevice */
  kTVMFFIDLTensorPtr = 7,    /*!< DLTensor* */
  kTVMFFIRawStr = 8,         /*!< const char* (raw string) */
  kTVMFFIByteArrayPtr = 9,   /*!< TVMFFIByteArray* */
  kTVMFFIObjectRValueRef = 10, /*!< R-value reference to ObjectRef */
  kTVMFFISmallStr = 11,      /*!< Small string on stack */
  kTVMFFISmallBytes = 12,    /*!< Small bytes on stack */

  /* Static objects: [64, 128) -- matches TVM 0.23.0 ABI */
  kTVMFFIStaticObjectBegin = 64,
  kTVMFFIObject = 64,        /*!< Base object type */
  kTVMFFIStr = 65,           /*!< String object */
  kTVMFFIBytes = 66,         /*!< Bytes object */
  kTVMFFIError = 67,         /*!< Error object */
  kTVMFFIFunction = 68,      /*!< Function object */
  kTVMFFIShape = 69,         /*!< Shape object */
  kTVMFFITensor = 70,        /*!< Tensor object (was NDArray=72 in 0.21) */
  kTVMFFIArray = 71,         /*!< Array object (was 69 in 0.21) */
  kTVMFFIMap = 72,           /*!< Map object (was 70 in 0.21) */
  kTVMFFIModule = 73,        /*!< Runtime module object */
  kTVMFFIOpaquePyObject = 74, /*!< Opaque Python object */
  kTVMFFIStaticObjectEnd,

  /* Dynamic objects: [128, +oo) */
  kTVMFFIDynObjectBegin = 128
#ifdef __cplusplus
};
#else
} TVMFFITypeIndex;
#endif

/*---------------------------------------------------------------------------
 * Handle Type
 *---------------------------------------------------------------------------*/

/*! \brief Handle to Object from C API's pov */
typedef void* TVMFFIObjectHandle;

/*---------------------------------------------------------------------------
 * Byte Array - Used by String and Bytes objects
 *---------------------------------------------------------------------------*/

/*!
 * \brief Byte array data structure.
 *
 * String and Bytes object layout = { TVMFFIObject, TVMFFIByteArray, ... }
 */
typedef struct {
  const char* data;
  size_t size;
} TVMFFIByteArray;

/*---------------------------------------------------------------------------
 * Object Header - Base for all heap-allocated objects
 *---------------------------------------------------------------------------*/

/*!
 * \brief C-based type of all FFI object header that allocates on heap.
 *
 * TVMFFIObject and TVMFFIAny share the common type_index header.
 * This structure must be at the beginning of all object types.
 */
typedef struct TVMFFIObject {
  int32_t type_index;    /*!< Type index of the object */
  int32_t ref_counter;   /*!< Reference count */
  union {
    /*! \brief Deleter to be invoked when reference counter goes to zero */
    void (*deleter)(struct TVMFFIObject* self);
    /*! \brief Ensure 8-byte alignment */
    int64_t __ensure_align;
  };
} TVMFFIObject;

/*---------------------------------------------------------------------------
 * Any Value Container - Polymorphic value type
 *---------------------------------------------------------------------------*/

/*!
 * \brief C-based type of all on stack Any value.
 *
 * Any value can hold on stack values like int,
 * as well as reference counted pointers to object.
 *
 * Layout: 16 bytes total (4 + 4 + 8)
 */
typedef struct TVMFFIAny {
  int32_t type_index;    /*!< Type index (TVMFFITypeIndex) */
  int32_t small_len;     /*!< Length for small-string optimization (reserved) */
  union {
    int64_t v_int64;           /*!< Integer value */
    double v_float64;          /*!< Floating-point value */
    void* v_ptr;               /*!< Generic pointer */
    const char* v_c_str;       /*!< Raw C string */
    TVMFFIObject* v_obj;       /*!< Object pointer */
    DLDataType v_dtype;        /*!< Data type */
    DLDevice v_device;         /*!< Device */
    char v_bytes[8];           /*!< Small string storage */
    uint64_t v_uint64;         /*!< Unsigned integer (for hashing) */
  };
} TVMFFIAny;

/*---------------------------------------------------------------------------
 * Shape Cell - Used by Shape objects
 *---------------------------------------------------------------------------*/

/*!
 * \brief Shape cell used in shape object following header.
 */
typedef struct {
  const int64_t* data;
  size_t size;
} TVMFFIShapeCell;

/*---------------------------------------------------------------------------
 * Safe Call Function Type
 *---------------------------------------------------------------------------*/

/*!
 * \brief Type that defines C-style safe call convention.
 *
 * \param handle The function handle
 * \param args The input arguments to the call
 * \param num_args Number of input arguments
 * \param result Store output result
 * \return 0 if successful, -1 if error
 *
 * IMPORTANT: caller must initialize result->type_index to kTVMFFINone
 */
typedef int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args,
                                   int32_t num_args, TVMFFIAny* result);

/*---------------------------------------------------------------------------
 * Function Cell - Used by Function objects
 *---------------------------------------------------------------------------*/

/*!
 * \brief Object cell for function object following header.
 */
typedef struct {
  TVMFFISafeCallType safe_call;
} TVMFFIFunctionCell;

/*---------------------------------------------------------------------------
 * Inline Operations for TVMFFIAny
 *---------------------------------------------------------------------------*/

/*!
 * \brief Initialize TVMFFIAny to None.
 */
static inline void TVMFFIAnySetNone(TVMFFIAny* any) {
  any->type_index = kTVMFFINone;
  any->small_len = 0;
  any->v_int64 = 0;
}

/*!
 * \brief Set integer value.
 */
static inline void TVMFFIAnySetInt(TVMFFIAny* any, int64_t value) {
  any->type_index = kTVMFFIInt;
  any->small_len = 0;
  any->v_int64 = value;
}

/*!
 * \brief Set boolean value.
 */
static inline void TVMFFIAnySetBool(TVMFFIAny* any, int value) {
  any->type_index = kTVMFFIBool;
  any->small_len = 0;
  any->v_int64 = value ? 1 : 0;
}

/*!
 * \brief Set float value.
 */
static inline void TVMFFIAnySetFloat(TVMFFIAny* any, double value) {
  any->type_index = kTVMFFIFloat;
  any->small_len = 0;
  any->v_float64 = value;
}

/*!
 * \brief Set opaque pointer value.
 * Note: Explicitly clears union to avoid garbage on 32-bit platforms.
 */
static inline void TVMFFIAnySetPtr(TVMFFIAny* any, void* ptr) {
  any->type_index = kTVMFFIOpaquePtr;
  any->small_len = 0;
  any->v_int64 = 0;  /* Clear full union first */
  any->v_ptr = ptr;
}

/*!
 * \brief Set raw string value (non-owning).
 */
static inline void TVMFFIAnySetRawStr(TVMFFIAny* any, const char* str) {
  any->type_index = kTVMFFIRawStr;
  any->small_len = 0;
  any->v_c_str = str;
}

/*!
 * \brief Set DLDataType value.
 */
static inline void TVMFFIAnySetDataType(TVMFFIAny* any, DLDataType dtype) {
  any->type_index = kTVMFFIDataType;
  any->small_len = 0;
  any->v_dtype = dtype;
}

/*!
 * \brief Set DLDevice value.
 */
static inline void TVMFFIAnySetDevice(TVMFFIAny* any, DLDevice device) {
  any->type_index = kTVMFFIDevice;
  any->small_len = 0;
  any->v_device = device;
}

/*!
 * \brief Set DLTensor pointer.
 */
static inline void TVMFFIAnySetDLTensor(TVMFFIAny* any, DLTensor* tensor) {
  any->type_index = kTVMFFIDLTensorPtr;
  any->small_len = 0;
  any->v_ptr = tensor;
}

/*!
 * \brief Set object reference with specific type index.
 * Note: Explicitly clears union to avoid garbage on 32-bit platforms.
 */
static inline void TVMFFIAnySetObject(TVMFFIAny* any, TVMFFIObject* obj,
                                       int32_t type_index) {
  any->type_index = type_index;
  any->small_len = 0;
  any->v_int64 = 0;  /* Clear full union first */
  any->v_obj = obj;
}

/*!
 * \brief Set NDArray object reference.
 * Note: Explicitly clears union to avoid garbage on 32-bit platforms.
 */
static inline void TVMFFIAnySetNDArray(TVMFFIAny* any, void* ndarray_obj) {
  any->type_index = kTVMFFITensor;
  any->small_len = 0;
  any->v_int64 = 0;  /* Clear full union first */
  any->v_obj = (TVMFFIObject*)ndarray_obj;
}

/*---------------------------------------------------------------------------
 * Value Accessors
 *---------------------------------------------------------------------------*/

static inline int TVMFFIAnyIsNone(const TVMFFIAny* any) {
  return any->type_index == kTVMFFINone;
}

static inline int TVMFFIAnyIsInt(const TVMFFIAny* any) {
  return any->type_index == kTVMFFIInt;
}

static inline int TVMFFIAnyIsFloat(const TVMFFIAny* any) {
  return any->type_index == kTVMFFIFloat;
}

static inline int TVMFFIAnyIsObject(const TVMFFIAny* any) {
  return any->type_index >= kTVMFFIStaticObjectBegin;
}

static inline int64_t TVMFFIAnyGetInt(const TVMFFIAny* any) {
  return any->v_int64;
}

static inline double TVMFFIAnyGetFloat(const TVMFFIAny* any) {
  return any->v_float64;
}

static inline void* TVMFFIAnyGetPtr(const TVMFFIAny* any) {
  return any->v_ptr;
}

static inline TVMFFIObject* TVMFFIAnyGetObject(const TVMFFIAny* any) {
  return any->v_obj;
}

static inline DLDataType TVMFFIAnyGetDataType(const TVMFFIAny* any) {
  return any->v_dtype;
}

static inline DLDevice TVMFFIAnyGetDevice(const TVMFFIAny* any) {
  return any->v_device;
}

/*---------------------------------------------------------------------------
 * Object Accessor Inline Functions (match TVM's c_api.h)
 *---------------------------------------------------------------------------*/

/*!
 * \brief Get the type index of an object.
 */
static inline int32_t TVMFFIObjectGetTypeIndex(TVMFFIObjectHandle obj) {
  return ((TVMFFIObject*)obj)->type_index;
}

/*!
 * \brief Get the byte array pointer from a string/bytes object.
 */
static inline TVMFFIByteArray* TVMFFIBytesGetByteArrayPtr(TVMFFIObjectHandle obj) {
  return (TVMFFIByteArray*)((char*)obj + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the shape cell pointer from a shape object.
 */
static inline TVMFFIShapeCell* TVMFFIShapeGetCellPtr(TVMFFIObjectHandle obj) {
  return (TVMFFIShapeCell*)((char*)obj + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the function cell pointer from a function object.
 */
static inline TVMFFIFunctionCell* TVMFFIFunctionGetCellPtr(TVMFFIObjectHandle obj) {
  return (TVMFFIFunctionCell*)((char*)obj + sizeof(TVMFFIObject));
}

/*!
 * \brief Get the DLTensor pointer from an NDArray object.
 */
static inline DLTensor* TVMFFINDArrayGetDLTensorPtr(TVMFFIObjectHandle obj) {
  return (DLTensor*)((char*)obj + sizeof(TVMFFIObject));
}

/*!
 * \brief Create a DLDevice from device type and id.
 */
static inline DLDevice TVMFFIDLDeviceFromIntPair(int32_t device_type,
                                                  int32_t device_id) {
  DLDevice dev;
  dev.device_type = (DLDeviceType)device_type;
  dev.device_id = device_id;
  return dev;
}

/*---------------------------------------------------------------------------
 * Function Declarations (implemented in ffi_types.c)
 *---------------------------------------------------------------------------*/

/*!
 * \brief Get type name from type index.
 * \param type_index Type index.
 * \return String name of the type, or "Unknown" if invalid.
 */
const char* TVMDSPGetTypeName(int32_t type_index);

/*!
 * \brief Increment object reference count.
 */
void TVMFFIObjectIncRef(TVMFFIObject* obj);

/*!
 * \brief Decrement object reference count.
 * If count reaches zero, deleter is called.
 */
void TVMFFIObjectDecRef(TVMFFIObject* obj);

/*!
 * \brief Free an object handle.
 * \return 0 on success.
 */
int TVMFFIObjectFree(TVMFFIObjectHandle handle);

/*!
 * \brief Move Any value, transferring ownership.
 * Source is set to None after move.
 */
void TVMFFIAnyMove(TVMFFIAny* src, TVMFFIAny* dst);

/*!
 * \brief Copy Any value with reference count increment.
 */
void TVMFFIAnyCopy(const TVMFFIAny* src, TVMFFIAny* dst);

/*!
 * \brief Clear Any value, decrementing reference if needed.
 */
void TVMFFIAnyClear(TVMFFIAny* any);

/*!
 * \brief Decrement ref-count of object in Any (if present), without clearing.
 *
 * Use this when you're about to overwrite the Any with a new value.
 * Use TVMFFIAnyClear() when you want to also reset the Any to None.
 */
static inline void TVMFFIAnyDecRef(const TVMFFIAny* any) {
  if (any != NULL && any->type_index >= kTVMFFIStaticObjectBegin) {
    TVMFFIObjectDecRef(any->v_obj);
  }
}

#ifdef __cplusplus
}
#endif

#endif /* TVM_RUNTIME_TI_DSP_FFI_FFI_TYPES_H_ */
