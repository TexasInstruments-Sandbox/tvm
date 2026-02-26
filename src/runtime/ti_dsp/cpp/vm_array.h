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
 * \file cpp/vm_array.h
 * \brief Typed wrapper for TVMFFIAny arrays (VM registers, constants)
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * AnyArray provides direct typed access to VM register and constant arrays
 * without FFI boxing/unboxing overhead. It wraps a TVMFFIAny* array and
 * provides typed getters and setters.
 *
 * PROBLEM IT SOLVES:
 *
 * The current FFI-based VM builtin calls require significant overhead:
 *
 *   // 7 statements, ~280 cycles overhead
 *   TVMBackendAnyListSetPackedArg(r, 2, stack_ffi_any, 0);
 *   SetFFIAnyInt(&((stack_ffi_any)[1]), (long)0);
 *   TVMBackendAnyListSetPackedArg(c, 5, stack_ffi_any, 2);
 *   TVMBackendAnyListSetPackedArg(c, 6, stack_ffi_any, 3);
 *   SetFFIAnyNone(&((stack_ffi_any)[4]));
 *   if (TVMFFIFunctionCall(vm_builtin_alloc_tensor, ...)) return -1;
 *   TVMBackendAnyListMoveFromPackedReturn(r, 3, stack_ffi_any, 4);
 *
 * WITH ANYARRAY:
 *
 *   // 1 statement, ~150 cycles (actual work only)
 *   r.SetNDArray(3, AllocTensor(r.GetStorage(2), 0, c.GetShape(5), c.GetDType(6)));
 *
 * =============================================================================
 * DESIGN PRINCIPLES
 * =============================================================================
 *
 * 1. NO RUNTIME VALIDATION
 *    - Assumes compiler has verified types at compile time
 *    - Direct casts without type checks for maximum performance
 *    - Debug builds can add assertions if needed
 *
 * 2. AUTOMATIC REFERENCE COUNTING
 *    - Setters automatically DecRef the old value before overwriting
 *    - New objects are assumed to have ref_count=1 from allocation
 *
 * 3. ZERO OVERHEAD ABSTRACTION
 *    - All methods are inline
 *    - No virtual functions
 *    - Direct memory access through typed getters
 *
 * =============================================================================
 * USAGE
 * =============================================================================
 *
 *   // In generated code prologue
 *   AnyArray _r(r);  // r is void* to register file
 *   AnyArray _c(c);  // c is void* to constants array
 *
 *   // Typed access
 *   TVMDSPStorage* storage = _r.GetStorage(2);
 *   TVMDSPShape* shape = _c.GetShape(5);
 *   DLDataType dtype = _c.GetDType(6);
 *
 *   // Setting with automatic DecRef
 *   _r.SetNDArray(3, AllocTensor(storage, 0, shape, dtype));
 *   _r.SetNone(5);  // Kill register (DecRef old value)
 *
 *   // Raw access when needed
 *   _r[10].v_int64 = 42;
 */

#ifndef TVM_RUNTIME_TI_DSP_CPP_VM_ARRAY_H_
#define TVM_RUNTIME_TI_DSP_CPP_VM_ARRAY_H_

#include "../ffi/ffi_types.h"
#include "../container/array.h"
#include "../container/ndarray.h"
#include "../container/shape.h"
#include "../vm/storage.h"

namespace tvm {
namespace dsp {
namespace vm {

/*!
 * \brief Typed wrapper for TVMFFIAny arrays (VM registers, constants)
 *
 * Provides direct typed access to array elements without FFI boxing/unboxing.
 * No runtime type validation - assumes compiler has verified types.
 */
class AnyArray {
 public:
  /*!
   * \brief Construct AnyArray wrapper
   * \param data Pointer to TVMFFIAny array
   */
  explicit AnyArray(TVMFFIAny* data) : data_(data) {}

  /*!
   * \brief Construct AnyArray from void* (for generated code compatibility)
   * \param data Pointer to TVMFFIAny array (as void*)
   */
  explicit AnyArray(void* data) : data_(static_cast<TVMFFIAny*>(data)) {}

  /*---------------------------------------------------------------------------
   * Typed Getters (no validation, direct cast)
   *---------------------------------------------------------------------------*/

  /*!
   * \brief Get storage object from array
   * \param idx Array index
   * \return Storage pointer (no ownership transfer)
   */
  TVMDSPStorage* GetStorage(int idx) const {
    return reinterpret_cast<TVMDSPStorage*>(data_[idx].v_obj);
  }

  /*!
   * \brief Get shape object from array
   * \param idx Array index
   * \return Shape pointer (no ownership transfer)
   */
  TVMDSPShape* GetShape(int idx) const {
    return reinterpret_cast<TVMDSPShape*>(data_[idx].v_obj);
  }

  /*!
   * \brief Get data type from array
   * \param idx Array index
   * \return DLDataType value
   *
   * Note: dtype is stored as int64 in FFIAny, we reinterpret the bits
   */
  DLDataType GetDType(int idx) const {
    DLDataType dtype;
    dtype.code = static_cast<uint8_t>(data_[idx].v_int64 & 0xFF);
    dtype.bits = static_cast<uint8_t>((data_[idx].v_int64 >> 8) & 0xFF);
    dtype.lanes = static_cast<uint16_t>((data_[idx].v_int64 >> 16) & 0xFFFF);
    return dtype;
  }

  /*!
   * \brief Get NDArray object from array
   * \param idx Array index
   * \return NDArray pointer (no ownership transfer)
   */
  TVMDSPNDArray* GetNDArray(int idx) const {
    return reinterpret_cast<TVMDSPNDArray*>(data_[idx].v_obj);
  }

  /*!
   * \brief Get integer value from array
   * \param idx Array index
   * \return int64_t value
   */
  int64_t GetInt(int idx) const {
    return data_[idx].v_int64;
  }

  /*!
   * \brief Get pointer value from array
   * \param idx Array index
   * \return void* pointer
   */
  void* GetPtr(int idx) const {
    return data_[idx].v_ptr;
  }

  /*!
   * \brief Get size value from shape at idx (common pattern for alloc_storage)
   * \param idx Array index containing shape object
   * \return First element of shape (size for 1D allocation)
   */
  int64_t GetSize(int idx) const {
    TVMDSPShape* shape = GetShape(idx);
    return shape ? shape->data[0] : 0;
  }

  /*!
   * \brief Get raw TVMFFIAny value
   * \param idx Array index
   * \return Const reference to TVMFFIAny
   */
  const TVMFFIAny& GetAny(int idx) const {
    return data_[idx];
  }

  /*---------------------------------------------------------------------------
   * Typed Setters (with automatic reference counting)
   *---------------------------------------------------------------------------*/

  /*!
   * \brief Set NDArray in array (with DecRef of old value)
   * \param idx Array index
   * \param arr NDArray pointer (ownership transferred, arr has ref_count=1)
   *
   * The old value at idx is DecRef'd before overwriting.
   * The new arr is assumed to already have ref_count=1 from allocation.
   */
  void SetNDArray(int idx, TVMDSPNDArray* arr) {
    DecRefOld(idx);
    data_[idx].type_index = kTVMFFITensor;
    data_[idx].small_len = 0;
    data_[idx].v_obj = reinterpret_cast<TVMFFIObject*>(arr);
  }

  /*!
   * \brief Set storage in array (with DecRef of old value)
   * \param idx Array index
   * \param storage Storage pointer (ownership transferred)
   */
  void SetStorage(int idx, TVMDSPStorage* storage) {
    DecRefOld(idx);
    data_[idx].type_index = TVM_DSP_STORAGE_TYPE_INDEX;
    data_[idx].small_len = 0;
    data_[idx].v_obj = reinterpret_cast<TVMFFIObject*>(storage);
  }

  /*!
   * \brief Set shape in array (with DecRef of old value)
   * \param idx Array index
   * \param shape Shape pointer (ownership transferred)
   */
  void SetShape(int idx, TVMDSPShape* shape) {
    DecRefOld(idx);
    data_[idx].type_index = kTVMFFIShape;
    data_[idx].small_len = 0;
    data_[idx].v_obj = reinterpret_cast<TVMFFIObject*>(shape);
  }

  /*!
   * \brief Set None value in array (with DecRef of old value)
   * \param idx Array index
   *
   * This is equivalent to "killing" a register - releases the old reference.
   */
  void SetNone(int idx) {
    DecRefOld(idx);
    data_[idx].type_index = kTVMFFINone;
    data_[idx].small_len = 0;
    data_[idx].v_obj = nullptr;
  }

  /*!
   * \brief Set Array (tuple) in array (with DecRef of old value)
   * \param idx Array index
   * \param arr TVMDSPArray pointer (ownership transferred, arr has ref_count=1)
   *
   * Used for multi-element make_tuple results.
   */
  void SetArray(int idx, TVMDSPArray* arr) {
    DecRefOld(idx);
    data_[idx].type_index = kTVMFFIArray;
    data_[idx].small_len = 0;
    data_[idx].v_obj = reinterpret_cast<TVMFFIObject*>(arr);
  }

  /*!
   * \brief Set integer value in array (with DecRef of old value)
   * \param idx Array index
   * \param value Integer value
   */
  void SetInt(int idx, int64_t value) {
    DecRefOld(idx);
    data_[idx].type_index = kTVMFFIInt;
    data_[idx].small_len = 0;
    data_[idx].v_int64 = value;
  }

  /*---------------------------------------------------------------------------
   * Cross-Array Operations (for argument marshaling)
   *---------------------------------------------------------------------------*/

  /*!
   * \brief Copy value from another AnyArray (with IncRef if object)
   * \param dst_idx Destination index in this array
   * \param src Source array
   * \param src_idx Source index in src array
   *
   * Copies the TVMFFIAny value and increments reference count for objects.
   * DecRefs the old value at dst_idx before overwriting.
   */
  void SetFrom(const AnyArray& src, int dst_idx, int src_idx) {
    DecRefOld(dst_idx);
    data_[dst_idx] = src.data_[src_idx];
    // IncRef for object types
    if (data_[dst_idx].type_index >= kTVMFFIStaticObjectBegin &&
        data_[dst_idx].v_obj != nullptr) {
      data_[dst_idx].v_obj->ref_counter++;
    }
  }

  /*!
   * \brief Copy value from another AnyArray without DecRef (for uninitialized slots)
   * \param dst_idx Destination index in this array
   * \param src Source array
   * \param src_idx Source index in src array
   *
   * Same as SetFrom but skips DecRefOld. Use ONLY when the destination slot
   * is known to be empty (None or zero-initialized). This is the case for:
   * - Stack arrays that are zero-initialized with `= {}`
   * - Slots that have just been cleared with SetNone
   *
   * Using this on a slot containing a live object will leak memory!
   */
  void SetFromUnchecked(const AnyArray& src, int dst_idx, int src_idx) {
    data_[dst_idx] = src.data_[src_idx];
    // IncRef for object types
    if (data_[dst_idx].type_index >= kTVMFFIStaticObjectBegin &&
        data_[dst_idx].v_obj != nullptr) {
      data_[dst_idx].v_obj->ref_counter++;
    }
  }

  /*!
   * \brief Move value from another AnyArray (no ref count change)
   * \param dst_idx Destination index in this array
   * \param src Source array (will be modified)
   * \param src_idx Source index in src array
   *
   * Moves the TVMFFIAny value without changing reference count.
   * The source slot is set to None after the move.
   * DecRefs the old value at dst_idx before overwriting.
   */
  void MoveFrom(AnyArray& src, int dst_idx, int src_idx) {
    DecRefOld(dst_idx);
    data_[dst_idx] = src.data_[src_idx];
    // Clear source (no DecRef - we moved ownership)
    src.data_[src_idx].type_index = kTVMFFINone;
    src.data_[src_idx].v_obj = nullptr;
  }

  /*---------------------------------------------------------------------------
   * Raw Access (for pass-through and special cases)
   *---------------------------------------------------------------------------*/

  /*!
   * \brief Get mutable reference to element
   * \param idx Array index
   * \return Reference to TVMFFIAny element
   */
  TVMFFIAny& operator[](int idx) {
    return data_[idx];
  }

  /*!
   * \brief Get const reference to element
   * \param idx Array index
   * \return Const reference to TVMFFIAny element
   */
  const TVMFFIAny& operator[](int idx) const {
    return data_[idx];
  }

  /*!
   * \brief Get raw pointer to array
   * \return Pointer to underlying TVMFFIAny array
   */
  TVMFFIAny* data() {
    return data_;
  }

  /*!
   * \brief Get const raw pointer to array
   * \return Const pointer to underlying TVMFFIAny array
   */
  const TVMFFIAny* data() const {
    return data_;
  }

 private:
  TVMFFIAny* data_;  /*!< Pointer to TVMFFIAny array */

  /*!
   * \brief Decrement reference count of old object value
   * \param idx Array index
   *
   * If the slot contains an object (type_index >= kTVMFFIStaticObjectBegin),
   * decrement its reference count. If ref_count becomes 0, call deleter.
   */
  void DecRefOld(int idx) {
    TVMFFIAny* slot = &data_[idx];
    if (slot->type_index >= kTVMFFIStaticObjectBegin && slot->v_obj != nullptr) {
      TVMFFIObject* obj = slot->v_obj;
      if (--obj->ref_counter == 0 && obj->deleter != nullptr) {
        obj->deleter(obj);
      }
    }
  }
};

}  // namespace vm
}  // namespace dsp
}  // namespace tvm

#endif  // TVM_RUNTIME_TI_DSP_CPP_VM_ARRAY_H_
