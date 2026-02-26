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
 * \file cpp/typed_handle.h
 * \brief Type-safe handle wrapper for FFI object pointers
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * TypedHandle<T> is a type-safe wrapper for raw pointers/handles. It provides
 * compile-time type safety for what would otherwise be void* handles.
 *
 * PROBLEM IT SOLVES:
 *
 * In C FFI code, objects are often passed as void* handles:
 *
 *   void* ndarray_handle = create_ndarray();
 *   void* shape_handle = create_shape();
 *
 *   // Easy to mix up handles - compiler won't catch this!
 *   process_ndarray(shape_handle);  // BUG: wrong handle type!
 *
 * WITH TYPED_HANDLE:
 *
 *   TypedHandle<NDArray> ndarray = create_ndarray();
 *   TypedHandle<Shape> shape = create_shape();
 *
 *   process_ndarray(shape);  // COMPILE ERROR! Type mismatch
 *
 * =============================================================================
 * C++ CONCEPTS USED (for C programmers)
 * =============================================================================
 *
 * 1. WRAPPER CLASS
 *    - TypedHandle wraps a raw pointer (T*)
 *    - Provides type safety without runtime overhead
 *    - The wrapped pointer is accessible via get()
 *
 * 2. EXPLICIT CONSTRUCTOR
 *    - Prevents accidental implicit conversions
 *    - You must explicitly create handles: TypedHandle<T>(ptr)
 *
 * 3. ARROW AND DEREFERENCE OPERATORS
 *    - operator-> allows: handle->method()
 *    - operator* allows: T& ref = *handle;
 *
 * 4. NULLPTR
 *    - C++11 type-safe null pointer
 *    - TypedHandle can be null (empty)
 *
 * =============================================================================
 * MEMORY MODEL
 * =============================================================================
 *
 * NO DYNAMIC ALLOCATION. TypedHandle is just a pointer wrapper:
 *
 *   TypedHandle<NDArray> handle(ptr);
 *
 *   Memory layout:
 *   +----------+
 *   | ptr_ (T*)|
 *   +----------+
 *
 *   Total size: sizeof(T*) = 4 or 8 bytes depending on platform
 *
 * OWNERSHIP:
 * TypedHandle is NON-OWNING. It does not manage the lifetime
 * of the pointed-to object. The object must be freed separately.
 *
 * =============================================================================
 * USAGE EXAMPLES
 * =============================================================================
 *
 * Creating handles:
 *
 *   // From typed pointer
 *   NDArray* raw_ptr = get_ndarray();
 *   TypedHandle<NDArray> handle(raw_ptr);
 *
 *   // From void* with explicit cast
 *   void* raw_handle = some_c_api();
 *   auto handle = TypedHandle<NDArray>::FromRaw(raw_handle);
 *
 *   // Null handle
 *   TypedHandle<NDArray> null_handle;  // or nullptr
 *
 * Accessing the wrapped object:
 *
 *   handle->some_method();       // Arrow operator
 *   NDArray& ref = *handle;      // Dereference
 *   NDArray* ptr = handle.get(); // Get raw pointer
 *
 * Null checking:
 *
 *   if (handle) { ... }          // Handle is valid
 *   if (handle.IsNull()) { ... } // Handle is null
 *
 * =============================================================================
 */

#ifndef TVM_DSP_RUNTIME_CPP_TYPED_HANDLE_H_
#define TVM_DSP_RUNTIME_CPP_TYPED_HANDLE_H_

#include <cstddef>     /* size_t, nullptr_t */
#include <cstdint>     /* int32_t */

/* Include FFI types for TVMFFIObject definition */
extern "C" {
#include "ffi_types.h"
}

namespace tvm {
namespace dsp {

/*
 * =============================================================================
 * HANDLE ERROR CODES
 * =============================================================================
 */

/*!
 * \brief Error codes specific to handle operations
 */
enum class HandleError {
  kNullHandle,      /*!< Handle is null */
  kTypeMismatch,    /*!< Type check failed during cast */
  kInvalidHandle    /*!< Handle points to invalid memory */
};

/*
 * =============================================================================
 * TYPED HANDLE CLASS
 * =============================================================================
 */

/*!
 * \brief Type-safe non-owning handle wrapper
 *
 * TypedHandle<T> wraps a pointer to T, providing type safety at compile time.
 * It does NOT own the pointed-to object and will not free it.
 *
 * \tparam T The type of the pointed-to object
 *
 * THREAD SAFETY:
 * TypedHandle itself is thread-safe for read operations. Modifications to
 * the pointed-to object require external synchronization.
 *
 * LIFETIME:
 * The pointed-to object must outlive the TypedHandle. Accessing a handle
 * after the object is destroyed is undefined behavior.
 */
template <typename T>
class TypedHandle {
 public:
  using element_type = T;         /*!< The type of the pointed-to object */
  using pointer = T*;             /*!< Pointer type */

  /*!
   * \brief Default constructor - creates null handle
   */
  TypedHandle() : ptr_(nullptr) {}

  /*!
   * \brief Construct from nullptr
   */
  TypedHandle(std::nullptr_t) : ptr_(nullptr) {}

  /*!
   * \brief Construct from typed pointer
   *
   * \param ptr Pointer to wrap (can be null)
   */
  explicit TypedHandle(T* ptr) : ptr_(ptr) {}

  /*!
   * \brief Copy constructor
   */
  TypedHandle(const TypedHandle& other) = default;

  /*!
   * \brief Copy assignment
   */
  TypedHandle& operator=(const TypedHandle& other) = default;

  /*!
   * \brief Assign nullptr
   */
  TypedHandle& operator=(std::nullptr_t) {
    ptr_ = nullptr;
    return *this;
  }

  /*!
   * \brief Create handle from raw void pointer
   *
   * \param raw_ptr Raw pointer to convert
   * \return TypedHandle<T> wrapping the cast pointer
   *
   * WARNING: This performs an unchecked cast.
   */
  static TypedHandle<T> FromRaw(void* raw_ptr) {
    return TypedHandle<T>(static_cast<T*>(raw_ptr));
  }

  /*!
   * \brief Get the raw pointer
   * \return The wrapped pointer (may be null)
   */
  T* get() const { return ptr_; }

  /*!
   * \brief Arrow operator for member access
   *
   * WARNING: Calling on null handle is undefined behavior!
   */
  T* operator->() const { return ptr_; }

  /*!
   * \brief Dereference operator
   *
   * WARNING: Calling on null handle is undefined behavior!
   */
  T& operator*() const { return *ptr_; }

  /*!
   * \brief Check if handle is null
   */
  bool IsNull() const { return ptr_ == nullptr; }

  /*!
   * \brief Check if handle is valid (non-null)
   */
  bool IsValid() const { return ptr_ != nullptr; }

  /*!
   * \brief Boolean conversion
   */
  explicit operator bool() const { return ptr_ != nullptr; }

  /*!
   * \brief Cast to another handle type (unchecked)
   *
   * \tparam U The target type
   * \return TypedHandle<U> pointing to the same memory
   */
  template <typename U>
  TypedHandle<U> Cast() const {
    return TypedHandle<U>(reinterpret_cast<U*>(ptr_));
  }

  /*!
   * \brief Cast to void handle
   */
  TypedHandle<void> ToVoid() const;

  /* Comparison operators */
  bool operator==(const TypedHandle& other) const { return ptr_ == other.ptr_; }
  bool operator!=(const TypedHandle& other) const { return ptr_ != other.ptr_; }
  bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }

  /*!
   * \brief Release the pointer and return it
   */
  T* Release() {
    T* temp = ptr_;
    ptr_ = nullptr;
    return temp;
  }

  /*!
   * \brief Reset the handle to a new pointer
   */
  void Reset(T* ptr = nullptr) { ptr_ = ptr; }

 private:
  T* ptr_;
};

/*
 * =============================================================================
 * VOID SPECIALIZATION
 * =============================================================================
 */

template <>
class TypedHandle<void> {
 public:
  TypedHandle() : ptr_(nullptr) {}
  TypedHandle(std::nullptr_t) : ptr_(nullptr) {}
  explicit TypedHandle(void* ptr) : ptr_(ptr) {}
  TypedHandle(const TypedHandle& other) = default;
  TypedHandle& operator=(const TypedHandle& other) = default;

  TypedHandle& operator=(std::nullptr_t) {
    ptr_ = nullptr;
    return *this;
  }

  void* get() const { return ptr_; }
  bool IsNull() const { return ptr_ == nullptr; }
  bool IsValid() const { return ptr_ != nullptr; }
  explicit operator bool() const { return ptr_ != nullptr; }

  template <typename U>
  TypedHandle<U> Cast() const {
    return TypedHandle<U>(static_cast<U*>(ptr_));
  }

  bool operator==(const TypedHandle& other) const { return ptr_ == other.ptr_; }
  bool operator!=(const TypedHandle& other) const { return ptr_ != other.ptr_; }
  bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }

  void* Release() {
    void* temp = ptr_;
    ptr_ = nullptr;
    return temp;
  }

  void Reset(void* ptr = nullptr) { ptr_ = ptr; }

 private:
  void* ptr_;
};

/* Implementation of ToVoid (needs void specialization first) */
template <typename T>
TypedHandle<void> TypedHandle<T>::ToVoid() const {
  return TypedHandle<void>(static_cast<void*>(ptr_));
}

/*
 * =============================================================================
 * FFI OBJECT HANDLE
 * =============================================================================
 */

/*!
 * \brief Handle for FFI objects with type index checking
 *
 * This class wraps FFI objects and provides runtime type checking.
 *
 * \tparam T The FFI object type (must have TVMFFIObject as first member)
 */
template <typename T>
class FFIHandle {
 public:
  FFIHandle() : ptr_(nullptr) {}
  FFIHandle(std::nullptr_t) : ptr_(nullptr) {}
  explicit FFIHandle(T* ptr) : ptr_(ptr) {}
  FFIHandle(const FFIHandle& other) = default;
  FFIHandle& operator=(const FFIHandle& other) = default;

  FFIHandle& operator=(std::nullptr_t) {
    ptr_ = nullptr;
    return *this;
  }

  /* Pointer access */
  T* get() const { return ptr_; }
  T* operator->() const { return ptr_; }
  T& operator*() const { return *ptr_; }

  /* State checking */
  bool IsNull() const { return ptr_ == nullptr; }
  bool IsValid() const { return ptr_ != nullptr; }
  explicit operator bool() const { return ptr_ != nullptr; }

  /* Comparison */
  bool operator==(const FFIHandle& other) const { return ptr_ == other.ptr_; }
  bool operator!=(const FFIHandle& other) const { return ptr_ != other.ptr_; }
  bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }

  /*!
   * \brief Create from raw pointer with type check
   *
   * \param raw_ptr Raw pointer to convert
   * \param expected_type The expected type_index
   * \param[out] out_handle Output handle (set on success)
   * \return HandleError::kNullHandle if null, HandleError::kTypeMismatch if
   *         type doesn't match, or kSuccess (cast to HandleError = 0) on success
   */
  static HandleError FromRawChecked(void* raw_ptr, int32_t expected_type,
                                    FFIHandle<T>* out_handle) {
    if (raw_ptr == nullptr) {
      return HandleError::kNullHandle;
    }

    /* Access type_index from TVMFFIObject header */
    TVMFFIObject* obj = static_cast<TVMFFIObject*>(raw_ptr);
    if (obj->type_index != expected_type) {
      return HandleError::kTypeMismatch;
    }

    *out_handle = FFIHandle<T>(static_cast<T*>(raw_ptr));
    return static_cast<HandleError>(0);  /* Success */
  }

  /*!
   * \brief Get the type index of the wrapped object
   */
  int32_t GetTypeIndex() const {
    if (ptr_ == nullptr) return -1;
    return reinterpret_cast<TVMFFIObject*>(ptr_)->type_index;
  }

  /*!
   * \brief Check if the object has the expected type
   */
  bool HasType(int32_t expected_type) const {
    if (ptr_ == nullptr) return false;
    return GetTypeIndex() == expected_type;
  }

 private:
  T* ptr_;
};

/*
 * =============================================================================
 * HELPER FUNCTIONS
 * =============================================================================
 */

/*!
 * \brief Create a TypedHandle with type deduction
 */
template <typename T>
TypedHandle<T> MakeHandle(T* ptr) {
  return TypedHandle<T>(ptr);
}

/*!
 * \brief Create a null TypedHandle
 */
template <typename T>
TypedHandle<T> NullHandle() {
  return TypedHandle<T>(nullptr);
}

/*!
 * \brief Check if a handle is valid (non-null)
 *
 * \param handle The handle to check
 * \param[out] error Set to HandleError::kNullHandle if null
 * \return true if valid, false if null
 */
template <typename T>
bool CheckHandle(const TypedHandle<T>& handle, HandleError* error) {
  if (handle.IsNull()) {
    if (error) *error = HandleError::kNullHandle;
    return false;
  }
  return true;
}

}  // namespace dsp
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_TYPED_HANDLE_H_ */
