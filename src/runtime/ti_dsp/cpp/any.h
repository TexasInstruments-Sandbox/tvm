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
 * \file cpp/any.h
 * \brief C++ Any type wrapper for TVM DSP Runtime
 *
 * Provides a C++ interface compatible with tvm::ffi::Any.
 * This is a simplified version for DSP use that wraps TVMFFIAny.
 */

#ifndef TVM_DSP_RUNTIME_CPP_ANY_H_
#define TVM_DSP_RUNTIME_CPP_ANY_H_

#include "object_ref.h"
#include "ndarray.h"

extern "C" {
#include "../ffi/ffi_types.h"
}

#include <cstddef>   /* std::nullptr_t */
#include <cstdint>
#include <utility>   /* std::move */
#include <exception>

namespace tvm {
namespace ffi {

// Forward declaration
class Any;

/*!
 * \brief Utility class for unsafe operations on Any
 *
 * This provides compatibility with TVM's AnyUnsafe operations
 * used in generated code.
 */
namespace details {

class AnyUnsafe {
 public:
  /*!
   * \brief Move a TVMFFIAny to an Any object
   *
   * This is used by generated code to move return values.
   */
  static Any MoveTVMFFIAnyToAny(TVMFFIAny&& src);
};

}  // namespace details

/*!
 * \brief Polymorphic value container compatible with TVM's Any
 *
 * Any can hold integers, floats, pointers, objects, etc.
 * It wraps TVMFFIAny and provides C++ semantics.
 */
class Any {
 public:
  /*! \brief Default constructor - creates None value */
  Any() {
    data_.type_index = kTVMFFINone;
    data_.small_len = 0;
    data_.v_int64 = 0;
  }

  /*! \brief Construct from nullptr */
  Any(std::nullptr_t) {
    data_.type_index = kTVMFFINone;
    data_.small_len = 0;
    data_.v_int64 = 0;
  }

  /*! \brief Construct from integer */
  Any(int64_t value) {
    data_.type_index = kTVMFFIInt;
    data_.small_len = 0;
    data_.v_int64 = value;
  }

  /*! \brief Construct from int */
  Any(int value) {
    data_.type_index = kTVMFFIInt;
    data_.small_len = 0;
    data_.v_int64 = static_cast<int64_t>(value);
  }

  /*! \brief Construct from double */
  Any(double value) {
    data_.type_index = kTVMFFIFloat;
    data_.small_len = 0;
    data_.v_float64 = value;
  }

  /*! \brief Construct from NDArray */
  Any(const runtime::NDArray& arr) {
    data_.type_index = kTVMFFITensor;
    data_.small_len = 0;
    data_.v_obj = arr.get();
    if (data_.v_obj) {
      reinterpret_cast<TVMFFIObject*>(data_.v_obj)->ref_counter++;
    }
  }

  /*! \brief Construct from raw TVMFFIAny */
  explicit Any(const TVMFFIAny& raw) : data_(raw) {
    if (TVMFFIAnyIsObject(&data_) && data_.v_obj) {
      reinterpret_cast<TVMFFIObject*>(data_.v_obj)->ref_counter++;
    }
  }

  /*! \brief Copy constructor */
  Any(const Any& other) : data_(other.data_) {
    if (TVMFFIAnyIsObject(&data_) && data_.v_obj) {
      reinterpret_cast<TVMFFIObject*>(data_.v_obj)->ref_counter++;
    }
  }

  /*! \brief Move constructor */
  Any(Any&& other) noexcept : data_(other.data_) {
    other.data_.type_index = kTVMFFINone;
    other.data_.small_len = 0;
    other.data_.v_int64 = 0;
  }

  /*! \brief Destructor */
  ~Any() {
    Clear();
  }

  /*! \brief Copy assignment */
  Any& operator=(const Any& other) {
    if (this != &other) {
      Clear();
      data_ = other.data_;
      if (TVMFFIAnyIsObject(&data_) && data_.v_obj) {
        reinterpret_cast<TVMFFIObject*>(data_.v_obj)->ref_counter++;
      }
    }
    return *this;
  }

  /*! \brief Move assignment */
  Any& operator=(Any&& other) noexcept {
    if (this != &other) {
      Clear();
      data_ = other.data_;
      other.data_.type_index = kTVMFFINone;
      other.data_.small_len = 0;
      other.data_.v_int64 = 0;
    }
    return *this;
  }

  /*! \brief Assign nullptr */
  Any& operator=(std::nullptr_t) {
    Clear();
    data_.type_index = kTVMFFINone;
    data_.small_len = 0;
    data_.v_int64 = 0;
    return *this;
  }

  /*! \brief Assign from void pointer */
  Any& operator=(void* ptr) {
    Clear();
    data_.type_index = kTVMFFIOpaquePtr;
    data_.small_len = 0;
    data_.v_ptr = ptr;
    return *this;
  }

  /*! \brief Assign from NDArray */
  Any& operator=(const runtime::NDArray& arr) {
    Clear();
    data_.type_index = kTVMFFITensor;
    data_.small_len = 0;
    data_.v_obj = arr.get();
    if (data_.v_obj) {
      reinterpret_cast<TVMFFIObject*>(data_.v_obj)->ref_counter++;
    }
    return *this;
  }

  /*!
   * \brief Type conversion template (stub for generated code)
   * \tparam T Target type
   * \return Optional-like wrapper (always has value for now)
   *
   * This is a simplified stub - real TVM has full type checking.
   */
  template<typename T>
  struct AsResult {
    T value_;
    bool has_value_;
    AsResult(T v) : value_(v), has_value_(true) {}
    AsResult() : value_(), has_value_(false) {}
    T value() const { return value_; }
    bool has_value() const { return has_value_; }
  };

  template<typename T>
  AsResult<T> as() const {
    /* Stub - generated code calls this but we return default */
    return AsResult<T>();
  }

  /*! \brief Get type index */
  int32_t type_index() const { return data_.type_index; }

  /*! \brief Check if value is None */
  bool IsNone() const { return data_.type_index == kTVMFFINone; }

  /*! \brief Check if value is an object */
  bool IsObject() const { return TVMFFIAnyIsObject(&data_) != 0; }

  /*! \brief Get as integer */
  int64_t AsInt() const { return data_.v_int64; }

  /*! \brief Get as double */
  double AsFloat() const { return data_.v_float64; }

  /*! \brief Get raw pointer to underlying TVMFFIAny */
  TVMFFIAny* ptr() { return &data_; }
  const TVMFFIAny* ptr() const { return &data_; }

  /*! \brief Get underlying TVMFFIAny by reference */
  TVMFFIAny& raw() { return data_; }
  const TVMFFIAny& raw() const { return data_; }

 private:
  TVMFFIAny data_;

  /*! \brief Release any held object */
  void Clear() {
    if (TVMFFIAnyIsObject(&data_) && data_.v_obj) {
      TVMFFIObject* obj = reinterpret_cast<TVMFFIObject*>(data_.v_obj);
      obj->ref_counter--;
      if (obj->ref_counter == 0 && obj->deleter) {
        obj->deleter(obj);
      }
    }
  }

  friend class details::AnyUnsafe;
};

// Implementation of AnyUnsafe::MoveTVMFFIAnyToAny
inline Any details::AnyUnsafe::MoveTVMFFIAnyToAny(TVMFFIAny&& src) {
  Any result;
  result.data_ = src;
  // Clear source without decrementing ref count (it's a move)
  src.type_index = kTVMFFINone;
  src.small_len = 0;
  src.v_int64 = 0;
  return result;
}

/*!
 * \brief Exception placeholder for generated code
 *
 * Generated code may throw this but DSP runtime doesn't use exceptions.
 */
class EnvErrorAlreadySet : public std::exception {
 public:
  const char* what() const noexcept override {
    return "EnvErrorAlreadySet (DSP stub)";
  }
};

namespace details {

/*!
 * \brief Exception placeholder for generated code
 */
class MoveFromSafeCallRaised : public std::exception {
 public:
  const char* what() const noexcept override {
    return "MoveFromSafeCallRaised (DSP stub)";
  }
};

}  // namespace details

}  // namespace ffi
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_ANY_H_ */
