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
 * \file cpp/packed_func.h
 * \brief C++ PackedFunc wrapper for TVM DSP Runtime
 *
 * Provides a C++ interface compatible with tvm::runtime::PackedFunc.
 */

#ifndef TVM_DSP_RUNTIME_CPP_PACKED_FUNC_H_
#define TVM_DSP_RUNTIME_CPP_PACKED_FUNC_H_

#include "any.h"

extern "C" {
#include "../registry/packed_func.h"
#include "../registry/registry.h"
}

#include <cstddef>   /* std::nullptr_t */
#include <utility>   /* std::move, std::forward */

namespace tvm {
namespace runtime {

/*!
 * \brief C++ wrapper for PackedFunc
 *
 * This class wraps TVMDSPPackedFunc and provides a callable interface
 * compatible with TVM's PackedFunc.
 */
class PackedFunc {
 public:
  /*! \brief Default constructor - null function */
  PackedFunc() : func_(nullptr) {}

  /*! \brief Construct from raw TVMDSPPackedFunc pointer */
  explicit PackedFunc(TVMDSPPackedFunc* func) : func_(func) {
    if (func_) {
      func_->ref_counter++;
    }
  }

  /*! \brief Copy constructor */
  PackedFunc(const PackedFunc& other) : func_(other.func_) {
    if (func_) {
      func_->ref_counter++;
    }
  }

  /*! \brief Move constructor */
  PackedFunc(PackedFunc&& other) noexcept : func_(other.func_) {
    other.func_ = nullptr;
  }

  /*! \brief Destructor */
  ~PackedFunc() {
    if (func_) {
      func_->ref_counter--;
      if (func_->ref_counter == 0 && func_->deleter) {
        func_->deleter(func_);
      }
    }
  }

  /*! \brief Copy assignment */
  PackedFunc& operator=(const PackedFunc& other) {
    if (this != &other) {
      if (func_) {
        func_->ref_counter--;
        if (func_->ref_counter == 0 && func_->deleter) {
          func_->deleter(func_);
        }
      }
      func_ = other.func_;
      if (func_) {
        func_->ref_counter++;
      }
    }
    return *this;
  }

  /*! \brief Move assignment */
  PackedFunc& operator=(PackedFunc&& other) noexcept {
    if (this != &other) {
      if (func_) {
        func_->ref_counter--;
        if (func_->ref_counter == 0 && func_->deleter) {
          func_->deleter(func_);
        }
      }
      func_ = other.func_;
      other.func_ = nullptr;
    }
    return *this;
  }

  /*! \brief Check if the function is valid */
  bool defined() const { return func_ != nullptr && func_->func != nullptr; }

  /*! \brief Check if null */
  bool operator==(std::nullptr_t) const { return !defined(); }
  bool operator!=(std::nullptr_t) const { return defined(); }

  /*!
   * \brief Call the packed function
   * \param args Array of arguments
   * \param num_args Number of arguments
   * \param result Output result
   * \return 0 on success, non-zero on error
   */
  int Call(const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) const {
    if (!defined()) {
      return -1;
    }
    return func_->func(args, num_args, result);
  }

  /*!
   * \brief Call operator with Any arguments
   * \param args Arguments as ffi::Any objects
   * \return Result as ffi::Any
   */
  template<typename... Args>
  ffi::Any operator()(Args&&... args) const {
    TVMFFIAny arg_values[sizeof...(Args) > 0 ? sizeof...(Args) : 1];
    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;

    if (defined()) {
      SetArgs(arg_values, 0, std::forward<Args>(args)...);
      func_->func(arg_values, sizeof...(Args), &result);
    }

    return ffi::details::AnyUnsafe::MoveTVMFFIAnyToAny(std::move(result));
  }

  /*! \brief Get raw function pointer */
  TVMDSPPackedFunc* get() const { return func_; }

 private:
  TVMDSPPackedFunc* func_;

  // Helper to set arguments (base case)
  static void SetArgs(TVMFFIAny* args, int index) {}

  // Helper to set int64_t argument
  static void SetArgs(TVMFFIAny* args, int index, int64_t value) {
    args[index].type_index = kTVMFFIInt;
    args[index].zero_padding = 0;
    args[index].v_int64 = value;
  }

  // Helper to set int argument
  static void SetArgs(TVMFFIAny* args, int index, int value) {
    args[index].type_index = kTVMFFIInt;
    args[index].zero_padding = 0;
    args[index].v_int64 = static_cast<int64_t>(value);
  }

  // Helper to set double argument
  static void SetArgs(TVMFFIAny* args, int index, double value) {
    args[index].type_index = kTVMFFIFloat;
    args[index].zero_padding = 0;
    args[index].v_float64 = value;
  }

  // Helper to set Any argument
  static void SetArgs(TVMFFIAny* args, int index, const ffi::Any& value) {
    args[index] = value.raw();
  }

  // Recursive helper to set multiple arguments
  template<typename T, typename... Rest>
  static void SetArgs(TVMFFIAny* args, int index, T&& first, Rest&&... rest) {
    SetArgs(args, index, std::forward<T>(first));
    SetArgs(args, index + 1, std::forward<Rest>(rest)...);
  }
};

/*!
 * \brief Get a global function by name
 * \param name Function name
 * \return PackedFunc if found, null PackedFunc otherwise
 */
inline PackedFunc GetGlobalFunc(const char* name) {
  TVMFFIObjectHandle handle = TVMRegistryLookup(name);
  if (handle) {
    return PackedFunc(reinterpret_cast<TVMDSPPackedFunc*>(handle));
  }
  return PackedFunc();
}

}  // namespace runtime
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_PACKED_FUNC_H_ */
