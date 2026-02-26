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
 * \file cpp/object_ref.h
 * \brief Minimal C++ object reference wrapper for TVM DSP Runtime
 *
 * Provides RAII-style object management compatible with TVM's C++ API.
 * This is a simplified version that wraps our C implementation.
 */

#ifndef TVM_DSP_RUNTIME_CPP_OBJECT_REF_H_
#define TVM_DSP_RUNTIME_CPP_OBJECT_REF_H_

extern "C" {
#include "../ffi/ffi_types.h"
#include "../ffi/object.h"
}

#include <cstddef>  /* std::nullptr_t */
#include <utility>  /* std::move */

namespace tvm {
namespace runtime {

/*!
 * \brief Base class for all object references
 *
 * ObjectRef is a smart pointer that manages reference counting
 * for TVM objects. It provides RAII semantics.
 */
class ObjectRef {
 public:
  /*! \brief Default constructor - null reference */
  ObjectRef() : data_(nullptr) {}

  /*! \brief Construct from raw pointer (takes ownership) */
  explicit ObjectRef(TVMFFIObject* data) : data_(data) {
    // Note: We assume data already has ref_count = 1 from allocation
  }

  /*! \brief Copy constructor - increments reference count */
  ObjectRef(const ObjectRef& other) : data_(other.data_) {
    IncRef();
  }

  /*! \brief Move constructor - transfers ownership */
  ObjectRef(ObjectRef&& other) noexcept : data_(other.data_) {
    other.data_ = nullptr;
  }

  /*! \brief Destructor - decrements reference count */
  ~ObjectRef() {
    DecRef();
  }

  /*! \brief Copy assignment */
  ObjectRef& operator=(const ObjectRef& other) {
    if (this != &other) {
      DecRef();
      data_ = other.data_;
      IncRef();
    }
    return *this;
  }

  /*! \brief Move assignment */
  ObjectRef& operator=(ObjectRef&& other) noexcept {
    if (this != &other) {
      DecRef();
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  /*! \brief Assign nullptr to release the reference */
  ObjectRef& operator=(std::nullptr_t) {
    DecRef();
    data_ = nullptr;
    return *this;
  }

  /*! \brief Check if reference is null */
  bool defined() const { return data_ != nullptr; }

  /*! \brief Check if reference is null */
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

  /*! \brief Get raw pointer (does not transfer ownership) */
  TVMFFIObject* get() const { return data_; }

  /*! \brief Get type index */
  int32_t type_index() const {
    return data_ ? data_->type_index : kTVMFFINone;
  }

 protected:
  /*! \brief Internal pointer to managed object */
  TVMFFIObject* data_;

  /*! \brief Increment reference count */
  void IncRef() {
    if (data_ != nullptr) {
      data_->ref_counter++;
    }
  }

  /*! \brief Decrement reference count, free if zero */
  void DecRef() {
    if (data_ != nullptr) {
      data_->ref_counter--;
      if (data_->ref_counter == 0 && data_->deleter != nullptr) {
        data_->deleter(data_);
      }
    }
  }
};

}  // namespace runtime
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_OBJECT_REF_H_ */
