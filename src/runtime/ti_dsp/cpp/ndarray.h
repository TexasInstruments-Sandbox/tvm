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
 * \file cpp/ndarray.h
 * \brief C++ NDArray wrapper for TVM DSP Runtime
 *
 * Provides a C++ interface compatible with tvm::runtime::NDArray.
 */

#ifndef TVM_DSP_RUNTIME_CPP_NDARRAY_H_
#define TVM_DSP_RUNTIME_CPP_NDARRAY_H_

#include "object_ref.h"

extern "C" {
#include "../container/ndarray.h"
}

#include <utility>  /* std::move */

namespace tvm {
namespace runtime {

/*!
 * \brief C++ wrapper for NDArray
 *
 * This class provides RAII-style management of NDArray objects
 * and an interface compatible with TVM's NDArray class.
 */
class NDArray : public ObjectRef {
 public:
  /*! \brief Default constructor - null array */
  NDArray() : ObjectRef() {}

  /*! \brief Construct from C NDArray pointer */
  explicit NDArray(TVMDSPNDArray* arr)
      : ObjectRef(reinterpret_cast<TVMFFIObject*>(arr)) {}

  /*! \brief Copy constructor */
  NDArray(const NDArray& other) : ObjectRef(other) {}

  /*! \brief Move constructor */
  NDArray(NDArray&& other) noexcept : ObjectRef(std::move(other)) {}

  /*! \brief Copy assignment */
  NDArray& operator=(const NDArray& other) {
    ObjectRef::operator=(other);
    return *this;
  }

  /*! \brief Move assignment */
  NDArray& operator=(NDArray&& other) noexcept {
    ObjectRef::operator=(std::move(other));
    return *this;
  }

  /*! \brief Get the underlying DLTensor pointer */
  DLTensor* operator->() const {
    TVMDSPNDArray* arr = get_mutable();
    /* DLTensor fields are embedded at offset sizeof(TVMFFIObject) */
    return arr ? TVMFFINDArrayGetDLTensorPtr(reinterpret_cast<TVMFFIObjectHandle>(arr)) : nullptr;
  }

  /*! \brief Get data pointer */
  void* data() const {
    TVMDSPNDArray* arr = get_mutable();
    return arr ? arr->data : nullptr;
  }

  /*! \brief Get shape array */
  const int64_t* shape() const {
    TVMDSPNDArray* arr = get_mutable();
    return arr ? arr->shape : nullptr;
  }

  /*! \brief Get number of dimensions */
  int ndim() const {
    TVMDSPNDArray* arr = get_mutable();
    return arr ? arr->ndim : 0;
  }

  /*! \brief Get data type */
  DLDataType dtype() const {
    TVMDSPNDArray* arr = get_mutable();
    if (arr) {
      return arr->dtype;
    }
    DLDataType dt = {0, 0, 0};
    return dt;
  }

  /*! \brief Get device */
  DLDevice device() const {
    TVMDSPNDArray* arr = get_mutable();
    if (arr) {
      return arr->device;
    }
    DLDevice dev = {kDLCPU, 0};
    return dev;
  }

  /*! \brief Check if the NDArray is contiguous */
  bool IsContiguous() const {
    TVMDSPNDArray* arr = get_mutable();
    return arr ? TVMDSPNDArrayIsContiguous(arr) : false;
  }

  /*! \brief Get total number of elements */
  int64_t Size() const {
    TVMDSPNDArray* arr = get_mutable();
    if (!arr) return 0;

    int64_t size = 1;
    for (int i = 0; i < arr->ndim; i++) {
      size *= arr->shape[i];
    }
    return size;
  }

  /*! \brief Get data size in bytes */
  size_t DataSize() const {
    TVMDSPNDArray* arr = get_mutable();
    return arr ? TVMDSPNDArrayDataSize(arr) : 0;
  }

  /*!
   * \brief Create an empty NDArray
   * \param shape The shape of the array
   * \param ndim Number of dimensions
   * \param dtype Data type
   * \param device Device
   * \return The created NDArray
   */
  static NDArray Empty(const int64_t* shape, int ndim,
                       DLDataType dtype, DLDevice device) {
    TVMDSPNDArray* arr = TVMDSPNDArrayAlloc(shape, ndim, dtype, device);
    NDArray ret;
    ret.data_ = reinterpret_cast<TVMFFIObject*>(arr);
    return ret;
  }

  /*! \brief Get raw C pointer */
  TVMDSPNDArray* get_mutable() const {
    return reinterpret_cast<TVMDSPNDArray*>(data_);
  }
};

}  // namespace runtime
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_NDARRAY_H_ */
