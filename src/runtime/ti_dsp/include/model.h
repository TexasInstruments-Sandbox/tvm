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
 * \file include/model.h
 * \brief C++14 Model API for TVM DSP Runtime (Public API)
 *
 * This provides a clean, RAII-based interface for running TVM-generated
 * models on TI DSP processors. See MODEL_API.md for full documentation.
 *
 * This is the only header file users need to include for the Model API.
 *
 * Usage:
 *   #include "model.h"
 *
 *   int main() {
 *     using namespace tvm::dsp;
 *
 *     Model model;
 *     if (model.Load(weights_data, weights_size) != ModelError::kSuccess) {
 *       return 1;
 *     }
 *
 *     auto input = NDArray::Float32(buffer, shape, 3);
 *     NDArray* output;
 *     if (model.Infer(&input, &output) != ModelError::kSuccess) {
 *       return 1;
 *     }
 *
 *     printf("Result: %f\n", output->DataAs<float>()[0]);
 *     return 0;
 *   }
 */
#ifndef TVM_RUNTIME_TI_DSP_CPP_MODEL_H_
#define TVM_RUNTIME_TI_DSP_CPP_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <dlpack/dlpack.h>

/* Forward declarations for C types */
struct TVMFFIAny;

namespace tvm {
namespace dsp {

// ============================================================
// NDArray - Tensor descriptor with constructors
// ============================================================

/*!
 * \brief N-dimensional array descriptor
 *
 * Simple POD-like structure with constructors for safe initialization.
 * Memory layout is compatible with TVMDSPNDArray.
 *
 * Memory ownership:
 * - Caller owns data and shape pointers
 * - NDArray is just a descriptor, not an owner
 */
struct NDArray {
  /*! \brief Pointer to tensor data (caller owns) */
  void* data;

  /*! \brief Pointer to shape array (caller owns) */
  int64_t* shape;

  /*! \brief Number of dimensions */
  int32_t ndim;

  /*! \brief Data type descriptor */
  DLDataType dtype;

  /*! \brief Reference counter (auto-set to 1) */
  int32_t ref_counter;

  // ----------------------------------------------------------
  // Constructors
  // ----------------------------------------------------------

  /*! \brief Default constructor - zero initialization */
  NDArray()
      : data(nullptr),
        shape(nullptr),
        ndim(0),
        dtype{0, 0, 0},
        ref_counter(1) {}

  /*!
   * \brief Main constructor
   * \param data Pointer to tensor data
   * \param shape Pointer to shape array
   * \param ndim Number of dimensions
   * \param dtype Data type descriptor
   */
  NDArray(void* data, int64_t* shape, int32_t ndim, DLDataType dtype)
      : data(data),
        shape(shape),
        ndim(ndim),
        dtype(dtype),
        ref_counter(1) {}

  // ----------------------------------------------------------
  // Factory Methods
  // ----------------------------------------------------------

  /*! \brief Create Float32 NDArray */
  static NDArray Float32(float* data, int64_t* shape, int32_t ndim) {
    return NDArray(static_cast<void*>(data), shape, ndim, {kDLFloat, 32, 1});
  }

  /*! \brief Create Float16 NDArray */
  static NDArray Float16(void* data, int64_t* shape, int32_t ndim) {
    return NDArray(data, shape, ndim, {kDLFloat, 16, 1});
  }

  /*! \brief Create Int32 NDArray */
  static NDArray Int32(int32_t* data, int64_t* shape, int32_t ndim) {
    return NDArray(static_cast<void*>(data), shape, ndim, {kDLInt, 32, 1});
  }

  /*! \brief Create Int8 NDArray */
  static NDArray Int8(int8_t* data, int64_t* shape, int32_t ndim) {
    return NDArray(static_cast<void*>(data), shape, ndim, {kDLInt, 8, 1});
  }

  /*! \brief Create UInt8 NDArray */
  static NDArray UInt8(uint8_t* data, int64_t* shape, int32_t ndim) {
    return NDArray(static_cast<void*>(data), shape, ndim, {kDLUInt, 8, 1});
  }

  // ----------------------------------------------------------
  // Utility Methods
  // ----------------------------------------------------------

  /*! \brief Check if array is valid */
  bool IsValid() const {
    return data != nullptr && shape != nullptr && ndim > 0;
  }

  /*! \brief Get total number of elements */
  int64_t NumElements() const {
    if (!shape || ndim <= 0) return 0;
    int64_t total = 1;
    for (int32_t i = 0; i < ndim; i++) {
      total *= shape[i];
    }
    return total;
  }

  /*! \brief Get size in bytes */
  size_t SizeBytes() const {
    return static_cast<size_t>(NumElements()) * (dtype.bits / 8);
  }

  /*! \brief Get typed data pointer */
  template <typename T>
  T* DataAs() const {
    return static_cast<T*>(data);
  }
};

// ============================================================
// MemoryPool - Memory pool identifier
// ============================================================

/*! \brief Memory pool identifier */
enum class MemoryPool {
  kFast = 0,  /*!< L2 SRAM - fast, limited (64KB on C66x) */
  kMain = 1   /*!< L3/DDR - slower, larger */
};

// ============================================================
// ModelError - Error codes for Model operations
// ============================================================

/*! \brief Error codes returned by Model methods */
enum class ModelError {
  kSuccess = 0,           /*!< Operation succeeded */
  kPlatformInitFailed,    /*!< Platform initialization failed */
  kConstantsParseFailed,  /*!< Constants parsing failed */
  kNullInput,             /*!< Null input pointer */
  kNotLoaded,             /*!< Model not loaded */
  kInferenceFailed,       /*!< Inference execution failed */
  kInvalidOutputType      /*!< Output type not supported */
};

// ============================================================
// MemoryStats - Memory statistics
// ============================================================

/*! \brief Memory statistics for a pool */
struct MemoryStats {
  size_t total_size;     /*!< Total pool size in bytes */
  size_t used_size;      /*!< Currently used bytes */
  size_t peak_used;      /*!< Peak usage during lifetime */
  uint32_t alloc_count;  /*!< Number of allocations */
  uint32_t free_count;   /*!< Number of frees */
};

// ============================================================
// Model - RAII model class
// ============================================================

/*!
 * \brief RAII-based model class for TVM DSP inference
 *
 * Manages the complete lifecycle:
 * - Platform initialization
 * - Constants parsing
 * - Inference execution
 * - Automatic cleanup
 *
 * Thread Safety: NOT thread-safe (single inference at a time)
 */
class Model {
 public:
  // ----------------------------------------------------------
  // Loading
  // ----------------------------------------------------------

  /*!
   * \brief Load a model and initialize the runtime
   *
   * \param weights_data Pointer to weights.bin data (nullptr for embedded)
   * \param weights_size Size of weights data in bytes
   * \return ModelError::kSuccess on success, error code on failure
   */
  ModelError Load(const void* weights_data = nullptr, size_t weights_size = 0);

  // ----------------------------------------------------------
  // Inference
  // ----------------------------------------------------------

  /*!
   * \brief Run inference on input tensors (single output)
   *
   * \param inputs Pointer to array of input NDArrays (caller retains ownership)
   * \param num_inputs Number of input tensors
   * \param output Output pointer - set to output NDArray on success
   * \return ModelError::kSuccess on success, error code on failure
   *
   * For multi-output models, this returns only the first output.
   * Use InferMulti() to get all outputs.
   * Output is valid until next Infer() call or destructor.
   */
  ModelError Infer(NDArray* inputs, int num_inputs, NDArray** output);

  /*! \brief Single-input convenience overload */
  ModelError Infer(NDArray* input, NDArray** output) {
    return Infer(input, 1, output);
  }

  /*!
   * \brief Run inference on input tensors (multi-output)
   *
   * \param inputs Pointer to array of input NDArrays (caller retains ownership)
   * \param num_inputs Number of input tensors
   * \param outputs Array of output NDArray pointers (max 8)
   * \param num_outputs Set to number of outputs on success
   * \return ModelError::kSuccess on success, error code on failure
   *
   * Outputs are valid until next Infer/InferMulti() call or destructor.
   */
  ModelError InferMulti(NDArray* inputs, int num_inputs, NDArray** outputs, int* num_outputs);

  /*! \brief Single-input convenience overload */
  ModelError InferMulti(NDArray* input, NDArray** outputs, int* num_outputs) {
    return InferMulti(input, 1, outputs, num_outputs);
  }

  /*!
   * \brief Get number of outputs from last inference
   * \return Number of output tensors (1 for single output, N for multi-output)
   */
  int OutputCount() const { return output_count_; }

  // ----------------------------------------------------------
  // Diagnostics
  // ----------------------------------------------------------

  /*! \brief Get cycle count from last inference */
  uint64_t LastInferenceCycles() const { return last_cycles_; }

  /*! \brief Get memory statistics for a pool */
  MemoryStats GetMemoryStats(MemoryPool pool) const;

  /*! \brief Get number of constants loaded */
  int ConstantCount() const { return const_count_; }

  /*! \brief Check if model is loaded and ready */
  bool IsLoaded() const { return initialized_; }

  // ----------------------------------------------------------
  // Lifecycle
  // ----------------------------------------------------------

  /*! \brief Default constructor - creates unloaded model */
  Model();

  /*! \brief Destructor - handles all cleanup */
  ~Model();

  /*! \brief Move constructor */
  Model(Model&& other) noexcept;

  /*! \brief Move assignment */
  Model& operator=(Model&& other) noexcept;

  /* Disable copy */
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

 private:
  /*! \brief Maximum inputs supported */
  static constexpr int kMaxInputs = 8;
  /*! \brief Maximum outputs supported (matches TVM_DSP_ARRAY_MAX_ELEMENTS) */
  static constexpr int kMaxOutputs = 8;

  /* Cleanup helper */
  void Cleanup();

  /* Internal inference implementation */
  ModelError InferInternal(NDArray* inputs, int num_inputs, struct TVMFFIAny* output_any);

  TVMFFIAny* constants_;
  int const_count_;
  uint64_t last_cycles_;
  bool initialized_;
  bool platform_initialized_;

  /* Multi-output support */
  int output_count_;
  NDArray output_views_[kMaxOutputs];
};

}  // namespace dsp
}  // namespace tvm

#endif  // TVM_RUNTIME_TI_DSP_CPP_MODEL_H_
