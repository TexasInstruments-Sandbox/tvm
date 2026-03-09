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
 * \file io/tensor_file.h
 * \brief Binary tensor file I/O for DSP applications
 *
 * Provides functions to read/write tensors from/to binary files.
 * Used for host testing - not available on embedded targets.
 *
 * File format:
 *   File Header (12 bytes):
 *     - magic       : uint32 = 0x54564D54 ("TVMT")
 *     - version     : uint32 = 1
 *     - num_tensors : uint32
 *
 *   Per Tensor:
 *     - ndim        : int32
 *     - shape[ndim] : int64[ndim]
 *     - dtype_code  : int32  (0=int, 1=uint, 2=float)
 *     - dtype_bits  : int32  (8, 16, 32, 64)
 *     - data_size   : int64  (bytes)
 *     - data        : uint8[data_size]
 */

#ifndef DSP_CPP_IO_TENSOR_FILE_H_
#define DSP_CPP_IO_TENSOR_FILE_H_

#include <cstdint>
#include <dlpack/dlpack.h>

/* Public API header for NDArray view */
#include "include/model.h"

/* File format constants */
#define TENSOR_FILE_MAGIC   0x54564D54  /* "TVMT" in ASCII */
#define TENSOR_FILE_VERSION 1

/* Maximum dimensions */
#define TENSOR_MAX_NDIM 8

/*!
 * \brief Owned N-dimensional array for file I/O
 *
 * This struct owns its data and shape arrays. Used by file I/O functions
 * to return tensors read from files. Convert to tvm::dsp::NDArray view
 * for inference using AsView().
 *
 * Memory ownership:
 * - data and shape are owned by this struct
 * - Must be freed using FreeTensors()
 */
struct OwnedNDArray {
  void* data;           /*!< Tensor data (owned) */
  int64_t* shape;       /*!< Shape array (owned) */
  int32_t ndim;         /*!< Number of dimensions */
  DLDataType dtype;     /*!< Data type */

  /*!
   * \brief Convert to non-owning NDArray view for inference
   * \return NDArray view that can be passed to Model::Infer()
   */
  tvm::dsp::NDArray AsView() const {
    return tvm::dsp::NDArray(data, shape, ndim, dtype);
  }

  /*!
   * \brief Get total number of elements
   */
  int64_t NumElements() const {
    if (!shape || ndim <= 0) return 0;
    int64_t total = 1;
    for (int32_t i = 0; i < ndim; i++) {
      total *= shape[i];
    }
    return total;
  }

  /*!
   * \brief Get size in bytes
   */
  size_t SizeBytes() const {
    return static_cast<size_t>(NumElements()) * (dtype.bits / 8);
  }
};

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Read tensors from binary file.
 *
 * Allocates OwnedNDArray for each tensor in the file. The caller is
 * responsible for freeing the returned array using FreeTensors().
 *
 * \param filename Path to input file
 * \param num_tensors Output: number of tensors read
 * \return Array of OwnedNDArray pointers, or NULL on error
 */
OwnedNDArray** ReadTensorsFromFile(const char* filename, int* num_tensors);

/*!
 * \brief Write tensors to binary file.
 *
 * Writes all tensors to the file in the standard format.
 *
 * \param filename Path to output file
 * \param tensors Array of OwnedNDArray pointers
 * \param num_tensors Number of tensors to write
 * \return 0 on success, -1 on error
 */
int WriteTensorsToFile(const char* filename,
                       OwnedNDArray** tensors,
                       int num_tensors);

/*!
 * \brief Write NDArray views to binary file.
 *
 * Convenience function to write tvm::dsp::NDArray views directly.
 *
 * \param filename Path to output file
 * \param tensors Array of NDArray pointers
 * \param num_tensors Number of tensors to write
 * \return 0 on success, -1 on error
 */
int WriteNDArraysToFile(const char* filename,
                        tvm::dsp::NDArray** tensors,
                        int num_tensors);

/*!
 * \brief Free tensor array returned by ReadTensorsFromFile.
 *
 * Frees data, shape, and the OwnedNDArray structs.
 *
 * \param tensors Array of OwnedNDArray pointers
 * \param num_tensors Number of tensors
 */
void FreeTensors(OwnedNDArray** tensors, int num_tensors);

#ifdef __cplusplus
}
#endif

#endif  /* DSP_CPP_IO_TENSOR_FILE_H_ */
