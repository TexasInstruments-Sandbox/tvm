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
 * \file io/weights_loader.h
 * \brief Weights loading for DSP applications
 *
 * Provides functions to load model weights from either:
 * - TVM_DSP_WEIGHTS_EMBEDDED: Linker-embedded weights
 * - TVM_DSP_WEIGHTS_PATH: Filesystem path (host emulation)
 *
 * Usage:
 *   #include "io/weights_loader.h"
 *   #include "model.h"
 *
 *   // Load weights and get pointer/size
 *   size_t size;
 *   const void* data = GetWeightsData(&size);
 *   if (!data) {
 *     return error;
 *   }
 *
 *   // Pass directly to Model::Load()
 *   tvm::dsp::Model model;
 *   model.Load(data, size);
 */

#ifndef DSP_CPP_IO_WEIGHTS_LOADER_H_
#define DSP_CPP_IO_WEIGHTS_LOADER_H_

#include <cstddef>

/*!
 * \brief Load and return weights data
 *
 * Loads weights from the configured source (embedded or filesystem)
 * and returns a pointer to the data. The data remains valid for the
 * lifetime of the application.
 *
 * \param size Output: Size of weights data in bytes
 * \return Pointer to weights data, or nullptr on error
 */
const void* GetWeightsData(size_t* size);

/*!
 * \brief Free weights buffer (if filesystem-loaded)
 *
 * Call this to release memory used by filesystem-loaded weights.
 * Has no effect for embedded weights.
 */
void FreeWeightsBuffer();

#endif  /* DSP_CPP_IO_WEIGHTS_LOADER_H_ */
