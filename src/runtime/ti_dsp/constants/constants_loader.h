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
 * \file constants/constants_loader.h
 * \brief DSP-compatible constants loader API
 *
 * This header provides the high-level constants loading API that handles:
 * - Weight data source initialization (embedded or filesystem)
 * - Constants parsing via the weights.bin parser
 * - C++ TVMGetConstants() for host emulation compatibility
 *
 * Usage:
 *   // Option 1: Filesystem loading (define TVM_DSP_WEIGHTS_PATH)
 *   int count = TVMDSPParseConstants();
 *
 *   // Option 2: Embedded weights (define TVM_DSP_WEIGHTS_EMBEDDED)
 *   int count = TVMDSPParseConstants();
 *
 *   // Option 3: External data
 *   TVMDSPSetWeightsData(data, size);
 *   int count = TVMDSPParseConstants();
 *
 *   // Access parsed constants
 *   TVMFFIAny* constants = TVMDSPConstantsGet(&count);
 */
#ifndef TVM_RUNTIME_TI_DSP_CONSTANTS_CONSTANTS_LOADER_H_
#define TVM_RUNTIME_TI_DSP_CONSTANTS_CONSTANTS_LOADER_H_

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Parse constants from weights data
 *
 * This is the main entry point for loading model constants. It:
 * 1. Initializes weights data source (embedded, filesystem, or external)
 * 2. Parses the weights.bin format
 * 3. Creates NDArray constants ready for use
 *
 * Call this once at startup. Subsequent calls return cached count.
 *
 * \return Number of constants parsed, or negative error code
 */
int TVMDSPParseConstants(void);

/*!
 * \brief Check if constants have been parsed
 * \return Non-zero if constants are ready for use
 */
int TVMDSPConstantsReady(void);

/*!
 * \brief Reset constants loader state
 *
 * Clears parsed constants and resets state. Useful for testing
 * or reloading different weights.
 */
void TVMDSPConstantsReset(void);

#ifdef __cplusplus
}
#endif

/*
 * C++ API (host emulation only)
 *
 * TVMGetConstants() returns std::vector<tvm::ffi::Any> for compatibility
 * with TVM-generated code that expects this signature.
 *
 * On C66x, use TVMDSPParseConstants() and TVMDSPConstantsGet() instead.
 */
#if defined(__cplusplus) && !defined(TVM_DSP_TARGET_C66X)
#include <vector>

namespace tvm {
namespace ffi {
class Any;
}  // namespace ffi
}  // namespace tvm

/*!
 * \brief Get model constants (C++ API)
 *
 * Returns vector of Any objects containing model's constant tensors.
 * This function is provided for compatibility with TVM-generated code.
 *
 * \return Vector of constants
 */
std::vector<tvm::ffi::Any> TVMGetConstants();
#endif

#endif  /* TVM_RUNTIME_TI_DSP_CONSTANTS_CONSTANTS_LOADER_H_ */
