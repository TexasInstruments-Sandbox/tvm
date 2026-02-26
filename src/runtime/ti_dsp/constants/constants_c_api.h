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
 * \file constants/constants_c_api.h
 * \brief Pure C API for constants loading
 *
 * This provides a C-only API that TVM-generated code can use to access
 * model constants without C++ dependencies.
 */

#ifndef TVM_RUNTIME_TI_DSP_CONSTANTS_C_API_H_
#define TVM_RUNTIME_TI_DSP_CONSTANTS_C_API_H_

#include "../ffi/ffi_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------------------------------
 * Weight Data Source API
 *
 * These functions provide access to the raw weights.bin data.
 * Implementation depends on build configuration (linker embedded, filesystem).
 *---------------------------------------------------------------------------*/

/*!
 * \brief Set the weights data source
 *
 * Call this before TVMDSPConstantsParse() to specify where weights come from.
 * If not called, parsing will fail with no data.
 *
 * \param data Pointer to weights.bin data (must remain valid!)
 * \param size Size of weights data in bytes
 */
void TVMDSPSetWeightsData(const void* data, size_t size);

/*!
 * \brief Get the weights data pointer
 * \return Pointer to weights data, or NULL if not set
 */
const void* TVMDSPGetWeightsData(void);

/*!
 * \brief Get the weights data size
 * \return Size of weights data in bytes, or 0 if not set
 */
size_t TVMDSPGetWeightsSize(void);

/*---------------------------------------------------------------------------
 * Constants Access API
 *
 * After parsing, use these to access individual constants.
 *---------------------------------------------------------------------------*/

/*!
 * \brief Load and parse constants
 *
 * This is a convenience function that:
 * 1. Gets weights data from TVMDSPGetWeightsData()
 * 2. Initializes the constants system
 * 3. Parses the weights.bin data
 *
 * \return Number of constants parsed (>= 0), or negative error code
 */
int TVMDSPLoadConstants(void);

/*!
 * \brief Get constant by index as TVMFFIAny pointer
 *
 * This provides direct access to the parsed constant array.
 * The returned pointer points to static storage - do not free.
 *
 * \param index Constant index (0 to num_constants - 1)
 * \return Pointer to TVMFFIAny, or NULL if index out of bounds
 */
TVMFFIAny* TVMDSPGetConstant(int index);

/*!
 * \brief Get all constants as array
 *
 * Returns pointer to the static constants array.
 *
 * \param count Output: number of constants (can be NULL)
 * \return Pointer to constants array, or NULL if not loaded
 */
TVMFFIAny* TVMDSPGetAllConstants(int* count);

/*!
 * \brief Check if constants are loaded
 * \return 1 if loaded, 0 otherwise
 */
int TVMDSPConstantsLoaded(void);

/*---------------------------------------------------------------------------
 * Linker Symbol Helpers (for embedded weights)
 *
 * These are typically defined by the linker when weights.bin is embedded.
 * They're declared here for reference.
 *---------------------------------------------------------------------------*/

#if defined(TVM_DSP_WEIGHTS_LINKER_EMBEDDED)
extern const char _binary_weights_bin_start[];
extern const char _binary_weights_bin_end[];
#endif

#ifdef __cplusplus
}
#endif

#endif  /* TVM_RUNTIME_TI_DSP_CONSTANTS_C_API_H_ */
