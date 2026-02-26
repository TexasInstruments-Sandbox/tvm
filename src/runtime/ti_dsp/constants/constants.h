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
 * \file constants/constants.h
 * \brief Constants loader for TVM DSP runtime
 *
 * This module provides zero-copy parsing of TVM's weights.bin format
 * for embedded DSP platforms without dynamic memory allocation.
 */

#ifndef TVM_RUNTIME_TI_DSP_CONSTANTS_CONSTANTS_H_
#define TVM_RUNTIME_TI_DSP_CONSTANTS_CONSTANTS_H_

#include "../ffi/ffi_types.h"
#include "../container/ndarray.h"
#include "../container/shape.h"
#include "../core/config.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------------------------------
 * Configuration
 *
 * Memory pools are dynamically allocated from L3 based on actual model needs.
 * These limits are only used for sanity checking.
 *---------------------------------------------------------------------------*/

/* Maximum constants per model (sanity check, from core/config.h) */
#ifndef TVM_DSP_CONST_MAX_SANITY
#define TVM_DSP_CONST_MAX_SANITY  TVM_DSP_MAX_CONSTANTS
#endif

/* Maximum dimensions for parsing (larger than container limit for format compat) */
#define TVM_DSP_PARSE_MAX_NDIM  16

/*---------------------------------------------------------------------------
 * Type Index Values
 *
 * These are imported from ffi_types.h for consistency.
 * The serialization format uses these type indices as discriminators.
 *---------------------------------------------------------------------------*/

/* Type indices are defined in ffi_types.h as TVMFFITypeIndex enum */

/* TVM NDArray magic number for validation */
#define TVM_NDARRAY_MAGIC  0xDD5E40F096B4A13FULL

/*---------------------------------------------------------------------------
 * Error Codes
 *---------------------------------------------------------------------------*/

#define TVM_DSP_CONST_SUCCESS           0
#define TVM_DSP_CONST_ERR_NULL_INPUT   -1
#define TVM_DSP_CONST_ERR_INVALID_MAGIC -2
#define TVM_DSP_CONST_ERR_TOO_MANY     -3
#define TVM_DSP_CONST_ERR_SHAPE_FULL   -4
#define TVM_DSP_CONST_ERR_STRING_FULL  -5
#define TVM_DSP_CONST_ERR_UNKNOWN_TYPE -6
#define TVM_DSP_CONST_ERR_BUFFER_END   -7
#define TVM_DSP_CONST_ERR_NDARRAY_FULL -8
#define TVM_DSP_CONST_ERR_NOT_INIT     -9
#define TVM_DSP_CONST_ERR_ALLOC_FAIL   -10

/*---------------------------------------------------------------------------
 * Container Types
 *
 * TVMDSPNDArray, TVMDSPShape are defined in container/ndarray.h and
 * container/shape.h respectively. TVMDSPString is defined below for
 * constants parsing.
 *---------------------------------------------------------------------------*/

/*!
 * \brief DSP String structure for constants
 *
 * TVM's String type - points to null-terminated string data.
 * Used for string constants in weights.bin.
 */
typedef struct {
  int32_t type_index;   /* Type index for String (kTVMFFIStr) */
  int32_t ref_counter;
  void (*deleter)(void*);
  const char* data;     /* Pointer to string data in pool */
  size_t size;          /* String length (not including null) */
} TVMDSPString;

/*---------------------------------------------------------------------------
 * Public API
 *---------------------------------------------------------------------------*/

/*!
 * \brief Initialize the constants system
 *
 * This resets all static pools and prepares for parsing.
 * Must be called before TVMDSPConstantsParse.
 */
void TVMDSPConstantsInit(void);

/*!
 * \brief Parse constants from binary weights data
 *
 * Parses TVM's weights.bin format and creates TVMFFIAny constants.
 * NDArray data pointers point directly into the input data (zero-copy).
 *
 * \param data Pointer to weights.bin data (must remain valid!)
 * \param size Size of weights.bin in bytes
 * \return Number of constants parsed (>= 0), or negative error code
 */
int TVMDSPConstantsParse(const void* data, size_t size);

/*!
 * \brief Get the parsed constants array
 *
 * Returns pointer to the static constants array and its size.
 *
 * \param count Output: number of constants (can be NULL)
 * \return Pointer to constants array, or NULL if not initialized
 */
TVMFFIAny* TVMDSPConstantsGet(int* count);

/*!
 * \brief Get a single constant by index
 *
 * \param index Constant index (0 to num_constants-1)
 * \return Pointer to constant, or NULL if out of bounds
 */
TVMFFIAny* TVMDSPConstantGetByIndex(int index);

/*!
 * \brief Get number of parsed constants
 * \return Number of constants, or 0 if not initialized
 */
int TVMDSPConstantsCount(void);

/*!
 * \brief Get error message for error code
 * \param err Error code from TVMDSPConstantsParse
 * \return Human-readable error message
 */
const char* TVMDSPConstantsErrorString(int err);

/*!
 * \brief Clean up and free all constants memory
 *
 * Frees all memory pools allocated for constants (NDArray pool, shape pool,
 * string pool, etc.) and resets the module to uninitialized state.
 *
 * After calling this function:
 * - All pointers returned by TVMDSPConstantsGet() become invalid
 * - TVMDSPConstantsCount() returns 0
 * - TVMDSPConstantsParse() can be called again to load new constants
 *
 * This function is safe to call multiple times or when not initialized.
 */
void TVMDSPConstantsCleanup(void);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_RUNTIME_TI_DSP_CONSTANTS_CONSTANTS_H_ */
