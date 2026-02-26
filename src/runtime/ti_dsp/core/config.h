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
 * \file core/config.h
 * \brief Centralized configuration constants for TVM DSP Runtime
 *
 * This header defines common configuration constants used across the runtime.
 * Platform-specific constants (L2/L3 sizes, etc.) are defined in platform headers.
 */

#ifndef TVM_RUNTIME_TI_DSP_CORE_CONFIG_H_
#define TVM_RUNTIME_TI_DSP_CORE_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Memory Allocation Thresholds
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Size threshold for L2 (fast) vs L3 (main) memory allocation.
 *
 * Allocations <= this threshold attempt L2 first, falling back to L3.
 * Allocations > this threshold go directly to L3.
 *
 * This balances keeping hot data in L2 while avoiding fragmentation.
 */
#ifndef TVM_DSP_L2_ALLOC_THRESHOLD
#define TVM_DSP_L2_ALLOC_THRESHOLD (32 * 1024)  /* 32KB */
#endif

/* ---------------------------------------------------------------------------
 * Container Limits
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Maximum number of dimensions for tensors.
 *
 * Used for inline shape storage in NDArray and Shape containers.
 * Matches TVM's typical limit and covers most deep learning models.
 */
#ifndef TVM_DSP_MAX_NDIM
#define TVM_DSP_MAX_NDIM 8
#endif

/* ---------------------------------------------------------------------------
 * Sanity Check Limits
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Maximum number of constants allowed (sanity check).
 *
 * Prevents loading corrupted weights files that claim unreasonable counts.
 */
#ifndef TVM_DSP_MAX_CONSTANTS
#define TVM_DSP_MAX_CONSTANTS 4096
#endif

#ifdef __cplusplus
}
#endif

#endif  /* TVM_RUNTIME_TI_DSP_CORE_CONFIG_H_ */
