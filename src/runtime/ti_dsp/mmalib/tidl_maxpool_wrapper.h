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

/**
 * @file tidl_maxpool_wrapper.h
 * @brief C-callable wrapper around TIDL's vectorized C7x spatial max pool kernel.
 *
 * This header does NOT include any TIDL headers so it can be safely included
 * in the generated lib0/lib1.c (which compiles for host, c7x, and c7x_host).
 * The implementation (tidl_maxpool_wrapper.cpp) is firmware-only and links
 * against the TIDL algo libraries at link time.
 */

#ifndef TVM_TIDL_MAXPOOL_WRAPPER_H_
#define TVM_TIDL_MAXPOOL_WRAPPER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Spatial max-pooling via TIDL's vectorized C7x kernel.
 *
 * Backed by TIDL_spatialMaxPool_ixX_oxX_init/exec which uses two Streaming
 * Engines and the 3-row vertical-max-plus-horizontal-shift trick to process
 * the pool in a single pass.  Expected ~30-60x faster than the plain-C
 * c7x_int8_max_pool fallback.
 *
 * The handle is cached after the first call for the given (C, H_in, W_in,
 * kH, kW, sH, sW, pH, pW) configuration; subsequent calls with identical
 * params skip reinit.  The cache holds one entry; a config change triggers
 * reinit.
 *
 * @param in           Input  [N, C, H_in, W_in], int8, NCHW
 * @param out          Output [N, C, H_out, W_out], int8, NCHW
 * @param N            Batch size (normally 1 for inference)
 * @param C            Number of channels
 * @param H_in, W_in   Input spatial dimensions
 * @param H_out, W_out Output spatial dimensions (caller must compute)
 * @param kH, kW       Pool window height and width
 * @param sH, sW       Stride height and width
 * @param pH, pW       Symmetric padding height and width
 * @return 0 on success, non-zero on TIDL error
 */
int32_t c7x_int8_max_pool_tidl(
    const void* in, void* out,
    int32_t N, int32_t C,
    int32_t H_in, int32_t W_in,
    int32_t H_out, int32_t W_out,
    int32_t kH, int32_t kW,
    int32_t sH, int32_t sW,
    int32_t pH, int32_t pW);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_TIDL_MAXPOOL_WRAPPER_H_ */
