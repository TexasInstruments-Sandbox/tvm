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
 * @file c7x_pool_relu.h
 * @brief C7x int8 max-pooling and relu kernels for NCHW tensors.
 *
 * These are C7x-native C implementations, NOT calls into the TIDL library.
 * Both operations are quantization-transparent (max and clip are monotone),
 * so dq→op→q == op(int8) when input and output zero-points are identical.
 */

#ifndef TVM_C7X_POOL_RELU_H_
#define TVM_C7X_POOL_RELU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Spatial max-pooling on int8 NCHW data.
 *
 * Because max is monotone, the output quantization parameters equal the
 * input parameters and no scale/zp conversion is needed.  Padding fills
 * with INT8_MIN (-128), the int8 representation of -infinity.
 *
 * @param in           Input  [N, C, H_in, W_in], int8, NCHW
 * @param out          Output [N, C, H_out, W_out], int8, NCHW
 * @param N, C         Batch and channel dimensions
 * @param H_in, W_in   Input spatial dimensions
 * @param H_out, W_out Output spatial dimensions
 * @param kH, kW       Pool window height and width
 * @param sH, sW       Stride height and width
 * @param pH, pW       Padding height and width (applied symmetrically)
 */
int32_t c7x_int8_max_pool(
    const void* in, void* out,
    int32_t N, int32_t C, int32_t H_in, int32_t W_in,
    int32_t H_out, int32_t W_out,
    int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW);

/**
 * @brief Element-wise relu on int8 data: out[i] = max(in[i], clip_lo).
 *
 * relu clips at float 0.0, which corresponds to the input zero_point in the
 * int8 domain.  Pass clip_lo = d_zp (the input quantization zero-point).
 * For symmetric quantization (d_zp == 0), this is a plain max(x, 0) on int8.
 *
 * @param in      Input tensor, int8, any layout (treated as flat)
 * @param out     Output tensor, int8, same shape as input
 * @param n       Total number of elements
 * @param clip_lo int8 value corresponding to float 0.0 (= input zero_point)
 */
/**
 * @brief Rescale int8 and clamp: out[i] = clamp(round(in[i] * combined_scale), lo, hi).
 *
 * For non-transparent dq→clip→q patterns (d_scale ≠ o_scale).
 * combined_scale = d_scale / o_scale is precomputed by the compiler pass.
 * clip_lo/hi are the int8 representations of the float clip bounds in the
 * output quantization domain.  Requires d_zp == o_zp == 0.
 */
int32_t c7x_int8_requantize_clamp(
    const void* in, void* out,
    int32_t n, float combined_scale,
    int32_t clip_lo, int32_t clip_hi);

/**
 * @brief Two-sided clamp on int8: out[i] = clamp(in[i], clip_lo, clip_hi).
 *
 * Handles ReLU6 and any transparent dq→clip→q pattern.  clip_lo and clip_hi
 * are int8 representations of the float bounds: round(bound / scale) + zp.
 */
int32_t c7x_int8_clamp(
    const void* in, void* out,
    int32_t n, int32_t clip_lo, int32_t clip_hi);

int32_t c7x_int8_relu(
    const void* in, void* out,
    int32_t n, int32_t clip_lo);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_C7X_POOL_RELU_H_ */
