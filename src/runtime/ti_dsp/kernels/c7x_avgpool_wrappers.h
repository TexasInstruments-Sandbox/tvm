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
 * @file c7x_avgpool_wrappers.h
 * @brief Quantized average-pooling kernels for int8 tensors.
 *
 * Pure C7x kernels — neither function calls into the TIDL algo library
 * (hence the `c7x_` prefix, not `tidl_`; see c7x_int8_max_pool_tidl for a
 * kernel that actually does).
 *
 * All tensors are NCHW layout.  Quantization parameters follow the
 * same convention as tidl_activation_wrappers.h:
 *   x_float = (in[i] - zx) * sx
 *   out[i]  = clamp(round(mean_float / sy) + zy, -128, 127)
 */

#ifndef TVM_C7X_AVGPOOL_WRAPPERS_H_
#define TVM_C7X_AVGPOOL_WRAPPERS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Global average pooling: output size 1×1.
 *
 * Computes mean of the full spatial extent [H, W] for each (b, c),
 * dequantizes input using (zx, sx), and requantizes output using (zy, sy).
 *
 * @param in     Input tensor  [N, C, H, W], int8, NCHW
 * @param out    Output tensor [N, C, 1, 1], int8, NCHW
 * @param N, C, H, W  Input dimensions
 * @param zx, sx Input zero-point and scale
 * @param zy, sy Output zero-point and scale
 */
int32_t c7x_int8_global_avg_pool(
    const void* in, void* out,
    int32_t N, int32_t C, int32_t H, int32_t W,
    int32_t zx, float sx, int32_t zy, float sy);

/**
 * @brief Spatial average pooling with explicit kernel, stride, and padding.
 *
 * Handles avg_pool2d and adaptive_avg_pool2d with output_size != (1,1).
 * count_include_pad=True (always divides by kH*kW regardless of padding).
 *
 * On __C7524__, the dominant stride=1/3×3/"same" case (kH=kW=3, sH=sW=1,
 * pH=pW=1, H_in==H_out, W_in==W_out) gets a Q13 fixed-point fast path for
 * interior output pixels (away from the 1-pixel pad border, where all 9
 * window taps are valid); the 1-pixel border and any other kernel/stride
 * combination use the scalar path below.
 *
 * @param in           Input  [N, C, H_in, W_in], int8, NCHW
 * @param out          Output [N, C, H_out, W_out], int8, NCHW
 * @param N, C         Batch and channel dimensions
 * @param H_in, W_in   Input spatial dimensions
 * @param H_out, W_out Output spatial dimensions
 * @param kH, kW       Kernel (pool window) height and width
 * @param sH, sW       Stride height and width
 * @param pH, pW       Padding height and width (symmetric; applied on both sides)
 * @param zx, sx       Input quantization parameters
 * @param zy, sy       Output quantization parameters
 */
int32_t c7x_int8_avg_pool(
    const void* in, void* out,
    int32_t N, int32_t C, int32_t H_in, int32_t W_in,
    int32_t H_out, int32_t W_out,
    int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
    int32_t zx, float sx, int32_t zy, float sy);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_C7X_AVGPOOL_WRAPPERS_H_ */
