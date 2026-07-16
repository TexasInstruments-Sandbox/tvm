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

#ifndef TVM_C7X_QUANTIZE_H_
#define TVM_C7X_QUANTIZE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Per-tensor float32 → int8 quantize.
 *
 * Implements out[i] = clamp(round(in[i] * inv_scale) + zp, -128, 127).
 * Uses round-to-nearest (VSPINT on C7524) matching torch.quantize_per_tensor.
 *
 * inv_scale = 1.0f / scale is precomputed by the FuseInputQuantize pass
 * to keep the hot loop division-free.
 *
 * @param in        Input tensor, float32, any layout (treated as flat)
 * @param out       Output tensor, int8, same number of elements as input
 * @param n         Total number of elements
 * @param inv_scale Reciprocal of quantization scale (= 1.0f / scale)
 * @param zp        Integer zero point (added after rounding)
 */
int32_t c7x_int8_quantize(
    const void* in, void* out,
    int32_t n, float inv_scale, int32_t zp);

/**
 * @brief Per-channel float32 → int8 quantize for a 3-channel (RGB) NCHW
 * input, folding a per-channel affine normalize (y = a*x + b) into the
 * quantize step itself: out = clamp(round(x*inv_scale_c + offset_c), -128, 127).
 *
 * Replaces FuseInputNormalizeQuantize's traced
 * take/expand_dims/multiply/add/concat/quantize chain (torchvision's
 * Inception3/GoogLeNet transform_input) with a single call: the affine's
 * scale/offset are folded into the quantize step's own scale/zero-point,
 * so no intermediate float32 tensor or concat is ever materialized.
 *
 * @param in      Input tensor, float32, NCHW layout, C=3
 * @param out     Output tensor, int8, same shape as input
 * @param N       Batch size
 * @param HW      H*W (elements per channel plane)
 * @param inv_scaleC/offsetC  Per-channel affine+quantize params, channel C
 */
int32_t c7x_int8_quantize_rgb(
    const void* in, void* out, int32_t N, int32_t HW,
    float inv_scale0, float offset0,
    float inv_scale1, float offset1,
    float inv_scale2, float offset2);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_C7X_QUANTIZE_H_ */
