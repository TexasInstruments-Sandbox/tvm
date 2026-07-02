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
 * @file tidl_activation_wrappers.h
 * @brief C-callable wrappers for TIDL quantized activation functions.
 *
 * Each function applies a non-linear activation element-wise on an int8
 * tensor with quantization parameters for dequant/requant.
 *
 * Signature for all functions:
 *   in  [n]  int8  — input tensor
 *   out [n]  int8  — output tensor
 *   n        int32 — total number of elements
 *   zx       int32 — input zero-point
 *   sx       float — input scale
 *   zy       int32 — output zero-point
 *   sy       float — output scale
 *
 * The math for each element i:
 *   x_f = (in[i] - zx) * sx
 *   y_f = activation(x_f)
 *   out[i] = clamp(round(y_f / sy) + zy, -128, 127)
 *
 * Implemented using standard C math (erff, expf).  On C7x, the TI compiler
 * auto-vectorises the loop with the TI MathLib SIMD primitives.
 */

#ifndef TVM_TIDL_ACTIVATION_WRAPPERS_H_
#define TVM_TIDL_ACTIVATION_WRAPPERS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t tidl_int8_gelu(
    const void* in, void* out, int32_t n,
    int32_t zx, float sx, int32_t zy, float sy);

int32_t tidl_int8_silu(
    const void* in, void* out, int32_t n,
    int32_t zx, float sx, int32_t zy, float sy);

int32_t tidl_int8_hardsigmoid(
    const void* in, void* out, int32_t n,
    int32_t zx, float sx, int32_t zy, float sy);

int32_t tidl_int8_hardswish(
    const void* in, void* out, int32_t n,
    int32_t zx, float sx, int32_t zy, float sy);

/* SE-block broadcast multiply: excitation[C] × feature_map[C×H_W] → out[C×H_W].
 * All shapes are NCHW with the excitation having trailing [1,1] spatial dims. */
int32_t tidl_int8_channel_scale_multiply(
    const void* excitation, const void* feature_map, void* out,
    int32_t C, int32_t H_W,
    float s_exc,  int32_t z_exc,
    float s_feat, int32_t z_feat,
    float s_out,  int32_t z_out);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_TIDL_ACTIVATION_WRAPPERS_H_ */
