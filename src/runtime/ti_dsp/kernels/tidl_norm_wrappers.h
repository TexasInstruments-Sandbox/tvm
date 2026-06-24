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
 * @file tidl_norm_wrappers.h
 * @brief Quantized normalization kernels (int8 input/output, float32 internals).
 *
 * Layer normalization for int8 quantized inputs.  The normalization
 * computation runs in float32 to avoid catastrophic cancellation.
 * Input is dequantized, normalized, affine-transformed, then requantized.
 */

#ifndef TVM_TIDL_NORM_WRAPPERS_H_
#define TVM_TIDL_NORM_WRAPPERS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Layer normalization on int8 input.
 *
 * For each of the `outer_size` tokens, normalizes the `norm_size` elements:
 *   x_f = (in[i] - zx) * sx
 *   mean = sum(x_f) / norm_size
 *   var  = sum((x_f - mean)^2) / norm_size
 *   x_hat = (x_f - mean) / sqrt(var + eps)
 *   y_f = weight[j] * x_hat + bias[j]
 *   out[i] = clamp(round(y_f / sy) + zy, -128, 127)
 *
 * @param in      Input  [outer_size, norm_size], int8 row-major
 * @param weight  Scale  [norm_size], float32 (gamma)
 * @param bias    Offset [norm_size], float32 (beta)
 * @param out     Output [outer_size, norm_size], int8 row-major
 * @param outer_size  Number of tokens (batch * sequence)
 * @param norm_size   Normalization dimension (hidden size)
 * @param eps         Stability constant (typically 1e-5)
 * @param zx, sx      Input quantization parameters
 * @param zy, sy      Output quantization parameters
 */
int32_t tidl_int8_layer_norm(
    const void* in, const void* weight, const void* bias, void* out,
    int32_t outer_size, int32_t norm_size,
    float eps,
    int32_t zx, float sx, int32_t zy, float sy);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_TIDL_NORM_WRAPPERS_H_ */
