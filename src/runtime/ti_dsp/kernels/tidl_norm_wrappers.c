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
 * @file tidl_norm_wrappers.c
 * @brief Int8 layer normalization.
 *
 * Computes layer norm in float32 after dequantizing the input.  This avoids
 * catastrophic cancellation in the mean/variance computation that would occur
 * with fixed-point arithmetic.  Only the I/O is int8.
 */

#include "tidl_norm_wrappers.h"

#include <math.h>
#include <stdint.h>

static inline int8_t rq(float y, int32_t zy, float sy) {
    int32_t v = (int32_t)(y / sy + 0.5f) + zy;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

int32_t tidl_int8_layer_norm(
        const void* in, const void* weight, const void* bias, void* out,
        int32_t outer_size, int32_t norm_size,
        float eps,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t*  in_p  = (const int8_t*)in;
    const float*   w_p   = (const float*)weight;
    const float*   b_p   = (const float*)bias;
    int8_t*        out_p = (int8_t*)out;
    for (int32_t t = 0; t < outer_size; t++) {
        const int8_t* row_in  = in_p  + t * norm_size;
        int8_t*       row_out = out_p + t * norm_size;

        /* Step 1: dequantize and compute mean */
        float mean = 0.0f;
        for (int32_t j = 0; j < norm_size; j++)
            mean += ((float)(row_in[j] - zx)) * sx;
        mean /= (float)norm_size;

        /* Step 2: compute variance */
        float var = 0.0f;
        for (int32_t j = 0; j < norm_size; j++) {
            float diff = ((float)(row_in[j] - zx)) * sx - mean;
            var += diff * diff;
        }
        var /= (float)norm_size;

        float inv_std = 1.0f / sqrtf(var + eps);

        /* Step 3: normalize, affine-transform, requantize */
        for (int32_t j = 0; j < norm_size; j++) {
            float x_f = ((float)(row_in[j] - zx)) * sx;
            float x_hat = (x_f - mean) * inv_std;
            float y_f = w_p[j] * x_hat + b_p[j];
            row_out[j] = rq(y_f, zy, sy);
        }
    }
    return 0;
}
