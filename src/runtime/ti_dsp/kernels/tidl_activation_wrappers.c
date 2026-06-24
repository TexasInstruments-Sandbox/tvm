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
 * @file tidl_activation_wrappers.c
 * @brief Element-wise quantized activation kernels.
 *
 * Implements the same element-wise quantized math as TIDL's *_nonLut
 * activation functions (TIDL_gelu_nonLut, TIDL_silu_nonLut, etc.).
 * Using standard C99 math avoids the complex TIDL batchNorm init/exec
 * infrastructure while producing identical results.
 *
 * On C7x, the TI compiler (cl7x) auto-vectorises these loops using the
 * TI C7x MathLib SIMD implementations of erff/expf.
 */

#include "tidl_activation_wrappers.h"

#include <math.h>
#include <stdint.h>

/* M_SQRT1_2 = 1/sqrt(2) ≈ 0.7071067811865476 */
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865476f
#endif

/* Dequantize one int8 element to float. */
static inline float dq(int8_t x, int32_t zp, float scale) {
    return ((float)(x - zp)) * scale;
}

/* Requantize float back to int8 with saturation. */
static inline int8_t rq(float y, int32_t zp, float scale) {
    int32_t v = (int32_t)(y / scale + 0.5f);
    v += zp;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

/* gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2))) */
static inline float _gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * M_SQRT1_2));
}

/* silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
static inline float _silu(float x) {
    return x / (1.0f + expf(-x));
}

/* hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1) */
static inline float _hardsigmoid(float x) {
    float v = x * (1.0f / 6.0f) + 0.5f;
    if (v <= 0.0f) return 0.0f;
    if (v >= 1.0f) return 1.0f;
    return v;
}

/* hardswish(x) = x * hardsigmoid(x) */
static inline float _hardswish(float x) {
    return x * _hardsigmoid(x);
}

int32_t tidl_int8_gelu(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq(_gelu(dq(p[i], zx, sx)), zy, sy);
    return 0;
}

int32_t tidl_int8_silu(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq(_silu(dq(p[i], zx, sx)), zy, sy);
    return 0;
}

int32_t tidl_int8_hardsigmoid(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq(_hardsigmoid(dq(p[i], zx, sx)), zy, sy);
    return 0;
}

int32_t tidl_int8_hardswish(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq(_hardswish(dq(p[i], zx, sx)), zy, sy);
    return 0;
}
