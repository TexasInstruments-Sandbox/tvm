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

#include <stdint.h>

#ifdef __C7000__

#include <c7x.h>

#define FLOAT_SIMD 8

static inline float hsum(__float8 v)
{
    return __get_vector_element(v, 0u) + __get_vector_element(v, 1u)
         + __get_vector_element(v, 2u) + __get_vector_element(v, 3u)
         + __get_vector_element(v, 4u) + __get_vector_element(v, 5u)
         + __get_vector_element(v, 6u) + __get_vector_element(v, 7u);
}

extern "C"
int32_t c7x_dequantize_vecmatmul(
    const void* activation_ptr,
    const void* weights_ptr,
    const void* scale_ptr,
    void* output_ptr,
    int32_t M, int32_t K, int32_t N)
{
    const float* __restrict activation = (const float*)activation_ptr;
    const int8_t* __restrict weights = (const int8_t*)weights_ptr;
    const float* __restrict scale = (const float*)scale_ptr;
    float* __restrict output = (float*)output_ptr;

    __SE_TEMPLATE_v1 se_tmpl = __gen_SE_TEMPLATE_v1();
    se_tmpl.ICNT0   = K;
    se_tmpl.ELETYPE = __SE_ELETYPE_8BIT;
    se_tmpl.VECLEN  = __SE_VECLEN_8ELEMS;
    se_tmpl.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;

    for (int32_t m = 0; m < M; m++) {
        const float* act_row = activation + m * K;

        for (int32_t n = 0; n < N; n++) {
            const int8_t* w_row = weights + n * K;

            __SE0_OPEN((void*)w_row, se_tmpl);

            __float8 acc0 = (__float8)(0.0f);
            __float8 acc1 = (__float8)(0.0f);
            __float8 acc2 = (__float8)(0.0f);
            __float8 acc3 = (__float8)(0.0f);

            const int32_t nvec = K / FLOAT_SIMD;
            const int32_t nvec4 = nvec & ~3;
            int32_t i = 0;

            /* No #pragma MUST_ITERATE(1,,): nvec4 can be 0 for small K --
             * see c7x_quantize.cpp's quantize_1plane for the full investigation. */
            for (; i < nvec4; i += 4) {
                __int8 w0 = __SE0ADV(int8);
                __int8 w1 = __SE0ADV(int8);
                __int8 w2 = __SE0ADV(int8);
                __int8 w3 = __SE0ADV(int8);
                acc0 += __int_to_float(w0) * *(__float8*)(act_row + (i+0) * FLOAT_SIMD);
                acc1 += __int_to_float(w1) * *(__float8*)(act_row + (i+1) * FLOAT_SIMD);
                acc2 += __int_to_float(w2) * *(__float8*)(act_row + (i+2) * FLOAT_SIMD);
                acc3 += __int_to_float(w3) * *(__float8*)(act_row + (i+3) * FLOAT_SIMD);
            }
            for (; i < nvec; i++) {
                acc0 += __int_to_float(__SE0ADV(int8)) * *(__float8*)(act_row + i * FLOAT_SIMD);
            }

            __SE0_CLOSE();

            __float8 total = (acc0 + acc1) + (acc2 + acc3);
            output[m * N + n] = hsum(total) * scale[n];
        }
    }
    return 0;
}

#else  /* !__C7000__ — scalar fallback for non-C7x builds */

extern "C"
int32_t c7x_dequantize_vecmatmul(
    const void* activation_ptr,
    const void* weights_ptr,
    const void* scale_ptr,
    void* output_ptr,
    int32_t M, int32_t K, int32_t N)
{
    const float* activation = (const float*)activation_ptr;
    const int8_t* weights = (const int8_t*)weights_ptr;
    const float* scale = (const float*)scale_ptr;
    float* output = (float*)output_ptr;

    for (int32_t m = 0; m < M; m++) {
        for (int32_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int32_t k = 0; k < K; k++) {
                acc += activation[m * K + k] * (float)weights[n * K + k];
            }
            output[m * N + n] = acc * scale[n];
        }
    }
    return 0;
}

#endif /* __C7000__ */
