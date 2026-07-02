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
 * @file c7x_pool_relu_wrappers.c
 * @brief C7x-native int8 max-pooling and relu kernels.
 *
 * These are plain C implementations compiled by cl7x — NOT wrappers around
 * the TIDL library.  The TVM TIR scalar back-end generates equivalent logic
 * but without the loop-form hints that let cl7x recognize the pattern, which
 * is why these purpose-written loops run significantly faster.
 *
 * max_pool: 18.6M → still limited by boundary-check branches in the pw loop
 * that prevent full vectorization.  Interior/border split is the next step
 * (see docs/dsp/resnet-mmalib-opt.md §Step 2).
 */

#include "c7x_pool_relu_wrappers.h"

#include <stdint.h>

int32_t c7x_int8_max_pool(
        const void* in, void* out,
        int32_t N, int32_t C, int32_t H_in, int32_t W_in,
        int32_t H_out, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW,
        int32_t pH, int32_t pW) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;

    for (int32_t b = 0; b < N; b++) {
        for (int32_t c = 0; c < C; c++) {
            const int8_t* in_bc  = p + (b * C + c) * H_in  * W_in;
            int8_t*       out_bc = q + (b * C + c) * H_out * W_out;

            /* Initialise all output positions to INT8_MIN.
             * INT8_MIN represents -infinity: padding regions fill with this
             * value so they never win the max comparison. */
            for (int32_t i = 0; i < H_out * W_out; i++)
                out_bc[i] = (int8_t)-128;

            /* Accumulate max over each kernel offset (kh, kw).
             * Inner pw loop is branch-free for the valid spatial region,
             * letting cl7x vectorize across output columns. */
            for (int32_t kh = 0; kh < kH; kh++) {
                for (int32_t kw = 0; kw < kW; kw++) {
                    for (int32_t ph = 0; ph < H_out; ph++) {
                        int32_t ih = ph * sH - pH + kh;
                        if (ih < 0 || ih >= H_in) continue;
                        const int8_t* in_row  = in_bc  + ih * W_in;
                        int8_t*       out_row = out_bc + ph * W_out;

                        /* pw loop: valid window columns only.
                         * Boundary offsets are computed once per (kh, kw, ph)
                         * row, so cl7x sees a contiguous inner loop. */
                        int32_t pw_lo = 0, pw_hi = W_out;
                        if (pW > kw)
                            pw_lo = (pW - kw + sW - 1) / sW;
                        if (W_in + pW <= kw + (W_out - 1) * sW)
                            pw_hi = (W_in + pW - kw - 1) / sW + 1;

                        for (int32_t pw = pw_lo; pw < pw_hi; pw++) {
                            int32_t iw = pw * sW - pW + kw;
                            int8_t v = in_row[iw];
                            if (v > out_row[pw]) out_row[pw] = v;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int32_t c7x_int8_requantize_clamp(
        const void* in, void* out,
        int32_t n, float combined_scale,
        int32_t clip_lo, int32_t clip_hi) {
    /* Non-transparent dq→clip→q: rescale int8 input and clamp.
     * combined_scale = d_scale / o_scale (precomputed by compiler pass).
     * out[i] = clamp(round(in[i] * combined_scale), clip_lo, clip_hi) */
    const int8_t* restrict p = (const int8_t*)in;
    int8_t*       restrict q = (int8_t*)out;

    for (int32_t i = 0; i < n; i++) {
        float v = (float)p[i] * combined_scale;
        int32_t qi = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        q[i] = (int8_t)(qi < clip_lo ? clip_lo : (qi > clip_hi ? clip_hi : qi));
    }
    return 0;
}

int32_t c7x_int8_clamp(
        const void* in, void* out,
        int32_t n, int32_t clip_lo, int32_t clip_hi) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;
    int8_t lo = (int8_t)clip_lo;
    int8_t hi = (int8_t)clip_hi;

    /* Flat loop — cl7x auto-vectorizes clamp(x, lo, hi) over int8 vectors.
     * Handles ReLU6 (lo=0, hi=round(6/scale)) and any general clip. */
    for (int32_t i = 0; i < n; i++) {
        int8_t v = p[i];
        q[i] = (v < lo) ? lo : ((v > hi) ? hi : v);
    }
    return 0;
}

int32_t c7x_int8_relu(
        const void* in, void* out,
        int32_t n, int32_t clip_lo) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;
    int8_t lo = (int8_t)clip_lo;

    /* Flat loop — cl7x auto-vectorizes max(x, lo) over int8 vectors. */
    for (int32_t i = 0; i < n; i++)
        q[i] = (p[i] > lo) ? p[i] : lo;

    return 0;
}

