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
 * @file tidl_avgpool_wrappers.c
 * @brief Quantized average-pooling for NCHW int8 tensors.
 *
 * Global pool uses integer accumulation (int32 sum, no intermediate float)
 * for accuracy.  Spatial pool dequantizes each element to float to avoid
 * scale-dependent rounding errors across the window.
 */

#include "tidl_avgpool_wrappers.h"

#include <stdint.h>
#include <math.h>

static inline int8_t rq(float y, int32_t zy, float sy) {
    int32_t v = (int32_t)(y / sy + 0.5f) + zy;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

int32_t tidl_int8_global_avg_pool(
        const void* in, void* out,
        int32_t N, int32_t C, int32_t H, int32_t W,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    int32_t hw = H * W;
    float inv_hw = sx / (float)hw;  /* sx / (H*W): fuse dequant+mean scale */

    for (int32_t b = 0; b < N; b++) {
        for (int32_t c = 0; c < C; c++) {
            const int8_t* plane = p + (b * C + c) * hw;
            int32_t sum = 0;
            for (int32_t i = 0; i < hw; i++)
                sum += (int32_t)plane[i] - zx;
            q_out[b * C + c] = rq((float)sum * inv_hw, zy, sy);
        }
    }
    return 0;
}

int32_t tidl_int8_avg_pool(
        const void* in, void* out,
        int32_t N, int32_t C, int32_t H_in, int32_t W_in,
        int32_t H_out, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* in_p = (const int8_t*)in;
    int8_t* out_p = (int8_t*)out;
    float inv_k = sx / (float)(kH * kW);  /* fuse dequant + 1/(kH*kW) */

    for (int32_t b = 0; b < N; b++) {
        for (int32_t c = 0; c < C; c++) {
            const int8_t* in_bc = in_p + (b * C + c) * H_in * W_in;
            int8_t*       out_bc = out_p + (b * C + c) * H_out * W_out;

            for (int32_t ph = 0; ph < H_out; ph++) {
                for (int32_t pw = 0; pw < W_out; pw++) {
                    int32_t ih_start = ph * sH - pH;
                    int32_t iw_start = pw * sW - pW;
                    int32_t sum = 0;

                    for (int32_t kh = 0; kh < kH; kh++) {
                        int32_t ih = ih_start + kh;
                        if (ih < 0 || ih >= H_in) continue;
                        for (int32_t kw = 0; kw < kW; kw++) {
                            int32_t iw = iw_start + kw;
                            if (iw < 0 || iw >= W_in) continue;
                            sum += (int32_t)in_bc[ih * W_in + iw] - zx;
                        }
                    }
                    out_bc[ph * W_out + pw] = rq((float)sum * inv_k, zy, sy);
                }
            }
        }
    }
    return 0;
}
