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
 * @file c7x_pool_relu_wrappers.cpp
 * @brief C7x-native int8 max-pooling, relu, clamp, and requantize kernels.
 *
 * c7x_int8_requantize_clamp, c7x_int8_relu, and c7x_int8_clamp all have SE
 * + integer fixed-point vectorized paths gated on #ifdef __C7524__: __int8
 * is an 8×int32 = 256-bit container specific to this variant; wider-vector
 * C7x parts would silently produce wrong results without the guard.  The
 * scalar fallback in the #else branch is the safe path for non-C7524
 * targets.
 *
 * max_pool is unchanged from the .c file.
 */

#include "c7x_pool_relu_wrappers.h"

#include <stdint.h>

/* Unconditional include (not gated on __C7524__): on the c7x_host g++
 * toolchain, __C7524__ is defined *by* <c7x.h> itself, not predefined by
 * the compiler — gating the include on the macro it defines is a
 * chicken-and-egg check that always evaluates false, silently disabling
 * the vectorized path on host emulation (the real c7x cross-compiler
 * predefines __C7524__ as a builtin before any header runs, so this only
 * broke host emulation, not hardware builds). See
 * c7x_avgpool_wrappers.cpp for the same fix. */
#include <c7x.h>

extern "C"
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

/* =========================================================================
 * c7x_int8_requantize_clamp — SE + integer fixed-point vectorized path
 *
 * Operation: out[i] = clamp(round(in[i] * combined_scale), clip_lo, clip_hi)
 *
 * combined_scale = d_scale / o_scale is precomputed by the compiler pass
 * (ti_fuse_qdq_tidl_relu.py) and passed as float32 at compile time.
 *
 * Vectorized implementation (C7524 only):
 *   - Q13 fixed-point: scale_q = round(combined_scale * 8192)
 *     Safe for combined_scale up to 255: max product = 127×8192×255 = 265M
 *     which fits in int32 (< INT32_MAX ≈ 2147M).
 *   - SE0 streams int8 input, sign-extends 4× to int32 (__int8 = 8×int32)
 *   - 4× unrolled loop matching tvm_int8_residual_add.cpp:156–196
 *     (single SE stream, no skip branch)
 *   - __vstore_pack_byte packs int32×8 → int8×8 in one D-unit cycle
 * ========================================================================= */

#ifdef __C7524__

static void requantize_clamp_vec(
        const int8_t* __restrict__ in,
        int8_t*       __restrict__ out,
        int32_t n, float combined_scale,
        int32_t clip_lo, int32_t clip_hi) {

    /* combined_scale = d_scale / o_scale is always positive; + 0.5f rounds
     * to nearest without needing <math.h> roundf(). */
    const int32_t SHIFT   = 13;
    const int32_t scale_q = (int32_t)(combined_scale * (float)(1 << SHIFT) + 0.5f);

    const __int8 scale_v = (__int8)scale_q;
    const __int8 lo_v    = (__int8)clip_lo;
    const __int8 hi_v    = (__int8)clip_hi;

    const int32_t nvec  = n / 8;
    const int32_t nvec4 = nvec & ~3;

    /* SE streams int8 input sign-extended to int32, same template as
     * tvm_int8_residual_add.cpp lines 141–145. */
    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = (uint32_t)(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(in), se);

    int32_t i = 0;

    /* 4× unrolled: four independent chains hide the 4–6 cycle SE latency. */
    #pragma MUST_ITERATE(1,,)
    for (; i < nvec4; i += 4) {
        __int8 vx0 = __SE0ADV(int8);
        __int8 vx1 = __SE0ADV(int8);
        __int8 vx2 = __SE0ADV(int8);
        __int8 vx3 = __SE0ADV(int8);

        __int8 a0 = (vx0 * scale_v) >> SHIFT;
        __int8 a1 = (vx1 * scale_v) >> SHIFT;
        __int8 a2 = (vx2 * scale_v) >> SHIFT;
        __int8 a3 = (vx3 * scale_v) >> SHIFT;

        a0 = __max(__min(a0, hi_v), lo_v);
        a1 = __max(__min(a1, hi_v), lo_v);
        a2 = __max(__min(a2, hi_v), lo_v);
        a3 = __max(__min(a3, hi_v), lo_v);

        __vstore_pack_byte((__char8*)(out + (i+0)*8), a0);
        __vstore_pack_byte((__char8*)(out + (i+1)*8), a1);
        __vstore_pack_byte((__char8*)(out + (i+2)*8), a2);
        __vstore_pack_byte((__char8*)(out + (i+3)*8), a3);
    }

    /* Cleanup: remaining full 8-element vectors (0–3). */
    for (; i < nvec; ++i) {
        __int8 vx = __SE0ADV(int8);
        __int8 a  = __max(__min((vx * scale_v) >> SHIFT, hi_v), lo_v);
        __vstore_pack_byte((__char8*)(out + i*8), a);
    }

    __SE0_CLOSE();

    /* Scalar tail: n % 8 remaining elements. */
    for (int32_t j = nvec * 8; j < n; ++j) {
        int32_t v = ((int32_t)in[j] * scale_q) >> SHIFT;
        out[j] = (int8_t)(v < clip_lo ? clip_lo : (v > clip_hi ? clip_hi : v));
    }
}

/* =========================================================================
 * relu_vec / clamp_vec — SE + integer vectorized paths, no rescale.
 *
 * relu/clamp are only lowered by ti_fuse_qdq_tidl_relu.py's
 * _check_relu/_check_clamp when input/output QDQ params are transparent
 * (d_zp == o_zp, and for clamp d_scale ~= o_scale too), so unlike
 * requantize_clamp there is no Q13 multiply/shift here -- just a plain
 * max (relu) or clamp (min-then-max) over the streamed int8 vector.
 * Same 4x-unrolled main loop + full-vector cleanup + scalar tail shape as
 * requantize_clamp_vec above.
 * ========================================================================= */

static void relu_vec(
        const int8_t* __restrict__ in,
        int8_t*       __restrict__ out,
        int32_t n, int32_t clip_lo) {

    const __int8 lo_v = (__int8)clip_lo;

    const int32_t nvec  = n / 8;
    const int32_t nvec4 = nvec & ~3;

    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = (uint32_t)(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(in), se);

    int32_t i = 0;

    #pragma MUST_ITERATE(1,,)
    for (; i < nvec4; i += 4) {
        __int8 vx0 = __SE0ADV(int8);
        __int8 vx1 = __SE0ADV(int8);
        __int8 vx2 = __SE0ADV(int8);
        __int8 vx3 = __SE0ADV(int8);

        __int8 a0 = __max(vx0, lo_v);
        __int8 a1 = __max(vx1, lo_v);
        __int8 a2 = __max(vx2, lo_v);
        __int8 a3 = __max(vx3, lo_v);

        __vstore_pack_byte((__char8*)(out + (i+0)*8), a0);
        __vstore_pack_byte((__char8*)(out + (i+1)*8), a1);
        __vstore_pack_byte((__char8*)(out + (i+2)*8), a2);
        __vstore_pack_byte((__char8*)(out + (i+3)*8), a3);
    }

    /* Cleanup: remaining full 8-element vectors (0-3). */
    for (; i < nvec; ++i) {
        __int8 vx = __SE0ADV(int8);
        __int8 a  = __max(vx, lo_v);
        __vstore_pack_byte((__char8*)(out + i*8), a);
    }

    __SE0_CLOSE();

    /* Scalar tail: n % 8 remaining elements. */
    for (int32_t j = nvec * 8; j < n; ++j)
        out[j] = (in[j] > clip_lo) ? in[j] : (int8_t)clip_lo;
}

static void clamp_vec(
        const int8_t* __restrict__ in,
        int8_t*       __restrict__ out,
        int32_t n, int32_t clip_lo, int32_t clip_hi) {

    const __int8 lo_v = (__int8)clip_lo;
    const __int8 hi_v = (__int8)clip_hi;

    const int32_t nvec  = n / 8;
    const int32_t nvec4 = nvec & ~3;

    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = (uint32_t)(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(in), se);

    int32_t i = 0;

    #pragma MUST_ITERATE(1,,)
    for (; i < nvec4; i += 4) {
        __int8 vx0 = __SE0ADV(int8);
        __int8 vx1 = __SE0ADV(int8);
        __int8 vx2 = __SE0ADV(int8);
        __int8 vx3 = __SE0ADV(int8);

        __int8 a0 = __max(__min(vx0, hi_v), lo_v);
        __int8 a1 = __max(__min(vx1, hi_v), lo_v);
        __int8 a2 = __max(__min(vx2, hi_v), lo_v);
        __int8 a3 = __max(__min(vx3, hi_v), lo_v);

        __vstore_pack_byte((__char8*)(out + (i+0)*8), a0);
        __vstore_pack_byte((__char8*)(out + (i+1)*8), a1);
        __vstore_pack_byte((__char8*)(out + (i+2)*8), a2);
        __vstore_pack_byte((__char8*)(out + (i+3)*8), a3);
    }

    /* Cleanup: remaining full 8-element vectors (0-3). */
    for (; i < nvec; ++i) {
        __int8 vx = __SE0ADV(int8);
        __int8 a  = __max(__min(vx, hi_v), lo_v);
        __vstore_pack_byte((__char8*)(out + i*8), a);
    }

    __SE0_CLOSE();

    /* Scalar tail: n % 8 remaining elements. */
    for (int32_t j = nvec * 8; j < n; ++j) {
        int8_t v = in[j];
        out[j] = (v < clip_lo) ? (int8_t)clip_lo : ((v > clip_hi) ? (int8_t)clip_hi : v);
    }
}

#endif  /* __C7524__ */

extern "C"
int32_t c7x_int8_requantize_clamp(
        const void* in, void* out,
        int32_t n, float combined_scale,
        int32_t clip_lo, int32_t clip_hi) {
#ifdef __C7524__
    requantize_clamp_vec(
        (const int8_t*)in, (int8_t*)out,
        n, combined_scale, clip_lo, clip_hi);
    return 0;
#else
    /* Scalar fallback for non-C7524 C7x variants. */
    const int8_t* __restrict__ p = (const int8_t*)in;
    int8_t*       __restrict__ q = (int8_t*)out;
    for (int32_t i = 0; i < n; i++) {
        float v = (float)p[i] * combined_scale;
        int32_t qi = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        q[i] = (int8_t)(qi < clip_lo ? clip_lo : (qi > clip_hi ? clip_hi : qi));
    }
    return 0;
#endif
}

extern "C"
int32_t c7x_int8_clamp(
        const void* in, void* out,
        int32_t n, int32_t clip_lo, int32_t clip_hi) {
#ifdef __C7524__
    clamp_vec((const int8_t*)in, (int8_t*)out, n, clip_lo, clip_hi);
    return 0;
#else
    /* Scalar fallback for non-C7524 C7x variants. Handles ReLU6 (lo=0,
     * hi=round(6/scale)) and any general clip. */
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;
    int8_t lo = (int8_t)clip_lo;
    int8_t hi = (int8_t)clip_hi;
    for (int32_t i = 0; i < n; i++) {
        int8_t v = p[i];
        q[i] = (v < lo) ? lo : ((v > hi) ? hi : v);
    }
    return 0;
#endif
}

extern "C"
int32_t c7x_int8_relu(
        const void* in, void* out,
        int32_t n, int32_t clip_lo) {
#ifdef __C7524__
    relu_vec((const int8_t*)in, (int8_t*)out, n, clip_lo);
    return 0;
#else
    /* Scalar fallback for non-C7524 C7x variants. */
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;
    int8_t lo = (int8_t)clip_lo;
    for (int32_t i = 0; i < n; i++)
        q[i] = (p[i] > lo) ? p[i] : lo;
    return 0;
#endif
}
