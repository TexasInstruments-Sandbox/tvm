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
 * @file tidl_activation_wrappers.cpp
 * @brief Element-wise quantized activation kernels.
 *
 * tidl_int8_hardswish: SE + float vectorized on C7524 (#ifdef __C7524__);
 *   scalar float fallback for other targets.  SE eliminates DDR scalar-read
 *   latency (~100 cycles/elem → ~2–4 cycles/elem on C7524).
 *
 * tidl_int8_channel_scale_multiply: SE + Q13 integer per-channel vectorized
 *   on C7524.  Handles the SE-block broadcast pattern [1,C,1,1] × [1,C,H,W]
 *   by looping over C channels with per-channel Q13 scale derived from the
 *   excitation vector, then streaming the H×W feature map via SE0.
 *
 * gelu, silu, hardsigmoid: scalar float only (applied to tiny tensors ≤C
 *   elements from global avg pool — not on the hot path).
 *
 * The #ifdef __C7524__ guard is required: __int8/__float8 = 256-bit containers
 * specific to the C7524 variant; wider-vector C7x parts would produce wrong
 * results without the guard.
 */

#include "tidl_activation_wrappers.h"

#include <math.h>
#include <stdint.h>

#ifdef __C7524__
#include <c7x.h>
#endif

/* M_SQRT1_2 = 1/sqrt(2) ≈ 0.7071067811865476 */
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865476f
#endif

/* =========================================================================
 * Scalar helpers (used by scalar fallbacks and gelu/silu/hardsigmoid)
 * ========================================================================= */

static inline float dq_f(int8_t x, int32_t zp, float scale) {
    return ((float)(x - zp)) * scale;
}

static inline int8_t rq_f(float y, int32_t zp, float scale) {
    int32_t v = (int32_t)(y / scale + 0.5f);
    v += zp;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

static inline float _gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * (float)M_SQRT1_2));
}

static inline float _silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float _hardsigmoid(float x) {
    float v = x * (1.0f / 6.0f) + 0.5f;
    if (v <= 0.0f) return 0.0f;
    if (v >= 1.0f) return 1.0f;
    return v;
}

static inline float _hardswish(float x) {
    return x * _hardsigmoid(x);
}

/* =========================================================================
 * gelu / silu / hardsigmoid — scalar only (tiny tensors, not hot)
 * ========================================================================= */

extern "C"
int32_t tidl_int8_gelu(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq_f(_gelu(dq_f(p[i], zx, sx)), zy, sy);
    return 0;
}

extern "C"
int32_t tidl_int8_silu(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq_f(_silu(dq_f(p[i], zx, sx)), zy, sy);
    return 0;
}

extern "C"
int32_t tidl_int8_hardsigmoid(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq_f(_hardsigmoid(dq_f(p[i], zx, sx)), zy, sy);
    return 0;
}

/* =========================================================================
 * tidl_int8_hardswish — SE + float vectorized on C7524
 *
 * Operation per element:
 *   x_f  = (in[i] - zx) * sx
 *   y_f  = x_f * clamp(x_f / 6 + 0.5, 0, 1)   // hardswish
 *   out[i] = clamp(floor(y_f / sy + zy), -128, 127)
 *
 * SE streams int8 input sign-extended to int32; cast to float8 via VSPISP;
 * all arithmetic in float; requantize via __float_to_int + __vstore_pack_byte.
 * Pattern from c7x_quantize.cpp (float SE) combined with int8 SE template
 * from tvm_int8_residual_add.cpp.
 * ========================================================================= */

#ifdef __C7524__

static void hardswish_vec(
        const int8_t* __restrict__ in,
        int8_t*       __restrict__ out,
        int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {

    const __float8 vzx    = (__float8)((float)zx);
    const __float8 vsx    = (__float8)sx;
    const __float8 vinv6  = (__float8)(1.0f / 6.0f);
    const __float8 v05    = (__float8)0.5f;
    const __float8 vzero  = (__float8)0.0f;
    const __float8 vone   = (__float8)1.0f;
    const __float8 vzy    = (__float8)((float)zy);
    const __float8 vinvsy = (__float8)(1.0f / sy);
    const __float8 vlo    = (__float8)(-128.0f);
    const __float8 vhi    = (__float8)(127.0f);

    const int32_t nvec  = n / 8;
    const int32_t nvec4 = nvec & ~3;

    /* SE streams int8 sign-extended to int32 (PROMOTE=4X_SIGNEXT). */
    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = (uint32_t)(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(in), se);

    int32_t i = 0;

    /* 4× unrolled: four independent float8 chains hide the SE load latency. */
    #pragma MUST_ITERATE(1,,)
    for (; i < nvec4; i += 4) {
        __float8 vf0 = __int_to_float(__SE0ADV(int8));
        __float8 vf1 = __int_to_float(__SE0ADV(int8));
        __float8 vf2 = __int_to_float(__SE0ADV(int8));
        __float8 vf3 = __int_to_float(__SE0ADV(int8));

        vf0 = (vf0 - vzx) * vsx;
        vf1 = (vf1 - vzx) * vsx;
        vf2 = (vf2 - vzx) * vsx;
        vf3 = (vf3 - vzx) * vsx;

        vf0 = vf0 * __max(vzero, __min(vone, vf0 * vinv6 + v05));
        vf1 = vf1 * __max(vzero, __min(vone, vf1 * vinv6 + v05));
        vf2 = vf2 * __max(vzero, __min(vone, vf2 * vinv6 + v05));
        vf3 = vf3 * __max(vzero, __min(vone, vf3 * vinv6 + v05));

        vf0 = __max(vlo, __min(vhi, vf0 * vinvsy + vzy));
        vf1 = __max(vlo, __min(vhi, vf1 * vinvsy + vzy));
        vf2 = __max(vlo, __min(vhi, vf2 * vinvsy + vzy));
        vf3 = __max(vlo, __min(vhi, vf3 * vinvsy + vzy));

        __vstore_pack_byte((__char8*)(out + (i+0)*8), __float_to_int(vf0));
        __vstore_pack_byte((__char8*)(out + (i+1)*8), __float_to_int(vf1));
        __vstore_pack_byte((__char8*)(out + (i+2)*8), __float_to_int(vf2));
        __vstore_pack_byte((__char8*)(out + (i+3)*8), __float_to_int(vf3));
    }

    for (; i < nvec; ++i) {
        __float8 vf = __int_to_float(__SE0ADV(int8));
        vf = (vf - vzx) * vsx;
        vf = vf * __max(vzero, __min(vone, vf * vinv6 + v05));
        vf = __max(vlo, __min(vhi, vf * vinvsy + vzy));
        __vstore_pack_byte((__char8*)(out + i*8), __float_to_int(vf));
    }

    __SE0_CLOSE();

    /* Scalar tail: n % 8 remaining elements. */
    for (int32_t j = nvec * 8; j < n; ++j)
        out[j] = rq_f(_hardswish(dq_f(in[j], zx, sx)), zy, sy);
}

#endif  /* __C7524__ */

extern "C"
int32_t tidl_int8_hardswish(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
#ifdef __C7524__
    hardswish_vec((const int8_t*)in, (int8_t*)out, n, zx, sx, zy, sy);
    return 0;
#else
    const int8_t* p = (const int8_t*)in;
    int8_t* q_out = (int8_t*)out;
    for (int32_t i = 0; i < n; i++)
        q_out[i] = rq_f(_hardswish(dq_f(p[i], zx, sx)), zy, sy);
    return 0;
#endif
}

/* =========================================================================
 * tidl_int8_channel_scale_multiply — SE + Q13 per-channel vectorized
 *
 * Handles the SE-block broadcast multiply: excitation[1,C,1,1] ×
 * feature_map[1,C,H,W].  For each channel c, the excitation scalar is
 * converted to a Q13 combined scale, then SE streams the H×W feature map
 * elements through the vectorized requantize loop.
 *
 * Operation per element at channel c, spatial position j:
 *   exc_f  = (excitation[c] - z_exc) * s_exc
 *   feat_f = (feature_map[c*H_W + j] - z_feat) * s_feat
 *   out    = clamp(round(exc_f * feat_f / s_out) + z_out, -128, 127)
 *
 * Integer form (Q13, SHIFT=13):
 *   scale_q = round(exc_f * s_feat / s_out * 8192)
 *   offset  = z_out - ((int64_t)z_feat * scale_q >> 13)
 *   out[j]  = clamp((feature_map[c*H_W+j] * scale_q >> 13) + offset, -128,127)
 *
 * Safe range: |feat| ≤ 127, scale_q = exc_f*s_feat/s_out*8192.
 * For typical SE blocks, exc_f ∈ [0,1] and s_feat≈s_out, so scale_q ≤ 8192;
 * max product = 127 × 8192 = 1,040,384 < INT32_MAX ✓.
 * ========================================================================= */

#ifdef __C7524__

static void channel_scale_multiply_vec(
        const int8_t* excitation,
        const int8_t* feature_map,
        int8_t*       out,
        int32_t C, int32_t H_W,
        float s_exc, int32_t z_exc,
        float s_feat, int32_t z_feat,
        float s_out, int32_t z_out) {

    const int32_t SHIFT = 13;
    const __int8 lo_v   = (__int8)(-128);
    const __int8 hi_v   = (__int8)(127);

    for (int32_t c = 0; c < C; c++) {
        float scale_f = (float)(excitation[c] - z_exc) * s_exc * s_feat / s_out;
        int32_t scale_q = (int32_t)(scale_f * (float)(1 << SHIFT) + 0.5f);
        /* Absorb z_feat into a per-channel offset so the inner loop only
         * needs a single multiply-shift-add per element. */
        int32_t offset = z_out - (int32_t)(((int64_t)z_feat * scale_q) >> SHIFT);

        const __int8 scale_v = (__int8)scale_q;
        const __int8 off_v   = (__int8)offset;

        const int8_t* fm_ch  = feature_map + (int32_t)c * H_W;
        int8_t*       out_ch = out         + (int32_t)c * H_W;

        const int32_t nvec  = H_W / 8;
        const int32_t nvec4 = nvec & ~3;

        __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
        se.ELETYPE = __SE_ELETYPE_8BIT;
        se.VECLEN  = __SE_VECLEN_8ELEMS;
        se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
        se.ICNT0   = (uint32_t)(nvec * 8);

        __SE0_OPEN(const_cast<int8_t*>(fm_ch), se);

        int32_t i = 0;
        #pragma MUST_ITERATE(1,,)
        for (; i < nvec4; i += 4) {
            __int8 vx0 = __SE0ADV(int8);
            __int8 vx1 = __SE0ADV(int8);
            __int8 vx2 = __SE0ADV(int8);
            __int8 vx3 = __SE0ADV(int8);

            __int8 a0 = __max(__min((vx0 * scale_v >> SHIFT) + off_v, hi_v), lo_v);
            __int8 a1 = __max(__min((vx1 * scale_v >> SHIFT) + off_v, hi_v), lo_v);
            __int8 a2 = __max(__min((vx2 * scale_v >> SHIFT) + off_v, hi_v), lo_v);
            __int8 a3 = __max(__min((vx3 * scale_v >> SHIFT) + off_v, hi_v), lo_v);

            __vstore_pack_byte((__char8*)(out_ch + (i+0)*8), a0);
            __vstore_pack_byte((__char8*)(out_ch + (i+1)*8), a1);
            __vstore_pack_byte((__char8*)(out_ch + (i+2)*8), a2);
            __vstore_pack_byte((__char8*)(out_ch + (i+3)*8), a3);
        }

        for (; i < nvec; ++i) {
            __int8 vx = __SE0ADV(int8);
            __int8 a  = __max(__min((vx * scale_v >> SHIFT) + off_v, hi_v), lo_v);
            __vstore_pack_byte((__char8*)(out_ch + i*8), a);
        }

        __SE0_CLOSE();

        /* Scalar tail: H_W % 8 remaining elements. */
        for (int32_t j = nvec * 8; j < H_W; ++j) {
            int32_t v = ((int32_t)fm_ch[j] * scale_q >> SHIFT) + offset;
            out_ch[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }
    }
}

#endif  /* __C7524__ */

extern "C"
int32_t tidl_int8_channel_scale_multiply(
        const void* excitation, const void* feature_map, void* out,
        int32_t C, int32_t H_W,
        float s_exc,  int32_t z_exc,
        float s_feat, int32_t z_feat,
        float s_out,  int32_t z_out) {
#ifdef __C7524__
    channel_scale_multiply_vec(
        (const int8_t*)excitation, (const int8_t*)feature_map, (int8_t*)out,
        C, H_W, s_exc, z_exc, s_feat, z_feat, s_out, z_out);
    return 0;
#else
    /* Scalar fallback for non-C7524 targets. */
    const int8_t* exc = (const int8_t*)excitation;
    const int8_t* fm  = (const int8_t*)feature_map;
    int8_t* dst       = (int8_t*)out;
    for (int32_t c = 0; c < C; c++) {
        float exc_f = (float)(exc[c] - z_exc) * s_exc;
        for (int32_t j = 0; j < H_W; j++) {
            float feat_f = (float)(fm[c * H_W + j] - z_feat) * s_feat;
            dst[c * H_W + j] = rq_f(exc_f * feat_f, z_out, s_out);
        }
    }
    return 0;
#endif
}
