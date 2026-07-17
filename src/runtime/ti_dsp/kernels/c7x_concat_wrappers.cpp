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
 * @file c7x_concat_wrappers.cpp
 * @brief Vectorized int8 channel-axis concat with per-input rescaling.
 *
 * c7x_int8_concat_rescale: up to 4 inputs, fixed-signature 4-slot API.
 *   - Transparent slot (s_i == s_out, z_i == z_out): memcpy, ~2 cycles/elem.
 *   - Rescale slot: SE streaming + Q13 integer fixed-point on C7524, scalar
 *     fallback on other targets.  Follows the same SE+Q13 pattern as
 *     c7x_int8_requantize_clamp in c7x_pool_relu_wrappers.cpp.
 *   - Slots with C_i == 0 are skipped entirely.
 *
 * The #ifdef __C7524__ guard is required: __int8/__float8 = 256-bit containers
 * specific to the C7524 variant; wider-vector parts would produce wrong results.
 */

#include "c7x_concat_wrappers.h"

#include <stdint.h>
#include <string.h>

#ifdef __C7524__
#include <c7x.h>
#endif

/* =========================================================================
 * Scalar helpers
 * ========================================================================= */

static inline int8_t rq_i(int32_t x, int32_t scale_q, int32_t offset) {
    /* Q13 fixed-point requantize: ((x * scale_q) >> 13) + offset, clamped. */
    int32_t v = ((int32_t)x * scale_q >> 13) + offset;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

/* =========================================================================
 * Per-slot rescale helpers
 * ========================================================================= */

#ifdef __C7524__

static void rescale_slot_vec(
        const int8_t* __restrict__ src,
        int8_t*       __restrict__ dst,
        int32_t n_elem,
        int32_t scale_q,
        int32_t offset) {
    const __int8 scale_v = (__int8)scale_q;
    const __int8 off_v   = (__int8)offset;
    const __int8 lo_v    = (__int8)(-128);
    const __int8 hi_v    = (__int8)(127);
    const int32_t SHIFT  = 13;

    const int32_t nvec  = n_elem / 8;
    const int32_t nvec4 = nvec & ~3;

    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = (uint32_t)(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(src), se);

    int32_t i = 0;

    /* No #pragma MUST_ITERATE(1,,): nvec4 = nvec & ~3 is exactly 0 for
     * small n_elem, making "at least 1 iteration" false -- see
     * c7x_quantize.cpp's quantize_vec for the full investigation (a
     * violated MUST_ITERATE(1,,) here caused a confirmed hardware
     * correctness bug for small inputs in that kernel). */
    for (; i < nvec4; i += 4) {
        __int8 vx0 = __SE0ADV(int8);
        __int8 vx1 = __SE0ADV(int8);
        __int8 vx2 = __SE0ADV(int8);
        __int8 vx3 = __SE0ADV(int8);

        __int8 a0 = __max(__min((vx0 * scale_v >> SHIFT) + off_v, hi_v), lo_v);
        __int8 a1 = __max(__min((vx1 * scale_v >> SHIFT) + off_v, hi_v), lo_v);
        __int8 a2 = __max(__min((vx2 * scale_v >> SHIFT) + off_v, hi_v), lo_v);
        __int8 a3 = __max(__min((vx3 * scale_v >> SHIFT) + off_v, hi_v), lo_v);

        __vstore_pack_byte((__char8*)(dst + (i+0)*8), a0);
        __vstore_pack_byte((__char8*)(dst + (i+1)*8), a1);
        __vstore_pack_byte((__char8*)(dst + (i+2)*8), a2);
        __vstore_pack_byte((__char8*)(dst + (i+3)*8), a3);
    }

    for (; i < nvec; ++i) {
        __int8 vx = __SE0ADV(int8);
        __int8 a  = __max(__min((vx * scale_v >> SHIFT) + off_v, hi_v), lo_v);
        __vstore_pack_byte((__char8*)(dst + i*8), a);
    }

    __SE0_CLOSE();

    /* Scalar tail: n_elem % 8 remaining elements. */
    for (int32_t j = nvec * 8; j < n_elem; ++j)
        dst[j] = rq_i((int32_t)src[j], scale_q, offset);
}

#endif  /* __C7524__ */

/* =========================================================================
 * Process one input slot into the output buffer
 * src       : input int8 data
 * dst       : start of this slot's channel slice in the output
 * n_elem    : C_i * HW
 * s_in, z_in: input QDQ params
 * s_out, z_out: output QDQ params
 * ========================================================================= */

static void process_slot(
        const int8_t* src,
        int8_t* dst,
        int32_t n_elem,
        float s_in, int32_t z_in,
        float s_out, int32_t z_out) {
    const int32_t SHIFT = 13;

    /* Transparent fast path: same scale and zero-point → direct copy. */
    if (s_in == s_out && z_in == z_out) {
        memcpy(dst, src, (size_t)n_elem);
        return;
    }

    /* Q13 fixed-point rescale. */
    int32_t scale_q = (int32_t)(s_in / s_out * (float)(1 << SHIFT) + 0.5f);
    int32_t offset  = z_out - (int32_t)(((int64_t)z_in * scale_q) >> SHIFT);

#ifdef __C7524__
    rescale_slot_vec(src, dst, n_elem, scale_q, offset);
#else
    for (int32_t j = 0; j < n_elem; ++j)
        dst[j] = rq_i((int32_t)src[j], scale_q, offset);
#endif
}

/* =========================================================================
 * Public API
 * ========================================================================= */

extern "C"
int32_t c7x_int8_concat_rescale(
        const void* in0, int32_t C0, float s0, int32_t z0,
        const void* in1, int32_t C1, float s1, int32_t z1,
        const void* in2, int32_t C2, float s2, int32_t z2,
        const void* in3, int32_t C3, float s3, int32_t z3,
        void* out, int32_t HW,
        float s_out, int32_t z_out) {
    int8_t* dst = (int8_t*)out;

    if (C0 > 0) {
        process_slot((const int8_t*)in0, dst, C0 * HW, s0, z0, s_out, z_out);
        dst += C0 * HW;
    }
    if (C1 > 0) {
        process_slot((const int8_t*)in1, dst, C1 * HW, s1, z1, s_out, z_out);
        dst += C1 * HW;
    }
    if (C2 > 0) {
        process_slot((const int8_t*)in2, dst, C2 * HW, s2, z2, s_out, z_out);
        dst += C2 * HW;
    }
    if (C3 > 0) {
        process_slot((const int8_t*)in3, dst, C3 * HW, s3, z3, s_out, z_out);
    }
    return 0;
}
