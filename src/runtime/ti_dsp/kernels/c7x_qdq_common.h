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
 * @file c7x_qdq_common.h
 * @brief Shared primitives for QDQ-glue element-wise/movement C7x kernels.
 *
 * Every function here is `static`/`static inline`: this header is included
 * by multiple kernel .cpp translation units (c7x_activation.cpp,
 * c7x_concat.cpp, and future c7x_rescale.cpp / c7x_sigmoid.cpp /
 * c7x_softmax.cpp), and each gets its own internal-linkage copy -- the same
 * convention those files already used for their own local statics, just
 * de-duplicated across files instead of copy-pasted.
 *
 * #ifdef __C7524__ guard convention: __int8/__float8/__char8/__SE_TEMPLATE_v1
 * etc. are 256-bit-container vector types specific to the C7524 core variant.
 * Any function using them must be compiled out on other targets, hence the
 * guard around the whole vector-intrinsics section below. The scalar helpers
 * (dq_f/rq_f/rq_i) have no such dependency and are always available.
 *
 * <c7x.h> is included unconditionally (not gated on __C7524__): on the
 * c7x_host g++ toolchain, __C7524__ is defined *by* <c7x.h> itself, not
 * predefined by the compiler -- gating the include on the macro it defines
 * is a chicken-and-egg check that always evaluates false, silently disabling
 * the vectorized path on host emulation (the real c7x cross-compiler
 * predefines __C7524__ as a builtin before any header runs, so gating only
 * ever broke host emulation, not hardware builds). See c7x_avgpool.cpp /
 * c7x_pool_relu.cpp / c7x_quantize.cpp for the same fix applied independently
 * before this header existed.
 *
 * MUST_ITERATE hazard: none of the vectorized loops below carry a
 * `#pragma MUST_ITERATE(1,,)` on their 4x-unrolled main loop, even though
 * that is the usual idiom for hinting a nonzero trip count to the compiler.
 * nvec4 (the 4x-unrolled trip count) can legitimately be 0 for small n --
 * asserting MUST_ITERATE(1,...) when the true count is 0 caused a real
 * firmware hang on c7x_dload hardware in earlier revisions of these kernels
 * (confirmed by deliberately restoring it during debugging), not just a
 * numerical mismatch. Do not add it back.
 */

#ifndef TVM_C7X_QDQ_COMMON_H_
#define TVM_C7X_QDQ_COMMON_H_

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <c7x.h>

/* =========================================================================
 * Scalar QDQ helpers (no vector-type dependency; always available)
 * ========================================================================= */

static inline float dq_f(int8_t x, int32_t zp, float scale) {
    return ((float)(x - zp)) * scale;
}

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline int8_t rq_f(float y, int32_t zp, float scale) {
    int32_t v = (int32_t)(y / scale + 0.5f);
    v += zp;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

static inline int8_t rq_i(int32_t x, int32_t scale_q, int32_t offset) {
    /* Q13 fixed-point requantize: ((x * scale_q) >> 13) + offset, clamped. */
    int32_t v = ((int32_t)x * scale_q >> 13) + offset;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

#ifdef __C7524__

/* =========================================================================
 * SE setup: int8 stream sign-extended to int32, VECLEN=8, PROMOTE=4X.
 * Shared config used by every int8-input vectorized kernel below.
 * ========================================================================= */

static inline __SE_TEMPLATE_v1 se_int8_signext_template(uint32_t icnt0) {
    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = icnt0;
    return se;
}

/* =========================================================================
 * Q13 fixed-point affine rescale core (from c7x_concat.cpp's
 * rescale_slot_vec): out[j] = clamp((src[j] * scale_q >> 13) + offset).
 * SE-streamed + 4x-unrolled main loop, single-vector remainder, scalar tail.
 * ========================================================================= */

static inline void rescale_i8_q13_vec(
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

    __SE_TEMPLATE_v1 se = se_int8_signext_template((uint32_t)(nvec * 8));

    __SE0_OPEN(const_cast<int8_t*>(src), se);

    int32_t i = 0;

    /* No #pragma MUST_ITERATE(1,,): see the header-level comment above. */
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

/* =========================================================================
 * Vectorized transcendentals for float-gate kernels (silu, sigmoid, softmax).
 * From c7x_activation.cpp's exp_taylor/vec_recip.
 * ========================================================================= */

/* (1<<16) << amt for amt>=0, (1<<16) >> -amt for amt<0 -- the reconstruction
 * core of exp_taylor's 2^yI step. amt must be in [-14,14]: (1<<16)<<14 ==
 * 1<<30 is the largest left shift that still fits in signed int32 (amt=15
 * gives 1<<31, which overflows to negative). Factored out so exp_taylor can
 * chain two of these safe rings instead of one -- see its comment below for
 * why. */
static inline __int8 _pow2_shift16(__int8 amt) {
    __vpred vpPos = __cmp_gt_pred(amt, (__int8)0);
    __int8  shiftL = __shift_left((__int8)(1 << 16), amt);
    __int8  shiftR = __shift_right((__int8)(1 << 16), (__int8)0 - amt);
    return __select(vpPos, shiftL, shiftR);
}

static inline __float8 exp_taylor(__float8 x) {
    const __float8 ln2           = (__float8)0.693147180559945f;
    const __float8 invln2        = (__float8)1.44269504090f;
    const __float8 oneBy6        = (__float8)0.1666667f;
    const __float8 oneBy24       = (__float8)0.0416667f;
    const __float8 one           = (__float8)1.0f;
    const __float8 half          = (__float8)0.5f;
    const __float8 zero          = (__float8)0.0f;
    const __float8 pkdOneBy65536 = (__float8)0.0000152587890625f;
    const __float8 fltMax        = (__float8)FLT_MAX;

    __float8 y  = invln2 * x;
    __int8   yI = __float_to_int(y);           /* round-to-nearest (VSPINT) */
    __float8 yf = y - __int_to_float(yI);

    __float8 r1 = yf * ln2;
    __float8 r2 = r1 * r1;
    __float8 r3 = r2 * r1;
    __float8 r4 = r2 * r2;
    __float8 twoPwF = one + r1 + r2 * half + r3 * oneBy6 + r4 * oneBy24;

    /* Reconstruct 2^yI * twoPwF via two chained _pow2_shift16 rings, each
     * clamped to [-14,14] (the widest single _pow2_shift16 call that can't
     * overflow int32), instead of one unclamped call -- doubles the exactly
     * -reconstructed range to yI in [-28,28] (x/ln2, i.e. |x| up to ~19.4)
     * with no extra risk: the second ring's amount is 0 whenever |yI|<=14,
     * making it a provable no-op there (unchanged behavior from before this
     * widening). Below the old +/-14..16 cutoff, sigmoid(x) (this function's
     * main caller, via 1/(1+exp(-x))) was hitting this clamp for |x| as
     * small as ~9.7 and collapsing to exactly 0.0/1.0 well before it's
     * mathematically indistinguishable from 0/1 in float32 (~|x|>20) --
     * turning many legitimately-distinct small sigmoid values into hard
     * ties. See c7x_int8_concat_sigmoid's kernel test / the yolo26n
     * match_frac regression this fixed for the concrete failure mode.
     * Beyond +/-28, the final saturate below is exact, not approximate:
     * float32 genuinely can't represent the difference from 0/FLT_MAX out
     * there. */
    __int8 fourteen    = (__int8)14;
    __int8 negFourteen = (__int8)(-14);
    __int8 yI_lo    = __max(__min(yI, fourteen), negFourteen);
    __int8 excess   = yI - yI_lo;
    __int8 excessLo = __max(__min(excess, fourteen), negFourteen);

    __float8 ePwX = twoPwF * __int_to_float(_pow2_shift16(yI_lo)) * pkdOneBy65536;
    ePwX = ePwX * __int_to_float(_pow2_shift16(excessLo)) * pkdOneBy65536;

    __vpred vpLo = __cmp_gt_pred((__int8)(-28), yI);
    ePwX = __select(vpLo, zero, ePwX);
    __vpred vpHi = __cmp_gt_pred(yI, (__int8)28);
    ePwX = __select(vpHi, fltMax, ePwX);

    return ePwX;
}

/* Vector reciprocal: __recip (VRCPSP) alone is only an ~8-bit-mantissa
 * seed; two Newton-Raphson iterations (x1 = x0*(2 - v*x0)) double the
 * mantissa accuracy each time, reaching full float32 precision. Plain `/`
 * on __float8 was tried first and rejected: it compiles to eight sequential
 * scalar __c7xabi_divf calls per vector op (checked via --keep_asm), not a
 * real vector instruction -- this refinement sequence is the actual
 * vectorized path. */
static inline __float8 vec_recip(__float8 v) {
    __float8 x = __recip(v);
    x = x * (((__float8)2.0f) - v * x);
    x = x * (((__float8)2.0f) - v * x);
    return x;
}

/* =========================================================================
 * Dequantize + sigmoid, float32 output, no self-multiply -- the core of
 * c7x_activation.cpp's c7x_int8_silu_f32out minus the self-gate multiply.
 * Shared with c7x_concat.cpp's c7x_int8_concat_sigmoid (bare-sigmoid concat
 * glue: dq -> concat -> sigmoid, no multiply, no trailing quantize).
 * ========================================================================= */

static inline void dequant_sigmoid_vec(
        const int8_t* __restrict__ in,
        float*        __restrict__ out,
        int32_t n,
        int32_t zx, float sx) {

    const __float8 vzx  = (__float8)((float)zx);
    const __float8 vsx  = (__float8)sx;
    const __float8 vone = (__float8)1.0f;

    const int32_t nvec  = n / 8;
    const int32_t nvec4 = nvec & ~3;

    __SE_TEMPLATE_v1 se = se_int8_signext_template((uint32_t)(nvec * 8));

    __SE0_OPEN(const_cast<int8_t*>(in), se);

    int32_t i = 0;

    /* No #pragma MUST_ITERATE(1,,): see the header-level comment above. */
    for (; i < nvec4; i += 4) {
        __float8 vf0 = __int_to_float(__SE0ADV(int8));
        __float8 vf1 = __int_to_float(__SE0ADV(int8));
        __float8 vf2 = __int_to_float(__SE0ADV(int8));
        __float8 vf3 = __int_to_float(__SE0ADV(int8));

        vf0 = (vf0 - vzx) * vsx;
        vf1 = (vf1 - vzx) * vsx;
        vf2 = (vf2 - vzx) * vsx;
        vf3 = (vf3 - vzx) * vsx;

        *(__float8*)(out + (i+0)*8) = vec_recip(exp_taylor((__float8)0.0f - vf0) + vone);
        *(__float8*)(out + (i+1)*8) = vec_recip(exp_taylor((__float8)0.0f - vf1) + vone);
        *(__float8*)(out + (i+2)*8) = vec_recip(exp_taylor((__float8)0.0f - vf2) + vone);
        *(__float8*)(out + (i+3)*8) = vec_recip(exp_taylor((__float8)0.0f - vf3) + vone);
    }

    for (; i < nvec; ++i) {
        __float8 vf = __int_to_float(__SE0ADV(int8));
        vf = (vf - vzx) * vsx;
        *(__float8*)(out + i*8) = vec_recip(exp_taylor((__float8)0.0f - vf) + vone);
    }

    __SE0_CLOSE();

    /* Scalar tail: n % 8 remaining elements. */
    for (int32_t j = nvec * 8; j < n; ++j)
        out[j] = sigmoid_f(dq_f(in[j], zx, sx));
}

#endif  /* __C7524__ */

#endif  /* TVM_C7X_QDQ_COMMON_H_ */
