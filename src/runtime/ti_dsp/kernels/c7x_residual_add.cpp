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

/*!
 * \file c7x_residual_add.cpp
 * \brief Quantized residual add with requantization — C7x vectorized.
 *
 * Computes the quantized residual add emitted by FuseInt8ResidualAdd:
 *
 *   out[i] = sat_i8(((x[i]-zp_x)*M_x + (skip[i]-zp_skip)*M_skip) >> shift + zp_out)
 *
 * === Int8 vectorized path ===
 *
 * Two Streaming Engine contexts (SE0 over x[], SE1 over skip[]) each deliver
 * 8 int8 values sign-extended to 8 int32 per fetch cycle.  C7x named intrinsics
 * handle the rest:
 *
 *   - __max / __min (VMAXW / VMINW): element-wise int32 max/min — used for
 *     relu and saturation clamp.
 *   - __vstore_pack_byte (VSTWSVPACKB): extracts the low byte of each int32
 *     lane and writes them to memory in a single store cycle.  Safe after
 *     clamping to [-128,127] because only the low 8 bits carry the result.
 *   - 4× loop unrolling: hides SE load latency (4–6 cycles on C7x) by
 *     keeping four independent multiply-accumulate chains in flight.
 *
 * No #ifdef __C7000__ guard: every build of our DSP runtime — cross-compile
 * (c7x) and host emulation (c7x_host) — defines __C7000__ and provides
 * the same intrinsics.
 *
 * M_x overflow note: FuseInt8ResidualAdd chooses shift such that
 * M_x * 128 < INT32_MAX, so the int32 multiply never overflows.
 *
 * === Int16 scalar path ===
 *
 * int16 × int32 products reach 32767 × M_x, which overflows int32 when
 * M_x > 65537.  int64 accumulators are mandatory.  A vectorized int16 path
 * would need __long4 / __long8 (int64 vectors); deferred because int16
 * residual add is not on the ResNet-18 hot path.
 *
 * === Params layout (16 bytes, shared by int8 and int16 variants) ===
 *
 *   [0..3]   M_x      (int32) - multiplier for x branch
 *   [4..7]   M_skip   (int32) - multiplier for skip branch
 *   [8..11]  shift    (int32) - right-shift to return to output scale
 *   [12]     zp_x     (int8)  - zero-point for x
 *   [13]     zp_skip  (int8)  - zero-point for skip
 *   [14]     zp_out   (int8)  - zero-point for output
 *   [15]     reserved
 */

#include "c7x_residual_add.h"

#include <c7x.h>
#include <stdint.h>

/* =========================================================================
 * Shared params parsing
 *
 * Both int8 and int16 variants use the same packed layout; parse once to
 * avoid duplicating the byte-offset arithmetic in each function body.
 * ========================================================================= */

struct ResidualAddParams {
    int32_t M_x, M_skip, shift;
    int32_t zp_x, zp_skip, zp_out;
};

static inline ResidualAddParams parse_params(const void* params) {
    const int32_t* p32 = reinterpret_cast<const int32_t*>(params);
    const int8_t*  p8  = reinterpret_cast<const int8_t*>(params);
    ResidualAddParams p;
    p.M_x     = p32[0];
    p.M_skip  = p32[1];
    p.shift   = p32[2];
    p.zp_x    = static_cast<int32_t>(p8[12]);
    p.zp_skip = static_cast<int32_t>(p8[13]);
    p.zp_out  = static_cast<int32_t>(p8[14]);
    return p;
}

/* =========================================================================
 * Int8 vectorized core
 * =========================================================================
 * SE setup: ELETYPE=8BIT, VECLEN=8ELEMS, PROMOTE=4X_SIGNEXT.
 * Each __SE0ADV(int8) / __SE1ADV(int8) returns __int8 (8×int32), with each
 * int8 byte sign-extended to a full int32 lane.
 *
 * Arithmetic uses C operators (+, -, *, >>) on __int8 vectors directly;
 * the compiler maps these to VADDW / VSUBW / VMPYWW / VSHRW.
 *
 * Clamp uses __max / __min (VMAXW / VMINW) — named C7x intrinsics, not
 * std::max / std::min, which don't have overloads for vector types.
 *
 * Store: __vstore_pack_byte (VSTWSVPACKB) extracts byte[0] of each int32
 * lane and writes them contiguously.  One instruction replaces 8 scalar
 * stores / __get_vector_element calls.
 * ========================================================================= */

static void residual_add_i8_vec(
        const int8_t* __restrict__ x,
        const int8_t* __restrict__ skip,
        int8_t* __restrict__ out,
        int32_t n,
        const ResidualAddParams& p,
        int32_t has_relu) {

    /* Splat each scalar param to all 8 int32 lanes (C7x broadcast via cast). */
    const __int8 M_x_v     = (__int8)p.M_x;
    const __int8 M_skip_v  = (__int8)p.M_skip;
    const __int8 zp_x_v    = (__int8)p.zp_x;
    const __int8 zp_skip_v = (__int8)p.zp_skip;
    const __int8 zp_out_v  = (__int8)p.zp_out;
    /* Named as _v to distinguish from the scalar members used in the tail. */
    const __int8 lo_v      = (__int8)(-128);
    const __int8 hi_v      = (__int8)(127);
    const __int8 zero_v    = (__int8)(0);
    const int32_t shift    = p.shift;

    const int32_t nvec  = n / 8;          /* full 8-element vector iterations  */
    const int32_t nvec4 = nvec & ~3;      /* 4x-unrolled portion               */

    /* Both SEs read 8-bit elements, sign-extend 4× to int32, deliver 8 per
     * advance.  ICNT0 covers exactly the vectorised portion; the scalar tail
     * reads x[] and skip[] directly so no over-read occurs. */
    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = static_cast<uint32_t>(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(x),    se);
    __SE1_OPEN(const_cast<int8_t*>(skip), se);

    int32_t i = 0;

    /* 4× unrolled loop: four independent accumulation chains allow the
     * compiler to fill the 4–6 cycle SE latency with useful work, targeting
     * a software-pipeline initiation interval of 1 for the inner stages.
     *
     * No #pragma MUST_ITERATE(1,,): nvec4 can be 0 for small n -- see
     * c7x_quantize.cpp's quantize_1plane for the full investigation. */
    for (; i < nvec4; i += 4) {
        __int8 vx0 = __SE0ADV(int8);  __int8 vsk0 = __SE1ADV(int8);
        __int8 vx1 = __SE0ADV(int8);  __int8 vsk1 = __SE1ADV(int8);
        __int8 vx2 = __SE0ADV(int8);  __int8 vsk2 = __SE1ADV(int8);
        __int8 vx3 = __SE0ADV(int8);  __int8 vsk3 = __SE1ADV(int8);

        /* Dequant + scale both branches, then accumulate in int32 domain. */
        __int8 a0 = (vx0 - zp_x_v) * M_x_v + (vsk0 - zp_skip_v) * M_skip_v;
        __int8 a1 = (vx1 - zp_x_v) * M_x_v + (vsk1 - zp_skip_v) * M_skip_v;
        __int8 a2 = (vx2 - zp_x_v) * M_x_v + (vsk2 - zp_skip_v) * M_skip_v;
        __int8 a3 = (vx3 - zp_x_v) * M_x_v + (vsk3 - zp_skip_v) * M_skip_v;

        /* Right-shift to output scale, then add output zero-point. */
        a0 = (a0 >> shift) + zp_out_v;
        a1 = (a1 >> shift) + zp_out_v;
        a2 = (a2 >> shift) + zp_out_v;
        a3 = (a3 >> shift) + zp_out_v;

        /* Optional relu: clip negative lanes to 0.
         * __max (VMAXW) is a named C7x intrinsic for element-wise int32 max;
         * std::max has no overload for __int8 and must not be used here. */
        if (has_relu) {
            a0 = __max(a0, zero_v);
            a1 = __max(a1, zero_v);
            a2 = __max(a2, zero_v);
            a3 = __max(a3, zero_v);
        }

        /* Saturate to int8 range using __max/__min (VMAXW/VMINW). */
        a0 = __max(__min(a0, hi_v), lo_v);
        a1 = __max(__min(a1, hi_v), lo_v);
        a2 = __max(__min(a2, hi_v), lo_v);
        a3 = __max(__min(a3, hi_v), lo_v);

        /* Pack int32×8 → int8×8 and store: VSTWSVPACKB extracts byte[0] of
         * each int32 lane (the result fits because we clamped to [-128,127])
         * and writes them to 8 contiguous bytes in one D-unit cycle. */
        __vstore_pack_byte(reinterpret_cast<__char8*>(out + (i+0)*8), a0);
        __vstore_pack_byte(reinterpret_cast<__char8*>(out + (i+1)*8), a1);
        __vstore_pack_byte(reinterpret_cast<__char8*>(out + (i+2)*8), a2);
        __vstore_pack_byte(reinterpret_cast<__char8*>(out + (i+3)*8), a3);
    }

    /* Cleanup: remaining full 8-element vectors (0–3 of them). */
    for (; i < nvec; ++i) {
        __int8 vx  = __SE0ADV(int8);
        __int8 vsk = __SE1ADV(int8);
        __int8 a   = (vx - zp_x_v) * M_x_v + (vsk - zp_skip_v) * M_skip_v;
        a = (a >> shift) + zp_out_v;
        if (has_relu) a = __max(a, zero_v);
        a = __max(__min(a, hi_v), lo_v);
        __vstore_pack_byte(reinterpret_cast<__char8*>(out + i*8), a);
    }

    __SE0_CLOSE();
    __SE1_CLOSE();

    /* Scalar tail: the remaining n % 8 elements (0–7, no vector iteration). */
    for (int32_t j = nvec * 8; j < n; ++j) {
        int32_t acc = (static_cast<int32_t>(x[j])    - p.zp_x)    * p.M_x
                    + (static_cast<int32_t>(skip[j]) - p.zp_skip) * p.M_skip;
        int32_t r = (acc >> p.shift) + p.zp_out;
        if (has_relu && r < 0) r = 0;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        out[j] = static_cast<int8_t>(r);
    }
}

/* =========================================================================
 * Int16 scalar core
 *
 * int16 × int32 products reach 32767 × 2^31 ≈ 7×10^13 before shifting —
 * this overflows int32, so int64 accumulators are required.
 *
 * A vectorized path would use __long4 (4×int64 on C7504) accumulators;
 * deferred as the int16 residual add is not on the current hot path.
 * ========================================================================= */

static void residual_add_i16_scalar(
        const int16_t* x, const int16_t* skip,
        int16_t* out, int32_t n,
        const ResidualAddParams& p, int32_t has_relu) {
    for (int32_t i = 0; i < n; ++i) {
        int64_t acc = (static_cast<int64_t>(x[i])    - p.zp_x)    * static_cast<int64_t>(p.M_x)
                    + (static_cast<int64_t>(skip[i]) - p.zp_skip) * static_cast<int64_t>(p.M_skip);
        int32_t r = static_cast<int32_t>(acc >> p.shift) + p.zp_out;
        if (has_relu && r < 0) r = 0;
        if (r < -32768) r = -32768;
        if (r >  32767) r =  32767;
        out[i] = static_cast<int16_t>(r);
    }
}

/* =========================================================================
 * Public C API — signatures unchanged from the original .c file.
 * ========================================================================= */

extern "C"
int32_t c7x_int8_residual_add_relu(
        const void* x_ptr, const void* skip_ptr,
        const void* params, void* output_ptr,
        int32_t num_elements, int32_t has_relu) {
    ResidualAddParams p = parse_params(params);
    residual_add_i8_vec(
        reinterpret_cast<const int8_t*>(x_ptr),
        reinterpret_cast<const int8_t*>(skip_ptr),
        reinterpret_cast<int8_t*>(output_ptr),
        num_elements, p, has_relu);
    return 0;
}

extern "C"
int32_t c7x_int16_residual_add_relu(
        const void* x_ptr, const void* skip_ptr,
        const void* params, void* output_ptr,
        int32_t num_elements, int32_t has_relu) {
    ResidualAddParams p = parse_params(params);
    residual_add_i16_scalar(
        reinterpret_cast<const int16_t*>(x_ptr),
        reinterpret_cast<const int16_t*>(skip_ptr),
        reinterpret_cast<int16_t*>(output_ptr),
        num_elements, p, has_relu);
    return 0;
}
