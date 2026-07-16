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
 * \file c7x_quantize.cpp
 * \brief Per-tensor and per-channel float32 → int8 quantize.
 *
 * c7x_int8_quantize: out[i] = clamp(round(in[i] * inv_scale) + zp, -128, 127)
 * where inv_scale = 1.0f / scale is precomputed by FuseInputQuantize.
 *
 * c7x_int8_quantize_rgb: the same formula with a per-channel (inv_scale,
 * offset) pair instead of one shared (inv_scale, zp) — folds a per-channel
 * affine normalize into the quantize step itself. See c7x_quantize.h and
 * FuseInputNormalizeQuantize for the derivation (Step 16).
 *
 * C7x vectorized path (8 float32 per cycle on C7524):
 *   VMPYSP  multiply float32×8 by inv_scale
 *   VADDSP  add offset (zp or the folded affine term, as float)
 *   VMAXSP  clamp low  (-128.0f)
 *   VMINSP  clamp high (+127.0f)
 *   VSPINT  round-to-nearest float→int32
 *   VSTWSVPACKB  pack int32×8 → int8×8 in one store
 *
 * Uses __float8 / __int8 (256-bit on C7524) following the same pattern as
 * tvm_dequantize_vecmatmul.cpp; scalar fallback for non-C7524 builds.
 *
 * Public interface uses void* (matching TVM call_extern convention) with
 * typed casts inside, same as tvm_dequantize_vecmatmul.cpp.
 *
 * KNOWN ISSUE (found via Step 16's unit tests, unrelated to that step's
 * fold logic -- present, unmodified, in quantize_vec's arithmetic since
 * before Step 16): on real c7x_dload hardware, small per-call element
 * counts (empirically, below ~64) can return wrong (near-zero-input)
 * results from the vectorized SE path; c7x_host emulation does not
 * reproduce this. Every production caller (FuseInputQuantize,
 * FuseInputNormalizeQuantize) always quantizes a full model input tensor
 * (thousands+ of elements), so this has not been observed to affect any
 * real model. Root cause not yet understood -- the failure pattern
 * doesn't cleanly correlate with the vectorized loop structure (e.g. a
 * 63-element call fails only in the scalar tail, not the vectorized
 * part). Flagged as a follow-up investigation, not fixed here; see
 * test_input_normalize_quantize_kernel.py's module docstring.
 */

#include "c7x_quantize.h"
#include <stdint.h>

/* Unconditional include (not gated on __C7524__): on the c7x_host g++
 * toolchain, __C7524__ is defined *by* <c7x.h> itself, not predefined by
 * the compiler — gating the include on the macro it defines is a
 * chicken-and-egg check that always evaluates false, silently disabling
 * the vectorized path on host emulation (the real c7x cross-compiler
 * predefines __C7524__ as a builtin before any header runs, so this only
 * broke host emulation, not hardware builds). See
 * c7x_avgpool_wrappers.cpp / c7x_pool_relu_wrappers.cpp for the same fix. */
#include <c7x.h>

#ifdef __C7524__

#define FLOAT_SIMD 8

/* Shared by c7x_int8_quantize (offset = zp) and c7x_int8_quantize_rgb
 * (offset = per-channel affine's folded additive term, see that
 * function's doc comment) — identical formula, offset is just a float
 * either way (the per-tensor caller's zp is exactly representable). */
static void quantize_vec(
        const float* __restrict__ in, int8_t* __restrict__ out,
        int32_t n, float inv_scale, float offset) {

    const __float8 vinv = (__float8)inv_scale;
    const __float8 voff = (__float8)offset;
    const __float8 vlo  = (__float8)(-128.0f);
    const __float8 vhi  = (__float8)(127.0f);

    const int32_t nvec  = n / FLOAT_SIMD;
    const int32_t nvec4 = nvec & ~3;   /* 4×-unrolled portion */

    /* SE streams float32 from DDR with hardware prefetch, bypassing the
     * cache-miss latency that limits direct __float8 pointer loads. */
    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ICNT0   = (uint32_t)(nvec * FLOAT_SIMD);
    se.ELETYPE = __SE_ELETYPE_32BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;

    __SE0_OPEN((void*)in, se);

    /* 4× unrolled: hides SE load latency (4–6 cycles) with independent chains */
    int32_t b = 0;
    #pragma MUST_ITERATE(1,,)
    for (; b < nvec4; b += 4) {
        __float8 vf0 = __SE0ADV(float8);
        __float8 vf1 = __SE0ADV(float8);
        __float8 vf2 = __SE0ADV(float8);
        __float8 vf3 = __SE0ADV(float8);
        __float8 vc0 = __max(vlo, __min(vhi, vf0 * vinv + voff));
        __float8 vc1 = __max(vlo, __min(vhi, vf1 * vinv + voff));
        __float8 vc2 = __max(vlo, __min(vhi, vf2 * vinv + voff));
        __float8 vc3 = __max(vlo, __min(vhi, vf3 * vinv + voff));
        __vstore_pack_byte((__char8*)(out + (b+0) * FLOAT_SIMD), __float_to_int(vc0));
        __vstore_pack_byte((__char8*)(out + (b+1) * FLOAT_SIMD), __float_to_int(vc1));
        __vstore_pack_byte((__char8*)(out + (b+2) * FLOAT_SIMD), __float_to_int(vc2));
        __vstore_pack_byte((__char8*)(out + (b+3) * FLOAT_SIMD), __float_to_int(vc3));
    }
    for (; b < nvec; b++) {
        __float8 vf = __SE0ADV(float8);
        __float8 vc = __max(vlo, __min(vhi, vf * vinv + voff));
        __vstore_pack_byte((__char8*)(out + b * FLOAT_SIMD), __float_to_int(vc));
    }

    __SE0_CLOSE();

    /* Scalar tail for n % 8 remaining elements. */
    for (int32_t i = nvec * FLOAT_SIMD; i < n; i++) {
        float v = in[i] * inv_scale + offset;
        int32_t qi = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        out[i] = (int8_t)(qi < -128 ? -128 : (qi > 127 ? 127 : qi));
    }
}

extern "C"
int32_t c7x_int8_quantize(
        const void* in_ptr, void* out_ptr,
        int32_t n, float inv_scale, int32_t zp) {
    quantize_vec((const float*)in_ptr, (int8_t*)out_ptr, n, inv_scale, (float)zp);
    return 0;
}

#define quantize_1plane quantize_vec

#else  /* scalar fallback for non-C7524 builds */

static void quantize_scalar(
        const float* in, int8_t* out, int32_t n, float inv_scale, float offset) {
    for (int32_t i = 0; i < n; i++) {
        float v = in[i] * inv_scale + offset;
        int32_t qi = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        out[i] = (int8_t)(qi < -128 ? -128 : (qi > 127 ? 127 : qi));
    }
}

extern "C"
int32_t c7x_int8_quantize(
        const void* in_ptr, void* out_ptr,
        int32_t n, float inv_scale, int32_t zp) {
    quantize_scalar((const float*)in_ptr, (int8_t*)out_ptr, n, inv_scale, (float)zp);
    return 0;
}

#define quantize_1plane quantize_scalar

#endif  /* __C7524__ */

/* Single definition for both branches above: quantize_1plane is macro'd to
 * whichever single-plane helper that branch defined. */
extern "C"
int32_t c7x_int8_quantize_rgb(
        const void* in_ptr, void* out_ptr, int32_t N, int32_t HW,
        float inv_scale0, float offset0,
        float inv_scale1, float offset1,
        float inv_scale2, float offset2) {
    const float* in  = (const float*)in_ptr;
    int8_t*      out = (int8_t*)out_ptr;
    const float inv_scale[3] = {inv_scale0, inv_scale1, inv_scale2};
    const float offset[3]    = {offset0, offset1, offset2};

    for (int32_t p = 0; p < N * 3; p++) {
        int32_t c = p % 3;
        quantize_1plane(in + (int64_t)p * HW, out + (int64_t)p * HW, HW,
                         inv_scale[c], offset[c]);
    }
    return 0;
}
