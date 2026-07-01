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
 * \brief Per-tensor float32 → int8 quantize.
 *
 * out[i] = clamp(round(in[i] / scale) + zp, -128, 127)
 * where inv_scale = 1.0f / scale is precomputed by FuseInputQuantize.
 *
 * C7x vectorized path (8 float32 per cycle on C7524):
 *   VMPYSP  multiply float32×8 by inv_scale
 *   VADDSP  add zp (as float)
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
 */

#include "c7x_quantize.h"
#include <stdint.h>

#ifdef __C7524__

#include <c7x.h>

#define FLOAT_SIMD 8

extern "C"
int32_t c7x_int8_quantize(
        const void* in_ptr, void* out_ptr,
        int32_t n, float inv_scale, int32_t zp) {

    int8_t* __restrict__ out = (int8_t*)out_ptr;

    const __float8 vinv = (__float8)inv_scale;
    const __float8 vzp  = (__float8)((float)zp);
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

    __SE0_OPEN((void*)in_ptr, se);

    /* 4× unrolled: hides SE load latency (4–6 cycles) with independent chains */
    int32_t b = 0;
    #pragma MUST_ITERATE(1,,)
    for (; b < nvec4; b += 4) {
        __float8 vf0 = __SE0ADV(float8);
        __float8 vf1 = __SE0ADV(float8);
        __float8 vf2 = __SE0ADV(float8);
        __float8 vf3 = __SE0ADV(float8);
        __float8 vc0 = __max(vlo, __min(vhi, vf0 * vinv + vzp));
        __float8 vc1 = __max(vlo, __min(vhi, vf1 * vinv + vzp));
        __float8 vc2 = __max(vlo, __min(vhi, vf2 * vinv + vzp));
        __float8 vc3 = __max(vlo, __min(vhi, vf3 * vinv + vzp));
        __vstore_pack_byte((__char8*)(out + (b+0) * FLOAT_SIMD), __float_to_int(vc0));
        __vstore_pack_byte((__char8*)(out + (b+1) * FLOAT_SIMD), __float_to_int(vc1));
        __vstore_pack_byte((__char8*)(out + (b+2) * FLOAT_SIMD), __float_to_int(vc2));
        __vstore_pack_byte((__char8*)(out + (b+3) * FLOAT_SIMD), __float_to_int(vc3));
    }
    for (; b < nvec; b++) {
        __float8 vf = __SE0ADV(float8);
        __float8 vc = __max(vlo, __min(vhi, vf * vinv + vzp));
        __vstore_pack_byte((__char8*)(out + b * FLOAT_SIMD), __float_to_int(vc));
    }

    __SE0_CLOSE();

    /* Scalar tail for n % 8 remaining elements. */
    const float* in_f = (const float*)in_ptr;
    for (int32_t i = nvec * FLOAT_SIMD; i < n; i++) {
        float v = in_f[i] * inv_scale + (float)zp;
        int32_t qi = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        out[i] = (int8_t)(qi < -128 ? -128 : (qi > 127 ? 127 : qi));
    }

    return 0;
}

#else  /* scalar fallback for non-C7524 builds */

extern "C"
int32_t c7x_int8_quantize(
        const void* in_ptr, void* out_ptr,
        int32_t n, float inv_scale, int32_t zp) {

    const float* in  = (const float*)in_ptr;
    int8_t*      out = (int8_t*)out_ptr;

    for (int32_t i = 0; i < n; i++) {
        float v = in[i] * inv_scale + (float)zp;
        int32_t qi = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        out[i] = (int8_t)(qi < -128 ? -128 : (qi > 127 ? 127 : qi));
    }
    return 0;
}

#endif  /* __C7524__ */
