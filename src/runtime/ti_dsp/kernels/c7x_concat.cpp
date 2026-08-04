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
 * @file c7x_concat.cpp
 * @brief Vectorized int8 concat kernels: channel-axis rescale, and last-axis
 *        dequantize+sigmoid.
 *
 * c7x_int8_concat_rescale: up to 4 inputs, fixed-signature 4-slot API.
 *   - Transparent slot (s_i == s_out, z_i == z_out): memcpy, ~2 cycles/elem.
 *   - Rescale slot: SE streaming + Q13 integer fixed-point on C7524, scalar
 *     fallback on other targets.  Follows the same SE+Q13 pattern as
 *     c7x_int8_requantize_clamp in c7x_pool_relu.cpp.
 *   - Slots with C_i == 0 are skipped entirely.
 *
 * c7x_int8_concat_sigmoid: same 4-slot API, but concatenating along the last
 * (flattened spatial/anchor) axis with a shared leading channel count C, and
 * ending in a sigmoid (float32 out) instead of a requantize -- the YOLO
 * multi-scale class-score glue (see ti_fuse_qdq_c7x_concat.py's
 * _make_concat_sigmoid_pattern). Slots with n_i == 0 are skipped entirely.
 *
 * The #ifdef __C7524__ guard is required: __int8/__float8 = 256-bit containers
 * specific to the C7524 variant; wider-vector parts would produce wrong results.
 */

#include "c7x_concat.h"

#include <stdint.h>
#include <string.h>

#include "c7x_qdq_common.h"

/* =========================================================================
 * Per-slot rescale helpers.  rq_i (scalar) and rescale_i8_q13_vec (SE
 * vectorized) come from c7x_qdq_common.h.
 * ========================================================================= */

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
    rescale_i8_q13_vec(src, dst, n_elem, scale_q, offset);
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

/* =========================================================================
 * c7x_int8_concat_sigmoid: last-axis concat + dequantize + sigmoid
 *
 * Branches share a leading channel count C; only the trailing width n_i
 * differs per branch (e.g. per-detection-scale anchor counts in the YOLO
 * multi-scale class-score glue: dq(reshape(x_i)) -> concat(axis=-1) ->
 * sigmoid, no multiply, no trailing quantize). Unlike c7x_int8_concat_rescale
 * (concat along the outermost non-batch axis, so branches are contiguous
 * blocks), concatenating along the last axis interleaves branches per
 * channel: output row c is [branch0 row c][branch1 row c]...
 *
 * dequant_sigmoid_vec (c7x_qdq_common.h) is c7x_int8_silu_f32out's SE +
 * exp_taylor/vec_recip core minus the self-gate multiply.
 * ========================================================================= */

static void process_branch_sigmoid(
        const int8_t* src,
        float* dst_base,
        int32_t C,
        int32_t n_elem,
        int32_t n_total,
        float s_in, int32_t z_in) {
    if (n_elem <= 0) return;

    for (int32_t c = 0; c < C; ++c) {
        const int8_t* row_src = src + (int64_t)c * n_elem;
        float* row_dst = dst_base + (int64_t)c * n_total;
#ifdef __C7524__
        dequant_sigmoid_vec(row_src, row_dst, n_elem, z_in, s_in);
#else
        for (int32_t j = 0; j < n_elem; ++j)
            row_dst[j] = sigmoid_f(dq_f(row_src[j], z_in, s_in));
#endif
    }
}

extern "C"
int32_t c7x_int8_concat_sigmoid(
        const void* in0, int32_t n0, float s0, int32_t z0,
        const void* in1, int32_t n1, float s1, int32_t z1,
        const void* in2, int32_t n2, float s2, int32_t z2,
        const void* in3, int32_t n3, float s3, int32_t z3,
        void* out, int32_t C) {
    const int32_t n_total = n0 + n1 + n2 + n3;
    float* dst = (float*)out;
    int32_t offset = 0;

    process_branch_sigmoid((const int8_t*)in0, dst + offset, C, n0, n_total, s0, z0);
    offset += n0;
    process_branch_sigmoid((const int8_t*)in1, dst + offset, C, n1, n_total, s1, z1);
    offset += n1;
    process_branch_sigmoid((const int8_t*)in2, dst + offset, C, n2, n_total, s2, z2);
    offset += n2;
    process_branch_sigmoid((const int8_t*)in3, dst + offset, C, n3, n_total, s3, z3);
    return 0;
}
