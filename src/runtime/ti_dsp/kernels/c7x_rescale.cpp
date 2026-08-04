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
 *
 * @file c7x_rescale.cpp
 * @brief c7x_int8_rescale (SE + Q13 vectorized, via c7x_qdq_common.h's
 * rescale_i8_q13_vec) and c7x_int8_resize2d_nearest2x (portable scalar
 * gather -- see the .h for why this one deliberately avoids C7524 permute
 * intrinsics: VDUP2B/VDUP8B et al. are documented for C7100/C7120/C7504
 * but not confirmed for C7524, and a wrong guess here would only be
 * observable on real hardware, not host emulation, since it's a "does
 * this instruction exist" risk rather than a numerical one).
 */

#include "c7x_rescale.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "c7x_qdq_common.h"

/* =========================================================================
 * c7x_int8_rescale
 * ========================================================================= */

extern "C"
int32_t c7x_int8_rescale(
        const void* in, void* out, int32_t n,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* src = (const int8_t*)in;
    int8_t* dst = (int8_t*)out;

    /* Transparent fast path: identical scale/zero-point -> direct copy. */
    if (sx == sy && zx == zy) {
        memcpy(dst, src, (size_t)n);
        return 0;
    }

    const int32_t SHIFT = 13;
    int32_t scale_q = (int32_t)(sx / sy * (float)(1 << SHIFT) + 0.5f);
    int32_t offset  = zy - (int32_t)(((int64_t)zx * scale_q) >> SHIFT);

#ifdef __C7524__
    rescale_i8_q13_vec(src, dst, n, scale_q, offset);
#else
    for (int32_t j = 0; j < n; ++j)
        dst[j] = rq_i((int32_t)src[j], scale_q, offset);
#endif
    return 0;
}

/* =========================================================================
 * c7x_int8_resize2d_nearest2x
 *
 * Row-at-a-time: width-double the input row into the first output row of
 * the pair via a scalar byte loop, then memcpy that row to produce the
 * second (identical) output row. Pure int8 data movement, no arithmetic,
 * so even the unvectorized width-doubling loop is far cheaper than the
 * scalar float32 dequantize+resize+requantize path it replaces.
 * ========================================================================= */

extern "C"
int32_t c7x_int8_resize2d_nearest2x(
        const void* in, void* out, int32_t C, int32_t H, int32_t W) {
    const int8_t* src = (const int8_t*)in;
    int8_t* dst = (int8_t*)out;
    const int32_t W2 = 2 * W;
    const int32_t H2 = 2 * H;

    for (int32_t c = 0; c < C; ++c) {
        const int8_t* in_plane  = src + (int64_t)c * H * W;
        int8_t*       out_plane = dst + (int64_t)c * H2 * W2;
        for (int32_t h = 0; h < H; ++h) {
            const int8_t* in_row   = in_plane + (int64_t)h * W;
            int8_t*       out_row0 = out_plane + (int64_t)(2 * h) * W2;
            int8_t*       out_row1 = out_row0 + W2;
            for (int32_t w = 0; w < W; ++w) {
                int8_t v = in_row[w];
                out_row0[2 * w]     = v;
                out_row0[2 * w + 1] = v;
            }
            memcpy(out_row1, out_row0, (size_t)W2);
        }
    }
    return 0;
}


/* =========================================================================
 * c7x_int8_fpn_upsample_concat[_ex]
 *
 * Branch 1: scalar SiLU (dq_f/rq_f from c7x_qdq_common.h + plain expf),
 * computed once per input pixel and replicated into its 2x2 output block
 * directly (no separate upsample pass over an intermediate buffer).
 * Branch 2: plain affine rescale, not SiLU -- see the doc comment above
 * that loop below for why.
 * One combined kernel rather than chained call_te ops -- see the .h.
 *
 * out1_presize (may be nullptr, in which case this behaves as the plain
 * c7x_int8_fpn_upsample_concat entry point below): also writes branch 1's
 * per-pixel float32 SiLU value, at the *pre-upsample* [C1,H,W] spatial
 * size, before it gets 2x2-replicated (and requantized) into out. This is
 * the exact SiLU result (not a requantized-then-dequantized approximation),
 * so a downstream consumer that needs the float value gets it losslessly.
 * Needed when branch 1's SiLU output is independently consumed elsewhere
 * in the graph too (see
 * the .h doc and ti_fuse_qdq_c7x_movement.py's pattern-2 docstring) --
 * FuseOpsByPattern promotes that shared value to an extra tuple output of
 * the matched composite, the same "is_tuple_out" situation
 * _ActivationLowerer._lower_single_input already handles for hardswish.
 * ========================================================================= */

static void fpn_upsample_concat_impl(
        const int8_t* p1, int32_t C1, int32_t H, int32_t W, int32_t z1, float s1,
        const int8_t* p2, int32_t C2, int32_t z2, float s2,
        int8_t* dst, float s_out, int32_t z_out,
        float* out1_presize) {
    const int32_t W2 = 2 * W;
    const int32_t H2 = 2 * H;
    const int32_t out_plane_hw = H2 * W2;

    /* Branch 1: SiLU(in1), upsampled 2x nearest, into out[0:C1]. */
    for (int32_t c = 0; c < C1; ++c) {
        const int8_t* in_plane  = p1 + (int64_t)c * H * W;
        int8_t*       out_plane = dst + (int64_t)c * out_plane_hw;
        float*        presize_plane = out1_presize ? out1_presize + (int64_t)c * H * W : nullptr;
        for (int32_t h = 0; h < H; ++h) {
            const int8_t* in_row   = in_plane + (int64_t)h * W;
            int8_t*       out_row0 = out_plane + (int64_t)(2 * h) * W2;
            int8_t*       out_row1 = out_row0 + W2;
            float*        presize_row = presize_plane ? presize_plane + (int64_t)h * W : nullptr;
            for (int32_t w = 0; w < W; ++w) {
                float xf = dq_f(in_row[w], z1, s1);
                float yf = xf / (1.0f + expf(-xf));
                int8_t v = rq_f(yf, z_out, s_out);
                out_row0[2 * w]     = v;
                out_row0[2 * w + 1] = v;
                /* Companion output is the exact float32 SiLU value, not the
                 * requantized int8 `v` -- so a downstream consumer gets the
                 * true result without this kernel's output-scale rounding. */
                if (presize_row) presize_row[w] = yf;
            }
            memcpy(out_row1, out_row0, (size_t)W2);
        }
    }

    /* Branch 2: plain affine rescale of in2 into out[C1:C1+C2] (already at
     * the upsampled spatial size). in2 is NOT re-activated with SiLU here:
     * on both real FPN sites this backs (yolov8n/yolo26n), branch 2's SiLU
     * output is also consumed elsewhere in the graph, so
     * FuseQDQToC7xActivation's own shared-output handling has already
     * lowered it to c7x_int8_silu there and left a plain dequantize (of
     * that already-SiLU'd, already-int8 result) feeding this concat --
     * see ti_fuse_qdq_c7x_movement.py's pattern-2 docstring. Applying SiLU
     * again here would double-activate it. */
    for (int32_t c = 0; c < C2; ++c) {
        const int8_t* in_plane  = p2 + (int64_t)c * out_plane_hw;
        int8_t*       out_plane = dst + (int64_t)(C1 + c) * out_plane_hw;
        for (int32_t j = 0; j < out_plane_hw; ++j) {
            out_plane[j] = rq_f(dq_f(in_plane[j], z2, s2), z_out, s_out);
        }
    }
}

extern "C"
int32_t c7x_int8_fpn_upsample_concat(
        const void* in1, int32_t C1, int32_t H, int32_t W, int32_t z1, float s1,
        const void* in2, int32_t C2, int32_t z2, float s2,
        void* out, float s_out, int32_t z_out) {
    fpn_upsample_concat_impl(
        (const int8_t*)in1, C1, H, W, z1, s1,
        (const int8_t*)in2, C2, z2, s2,
        (int8_t*)out, s_out, z_out,
        nullptr);
    return 0;
}

extern "C"
int32_t c7x_int8_fpn_upsample_concat_ex(
        const void* in1, int32_t C1, int32_t H, int32_t W, int32_t z1, float s1,
        const void* in2, int32_t C2, int32_t z2, float s2,
        void* out, float s_out, int32_t z_out,
        void* out1_presize) {
    fpn_upsample_concat_impl(
        (const int8_t*)in1, C1, H, W, z1, s1,
        (const int8_t*)in2, C2, z2, s2,
        (int8_t*)out, s_out, z_out,
        (float*)out1_presize);
    return 0;
}
