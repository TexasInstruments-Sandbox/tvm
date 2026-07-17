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
 * \file c7x_avgpool.cpp
 * \brief Quantized average-pooling for NCHW int8 tensors — C7x optimized.
 *
 * Renamed from tidl_avgpool_wrappers.c: neither kernel calls into the TIDL
 * algo library (TIDL's own TIDL_spatialAvgPool_ixX_oxX_* C7x exec path
 * assumes symmetric, zero-point-free quantization and would silently
 * mis-round our general PT2E per-tensor affine case — see
 * docs/dsp/quantized_model_optimization.md Step 12), so the `tidl_` prefix
 * was misleading; both kernels are pure C7x, hence `c7x_`.
 *
 * Global pool uses integer accumulation (int32 sum, no intermediate float)
 * for accuracy and is unchanged from the original implementation.
 *
 * Spatial pool's scalar path dequantizes each element to float to avoid
 * scale-dependent rounding errors across the window; on __C7524__, the
 * dominant stride=1/3×3/"same" shape additionally gets a Q13 fixed-point
 * interior fast path (see avg_pool_interior_fast below). Any other
 * kernel/stride combination keeps using the scalar path.
 */

#include "c7x_avgpool.h"

#include <stdint.h>
#include <math.h>
#include <string.h>

/* Unconditional include (not gated on __C7524__ like other kernels in this
 * directory): on the c7x_host g++ toolchain, __C7524__ is defined *by*
 * <c7x.h> itself, not predefined by the compiler — gating the include on
 * the macro it defines is a chicken-and-egg check that always evaluates
 * false, silently disabling the vectorized path on host emulation (the
 * real c7x cross-compiler predefines __C7524__ as a builtin before any
 * header runs, so this only breaks host emulation, not hardware builds). */
#include <c7x.h>

static inline int8_t rq(float y, int32_t zy, float sy) {
    int32_t v = (int32_t)(y / sy + 0.5f) + zy;
    if (v < -128) v = -128;
    if (v >  127) v =  127;
    return (int8_t)v;
}

extern "C"
int32_t c7x_int8_global_avg_pool(
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

/* =========================================================================
 * Scalar spatial avg pool — reference implementation, also used as:
 *   - the correctness fallback for non-C7524 targets and unsupported
 *     kernel/stride/width combinations
 *   - the border handler for the fast path below (ph_lo/ph_hi/pw_lo/pw_hi
 *     restrict which output rectangle gets (re)computed)
 * ========================================================================= */

static void avg_pool_scalar_rect(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
        int32_t zx, float inv_k, int32_t zy, float sy,
        int32_t ph_lo, int32_t ph_hi, int32_t pw_lo, int32_t pw_hi) {
    for (int32_t ph = ph_lo; ph < ph_hi; ph++) {
        int32_t ih_start = ph * sH - pH;
        for (int32_t pw = pw_lo; pw < pw_hi; pw++) {
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

#ifdef __C7524__

/* Compile-time cap on the fast-path spatial size: avg_pool_interior_fast's
 * row_scratch buffer below is fixed-size, sized to this capacity.
 * InceptionV3's largest spatial avg_pool instance is 35x35
 * (docs/dsp/quantized_model_optimization.md Step 12); 40 leaves comfortable
 * margin. Anything larger falls back to the scalar avg_pool_scalar_rect
 * path via the fastpath eligibility check in c7x_int8_avg_pool below. */
static const int32_t kFastpathMaxDim = 40;

/* =========================================================================
 * avg_pool_interior_fast — 3x3/stride=1/"same" interior pooling, SE-vectorized.
 *
 * Third attempt at vectorizing this kernel (see
 * docs/dsp/quantized_model_optimization.md Step 12 for the first two
 * discarded attempts and their root causes, and this step's own record for
 * a third: a version that computed the box filter via a separable
 * vertical-then-horizontal sum, round-tripping partial sums through
 * scratch memory between SE passes, hung on real c7x_dload hardware —
 * passed on host emulation, but that version needed 5 sequential SE
 * open/close cycles per call and one mismatched-template simultaneous
 * SE0/SE1 open, neither pattern exercised by any other kernel in this
 * codebase or documented as safe/unsafe in the SE reference material).
 *
 * This version avoids both of those by construction: ONE SE0 open per
 * call (SE1 unused), one consistent template throughout, and the 9-tap
 * sum for each output pixel accumulates entirely in a vector register —
 * never written to memory and read back via SE.
 *
 * The 9 taps for one call are addressed as a single 5D SE stream, one
 * dimension per loop level (innermost first):
 *   dim0 (ICNT0=8):            one vector's worth of contiguous columns
 *   dim1 (ICNT1=3, DIM1=1):    kw, the horizontal tap (0/1/2)
 *   dim2 (ICNT2=3, DIM2=W):    kh, the vertical tap (0/1/2)
 *   dim3 (ICNT3=numBlocks, DIM3=8):  successive 8-wide column blocks
 *   dim4 (ICNT4=rows, DIM4=W): successive output rows
 * Summing the 9 reads for one (row, block) — one full sweep of dim1 x
 * dim2 — gives sum9 for 8 output pixels at once; no manual pointer
 * arithmetic for the shifted taps, and no alignment concern since SE
 * reads (unlike raw vector-load casts) tolerate any byte offset.
 *
 * Q13 rescale + clamp + pack happens on that same register immediately
 * (matching c7x_int8_requantize_clamp's idiom), then the packed 8 bytes
 * land in row_scratch — a small, 0-based, naturally-aligned local buffer.
 * row_scratch is never re-read by SE; it's only ever memcpy'd (alignment-
 * agnostic) into out_bc's +1-column-offset interior, once per row. This
 * keeps the one vector store this function does (into row_scratch) at a
 * self-consistent 8-aligned offset, while still landing the final result
 * at out_bc's not-necessarily-8-aligned interior offset.
 *
 * Any trailing columns in a row that don't fill a full 8-wide block
 * (interior width not a multiple of 8) are computed by a plain scalar
 * loop using the same Q13 formula, writing directly into out_bc — plain
 * scalar int8 stores have no alignment requirement either way.
 *
 * Deliberately NOT applied here: reusing vertical row-sums across
 * adjacent output rows (each row's 3 taps are re-read from scratch for
 * the next row instead of reusing 2 of the 3 already read). Left as the
 * same deferred future work as Step 12's discarded ~3x tap-reuse
 * optimization.
 *
 * Only computes the interior rectangle: rows [1, H-2], columns [1, W-2].
 * The 1-pixel border is left to avg_pool_scalar_rect, same as before. H
 * and W are bounded to kFastpathMaxDim by the caller's fastpath
 * eligibility check below, since row_scratch is fixed-size.
 * ========================================================================= */

static void avg_pool_interior_fast(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t H, int32_t W,
        int32_t zx, int32_t scale_q, int32_t zy) {
    /* Q13 fixed-point: combined_scale = (sx/9)/sy.
     * |sum9 - 9*zx| <= 9*255 = 2295 (9 signed int8 taps, zero-point offset).
     * scale_q * 2295 must stay within int32: scale_q < 2^31/2295 ~ 2^19.8,
     * i.e. combined_scale < ~114 at SHIFT=13 — matches the Q13 convention
     * used throughout this codebase (c7x_int8_requantize_clamp,
     * c7x_int8_concat_rescale) and covers combined_scale's typical <2
     * range (sx, sy are both activation quant scales of similar magnitude)
     * with wide margin. */
    const int32_t SHIFT      = 13;
    const int32_t zx9        = 9 * zx;
    const int32_t rows       = H - 2;          /* interior output rows */
    const int32_t interior_w = W - 2;          /* interior output columns */
    const int32_t eleCount   = 8;               /* C7524: 8 int32 lanes/advance */
    const int32_t numBlocks  = interior_w / eleCount;
    const int32_t colRem     = interior_w - numBlocks * eleCount;

    if (numBlocks > 0) {
        __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
        se.ELETYPE = __SE_ELETYPE_8BIT;
        se.VECLEN  = __SE_VECLEN_8ELEMS;
        se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
        se.DIMFMT  = __SE_DIMFMT_5D;
        se.ICNT0   = eleCount;
        se.ICNT1   = 3; se.DIM1 = 1;                 /* kw */
        se.ICNT2   = 3; se.DIM2 = W;                  /* kh */
        se.ICNT3   = numBlocks; se.DIM3 = eleCount;   /* column blocks */
        se.ICNT4   = rows; se.DIM4 = W;                /* output rows */

        __SE0_OPEN(const_cast<int8_t*>(in_bc), se);

        const __int8 scale_v = (__int8)scale_q;
        const __int8 zx9_v   = (__int8)zx9;
        const __int8 zy_v    = (__int8)zy;
        const __int8 lo_v    = (__int8)(-128);
        const __int8 hi_v    = (__int8)(127);

        int8_t row_scratch[kFastpathMaxDim];

        for (int32_t r = 0; r < rows; r++) {
            for (int32_t b = 0; b < numBlocks; b++) {
                __int8 sum9 = __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);
                sum9 += __SE0ADV(int8);

                __int8 a = ((sum9 - zx9_v) * scale_v >> SHIFT) + zy_v;
                a = __max(__min(a, hi_v), lo_v);
                __vstore_pack_byte(
                    reinterpret_cast<__char8*>(row_scratch + b * eleCount), a);
            }
            memcpy(out_bc + (r + 1) * W + 1, row_scratch, numBlocks * eleCount);
        }

        __SE0_CLOSE();
    }

    if (colRem > 0) {
        for (int32_t r = 0; r < rows; r++) {
            const int32_t ph = r + 1;
            int8_t* out_row = out_bc + ph * W;
            for (int32_t k = 0; k < colRem; k++) {
                const int32_t pw = 1 + numBlocks * eleCount + k;
                int32_t sum9 = 0;
                for (int32_t kh = 0; kh < 3; kh++) {
                    const int8_t* row = in_bc + (ph - 1 + kh) * W;
                    sum9 += (int32_t)row[pw - 1] + (int32_t)row[pw] +
                            (int32_t)row[pw + 1];
                }
                int32_t v = ((sum9 - zx9) * scale_q) >> SHIFT;
                v += zy;
                out_row[pw] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
        }
    }
}

#endif  /* __C7524__ */

extern "C"
int32_t c7x_int8_avg_pool(
        const void* in, void* out,
        int32_t N, int32_t C, int32_t H_in, int32_t W_in,
        int32_t H_out, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* in_p = (const int8_t*)in;
    int8_t* out_p = (int8_t*)out;
    float inv_k = sx / (float)(kH * kW);  /* fuse dequant + 1/(kH*kW) */

#ifdef __C7524__
    const bool fastpath =
        kH == 3 && kW == 3 && sH == 1 && sW == 1 && pH == 1 && pW == 1 &&
        H_in == H_out && W_in == W_out && H_out >= 3 && W_out >= 3 &&
        H_out <= kFastpathMaxDim && W_out <= kFastpathMaxDim;
    /* Per-call constant (independent of b/c) — computed once, not per
     * channel, unlike a naive per-channel recompute of the same value. */
    const int32_t SHIFT = 13;
    const float combined_scale = inv_k / sy;
    const int32_t scale_q =
        (int32_t)(combined_scale * (float)(1 << SHIFT) + 0.5f);
    /* Border regions around the already-computed interior rectangle
     * [1, H_out-2] x [1, W_out-2]: top row, bottom row, then the
     * left/right columns of the remaining interior rows. */
    const int32_t border_ranges[4][4] = {
        {0, 1, 0, W_out},
        {H_out - 1, H_out, 0, W_out},
        {1, H_out - 1, 0, 1},
        {1, H_out - 1, W_out - 1, W_out},
    };
#endif

    for (int32_t b = 0; b < N; b++) {
        for (int32_t c = 0; c < C; c++) {
            const int8_t* in_bc  = in_p  + (b * C + c) * H_in  * W_in;
            int8_t*       out_bc = out_p + (b * C + c) * H_out * W_out;

#ifdef __C7524__
            if (fastpath) {
                avg_pool_interior_fast(in_bc, out_bc, H_out, W_out,
                                       zx, scale_q, zy);
                for (int32_t r = 0; r < 4; r++) {
                    avg_pool_scalar_rect(
                        in_bc, out_bc, H_in, W_in, H_out, W_out,
                        kH, kW, sH, sW, pH, pW, zx, inv_k, zy, sy,
                        border_ranges[r][0], border_ranges[r][1],
                        border_ranges[r][2], border_ranges[r][3]);
                }
                continue;
            }
#endif
            avg_pool_scalar_rect(in_bc, out_bc, H_in, W_in, H_out, W_out,
                                 kH, kW, sH, sW, pH, pW, zx, inv_k, zy, sy,
                                 0, H_out, 0, W_out);
        }
    }
    return 0;
}
