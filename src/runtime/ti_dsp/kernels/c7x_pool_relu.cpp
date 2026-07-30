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
 * @file c7x_pool_relu.cpp
 * @brief C7x-native int8 max-pooling, relu, clamp, and requantize kernels.
 *
 * c7x_int8_requantize_clamp, c7x_int8_relu, and c7x_int8_clamp all have SE
 * + integer fixed-point vectorized paths gated on #ifdef __C7524__: __int8
 * is an 8×int32 = 256-bit container specific to this variant; wider-vector
 * C7x parts would silently produce wrong results without the guard.  The
 * scalar fallback in the #else branch is the safe path for non-C7524
 * targets.
 *
 * c7x_int8_max_pool has its own SE + PROMOTE_4X_SIGNEXT vectorized interior
 * fast path (max_pool_interior_fast, #ifdef __C7524__) for the 3x3 and 2x2
 * shapes in its eligibility table; max_pool_scalar_rect is both the border
 * handler for that fast path and the full fallback for every other shape.
 */

#include "c7x_pool_relu.h"

#include <stdint.h>
#include <string.h>

/* Unconditional include (not gated on __C7524__): on the c7x_host g++
 * toolchain, __C7524__ is defined *by* <c7x.h> itself, not predefined by
 * the compiler — gating the include on the macro it defines is a
 * chicken-and-egg check that always evaluates false, silently disabling
 * the vectorized path on host emulation (the real c7x cross-compiler
 * predefines __C7524__ as a builtin before any header runs, so this only
 * broke host emulation, not hardware builds). See
 * c7x_avgpool.cpp for the same fix. */
#include <c7x.h>

/* =========================================================================
 * Scalar spatial max pool — reference implementation, also used as:
 *   - the correctness fallback for non-C7524 targets and unsupported
 *     kernel/stride/pad combinations
 *   - the border handler for the fast path below (ph_lo/ph_hi/pw_lo/pw_hi
 *     restrict which output rectangle gets (re)computed)
 *   - the column-remainder handler when the fast path's interior width
 *     isn't a multiple of 8
 *
 * Computes each output pixel directly (rather than the original's
 * init-to-INT8_MIN-then-accumulate passes) so it can be safely called on an
 * arbitrary sub-rectangle without disturbing pixels outside it. Produces the
 * same result: max is commutative/associative, so the set of valid (ih, iw)
 * taps per output pixel — unchanged here — determines the result, not the
 * order they're visited in.
 * ========================================================================= */

static void max_pool_scalar_rect(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t H_in, int32_t W_in, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
        int32_t ph_lo, int32_t ph_hi, int32_t pw_lo, int32_t pw_hi) {
    for (int32_t ph = ph_lo; ph < ph_hi; ph++) {
        int32_t ih0 = ph * sH - pH;
        for (int32_t pw = pw_lo; pw < pw_hi; pw++) {
            int32_t iw0 = pw * sW - pW;
            int8_t m = (int8_t)-128;
            for (int32_t kh = 0; kh < kH; kh++) {
                int32_t ih = ih0 + kh;
                if (ih < 0 || ih >= H_in) continue;
                const int8_t* row = in_bc + ih * W_in;
                for (int32_t kw = 0; kw < kW; kw++) {
                    int32_t iw = iw0 + kw;
                    if (iw < 0 || iw >= W_in) continue;
                    int8_t v = row[iw];
                    if (v > m) m = v;
                }
            }
            out_bc[ph * W_out + pw] = m;
        }
    }
}

#ifdef __C7524__

/* Compile-time cap on the fast-path interior width: max_pool_interior_fast's
 * row_scratch buffer below is fixed-size, sized to this capacity. Covers the
 * 56-wide ResNet-18 output and typical VGG-style pool widths with margin.
 * Wider outputs fall back to max_pool_scalar_rect via the fastpath
 * eligibility check in c7x_int8_max_pool below. */
static const int32_t kFastpathMaxWidth = 128;

/* C7524: 8 int32 lanes/advance. Shared between max_pool_interior_fast and
 * its caller's numBlocks/remainder-range computation below, so the two
 * never drift apart. */
static const int32_t kFastpathEleCount = 8;

/* C7524: 32 int8 lanes/advance -- the full vector width used (unpromoted)
 * by max_pool_interior_fast_dualrow below. Same sharing rationale as
 * kFastpathEleCount: the dispatcher's own numBlocks/remainder computation
 * for the dualrow path must use the identical value. */
static const int32_t kDualRowVecWidth = 32;

/* =========================================================================
 * max_pool_interior_fast — SE-vectorized windowed max, one call per
 * (kH, kW, sW) shape. Templated instead of duplicated per shape: kH*kW (the
 * tap-reduction loop trip count) becomes a compile-time constant per
 * instantiation, so the optimizer fully unrolls it (9 taps for 3x3, 4 for
 * 2x2) from one shared definition. Precedent for templating on shape/dtype
 * constants already exists in this codebase: mmalib_wrappers.cpp's
 * matmul_impl<ElemT, MmalibDtype, SatMin, SatMax>.
 *
 * Unlike avg_pool_interior_fast (c7x_avgpool.cpp), there is no Q13
 * rescale/clamp here: max is monotone, so QDQ is transparent (input and
 * output quant params are always equal) and the packed int8 max *is* the
 * output.
 *
 * Stride-2 columns use the SE's hardware DECIM_2 decimation instead of
 * TIDL's __pack_consec_low/high deinterleave trick: ICNT0 fetches 8*sW
 * contiguous input columns and keeps every sW-th one, so each of the kH*kW
 * tap reads already delivers exactly the 8 kept output columns for this
 * block — no wasted lanes, no manual deinterleave. One SE0 open, one
 * consistent 5D template, no SE1, no DECDIM, no circular addressing — the
 * same "one open, one template" structure that fixed avgpool's earlier
 * hardware-hang history (see c7x_avgpool.cpp's avg_pool_interior_fast
 * comment).
 *
 * Only computes the interior rectangle [ph_lo, ph_hi) x [pw_lo, pw_hi) —
 * the region where every tap of every output pixel is in-bounds, computed
 * by the caller. Any column remainder (interior width not a multiple of 8)
 * and the border outside the interior are left to max_pool_scalar_rect.
 * ========================================================================= */

template <int32_t kH, int32_t kW, int32_t sW>
static void max_pool_interior_fast(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t W_in, int32_t sH, int32_t W_out, int32_t pH, int32_t pW,
        int32_t ph_lo, int32_t ph_hi, int32_t pw_lo, int32_t pw_hi) {
    const int32_t rows        = ph_hi - ph_lo;
    const int32_t interior_w  = pw_hi - pw_lo;
    const int32_t numBlocks   = interior_w / kFastpathEleCount;

    if (rows <= 0 || numBlocks <= 0) return;

    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.DIMFMT  = __SE_DIMFMT_5D;
    se.DECIM   = (sW == 2) ? __SE_DECIM_2 : __SE_DECIM_OFF;
    se.ICNT0   = kFastpathEleCount * sW;
    se.ICNT1   = kW; se.DIM1 = 1;                    /* kw: horizontal tap */
    se.ICNT2   = kH; se.DIM2 = W_in;                  /* kh: vertical tap */
    se.ICNT3   = numBlocks; se.DIM3 = kFastpathEleCount * sW;  /* column blocks */
    se.ICNT4   = rows; se.DIM4 = sH * W_in;           /* output rows */

    const int8_t* base =
        in_bc + (ph_lo * sH - pH) * W_in + (pw_lo * sW - pW);
    __SE0_OPEN(const_cast<int8_t*>(base), se);

    const __int8 lo_v = (__int8)(-128);
    int8_t row_scratch[kFastpathMaxWidth];

    for (int32_t r = 0; r < rows; r++) {
        for (int32_t b = 0; b < numBlocks; b++) {
            __int8 m = lo_v;
            for (int32_t t = 0; t < kH * kW; t++)
                m = __max(m, __SE0ADV(int8));
            __vstore_pack_byte(
                reinterpret_cast<__char8*>(row_scratch + b * kFastpathEleCount), m);
        }
        memcpy(out_bc + (ph_lo + r) * W_out + pw_lo, row_scratch,
               numBlocks * kFastpathEleCount);
    }

    __SE0_CLOSE();
}

/* =========================================================================
 * max_pool_interior_fast_dualrow — two output rows per pass, stride-1
 * shapes only (kW deinterleave for stride-2 shapes is future work; see
 * maxpool_vectorized_notidl.md and the two-output-rows-per-pass follow-on
 * plan). Structurally different from max_pool_interior_fast above, not an
 * extension of it: this fetches each of the kH window rows once (kH SE
 * advances, not kH*kW) and gets the kW-1 remaining horizontal taps by
 * *shifting* the vertically-reduced row instead of re-fetching — so the
 * full window costs kH fetches + (kH-1)+(kW-1) __max calls, not kH*kW
 * fetches + kH*kW-1 max calls. This restructuring, not just adding a
 * second SE stream, is what a real accelerated max-pool kernel actually
 * does; verified against (not copied from) TI's own C7x max-pool
 * implementation for understanding only.
 *
 * No PROMOTE here (unlike max_pool_interior_fast): __max exists directly
 * on unpromoted __char32/__uchar32, and max has no overflow/accuracy need
 * for int32 headroom the way rescale arithmetic does. That also means a
 * fetch covers 32 raw int8 columns per SE advance (this device's full
 * vector width) instead of 8 post-promotion lanes.
 *
 * Two Streaming Engines process two *adjacent* output rows concurrently:
 * SE0's kH-row window is this pair's first output row's; SE1's is the
 * next output row's, based exactly sH input rows below SE0's (same
 * template, offset base address only -- the same dual-SE pattern already
 * proven in this codebase by c7x_residual_add.cpp, not the mismatched-
 * template pattern avgpool's history flagged as a hang risk).
 *
 * Horizontal-shift boundary handling: __shift_right_full zero-fills the
 * vacated tail lanes rather than pulling in the next block's data (I
 * confirmed this empirically -- host-emulation intrinsic behavior isn't
 * fully nailed down by the ISA doc comments alone), so the last kW-1
 * lanes of every 32-lane fetch are unusable as outputs. Each SE advance
 * therefore only yields `kVecWidth - (kW - 1)` valid output columns, and
 * consecutive column-block fetches overlap in their raw reads by kW-1
 * columns to cover the gap -- DIM2 below is the output-block width, not
 * the fetch width. Mirrors this file's existing scratch+memcpy idiom, one
 * level narrower: a plain vector store (no __vstore_pack_byte -- m0/m1 are
 * already byte-wide, nothing to pack down from) writes the full (partially
 * garbage) 32-lane result into a small scratch, and only the valid prefix
 * gets memcpy'd into the real per-block position -- no predicated store
 * needed.
 * ========================================================================= */

/* __shift_right_full's shift amount must be a genuine compile-time
 * constant on the real cl7x cross-compiler (host emulation's C++ library
 * implementation is more lenient and accepts a runtime argument, which is
 * how this got past c7x_host but failed cl7x with "input argument is not
 * a valid constant"). ShiftBytes as a template parameter, instantiated
 * only at literal call sites below, keeps the argument a literal. */
template <int32_t ShiftBytes>
static inline __char32 shift_right_bytes(__char32 v) {
    __long4 s = __shift_right_full(
        *reinterpret_cast<__long4*>(&v), (uint8_t)(8 * ShiftBytes));
    return *reinterpret_cast<__char32*>(&s);
}

template <int32_t kH, int32_t kW>
static void max_pool_interior_fast_dualrow(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t W_in, int32_t sH, int32_t W_out, int32_t pH, int32_t pW,
        int32_t ph_lo, int32_t pw_lo, int32_t pw_hi, int32_t row_pairs,
        int32_t numBlocks) {
    static_assert(kW == 2 || kW == 3,
                  "horizontal reduction below is hand-unrolled for kW 2-3 only");
    const int32_t block_out_width = kDualRowVecWidth - (kW - 1);

    if (row_pairs <= 0 || numBlocks <= 0) return;

    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_32ELEMS;
    se.DIMFMT  = __SE_DIMFMT_4D;
    se.ICNT0   = kDualRowVecWidth;
    se.ICNT1   = kH; se.DIM1 = W_in;                     /* kh: vertical tap */
    se.ICNT2   = numBlocks; se.DIM2 = block_out_width;    /* column blocks */
    se.ICNT3   = row_pairs; se.DIM3 = 2 * sH * W_in;      /* row-pair loop */

    const int8_t* base0 = in_bc + (ph_lo * sH - pH) * W_in + (pw_lo - pW);
    const int8_t* base1 = base0 + sH * W_in;
    __SE0_OPEN(const_cast<int8_t*>(base0), se);
    __SE1_OPEN(const_cast<int8_t*>(base1), se);

    int8_t row_scratch0[kFastpathMaxWidth];
    int8_t row_scratch1[kFastpathMaxWidth];
    int8_t tmp0[kDualRowVecWidth];
    int8_t tmp1[kDualRowVecWidth];

    for (int32_t rp = 0; rp < row_pairs; rp++) {
        for (int32_t b = 0; b < numBlocks; b++) {
            /* Vertical max: kH row-fetches, kH-1 __max calls. */
            __char32 m0 = __SE0ADV(char32);
            for (int32_t kh = 1; kh < kH; kh++) m0 = __max(m0, __SE0ADV(char32));
            __char32 m1 = __SE1ADV(char32);
            for (int32_t kh = 1; kh < kH; kh++) m1 = __max(m1, __SE1ADV(char32));

            /* Horizontal max: kW-1 taps via register shift of the
             * vertically-reduced row (no re-fetch). Each shift is taken
             * from the original vertically-maxed row (vmax0/vmax1), not
             * the progressively-updated m0/m1 -- shifting an
             * already-shifted result would combine kw+1 taps instead of
             * exactly kw taps by the last iteration. Tail (kW-1) lanes of
             * the shifted view are zero-filled (confirmed empirically),
             * not real neighbor data -- harmless here since only the
             * leading block_out_width lanes of m0/m1 ever get copied out
             * below. Unrolled by hand (kW == 2 or 3 only, matching this
             * file's eligible-shape table) rather than a runtime loop over
             * shift amounts: see shift_right_bytes's comment. */
            const __char32 vmax0 = m0;
            const __char32 vmax1 = m1;
            m0 = __max(m0, shift_right_bytes<1>(vmax0));
            m1 = __max(m1, shift_right_bytes<1>(vmax1));
            if (kW >= 3) {
                m0 = __max(m0, shift_right_bytes<2>(vmax0));
                m1 = __max(m1, shift_right_bytes<2>(vmax1));
            }

            *reinterpret_cast<__char32*>(tmp0) = m0;
            memcpy(row_scratch0 + b * block_out_width, tmp0, block_out_width);
            *reinterpret_cast<__char32*>(tmp1) = m1;
            memcpy(row_scratch1 + b * block_out_width, tmp1, block_out_width);
        }
        memcpy(out_bc + (ph_lo + 2 * rp) * W_out + pw_lo,
               row_scratch0, numBlocks * block_out_width);
        memcpy(out_bc + (ph_lo + 2 * rp + 1) * W_out + pw_lo,
               row_scratch1, numBlocks * block_out_width);
    }

    __SE0_CLOSE();
    __SE1_CLOSE();
}

/* =========================================================================
 * max_pool_interior_fast_dualrow_s2 — two output rows per pass, stride-2
 * shapes (3x3/s2, 2x2/s2). Structurally different from the stride-1
 * dualrow path above, not a variant of it: stride-2 needs to both
 * deinterleave (keep every 2nd raw column) and combine the two streams'
 * rows, which __pack_consec_low/high do simultaneously. This is *why*
 * this path doesn't use DECIM_2 (the mechanism the older single-row
 * max_pool_interior_fast uses for its stride-2 shapes): DECIM_2 discards
 * exactly the odd-column data a register-shift tap reuse would need, so
 * it doesn't compose with this design the way pack_consec does.
 *
 * Semantics confirmed empirically against this exact C7524 target via a
 * standalone host-emulation probe before this function was written (not
 * assumed from documentation, which doesn't disambiguate the byte-level
 * VBPACKH/VBPACKL behavior clearly enough to be sure):
 *   pack_consec_low(A, B)  = { evens(B) in lanes [0,16), evens(A) in [16,32) }
 *   pack_consec_high(A, B) = { odds(B)  in lanes [0,16), odds(A)  in [16,32) }
 * So pack_consec_low(vmax1, vmax0) puts stream0's (vmax0's) even columns
 * in the LOW half and stream1's (vmax1's) even columns in the HIGH half:
 * one combined vector holding both output rows, deinterleaved by stride
 * in the same operation -- no separate row-combine step needed.
 *
 * For kW==3, the third tap (kw=2) is "the next even column" relative to
 * vA's own lanes: a 1-lane shift of vA itself (shift_right_bytes<1>), not
 * a further pack_consec call. That shift crosses from the low half into
 * the high half at lane 15->16, so lane 15 (stream0's last lane) and lane
 * 31 (stream1's last lane, shifted-in as zero) are not valid outputs --
 * confirmed by the same probe. So each half yields only
 * (kDualRowVecWidth/2)-1 valid output columns for kW==3, not
 * kDualRowVecWidth/2; for kW==2 no shift is needed (hmax=max(vA,vB)
 * directly) and all kDualRowVecWidth/2 lanes per half are valid. This was
 * cross-checked against (not copied from) TI's own reported tile-advance
 * for the equivalent 3x3/stride-2 case, for understanding only.
 * ========================================================================= */

/* Vertical max (kH row-fetches from whichever SE0/SE1 session is currently
 * open) + horizontal pack_consec/shift combination -- the per-block body
 * shared by max_pool_interior_fast_dualrow_s2's main block loop and its
 * tail-overlap pass below, so the combination logic isn't duplicated
 * between them. */
template <int32_t kH, int32_t kW>
static inline __char32 dualrow_s2_block_hmax() {
    __char32 vmax0 = __SE0ADV(char32);
    for (int32_t kh = 1; kh < kH; kh++) vmax0 = __max(vmax0, __SE0ADV(char32));
    __char32 vmax1 = __SE1ADV(char32);
    for (int32_t kh = 1; kh < kH; kh++) vmax1 = __max(vmax1, __SE1ADV(char32));

    __char32 vA = __pack_consec_low(vmax1, vmax0);
    __char32 vB = __pack_consec_high(vmax1, vmax0);
    if (kW == 3) return __max(vA, __max(vB, shift_right_bytes<1>(vA)));
    return __max(vA, vB);
}

/* Open SE0/SE1 at (pw_lo_local, numBlocks_local) and run the full
 * row-pair x block loop -- shared by max_pool_interior_fast_dualrow_s2's
 * main pass (numBlocks_local = num_blocks_dual, at pw_lo) and its
 * tail-overlap pass (numBlocks_local = 1, at a different offset). The
 * tail is not a structurally distinct case: with ICNT2 = 1 the "column
 * blocks" loop dimension simply doesn't iterate a second time, so one
 * shared 4D template covers both -- no separate 3D template needed. */
template <int32_t kH, int32_t kW>
static void dualrow_s2_process_blocks(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t W_in, int32_t sH, int32_t W_out, int32_t pH, int32_t pW,
        int32_t ph_lo, int32_t pw_lo_local, int32_t row_pairs,
        int32_t numBlocks_local) {
    const int32_t half_width = kDualRowVecWidth / 2;                 /* 16 */
    const int32_t block_out_width = half_width - (kW == 3 ? 1 : 0);  /* 15 or 16 */
    const int32_t raw_block_advance = 2 * block_out_width;           /* sW == 2 */

    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_32ELEMS;
    se.DIMFMT  = __SE_DIMFMT_4D;
    se.ICNT0   = kDualRowVecWidth;
    se.ICNT1   = kH; se.DIM1 = W_in;                              /* kh: vertical tap */
    se.ICNT2   = numBlocks_local; se.DIM2 = raw_block_advance;    /* column blocks */
    se.ICNT3   = row_pairs; se.DIM3 = 2 * sH * W_in;              /* row-pair loop */

    const int8_t* base0 = in_bc + (ph_lo * sH - pH) * W_in + (pw_lo_local * 2 - pW);
    const int8_t* base1 = base0 + sH * W_in;
    __SE0_OPEN(const_cast<int8_t*>(base0), se);
    __SE1_OPEN(const_cast<int8_t*>(base1), se);

    int8_t row_scratch0[kFastpathMaxWidth];
    int8_t row_scratch1[kFastpathMaxWidth];
    int8_t tmp[kDualRowVecWidth];

    for (int32_t rp = 0; rp < row_pairs; rp++) {
        for (int32_t b = 0; b < numBlocks_local; b++) {
            __char32 hmax = dualrow_s2_block_hmax<kH, kW>();
            *reinterpret_cast<__char32*>(tmp) = hmax;
            memcpy(row_scratch0 + b * block_out_width, tmp, block_out_width);
            memcpy(row_scratch1 + b * block_out_width, tmp + half_width, block_out_width);
        }
        memcpy(out_bc + (ph_lo + 2 * rp) * W_out + pw_lo_local,
               row_scratch0, numBlocks_local * block_out_width);
        memcpy(out_bc + (ph_lo + 2 * rp + 1) * W_out + pw_lo_local,
               row_scratch1, numBlocks_local * block_out_width);
    }

    __SE0_CLOSE();
    __SE1_CLOSE();
}

template <int32_t kH, int32_t kW>
static void max_pool_interior_fast_dualrow_s2(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t W_in, int32_t sH, int32_t W_out, int32_t pH, int32_t pW,
        int32_t ph_lo, int32_t pw_lo, int32_t pw_hi, int32_t row_pairs,
        int32_t numBlocks) {
    static_assert(kW == 2 || kW == 3,
                  "horizontal combination below is hand-unrolled for kW 2-3 only");
    const int32_t block_out_width = kDualRowVecWidth / 2 - (kW == 3 ? 1 : 0);

    if (row_pairs <= 0) return;

    if (numBlocks > 0) {
        dualrow_s2_process_blocks<kH, kW>(in_bc, out_bc, W_in, sH, W_out, pH, pW,
                                           ph_lo, pw_lo, row_pairs, numBlocks);
    }

    /* Tail-overlap block: if the interior width isn't a multiple of
     * block_out_width, do ONE more block positioned so it lands exactly on
     * pw_hi -- overlapping (redundantly recomputing, harmlessly) the tail
     * of the last regular block -- instead of falling back to the much
     * slower scalar remainder path. This is what turned stage 3 from a
     * regression on the ResNet-18 shape (interior_w=55, block_out_width=15
     * leaves a 10-column remainder) into a net win: the caller skips its
     * own remainder max_pool_scalar_rect call whenever this path is taken
     * (see c7x_int8_max_pool below) since this fully covers [pw_lo, pw_hi)
     * on its own. Only applies when there's at least one full block's
     * worth of width to overlap into; narrower interiors are left entirely
     * to the caller's scalar fallback, same as the numBlocks<=0 case. */
    const int32_t interior_w = pw_hi - pw_lo;
    if (interior_w > numBlocks * block_out_width && interior_w >= block_out_width) {
        dualrow_s2_process_blocks<kH, kW>(in_bc, out_bc, W_in, sH, W_out, pH, pW,
                                           ph_lo, pw_hi - block_out_width,
                                           row_pairs, /*numBlocks_local=*/1);
    }
}

/* Shared by both fastpath branches in c7x_int8_max_pool below: apply the
 * (up to 4) border strips outside the interior rectangle, skipping any
 * range collapsed to empty (e.g. the common case where padding only
 * touches one edge, as in the ResNet-18 3x3/s2/p1 shape, or not at all, as
 * in the 2x2/s2/p0 shape). */
static void apply_border_ranges(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t H_in, int32_t W_in, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
        const int32_t border_ranges[4][4]) {
    for (int32_t r = 0; r < 4; r++) {
        if (border_ranges[r][0] >= border_ranges[r][1] ||
            border_ranges[r][2] >= border_ranges[r][3])
            continue;
        max_pool_scalar_rect(
            in_bc, out_bc, H_in, W_in, W_out, kH, kW, sH, sW, pH, pW,
            border_ranges[r][0], border_ranges[r][1],
            border_ranges[r][2], border_ranges[r][3]);
    }
}

using FastPathFn = void (*)(const int8_t*, int8_t*, int32_t, int32_t,
                             int32_t, int32_t, int32_t, int32_t,
                             int32_t, int32_t, int32_t);

/* Shared by both fastpath branches in c7x_int8_max_pool below: when the
 * dual-row interior row count is odd, the one trailing row falls back to
 * single_row_fn (always max_pool_interior_fast<kH,kW,sW> for whichever
 * shape is active), plus its own column-remainder split -- block width
 * kFastpathEleCount, not either dual-row path's wider block. */
static void apply_trailing_row_fallback(
        const int8_t* in_bc, int8_t* out_bc,
        int32_t H_in, int32_t W_in, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW,
        int32_t trailing_ph, int32_t ph_hi, int32_t pw_lo, int32_t pw_hi,
        FastPathFn single_row_fn) {
    if (trailing_ph >= ph_hi) return;
    single_row_fn(in_bc, out_bc, W_in, sH, W_out, pH, pW,
                  trailing_ph, trailing_ph + 1, pw_lo, pw_hi);
    const int32_t num_blocks_single = (pw_hi - pw_lo) / kFastpathEleCount;
    max_pool_scalar_rect(
        in_bc, out_bc, H_in, W_in, W_out, kH, kW, sH, sW, pH, pW,
        trailing_ph, trailing_ph + 1,
        pw_lo + num_blocks_single * kFastpathEleCount, pw_hi);
}

#endif  /* __C7524__ */

extern "C"
int32_t c7x_int8_max_pool(
        const void* in, void* out,
        int32_t N, int32_t C, int32_t H_in, int32_t W_in,
        int32_t H_out, int32_t W_out,
        int32_t kH, int32_t kW, int32_t sH, int32_t sW,
        int32_t pH, int32_t pW) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;

#ifdef __C7524__
    const bool shape_3x3_s1 =
        kH == 3 && kW == 3 && sH == 1 && sW == 1 && pH == 1 && pW == 1;
    const bool shape_3x3_s2 =
        kH == 3 && kW == 3 && sH == 2 && sW == 2 && pH == 1 && pW == 1;
    const bool shape_2x2_s2 =
        kH == 2 && kW == 2 && sH == 2 && sW == 2 && pH == 0 && pW == 0;
    const bool fastpath = (shape_3x3_s1 || shape_3x3_s2 || shape_2x2_s2) &&
                          W_out <= kFastpathMaxWidth;

    if (fastpath) {
        /* Interior rectangle: the region where every tap of every output
         * pixel is in-bounds, from the tightest per-tap constraint (kh=0/
         * kw=0 for the lower bound, kh=kH-1/kw=kW-1 for the upper bound) —
         * same arithmetic the scalar kernel below uses per-tap, generalized
         * across the whole window. May be empty (ph_lo==ph_hi and/or
         * pw_lo==pw_hi) for tiny inputs; max_pool_scalar_rect is a no-op on
         * an empty range, so no separate guard is needed for that case. */
        const int32_t ph_lo = (pH + sH - 1) / sH;
        const int32_t ph_hi = (H_in - kH + pH) / sH + 1;
        const int32_t pw_lo = (pW + sW - 1) / sW;
        const int32_t pw_hi = (W_in - kW + pW) / sW + 1;

        /* Border strips outside the interior rectangle: top/bottom
         * full-width, then left/right of the middle row band — same
         * partition avg_pool's border_ranges uses. Shape-agnostic (both
         * branches below cover the full [ph_lo,ph_hi) interior between
         * them), so computed once for either path. */
        const int32_t border_ranges[4][4] = {
            {0, ph_lo, 0, W_out},
            {ph_hi, H_out, 0, W_out},
            {ph_lo, ph_hi, 0, pw_lo},
            {ph_lo, ph_hi, pw_hi, W_out},
        };

        /* Row-pairing and the trailing-odd-row split are shape-agnostic
         * (both dual-row paths below pair up interior rows the same way),
         * so computed once for either branch. The trailing row always
         * falls back to the single-row max_pool_interior_fast, whichever
         * shape's instantiation applies. */
        const int32_t row_pairs = (ph_hi - ph_lo) / 2;
        const int32_t trailing_ph = ph_lo + row_pairs * 2;
        const FastPathFn single_row_fn =
            shape_3x3_s1 ? &max_pool_interior_fast<3, 3, 1>
            : shape_3x3_s2 ? &max_pool_interior_fast<3, 3, 2>
                           : &max_pool_interior_fast<2, 2, 2>;

        if (shape_3x3_s1) {
            /* Two-output-rows-per-pass path (max_pool_interior_fast_dualrow):
             * pairs of adjacent interior rows via SE0+SE1; a trailing odd
             * row (if the interior row count is odd) falls back to the
             * single-row max_pool_interior_fast, which has its own,
             * differently-sized column-remainder split (its block width is
             * kFastpathEleCount, not the dualrow path's 32-(kW-1)). */
            const int32_t dual_block_width = kDualRowVecWidth - (kW - 1);
            const int32_t num_blocks_dual = (pw_hi - pw_lo) / dual_block_width;

            for (int32_t b = 0; b < N; b++) {
                for (int32_t c = 0; c < C; c++) {
                    const int8_t* in_bc  = p + (b * C + c) * H_in  * W_in;
                    int8_t*       out_bc = q + (b * C + c) * H_out * W_out;

                    max_pool_interior_fast_dualrow<3, 3>(
                        in_bc, out_bc, W_in, sH, W_out, pH, pW,
                        ph_lo, pw_lo, pw_hi, row_pairs, num_blocks_dual);
                    max_pool_scalar_rect(
                        in_bc, out_bc, H_in, W_in, W_out, kH, kW, sH, sW, pH, pW,
                        ph_lo, trailing_ph,
                        pw_lo + num_blocks_dual * dual_block_width, pw_hi);

                    apply_trailing_row_fallback(
                        in_bc, out_bc, H_in, W_in, W_out, kH, kW, sH, sW, pH, pW,
                        trailing_ph, ph_hi, pw_lo, pw_hi, single_row_fn);
                    apply_border_ranges(in_bc, out_bc, H_in, W_in, W_out,
                                        kH, kW, sH, sW, pH, pW, border_ranges);
                }
            }
            return 0;
        }

        /* shape_3x3_s2 / shape_2x2_s2: two-output-rows-per-pass stride-2
         * path (max_pool_interior_fast_dualrow_s2). Same row-pairing +
         * trailing-odd-row-fallback structure as shape_3x3_s1 above, but a
         * different block width (pack_consec-based, half-vector-width
         * minus 1 for kW==3, or the full half-vector-width for kW==2 --
         * see that function's comment) and its own SE1 kh-tap dispatch
         * since kH/kW differ (3 vs 2) between the two shapes sharing this
         * branch. */
        const int32_t dual_block_width =
            kDualRowVecWidth / 2 - (shape_3x3_s2 ? 1 : 0);
        const int32_t num_blocks_dual = (pw_hi - pw_lo) / dual_block_width;

        /* Shape -> template instantiation is a per-call constant: pick
         * once instead of re-branching every (b, c). */
        using DualRowS2Fn = void (*)(const int8_t*, int8_t*, int32_t, int32_t,
                                      int32_t, int32_t, int32_t, int32_t,
                                      int32_t, int32_t, int32_t, int32_t);
        const DualRowS2Fn dualrow_s2_fn =
            shape_3x3_s2 ? &max_pool_interior_fast_dualrow_s2<3, 3>
                         : &max_pool_interior_fast_dualrow_s2<2, 2>;

        for (int32_t b = 0; b < N; b++) {
            for (int32_t c = 0; c < C; c++) {
                const int8_t* in_bc  = p + (b * C + c) * H_in  * W_in;
                int8_t*       out_bc = q + (b * C + c) * H_out * W_out;

                dualrow_s2_fn(in_bc, out_bc, W_in, sH, W_out, pH, pW,
                              ph_lo, pw_lo, pw_hi, row_pairs, num_blocks_dual);
                /* max_pool_interior_fast_dualrow_s2 covers the *entire*
                 * [pw_lo, pw_hi) width itself via its tail-overlap block
                 * whenever there's at least one full block (num_blocks_dual
                 * > 0) -- no remainder call needed in that case. Only
                 * num_blocks_dual == 0 (which the function does nothing
                 * for) needs the full width scalar'd here. */
                if (num_blocks_dual == 0) {
                    max_pool_scalar_rect(
                        in_bc, out_bc, H_in, W_in, W_out, kH, kW, sH, sW, pH, pW,
                        ph_lo, trailing_ph, pw_lo, pw_hi);
                }

                apply_trailing_row_fallback(
                    in_bc, out_bc, H_in, W_in, W_out, kH, kW, sH, sW, pH, pW,
                    trailing_ph, ph_hi, pw_lo, pw_hi, single_row_fn);
                apply_border_ranges(in_bc, out_bc, H_in, W_in, W_out,
                                    kH, kW, sH, sW, pH, pW, border_ranges);
            }
        }
        return 0;
    }
#endif

    for (int32_t b = 0; b < N; b++) {
        for (int32_t c = 0; c < C; c++) {
            const int8_t* in_bc  = p + (b * C + c) * H_in  * W_in;
            int8_t*       out_bc = q + (b * C + c) * H_out * W_out;
            max_pool_scalar_rect(in_bc, out_bc, H_in, W_in, W_out,
                                 kH, kW, sH, sW, pH, pW,
                                 0, H_out, 0, W_out);
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
 * (ti_fuse_qdq_c7x_relu.py) and passed as float32 at compile time.
 *
 * Vectorized implementation (C7524 only):
 *   - Q13 fixed-point: scale_q = round(combined_scale * 8192)
 *     Safe for combined_scale up to 255: max product = 127×8192×255 = 265M
 *     which fits in int32 (< INT32_MAX ≈ 2147M).
 *   - SE0 streams int8 input, sign-extends 4× to int32 (__int8 = 8×int32)
 *   - 4× unrolled loop matching c7x_residual_add.cpp:156–196
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
     * c7x_residual_add.cpp lines 141–145. */
    __SE_TEMPLATE_v1 se = __gen_SE_TEMPLATE_v1();
    se.ELETYPE = __SE_ELETYPE_8BIT;
    se.VECLEN  = __SE_VECLEN_8ELEMS;
    se.PROMOTE = __SE_PROMOTE_4X_SIGNEXT;
    se.ICNT0   = (uint32_t)(nvec * 8);

    __SE0_OPEN(const_cast<int8_t*>(in), se);

    int32_t i = 0;

    /* 4× unrolled: four independent chains hide the 4–6 cycle SE latency.
     *
     * No #pragma MUST_ITERATE(1,,): nvec4 can be 0 for small n -- see
     * c7x_quantize.cpp's quantize_1plane for the full investigation. */
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
 * relu/clamp are only lowered by ti_fuse_qdq_c7x_relu.py's
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

    /* No #pragma MUST_ITERATE(1,,): nvec4 can be 0 for small n -- see
     * c7x_quantize.cpp's quantize_1plane for the full investigation. */
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

    /* No #pragma MUST_ITERATE(1,,): nvec4 can be 0 for small n -- see
     * c7x_quantize.cpp's quantize_1plane for the full investigation. */
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
