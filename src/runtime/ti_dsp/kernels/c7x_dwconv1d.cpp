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
 * \file c7x_dwconv1d.cpp
 * \brief int8 depthwise 1-D convolution with per-channel requantize.
 *
 * C7524 path. Each channel's input is copied once into a zero-padded line
 * buffer (for stride 2, as its even and odd samples, one after the other, so
 * every tap reads consecutive bytes). Outputs are computed 32 at a time in
 * four __int8 (8 x int32) accumulators that stay in registers across all
 * taps: per tap, four unaligned 8-byte loads, widened and multiplied by the
 * broadcast weight. The accumulators are then requantized as vectors and
 * stored as bytes.
 *
 * Keeping the accumulators in registers is what makes this fast. With them
 * in memory (an int32 row updated once per tap) the same loop ran about 5x
 * slower on BeagleY-AI, whose firmware runs L1D write-through; the
 * unscheduled generated code TVM emits for these convs (a padded copy, an
 * int32 and a float intermediate in DDR, and a pass per requantize step) ran
 * about 7x slower than this kernel.
 *
 * The requantize matches the generic path bit for bit: (float)acc * scale in
 * float32, then round half to even and clip. Clipping before rounding gives
 * the same result because both bounds are integers, and keeps VSPINT inside
 * its range.
 */

#include "c7x_dwconv1d.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Unconditional include: on the c7x_host g++ toolchain <c7x.h> is what
 * defines __C7524__ (see c7x_quantize.cpp). */
#include <c7x.h>

#define BLOCK 32

/* Firmware runs one inference at a time, so one line buffer serves every
 * call. */
static int8_t s_line[C7X_DWCONV1D_MAX_LINE + BLOCK] __attribute__((aligned(64)));
static int32_t s_tap_offset[C7X_DWCONV1D_MAX_TAPS];

/* Bytes of line buffer one channel needs: outputs are computed in whole
 * blocks of BLOCK, so the last block reads past T_out. */
static int32_t line_bytes(int32_t T_out, int32_t K, int32_t stride, int32_t dilation) {
  int32_t t_blk = (T_out + BLOCK - 1) / BLOCK * BLOCK;
  int32_t span = (K - 1) * dilation;
  if (stride == 1) return t_blk + span;
  return 2 * (t_blk + span / 2 + 1); /* stride 2: even and odd phases */
}

/* Fill the line buffer for one channel. Stride 1: xpad itself. Stride 2:
 * xpad's even samples, then its odd samples, each half_len long. */
static void fill_line(const int8_t* xc, int32_t T_in, int32_t pad_left, int32_t len, int32_t stride) {
  if (stride == 1) {
    int32_t start = pad_left < len ? pad_left : len;
    int32_t n = T_in < len - start ? T_in : len - start;
    memset(s_line, 0, (size_t)start);
    memcpy(s_line + start, xc, (size_t)n);
    memset(s_line + start + n, 0, (size_t)(len - start - n + BLOCK));
    return;
  }
  int32_t half = len / 2;
  for (int32_t phase = 0; phase < 2; ++phase) {
    int8_t* dst = s_line + phase * half;
    for (int32_t j = 0; j < half; ++j) {
      int32_t i = 2 * j + phase - pad_left;
      dst[j] = (i >= 0 && i < T_in) ? xc[i] : (int8_t)0;
    }
  }
  memset(s_line + len, 0, BLOCK);
}

#ifdef __C7524__

static inline __int8 requantize(__int8 a, __float8 vs) {
  __float8 f = __int_to_float(a) * vs;
  f = __max(__min(f, (__float8)127.0f), (__float8)-128.0f);
  return __float_to_int(f);
}

/* Store the first n (< 8) lanes of q as bytes. */
static inline void store_partial(int8_t* dst, __int8 q, int32_t n) {
  int8_t tmp[8];
  __vstore_pack_byte((__char8*)tmp, q);
  for (int32_t j = 0; j < n; ++j) dst[j] = tmp[j];
}

#if defined(__TI_COMPILER_VERSION__) && !defined(C7X_HOST_EMULATION)
#define DWCONV1D_UNROLL _Pragma("UNROLL(4)")
#else
#define DWCONV1D_UNROLL
#endif

/* One tap into the four accumulators, reading 32 consecutive bytes at p. */
#define DWCONV1D_MAC(p)                                       \
  do {                                                        \
    a0 += __convert_int8(*(const __char8*)((p) + 0)) * wk;    \
    a1 += __convert_int8(*(const __char8*)((p) + 8)) * wk;    \
    a2 += __convert_int8(*(const __char8*)((p) + 16)) * wk;   \
    a3 += __convert_int8(*(const __char8*)((p) + 24)) * wk;   \
  } while (0)

extern "C" int32_t c7x_int8_dwconv1d(const void* x_ptr, const void* w_ptr, const void* scale_ptr,
                                     void* y_ptr, int32_t C, int32_t T_in, int32_t T_out, int32_t K,
                                     int32_t stride, int32_t dilation, int32_t pad_left) {
  const int8_t* x = (const int8_t*)x_ptr;
  const int8_t* w = (const int8_t*)w_ptr;
  const float* scale = (const float*)scale_ptr;
  int8_t* y = (int8_t*)y_ptr;
  if ((stride != 1 && stride != 2) || dilation < 1 || K < 1 || K > C7X_DWCONV1D_MAX_TAPS ||
      pad_left < 0 || T_out < 1) {
    return -1;
  }
  int32_t len = line_bytes(T_out, K, stride, dilation);
  if (len > C7X_DWCONV1D_MAX_LINE) return -1;

  /* Stride 2: each tap's first byte in the even/odd line, for output 0. */
  int32_t half = len / 2;
  if (stride == 2) {
    for (int32_t k = 0; k < K; ++k) {
      int32_t off = k * dilation;
      s_tap_offset[k] = (off & 1) * half + (off >> 1);
    }
  }

  for (int32_t c = 0; c < C; ++c) {
    fill_line(x + c * T_in, T_in, pad_left, len, stride);
    const int8_t* wc = w + c * K;
    const __float8 vs = (__float8)scale[c];
    int8_t* yc = y + c * T_out;
    for (int32_t t0 = 0; t0 < T_out; t0 += BLOCK) {
      __int8 a0 = (__int8)0, a1 = (__int8)0, a2 = (__int8)0, a3 = (__int8)0;
      const int8_t* xb = s_line + t0;
      /* Three loops so each indexes the line as simply as it can: cl7x
       * pipelines the dilation-1 loop at II 9 for four taps (about 14 MACs a
       * cycle) and the other two at II 11. UNROLL(4): fully unrolled, the
       * loops run out of registers and don't pipeline. */
      if (stride == 1 && dilation == 1) {
        DWCONV1D_UNROLL
        for (int32_t k = 0; k < K; ++k) {
          int32_t wv = wc[k];
          __int8 wk = (__int8)wv;
          DWCONV1D_MAC(xb + k);
        }
      } else if (stride == 1) {
        const int8_t* xk = xb;
        DWCONV1D_UNROLL
        for (int32_t k = 0; k < K; ++k) {
          int32_t wv = wc[k];
          __int8 wk = (__int8)wv;
          DWCONV1D_MAC(xk);
          xk += dilation;
        }
      } else {
        DWCONV1D_UNROLL
        for (int32_t k = 0; k < K; ++k) {
          int32_t wv = wc[k];
          __int8 wk = (__int8)wv;
          DWCONV1D_MAC(xb + s_tap_offset[k]);
        }
      }
      int8_t* yb = yc + t0;
      int32_t rem = T_out - t0;
      if (rem >= BLOCK) {
        __vstore_pack_byte((__char8*)(yb + 0), requantize(a0, vs));
        __vstore_pack_byte((__char8*)(yb + 8), requantize(a1, vs));
        __vstore_pack_byte((__char8*)(yb + 16), requantize(a2, vs));
        __vstore_pack_byte((__char8*)(yb + 24), requantize(a3, vs));
      } else {
        __int8 q[4] = {requantize(a0, vs), requantize(a1, vs), requantize(a2, vs),
                       requantize(a3, vs)};
        for (int32_t i = 0; i < 4 && rem > 0; ++i, rem -= 8) {
          if (rem >= 8) {
            __vstore_pack_byte((__char8*)(yb + 8 * i), q[i]);
          } else {
            store_partial(yb + 8 * i, q[i], rem);
          }
        }
      }
    }
  }
  return 0;
}

#else /* scalar fallback for non-C7524 builds */

extern "C" int32_t c7x_int8_dwconv1d(const void* x_ptr, const void* w_ptr, const void* scale_ptr,
                                     void* y_ptr, int32_t C, int32_t T_in, int32_t T_out, int32_t K,
                                     int32_t stride, int32_t dilation, int32_t pad_left) {
  const int8_t* x = (const int8_t*)x_ptr;
  const int8_t* w = (const int8_t*)w_ptr;
  const float* scale = (const float*)scale_ptr;
  int8_t* y = (int8_t*)y_ptr;
  if ((stride != 1 && stride != 2) || dilation < 1 || K < 1 || K > C7X_DWCONV1D_MAX_TAPS ||
      pad_left < 0 || T_out < 1 ||
      line_bytes(T_out, K, stride, dilation) > C7X_DWCONV1D_MAX_LINE) {
    return -1;
  }
  for (int32_t c = 0; c < C; ++c) {
    const int8_t* xc = x + c * T_in;
    const int8_t* wc = w + c * K;
    for (int32_t t = 0; t < T_out; ++t) {
      int32_t acc = 0;
      for (int32_t k = 0; k < K; ++k) {
        int32_t i = t * stride + k * dilation - pad_left;
        if (i >= 0 && i < T_in) acc += (int32_t)xc[i] * (int32_t)wc[k];
      }
      float f = nearbyintf((float)acc * scale[c]);
      f = f > 127.0f ? 127.0f : (f < -128.0f ? -128.0f : f);
      y[c * T_out + t] = (int8_t)f;
    }
  }
  (void)fill_line;
  return 0;
}

#endif /* __C7524__ */
