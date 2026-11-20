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
 * every tap reads consecutive bytes); c7x_int8_dwconv1d_relu applies a ReLU
 * during that copy. Outputs are computed 128 at a time in sixteen __int8
 * (8 x int32) accumulators that stay in registers across all taps: per tap,
 * sixteen unaligned 8-byte loads, widened and multiplied by the broadcast
 * weight. The accumulators are then requantized as vectors and stored as
 * bytes. While a channel computes, the next channel's input is prefetched.
 *
 * Measured on BeagleY-AI, QuartzNet 15x5's 77 depthwise convs:
 * - Accumulators in registers are what makes this fast. With them in memory
 *   (an int32 row updated once per tap) the loop ran about 5x slower: the
 *   firmware runs L1D write-through, so every store goes on to L2. For the
 *   same reason the line buffer is filled with memcpy or 32-byte vector
 *   stores; a byte-at-a-time copy made the kernel 2.5x slower.
 * - 128 outputs per block, against 32, pays each tap loop's pipeline start-up
 *   a quarter as often (cl7x pipelines the 16-accumulator loop at II 9 per
 *   tap): 246 M -> 181 M cycles.
 * - Prefetching the next channel's input hides most of its DDR misses:
 *   181 M -> 136 M cycles.
 * - c7x_int8_dwconv1d_relu removes 60 ReLU passes (13.5 M cycles) for about
 *   1 M of extra copy time.
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

#define BLOCK 128

/* Firmware runs one inference at a time, so one line buffer serves every
 * call. s_line points into it: at stride 1 it is offset so that the input's
 * first byte (after the left padding) lands on a 32-byte boundary, and the
 * ReLU copy's vector stores are aligned. Unaligned, they made the ReLU
 * variant about 15% slower; the kernel's loads are unaligned either way. */
static int8_t s_line_storage[C7X_DWCONV1D_MAX_LINE + BLOCK + 32] __attribute__((aligned(64)));
static int8_t* s_line = s_line_storage;
static int32_t s_tap_offset[C7X_DWCONV1D_MAX_TAPS];

/* Bytes of line buffer one channel needs: outputs are computed in whole
 * blocks of BLOCK, so the last block reads past T_out. */
static int32_t line_bytes(int32_t T_out, int32_t K, int32_t stride, int32_t dilation) {
  int32_t t_blk = (T_out + BLOCK - 1) / BLOCK * BLOCK;
  int32_t span = (K - 1) * dilation;
  if (stride == 1) return t_blk + span;
  return 2 * (t_blk + span / 2 + 1); /* stride 2: even and odd phases */
}

static inline int8_t relu_if(int8_t v, int32_t relu) { return relu && v < 0 ? (int8_t)0 : v; }

/* dst[0:n] = relu(src[0:n]), with wide stores (see the file comment). */
static void copy_relu(int8_t* dst, const int8_t* src, int32_t n) {
  int32_t i = 0;
#ifdef __C7524__
  for (; i + 32 <= n; i += 32) {
    *(__char32*)(dst + i) = __max(*(const __char32*)(src + i), (__char32)0);
  }
#endif
  for (; i < n; ++i) dst[i] = relu_if(src[i], 1);
}

/* Fill the line buffer for one channel. Stride 1: xpad itself. Stride 2:
 * xpad's even samples, then its odd samples, each half_len long. */
static void fill_line(const int8_t* xc, int32_t T_in, int32_t pad_left, int32_t len, int32_t stride,
                      int32_t relu) {
  if (stride == 1) {
    int32_t start = pad_left < len ? pad_left : len;
    int32_t n = T_in < len - start ? T_in : len - start;
    s_line = s_line_storage + ((32 - (start & 31)) & 31);
    memset(s_line, 0, (size_t)start);
    if (relu) {
      copy_relu(s_line + start, xc, n);
    } else {
      memcpy(s_line + start, xc, (size_t)n);
    }
    memset(s_line + start + n, 0, (size_t)(len - start - n + BLOCK));
    return;
  }
  int32_t half = len / 2;
  s_line = s_line_storage;
  for (int32_t phase = 0; phase < 2; ++phase) {
    int8_t* dst = s_line + phase * half;
    for (int32_t j = 0; j < half; ++j) {
      int32_t i = 2 * j + phase - pad_left;
      dst[j] = (i >= 0 && i < T_in) ? relu_if(xc[i], relu) : (int8_t)0;
    }
  }
  memset(s_line + len, 0, BLOCK);
}

static int32_t check_params(int32_t stride, int32_t dilation, int32_t K, int32_t pad_left,
                            int32_t T_out) {
  if ((stride != 1 && stride != 2) || dilation < 1 || K < 1 || K > C7X_DWCONV1D_MAX_TAPS ||
      pad_left < 0 || T_out < 1 || line_bytes(T_out, K, stride, dilation) > C7X_DWCONV1D_MAX_LINE) {
    return -1;
  }
  return 0;
}

#ifdef __C7524__

static inline __int8 requantize(__int8 a, __float8 vs) {
  __float8 f = __int_to_float(a) * vs;
  f = __max(__min(f, (__float8)127.0f), (__float8)-128.0f);
  return __float_to_int(f);
}

#if defined(__TI_COMPILER_VERSION__) && !defined(C7X_HOST_EMULATION)
#define DWCONV1D_NO_UNROLL _Pragma("UNROLL(1)")
#define DWCONV1D_PREFETCH(p) __prefetch((void*)(p), 0)
#else
#define DWCONV1D_NO_UNROLL
#define DWCONV1D_PREFETCH(p) ((void)(p))
#endif

/* The sixteen accumulators, one per 8 outputs of the block. */
#define DWCONV1D_ACC(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) \
                        X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)
#define DWCONV1D_DECL(i) __int8 a##i = (__int8)0;
#define DWCONV1D_MAC_ONE(i) a##i += __convert_int8(*(const __char8*)(xk_ + 8 * (i))) * wk;
#define DWCONV1D_STORE(i) __vstore_pack_byte((__char8*)(yb + 8 * (i)), requantize(a##i, vs));
#define DWCONV1D_STORE_PART(i)                                                       \
  if (rem > 8 * (i)) {                                                               \
    int32_t n_ = rem - 8 * (i) >= 8 ? 8 : rem - 8 * (i);                             \
    int8_t tmp_[8];                                                                  \
    __vstore_pack_byte((__char8*)tmp_, requantize(a##i, vs));                        \
    for (int32_t j_ = 0; j_ < n_; ++j_) yb[8 * (i) + j_] = tmp_[j_];                 \
  }

/* One tap into the accumulators, reading BLOCK consecutive bytes at p. */
#define DWCONV1D_MAC(p)          \
  do {                           \
    const int8_t* xk_ = (p);     \
    DWCONV1D_ACC(DWCONV1D_MAC_ONE) \
  } while (0)

static int32_t dwconv1d(const void* x_ptr, const void* w_ptr, const void* scale_ptr, void* y_ptr,
                        int32_t C, int32_t T_in, int32_t T_out, int32_t K, int32_t stride,
                        int32_t dilation, int32_t pad_left, int32_t relu) {
  const int8_t* x = (const int8_t*)x_ptr;
  const int8_t* w = (const int8_t*)w_ptr;
  const float* scale = (const float*)scale_ptr;
  int8_t* y = (int8_t*)y_ptr;
  if (check_params(stride, dilation, K, pad_left, T_out) != 0) return -1;
  int32_t len = line_bytes(T_out, K, stride, dilation);

  /* Stride 2: each tap's first byte in the even/odd line, for output 0. */
  int32_t half = len / 2;
  if (stride == 2) {
    for (int32_t k = 0; k < K; ++k) {
      int32_t off = k * dilation;
      s_tap_offset[k] = (off & 1) * half + (off >> 1);
    }
  }

  for (int32_t c = 0; c < C; ++c) {
    const int8_t* xc = x + c * T_in;
    if (c + 1 < C) {
      for (int32_t i = 0; i < T_in; i += 64) DWCONV1D_PREFETCH(xc + T_in + i);
    }
    fill_line(xc, T_in, pad_left, len, stride, relu);
    const int8_t* wc = w + c * K;
    const __float8 vs = (__float8)scale[c];
    int8_t* yc = y + c * T_out;
    for (int32_t t0 = 0; t0 < T_out; t0 += BLOCK) {
      DWCONV1D_ACC(DWCONV1D_DECL)
      const int8_t* xb = s_line + t0;
      /* Three loops so each indexes the line as simply as it can. cl7x
       * pipelines them at II 9 per tap; unrolled, they run out of registers. */
      if (stride == 1 && dilation == 1) {
        DWCONV1D_NO_UNROLL
        for (int32_t k = 0; k < K; ++k) {
          int32_t wv = wc[k];
          __int8 wk = (__int8)wv;
          DWCONV1D_MAC(xb + k);
        }
      } else if (stride == 1) {
        const int8_t* xk = xb;
        DWCONV1D_NO_UNROLL
        for (int32_t k = 0; k < K; ++k) {
          int32_t wv = wc[k];
          __int8 wk = (__int8)wv;
          DWCONV1D_MAC(xk);
          xk += dilation;
        }
      } else {
        DWCONV1D_NO_UNROLL
        for (int32_t k = 0; k < K; ++k) {
          int32_t wv = wc[k];
          __int8 wk = (__int8)wv;
          DWCONV1D_MAC(xb + s_tap_offset[k]);
        }
      }
      int8_t* yb = yc + t0;
      int32_t rem = T_out - t0;
      if (rem >= BLOCK) {
        DWCONV1D_ACC(DWCONV1D_STORE)
      } else {
        DWCONV1D_ACC(DWCONV1D_STORE_PART)
      }
    }
  }
  return 0;
}

#else /* scalar fallback for non-C7524 builds */

static int32_t dwconv1d(const void* x_ptr, const void* w_ptr, const void* scale_ptr, void* y_ptr,
                        int32_t C, int32_t T_in, int32_t T_out, int32_t K, int32_t stride,
                        int32_t dilation, int32_t pad_left, int32_t relu) {
  const int8_t* x = (const int8_t*)x_ptr;
  const int8_t* w = (const int8_t*)w_ptr;
  const float* scale = (const float*)scale_ptr;
  int8_t* y = (int8_t*)y_ptr;
  if (check_params(stride, dilation, K, pad_left, T_out) != 0) return -1;
  for (int32_t c = 0; c < C; ++c) {
    const int8_t* xc = x + c * T_in;
    const int8_t* wc = w + c * K;
    for (int32_t t = 0; t < T_out; ++t) {
      int32_t acc = 0;
      for (int32_t k = 0; k < K; ++k) {
        int32_t i = t * stride + k * dilation - pad_left;
        if (i >= 0 && i < T_in) acc += (int32_t)relu_if(xc[i], relu) * (int32_t)wc[k];
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

extern "C" int32_t c7x_int8_dwconv1d(const void* x, const void* w, const void* scale, void* y,
                                     int32_t C, int32_t T_in, int32_t T_out, int32_t K,
                                     int32_t stride, int32_t dilation, int32_t pad_left) {
  return dwconv1d(x, w, scale, y, C, T_in, T_out, K, stride, dilation, pad_left, 0);
}

extern "C" int32_t c7x_int8_dwconv1d_relu(const void* x, const void* w, const void* scale,
                                          void* y, int32_t C, int32_t T_in, int32_t T_out,
                                          int32_t K, int32_t stride, int32_t dilation,
                                          int32_t pad_left) {
  return dwconv1d(x, w, scale, y, C, T_in, T_out, K, stride, dilation, pad_left, 1);
}
