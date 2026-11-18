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
 * @file c7x_dwconv1d.h
 * @brief int8 depthwise 1-D convolution with per-channel requantize.
 *
 * Computes, for each channel c and output position t,
 *
 *   acc      = sum_k w[c, k] * xpad[c, t * stride + k * dilation]   (int32)
 *   y[c, t]  = clamp(round_half_even((float)acc * scale[c]), -128, 127)
 *
 * where xpad is x with pad_left zeros before it and zeros after it. This is
 * the arithmetic FuseQDQToInt8Conv2D lowers a symmetric QDQ depthwise conv to
 * (a group_conv2d with int32 output, then astype, multiply, round, clip,
 * astype), so the kernel's output is bit-identical to that path's.
 * FuseQDQToC7xDwConv1d routes 1xK depthwise convs on a [1, C, 1, T] tensor
 * here.
 *
 * Layout: x [C, T_in], w [C, K], scale [C], y [C, T_out], all contiguous.
 */

#ifndef TVM_C7X_DWCONV1D_H_
#define TVM_C7X_DWCONV1D_H_

#include <stdint.h>

/** Longest zero-padded input line (bytes) one channel may need. A channel
 *  needs ceil(T_out / 32) * 32 + (K - 1) * dilation bytes at stride 1, and
 *  2 * (ceil(T_out / 32) * 32 + (K - 1) * dilation / 2 + 1) at stride 2
 *  (line_bytes in c7x_dwconv1d.cpp; FuseQDQToC7xDwConv1d checks the same). */
#define C7X_DWCONV1D_MAX_LINE 16384
/** Most taps a kernel may have. */
#define C7X_DWCONV1D_MAX_TAPS 1024

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief int8 depthwise 1-D conv + per-channel requantize.
 *
 * @param x         Input  [C, T_in], int8
 * @param w         Weights [C, K], int8
 * @param scale     Per-channel requantize multiplier [C], float32
 * @param y         Output [C, T_out], int8
 * @param C         Channels
 * @param T_in      Input length
 * @param T_out     Output length
 * @param K         Taps
 * @param stride    1 or 2
 * @param dilation  >= 1
 * @param pad_left  Zeros before the input; the input is zero past its end
 * @return 0, or -1 for parameters outside the limits above
 */
int32_t c7x_int8_dwconv1d(const void* x, const void* w, const void* scale, void* y, int32_t C,
                          int32_t T_in, int32_t T_out, int32_t K, int32_t stride,
                          int32_t dilation, int32_t pad_left);

#ifdef __cplusplus
}
#endif

#endif /* TVM_C7X_DWCONV1D_H_ */
