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
 * @file c7x_softmax.h
 * @brief Fused dequantize -> transpose -> softmax -> quantize for YOLOv8's
 * DFL (Distribution Focal Loss) head.
 *
 * The DFL head's real compiled shape (see
 * ti_fuse_qdq_c7x_activation.py's DFL softmax pattern for the exact match)
 * is always:
 *
 *   dq(x[B,A,K,N]) -> permute_dims(axes=[0,2,1,3]) -> softmax(axis=1)
 *     -> quantize -> [B,K,A,N]
 *
 * i.e. softmax reduces over the pre-permute axis 2 ("K", the reg_max
 * distribution bins -- architecturally fixed at 16 for every YOLOv8/v5
 * size), for each of the B*A independent (batch, box-coordinate) groups,
 * over N anchors. Rather than materializing the permuted tensor and then
 * reducing over its new axis 1, this kernel reduces directly over the
 * pre-permute memory layout and writes straight to the post-permute int8
 * output -- fusing away both the transpose's own data movement and the
 * surrounding dequantize/quantize scalar loops into one call.
 */

#ifndef TVM_C7X_SOFTMAX_H_
#define TVM_C7X_SOFTMAX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* in:  int8[B][A][K][N], single scalar (zx, sx) dequant.
 * out: int8[B][K][A][N], single scalar (zy, sy) requant.
 * Softmax normalizes over K, independently for each of the B*A*N
 * (batch, box-coord, anchor) positions. */
int32_t c7x_int8_dfl_softmax(
    const void* in, void* out,
    int32_t B, int32_t A, int32_t K, int32_t N,
    int32_t zx, float sx, int32_t zy, float sy);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_C7X_SOFTMAX_H_ */
