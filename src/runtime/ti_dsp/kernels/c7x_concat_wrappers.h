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
 * @file c7x_concat_wrappers.h
 * @brief Vectorized int8 channel-axis concatenation with per-input rescaling.
 *
 * Handles the common Inception-module pattern:
 *   dq(x1, s1, z1) | dq(x2, s2, z2) | ... → concat(axis=1) → q(s_out, z_out)
 *
 * Fixed 4-slot signature; set C_i=0 for unused slots (kernel skips them).
 * Transparent fast path: if s_i == s_out && z_i == z_out, uses memcpy.
 * Vectorized path: SE streaming + Q13 integer fixed-point, #ifdef __C7524__.
 */

#ifndef TVM_C7X_CONCAT_WRAPPERS_H_
#define TVM_C7X_CONCAT_WRAPPERS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Concatenate up to 4 int8 NCHW tensors along the channel axis with
 * per-input requantization.
 *
 * Each input i covers C_i channels; all inputs share the same H×W.
 * Set C_i = 0 for unused slots — the kernel skips them entirely.
 *
 * @param in0..in3  Input data pointers (ignored if C_i == 0)
 * @param C0..C3    Channel count per input (set 0 to disable)
 * @param s0..s3    Input quantization scales
 * @param z0..z3    Input zero-points
 * @param out       Output buffer [N=1, C0+C1+C2+C3, H, W]
 * @param HW        H * W (spatial size, same for all inputs)
 * @param s_out     Output quantization scale
 * @param z_out     Output zero-point
 */
int32_t c7x_int8_concat_rescale(
    const void* in0, int32_t C0, float s0, int32_t z0,
    const void* in1, int32_t C1, float s1, int32_t z1,
    const void* in2, int32_t C2, float s2, int32_t z2,
    const void* in3, int32_t C3, float s3, int32_t z3,
    void* out, int32_t HW,
    float s_out, int32_t z_out);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_C7X_CONCAT_WRAPPERS_H_ */
