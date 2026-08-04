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
 * @file c7x_softmax.cpp
 * @brief c7x_int8_dfl_softmax -- fused dequantize/transpose/softmax/quantize
 * for the YOLOv8 DFL head. See c7x_softmax.h for the exact shape this backs.
 *
 * Scalar-only (no #ifdef __C7524__ vectorized path): this is the low-ROI
 * side of Step 3 (6.6% of yolov8n, ~1.3% of yolo26n's unrelated attention
 * softmax -- see yolo_head_qdq_movement_fusion.md's Motivation table), so a
 * correct, portable 3-pass reduction was judged proportionate over a full
 * SE-vectorized rewrite. Fusing the 4 ops into one call is a real win
 * on its own: it eliminates the permute_dims's own physical transpose
 * (a full extra tensor materialization) in addition to fusing the
 * dequantize/softmax/quantize scalar loops that FuseTIR would otherwise
 * leave as separate float32 loops with per-element expf() calls.
 *
 * Algorithm, per (b, a) group of B*A independent softmax reductions (each
 * over K values, for each of N anchors):
 *   1. max[n]   = max over k of dq_f(in[b,a,k,n])
 *   2. exp[k,n] = exp(dq_f(in[b,a,k,n]) - max[n]); sum[n] = sum over k of exp[k,n]
 *   3. out[b,k,a,n] = rq_f(exp[k,n] / sum[n])
 *
 * exp() is computed exactly once per element (cached in a K*N-sized scratch
 * buffer in pass 2, read back in pass 3) rather than twice -- an earlier
 * version recomputed it in pass 3 to avoid the K*N buffer, but expf() turned
 * out to dominate this kernel's cost enough on real hardware (measured on
 * yolov8n's actual DFL shape) that halving the call count was worth the
 * extra scratch allocation. Three float scratch buffers (max/sum sized N,
 * exp sized K*N) are allocated once for the largest group and reused across
 * all B*A groups.
 */

#include "c7x_softmax.h"

#include <math.h>
#include <stdint.h>

#include "c7x_qdq_common.h"

extern "C" void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                           int dtype_code_hint, int dtype_bits_hint);
extern "C" int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

namespace {

struct Workspace {
    void* ptr = nullptr;
    ~Workspace() {
        if (ptr) TVMBackendFreeWorkspace(1, 0, ptr);
    }
    float* alloc_floats(int64_t count) {
        ptr = TVMBackendAllocWorkspace(1, 0, static_cast<uint64_t>(count) * sizeof(float), 2, 32);
        return static_cast<float*>(ptr);
    }
};

}  // namespace

extern "C"
int32_t c7x_int8_dfl_softmax(
        const void* in, void* out,
        int32_t B, int32_t A, int32_t K, int32_t N,
        int32_t zx, float sx, int32_t zy, float sy) {
    const int8_t* p = (const int8_t*)in;
    int8_t* q = (int8_t*)out;

    Workspace max_ws;
    Workspace sum_ws;
    Workspace exp_ws;
    float* maxv = max_ws.alloc_floats(N);
    float* sumv = sum_ws.alloc_floats(N);
    float* expv = exp_ws.alloc_floats((int64_t)K * N);
    if (!maxv || !sumv || !expv) return -1;

    const int64_t KN = (int64_t)K * N;
    const int64_t AN = (int64_t)A * N;

    for (int32_t b = 0; b < B; b++) {
        for (int32_t a = 0; a < A; a++) {
            const int8_t* grp_in = p + ((int64_t)b * A + a) * KN;
            int8_t* grp_out = q + (int64_t)b * K * AN + (int64_t)a * N;

            for (int32_t n = 0; n < N; n++) maxv[n] = -3.402823e38f;
            for (int32_t k = 0; k < K; k++) {
                const int8_t* row = grp_in + (int64_t)k * N;
                for (int32_t n = 0; n < N; n++) {
                    float xf = dq_f(row[n], zx, sx);
                    if (xf > maxv[n]) maxv[n] = xf;
                }
            }

            /* exp() is computed exactly once per element here and cached in
             * expv -- pass 3 below reads it back rather than recomputing
             * expf() a second time, halving the transcendental call count
             * (measured: this cut the kernel's hardware cycle cost by ~2x,
             * see yolo_head_qdq_movement_fusion.md's Performance section). */
            for (int32_t n = 0; n < N; n++) sumv[n] = 0.0f;
            for (int32_t k = 0; k < K; k++) {
                const int8_t* row = grp_in + (int64_t)k * N;
                float* erow = expv + (int64_t)k * N;
                for (int32_t n = 0; n < N; n++) {
                    float xf = dq_f(row[n], zx, sx);
                    float e = expf(xf - maxv[n]);
                    erow[n] = e;
                    sumv[n] += e;
                }
            }

            for (int32_t k = 0; k < K; k++) {
                const float* erow = expv + (int64_t)k * N;
                int8_t* orow = grp_out + (int64_t)k * AN;
                for (int32_t n = 0; n < N; n++) {
                    float sm = erow[n] / sumv[n];
                    orow[n] = rq_f(sm, zy, sy);
                }
            }
        }
    }

    return 0;
}
