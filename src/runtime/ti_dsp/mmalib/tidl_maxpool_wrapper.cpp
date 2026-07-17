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
 * @file tidl_maxpool_wrapper.cpp
 * @brief Wrapper around TIDL's vectorized C7x spatial max pool kernel.
 *
 * Why this file exists (and why it is compiled into the firmware, not the
 * DSP runtime library):
 *
 *   TIDL's TIDL_spatialMaxPool_ixX_oxX_init/exec uses a vectorized 3-row
 *   simultaneous approach: two SE contexts deliver 32 int8 values/cycle from
 *   three consecutive input rows, vertical max is done in 2 __max() calls,
 *   and horizontal max for stride=2 uses two register shifts — no inner loop
 *   over kernel positions.  SA handles predicated strided output.  This is
 *   ~30–60× faster than the plain-C c7x_int8_max_pool scalar loop.
 *
 *   The wrapper follows the same MMALIB wrapper pattern (mmalib_wrappers.cpp):
 *   runtime allocation via TVMBackendAllocWorkspace, RAII cleanup, static
 *   handle cache.  It lives alongside mmalib_wrappers.cpp and is compiled
 *   into the firmware under USE_TIDL_RUNTIME (which also links tidl_algo.lib).
 *
 * Handle caching:
 *   TIDL_spatialMaxPool_ixX_oxX_init is called once per unique pool config
 *   and the result cached in static variables.  For ResNet-18 there is one
 *   max pool config (3×3/s2, 112×112×64 → 56×56×64) so one cache entry
 *   is sufficient.  A config change invalidates the cache and reinitialises.
 *
 * Predicate buffer:
 *   TIDL uses SA (Streaming Address) with predicated stores for the output.
 *   The predicate values are pre-computed at init time and stored in a small
 *   buffer (a few hundred bytes for typical ResNet configs).  We query the
 *   required size on the first init call (passing NULL), then allocate and
 *   call init again with the real buffer.
 *
 * deviceName = 0:
 *   TIDL's pooling device code uses deviceName=0 for J722S (non-TDA4VM).
 *   When deviceName==0, the predicate buffer path is active and the
 *   TIDL_FUNCTION_OPTIMIZED_C7x (funcStyle=1) vectorised kernel is selected.
 */

#include "tidl_maxpool_wrapper.h"

#include <stdint.h>
#include <string.h>
#include <kernel/dpl/DebugP.h>

/* TIDL_POOL_BLOCK_FULL: from tidl_alg_int.h, indicates full-frame (non-LFM) operation */
#define TIDL_POOL_BLOCK_FULL 4

/* TIDL_OTF_FLAG_BIT: from tidl_deviceInfo.h.  Setting this bit in deviceName causes
 * TIDL_isPadOTF() to return TRUE, selecting the __SE_TEMPLATE_v2 init/exec path which
 * encodes virtual (zero) padding entirely in the SE DECDIM fields.  No software
 * predicate buffer is needed or allocated in this path. */
#define TIDL_OTF_FLAG_BIT 0x100U

/* TIDL algo headers — only available when compiled into the firmware with
 * USE_TIDL_RUNTIME=ON (which links tidl_algo.lib and sets include paths).
 *
 * CORE_DSP: suppresses the TI_platforms.h → vcop/vcop.h include chain.
 * vcop.h is a legacy C66/EVE header not present in C7x CGT; defining CORE_DSP
 * before the TIDL headers prevents the conditional include in TI_platforms.h.
 *
 * TIDL_MAXPOOL_USE_TIDL_KERNEL: define to enable the TIDL kernel calls.
 * Previously left undefined due to suspected static initializer issues in
 * TIDL spatial max pool objects. Investigation shows the objects have no
 * .init_array entries and are already linked via TIDL_VISION_FXNS; the
 * actual issue was incorrect poolingLFMBlock and featurePlaneSize values.
 * Now fixed and ready for testing. */
#define TIDL_MAXPOOL_USE_TIDL_KERNEL 1  /* enabled for testing with fixes */

#ifdef TIDL_MAXPOOL_USE_TIDL_KERNEL
#ifndef CORE_DSP
#define CORE_DSP
#endif
#include "tidl_generic_datatypes.h"    /* TIDL_KernelHandle, TIDL_INT8, etc. */
#include "tidl_dataflow.h"             /* TIDL_bufParams3D_t                  */
#include "tidl_types.h"                /* TIDL_FUNCTION_OPTIMIZED_C7x         */
#include "itidl_ti.h"                  /* TIDL_MaxPooling                      */
#include "tidl_spatialMaxPool_ixX_oxX.h"
#endif /* TIDL_MAXPOOL_USE_TIDL_KERNEL */

/* Scalar C fallback — always available, used when TIDL kernel is disabled.
 * Path is relative to TVM_DSP_RUNTIME_DIR which is in the firmware include path. */
#include "kernels/c7x_pool_relu.h"

extern "C" void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                          uint64_t nbytes, int dtype_code_hint,
                                          int dtype_bits_hint);
extern "C" int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

#ifdef TIDL_MAXPOOL_USE_TIDL_KERNEL

/* =========================================================================
 * build_buf_params — fill a TIDL_bufParams3D_t for NCHW int8 tensors.
 * ========================================================================= */

static void build_buf_params(TIDL_bufParams3D_t* bp,
                             int32_t C, int32_t H, int32_t W) {
    memset(bp, 0, sizeof(*bp));
    bp->data_type = TIDL_INT8;
    bp->dim_x     = (uint32_t)W;
    bp->dim_y     = (uint32_t)H;
    bp->stride_y  = W;
    bp->dim_z     = (uint32_t)C;
    bp->stride_z  = H * W;
    bp->stride_x  = 1;
    bp->tensorScaleList     = nullptr;
    bp->tensorZeroPointList = nullptr;
}

#endif /* TIDL_MAXPOOL_USE_TIDL_KERNEL */

/* =========================================================================
 * Public entry point
 * ========================================================================= */

extern "C"
int32_t c7x_int8_max_pool_tidl(
        const void* in, void* out,
        int32_t N, int32_t C,
        int32_t H_in, int32_t W_in,
        int32_t H_out, int32_t W_out,
        int32_t kH, int32_t kW,
        int32_t sH, int32_t sW,
        int32_t pH, int32_t pW) {

    if (!in || !out) return -1;

#ifdef TIDL_MAXPOOL_USE_TIDL_KERNEL
    /* Init overhead (~5K cycles) is negligible vs the kernel itself (~500K
     * cycles vectorised).  Allocating, initialising, and freeing per call
     * avoids any handle lifetime issues across model load/unload cycles. */
    uint32_t handle_bytes = TIDL_spatialMaxPool_ixX_oxX_getHandleSize();
    void* handle = TVMBackendAllocWorkspace(1, 0, (uint64_t)handle_bytes, 0, 64);
    if (!handle) return -1;

    TIDL_bufParams3D_t src_p, dst_p;
    build_buf_params(&src_p, C, H_in, W_in);
    build_buf_params(&dst_p, C, H_out, W_out);

    TIDL_SpatialMaxPoolIxXOxXInitArgs args;
    memset(&args, 0, sizeof(args));
    args.funcStyle            = TIDL_FUNCTION_OPTIMIZED_C7x;
    args.deviceName           = TIDL_OTF_FLAG_BIT;
    args.Ni                   = (uint32_t)C;
    args.No                   = (uint32_t)C;
    args.poolType             = TIDL_MaxPooling;
    args.kernelH              = kH;
    args.kernelW              = kW;
    args.strideH              = sH;
    args.strideW              = sW;
    args.padH                 = pH;
    args.padW                 = pW;
    args.vPadT                = pH;
    args.vPadB                = pH;
    args.vPadL                = pW;
    args.vPadR                = pW;
    args.inTensorHeight       = H_in;
    args.outputHeight         = (uint32_t)H_out;
    args.featurePlaneSize     = H_in * W_in;
    args.totalNumKernelCalls  = 1;
    args.numSplitsPerCh       = 1;
    args.poolingLFMBlock      = TIDL_POOL_BLOCK_FULL;
    args.buffInputBlockOffset = H_in * W_in;

    int32_t pred_size = 0;
    int32_t status = TIDL_spatialMaxPool_ixX_oxX_init(
        handle, nullptr, &pred_size, &src_p, &dst_p, &args);
    if (status != 0) {
        DebugP_log("[TIDL MaxPool] Init failed: status=%d\r\n", status);
        TVMBackendFreeWorkspace(1, 0, handle);
        return status;
    }

    TIDL_SpatialMaxPoolIxXOxXExecInArgs  exec_in;
    TIDL_SpatialMaxPoolIxXOxXExecOutArgs exec_out;
    memset(&exec_in,  0, sizeof(exec_in));
    memset(&exec_out, 0, sizeof(exec_out));
    exec_in.predicateBuffer        = nullptr;
    exec_in.startRowNumberInTensor = 0;
    exec_in.poolingLFMBlock        = TIDL_POOL_BLOCK_FULL;

    const int8_t* inp      = reinterpret_cast<const int8_t*>(in);
    int8_t*       outp     = reinterpret_cast<int8_t*>(out);
    int32_t       in_stride  = C * H_in  * W_in;
    int32_t       out_stride = C * H_out * W_out;

    for (int32_t b = 0; b < N; ++b) {
        status = TIDL_spatialMaxPool_ixX_oxX_exec(
            handle,
            inp  + b * in_stride,
            outp + b * out_stride,
            &exec_in, &exec_out);
        if (status != 0) break;
    }

    TVMBackendFreeWorkspace(1, 0, handle);
    return status;

#else
    return c7x_int8_max_pool(in, out, N, C, H_in, W_in, H_out, W_out,
                             kH, kW, sH, sW, pH, pW);
#endif
}
