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
 * @file c7x_rescale.h
 * @brief Standalone int8 QDQ-glue movement kernels: flat rescale and
 * nearest-neighbor 2x upsample.
 *
 * These back FuseQDQToC7xMovement (ti_fuse_qdq_c7x_movement.py), which
 * targets the dequantize -> {reshape | resize2d-nearest} -> quantize glue
 * left as scalar float32 TIR loops by the generic LegalizeOps path.
 */

#ifndef TVM_C7X_RESCALE_H_
#define TVM_C7X_RESCALE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Flat int8 -> int8 Q13 affine rescale: out[i] = quant(dequant(in[i])).
 *
 * out[i] = clamp(round(((in[i] - zx) * sx) / sy) + zy, -128, 127)
 *
 * Shape-agnostic -- operates on a flat buffer of n elements, so it applies
 * unchanged to any dequantize -> <injective, order-preserving movement op
 * (reshape, transparent-shape split/slice)> -> quantize chain, regardless
 * of the tensor's rank. Transparent fast path (sx==sy && zx==zy): memcpy.
 *
 * @param in   Input int8 buffer, n elements
 * @param out  Output int8 buffer, n elements
 * @param n    Element count
 * @param zx   Input zero-point
 * @param sx   Input scale
 * @param zy   Output zero-point
 * @param sy   Output scale
 */
int32_t c7x_int8_rescale(
    const void* in, void* out, int32_t n,
    int32_t zx, float sx, int32_t zy, float sy);

/**
 * Nearest-neighbor 2x spatial upsample, NCHW int8, exact integer factor.
 *
 * out[c, 2h+dh, 2w+dw] = in[c, h, w]  for dh, dw in {0, 1}
 *
 * Pure data movement -- no scale/zero-point parameters. Matches
 * relax.image.resize2d(method="nearest_neighbor",
 * coordinate_transformation_mode="half_pixel", rounding_method="round")
 * for an exact 2x upsample: half_pixel's src index (dst+0.5)*0.5-0.5,
 * rounded, reduces exactly to floor(dst/2) for every dst in range -- i.e.
 * plain 2x2 block replication, verified by direct calculation, not by
 * approximation. Caller must pre-rescale the input to the desired output
 * quantization (e.g. via c7x_int8_rescale or a fused activation kernel
 * targeting that scale directly) before calling this.
 *
 * @param in   Input int8 buffer, C*H*W elements, NCHW
 * @param out  Output int8 buffer, C*2H*2W elements, NCHW
 * @param C    Channel count
 * @param H    Input height
 * @param W    Input width
 */
int32_t c7x_int8_resize2d_nearest2x(
    const void* in, void* out, int32_t C, int32_t H, int32_t W);

/**
 * FPN upsample-concat: SiLU(in1) upsampled 2x nearest, concatenated with a
 * plain rescale of in2, along the channel axis (in1's channels first).
 *
 * out[0:C1, 2h+dh, 2w+dw]  = quant(silu(dequant(in1[c,h,w])))   dh,dw in {0,1}
 * out[C1:C1+C2, h2, w2]    = quant(dequant(in2[c,h2,w2]))
 *
 * in2 is rescaled, NOT re-activated with SiLU: on both real FPN sites this
 * backs (yolov8n/yolo26n), branch 2's own SiLU output is also consumed
 * elsewhere in the graph, so by the time FuseQDQToC7xMovement's pattern
 * runs, FuseQDQToC7xActivation's shared-output handling has already
 * lowered that SiLU to its own c7x_int8_silu call and left a plain
 * dequantize (of that already-SiLU'd, already-int8 result) feeding this
 * concat -- confirmed by direct inspection of the compiled graph, not
 * assumed; see ti_fuse_qdq_c7x_movement.py's pattern-2 docstring for the
 * full account, including why the pattern requires this shape rather than
 * SiLU on both branches (the naively-symmetric version doesn't structurally
 * exist in the graph by the time this pass runs).
 *
 * Both branches target the same (s_out, z_out) directly -- no intermediate
 * QDQ roundtrip. in2 must already be at the upsampled (2H, 2W) spatial size.
 * Single call_extern for the whole composite (dq->sigmoid->multiply->
 * resize2d->A ; dq->B ; concat([A,B],axis=1)->q): chaining this as separate
 * call_te ops per branch was tried and reverted -- an earlier (all-SiLU)
 * version of this composite crashed an unrelated *later* pass's own
 * FuseOpsByPattern call on the real compiled graphs ("Variable ... could
 * not be found in any group", from TVM's OperatorFusor) when matched as
 * two independent SiLU sub-diamonds joined only at the concat, even though
 * it worked fine in isolation on small synthetic graphs. One call_tir per
 * composite is what every other FuseQDQToC7x* lowerer in this codebase
 * already does; this kernel follows that precedent instead of re-relying
 * on multi-step Relax-level chaining.
 *
 * @param in1    Branch 1 (gets upsampled) int8 buffer, C1*H*W elements, NCHW
 * @param C1     Branch 1 channel count
 * @param H      Branch 1 input height
 * @param W      Branch 1 input width
 * @param z1     Branch 1 input zero-point
 * @param s1     Branch 1 input scale
 * @param in2    Branch 2 (skip) int8 buffer, C2*2H*2W elements, NCHW
 * @param C2     Branch 2 channel count
 * @param z2     Branch 2 input zero-point
 * @param s2     Branch 2 input scale
 * @param out    Output int8 buffer, (C1+C2)*2H*2W elements, NCHW
 * @param s_out  Output scale
 * @param z_out  Output zero-point
 */
int32_t c7x_int8_fpn_upsample_concat(
    const void* in1, int32_t C1, int32_t H, int32_t W, int32_t z1, float s1,
    const void* in2, int32_t C2, int32_t z2, float s2,
    void* out, float s_out, int32_t z_out);

/**
 * Same as c7x_int8_fpn_upsample_concat, plus an extra output: branch 1's
 * per-pixel float32 SiLU value at the pre-upsample [C1,H,W] spatial size
 * (i.e. silu(dequant(in1)) before the 2x2 replication + requantize into
 * out). This is the exact float32 SiLU result, NOT quant(silu(...))
 * requantized to the output scale -- so a downstream consumer that needs
 * the float value gets it losslessly, without this kernel's output-scale
 * quantization error.
 *
 * Needed when branch 1's SiLU output is independently consumed elsewhere
 * in the graph too, not just by this FPN concat: FuseOpsByPattern then
 * promotes that shared (float32) value to an extra tuple output of the
 * matched composite (the same "is_tuple_out" situation
 * _ActivationLowerer._lower_single_input already handles for hardswish in
 * ti_fuse_qdq_c7x_activation.py, except here the companion is delivered as
 * exact float32 rather than reconstructed via dequantize) -- confirmed on
 * both real FPN sites this backs (yolov8n/yolo26n), not assumed.
 *
 * @param out1_presize  Output float32 buffer, C1*H*W elements, NCHW (branch
 *                      1's pre-upsample float32 SiLU value)
 */
int32_t c7x_int8_fpn_upsample_concat_ex(
    const void* in1, int32_t C1, int32_t H, int32_t W, int32_t z1, float s1,
    const void* in2, int32_t C2, int32_t z2, float s2,
    void* out, float s_out, int32_t z_out,
    void* out1_presize);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_C7X_RESCALE_H_ */
