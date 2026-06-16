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
 * \file tvm_int8_residual_add.c
 * \brief Integer residual add with requantization for C7x.
 *
 * Replaces the float-domain sequence:
 *   dequantize(x) + dequantize(skip) -> [relu] -> quantize(out)
 *
 * With fixed-point integer arithmetic:
 *   out[i] = sat_i8((((x[i]-zp_x)*M_x + (skip[i]-zp_skip)*M_skip) >> shift) + zp_out)
 *
 * Parameters are packed in a 16-byte buffer:
 *   [0..3]   M_x      (int32) - multiplier for input x
 *   [4..7]   M_skip   (int32) - multiplier for skip connection
 *   [8..11]  shift    (int32) - right-shift for requantization
 *   [12]     zp_x     (int8)  - zero point for input x
 *   [13]     zp_skip  (int8)  - zero point for skip input
 *   [14]     zp_out   (int8)  - zero point for output
 *   [15]     reserved
 */

#include <stdint.h>

/*
 * tvm_int16_residual_add_relu — int16 variant of the residual add kernel.
 *
 * Identical algorithm to tvm_int8_residual_add_relu; only the element type
 * and saturation range differ.  The packed params layout is unchanged:
 *   [0..3]   M_x      (int32)
 *   [4..7]   M_skip   (int32)
 *   [8..11]  shift    (int32)
 *   [12]     zp_x     (int8)  — always 0 for symmetric int16 quant
 *   [13]     zp_skip  (int8)  — always 0 for symmetric int16 quant
 *   [14]     zp_out   (int8)  — always 0 for symmetric int16 quant
 *   [15]     reserved
 *
 * Saturation: clamp to [-32768, 32767] instead of [-128, 127].
 *
 * Accumulator: int64_t.  int16 inputs × int32 M values can produce up to
 * 32767 × 2^31 ≈ 7×10^13 before shifting, which overflows int32.  Using
 * int64 for the accumulator prevents silent wrap-around when
 * (x_scale + skip_scale) / o_scale > 2.0.  The final shifted result fits
 * comfortably in int32 before clamping to int16.
 */
int32_t tvm_int16_residual_add_relu(
    const void* x_ptr, const void* skip_ptr,
    const void* params, void* output_ptr,
    int32_t num_elements, int32_t has_relu)
{
    const int16_t* x    = (const int16_t*)x_ptr;
    const int16_t* skip = (const int16_t*)skip_ptr;
    int16_t* output     = (int16_t*)output_ptr;
    const int32_t* p32  = (const int32_t*)params;
    const int8_t*  p8   = (const int8_t*)params;

    int32_t M_x    = p32[0];
    int32_t M_skip = p32[1];
    int32_t shift  = p32[2];
    int32_t zp_x    = (int32_t)p8[12];
    int32_t zp_skip = (int32_t)p8[13];
    int32_t zp_out  = (int32_t)p8[14];

    int32_t i;
    for (i = 0; i < num_elements; i++) {
        /* int64 accumulator: 32767 * M (up to 2^31) * 2 operands can exceed INT32_MAX */
        int64_t acc = ((int64_t)x[i] - zp_x) * (int64_t)M_x
                    + ((int64_t)skip[i] - zp_skip) * (int64_t)M_skip;
        int32_t result = (int32_t)(acc >> shift) + zp_out;
        if (has_relu && result < 0) {
            result = 0;
        }
        if (result < -32768) result = -32768;
        if (result >  32767) result =  32767;
        output[i] = (int16_t)result;
    }
    return 0;
}

int32_t tvm_int8_residual_add_relu(
    const void* x_ptr, const void* skip_ptr,
    const void* params, void* output_ptr,
    int32_t num_elements, int32_t has_relu)
{
    const int8_t* x = (const int8_t*)x_ptr;
    const int8_t* skip = (const int8_t*)skip_ptr;
    int8_t* output = (int8_t*)output_ptr;
    const int32_t* p32 = (const int32_t*)params;
    const int8_t* p8 = (const int8_t*)params;

    int32_t M_x    = p32[0];
    int32_t M_skip = p32[1];
    int32_t shift  = p32[2];
    int32_t zp_x    = (int32_t)p8[12];
    int32_t zp_skip = (int32_t)p8[13];
    int32_t zp_out  = (int32_t)p8[14];

    int32_t i;
    for (i = 0; i < num_elements; i++) {
        int32_t acc = ((int32_t)x[i] - zp_x) * M_x
                    + ((int32_t)skip[i] - zp_skip) * M_skip;
        int32_t result = (acc >> shift) + zp_out;
        if (has_relu && result < 0) {
            result = 0;
        }
        if (result < -128) result = -128;
        if (result > 127) result = 127;
        output[i] = (int8_t)result;
    }
    return 0;
}
