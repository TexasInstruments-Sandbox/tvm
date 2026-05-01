#ifndef TVM_RUNTIME_TI_DSP_MMALIB_WRAPPERS_H_
#define TVM_RUNTIME_TI_DSP_MMALIB_WRAPPERS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// DLOAD export: firmware exposes these symbols to dynamically loaded modules.
#if defined(TVM_DSP_TARGET_C7X) && defined(__TI_COMPILER_VERSION__)
#define TVM_MMALIB_EXPORT __declspec(dllexport)
#else
#define TVM_MMALIB_EXPORT
#endif

// Matrix multiply: C[M,N] = A[M,K] x B[K,N]
// Accumulator >> shift, saturated to output dtype.
// Dimensions: multiples of 64 (int8) or 32 (int16).

TVM_MMALIB_EXPORT
int32_t mmalib_matmul_i8(void* src0, void* src1, void* dst,
                         int32_t M, int32_t K, int32_t N, int32_t shift);

TVM_MMALIB_EXPORT
int32_t mmalib_matmul_i16(void* src0, void* src1, void* dst,
                          int32_t M, int32_t K, int32_t N, int32_t shift);

// Conv2d with per-channel bias/scale/shift:
//   output[ch] = saturate((conv_accum[ch] + bias[ch]) * scale[ch] >> shift[ch])
//
// Layouts: input NCHW (N=1), kernel OIHW, output NCHW.
// groups=1, dilation=1.
// bias: int32[C_out] for i8, int64[C_out] for i16. NULL → zero.
// scale: uint8[C_out]. NULL → 1.
// shift: uint8[C_out]. NULL → 0.

TVM_MMALIB_EXPORT
int32_t mmalib_conv2d_i8(void* input, void* kernel,
                         void* bias, void* scale, void* shift,
                         void* output,
                         int32_t C_in, int32_t H_in, int32_t W_in,
                         int32_t C_out, int32_t KH, int32_t KW,
                         int32_t stride_h, int32_t stride_w,
                         int32_t pad_top, int32_t pad_bottom,
                         int32_t pad_left, int32_t pad_right);

// Compatibility entry point (no bias/scale, uniform shift).
TVM_MMALIB_EXPORT
int32_t mmalib_conv2d_i16(void* input, void* kernel, void* output,
                          int32_t C_in, int32_t H_in, int32_t W_in,
                          int32_t C_out, int32_t KH, int32_t KW,
                          int32_t stride_h, int32_t stride_w,
                          int32_t pad_top, int32_t pad_bottom,
                          int32_t pad_left, int32_t pad_right,
                          int32_t shift);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_TI_DSP_MMALIB_WRAPPERS_H_
