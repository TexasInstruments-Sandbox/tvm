/**
 * @file mmalib_wrappers.h
 * @brief C-callable wrappers around TI's MMALIB accelerated kernels.
 *
 * These wrappers provide a simplified, stable ABI for invoking MMALIB's
 * matrix-multiply and convolution routines from TVM-generated code running
 * on the C7x DSP. They handle buffer descriptor setup, handle allocation,
 * output-channel tiling, and NULL-defaulting of optional parameters internally.
 *
 * Linkage:
 *   - On C7x firmware (DLOAD), symbols are exported via __declspec(dllexport)
 *     so dynamically loaded modules can resolve them at runtime.
 *   - On host emulation builds, they link as normal C functions.
 *
 * Return convention:
 *   All functions return 0 (MMALIB_SUCCESS) on success, or a negative/non-zero
 *   MMALIB_STATUS error code on failure. -1 indicates a precondition violation
 *   (NULL pointer, buffer overflow, etc.) caught before calling MMALIB.
 *
 * Thread safety:
 *   These functions are NOT thread-safe. Each uses stack-allocated handle buffers
 *   and assumes exclusive access to the MMA hardware during execution.
 */

#ifndef TVM_RUNTIME_TI_DSP_MMALIB_WRAPPERS_H_
#define TVM_RUNTIME_TI_DSP_MMALIB_WRAPPERS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DLOAD export macro: on C7x firmware, marks symbols for dynamic linking so
 * that modules loaded at runtime via the DLOAD mechanism can resolve them.
 * On host emulation or non-TI compilers, this is a no-op.
 */
#if defined(TVM_DSP_TARGET_C7X) && defined(__TI_COMPILER_VERSION__)
#define TVM_MMALIB_EXPORT __declspec(dllexport)
#else
#define TVM_MMALIB_EXPORT
#endif

/* =========================================================================
 * Matrix Multiply (no bias)
 * =========================================================================
 * C[M,N] = saturate(A[M,K] × B[K,N] >> shift)
 *
 * Layout: all matrices are row-major, contiguous.
 * Alignment: all pointers must be 64-byte aligned.
 * Dimension constraints:
 *   - int8:  M, K, N must be multiples of 64
 *   - int16: M, K, N must be multiples of 32
 * ========================================================================= */

/**
 * @brief Int8 matrix multiply: C = saturate_i8(A × B >> shift)
 * @param src0  Input matrix A [M, K], int8, 64-byte aligned
 * @param src1  Input matrix B [K, N], int8, 64-byte aligned
 * @param dst   Output matrix C [M, N], int8, 64-byte aligned
 * @param M     Number of rows in A / rows in C
 * @param K     Inner dimension (columns of A, rows of B)
 * @param N     Number of columns in B / columns in C
 * @param shift Right-shift applied to 32-bit accumulator before saturation
 * @return 0 on success, non-zero MMALIB error code on failure
 */
TVM_MMALIB_EXPORT
int32_t mmalib_matmul_i8(void* src0, void* src1, void* dst,
                         int32_t M, int32_t K, int32_t N, int32_t shift);

/**
 * @brief Int16 matrix multiply: C = saturate_i16(A × B >> shift)
 * @param src0  Input matrix A [M, K], int16, 64-byte aligned
 * @param src1  Input matrix B [K, N], int16, 64-byte aligned
 * @param dst   Output matrix C [M, N], int16, 64-byte aligned
 * @param M     Number of rows in A / rows in C
 * @param K     Inner dimension (columns of A, rows of B)
 * @param N     Number of columns in B / columns in C
 * @param shift Right-shift applied to 64-bit accumulator before saturation
 * @return 0 on success, non-zero MMALIB error code on failure
 */
TVM_MMALIB_EXPORT
int32_t mmalib_matmul_i16(void* src0, void* src1, void* dst,
                          int32_t M, int32_t K, int32_t N, int32_t shift);

/* =========================================================================
 * 2D Convolution (standard, groups=1)
 * =========================================================================
 * Per-channel quantized convolution:
 *   output[ch] = saturate((conv_accum[ch] + bias[ch]) * scale[ch] >> shift[ch])
 *
 * Layouts (all contiguous, N=1):
 *   input:  NCHW — [1, C_in, H_in, W_in]
 *   kernel: OIHW — [C_out, C_in, KH, KW]
 *   output: NCHW — [1, C_out, H_out, W_out]
 *
 * Constraints: groups=1, dilation=1, C_out <= 1024.
 * For stride > 1, output channels are tiled in chunks of 64 (i8) or 32 (i16).
 * ========================================================================= */

/**
 * @brief Int8 2D convolution with per-channel bias, scale, and shift.
 * @param input   Input tensor [1, C_in, H_in, W_in], int8, 64-byte aligned
 * @param kernel  Weight tensor [C_out, C_in, KH, KW], int8, 64-byte aligned
 * @param bias    Per-channel bias [C_out], int32. NULL defaults to zero.
 * @param scale   Per-channel scale [C_out], uint8. NULL defaults to 1.
 * @param shift   Per-channel shift [C_out], uint8. NULL defaults to 0.
 * @param output  Output tensor [1, C_out, H_out, W_out], int8, 64-byte aligned
 * @param C_in    Number of input channels
 * @param H_in    Input spatial height
 * @param W_in    Input spatial width
 * @param C_out   Number of output channels (max 1024)
 * @param KH      Kernel height
 * @param KW      Kernel width
 * @param stride_h  Vertical stride
 * @param stride_w  Horizontal stride
 * @param pad_top    Top padding (zero-fill)
 * @param pad_bottom Bottom padding
 * @param pad_left   Left padding
 * @param pad_right  Right padding
 * @return 0 on success, non-zero MMALIB error code on failure
 */
TVM_MMALIB_EXPORT
int32_t mmalib_conv2d_i8(void* input, void* kernel,
                         void* bias, void* scale, void* shift,
                         void* output,
                         int32_t C_in, int32_t H_in, int32_t W_in,
                         int32_t C_out, int32_t KH, int32_t KW,
                         int32_t stride_h, int32_t stride_w,
                         int32_t pad_top, int32_t pad_bottom,
                         int32_t pad_left, int32_t pad_right);

/**
 * @brief Int16 2D convolution with uniform shift (compatibility entry point).
 *
 * Simplified interface: no per-channel bias/scale; a single shift value is
 * broadcast to all output channels. Bias defaults to zero, scale to 1.
 *
 * @param input   Input tensor [1, C_in, H_in, W_in], int16, 64-byte aligned
 * @param kernel  Weight tensor [C_out, C_in, KH, KW], int16, 64-byte aligned
 * @param output  Output tensor [1, C_out, H_out, W_out], int16, 64-byte aligned
 * @param C_in    Number of input channels
 * @param H_in    Input spatial height
 * @param W_in    Input spatial width
 * @param C_out   Number of output channels (max 1024)
 * @param KH      Kernel height
 * @param KW      Kernel width
 * @param stride_h  Vertical stride
 * @param stride_w  Horizontal stride
 * @param pad_top    Top padding
 * @param pad_bottom Bottom padding
 * @param pad_left   Left padding
 * @param pad_right  Right padding
 * @param shift   Uniform right-shift applied to all channels
 * @return 0 on success, non-zero MMALIB error code on failure
 */
TVM_MMALIB_EXPORT
int32_t mmalib_conv2d_i16(void* input, void* kernel, void* output,
                          int32_t C_in, int32_t H_in, int32_t W_in,
                          int32_t C_out, int32_t KH, int32_t KW,
                          int32_t stride_h, int32_t stride_w,
                          int32_t pad_top, int32_t pad_bottom,
                          int32_t pad_left, int32_t pad_right,
                          int32_t shift);

/* =========================================================================
 * Depthwise Convolution (groups == channels)
 * =========================================================================
 * Each input channel is convolved independently with its own KH×KW filter.
 * Uses MMALIB's column-based highPrecision kernel which performs runtime
 * weight reordering internally.
 *
 * Layouts (all contiguous, N=1):
 *   input:  NCHW — [1, channels, H_in, W_in]
 *   output: NCHW — [1, channels, H_out, W_out]
 *   weights: natural order [num_groups, 1, KH, KW] — reordered at runtime
 *
 * Constraints:
 *   - Supported kernels: 3×3, 5×5, 7×7
 *   - Strides: 1 or 2
 *   - Dilation: 1 only
 *   - num_groups == channels (true depthwise)
 * ========================================================================= */

/**
 * @brief Int8 depthwise convolution with per-group bias/scale/shift.
 * @param input    Input tensor [1, channels, H_in, W_in], int8, 64-byte aligned
 * @param reordered_weights  Weight tensor [num_groups, 1, KH, KW] in natural
 *                           order — will be reordered internally via MMALIB API
 * @param bias     Per-group bias [num_groups], int32. NULL defaults to zero.
 * @param scale    Per-group scale [num_groups], uint8. NULL defaults to 1.
 * @param shift    Per-group shift [num_groups], uint8. NULL defaults to 0.
 * @param output   Output tensor [1, channels, H_out, W_out], int8, 64-byte aligned
 * @param channels Number of input/output channels
 * @param H_in     Input spatial height
 * @param W_in     Input spatial width
 * @param KH       Kernel height (3, 5, or 7)
 * @param KW       Kernel width (3, 5, or 7)
 * @param stride_h Vertical stride (1 or 2)
 * @param stride_w Horizontal stride (1 or 2)
 * @param pad_top    Top padding (zero-fill)
 * @param pad_bottom Bottom padding
 * @param pad_left   Left padding
 * @param pad_right  Right padding
 * @param num_groups Number of groups (must equal channels for depthwise)
 * @return 0 on success, non-zero MMALIB error code on failure
 */
TVM_MMALIB_EXPORT
int32_t mmalib_depthwise_conv2d_i8(void* input, void* reordered_weights,
                                   void* bias, void* scale, void* shift,
                                   void* output,
                                   int32_t channels, int32_t H_in, int32_t W_in,
                                   int32_t KH, int32_t KW,
                                   int32_t stride_h, int32_t stride_w,
                                   int32_t pad_top, int32_t pad_bottom,
                                   int32_t pad_left, int32_t pad_right,
                                   int32_t num_groups);

/* =========================================================================
 * Matrix Multiply with Bias (fully-connected layer)
 * =========================================================================
 * Quantized FC layer:
 *   output[m,n] = sat_i8((sum_k(input[m,k] * weight[n,k]) + bias[n])
 *                         * scale[n] >> shift[n])
 *
 * Weight layout: [N, K] row-major; transposed internally (bTranspose=1).
 * This matches the standard PyTorch nn.Linear weight convention.
 *
 * Dimension constraints: K and N must be multiples of 64.
 * ========================================================================= */

/**
 * @brief Int8 matrix multiply with per-column bias, scale, and shift.
 *
 * Computes a fully-connected (dense) layer with quantized output. The weight
 * matrix is stored in [N, K] layout and transposed internally by MMALIB.
 *
 * @param input   Activation matrix A [M, K], int8, 64-byte aligned
 * @param weights Weight matrix B [N, K], int8, 64-byte aligned (transposed internally)
 * @param bias    Per-output bias [N], int32. NULL defaults to zero.
 * @param scale   Per-output scale [N], uint8. NULL defaults to 1.
 * @param shift   Per-output shift [N], uint8. NULL defaults to 0.
 * @param output  Output matrix C [M, N], int8, 64-byte aligned
 * @param M       Batch size (number of rows in input)
 * @param K       Input features (must be multiple of 64)
 * @param N       Output features (must be multiple of 64)
 * @return 0 on success, non-zero MMALIB error code on failure
 */
TVM_MMALIB_EXPORT
int32_t mmalib_matmul_bias_i8(void* input, void* weights,
                              void* bias, void* scale, void* shift,
                              void* output,
                              int32_t M, int32_t K, int32_t N);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_TI_DSP_MMALIB_WRAPPERS_H_
