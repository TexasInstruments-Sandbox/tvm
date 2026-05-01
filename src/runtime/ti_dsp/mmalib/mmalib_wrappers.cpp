#include "mmalib_wrappers.h"
#include "mmalib.h"

#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------
// Template implementation for matmul across int8/int16
// -----------------------------------------------------------------------

template <typename ElemT, int MmalibDtype, int SatMin, int SatMax>
static int32_t matmul_impl(void* src0, void* src1, void* dst,
                           int32_t M, int32_t K, int32_t N, int32_t shift) {
    if (!src0 || !src1 || !dst) {
        return -1;
    }

    int32_t elem_size = (int32_t)sizeof(ElemT);

    MMALIB_bufParams3D_t src0_params;
    src0_params.data_type = MmalibDtype;
    src0_params.dim_x = (uint32_t)K;
    src0_params.dim_y = (uint32_t)M;
    src0_params.stride_y = K * elem_size;
    src0_params.dim_z = 1;
    src0_params.stride_z = M * K * elem_size;

    MMALIB_bufParams3D_t src1_params;
    src1_params.data_type = MmalibDtype;
    src1_params.dim_x = (uint32_t)N;
    src1_params.dim_y = (uint32_t)K;
    src1_params.stride_y = N * elem_size;
    src1_params.dim_z = 1;
    src1_params.stride_z = K * N * elem_size;

    MMALIB_bufParams3D_t dst_params;
    dst_params.data_type = MmalibDtype;
    dst_params.dim_x = (uint32_t)N;
    dst_params.dim_y = (uint32_t)M;
    dst_params.stride_y = N * elem_size;
    dst_params.dim_z = 1;
    dst_params.stride_z = M * N * elem_size;

    MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_InitArgs init_args;
    memset(&init_args, 0, sizeof(init_args));
    init_args.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
    init_args.shift = (int8_t)shift;
    init_args.activationType = MMALIB_SATURATION;
    init_args.pSatMax = SatMax;
    init_args.pSatMin = SatMin;
    init_args.bTranspose =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLY_IXX_IXX_OXX_B_NON_TRANSPOSED;

    int32_t handle_size =
        MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_getHandleSize(&init_args);
    uint8_t handle_buf[1024] __attribute__((aligned(64)));
    if (handle_size > (int32_t)sizeof(handle_buf)) {
        return -1;
    }

    MMALIB_STATUS status = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_init(
        handle_buf, &src0_params, &src1_params, &dst_params, &init_args);
    if (status != MMALIB_SUCCESS) {
        return (int32_t)status;
    }

    status = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_exec(
        handle_buf, src0, src1, dst);
    return (int32_t)status;
}

// -----------------------------------------------------------------------
// Template implementation for conv2d across int8/int16
//
// Data layouts (all contiguous):
//   input:  NCHW — channels at stride H_in*W_in elements
//   kernel: OIHW — output channels at stride C_in*KH*KW elements
//   output: NCHW — channels at stride H_out*W_out elements
//
// Buffer descriptors:
//   src0 (kernel): 2D, dim_x=C_in*KH*KW, dim_y=C_out
//   src1 (input):  2D, dim_x=H_in*W_in,  dim_y=C_in
//   dst  (output): 3D, dim_x=H_out*W_out, dim_y=C_out, dim_z=1
//
// Exec argument order: (handle, kernel, input, bias, scale, shift, output)
// Bias/scale/shift must be non-NULL; pass zeros for identity.
// -----------------------------------------------------------------------

// MMA_SIZE: maximum output channels per MMALIB exec call for strided conv.
// For stride=1, MMALIB handles any C_out. For stride>1, subMChannels must
// be <= MMA_SIZE. We tile over output channels in chunks of MMA_SIZE.
static constexpr int32_t MMA_SIZE_I8 = 64;
static constexpr int32_t MMA_SIZE_I16 = 32;

template <typename ElemT, int MmalibDtype, int BiasDtype,
          int SatMin, int SatMax, int MmaSize>
static int32_t conv2d_impl(void* input, void* kernel,
                           void* bias, void* scale, void* shift,
                           void* output,
                           int32_t C_in, int32_t H_in, int32_t W_in,
                           int32_t C_out, int32_t KH, int32_t KW,
                           int32_t stride_h, int32_t stride_w,
                           int32_t pad_top, int32_t pad_bottom,
                           int32_t pad_left, int32_t pad_right) {
    if (!input || !kernel || !output) {
        return -1;
    }
    if (C_out > 1024) {
        return -1;
    }

    int32_t elem_size = (int32_t)sizeof(ElemT);
    int32_t H_out = (H_in + pad_top + pad_bottom - KH) / stride_h + 1;
    int32_t W_out = (W_in + pad_left + pad_right - KW) / stride_w + 1;
    int32_t kDim = C_in * KH * KW;
    int32_t inChOffset = H_in * W_in;
    int32_t outChSize = H_out * W_out;
    int32_t bias_elem_size = (BiasDtype == MMALIB_INT64) ? 8 : 4;

    // Determine chunk size for output-channel tiling.
    // stride>1 requires subMChannels <= MmaSize; stride=1 can do all at once.
    int32_t chunk = (stride_h > 1 || stride_w > 1) ?
        (C_out > MmaSize ? MmaSize : C_out) : C_out;

    // Default bias/scale/shift when caller passes NULL
    uint8_t default_bias[4096] __attribute__((aligned(64)));
    uint8_t default_scale[1024] __attribute__((aligned(64)));
    uint8_t default_shift[1024] __attribute__((aligned(64)));
    if (!bias) {
        memset(default_bias, 0, C_out * bias_elem_size);
        bias = default_bias;
    }
    if (!scale) {
        memset(default_scale, 1, C_out);
        scale = default_scale;
    }
    if (!shift) {
        memset(default_shift, 0, C_out);
        shift = default_shift;
    }

    // Process output channels in chunks
    for (int32_t co_base = 0; co_base < C_out; co_base += chunk) {
        int32_t sub = (co_base + chunk <= C_out) ? chunk : (C_out - co_base);

        // Kernel buffer: offset by co_base rows (each row = kDim elements)
        uint8_t* k_ptr = (uint8_t*)kernel + co_base * kDim * elem_size;
        // Bias: offset by co_base elements
        uint8_t* b_ptr = (uint8_t*)bias + co_base * bias_elem_size;
        // Scale/shift: offset by co_base elements
        uint8_t* s_ptr = (uint8_t*)scale + co_base;
        uint8_t* sh_ptr = (uint8_t*)shift + co_base;
        // Output: offset by co_base channels (each channel = outChSize elements)
        uint8_t* o_ptr = (uint8_t*)output + co_base * outChSize * elem_size;

        MMALIB_bufParams2D_t src0_addr;
        src0_addr.data_type = MmalibDtype;
        src0_addr.dim_x = (uint32_t)kDim;
        src0_addr.dim_y = (uint32_t)sub;
        src0_addr.stride_y = kDim * elem_size;

        MMALIB_bufParams2D_t src1_addr;
        src1_addr.data_type = MmalibDtype;
        src1_addr.dim_x = (uint32_t)inChOffset;
        src1_addr.dim_y = (uint32_t)C_in;
        src1_addr.stride_y = inChOffset * elem_size;

        MMALIB_bufParams2D_t src2_addr;
        memset(&src2_addr, 0, sizeof(src2_addr));
        src2_addr.data_type = BiasDtype;
        src2_addr.dim_x = (uint32_t)sub;
        src2_addr.dim_y = 1;
        src2_addr.stride_y = sub * bias_elem_size;

        MMALIB_bufParams1D_t src3_addr;
        src3_addr.data_type = MMALIB_UINT8;
        src3_addr.dim_x = (uint32_t)sub;

        MMALIB_bufParams3D_t dst_addr;
        dst_addr.data_type = MmalibDtype;
        dst_addr.dim_x = (uint32_t)outChSize;
        dst_addr.dim_y = (uint32_t)sub;
        dst_addr.stride_y = outChSize * elem_size;
        dst_addr.dim_z = 1;
        dst_addr.stride_z = sub * outChSize * elem_size;

        MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_InitArgs init_args;
        memset(&init_args, 0, sizeof(init_args));
        init_args.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
        init_args.No = sub;
        init_args.inChOffset = inChOffset;
        init_args.validColsIn = H_in * W_in;
        init_args.validColsPerRowIn = W_in;
        init_args.validRowsIn = H_in;
        init_args.inputPitchPerRow = W_in * elem_size;
        init_args.outputPitchPerRow = W_out * elem_size;
        init_args.inWidth = W_in;
        init_args.maxHeight = H_in;
        init_args.Fr = KH;
        init_args.Fc = KW;
        init_args.strideX = stride_w;
        init_args.strideY = stride_h;
        init_args.dilationX = 1;
        init_args.dilationY = 1;
        init_args.padTop = pad_top;
        init_args.padBottom = pad_bottom;
        init_args.padLeft = pad_left;
        init_args.padRight = pad_right;
        init_args.validColsOutBottom = H_out * W_out;
        init_args.bias = 0;
        init_args.activationType = MMALIB_SATURATION;
        init_args.pSatMin = SatMin;
        init_args.pSatMax = SatMax;
        init_args.mode = MMALIB_LINEAR;
        init_args.subMChannels = sub;
        init_args.numGroupsPerKernel = 1;

        int32_t handle_size =
            MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_getHandleSize(&init_args);
        uint8_t handle_buf[4096] __attribute__((aligned(64)));
        if (handle_size > (int32_t)sizeof(handle_buf)) {
            return -1;
        }

        MMALIB_STATUS status = MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_init(
            handle_buf, &src0_addr, &src1_addr, &src2_addr, &src3_addr,
            &dst_addr, &init_args);
        if (status != MMALIB_SUCCESS) {
            return (int32_t)status;
        }

        MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_ExecInArgs exec_in;
        memset(&exec_in, 0, sizeof(exec_in));
        exec_in.validColsIn = H_in * W_in;
        exec_in.validColsPerRowIn = W_in;
        exec_in.validRowsIn = H_in;
        exec_in.subMChannels = sub;
        exec_in.quantMethod = 1;

        MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_ExecOutArgs exec_out;
        memset(&exec_out, 0, sizeof(exec_out));

        status = MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_exec(
            handle_buf, k_ptr, input, b_ptr, s_ptr, sh_ptr,
            o_ptr, &exec_in, &exec_out);
        if (status != MMALIB_SUCCESS) {
            return (int32_t)status;
        }
    }

    return (int32_t)MMALIB_SUCCESS;
}

// -----------------------------------------------------------------------
// Extern "C" entry points
// -----------------------------------------------------------------------

int32_t mmalib_matmul_i8(void* src0, void* src1, void* dst,
                         int32_t M, int32_t K, int32_t N, int32_t shift) {
    return matmul_impl<int8_t, MMALIB_INT8, -128, 127>(src0, src1, dst, M, K, N, shift);
}

int32_t mmalib_matmul_i16(void* src0, void* src1, void* dst,
                          int32_t M, int32_t K, int32_t N, int32_t shift) {
    return matmul_impl<int16_t, MMALIB_INT16, -32768, 32767>(src0, src1, dst, M, K, N, shift);
}

int32_t mmalib_conv2d_i8(void* input, void* kernel,
                         void* bias, void* scale, void* shift,
                         void* output,
                         int32_t C_in, int32_t H_in, int32_t W_in,
                         int32_t C_out, int32_t KH, int32_t KW,
                         int32_t stride_h, int32_t stride_w,
                         int32_t pad_top, int32_t pad_bottom,
                         int32_t pad_left, int32_t pad_right) {
    return conv2d_impl<int8_t, MMALIB_INT8, MMALIB_INT32, -128, 127, MMA_SIZE_I8>(
        input, kernel, bias, scale, shift, output,
        C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);
}

int32_t mmalib_conv2d_i16(void* input, void* kernel, void* output,
                          int32_t C_in, int32_t H_in, int32_t W_in,
                          int32_t C_out, int32_t KH, int32_t KW,
                          int32_t stride_h, int32_t stride_w,
                          int32_t pad_top, int32_t pad_bottom,
                          int32_t pad_left, int32_t pad_right,
                          int32_t shift) {
    uint8_t shift_buf[1024] __attribute__((aligned(64)));
    memset(shift_buf, (uint8_t)shift, C_out);

    return conv2d_impl<int16_t, MMALIB_INT16, MMALIB_INT64, -32768, 32767, MMA_SIZE_I16>(
        input, kernel, NULL, NULL, shift_buf, output,
        C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);
}
