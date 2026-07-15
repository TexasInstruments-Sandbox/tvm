/**
 * @file mmalib_wrappers.cpp
 * @brief Implementation of C-callable MMALIB wrapper functions.
 *
 * This file bridges TVM-generated code and TI's MMALIB library by providing
 * simplified entry points that handle:
 *   1. MMALIB buffer descriptor construction from flat dimension parameters
 *   2. Dynamic buffer allocation via TVMBackendAllocWorkspace (no stack arrays)
 *   3. Output-channel tiling for strided convolutions (MMA HW limitation)
 *   4. NULL-defaulting of optional bias/scale/shift to identity values
 *   5. Runtime weight reordering for depthwise convolution
 *
 * The C++ templates (matmul_impl, conv2d_impl) are internal and instantiated
 * only for int8/int16. The extern "C" entry points at the bottom are the
 * public API exported to dynamically loaded modules via DLOAD.
 *
 * Memory: all temporary buffers are allocated via TVMBackendAllocWorkspace
 * (64-byte aligned DDR) and freed automatically via RAII. No stack arrays.
 */

#include "mmalib_wrappers.h"
#include "mmalib.h"

#include <stdlib.h>
#include <string.h>

extern "C" void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                          uint64_t nbytes, int dtype_code_hint,
                                          int dtype_bits_hint);
extern "C" int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

struct Workspace {
    void* ptr = nullptr;
    ~Workspace() { if (ptr) TVMBackendFreeWorkspace(1, 0, ptr); }
    void* alloc(int32_t n) { ptr = TVMBackendAllocWorkspace(1, 0, (uint64_t)n, 0, 8); return ptr; }
};

/* =========================================================================
 * Template: matmul_impl — shared logic for int8/int16 matrix multiply
 *
 * Instantiated as mmalib_matmul_i8 and mmalib_matmul_i16.
 * Uses MMALIB_LINALG_matrixMatrixMultiply (non-bias variant).
 * B is NOT transposed; both A and B are in natural row-major order.
 * ========================================================================= */

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
    Workspace handle;
    if (!handle.alloc(handle_size)) return -1;

    MMALIB_STATUS status = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_init(
        handle.ptr, &src0_params, &src1_params, &dst_params, &init_args);
    if (status != MMALIB_SUCCESS) {
        return (int32_t)status;
    }

    status = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_exec(
        handle.ptr, src0, src1, dst);
    return (int32_t)status;
}

/* =========================================================================
 * Template: conv2d_impl — shared logic for int8/int16 convolution
 *
 * Instantiated as mmalib_conv2d_i8 and mmalib_conv2d_i16.
 * Uses MMALIB_CNN_convolveBias_row (row-based convolution with bias).
 *
 * Data layouts (all contiguous, batch N=1):
 *   input:  NCHW — channels at stride H_in*W_in elements
 *   kernel: OIHW — output channels at stride C_in*KH*KW elements
 *   output: NCHW — channels at stride H_out*W_out elements
 *
 * Buffer descriptors passed to MMALIB:
 *   src0 (kernel): 2D [C_out, C_in*KH*KW] — each row is one output filter
 *   src1 (input):  2D [C_in, H_in*W_in]   — each row is one input channel
 *   src2 (bias):   2D [C_out, 1]
 *   src3 (scale):  1D [C_out]
 *   dst  (output): 3D [1, C_out, H_out*W_out]
 *
 * Output-channel tiling:
 *   For stride > 1, the MMA hardware requires subMChannels <= MMA_SIZE.
 *   We process C_out in chunks of MMA_SIZE, advancing kernel/bias/scale/shift/
 *   output pointers by the chunk offset each iteration.
 *
 * Exec argument order: (handle, kernel, input, bias, scale, shift, output)
 * Bias/scale/shift are required non-NULL by MMALIB; this wrapper substitutes
 * identity defaults (zero bias, scale=1, shift=0) when the caller passes NULL.
 * ========================================================================= */

// MMA_SIZE: derived from MMALIB header (MMALIB_MMA_SIZE_8_BIT).
// On C7504 (AM67A/J722S): 32 for int8, 16 for int16.
// Used as subMChannels limit for strided convolution tiling.
static constexpr int32_t MMA_SIZE_I8 = MMALIB_MMA_SIZE_8_BIT;
static constexpr int32_t MMA_SIZE_I16 = MMALIB_MMA_SIZE_8_BIT / 2;

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

    int32_t chunk = (stride_h > 1 || stride_w > 1) ?
        (C_out > MmaSize ? MmaSize : C_out) : C_out;

    // Default bias/scale/shift when caller passes NULL
    Workspace wb, ws, wsh;
    if (!bias) {
        if (!wb.alloc(C_out * bias_elem_size)) return -1;
        memset(wb.ptr, 0, C_out * bias_elem_size);
        bias = wb.ptr;
    }
    if (!scale) {
        if (!ws.alloc(C_out)) return -1;
        memset(ws.ptr, 1, C_out);
        scale = ws.ptr;
    }
    if (!shift) {
        if (!wsh.alloc(C_out)) return -1;
        memset(wsh.ptr, 0, C_out);
        shift = wsh.ptr;
    }

    int32_t result = (int32_t)MMALIB_SUCCESS;

    for (int32_t co_base = 0; co_base < C_out; co_base += chunk) {
        int32_t sub = (co_base + chunk <= C_out) ? chunk : (C_out - co_base);

        uint8_t* k_ptr = (uint8_t*)kernel + co_base * kDim * elem_size;
        uint8_t* b_ptr = (uint8_t*)bias + co_base * bias_elem_size;
        uint8_t* s_ptr = (uint8_t*)scale + co_base;
        uint8_t* sh_ptr = (uint8_t*)shift + co_base;
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
        Workspace handle;
        if (!handle.alloc(handle_size)) { result = -1; break; }

        MMALIB_STATUS status = MMALIB_CNN_convolveBias_row_ixX_ixX_oxX_init(
            handle.ptr, &src0_addr, &src1_addr, &src2_addr, &src3_addr,
            &dst_addr, &init_args);
        if (status != MMALIB_SUCCESS) {
            result = (int32_t)status;
            break;
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
            handle.ptr, k_ptr, input, b_ptr, s_ptr, sh_ptr,
            o_ptr, &exec_in, &exec_out);
        if (status != MMALIB_SUCCESS) {
            result = (int32_t)status;
            break;
        }
    }

    return result;
}

/* =========================================================================
 * Extern "C" entry points — public API
 *
 * These are the symbols exported to dynamically loaded TVM modules.
 * Each instantiates the appropriate template with dtype-specific parameters.
 * ========================================================================= */

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

int32_t mmalib_conv2d_i8_sliced(void* input,       /* L2: already DMA'd by caller */
                                 void* kernel_tile, /* L2: weight slice for this OC tile */
                                 void* bias,        /* DDR: full bias; offset by oc_start */
                                 void* scale,       /* DDR: full scale; offset by oc_start */
                                 void* shift,       /* DDR: full shift; offset by oc_start */
                                 void* output,      /* DDR: full output; offset by oc_start */
                                 int32_t C_in, int32_t H_in, int32_t W_in,
                                 int32_t C_out_tile,
                                 int32_t KH, int32_t KW,
                                 int32_t stride_h, int32_t stride_w,
                                 int32_t pad_top, int32_t pad_bottom,
                                 int32_t pad_left, int32_t pad_right,
                                 int32_t oc_start) {
    int32_t H_out    = (H_in + pad_top + pad_bottom - KH) / stride_h + 1;
    int32_t W_out    = (W_in + pad_left + pad_right  - KW) / stride_w + 1;

    /* Advance per-channel DDR pointers to the start of this OC tile. */
    bias   = (uint8_t*)bias   + (int64_t)oc_start * 4; /* int32 bias */
    scale  = (uint8_t*)scale  + oc_start;               /* uint8 scale */
    shift  = (uint8_t*)shift  + oc_start;               /* uint8 shift */
    output = (uint8_t*)output + (int64_t)oc_start * H_out * W_out; /* int8 output */

    return conv2d_impl<int8_t, MMALIB_INT8, MMALIB_INT32, -128, 127, MMA_SIZE_I8>(
        input, kernel_tile, bias, scale, shift, output,
        C_in, H_in, W_in, C_out_tile, KH, KW,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);
}

/* =========================================================================
 * mmalib_conv2d_i8_grouped_loop — grouped convolution via a C loop over
 * conv2d_impl, one call per group. See mmalib_wrappers.h for the full
 * rationale (this exists to collapse each grouped-conv layer's TVM
 * lowering to a single call_extern, without touching MMALIB's untested
 * numGroupsPerKernel field).
 *
 * PyTorch's native grouped weight layout ([C_out, C_in/groups, KH, KW])
 * is already contiguous per group, so each group's slice of input/
 * kernel/bias/scale/shift/output is just a flat pointer offset -- no
 * reshuffling, no relax.op.strided_slice/concat needed by the caller.
 * ========================================================================= */

int32_t mmalib_conv2d_i8_grouped_loop(void* input, void* kernel,
                                       void* bias, void* scale, void* shift,
                                       void* output,
                                       int32_t C_in, int32_t H_in, int32_t W_in,
                                       int32_t C_out, int32_t KH, int32_t KW,
                                       int32_t stride_h, int32_t stride_w,
                                       int32_t pad_top, int32_t pad_bottom,
                                       int32_t pad_left, int32_t pad_right,
                                       int32_t groups) {
    if (!input || !kernel || !bias || !scale || !shift || !output) {
        return -1;
    }
    if (groups <= 1 || C_in % groups != 0 || C_out % groups != 0) {
        return -1;
    }
    if (groups == C_in) {
        return -1;  // true depthwise -- mmalib_depthwise_conv2d_i8's job
    }

    int32_t C_in_g = C_in / groups;
    int32_t C_out_g = C_out / groups;
    int32_t elem_size = (int32_t)sizeof(int8_t);
    int32_t bias_elem_size = (int32_t)sizeof(int32_t);
    int32_t H_out = (H_in + pad_top + pad_bottom - KH) / stride_h + 1;
    int32_t W_out = (W_in + pad_left + pad_right - KW) / stride_w + 1;
    int32_t in_ch_stride = H_in * W_in;
    int32_t out_ch_stride = H_out * W_out;
    int32_t kernel_ch_stride = C_in_g * KH * KW;

    for (int32_t g = 0; g < groups; g++) {
        uint8_t* in_ptr = (uint8_t*)input + (int64_t)g * C_in_g * in_ch_stride * elem_size;
        uint8_t* k_ptr = (uint8_t*)kernel + (int64_t)g * C_out_g * kernel_ch_stride * elem_size;
        uint8_t* b_ptr = (uint8_t*)bias + (int64_t)g * C_out_g * bias_elem_size;
        uint8_t* s_ptr = (uint8_t*)scale + (int64_t)g * C_out_g;
        uint8_t* sh_ptr = (uint8_t*)shift + (int64_t)g * C_out_g;
        uint8_t* o_ptr = (uint8_t*)output + (int64_t)g * C_out_g * out_ch_stride * elem_size;

        int32_t status = conv2d_impl<int8_t, MMALIB_INT8, MMALIB_INT32, -128, 127, MMA_SIZE_I8>(
            in_ptr, k_ptr, b_ptr, s_ptr, sh_ptr, o_ptr,
            C_in_g, H_in, W_in, C_out_g, KH, KW,
            stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);
        if (status != 0) {
            /* Zero the remaining, not-yet-processed groups' output instead
             * of leaving whatever was previously in that DDR region --
             * deterministic over leaking stale/uninitialized data. */
            int64_t remaining_bytes =
                (int64_t)(groups - g) * C_out_g * out_ch_stride * elem_size;
            memset(o_ptr, 0, (size_t)remaining_bytes);
            return status;
        }
    }
    return 0;
}

int32_t mmalib_conv2d_i16(void* input, void* kernel,
                          void* bias, void* scale, void* shift,
                          void* output,
                          int32_t C_in, int32_t H_in, int32_t W_in,
                          int32_t C_out, int32_t KH, int32_t KW,
                          int32_t stride_h, int32_t stride_w,
                          int32_t pad_top, int32_t pad_bottom,
                          int32_t pad_left, int32_t pad_right) {
    // conv2d_impl handles NULL bias/scale/shift by substituting identity values
    // (zero bias, scale=1, shift=0), so callers may pass NULL for any of them.
    return conv2d_impl<int16_t, MMALIB_INT16, MMALIB_INT64, -32768, 32767, MMA_SIZE_I16>(
        input, kernel, bias, scale, shift, output,
        C_in, H_in, W_in, C_out, KH, KW,
        stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);
}

/* =========================================================================
 * Depthwise conv2d (column-based, highPrecision variant)
 *
 * Uses MMALIB_CNN_convolve_col_smallNo_highPrecision which is optimized for
 * depthwise (Ni=1, No=1) convolution with per-group scale/shift quantization.
 *
 * Implementation steps:
 *   1. Runtime weight reorder — transforms natural [G,1,KH,KW] layout into
 *      MMALIB's internal column-interleaved format via reorderWeights_exec().
 *   2. Single exec call — all groups processed together. MMALIB iterates
 *      over groups internally using groupOffset and numGroupsPerKernel.
 *
 * Data layouts:
 *   input:  NCHW — each channel is a contiguous H_in*W_in block
 *   output: NCHW — each channel is a contiguous H_out*W_out block
 *   weights: natural order [num_groups, 1, KH, KW] (reordered at runtime)
 *
 * Key init_args fields:
 *   - columnOffset: spacing between column pairs in MMA input (2*MMA_SIZE*stride)
 *   - inPairOffset: spacing between elements of an input pair (MMA_SIZE*stride)
 *   - outPairOffset: spacing between output pair elements (MMA_SIZE)
 *   - groupOffset: byte offset between groups in input (H_in*W_in)
 *   - blockFeaturePitch: row stride in input feature map (W_in)
 * ========================================================================= */

int32_t mmalib_depthwise_conv2d_i8(void* input, void* weights,
                                   void* bias, void* scale, void* shift,
                                   void* output,
                                   int32_t channels, int32_t H_in, int32_t W_in,
                                   int32_t KH, int32_t KW,
                                   int32_t stride_h, int32_t stride_w,
                                   int32_t pad_top, int32_t pad_bottom,
                                   int32_t pad_left, int32_t pad_right,
                                   int32_t num_groups) {
    if (!input || !weights || !output) {
        return -1;
    }

    const int32_t mma_size = MMALIB_MMA_SIZE_8_BIT;

    int32_t H_out = (H_in + pad_top + pad_bottom - KH) / stride_h + 1;
    int32_t W_out = (W_in + pad_left + pad_right - KW) / stride_w + 1;

    // Default bias/scale/shift when caller passes NULL
    Workspace wb, ws, wsh;
    if (!bias) {
        if (!wb.alloc(num_groups * 4)) return -1;
        memset(wb.ptr, 0, num_groups * 4);
        bias = wb.ptr;
    }
    if (!scale) {
        if (!ws.alloc(num_groups)) return -1;
        memset(ws.ptr, 1, num_groups);
        scale = ws.ptr;
    }
    if (!shift) {
        if (!wsh.alloc(num_groups)) return -1;
        memset(wsh.ptr, 0, num_groups);
        shift = wsh.ptr;
    }

    // --- Runtime weight reorder via MMALIB API ---
    MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_Args rw_args;
    memset(&rw_args, 0, sizeof(rw_args));
    rw_args.dataType = MMALIB_INT8;
    rw_args.Ni = 1;
    rw_args.No = 1;
    rw_args.Fr = KH;
    rw_args.Fc = KW;
    rw_args.strideX = stride_w;
    rw_args.strideY = stride_h;
    rw_args.dilationX = 1;
    rw_args.featureWidth = W_in;
    rw_args.blockFeatureHeight = H_in;
    rw_args.topPad = pad_top;
    rw_args.bottomPad = pad_bottom;
    rw_args.leftPad = pad_left;
    rw_args.rightPad = pad_right;
    rw_args.numGroupsPerKernel = num_groups;

    int32_t reorder_size =
        MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_getMemorySize(&rw_args);
    if (reorder_size <= 0) return -1;

    Workspace wreorder;
    if (!wreorder.alloc(reorder_size)) return -1;

    MMALIB_bufParams3D_t src0_addr;
    memset(&src0_addr, 0, sizeof(src0_addr));
    MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_fillBufParams(
        &rw_args, &src0_addr);

    MMALIB_bufParams3D_t nat_weights_addr;
    nat_weights_addr.data_type = MMALIB_INT8;
    nat_weights_addr.dim_x = (uint32_t)(1 * KH * KW);
    nat_weights_addr.dim_y = 1;
    nat_weights_addr.stride_y = 1 * KH * KW;
    nat_weights_addr.dim_z = (uint32_t)num_groups;
    nat_weights_addr.stride_z = 1 * nat_weights_addr.stride_y;

    MMALIB_bufParams2D_t bias_rw_addr;
    memset(&bias_rw_addr, 0, sizeof(bias_rw_addr));
    bias_rw_addr.data_type = MMALIB_INT32;
    bias_rw_addr.dim_x = 1;
    bias_rw_addr.dim_y = (uint32_t)num_groups;
    bias_rw_addr.stride_y = 1;

    MMALIB_STATUS status =
        MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_exec(
            HIGHPRECISION_REORDERWEIGHTS,
            &rw_args,
            &nat_weights_addr,
            weights,
            &bias_rw_addr,
            NULL,
            &src0_addr,
            wreorder.ptr);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    // --- Depthwise convolution ---

    // Output row stride must be 64-byte aligned so that pair writes
    // (at offset outPairOffset) don't overflow into adjacent rows.
    int32_t out_stride_y = (W_out + 63) & ~63;
    bool needs_compact = (out_stride_y != W_out);

    MMALIB_bufParams2D_t src1_addr;
    src1_addr.data_type = MMALIB_INT8;
    src1_addr.dim_x = (uint32_t)W_in;
    src1_addr.dim_y = (uint32_t)(H_in * num_groups);
    src1_addr.stride_y = W_in;

    MMALIB_bufParams2D_t src2_addr;
    src2_addr.data_type = MMALIB_INT32;
    src2_addr.dim_x = 1;
    src2_addr.dim_y = (uint32_t)num_groups;
    src2_addr.stride_y = 1;

    MMALIB_bufParams1D_t src3_addr;
    src3_addr.data_type = MMALIB_UINT8;
    src3_addr.dim_x = (uint32_t)(num_groups * 1);

    MMALIB_bufParams3D_t dst_addr;
    dst_addr.data_type = MMALIB_INT8;
    dst_addr.dim_x = (uint32_t)W_out;
    dst_addr.dim_y = (uint32_t)H_out;
    dst_addr.stride_y = out_stride_y;
    dst_addr.dim_z = (uint32_t)num_groups;
    dst_addr.stride_z = H_out * out_stride_y;

    MMALIB_CNN_convolve_col_smallNo_highPrecision_InitArgs init_args;
    memset(&init_args, 0, sizeof(init_args));
    init_args.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
    init_args.Ni = 1;
    init_args.No = 1;
    init_args.Fr = KH;
    init_args.Fc = KW;
    init_args.shift = 0;
    init_args.shiftMethod = 1;
    init_args.strideX = stride_w;
    init_args.strideY = stride_h;
    init_args.dilationX = 1;
    init_args.dilationY = 1;
    init_args.topPad = pad_top;
    init_args.bottomPad = pad_bottom;
    init_args.leftPad = pad_left;
    init_args.rightPad = pad_right;
    init_args.activationType = MMALIB_SATURATION;
    init_args.pSatMin = -128;
    init_args.pSatMax = 127;
    init_args.featureWidth = W_in;
    init_args.blockFeatureHeight = H_in;
    init_args.blockFeaturePitch = W_in;
    init_args.columnOffset = mma_size * 2 * stride_w;
    init_args.inPairOffset = mma_size * stride_w;
    init_args.groupOffset = H_in * W_in;
    init_args.inChOffset = W_in;
    init_args.outPairOffset = mma_size;
    init_args.numGroupsPerKernel = num_groups;

    int32_t handle_size =
        MMALIB_CNN_convolve_col_smallNo_highPrecision_getHandleSize(&init_args);
    Workspace whandle;
    if (!whandle.alloc(handle_size)) return -1;

    Workspace wpadded;
    void* dst_ptr = output;
    if (needs_compact) {
        int32_t padded_size = num_groups * H_out * out_stride_y;
        if (!wpadded.alloc(padded_size)) return -1;
        dst_ptr = wpadded.ptr;
    }

    status = MMALIB_CNN_convolve_col_smallNo_highPrecision_init(
        whandle.ptr, &src0_addr, &src1_addr, &src2_addr, &src3_addr,
        &dst_addr, &init_args);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    MMALIB_CNN_convolve_col_smallNo_highPrecision_ExecInArgs exec_in;
    memset(&exec_in, 0, sizeof(exec_in));
    exec_in.blockFeatureWidth = W_in;
    exec_in.padFillValue = 0;

    MMALIB_CNN_convolve_col_smallNo_highPrecision_ExecOutArgs exec_out;
    memset(&exec_out, 0, sizeof(exec_out));

    status = MMALIB_CNN_convolve_col_smallNo_highPrecision_exec(
        whandle.ptr, wreorder.ptr, input, bias, scale,
        (uint8_t*)shift, dst_ptr, &exec_in, &exec_out);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    if (needs_compact) {
        uint8_t* src = (uint8_t*)dst_ptr;
        uint8_t* dst = (uint8_t*)output;
        for (int32_t g = 0; g < num_groups; g++) {
            for (int32_t r = 0; r < H_out; r++) {
                memcpy(dst, src, W_out);
                src += out_stride_y;
                dst += W_out;
            }
        }
    }

    return (int32_t)MMALIB_SUCCESS;
}

/* =========================================================================
 * Int16 depthwise conv2d — same algorithm as mmalib_depthwise_conv2d_i8,
 * adapted for 16-bit elements.
 *
 * IMPORTANT: MMALIB only supports 3×3 kernels for INT16 depthwise
 * (convolve_col_smallNo_highPrecision).  5×5 and 7×7 are not implemented
 * (tracked as MMALIB-882).  The TVM FuseMMALIBQDQDwConv2dI16 pass
 * enforces this constraint at compile time; this wrapper returns -1 if
 * KH != 3 as a runtime safeguard.
 *
 * Key differences from the int8 version:
 *   - MMA vector width is 16 elements (vs 32 for int8) because each int16
 *     element is 2 bytes and the MMA reads 32-byte vectors.
 *   - All MMALIB_bufParams stride fields are in BYTES, not elements, so
 *     every stride involving element size must be multiplied by elem_size=2.
 *   - Bias is int64 (wider accumulator matches int16 × int16 → int32+ path).
 *   - Saturation bounds: [-32768, 32767] instead of [-128, 127].
 *   - Output row alignment: 32 int16 elements = 64 bytes (same physical
 *     alignment as i8's 64 elements = 64 bytes, but fewer elements).
 *   - groupOffset and blockFeaturePitch also scale by elem_size.
 * ========================================================================= */
int32_t mmalib_depthwise_conv2d_i16(void* input, void* weights,
                                    void* bias, void* scale, void* shift,
                                    void* output,
                                    int32_t channels, int32_t H_in, int32_t W_in,
                                    int32_t KH, int32_t KW,
                                    int32_t stride_h, int32_t stride_w,
                                    int32_t pad_top, int32_t pad_bottom,
                                    int32_t pad_left, int32_t pad_right,
                                    int32_t num_groups) {
    if (!input || !weights || !output) {
        return -1;
    }
    // Only 3×3 kernels are supported for INT16 (MMALIB-882: 5×5/7×7 not implemented)
    if (KH != 3 || KW != 3) {
        return -1;
    }

    // MMA_SIZE_I16 = 16: the MMA reads/writes 16 int16 elements per vector
    // (MMA_SIZE_I8 = 32 for int8; both equal one 32-byte hardware vector).
    const int32_t mma_size = MMA_SIZE_I16;
    const int32_t elem_size = (int32_t)sizeof(int16_t);  // 2 bytes

    int32_t H_out = (H_in + pad_top + pad_bottom - KH) / stride_h + 1;
    int32_t W_out = (W_in + pad_left + pad_right - KW) / stride_w + 1;

    // NULL-default bias/scale/shift to identity values.
    // Bias is int64 (8 bytes per group) because the int16 accumulator is wider.
    Workspace wb, ws, wsh;
    if (!bias) {
        if (!wb.alloc(num_groups * 8)) return -1;
        memset(wb.ptr, 0, num_groups * 8);
        bias = wb.ptr;
    }
    if (!scale) {
        if (!ws.alloc(num_groups)) return -1;
        memset(ws.ptr, 1, num_groups);  // uint8 value 1 → scale factor = 1
        scale = ws.ptr;
    }
    if (!shift) {
        if (!wsh.alloc(num_groups)) return -1;
        memset(wsh.ptr, 0, num_groups);  // uint8 value 0 → no right-shift
        shift = wsh.ptr;
    }

    // -------------------------------------------------------------------------
    // Step 1: Runtime weight reorder
    //
    // MMALIB requires weights in a column-interleaved internal format.
    // The reorderWeights API transforms our natural [G, 1, KH, KW] layout.
    // Setting dataType=MMALIB_INT16 makes MMALIB use the int16 reorder path.
    // -------------------------------------------------------------------------
    MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_Args rw_args;
    memset(&rw_args, 0, sizeof(rw_args));
    rw_args.dataType = MMALIB_INT16;  // int16 weights (vs MMALIB_INT8 for i8)
    rw_args.Ni = 1;                   // depthwise: 1 input channel per group
    rw_args.No = 1;                   // depthwise: 1 output channel per group
    rw_args.Fr = KH;
    rw_args.Fc = KW;
    rw_args.strideX = stride_w;
    rw_args.strideY = stride_h;
    rw_args.dilationX = 1;
    rw_args.featureWidth = W_in;
    rw_args.blockFeatureHeight = H_in;
    rw_args.topPad = pad_top;
    rw_args.bottomPad = pad_bottom;
    rw_args.leftPad = pad_left;
    rw_args.rightPad = pad_right;
    rw_args.numGroupsPerKernel = num_groups;

    int32_t reorder_size =
        MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_getMemorySize(&rw_args);
    if (reorder_size <= 0) return -1;

    Workspace wreorder;
    if (!wreorder.alloc(reorder_size)) return -1;

    // fillBufParams fills src0_addr with the reordered-weight buffer descriptor.
    MMALIB_bufParams3D_t src0_addr;
    memset(&src0_addr, 0, sizeof(src0_addr));
    MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_fillBufParams(
        &rw_args, &src0_addr);

    // Natural weight layout: [num_groups, 1, KH, KW] in row-major int16.
    // stride_y and stride_z are in BYTES: multiply by elem_size.
    MMALIB_bufParams3D_t nat_weights_addr;
    nat_weights_addr.data_type = MMALIB_INT16;
    nat_weights_addr.dim_x    = (uint32_t)(1 * KH * KW);
    nat_weights_addr.dim_y    = 1;
    nat_weights_addr.stride_y = 1 * KH * KW * elem_size;   // bytes per row
    nat_weights_addr.dim_z    = (uint32_t)num_groups;
    nat_weights_addr.stride_z = 1 * nat_weights_addr.stride_y;

    // Bias descriptor for the reorder call — we pass NULL data but MMALIB
    // still uses the descriptor type to size its internal structures.
    MMALIB_bufParams2D_t bias_rw_addr;
    memset(&bias_rw_addr, 0, sizeof(bias_rw_addr));
    bias_rw_addr.data_type = MMALIB_INT64;  // int64 bias for int16 path
    bias_rw_addr.dim_x     = 1;
    bias_rw_addr.dim_y     = (uint32_t)num_groups;
    bias_rw_addr.stride_y  = 1;

    MMALIB_STATUS status =
        MMALIB_CNN_convolve_col_smallNo_highPrecision_reorderWeights_exec(
            HIGHPRECISION_REORDERWEIGHTS,
            &rw_args,
            &nat_weights_addr,
            weights,
            &bias_rw_addr,
            NULL,          // bias not needed in the reorder step
            &src0_addr,
            wreorder.ptr);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    // -------------------------------------------------------------------------
    // Step 2: Depthwise convolution
    // -------------------------------------------------------------------------

    // Output row stride must be 64-byte aligned so pair-writes (at offset
    // outPairOffset = mma_size elements) don't spill into the next row.
    // For int16: 32 elements × 2 bytes = 64 bytes per aligned row.
    // (Int8 version uses 64 elements × 1 byte = 64 bytes — same constraint.)
    int32_t out_stride_y = (W_out + 31) & ~31;
    bool needs_compact = (out_stride_y != W_out);

    // src1: input feature map [C×H_in, W_in] — rows are contiguous channels
    MMALIB_bufParams2D_t src1_addr;
    src1_addr.data_type = MMALIB_INT16;
    src1_addr.dim_x     = (uint32_t)W_in;
    src1_addr.dim_y     = (uint32_t)(H_in * num_groups);
    src1_addr.stride_y  = W_in * elem_size;     // bytes per input row

    // src2: per-group bias [num_groups, 1] int64.
    // stride_y is in ELEMENTS (not bytes) — confirmed by real-world MMALIB usage:
    // stride_y = 1 regardless of whether bias is INT32 or INT64.
    MMALIB_bufParams2D_t src2_addr;
    src2_addr.data_type = MMALIB_INT64;
    src2_addr.dim_x     = 1;
    src2_addr.dim_y     = (uint32_t)num_groups;
    src2_addr.stride_y  = 1;

    // src3: per-group scale [num_groups] uint8
    MMALIB_bufParams1D_t src3_addr;
    src3_addr.data_type = MMALIB_UINT8;
    src3_addr.dim_x     = (uint32_t)(num_groups * 1);

    // dst: output feature map [num_groups, H_out, out_stride_y] int16
    MMALIB_bufParams3D_t dst_addr;
    dst_addr.data_type = MMALIB_INT16;
    dst_addr.dim_x     = (uint32_t)W_out;
    dst_addr.dim_y     = (uint32_t)H_out;
    dst_addr.stride_y  = out_stride_y * elem_size;          // bytes per output row
    dst_addr.dim_z     = (uint32_t)num_groups;
    dst_addr.stride_z  = H_out * out_stride_y * elem_size;  // bytes per group

    MMALIB_CNN_convolve_col_smallNo_highPrecision_InitArgs init_args;
    memset(&init_args, 0, sizeof(init_args));
    init_args.funcStyle       = MMALIB_FUNCTION_OPTIMIZED;
    init_args.Ni              = 1;
    init_args.No              = 1;
    init_args.Fr              = KH;
    init_args.Fc              = KW;
    init_args.shift           = 0;  // per-group shift from src3 (shiftMethod=1)
    init_args.shiftMethod     = 1;  // use per-group shift vector in src3
    init_args.strideX         = stride_w;
    init_args.strideY         = stride_h;
    init_args.dilationX       = 1;
    init_args.dilationY       = 1;
    init_args.topPad          = pad_top;
    init_args.bottomPad       = pad_bottom;
    init_args.leftPad         = pad_left;
    init_args.rightPad        = pad_right;
    init_args.activationType  = MMALIB_SATURATION;
    init_args.pSatMin         = -32768;  // int16 range (vs -128 for int8)
    init_args.pSatMax         =  32767;
    init_args.featureWidth    = W_in;
    init_args.blockFeatureHeight = H_in;
    // blockFeaturePitch: row stride in the input feature map, in BYTES.
    init_args.blockFeaturePitch = W_in * elem_size;

    // inChOffset and groupOffset: in ELEMENTS (not bytes) per MMALIB header docs.
    // For i8 (elem_size=1), elements == bytes so the values look the same as i8.
    init_args.inChOffset  = W_in;         // elements between rows in one group
    init_args.groupOffset = H_in * W_in;  // elements between groups in input

    // MMA column/pair offsets: in ELEMENTS. mma_size=16 for int16.
    init_args.columnOffset    = mma_size * 2 * stride_w;
    init_args.inPairOffset    = mma_size * stride_w;
    init_args.outPairOffset   = mma_size;
    init_args.numGroupsPerKernel = num_groups;

    int32_t handle_size =
        MMALIB_CNN_convolve_col_smallNo_highPrecision_getHandleSize(&init_args);
    Workspace whandle;
    if (!whandle.alloc(handle_size)) return -1;

    // Allocate a padded output buffer when W_out is not 32-element aligned.
    // After exec, compact the padded output into the caller's contiguous buffer.
    Workspace wpadded;
    void* dst_ptr = output;
    if (needs_compact) {
        int32_t padded_size = num_groups * H_out * out_stride_y * elem_size;
        if (!wpadded.alloc(padded_size)) return -1;
        dst_ptr = wpadded.ptr;
    }

    status = MMALIB_CNN_convolve_col_smallNo_highPrecision_init(
        whandle.ptr, &src0_addr, &src1_addr, &src2_addr, &src3_addr,
        &dst_addr, &init_args);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    MMALIB_CNN_convolve_col_smallNo_highPrecision_ExecInArgs exec_in;
    memset(&exec_in, 0, sizeof(exec_in));
    exec_in.blockFeatureWidth = W_in;
    exec_in.padFillValue      = 0;

    MMALIB_CNN_convolve_col_smallNo_highPrecision_ExecOutArgs exec_out;
    memset(&exec_out, 0, sizeof(exec_out));

    status = MMALIB_CNN_convolve_col_smallNo_highPrecision_exec(
        whandle.ptr, wreorder.ptr, input, bias, scale,
        (uint8_t*)shift, dst_ptr, &exec_in, &exec_out);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    // Remove row padding: copy W_out int16 elements per row, skipping the
    // out_stride_y - W_out padding elements at the end of each padded row.
    if (needs_compact) {
        uint16_t* src = (uint16_t*)dst_ptr;
        uint16_t* dst = (uint16_t*)output;
        for (int32_t g = 0; g < num_groups; g++) {
            for (int32_t r = 0; r < H_out; r++) {
                memcpy(dst, src, W_out * elem_size);
                src += out_stride_y;  // advance by padded row width (elements)
                dst += W_out;         // advance by actual row width (elements)
            }
        }
    }

    return (int32_t)MMALIB_SUCCESS;
}

/* =========================================================================
 * Matrix multiply with bias (LINALG matrixMatrixMultiplyBias)
 *
 * Implements a quantized fully-connected layer:
 *   C[m,n] = sat_i8((A[m,:] · B[n,:] + bias[n]) * scale[n] >> shift[n])
 *
 * Uses bTranspose=1 so weights are stored in [N, K] (PyTorch convention)
 * and transposed internally by the MMA hardware — no reordering needed.
 *
 * Buffer layouts (all row-major, contiguous):
 *   input  (A): [M, K] — activations
 *   weight (B): [N, K] — stored transposed, HW reads as [K, N]
 *   bias:       [N]    — int32, added after matmul accumulation
 *   scale:      [N]    — uint8, per-column multiplicative scale
 *   shift:      [N]    — uint8, per-column right-shift
 *   output (C): [M, N] — quantized result
 *
 * Scale/shift application order: (accum + bias) * scale >> shift
 * This matches the QDQ (quantize-dequantize-quantize) pattern used by
 * TVM's quantization passes.
 * ========================================================================= */

int32_t mmalib_matmul_bias_i8(void* input, void* weights,
                              void* bias, void* scale, void* shift,
                              void* output,
                              int32_t M, int32_t K, int32_t N) {
    if (!input || !weights || !output) {
        return -1;
    }

    Workspace wb, ws, wsh;
    if (!bias) {
        if (!wb.alloc(N * 4)) return -1;
        memset(wb.ptr, 0, N * 4);
        bias = wb.ptr;
    }
    if (!scale) {
        if (!ws.alloc(N)) return -1;
        memset(ws.ptr, 1, N);
        scale = ws.ptr;
    }
    if (!shift) {
        if (!wsh.alloc(N)) return -1;
        memset(wsh.ptr, 0, N);
        shift = wsh.ptr;
    }

    MMALIB_bufParams3D_t src0_params;
    src0_params.data_type = MMALIB_INT8;
    src0_params.dim_x = (uint32_t)K;
    src0_params.dim_y = (uint32_t)M;
    src0_params.stride_y = K;
    src0_params.dim_z = 1;
    src0_params.stride_z = M * K;

    MMALIB_bufParams3D_t src1_params;
    src1_params.data_type = MMALIB_INT8;
    src1_params.dim_x = (uint32_t)K;
    src1_params.dim_y = (uint32_t)N;
    src1_params.stride_y = K;
    src1_params.dim_z = 1;
    src1_params.stride_z = N * K;

    MMALIB_bufParams2D_t src2_params;
    src2_params.data_type = MMALIB_INT32;
    src2_params.dim_x = (uint32_t)N;
    src2_params.dim_y = 1;
    src2_params.stride_y = N * 4;

    MMALIB_bufParams2D_t src3_params;
    src3_params.data_type = MMALIB_INT8;
    src3_params.dim_x = (uint32_t)N;
    src3_params.dim_y = 1;
    src3_params.stride_y = N;

    MMALIB_bufParams3D_t dst_params;
    dst_params.data_type = MMALIB_INT8;
    dst_params.dim_x = (uint32_t)N;
    dst_params.dim_y = (uint32_t)M;
    dst_params.stride_y = N;
    dst_params.dim_z = 1;
    dst_params.stride_z = M * N;

    MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_InitArgs init_args;
    memset(&init_args, 0, sizeof(init_args));
    init_args.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
    init_args.activationType = MMALIB_SATURATION;
    init_args.pSatMin = -128;
    init_args.pSatMax = 127;
    init_args.bTranspose =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_B_TRANSPOSED;
    init_args.biasOrder =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_BIAS_ORDER_ROW;
    init_args.scaleAndShiftFlag =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_SCALE_SHIFT_VECTOR;
    init_args.scaleShiftOrder =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_SCALE_SHIFT_ORDER_ROW;
    init_args.interleavedFlag = 0;

    int32_t handle_size =
        MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_getHandleSize(&init_args);
    Workspace whandle;
    if (!whandle.alloc(handle_size)) return -1;

    MMALIB_STATUS status = MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_init(
        whandle.ptr, &src0_params, &src1_params, &src2_params, &src3_params,
        &dst_params, &init_args);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    status = MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_exec(
        whandle.ptr, input, weights, bias, scale, shift, output);
    return (int32_t)status;
}

int32_t mmalib_matmul_bias_i16(void* input, void* weights,
                               void* bias, void* scale, void* shift,
                               void* output,
                               int32_t M, int32_t K, int32_t N) {
    if (!input || !weights || !output) {
        return -1;
    }

    Workspace wb, ws, wsh;
    if (!bias) {
        if (!wb.alloc(N * 8)) return -1;  // int64 bias for int16
        memset(wb.ptr, 0, N * 8);
        bias = wb.ptr;
    }
    if (!scale) {
        if (!ws.alloc(N)) return -1;
        memset(ws.ptr, 1, N);
        scale = ws.ptr;
    }
    if (!shift) {
        if (!wsh.alloc(N)) return -1;
        memset(wsh.ptr, 0, N);
        shift = wsh.ptr;
    }

    MMALIB_bufParams3D_t src0_params;
    src0_params.data_type = MMALIB_INT16;
    src0_params.dim_x = (uint32_t)K;
    src0_params.dim_y = (uint32_t)M;
    src0_params.stride_y = K * 2;
    src0_params.dim_z = 1;
    src0_params.stride_z = M * K * 2;

    MMALIB_bufParams3D_t src1_params;
    src1_params.data_type = MMALIB_INT16;
    src1_params.dim_x = (uint32_t)K;
    src1_params.dim_y = (uint32_t)N;
    src1_params.stride_y = K * 2;
    src1_params.dim_z = 1;
    src1_params.stride_z = N * K * 2;

    MMALIB_bufParams2D_t src2_params;
    src2_params.data_type = MMALIB_INT64;
    src2_params.dim_x = (uint32_t)N;
    src2_params.dim_y = 1;
    src2_params.stride_y = N * 8;

    MMALIB_bufParams2D_t src3_params;
    src3_params.data_type = MMALIB_INT8;
    src3_params.dim_x = (uint32_t)N;
    src3_params.dim_y = 1;
    src3_params.stride_y = N;

    MMALIB_bufParams3D_t dst_params;
    dst_params.data_type = MMALIB_INT16;
    dst_params.dim_x = (uint32_t)N;
    dst_params.dim_y = (uint32_t)M;
    dst_params.stride_y = N * 2;
    dst_params.dim_z = 1;
    dst_params.stride_z = M * N * 2;

    MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_InitArgs init_args;
    memset(&init_args, 0, sizeof(init_args));
    init_args.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
    init_args.activationType = MMALIB_SATURATION;
    init_args.pSatMin = -32768;
    init_args.pSatMax = 32767;
    init_args.bTranspose =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_B_TRANSPOSED;
    init_args.biasOrder =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_BIAS_ORDER_ROW;
    init_args.scaleAndShiftFlag =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_SCALE_SHIFT_VECTOR;
    init_args.scaleShiftOrder =
        MMALIB_LINALG_MATRIXMATRIXMULTIPLYBIAS_IXX_IXX_OXX_SCALE_SHIFT_ORDER_ROW;
    init_args.interleavedFlag = 0;

    int32_t handle_size =
        MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_getHandleSize(&init_args);
    Workspace whandle;
    if (!whandle.alloc(handle_size)) return -1;

    MMALIB_STATUS status = MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_init(
        whandle.ptr, &src0_params, &src1_params, &src2_params, &src3_params,
        &dst_params, &init_args);
    if (status != MMALIB_SUCCESS) return (int32_t)status;

    status = MMALIB_LINALG_matrixMatrixMultiplyBias_ixX_ixX_oxX_exec(
        whandle.ptr, input, weights, bias, scale, shift, output);
    return (int32_t)status;
}