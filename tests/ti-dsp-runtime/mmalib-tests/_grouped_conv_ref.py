"""Shared numpy reference for grouped conv2d + per-channel bias/scale/shift,
matching MMALIB's quantMethod=1 rounded right-shift convention. Used by
test_mmalib_conv2d_i8_grouped_loop_dsp.py and test_mmalib_loop_only_chain_dsp.py
(both call mmalib_conv2d_i8_grouped_loop directly with lowered, already-
requantized int8/uint8 constants -- not the float-domain QDQ reference in
test_mmalib_qdq_grouped_conv2d_i8_dsp.py, which needs different math).
"""

import numpy as np


def numpy_grouped_conv2d_i8(
    input_np,
    kernel_np,
    bias_np,
    scale_np,
    shift_np,
    C_in,
    H_in,
    W_in,
    C_out,
    KH,
    KW,
    stride,
    pad,
    groups,
):
    H_out = (H_in + 2 * pad - KH) // stride + 1
    W_out = (W_in + 2 * pad - KW) // stride + 1
    C_in_g = C_in // groups
    C_out_g = C_out // groups

    padded = np.pad(input_np[0].astype(np.int32), ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    out = np.zeros((C_out, H_out, W_out), dtype=np.int32)
    for g in range(groups):
        ci0 = g * C_in_g
        co0 = g * C_out_g
        for co in range(C_out_g):
            k = kernel_np[co0 + co].astype(np.int32)
            for oh in range(H_out):
                for ow in range(W_out):
                    ih0, iw0 = oh * stride, ow * stride
                    patch = padded[ci0 : ci0 + C_in_g, ih0 : ih0 + KH, iw0 : iw0 + KW]
                    out[co0 + co, oh, ow] = int(np.sum(patch * k))

    biased = out + bias_np.reshape(C_out, 1, 1).astype(np.int32)
    scaled = biased.astype(np.int64) * scale_np.reshape(C_out, 1, 1).astype(np.int64)
    shift = shift_np.reshape(C_out, 1, 1).astype(np.int64)
    rounding = np.where(shift > 0, np.int64(1) << np.maximum(shift - 1, 0), 0)
    shifted = (scaled + rounding) >> shift
    return np.clip(shifted, -128, 127).astype(np.int8).reshape(1, C_out, H_out, W_out)
