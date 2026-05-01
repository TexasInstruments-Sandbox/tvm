"""
MMALIB direct conv2d integration test.

Tests int16 2D convolution using MMALIB on C7x (host emulation).
Validates: Relax IR → MMALIB legalization → C codegen → link → run.

Usage:
    pytest test_mmalib_conv2d_dsp.py -v --dsp-mode=c7x_host
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax
from tvm.relax import TensorStructInfo

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


def _create_int16_conv2d_model(
    c_in: int, h: int, w: int, c_out: int, kh: int, kw: int, padding: int, seed: int = 42
):
    """Create a Relax IRModule with int16 conv2d and constant kernel."""
    rng = np.random.default_rng(seed)

    input_data = rng.integers(-4, 4, size=(1, c_in, h, w), dtype=np.int16)
    kernel_data = rng.integers(-4, 4, size=(c_out, c_in, kh, kw), dtype=np.int16)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, c_in, h, w), "int16"))
    w_const = relax.Constant(kernel_data)

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            out = bb.emit(
                relax.op.nn.conv2d(
                    x,
                    w_const,
                    strides=(1, 1),
                    padding=(padding, padding, padding, padding),
                    dilation=(1, 1),
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_dtype="int16",
                )
            )
            bb.emit_output(out)
        bb.emit_func_output(out)

    mod = bb.finalize()
    return mod, input_data, kernel_data


def _numpy_conv2d_i16(input_data, kernel_data, padding):
    """Reference int16 conv2d (NCHW, no bias, shift=0)."""
    n, c_in, h_in, w_in = input_data.shape
    c_out, _, kh, kw = kernel_data.shape

    h_out = h_in + 2 * padding - kh + 1
    w_out = w_in + 2 * padding - kw + 1

    # Pad input
    padded = np.pad(
        input_data,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    output = np.zeros((n, c_out, h_out, w_out), dtype=np.int64)
    for co in range(c_out):
        for ci in range(c_in):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch = padded[0, ci, oh : oh + kh, ow : ow + kw]
                    output[0, co, oh, ow] += np.sum(
                        patch.astype(np.int64) * kernel_data[co, ci].astype(np.int64)
                    )

    output = np.clip(output, -32768, 32767)
    return output.astype(np.int16)


@pytest.mark.skip(reason="int16 not used in practice (PyTorch has no int16 quantization)")
def test_mmalib_conv2d_i16(dsp_mode, record_cycles):
    """Test int16 conv2d via MMALIB."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    c_in, h, w = 16, 8, 8
    c_out, kh, kw = 32, 3, 3
    padding = 1

    mod, input_data, kernel_data = _create_int16_conv2d_model(c_in, h, w, c_out, kh, kw, padding)

    ref_output = _numpy_conv2d_i16(input_data, kernel_data, padding)

    target_string = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        profile=True,
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_output = results.get(result_key)
    assert dsp_output is not None, f"No {result_key} returned"

    dsp_output_i16 = dsp_output.astype(np.int16).reshape(1, c_out, h, w)
    max_diff = int(
        np.max(np.abs(dsp_output_i16.astype(np.int32) - ref_output.astype(np.int32)))
    )

    cycles = results.get("c7x_dload_cycles", 0)
    print(f"Conv2d i16: max_diff={max_diff}, cycles={cycles:,}")
    record_cycles("mmalib_conv2d_i16", cycles)

    assert max_diff == 0, f"MMALIB conv2d mismatch: max_diff={max_diff}"
