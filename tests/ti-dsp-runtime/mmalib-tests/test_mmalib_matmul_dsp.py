"""
MMALIB int8 matmul integration test.

Tests int8 matrix multiplication using MMALIB on C7x.
Validates: Relax IR → MMALIB legalization → C codegen → link → run.

Usage:
    pytest test_mmalib_matmul_dsp.py -v --dsp-mode=c7x_host
    pytest test_mmalib_matmul_dsp.py -v --dsp-mode=c7x_dload
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


def _create_int8_matmul_model(m: int, k: int, n: int, seed: int = 42):
    """Create a Relax IRModule with int8 matmul and constant weight."""
    rng = np.random.default_rng(seed)

    weight_data = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
    input_data = rng.integers(-8, 8, size=(m, k), dtype=np.int8)

    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((m, k), "int8"))
    w_const = relax.Constant(weight_data)

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            out = bb.emit(relax.op.matmul(x, w_const, out_dtype="int8"))
            bb.emit_output(out)
        bb.emit_func_output(out)

    mod = bb.finalize()
    return mod, input_data, weight_data


def _numpy_matmul_i8(a, b, shift=0):
    """Reference int8 matmul matching MMALIB behavior."""
    result = a.astype(np.int64) @ b.astype(np.int64)
    if shift > 0:
        result = (result + (1 << (shift - 1))) >> shift
    result = np.clip(result, -128, 127)
    return result.astype(np.int8)


@pytest.mark.skip(reason="int8 matmul legalization not yet implemented (only conv2d uses MMALIB i8)")
def test_mmalib_matmul_i8(dsp_mode, record_cycles):
    """Test int8 matmul via MMALIB."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("MMALIB test requires c7x_host or c7x_dload")

    m, k, n = 64, 128, 64
    mod, input_data, weight_data = _create_int8_matmul_model(m, k, n)

    # Numpy reference (shift=0)
    ref_output = _numpy_matmul_i8(input_data, weight_data, shift=0)

    # Compile and run with MMALIB
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

    dsp_output_i8 = dsp_output.astype(np.int8).reshape(m, n)
    max_diff = int(
        np.max(np.abs(dsp_output_i8.astype(np.int32) - ref_output.astype(np.int32)))
    )

    cycles = results.get("c7x_dload_cycles", 0)
    print(f"Matmul i8: max_diff={max_diff}, cycles={cycles:,}")
    record_cycles("mmalib_matmul_i8", cycles)

    assert max_diff == 0, f"MMALIB matmul mismatch: max_diff={max_diff}"
