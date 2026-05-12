"""
MMALIB int16 matmul (non-bias) integration test for SmolLM MLP offload.

Tests mmalib_matmul_i16 with float→int16 quantization and shift-based
overflow prevention. Validates that the dequantized output matches the
float reference to high precision (max_diff < 0.01 relative).

This tests the end-to-end flow:
  float_activation → quantize_int16 → mmalib_matmul_i16(shift) → dequant → float

Usage:
    pytest test_mmalib_matmul_i16_dsp.py -v --dsp-mode=c7x_dload
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir
from tvm.relax import TensorStructInfo

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


def _compute_shift(w_i16, x_scale, K):
    """Compute shift to prevent int16 output overflow.

    The accumulator for int16×int16 matmul can reach:
      max_accum = max|x_i16| * max|w_i16| * K

    The shift ensures: max_accum >> shift <= 32767
    """
    max_w = int(np.abs(w_i16).max())
    max_x = 32767  # worst case for int16 input
    max_accum = max_w * max_x * K
    if max_accum <= 32767:
        return 0
    return math.ceil(math.log2(max_accum / 32767))


def _build_i16_matmul_model(M, K, N, x_i16, w_i16_KN, shift):
    """Build a Relax model with a single mmalib_matmul_i16 call."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((M, K), "int16"))

    def te_matmul(data_t, w_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "mmalib_matmul_i16",
                ins[0].data,
                ins[1].data,
                outs[0].data,
                M,
                K,
                N,
                shift,
            )

        return te.extern(
            [M, N], [data_t, w_t], fcompute, name="matmul_i16", dtype="int16"
        )

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit(
                bb.call_te(
                    te_matmul,
                    x,
                    relax.Constant(w_i16_KN),
                    primfunc_name_hint="matmul_i16",
                )
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize()


def _run_i16_matmul_test(dsp_mode, M, K, N, seed=42):
    """Full float→int16→mmalib→dequant→float test.

    1. Generate random float activation and weight
    2. Quantize both to int16
    3. Compute shift for overflow prevention
    4. Run mmalib_matmul_i16 on DSP
    5. Dequantize output to float
    6. Compare against numpy float matmul reference
    """
    rng = np.random.default_rng(seed)

    # Float data (simulating RMSNorm output and weight)
    x_float = rng.uniform(-5.0, 5.0, size=(M, K)).astype(np.float32)
    w_float = rng.uniform(-0.05, 0.05, size=(N, K)).astype(np.float32)

    # Float reference
    float_ref = x_float @ w_float.T  # [M, N]

    # Quantize activation to int16 (per-tensor symmetric)
    x_scale = float(np.abs(x_float).max()) / 32767.0
    x_i16 = np.clip(np.round(x_float / x_scale), -32768, 32767).astype(np.int16)

    # Quantize weight to int16 (per-channel symmetric, axis=0 of [N, K])
    w_scale = np.abs(w_float).max(axis=1) / 32767.0  # [N]
    w_scale = np.maximum(w_scale, 1e-10).astype(np.float32)
    w_i16 = np.clip(
        np.round(w_float / w_scale.reshape(N, 1)), -32768, 32767
    ).astype(np.int16)

    # Weight in [K, N] layout for non-transposed mmalib_matmul_i16
    w_i16_KN = np.ascontiguousarray(w_i16.T)  # [K, N]

    # Compute shift
    shift = _compute_shift(w_i16, x_scale, K)

    # Build and run
    mod = _build_i16_matmul_model(M, K, N, x_i16, w_i16_KN, shift)
    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=x_i16,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_i16 = results[result_key].astype(np.int16).reshape(M, N)

    # Dequantize: out_float = out_i16 * (2^shift) * x_scale * w_scale[n]
    dequant_scale = (1 << shift) * x_scale * w_scale.reshape(1, N)
    dsp_float = dsp_i16.astype(np.float64) * dequant_scale

    # Compare
    diff = np.abs(dsp_float - float_ref.astype(np.float64))
    max_diff = float(diff.max())
    output_range = float(np.abs(float_ref).max())
    relative_err = max_diff / output_range if output_range > 0 else 0

    cycles = results.get("c7x_dload_cycles", 0)
    return max_diff, relative_err, cycles, shift


@pytest.mark.c7x_only
def test_mmalib_matmul_i16_gate_proj(dsp_mode, record_cycles):
    """Int16 matmul for SmolLM gate_proj (M=64, K=576, N=1536)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")

    max_diff, rel_err, cycles, shift = _run_i16_matmul_test(
        dsp_mode, M=64, K=576, N=1536
    )

    print(f"\n  gate_proj i16: max_diff={max_diff:.4f}, rel_err={rel_err:.6f}, shift={shift}")
    assert rel_err < 0.001, f"Relative error too large: {rel_err:.4f}"

    record_cycles("mmalib_matmul_i16_gate_proj", cycles)


@pytest.mark.c7x_only
def test_mmalib_matmul_i16_down_proj(dsp_mode, record_cycles):
    """Int16 matmul for SmolLM down_proj (M=64, K=1536, N=576)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")

    max_diff, rel_err, cycles, shift = _run_i16_matmul_test(
        dsp_mode, M=64, K=1536, N=576
    )

    print(f"\n  down_proj i16: max_diff={max_diff:.4f}, rel_err={rel_err:.6f}, shift={shift}")
    assert rel_err < 0.001, f"Relative error too large: {rel_err:.4f}"

    record_cycles("mmalib_matmul_i16_down_proj", cycles)


@pytest.mark.c7x_only
def test_mmalib_matmul_i16_small(dsp_mode, record_cycles):
    """Int16 matmul with minimal dimensions (M=32, K=32, N=32)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")

    max_diff, rel_err, cycles, shift = _run_i16_matmul_test(
        dsp_mode, M=32, K=32, N=32, seed=123
    )

    print(f"\n  small i16: max_diff={max_diff:.4f}, rel_err={rel_err:.6f}, shift={shift}")
    assert rel_err < 0.001, f"Relative error too large: {rel_err:.4f}"

    record_cycles("mmalib_matmul_i16_32x32", cycles)
