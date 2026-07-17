"""Unit test for c7x_dequantize_vecmatmul kernel.

Directly invokes the kernel via call_extern with known inputs and
verifies output matches numpy reference. Tests the kernel in isolation,
independent of the FuseDequantizeMatmul pass.

Usage:
    pytest test_vecmatmul_kernel.py -v --dsp-mode=c7x_host
    pytest test_vecmatmul_kernel.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402
    compile_and_run_dsp,
    get_target_string,
    set_current_test_name,
)


def _numpy_ref(activation, weights_int8, scale):
    """Compute expected output: output[m,n] = sum_k(act[m,k] * w[n,k]) * scale[n]"""
    # weights_int8: [N, K], activation: [M, K], scale: [N]
    acc = activation @ weights_int8.astype(np.float32).T  # [M, N]
    return acc * scale.reshape(1, -1)


def _build_vecmatmul_module(M, K, N, w_int8, scale):
    """Build a Relax module that calls c7x_dequantize_vecmatmul directly."""

    def te_vecmatmul(act_t, w_t, scale_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "c7x_dequantize_vecmatmul",
                ins[0].data,
                ins[1].data,
                ins[2].data,
                outs[0].data,
                M,
                K,
                N,
            )

        return te.extern(
            [M, N],
            [act_t, w_t, scale_t],
            fcompute,
            name="dequantize_vecmatmul",
            dtype="float32",
        )

    bb = relax.BlockBuilder()

    act_var = relax.Var("activation", relax.TensorStructInfo([M, K], "float32"))
    w_var = relax.Var("weights", relax.TensorStructInfo([N, K], "int8"))
    s_var = relax.Var("scale", relax.TensorStructInfo([N], "float32"))

    with bb.function("main", [act_var, w_var, s_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(
                te_vecmatmul,
                act_var,
                w_var,
                s_var,
                primfunc_name_hint="dequantize_vecmatmul",
            )
            out = bb.emit_output(result)
        bb.emit_func_output(out)

    mod = bb.finalize()

    # Bind weight and scale as constants, leaving only activation as runtime input
    params = {
        mod["main"].params[1]: w_int8,
        mod["main"].params[2]: scale,
    }
    mod = relax.transform.BindParams(func_name="main", params=params)(mod)

    return mod


def _run_vecmatmul_test(dsp_mode, M, K, N, seed=42):
    """Build, compile, run, verify."""
    rng = np.random.default_rng(seed)

    w_float = rng.uniform(-0.1, 0.1, (N, K)).astype(np.float32)
    scale = (np.abs(w_float).max(axis=1) / 127.0).astype(np.float32)
    scale = np.maximum(scale, 1e-10).astype(np.float32)
    w_int8 = np.clip(np.round(w_float / scale.reshape(-1, 1)), -128, 127).astype(np.int8)

    activation = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    ref = _numpy_ref(activation, w_int8, scale)

    mod = _build_vecmatmul_module(M, K, N, w_int8, scale)

    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=activation,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )

    result_key = f"{dsp_mode}_result"
    output = results[result_key].reshape(ref.shape)

    max_diff = float(np.abs(output - ref).max())
    ref_range = float(np.abs(ref).max())
    rel_err = max_diff / ref_range if ref_range > 0 else 0

    cycles = results.get("c7x_dload_cycles", 0)

    print(f"\n  vecmatmul M={M} K={K} N={N}")
    print(f"    max_diff={max_diff:.4e}  rel_err={rel_err:.6f}")
    if cycles:
        print(f"    cycles={cycles:,}")

    return output, ref, rel_err, cycles


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_vecmatmul_decode_576x576(dsp_mode, record_cycles):
    """Decode (M=1): Q/K/V/O projection size."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    test_name = f"vecmatmul_1x576x576_{dsp_mode}"
    set_current_test_name(test_name)
    try:
        _, _, rel_err, cycles = _run_vecmatmul_test(dsp_mode, M=1, K=576, N=576)
        record_cycles("vecmatmul_1x576x576", cycles)
        assert rel_err < 1e-5, f"Relative error too large: {rel_err:.6e}"
    finally:
        set_current_test_name(None)


@pytest.mark.quick
def test_vecmatmul_decode_576x1536(dsp_mode, record_cycles):
    """Decode (M=1): gate_proj / up_proj size."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    test_name = f"vecmatmul_1x576x1536_{dsp_mode}"
    set_current_test_name(test_name)
    try:
        _, _, rel_err, cycles = _run_vecmatmul_test(dsp_mode, M=1, K=576, N=1536)
        record_cycles("vecmatmul_1x576x1536", cycles)
        assert rel_err < 1e-5, f"Relative error too large: {rel_err:.6e}"
    finally:
        set_current_test_name(None)


@pytest.mark.quick
def test_vecmatmul_decode_1536x576(dsp_mode, record_cycles):
    """Decode (M=1): down_proj size."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    test_name = f"vecmatmul_1x1536x576_{dsp_mode}"
    set_current_test_name(test_name)
    try:
        _, _, rel_err, cycles = _run_vecmatmul_test(dsp_mode, M=1, K=1536, N=576)
        record_cycles("vecmatmul_1x1536x576", cycles)
        assert rel_err < 1e-5, f"Relative error too large: {rel_err:.6e}"
    finally:
        set_current_test_name(None)


@pytest.mark.quick
def test_vecmatmul_prefill_64x576x576(dsp_mode, record_cycles):
    """Prefill (M=64): Q projection size."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    test_name = f"vecmatmul_64x576x576_{dsp_mode}"
    set_current_test_name(test_name)
    try:
        _, _, rel_err, cycles = _run_vecmatmul_test(dsp_mode, M=64, K=576, N=576)
        record_cycles("vecmatmul_64x576x576", cycles)
        assert rel_err < 1e-4, f"Relative error too large: {rel_err:.6e}"
    finally:
        set_current_test_name(None)
