"""Unit test for tvm_sdpa_decode kernel.

Directly invokes the kernel via call_extern with known inputs and
verifies output matches numpy reference. Tests GQA head expansion,
Q×K^T dot product, softmax, and attention×V in one fused call.

Usage:
    pytest test_sdpa_decode_kernel.py -v --dsp-mode=c7x_host
    pytest test_sdpa_decode_kernel.py -v --dsp-mode=c7x_dload
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


def _numpy_sdpa_decode(Q, K_cache, V_cache, mask, num_q_heads, num_kv_heads,
                       head_dim, max_cache_len):
    """Numpy reference for decode SDPA with GQA."""
    heads_per_group = num_q_heads // num_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    output = np.zeros((num_q_heads, head_dim), dtype=np.float32)

    for qh in range(num_q_heads):
        kv_h = qh // heads_per_group
        q_vec = Q[qh, :]
        k_head = K_cache[kv_h, :, :]
        v_head = V_cache[kv_h, :, :]

        scores = (q_vec @ k_head.T) * scale + mask
        scores_max = scores.max()
        scores_exp = np.exp(scores - scores_max)
        scores_softmax = scores_exp / scores_exp.sum()
        output[qh, :] = scores_softmax @ v_head

    return output


def _build_sdpa_module(num_q_heads, num_kv_heads, head_dim, max_cache_len):
    """Build a Relax module that calls tvm_sdpa_decode directly."""

    def te_sdpa(q_t, k_t, v_t, mask_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "tvm_sdpa_decode",
                ins[0].data,
                ins[1].data,
                ins[2].data,
                ins[3].data,
                outs[0].data,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_cache_len,
            )

        return te.extern(
            [num_q_heads, head_dim],
            [q_t, k_t, v_t, mask_t],
            fcompute,
            name="sdpa_decode",
            dtype="float32",
        )

    bb = relax.BlockBuilder()

    q_var = relax.Var("Q", relax.TensorStructInfo([num_q_heads, head_dim], "float32"))
    k_var = relax.Var("K_cache", relax.TensorStructInfo(
        [num_kv_heads, max_cache_len, head_dim], "float32"))
    v_var = relax.Var("V_cache", relax.TensorStructInfo(
        [num_kv_heads, max_cache_len, head_dim], "float32"))
    mask_var = relax.Var("mask", relax.TensorStructInfo([max_cache_len], "float32"))

    with bb.function("main", [q_var, k_var, v_var, mask_var], attrs={"num_input": 4}):
        with bb.dataflow():
            result = bb.emit_te(
                te_sdpa,
                q_var,
                k_var,
                v_var,
                mask_var,
                primfunc_name_hint="sdpa_decode",
            )
            out = bb.emit_output(result)
        bb.emit_func_output(out)

    return bb.finalize()


def _run_sdpa_test(dsp_mode, num_q_heads, num_kv_heads, head_dim,
                   max_cache_len, cache_pos, seed=42):
    """Build, compile, run, verify."""
    rng = np.random.default_rng(seed)

    Q = rng.uniform(-1.0, 1.0, (num_q_heads, head_dim)).astype(np.float32)
    K_cache = rng.uniform(-0.5, 0.5,
                          (num_kv_heads, max_cache_len, head_dim)).astype(np.float32)
    V_cache = rng.uniform(-0.5, 0.5,
                          (num_kv_heads, max_cache_len, head_dim)).astype(np.float32)

    mask = np.full(max_cache_len, -3.4e38, dtype=np.float32)
    mask[:cache_pos] = 0.0

    ref = _numpy_sdpa_decode(Q, K_cache, V_cache, mask, num_q_heads,
                             num_kv_heads, head_dim, max_cache_len)

    mod = _build_sdpa_module(num_q_heads, num_kv_heads, head_dim, max_cache_len)

    target = get_target_string(dsp_mode, use_cpp_api=True)
    inputs = [Q, K_cache, V_cache, mask]
    results = compile_and_run_dsp(
        mod=mod,
        input_data=inputs,
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

    print(f"\n  sdpa_decode q_heads={num_q_heads} kv_heads={num_kv_heads} "
          f"head_dim={head_dim} cache_len={max_cache_len} cache_pos={cache_pos}")
    print(f"    max_diff={max_diff:.4e}  rel_err={rel_err:.6f}")
    if cycles:
        print(f"    cycles={cycles:,}")

    assert rel_err < 1e-4, (
        f"SDPA output mismatch: max_diff={max_diff:.4e}, rel_err={rel_err:.6f}"
    )

    return output, ref, rel_err, cycles


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_sdpa_smollm_dims(dsp_mode, record_cycles):
    """SmolLM-135M dimensions: 9 query heads, 3 KV heads, 64 head_dim, 256 cache."""
    set_current_test_name("sdpa_smollm")
    _, _, rel_err, cycles = _run_sdpa_test(
        dsp_mode,
        num_q_heads=9,
        num_kv_heads=3,
        head_dim=64,
        max_cache_len=256,
        cache_pos=16,
    )
    if cycles:
        record_cycles("sdpa_smollm_9h_3kv_64d_256c", cycles)


@pytest.mark.quick
def test_sdpa_partial_cache(dsp_mode, record_cycles):
    """Test with partially filled cache (cache_pos=1, first decode step)."""
    set_current_test_name("sdpa_partial")
    _, _, rel_err, cycles = _run_sdpa_test(
        dsp_mode,
        num_q_heads=9,
        num_kv_heads=3,
        head_dim=64,
        max_cache_len=256,
        cache_pos=1,
    )
    if cycles:
        record_cycles("sdpa_partial_1pos", cycles)


@pytest.mark.quick
def test_sdpa_full_cache(dsp_mode, record_cycles):
    """Test with fully filled cache (all positions valid)."""
    set_current_test_name("sdpa_full")
    _, _, rel_err, cycles = _run_sdpa_test(
        dsp_mode,
        num_q_heads=9,
        num_kv_heads=3,
        head_dim=64,
        max_cache_len=256,
        cache_pos=256,
    )
    if cycles:
        record_cycles("sdpa_full_256pos", cycles)


@pytest.mark.quick
def test_sdpa_llama_dims(dsp_mode, record_cycles):
    """Llama-style dimensions: 32 query heads, 8 KV heads, 128 head_dim."""
    set_current_test_name("sdpa_llama")
    _, _, rel_err, cycles = _run_sdpa_test(
        dsp_mode,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_cache_len=64,
        cache_pos=32,
    )
    if cycles:
        record_cycles("sdpa_llama_32h_8kv_128d_64c", cycles)
