"""
FuseDequantizeMatmul benchmark — INT8 weight-only matmul at SmolLM dimensions.

Measures cycle count for the scalar FuseDequantizeMatmul kernel on C7x
hardware. This is the weight-only INT8 path (activations stay float32)
that gives full accuracy. Used as a baseline before vectorization/DMA
optimization.

Usage:
    pytest test_dequantize_matmul_dsp.py -v --dsp-mode=c7x_dload
    pytest test_dequantize_matmul_dsp.py -v --dsp-mode=c7x_host
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.export import export

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402
    compile_and_run_dsp,
    get_target_string,
    set_current_test_name,
)


class QuantizedLinear(nn.Module):
    def __init__(self, weight_int8, scale):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)

    def forward(self, x):
        return x @ (self.weight_int8.float() * self.scale).T


def _build_dequantize_matmul_model(M, K, N, seed=42):
    """Build a minimal model with one FuseDequantizeMatmul kernel.

    Returns (tvm_mod, input_data, ref_output, cycles_name).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    w_float = rng.uniform(-0.1, 0.1, (N, K)).astype(np.float32)
    scale = np.abs(w_float).max(axis=1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-10).astype(np.float32)
    w_int8 = np.clip(np.round(w_float / scale), -128, 127).astype(np.int8)

    model = QuantizedLinear(
        torch.from_numpy(w_int8), torch.from_numpy(scale)
    )
    model.eval()

    x = torch.from_numpy(rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32))

    with torch.no_grad():
        ref = model(x).numpy()

    with torch.no_grad():
        ep = export(model, (x,))
        mod = from_exported_program(ep, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    weight_params = mod["main"].params[1:]
    mod = relax.transform.BindParams(
        func_name="main", params=dict(zip(weight_params, params["main"]))
    )(mod)

    mod = relax.transform.RewriteDequantize()(mod)
    mod = relax.transform.DeadCodeElimination()(mod)

    return mod, x.numpy(), ref


def _run_test(dsp_mode, M, K, N, name_suffix):
    """Compile, run, verify, return cycles."""
    test_name = f"dequantize_matmul_{M}x{K}x{N}_{dsp_mode}"
    set_current_test_name(test_name)
    try:
        mod, input_data, ref = _build_dequantize_matmul_model(M, K, N)

        target = get_target_string(dsp_mode, use_cpp_api=True)
        results = compile_and_run_dsp(
            mod=mod,
            input_data=input_data,
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

        print(f"\n  {name_suffix}: M={M} K={K} N={N}")
        print(f"    max_diff={max_diff:.4e}  rel_err={rel_err:.6f}")
        if cycles:
            print(f"    cycles={cycles:,}")

        assert rel_err < 0.01, f"Relative error too large: {rel_err:.4f}"
        return cycles
    finally:
        set_current_test_name(None)


@pytest.mark.c7x_only
def test_dequantize_matmul_q_proj(dsp_mode, record_cycles):
    """FuseDequantizeMatmul at q_proj size (M=64, K=576, N=576)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    cycles = _run_test(dsp_mode, M=64, K=576, N=576, name_suffix="q_proj")
    record_cycles("dequantize_matmul_576x576", cycles)


@pytest.mark.c7x_only
def test_dequantize_matmul_gate_proj(dsp_mode, record_cycles):
    """FuseDequantizeMatmul at gate_proj size (M=64, K=576, N=1536)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    cycles = _run_test(dsp_mode, M=64, K=576, N=1536, name_suffix="gate_proj")
    record_cycles("dequantize_matmul_576x1536", cycles)


@pytest.mark.c7x_only
def test_dequantize_matmul_down_proj(dsp_mode, record_cycles):
    """FuseDequantizeMatmul at down_proj size (M=64, K=1536, N=576)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("Requires c7x_host or c7x_dload")
    cycles = _run_test(dsp_mode, M=64, K=1536, N=576, name_suffix="down_proj")
    record_cycles("dequantize_matmul_1536x576", cycles)
