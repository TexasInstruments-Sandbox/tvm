#!/usr/bin/env python
"""
Test individual transformer components on c7x_host vs c7x_dload.

Isolates which operations (RMSNorm, attention, MLP, etc.) contribute to
cross-platform float divergence.  Uses random weights — no model download
needed.

Usage:
    python test_component_divergence.py c7x_host
    python test_component_divergence.py c7x_dload

Expected result: all components show <0.1 max_diff on both modes,
confirming that individual operations are precise.  The SmolLM
divergence comes from ill-conditioned trained weights amplifying
tiny per-operation differences through 30 layers.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import export

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402
    INPUT_BIN_FILE,
    build_dsp_c7x_host,
    build_dsp_dynmod,
    compile_for_dsp,
    run_dsp_dload,
    run_dsp_host,
    set_current_board,
    write_tensors_to_file,
)


class QuantizedLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        w = torch.randn(out_f, in_f) * 0.05
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp(min=1e-10)
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale)

    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        return F.linear(x, w)


class RMSNormOnly(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.RMSNorm(dim)

    def forward(self, x):
        return self.norm(x)


class NormLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.RMSNorm(dim)
        self.linear = QuantizedLinear(dim, dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = QuantizedLinear(dim, dim)
        self.k_proj = QuantizedLinear(dim, dim)
        self.v_proj = QuantizedLinear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class SimpleMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = QuantizedLinear(dim, hidden_dim)
        self.up = QuantizedLinear(dim, hidden_dim)
        self.down = QuantizedLinear(hidden_dim, dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(dim)
        self.attn = SimpleAttention(dim)
        self.norm2 = torch.nn.RMSNorm(dim)
        self.mlp = SimpleMLP(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def compile_and_test(model, x, dsp_mode, test_name, apply_dequant=True):
    model.eval()

    with torch.no_grad():
        ref = model(x).numpy()

    with torch.no_grad():
        ep = export(model, (x,))
        mod = from_exported_program(ep, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    n_user_inputs = 1
    weight_params = mod["main"].params[n_user_inputs:]
    func_params_dict = dict(zip(weight_params, params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    if apply_dequant:
        mod = relax.transform.RewriteDequantize()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)

    target_string = "c_static -mcpu=c7x"
    artifacts = Path(f"/tmp/xfmr_test_{test_name}_{dsp_mode}")
    artifacts.mkdir(parents=True, exist_ok=True)
    generated_dir = compile_for_dsp(mod, target_string, output_dir=artifacts)

    input_data = [x.numpy()]

    if dsp_mode == "c7x_host":
        build_dir = artifacts / "build"
        exe = build_dsp_c7x_host(generated_dir, build_dir=build_dir)
        write_tensors_to_file(input_data, str(build_dir / INPUT_BIN_FILE))
        result = run_dsp_host(exe)
    elif dsp_mode == "c7x_dload":
        build_dir = artifacts / "build-dynmod"
        weights_path = generated_dir / "weights.bin"
        module_path = build_dsp_dynmod(
            generated_dir, build_dir=build_dir, weights_file=weights_path
        )
        result, _, _ = run_dsp_dload(module_path, weights_path, input_data, embedded_weights=True)
    else:
        raise ValueError(f"Unknown mode: {dsp_mode}")

    diff = np.abs(result - ref)
    cos = np.dot(result.flatten(), ref.flatten()) / (
        np.linalg.norm(result) * np.linalg.norm(ref) + 1e-10
    )
    print(f"  {test_name:20s}  max_diff={diff.max():.4e}  mean={diff.mean():.4e}  cos={cos:.6f}")
    return diff.max()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "c7x_host"
    set_current_board(sys.argv[2] if len(sys.argv) > 2 else None)
    dim = 576
    hidden = 1536
    seq_len = 16

    torch.manual_seed(42)
    np.random.seed(42)
    x = torch.randn(1, seq_len, dim)

    print(f"\n=== Component tests on {mode} (dim={dim}, seq={seq_len}) ===")

    torch.manual_seed(42)
    compile_and_test(RMSNormOnly(dim), x, mode, "rmsnorm", apply_dequant=False)

    torch.manual_seed(42)
    compile_and_test(NormLinear(dim), x, mode, "norm_linear")

    torch.manual_seed(42)
    compile_and_test(SimpleAttention(dim), x, mode, "attention")

    torch.manual_seed(42)
    compile_and_test(SimpleMLP(dim, hidden), x, mode, "mlp")

    torch.manual_seed(42)
    compile_and_test(TransformerBlock(dim, hidden), x, mode, "xfmr_block")
