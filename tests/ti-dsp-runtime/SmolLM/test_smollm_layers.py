#!/usr/bin/env python
"""
Test SmolLM-135M with truncated layers on c7x_host and c7x_dload.

Compiles the real SmolLM model (INT8 quantized) with only the first N
transformer layers and compares against PyTorch reference.  This reveals
how logit divergence grows with depth and confirms that c7x_dload error
is dominated by lm_head amplification (sigma_max=626), not layer
accumulation.

Usage:
    # Sweep layer counts on c7x_dload (needs AM67A hardware):
    python test_smollm_layers.py c7x_dload 1 2

    # Same sweep on c7x_host (no hardware needed):
    python test_smollm_layers.py c7x_host 1 2

    # FP32 vs INT8 comparison at 1 layer:
    python test_smollm_layers.py c7x_dload 1 --fp32

Expected results (c7x_dload):
    1 layer:  max_diff ~1.9  (hidden-state diff amplified by lm_head)
    2 layers: max_diff ~45   (further amplification through attention)
    4+ layers: max_diff plateaus ~31-34 (logits bounded by lm_head range)

Expected results (c7x_host):
    1 layer:  max_diff ~0.12
    30 layers: max_diff ~0.19  (error stays bounded)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import export
from transformers import AutoModelForCausalLM

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
_MODEL_DIR = _THIS_DIR / "model"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402
    INPUT_BIN_FILE,
    add_board_arg,
    build_dsp_c7x_host,
    build_dsp_dynmod,
    compile_for_dsp,
    run_dsp_dload,
    run_dsp_host,
    write_tensors_to_file,
)


class QuantizedLinear(nn.Module):
    def __init__(self, weight_int8, scale, bias=None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        return F.linear(x, w, self.bias)


def quantize_linear(linear):
    weight = linear.weight.data
    scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-10)
    weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return QuantizedLinear(weight_int8, scale, linear.bias)


def quantize_linears_recursive(module, prefix="", skip_lm_head=True):
    count = 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if skip_lm_head and name == "lm_head":
                continue
            setattr(module, name, quantize_linear(child))
            count += 1
        else:
            count += quantize_linears_recursive(child, full_name, skip_lm_head)
    return count


class SmolLMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


def test_n_layers(dsp_mode, n_layers, seq_len=16, quantize=True):
    torch.manual_seed(42)
    np.random.seed(42)

    model = AutoModelForCausalLM.from_pretrained(
        str(_MODEL_DIR), dtype=torch.float32, local_files_only=True
    )
    model.eval()
    model.model.layers = model.model.layers[:n_layers]

    if quantize:
        quantize_linears_recursive(model)

    wrapper = SmolLMWrapper(model)
    wrapper.eval()

    input_ids = torch.randint(0, 1000, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    example_args = (input_ids, position_ids, attention_mask)

    with torch.no_grad():
        ref = wrapper(*example_args).numpy()

    with torch.no_grad():
        ep = export(wrapper, example_args)
        mod = from_exported_program(ep, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    n_user_inputs = 3
    weight_params = mod["main"].params[n_user_inputs:]
    func_params_dict = dict(zip(weight_params, params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    if quantize:
        mod = relax.transform.RewriteDequantize()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)

    q_label = "INT8" if quantize else "FP32"
    target_string = "c_static -mcpu=c7x"
    artifacts = Path(f"/tmp/smollm_trunc_{q_label}_{dsp_mode}_{n_layers}L")
    artifacts.mkdir(parents=True, exist_ok=True)
    generated_dir = compile_for_dsp(mod, target_string, output_dir=artifacts)

    input_data = [input_ids.numpy(), position_ids.numpy(), attention_mask.numpy()]

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
    print(
        f"  {q_label} {n_layers:2d}L  max_diff={diff.max():.4e}"
        f"  mean={diff.mean():.4e}  cos={cos:.6f}"
    )
    return diff.max()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmolLM truncated layer test")
    parser.add_argument("dsp_mode", choices=["c7x_host", "c7x_dload"], help="DSP execution mode")
    parser.add_argument("layers", nargs="*", type=int, default=[1, 2], help="Layer counts to test")
    parser.add_argument("--fp32", action="store_true", help="Also test FP32")
    add_board_arg(parser)
    args = parser.parse_args()

    print(f"\n=== SmolLM-135M truncated on {args.dsp_mode} ===")
    for n in args.layers:
        test_n_layers(args.dsp_mode, n, quantize=True)
        if args.fp32:
            test_n_layers(args.dsp_mode, n, quantize=False)
