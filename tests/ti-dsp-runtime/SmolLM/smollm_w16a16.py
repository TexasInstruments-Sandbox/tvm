#!/usr/bin/env python3
"""SmolLM-135M W16A16 MMALIB offload via int16 non-bias matmul.

All linear projections are offloaded to the C7x MMA accelerator using
mmalib_matmul_i16 with dynamic per-tensor activation quantization.

Flow:
  1. SmoothQuant: equalize activation ranges across channels
  2. Weight-only INT8: all linear layers get QuantizedLinear
  3. torch.export → TVM Relax IR
  4. Pipeline (-mmalib=1):
     - RewriteDequantize: int8 weight pattern → R.dequantize
     - LegalizeMLPToMMALIBInt16: dequantize+matmul → int16 MMALIB
       (dynamic quantize → mmalib_matmul_i16 → dequantize)
     - Remaining ops: scalar C7x loops (RMSNorm, SiLU, softmax, RoPE)

Accuracy: ~84% top-1 at 1 layer vs float reference.
Speedup: ~9-10x vs scalar FuseDequantizeMatmul baseline.

Usage:
    python smollm_w16a16.py test --dsp-mode c7x_dload
    python smollm_w16a16.py test --dsp-mode c7x_dload --num-layers 5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.export import export

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import (  # noqa: E402
    add_board_arg,
    compile_and_run_dsp,
    get_target_string,
    set_current_test_name,
)
from smollm_c7x import SmolLMWrapper, quantize_linear  # noqa: E402

logger = logging.getLogger(__name__)
_DEFAULT_MODEL_DIR = _THIS_DIR / "model"

_CALIBRATION_PROMPTS = [
    "The capital of France is Paris, which is known for",
    "In machine learning, gradient descent minimizes the loss function by",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1)",
    "Question: What is photosynthesis?\nAnswer: Photosynthesis is the process by which",
    "The transformer architecture uses self-attention to process sequences",
    "import numpy as np\nimport torch\n\nclass Model(nn.Module):",
    "Once upon a time in a distant land, there lived a wise old",
    "The quick brown fox jumps over the lazy dog. This sentence contains",
]


def collect_act_scales(model, tokenizer, seq_len=64, n_samples=8):
    """Collect per-channel activation max magnitudes for SmoothQuant."""
    import functools

    act_scales = {}

    def hook_fn(module, input, output, name):
        x = input[0] if isinstance(input, tuple) else input
        hidden = x.shape[-1]
        channel_max = x.view(-1, hidden).abs().max(dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], channel_max)
        else:
            act_scales[name] = channel_max

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(functools.partial(hook_fn, name=name))
            )

    model.eval()
    with torch.no_grad():
        for prompt in _CALIBRATION_PROMPTS[:n_samples]:
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) >= seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens = tokens + [tokenizer.eos_token_id] * (seq_len - len(tokens))
            model(torch.tensor([tokens], dtype=torch.long))

    for h in hooks:
        h.remove()
    return act_scales


def smooth_model(model, act_scales, alpha=0.5):
    """Apply SmoothQuant (Xiao et al., ICML 2023) in-place.

    Migrates quantization difficulty from activations to weights by
    absorbing a per-channel scaling factor into the preceding RMSNorm
    and the linear layer weights. No runtime cost — just modifies weights.
    """
    count = 0
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}"
        attn_key = f"{prefix}.self_attn.q_proj"
        if attn_key in act_scales:
            _smooth_ln_fcs(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
                act_scales[attn_key], alpha,
            )
            count += 3
        mlp_key = f"{prefix}.mlp.gate_proj"
        if mlp_key in act_scales:
            _smooth_ln_fcs(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj],
                act_scales[mlp_key], alpha,
            )
            count += 2
    logger.info("SmoothQuant: smoothed %d linear layers (alpha=%.2f)", count, alpha)


def _smooth_ln_fcs(ln, fcs, act_scale, alpha):
    """Smooth one (RMSNorm, [Linear...]) group."""
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scale = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    device = fcs[0].weight.device
    act_scale = act_scale.to(device).clamp(min=1e-5)
    scales = (act_scale.pow(alpha) / weight_scale.pow(1 - alpha)).clamp(min=1e-5)
    ln.weight.data.div_(scales)
    for fc in fcs:
        fc.weight.data.mul_(scales.view(1, -1))


def create_model(model_dir, seq_len=64, num_layers=30, smooth_alpha=0.5):
    """Load SmolLM, apply SmoothQuant, quantize all weights to int8.

    The int8 weights are sign-extended to int16 by LegalizeMLPToMMALIBInt16
    at compile time. Activations are quantized dynamically to int16 at
    runtime (per-tensor scale computed from max|x|).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] Loading model ({num_layers} layers) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True
    )
    model.eval()
    model.model.layers = model.model.layers[:num_layers]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    print(f"[2/4] SmoothQuant (alpha={smooth_alpha}) ...")
    act_scales = collect_act_scales(model, tokenizer, seq_len=seq_len, n_samples=8)
    smooth_model(model, act_scales, alpha=smooth_alpha)

    print("[3/4] Weight-only INT8 quantization (all layers) ...")
    count = 0
    for layer in model.model.layers:
        layer.self_attn.q_proj = quantize_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = quantize_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = quantize_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = quantize_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = quantize_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = quantize_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = quantize_linear(layer.mlp.down_proj)
        count += 7
    model.lm_head = quantize_linear(model.lm_head)
    count += 1
    print(f"  Quantized {count} layers")

    print("[4/4] Exporting to TVM ...")
    wrapper = SmolLMWrapper(model)
    wrapper.eval()
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        float_ref = wrapper(input_ids, position_ids, attention_mask).numpy()
        ep = export(wrapper, (input_ids, position_ids, attention_mask))
        mod = from_exported_program(ep, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    wp = mod["main"].params[3:]
    mod = relax.transform.BindParams(func_name="main", params=dict(zip(wp, params["main"])))(mod)

    input_data = (
        input_ids.numpy(),
        position_ids.numpy(),
        attention_mask.numpy(),
    )
    return mod, input_data, float_ref


def cmd_test(args):
    """Compile and run SmolLM with int16 MMALIB, compare to float reference."""
    set_current_test_name(f"smollm_w16a16_{args.num_layers}L_{args.dsp_mode}")
    mod, input_data, float_ref = create_model(
        args.model_dir,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        smooth_alpha=args.smooth_alpha,
    )

    target_string = get_target_string(args.dsp_mode, use_cpp_api=True) + " -mmalib=1"
    reassoc = "--fp_reassoc=off" if args.fp_reassoc_off else "default"
    print(f"\n[Compile] target: {target_string} ({reassoc})")

    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=args.dsp_mode,
        fp_reassoc_off=args.fp_reassoc_off,
    )

    result_key = "c7x_host_result" if args.dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_output = results[result_key].reshape(float_ref.shape)
    cycles = results.get("c7x_dload_cycles", 0)

    dsp_tokens = np.argmax(dsp_output.reshape(-1, dsp_output.shape[-1]), axis=-1)
    ref_tokens = np.argmax(float_ref.reshape(-1, float_ref.shape[-1]), axis=-1)
    top1 = (dsp_tokens == ref_tokens).mean()
    cos = np.dot(dsp_output.flatten(), float_ref.flatten()) / (
        np.linalg.norm(dsp_output) * np.linalg.norm(float_ref) + 1e-10
    )
    max_diff = np.abs(dsp_output - float_ref).max()

    print(f"\n[Result] SmolLM {args.num_layers}-layer on {args.dsp_mode}:")
    print(f"  Top-1 accuracy: {top1*100:.1f}%")
    print(f"  Cosine similarity: {cos:.4f}")
    print(f"  Max logit diff: {max_diff:.2f}")
    if cycles:
        print(f"  Cycles: {cycles:,}")
        print(f"  Speedup vs baseline (47B): {47_000_000_000 / cycles:.1f}x")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SmolLM-135M W16A16 MMALIB offload (int16 non-bias matmul)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    p_test = subparsers.add_parser("test", help="Compile and run on DSP")
    p_test.add_argument(
        "--model-dir", type=Path, default=_DEFAULT_MODEL_DIR, help="Model directory"
    )
    p_test.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    p_test.add_argument("--num-layers", type=int, default=30, help="Number of layers")
    p_test.add_argument(
        "--smooth-alpha", type=float, default=0.5, dest="smooth_alpha",
        help="SmoothQuant alpha",
    )
    p_test.add_argument(
        "--dsp-mode", choices=["c7x_host", "c7x_dload"], default="c7x_dload",
        help="DSP mode",
    )
    p_test.add_argument(
        "--fp-reassoc-off", action="store_true", default=False,
        dest="fp_reassoc_off",
        help="Compile with --fp_reassoc=off (c7x_dload only)",
    )
    add_board_arg(p_test)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    if args.command == "test":
        return cmd_test(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
