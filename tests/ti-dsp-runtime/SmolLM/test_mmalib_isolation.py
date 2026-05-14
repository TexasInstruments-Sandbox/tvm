#!/usr/bin/env python3
"""Isolate MMALIB int16 accuracy per linear layer on c7x_dload.

Tests each of the 8 MMALIB matmul_i16 calls individually with real
SmolLM weights to identify which layers produce incorrect results on
hardware vs the numpy reference.

For each linear layer (q/k/v/o_proj, gate/up/down_proj, lm_head):
  1. Extract real int8 weight and per-channel scale from SmoothQuant model
  2. Sign-extend to int16, compute shift via _compute_shift()
  3. Build a minimal TVM model: int16 input → mmalib_matmul_i16 → int16 output
  4. Quantize a deterministic float input to int16, feed to DSP
  5. Compare raw int16 output against numpy reference
  6. Also dequantize both and compare float error
  7. Report per-layer max_diff and PASS/FAIL

This uses the same model pattern as test_mmalib_matmul_i16_dsp.py (proven
to work on both c7x_host and c7x_dload) but with real SmolLM weights
instead of random data.

Usage:
    python test_mmalib_isolation.py --dsp-mode c7x_dload
    python test_mmalib_isolation.py --dsp-mode c7x_host
    python test_mmalib_isolation.py --dsp-mode c7x_dload --layer q_proj
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tvm import relax, te, tir
from tvm.relax import TensorStructInfo

_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
_MODEL_DIR = _THIS_DIR / "model"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from smollm_c7x import quantize_linear  # noqa: E402

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

MMA_SIZE_I16 = 16


def _compute_shift(w_i16_KN):
    """Compute global shift to prevent int16 output overflow (same as pass)."""
    row_l1 = np.abs(w_i16_KN).sum(axis=0)
    max_l1 = int(row_l1.max())
    max_accum = 32767 * max_l1
    if max_accum <= 32767:
        return 0
    return math.ceil(math.log2(max_accum / 32767))


def _build_i16_matmul_model(M, K, N, w_i16_KN, shift):
    """Build a minimal TVM model: int16 input → mmalib_matmul_i16 → int16 output.

    Matches the pattern from test_mmalib_matmul_i16_dsp.py which is proven
    to work on both c7x_host and c7x_dload.
    """
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
                M, K, N, shift,
            )
        return te.extern(
            [M, N], [data_t, w_t], fcompute, name="matmul_i16", dtype="int16"
        )

    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit(
                bb.call_te(
                    te_matmul, x, relax.Constant(w_i16_KN),
                    primfunc_name_hint="matmul_i16",
                )
            )
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize()


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


def _smooth_ln_fcs(ln, fcs, act_scale, alpha):
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


def smooth_model(model, act_scales, alpha=0.5):
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}"
        attn_key = f"{prefix}.self_attn.q_proj"
        if attn_key in act_scales:
            _smooth_ln_fcs(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
                act_scales[attn_key], alpha,
            )
        mlp_key = f"{prefix}.mlp.gate_proj"
        if mlp_key in act_scales:
            _smooth_ln_fcs(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj],
                act_scales[mlp_key], alpha,
            )


def load_quantized_model(model_dir, num_layers=1, smooth_alpha=0.5, seq_len=64):
    """Load SmolLM, apply SmoothQuant + int8 quantization, return layer weights."""
    print(f"Loading model ({num_layers} layers) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True
    )
    model.eval()
    model.model.layers = model.model.layers[:num_layers]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    print(f"SmoothQuant (alpha={smooth_alpha}) ...")
    act_scales = collect_act_scales(model, tokenizer, seq_len=seq_len, n_samples=8)
    smooth_model(model, act_scales, alpha=smooth_alpha)

    print("INT8 quantization ...")
    layers_info = {}
    layer = model.model.layers[0]

    layer.self_attn.q_proj = quantize_linear(layer.self_attn.q_proj)
    layer.self_attn.k_proj = quantize_linear(layer.self_attn.k_proj)
    layer.self_attn.v_proj = quantize_linear(layer.self_attn.v_proj)
    layer.self_attn.o_proj = quantize_linear(layer.self_attn.o_proj)
    layer.mlp.gate_proj = quantize_linear(layer.mlp.gate_proj)
    layer.mlp.up_proj = quantize_linear(layer.mlp.up_proj)
    layer.mlp.down_proj = quantize_linear(layer.mlp.down_proj)
    model.lm_head = quantize_linear(model.lm_head)

    for name, mod in [
        ("q_proj", layer.self_attn.q_proj),
        ("k_proj", layer.self_attn.k_proj),
        ("v_proj", layer.self_attn.v_proj),
        ("o_proj", layer.self_attn.o_proj),
        ("gate_proj", layer.mlp.gate_proj),
        ("up_proj", layer.mlp.up_proj),
        ("down_proj", layer.mlp.down_proj),
        ("lm_head", model.lm_head),
    ]:
        w_int8 = mod.weight_int8.numpy()
        w_scale = mod.scale.numpy().flatten()
        N, K = w_int8.shape
        layers_info[name] = {
            "w_int8": w_int8,
            "w_scale": w_scale,
            "N": N,
            "K": K,
        }

    return layers_info


def test_layer(layer_name, layer_info, M, dsp_mode):
    """Test one MMALIB layer in isolation on DSP.

    Uses int16 input/output (no float ops in the graph) to isolate the
    MMALIB kernel behavior from any surrounding code.
    """
    w_int8 = layer_info["w_int8"]
    w_scale = layer_info["w_scale"]
    N = layer_info["N"]
    K = layer_info["K"]

    # Check alignment
    if M % MMA_SIZE_I16 != 0 or K % MMA_SIZE_I16 != 0 or N % MMA_SIZE_I16 != 0:
        print(f"  {layer_name:12s}: SKIP (not aligned: M={M} K={K} N={N})")
        return None

    # Sign-extend int8 -> int16
    w_i16 = w_int8.astype(np.int16)
    w_i16_KN = np.ascontiguousarray(w_i16.T)  # [K, N]

    # Compute shift
    shift = _compute_shift(w_i16_KN)

    # Create deterministic float32 input, then quantize to int16
    np.random.seed(42)
    x_float = np.random.uniform(-5.0, 5.0, (M, K)).astype(np.float32)
    x_scale = float(np.abs(x_float).max()) / 32767.0
    x_scale = max(x_scale, 1e-10)
    x_i16 = np.clip(np.round(x_float / x_scale), -32768, 32767).astype(np.int16)

    # Numpy reference: int16 matmul with shift
    accum = x_i16.astype(np.int64) @ w_i16_KN.astype(np.int64)
    ref_i16 = np.clip(accum >> shift, -32768, 32767).astype(np.int16)

    # Build TVM model (int16 in → int16 out)
    mod = _build_i16_matmul_model(M, K, N, w_i16_KN, shift)

    # Compile and run
    target_string = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=x_i16,
        target_string=target_string,
        execution_mode=dsp_mode,
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_i16 = results[result_key].astype(np.int16).reshape(M, N)

    # Compare raw int16 output
    i16_diff = np.abs(dsp_i16.astype(np.int32) - ref_i16.astype(np.int32))
    i16_max_diff = int(i16_diff.max())
    i16_mean_diff = float(i16_diff.mean())

    # Also compare dequantized float output for context
    dequant_scale = (1 << shift) * x_scale * w_scale.reshape(1, N)
    dsp_float = dsp_i16.astype(np.float64) * dequant_scale
    ref_float = ref_i16.astype(np.float64) * dequant_scale
    float_diff = np.abs(dsp_float - ref_float)
    float_max_diff = float(float_diff.max())

    # MMALIB rounds the shifted accumulator differently from Python's >>
    # (which truncates toward -inf). Allow ±1 LSB difference.
    passed = i16_max_diff <= 1
    status = "PASS" if passed else "FAIL"

    print(
        f"  {layer_name:12s}: [{status}] i16_max_diff={i16_max_diff} "
        f"i16_mean={i16_mean_diff:.2f} float_max_diff={float_max_diff:.4f} "
        f"(M={M} K={K} N={N} shift={shift})"
    )

    if not passed:
        # Print diagnostic details
        worst_idx = np.unravel_index(i16_diff.argmax(), i16_diff.shape)
        print(
            f"    Worst at [{worst_idx[0]}, {worst_idx[1]}]: "
            f"dsp={dsp_i16[worst_idx]} ref={ref_i16[worst_idx]} "
            f"diff={i16_diff[worst_idx]}"
        )
        # Sample first row
        print(f"    DSP i16[0,:8]:  {dsp_i16[0, :8]}")
        print(f"    Ref i16[0,:8]:  {ref_i16[0, :8]}")
        # Check position 32 (from plan: position-dependent errors)
        if M > 32:
            print(f"    DSP i16[32,:8]: {dsp_i16[32, :8]}")
            print(f"    Ref i16[32,:8]: {ref_i16[32, :8]}")
        # Error distribution across rows
        row_max_diff = i16_diff.max(axis=1)
        nonzero_rows = np.where(row_max_diff > 0)[0]
        if len(nonzero_rows) <= 10:
            print(f"    Rows with errors: {nonzero_rows.tolist()}")
        else:
            print(
                f"    Rows with errors: {len(nonzero_rows)}/{M} "
                f"(first 5: {nonzero_rows[:5].tolist()})"
            )
        print(f"    Per-row max diff: min={row_max_diff.min()} max={row_max_diff.max()}")

    return {
        "i16_max_diff": i16_max_diff,
        "i16_mean_diff": i16_mean_diff,
        "float_max_diff": float_max_diff,
        "passed": passed,
        "shift": shift,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MMALIB int16 isolation test per linear layer"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=_MODEL_DIR, help="Model directory"
    )
    parser.add_argument(
        "--dsp-mode", choices=["c7x_host", "c7x_dload"], default="c7x_dload",
        help="DSP execution mode",
    )
    parser.add_argument(
        "--seq-len", type=int, default=64, help="M dimension (seq_len)"
    )
    parser.add_argument(
        "--layer", type=str, default=None,
        help="Test only this layer (e.g., q_proj, lm_head)",
    )
    args = parser.parse_args()

    M = args.seq_len

    print("=" * 70)
    print(f"MMALIB Int16 Isolation Test — {args.dsp_mode}, M={M}")
    print("=" * 70)

    layers_info = load_quantized_model(
        args.model_dir, num_layers=1, smooth_alpha=0.5, seq_len=M
    )

    print(f"\nTesting {len(layers_info)} layers on {args.dsp_mode}:\n")

    results = {}
    all_layers = list(layers_info.keys())
    if args.layer:
        if args.layer not in all_layers:
            print(f"ERROR: unknown layer '{args.layer}'. Available: {all_layers}")
            return 1
        all_layers = [args.layer]

    for layer_name in all_layers:
        result = test_layer(layer_name, layers_info[layer_name], M, args.dsp_mode)
        if result is not None:
            results[layer_name] = result

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    n_pass = sum(1 for r in results.values() if r["passed"])
    n_fail = sum(1 for r in results.values() if not r["passed"])
    print(f"  PASS: {n_pass}  FAIL: {n_fail}")
    if n_fail > 0:
        print("  Failed layers:")
        for name, r in results.items():
            if not r["passed"]:
                print(f"    {name}: i16_max_diff={r['i16_max_diff']}")
    print("=" * 70)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
