#!/usr/bin/env python3
"""Test SmolLM with MMALIB on exactly ONE linear layer at a time.

Compiles the full SmolLM 1-layer model (SmoothQuant + INT8) with the
LegalizeMLPToMMALIBInt16 pass restricted to convert only one specific
linear layer. All other layers remain as scalar FuseDequantizeMatmul.

This isolates whether the MMALIB bug is:
  (a) In a single MMALIB call's interaction with surrounding scalar code
  (b) In how multiple MMALIB calls interact (memory reuse, stale data)

Usage:
    # Test each layer individually on c7x_dload:
    python test_mmalib_single_layer.py --dsp-mode c7x_dload

    # Test only layer index 0 (q_proj):
    python test_mmalib_single_layer.py --dsp-mode c7x_dload --layer-idx 0

    # Compare with all-MMALIB:
    python test_mmalib_single_layer.py --dsp-mode c7x_dload --all-mmalib
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import add_board_arg, compile_and_run_dsp, get_target_string  # noqa: E402
from smollm_c7x import SmolLMWrapper, quantize_linear  # noqa: E402
from smollm_w16a16 import (  # noqa: E402
    collect_act_scales,
    smooth_model,
)

logger = logging.getLogger(__name__)
_DEFAULT_MODEL_DIR = _THIS_DIR / "model"

# SmolLM layer order as targeted by LegalizeMLPToMMALIBInt16:
# The pass processes matmul ops in IR order, which corresponds to
# the model's forward execution order.
LAYER_NAMES = [
    "q_proj",      # 0: M=64, K=576, N=576
    "k_proj",      # 1: M=64, K=576, N=192
    "v_proj",      # 2: M=64, K=576, N=192
    "o_proj",      # 3: M=64, K=576, N=576
    "gate_proj",   # 4: M=64, K=576, N=1536
    "up_proj",     # 5: M=64, K=576, N=1536
    "down_proj",   # 6: M=64, K=1536, N=576
    "lm_head",     # 7: M=64, K=576, N=49152
]


def _make_selective_mmalib_pass(target_indices):
    """Create a pass that only converts specific layer indices to MMALIB.

    Uses the same pattern matching as LegalizeMLPToMMALIBInt16 but skips
    layers whose index is not in target_indices.
    """
    from tvm.ir.module import IRModule
    from tvm.ir.transform import PassContext
    from tvm.relax.transform.ti_mmalib_i16_fc import (
        _MMALIBInt16FCMutator,
        _pre_scan_bindings,
    )

    @tvm.transform.module_pass(opt_level=0, name="SelectiveMMALIBInt16")
    class SelectiveMMALIBInt16:
        def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
            binding_map = {}
            for _, func in mod.functions_items():
                if isinstance(func, relax.Function):
                    if "Composite" in (func.attrs or {}):
                        continue
                    binding_map = _pre_scan_bindings(func)
                    break

            lowerer = _MMALIBInt16FCMutator(mod, binding_map)

            # Monkey-patch to only convert specified indices
            orig_try_lower = lowerer._try_lower_matmul

            def selective_try_lower(call):
                # Check what the current count would be if we match
                result = orig_try_lower(call)
                if result is not None:
                    # lowerer.count was already incremented by orig_try_lower
                    current_idx = lowerer.count - 1
                    if current_idx not in target_indices:
                        # Revert: decrement count, return None to skip
                        lowerer.count -= 1
                        return None
                return result

            lowerer._try_lower_matmul = selective_try_lower

            for gv, func in mod.functions_items():
                if isinstance(func, relax.Function):
                    if "Composite" in (func.attrs or {}):
                        continue
                    func = lowerer.visit_expr(func)
                    lowerer.builder_.update_func(gv, func)
            mod = lowerer.builder_.get()

            if lowerer.count > 0:
                mod = relax.transform.DeadCodeElimination()(mod)

            return mod

    return SelectiveMMALIBInt16()


def create_model(model_dir, seq_len=64, num_layers=1, smooth_alpha=0.5):
    """Load SmolLM, SmoothQuant, INT8 quantize, export to Relax (no MMALIB pass)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True
    )
    model.eval()
    model.model.layers = model.model.layers[:num_layers]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    act_scales = collect_act_scales(model, tokenizer, seq_len=seq_len, n_samples=8)
    smooth_model(model, act_scales, alpha=smooth_alpha)

    for layer in model.model.layers:
        layer.self_attn.q_proj = quantize_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = quantize_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = quantize_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = quantize_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = quantize_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = quantize_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = quantize_linear(layer.mlp.down_proj)
    model.lm_head = quantize_linear(model.lm_head)

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
    mod = relax.transform.BindParams(
        func_name="main", params=dict(zip(wp, params["main"]))
    )(mod)

    input_data = (
        input_ids.numpy(),
        position_ids.numpy(),
        attention_mask.numpy(),
    )
    return mod, input_data, float_ref


def run_with_mmalib(mod, input_data, float_ref, target_indices, dsp_mode, label):
    """Apply selective MMALIB pass and run on DSP."""
    # Apply RewriteDequantize first (needed for pattern matching)
    test_mod = relax.transform.RewriteDequantize()(mod)
    test_mod = relax.transform.DeadCodeElimination()(test_mod)

    # Apply selective MMALIB pass
    if target_indices is None:
        # All layers
        test_mod = relax.transform.LegalizeMLPToMMALIBInt16()(test_mod)
    else:
        selective_pass = _make_selective_mmalib_pass(target_indices)
        test_mod = selective_pass(test_mod)

    # Compile and run
    target_string = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=test_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_output = results[result_key].reshape(float_ref.shape)

    dsp_tokens = np.argmax(dsp_output.reshape(-1, dsp_output.shape[-1]), axis=-1)
    ref_tokens = np.argmax(float_ref.reshape(-1, float_ref.shape[-1]), axis=-1)
    top1 = (dsp_tokens == ref_tokens).mean()
    cos = np.dot(dsp_output.flatten(), float_ref.flatten()) / (
        np.linalg.norm(dsp_output) * np.linalg.norm(float_ref) + 1e-10
    )
    max_diff = np.abs(dsp_output - float_ref).max()

    passed = top1 > 0.8
    status = "PASS" if passed else "FAIL"

    print(
        f"  {label:30s}: [{status}] top1={top1*100:.1f}% "
        f"cos={cos:.4f} max_diff={max_diff:.2f}"
    )

    return {"top1": top1, "cos": cos, "max_diff": max_diff, "passed": passed}


def main():
    parser = argparse.ArgumentParser(
        description="SmolLM single-MMALIB-layer test"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=_DEFAULT_MODEL_DIR, help="Model directory"
    )
    parser.add_argument(
        "--dsp-mode", choices=["c7x_host", "c7x_dload"], default="c7x_dload",
    )
    parser.add_argument(
        "--seq-len", type=int, default=64,
    )
    parser.add_argument(
        "--layer-idx", type=int, default=None,
        help="Test only this layer index (0-7). If not set, tests all.",
    )
    parser.add_argument(
        "--all-mmalib", action="store_true",
        help="Also test with all layers as MMALIB (reproduces the bug).",
    )
    add_board_arg(parser)
    args = parser.parse_args()

    print("=" * 70)
    print(f"SmolLM Single-MMALIB-Layer Test — {args.dsp_mode}")
    print("=" * 70)

    print("\n[1/2] Preparing model ...")
    mod, input_data, float_ref = create_model(
        args.model_dir, seq_len=args.seq_len, num_layers=1
    )

    print(f"\n[2/2] Testing on {args.dsp_mode}:\n")

    results = {}

    if args.layer_idx is not None:
        indices = [args.layer_idx]
    else:
        indices = list(range(len(LAYER_NAMES)))

    for idx in indices:
        name = LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"layer_{idx}"
        label = f"only {name} (idx={idx})"
        result = run_with_mmalib(
            mod, input_data, float_ref, {idx}, args.dsp_mode, label
        )
        results[name] = result

    if args.all_mmalib:
        result = run_with_mmalib(
            mod, input_data, float_ref, None, args.dsp_mode, "ALL MMALIB"
        )
        results["ALL"] = result

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {name:15s}: [{status}] top1={r['top1']*100:.1f}%")
    print("=" * 70)

    n_fail = sum(1 for r in results.values() if not r["passed"])
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
