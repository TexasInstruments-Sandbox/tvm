#!/usr/bin/env python
"""
SmolLM-135M C7x test — float32 and INT8 weight-only quantization.

Compiles SmolLM-135M-Instruct to the c_static C7x backend, builds for
C7x host emulation or DLOAD hardware, and compares logits against the
PyTorch reference.

Prerequisites:
    export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS

Download weights once (if not already present):
    python -c "
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    m = AutoModelForCausalLM.from_pretrained(
        'HuggingFaceTB/SmolLM-135M-Instruct', dtype=torch.float32)
    t = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M-Instruct')
    m.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
    t.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
    "

Usage:
    # Float32 on c7x_host (~621 MB weights)
    python smollm_c7x_host.py

    # INT8 weight-only on c7x_host (~237 MB weights)
    python smollm_c7x_host.py --quantize

    # INT8 on AM67A hardware
    python smollm_c7x_host.py --quantize --dsp-mode c7x_dload

    # Options
    python smollm_c7x_host.py --seq-len 32
    python smollm_c7x_host.py -v          # verbose (debug logging)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import export
from transformers import AutoModelForCausalLM

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402, I001
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)

logger = logging.getLogger(__name__)

# Default local model directory (sibling of this script)
_DEFAULT_MODEL_DIR = _THIS_DIR / "model"


# ---------------------------------------------------------------------------
# Wrapper — provides explicit position_ids and attention_mask so the
# exported graph avoids cumsum (not yet lowerable to TIR).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Manual per-channel INT8 weight quantization
# ---------------------------------------------------------------------------


class QuantizedLinear(nn.Module):
    """Linear layer with INT8 per-channel weight quantization.

    Stores weight_int8 (int8) and scale (float32) as buffers.
    Forward computes: F.linear(x, weight_int8.float() * scale, bias)

    The int8.float() * scale pattern is what RewriteDequantize rewrites
    into R.dequantize() to prevent TVM constant folding.
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias=None):
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


def quantize_linear(linear: nn.Linear) -> QuantizedLinear:
    """Quantize a single nn.Linear to per-channel INT8."""
    weight = linear.weight.data  # [out_features, in_features]
    # Per-channel (axis=0): one scale per output channel
    scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-10)
    weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return QuantizedLinear(weight_int8, scale, linear.bias)


def quantize_linears(module: nn.Module) -> nn.Module:
    """Replace all nn.Linear layers with QuantizedLinear (in-place).

    Skips the lm_head (embedding) layer to keep it in float32, as it
    is shared with the input embedding in SmolLM.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Skip lm_head — keep embedding weights in float32
            if name == "lm_head":
                logger.info("  Skipping %s (lm_head)", name)
                continue
            q = quantize_linear(child)
            setattr(module, name, q)
            count += 1
        else:
            sub_count = _quantize_linears_recursive(child, name)
            count += sub_count
    print(f"  Quantized {count} Linear layers to INT8")
    return module


def _quantize_linears_recursive(module: nn.Module, prefix: str) -> int:
    count = 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}"
        if isinstance(child, nn.Linear):
            q = quantize_linear(child)
            setattr(module, name, q)
            count += 1
            logger.debug("  Quantized %s", full_name)
        else:
            count += _quantize_linears_recursive(child, full_name)
    return count


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_smollm_model(
    model_dir: Path = _DEFAULT_MODEL_DIR,
    seq_len: int = 16,
    seed: int = 42,
    quantize: bool = False,
) -> tuple:
    """
    Load SmolLM-135M, export to Relax, and bind parameters.

    Args:
        model_dir: Path to local HuggingFace model directory.
        seq_len: Input sequence length.
        seed: Random seed for reproducibility.
        quantize: If True, apply per-channel INT8 weight quantization
            and run the RewriteDequantize pass.

    Returns:
        (tvm_mod, input_data_tuple, ref_output)
        - tvm_mod: IRModule with params bound, ready for compile_for_dsp
        - input_data_tuple: tuple of numpy arrays (input_ids, position_ids, attention_mask)
        - ref_output: PyTorch reference logits as numpy array
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local model not found at {model_dir}.\n"
            "Download once with:\n"
            "  python -c \"\n"
            "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "  import torch\n"
            "  m = AutoModelForCausalLM.from_pretrained("
            "'HuggingFaceTB/SmolLM-135M-Instruct', dtype=torch.float32)\n"
            "  t = AutoTokenizer.from_pretrained("
            "'HuggingFaceTB/SmolLM-135M-Instruct')\n"
            f"  m.save_pretrained('{model_dir}')\n"
            f"  t.save_pretrained('{model_dir}')\n"
            '  "'
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load from local directory — no network access
    print(f"  Loading model from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True
    )
    model.eval()

    # Apply quantization before wrapping
    if quantize:
        print("  Applying per-channel INT8 weight quantization ...")
        quantize_linears(model)

    wrapper = SmolLMWrapper(model)
    wrapper.eval()

    # Deterministic inputs
    input_ids = torch.randint(0, 1000, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    example_args = (input_ids, position_ids, attention_mask)

    # PyTorch reference
    print("  Running PyTorch reference ...")
    with torch.no_grad():
        ref_output = wrapper(*example_args).numpy()

    # Export to Relax
    print("  torch.export ...")
    with torch.no_grad():
        ep = export(wrapper, example_args)
        mod = from_exported_program(ep, keep_params_as_input=True)

    # Bind parameters — 3 user inputs come first, rest are weights
    print("  Binding parameters ...")
    mod, params = relax.frontend.detach_params(mod)
    n_user_inputs = 3  # input_ids, position_ids, attention_mask
    weight_params = mod["main"].params[n_user_inputs:]
    func_params_dict = dict(zip(weight_params, params["main"]))
    mod = relax.transform.BindParams(  # pyright: ignore[reportArgumentType]
        func_name="main", params=func_params_dict
    )(mod)

    # Apply RewriteDequantize pass for quantized models
    if quantize:
        print("  Running RewriteDequantize pass ...")
        mod = relax.transform.RewriteDequantize()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)

    input_data = (
        input_ids.numpy(),
        position_ids.numpy(),
        attention_mask.numpy(),
    )

    return mod, input_data, ref_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SmolLM-135M C7x test")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        help="Path to local SmolLM model directory",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence length (default: 16)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply per-channel INT8 weight-only quantization",
    )
    parser.add_argument(
        "--dsp-mode",
        choices=["c7x_host", "c7x_dload"],
        default="c7x_host",
        help="DSP execution mode (default: c7x_host)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    mode_label = args.dsp_mode
    quant_label = "INT8" if args.quantize else "FP32"
    print("=" * 70)
    print(f"SmolLM-135M  {quant_label}  {mode_label}")
    print("=" * 70)

    # Step 1: Create model
    print("\n[1/3] Preparing model ...")
    tvm_mod, input_data, ref_output = create_smollm_model(
        model_dir=args.model_dir,
        seq_len=args.seq_len,
        quantize=args.quantize,
    )
    print(f"  Input shapes: {[a.shape for a in input_data]}")
    print(f"  Ref output shape: {ref_output.shape}")
    print(f"  Ref output range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")

    # Step 2: Compile and run
    target_string = get_target_string(args.dsp_mode, use_cpp_api=True)
    print(f"\n[2/3] Compile + run (target: {target_string}) ...")

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=args.dsp_mode,
        timeout_ms=300000,
    )

    # Step 3: Compare
    result_key = f"{args.dsp_mode}_result"
    error_key = f"{args.dsp_mode}_error"

    # Float32 accumulates error through 30 transformer layers; allow ~0.5
    # INT8 weight quantization adds another ~2.0 on top
    atol = 2.5 if args.quantize else 0.5
    rtol = 1e-1 if args.quantize else 5e-2

    print("\n[3/3] Results")
    if result_key in dsp_results:
        result = dsp_results[result_key]
        print(f"  DSP output shape: {result.shape}")
        print(f"  DSP output range: [{result.min():.4f}, {result.max():.4f}]")
        comparison = compare_results(
            dsp_results, ref_output, "PyTorch", rtol=rtol, atol=atol
        )
        passed_key = f"{args.dsp_mode}_vs_ref_passed"
        passed = comparison.get(passed_key, False)
        max_diff = np.max(np.abs(result - ref_output))
        status = "PASS" if passed else "FAIL"
        print(f"  Max abs diff: {max_diff:.2e}  (atol={atol}, rtol={rtol})  [{status}]")
    elif error_key in dsp_results:
        print(f"  ERROR: {dsp_results[error_key]}")
        passed = False
    else:
        print(f"  Unexpected result keys: {list(dsp_results.keys())}")
        passed = False

    print("\n" + "=" * 70)
    print(f"  {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
