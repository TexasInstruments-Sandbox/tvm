#!/usr/bin/env python3
"""Benchmark DMA tiling impact and analyze per-kernel DMA usage.

Compiles a model for c7x_host to inspect the generated code (DMA
calls per kernel), then runs on c7x_dload for real cycle counts.

Supports two comparison modes:
  --compare    Run twice (DMA vs no-DMA) and report speedup
  (default)    Single run with DMA tiling stats

Usage:
    # Analyze DMA stats + get baseline cycles for quantized ResNet-18
    python bench_dma_tiling.py --model qresnet18 --dsp-mode c7x_dload

    # Compare DMA vs no-DMA for quantized conv2d stack
    python bench_dma_tiling.py --model qconv2d_stack --dsp-mode c7x_dload --compare

    # Just analyze generated code (no hardware needed)
    python bench_dma_tiling.py --model qresnet18 --dsp-mode c7x_host
"""

import argparse
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from model_utils import create_quantized_conv2d_stack_model  # noqa: E402

import numpy as np  # noqa: E402
import tvm.relax.transform.schedule_c7x_dma as _dma_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _create_qresnet18():
    """Create quantized ResNet-18 (INT8, PT2E, 224x224 input)."""
    import torch
    from torch.export import export
    from torchvision.models.resnet import ResNet18_Weights, resnet18
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from tvm import relax
    from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(torch_model, example_args)
    model_gm = exported_program.module()

    quantizer = C7xMMAQuantizer(dtype="int8", symmetric_activations=True)
    prepared = prepare_pt2e(model_gm, quantizer)
    with torch.no_grad():
        for _ in range(10):
            prepared(torch.randn(1, 3, 224, 224, dtype=torch.float32))
    quantized_gm = convert_pt2e(prepared)

    with torch.no_grad():
        exported_program_q = export(quantized_gm, example_args)
        mod = from_exported_program(exported_program_q, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    np.random.seed(42)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        ref = quantized_gm(torch.from_numpy(input_data)).numpy()

    return mod, input_data, ref


def _create_qconv2d_stack():
    """Create quantized conv2d stack (INT8, PT2E, 56x56 input)."""
    import torch

    tvm_mod, quantized_gm, input_data = create_quantized_conv2d_stack_model()
    with torch.no_grad():
        ref = quantized_gm(torch.from_numpy(input_data)).numpy()
    return tvm_mod, input_data, ref


MODELS = {
    "qresnet18": ("Quantized ResNet-18 (224x224)", _create_qresnet18),
    "qconv2d_stack": ("Quantized Conv2D Stack (56x56)", _create_qconv2d_stack),
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_generated_code(lib0_path):
    """Parse lib0.c to find DMA calls per kernel function."""
    text = Path(lib0_path).read_text()

    # Match function definitions (ending with '{'), not forward declarations
    func_pattern = re.compile(
        r"^(?:TVM_DLL )?int(?:32_t)? (fused_\w+)\([^)]*\)\s*\{", re.MULTILINE
    )
    dma_copy_re = re.compile(r"tvm_dsp_dma_copy")
    dma_wait_re = re.compile(r"tvm_dsp_dma_wait")
    l2_alloc_re = re.compile(r"tvm_l2_alloc")

    funcs = []
    seen = set()
    matches = list(func_pattern.finditer(text))

    for i, m in enumerate(matches):
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)

        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]

        n_copy = len(dma_copy_re.findall(body))
        n_wait = len(dma_wait_re.findall(body))
        n_l2 = len(l2_alloc_re.findall(body))

        funcs.append({
            "name": name,
            "dma_copy": n_copy,
            "dma_wait": n_wait,
            "l2_alloc": n_l2,
            "has_dma": n_copy > 0,
        })

    return funcs


def print_dma_stats(funcs):
    """Print DMA tiling summary table."""
    conv_funcs = [f for f in funcs if "conv2d" in f["name"].lower()]
    dma_funcs = [f for f in funcs if f["has_dma"]]

    print(f"  Total kernel functions:  {len(funcs)}")
    print(f"  Conv2d-related kernels:  {len(conv_funcs)}")
    print(f"  Kernels with DMA tiling: {len(dma_funcs)}")
    print()

    # Print conv2d kernels with DMA status
    print(f"  {'Conv2d Kernel':<65} {'DMA':>5} {'L2':>3}")
    print(f"  {'-'*65} {'-'*5} {'-'*3}")
    for f in conv_funcs:
        dma_str = f"{f['dma_copy']}c/{f['dma_wait']}w" if f["has_dma"] else "--"
        l2_str = str(f["l2_alloc"]) if f["l2_alloc"] else "--"
        name = f["name"]
        if len(name) > 63:
            name = name[:60] + "..."
        print(f"  {name:<65} {dma_str:>5} {l2_str:>3}")

    print()
    total_copy = sum(f["dma_copy"] for f in funcs)
    total_wait = sum(f["dma_wait"] for f in funcs)
    total_l2 = sum(f["l2_alloc"] for f in funcs)
    print(f"  Total tvm_dsp_dma_copy calls: {total_copy}")
    print(f"  Total tvm_dsp_dma_wait calls: {total_wait}")
    print(f"  Total tvm_l2_alloc calls:     {total_l2}")

    return len(dma_funcs), len(conv_funcs)


def extract_cycles(dsp_results):
    """Extract cycle count from c7x_dload output."""
    stdout = dsp_results.get("c7x_dload_stdout", "")
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{") and '"cycles"' in line:
            try:
                return json.loads(line).get("cycles")
            except json.JSONDecodeError:
                pass
    m = re.search(r"Inference complete:\s*(\d+)\s*cycles", stdout)
    return int(m.group(1)) if m else None


def noop_schedule(func, l2_budget):
    """No-op: skip DMA tiling."""
    return func


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DMA tiling impact on C7x DSP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Model to benchmark",
    )
    parser.add_argument(
        "--dsp-mode",
        required=True,
        choices=["c7x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run DMA vs no-DMA comparison (2x slower)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for averaging (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300000,
        help="Execution timeout in ms (default: 300000)",
    )
    args = parser.parse_args()

    model_desc, model_fn = MODELS[args.model]

    print("=" * 70)
    print(f"DMA Tiling Benchmark: {model_desc}")
    print(f"Mode: {args.dsp_mode}")
    print("=" * 70)

    # Create model
    print("\n[1] Creating model...")
    tvm_mod, input_data, ref = model_fn()
    print(f"    Input shape:  {input_data.shape}")
    print(f"    Output shape: {ref.shape}")

    target = get_target_string(args.dsp_mode, use_cpp_api=True)

    # Step 1: Compile to get generated code (DMA analysis)
    print(f"\n[2] Compiling for {args.dsp_mode} (DMA analysis)...")
    import os
    os.environ["DSP_KEEP_TEMP"] = "1"

    res = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target,
        execution_mode=args.dsp_mode,
        timeout_ms=args.timeout,
    )

    # Find and analyze generated code
    lib0 = None
    if "generated_dir" in res:
        lib0 = res["generated_dir"] / "lib0.c"
    if not lib0 or not lib0.exists():
        import glob
        candidates = sorted(
            glob.glob("/tmp/tvm_dsp_*/lib0.c"),
            key=lambda p: Path(p).stat().st_mtime,
        )
        if candidates:
            lib0 = Path(candidates[-1])

    if lib0 and lib0.exists():
        print(f"    Generated code: {lib0}")
        funcs = analyze_generated_code(lib0)

        print(f"\n{'=' * 70}")
        print("DMA Tiling Summary")
        print(f"{'=' * 70}")
        n_dma, n_conv = print_dma_stats(funcs)
    else:
        print("    WARNING: could not find generated lib0.c")

    # Report correctness + cycles for DMA run
    result_key = f"{args.dsp_mode.replace('-', '_')}_result"
    if result_key in res:
        diff = np.max(np.abs(res[result_key] - ref))
        print(f"\n    [DMA] max_diff vs reference: {diff:.2e}")

    cycles_dma = extract_cycles(res) if args.dsp_mode == "c7x_dload" else None
    if cycles_dma:
        print(f"    [DMA] {cycles_dma:,} cycles ({cycles_dma / 1_000_000:.1f} ms @ 1 GHz)")

    # Additional DMA runs for averaging
    all_dma_cycles = [cycles_dma] if cycles_dma else []
    for i in range(1, args.runs):
        print(f"\n    [DMA] Run {i + 1}/{args.runs}...")
        r = compile_and_run_dsp(
            mod=tvm_mod, input_data=input_data, target_string=target,
            execution_mode=args.dsp_mode, timeout_ms=args.timeout,
        )
        c = extract_cycles(r)
        if c:
            all_dma_cycles.append(c)
            print(f"    [DMA] {c:,} cycles")

    # Step 2: Optional no-DMA comparison
    all_nodma_cycles = []
    if args.compare and args.dsp_mode == "c7x_dload":
        print(f"\n[3] Running WITHOUT DMA tiling ({args.runs} run(s))...")
        for i in range(args.runs):
            print(f"\n    [NO-DMA] Run {i + 1}/{args.runs}...")
            with patch.object(_dma_mod, "_schedule_conv2d_nhwc", noop_schedule), \
                 patch.object(_dma_mod, "_schedule_conv2d", noop_schedule):
                r = compile_and_run_dsp(
                    mod=tvm_mod, input_data=input_data, target_string=target,
                    execution_mode=args.dsp_mode, timeout_ms=args.timeout,
                )
            c = extract_cycles(r)
            if c:
                all_nodma_cycles.append(c)
                print(f"    [NO-DMA] {c:,} cycles")
            if result_key in r:
                diff = np.max(np.abs(r[result_key] - ref))
                print(f"    [NO-DMA] max_diff vs reference: {diff:.2e}")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {model_desc} on C7x (J722S)")
    print(f"{'=' * 70}")

    if all_dma_cycles:
        avg = sum(all_dma_cycles) / len(all_dma_cycles)
        mn = min(all_dma_cycles)
        print(f"  DMA:    avg={avg:,.0f}  min={mn:,}  ({mn / 1_000_000:.1f} ms @ 1 GHz)")
        if len(all_dma_cycles) > 1:
            print(f"          runs={all_dma_cycles}")

    if all_nodma_cycles:
        avg_n = sum(all_nodma_cycles) / len(all_nodma_cycles)
        mn_n = min(all_nodma_cycles)
        print(f"  NO-DMA: avg={avg_n:,.0f}  min={mn_n:,}  ({mn_n / 1_000_000:.1f} ms @ 1 GHz)")
        if len(all_nodma_cycles) > 1:
            print(f"          runs={all_nodma_cycles}")

    if all_dma_cycles and all_nodma_cycles:
        avg_d = sum(all_dma_cycles) / len(all_dma_cycles)
        avg_n = sum(all_nodma_cycles) / len(all_nodma_cycles)
        pct = (avg_n - avg_d) / avg_n * 100
        print(f"\n  DMA effect: {pct:+.1f}% ({'DMA faster' if pct > 0 else 'NO-DMA faster'})")
        print(f"  Ratio: {avg_n / avg_d:.3f}x")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
