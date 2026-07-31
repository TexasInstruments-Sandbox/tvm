#!/usr/bin/env python
"""
Quantized Conv2D Stack DSP test.

Tests an INT8 quantized 4-layer conv2d + batch_norm + relu stack on DSP
comparing against PyTorch quantized reference. Uses PT2E static
quantization with C7xMMAQuantizer to produce a QDQ graph.

The model isolates the most expensive conv2d configurations from
ResNet18 without skip connections, making the generated code easy to
inspect for DMA and tiling optimization.

Usage:
    # Run with C66x host emulation
    pytest test_quantized_conv2d_stack_dsp.py -v

    # Run via DLOAD pipeline (c7x_compute)
    pytest test_quantized_conv2d_stack_dsp.py -v --dsp-mode=c7x_dload

    # Run as standalone script
    python test_quantized_conv2d_stack_dsp.py --dsp-mode c66x_host
    python test_quantized_conv2d_stack_dsp.py --dsp-mode c7x_dload
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    add_board_arg,
    assert_dsp_comparison,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)
from model_utils import create_quantized_conv2d_stack_model  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_quantized_conv2d_stack_dsp_test(
    dsp_mode: str,
    timeout_ms: int = 60000,
    profile_layers: bool = False,
    use_cpp_api: bool = False,
    save_artifacts: str | None = None,
) -> dict:
    """
    Run quantized Conv2D stack model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c66x_host", "c7x_host", "c7x_dload", or "c66x")
        timeout_ms: Hardware execution timeout
        profile_layers: Enable per-layer cycle profiling
        use_cpp_api: Enable direct VM builtin calls (bypass FFI)
        save_artifacts: Directory to copy build artifacts to

    Returns:
        Dictionary with test results and comparison metrics
    """
    # Create model
    tvm_mod, quantized_gm, input_data = create_quantized_conv2d_stack_model()

    # Run PyTorch quantized reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = quantized_gm(torch_input).numpy()

    # Build target string with optional flags
    target_string = get_target_string(
        dsp_mode, profile_layers=profile_layers, use_cpp_api=use_cpp_api
    )

    # Run on DSP
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
    )

    # Copy build artifacts if requested
    if save_artifacts and "generated_dir" in dsp_results:
        _copy_artifacts(dsp_results["generated_dir"], save_artifacts)

    # Compare results — QDQ rounding amplifies float32 differences
    comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=1e-1)

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


def _copy_artifacts(generated_dir: Path, dest_dir: str) -> None:
    """Copy build artifacts to specified directory."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for artifact in ["lib0.c", "weights.bin", "devc.c"]:
        src = generated_dir / artifact
        if src.exists():
            shutil.copy2(src, dest / artifact)
            print(f"  Copied {artifact} to {dest}")
    print(f"  Build artifacts saved to: {dest}")


@pytest.mark.quick
@pytest.mark.core
def test_quantized_conv2d_stack_dsp(dsp_mode, dsp_timeout, use_cpp_api, profile_layers):
    """Test quantized Conv2D stack model on DSP comparing against PyTorch reference.

    Uses PT2E static quantization (INT8 QDQ graph).
    """
    results = _run_quantized_conv2d_stack_dsp_test(
        dsp_mode, dsp_timeout, use_cpp_api=use_cpp_api, profile_layers=profile_layers
    )

    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    parser = argparse.ArgumentParser(description="Quantized Conv2D Stack DSP Test")
    parser.add_argument(
        "--dsp-mode",
        default=None,
        choices=["c66x_host", "c7x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument(
        "--save-artifacts",
        type=str,
        metavar="DIR",
        help="Copy build artifacts (lib0.c, weights.bin, devc.c) to DIR",
    )
    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help="Enable per-layer cycle profiling",
    )
    parser.add_argument(
        "--use-cpp-api",
        action="store_true",
        help="Enable direct VM builtin calls (bypass FFI dispatch)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    add_board_arg(parser)
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    # Determine execution mode
    dsp_mode = args.dsp_mode
    if dsp_mode is None:
        parser.print_help()
        return 1

    print("=" * 70)
    print(f"Quantized Conv2D Stack DSP Test (mode: {dsp_mode})")
    print("=" * 70)

    if args.profile_layers:
        print("Layer profiling enabled")

    # Create model
    print("\n[1/3] Creating quantized Conv2D stack model (PT2E INT8)...")
    tvm_mod, quantized_gm, input_data = create_quantized_conv2d_stack_model()

    print(f"  Input shape: {input_data.shape}")

    # Run PyTorch reference
    print("\n[2/3] Running PyTorch quantized reference inference...")
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = quantized_gm(torch_input).numpy()

    print(f"  Output shape: {torch_result.shape}")
    print(f"  Output range: [{torch_result.min():.6f}, {torch_result.max():.6f}]")

    # Build target string
    target_string = get_target_string(
        dsp_mode, profile_layers=args.profile_layers, use_cpp_api=args.use_cpp_api
    )
    if args.use_cpp_api:
        print("Direct VM calls enabled")

    # Run on DSP
    print(f"\n[3/3] DSP Compilation and Execution (mode: {dsp_mode})...")
    print(f"  Target: {target_string}")

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
    )

    # Copy build artifacts if requested
    if args.save_artifacts and "generated_dir" in dsp_results:
        _copy_artifacts(dsp_results["generated_dir"], args.save_artifacts)

    # Print host results
    if "c66x_host_result" in dsp_results:
        c66x_host_result = dsp_results["c66x_host_result"]
        print(f"\n[C66x Host] Output shape: {c66x_host_result.shape}")
        print(f"[C66x Host] Output range: [{c66x_host_result.min():.6f}, {c66x_host_result.max():.6f}]")

    # Print DLOAD results
    if "c7x_dload_result" in dsp_results:
        c7x_dload_result = dsp_results["c7x_dload_result"]
        print(f"\n[C7x DLOAD] Output shape: {c7x_dload_result.shape}")
        print(f"[C7x DLOAD] Output range: [{c7x_dload_result.min():.6f}, {c7x_dload_result.max():.6f}]")

        if "c7x_dload_stdout" in dsp_results:
            stdout = dsp_results["c7x_dload_stdout"]
            cycles_match = re.search(r"Inference complete:\s*(\d+)\s*cycles", stdout)
            if cycles_match:
                cycles = int(cycles_match.group(1))
                time_ms = cycles / 1_000_000
                print(f"[C7x DLOAD] Inference cycles: {cycles:,} ({time_ms:.3f} ms at 1 GHz)")

            if args.profile_layers and "TVM Layer Profile" in stdout:
                profile_start = stdout.find("===== TVM Layer Profile =====")
                profile_end = stdout.find("=============================", profile_start + 1)
                if profile_start != -1 and profile_end != -1:
                    profile_section = stdout[profile_start : profile_end + 29]
                    print(f"\n{profile_section}")

    elif "c7x_dload_error" in dsp_results:
        print(f"\n[C7x DLOAD] SKIPPED: {dsp_results['c7x_dload_error']}")

    # Compare results
    print("\n[Comparison] vs PyTorch quantized:")
    compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=1e-1)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = True

    if "c66x_host_result" in dsp_results:
        c66x_host_diff = np.max(np.abs(dsp_results["c66x_host_result"] - torch_result))
        c66x_host_passed = np.allclose(dsp_results["c66x_host_result"], torch_result, rtol=1e-1, atol=1e-1)
        status = "PASS" if c66x_host_passed else "FAIL"
        print(f"  C66x Host vs PyTorch:    {c66x_host_diff:.2e} [{status}]")
        passed = passed and c66x_host_passed

    if "c7x_dload_result" in dsp_results:
        c7x_dload_diff = np.max(np.abs(dsp_results["c7x_dload_result"] - torch_result))
        c7x_dload_passed = np.allclose(
            dsp_results["c7x_dload_result"], torch_result, rtol=1e-1, atol=1e-1
        )
        status = "PASS" if c7x_dload_passed else "FAIL"
        print(f"  C7x DLOAD vs PyTorch:   {c7x_dload_diff:.2e} [{status}]")
        passed = passed and c7x_dload_passed

    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
