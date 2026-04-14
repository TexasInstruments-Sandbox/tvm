#!/usr/bin/env python
"""
LeNet-5 DSP test.

Tests LeNet-5 CNN on DSP (host emulation, C66x, and/or DLOAD) comparing
against PyTorch reference.

LeNet-5 is a classic convolutional neural network for MNIST digit classification.
It's small enough (~44K params, ~176KB) to fit in DSP memory.

Usage:
    # Run with C66x host emulation
    pytest test_lenet_dsp.py -v

    # Run on C66x hardware
    pytest test_lenet_dsp.py -v --dsp-mode=c66x

    # Run via DLOAD pipeline (c7x_compute)
    pytest test_lenet_dsp.py -v --dsp-mode=c7x_dload

    # Run as standalone script
    python test_lenet_dsp.py --dsp-mode c66x_host
    python test_lenet_dsp.py --dsp-mode c66x
    python test_lenet_dsp.py --dsp-mode c7x_dload
"""

import argparse
import logging
import sys

import pytest
from pathlib import Path

import torch

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, compare_results, get_target_string, assert_dsp_comparison  # noqa: E402  # pyright: ignore[reportMissingImports]
from model_utils import create_lenet_model  # noqa: E402  # pyright: ignore[reportMissingImports]

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_lenet_dsp_test(
    dsp_mode: str,
    timeout_ms: int = 60000,
    profile_layers: bool = False,
    use_cpp_api: bool = False,
) -> dict:
    """
    Run LeNet-5 model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c66x_host", "c66x", "c7x_host", or "c7x_dload")
        timeout_ms: C66x execution timeout
        profile_layers: Enable per-layer cycle profiling (C66x only)
        use_cpp_api: Enable direct VM builtin calls (bypass FFI)

    Returns:
        Dictionary with test results and comparison metrics
    """
    # Create model
    tvm_mod, torch_model, input_data = create_lenet_model()

    # Run PyTorch reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = torch_model(torch_input).numpy()

    # Build target string with optional flags
    target_string = get_target_string(dsp_mode, profile_layers=profile_layers, use_cpp_api=use_cpp_api)

    # Run on DSP
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
    )

    # Compare results
    comparison = compare_results(dsp_results, torch_result, "PyTorch")

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


@pytest.mark.core
def test_lenet_dsp(dsp_mode, dsp_timeout, use_cpp_api):
    """Test LeNet-5 model on DSP comparing against PyTorch reference."""
    results = _run_lenet_dsp_test(dsp_mode, dsp_timeout, use_cpp_api=use_cpp_api)
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run LeNet-5 DSP test as standalone script."""
    parser = argparse.ArgumentParser(description="LeNet-5 DSP Test")
    parser.add_argument(
        "--dsp-mode",
        default=None,
        choices=["c66x_host", "c66x", "c7x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60000,
        help="C66x execution timeout in ms (default: 60000)",
    )
    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help="Enable per-layer cycle profiling (C66x only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Determine execution mode
    mode = args.dsp_mode
    if mode is None:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("LeNet-5 DSP Test")
    print(f"  Mode: {mode}")
    print("  Input: MNIST 28x28 grayscale")
    print("  Output: 10 class logits")
    print("  Estimated params: ~44K (~176KB)")
    print(f"  Timeout: {args.timeout}ms")
    print(f"{'=' * 60}\n")

    try:
        results = _run_lenet_dsp_test(
            dsp_mode=mode,
            timeout_ms=args.timeout,
            profile_layers=args.profile_layers,
        )

        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        comparison = results["comparison"]
        all_passed = True
        if "c66x_host_vs_ref_max_diff" in comparison:
            passed = comparison["c66x_host_vs_ref_passed"]
            all_passed = all_passed and passed
            print("\nC66x Host vs PyTorch:")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")
            print(f"  Max diff: {comparison['c66x_host_vs_ref_max_diff']:.6e}")
        if "c66x_vs_ref_max_diff" in comparison:
            passed = comparison["c66x_vs_ref_passed"]
            all_passed = all_passed and passed
            print("\nC66x vs PyTorch:")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")
            print(f"  Max diff: {comparison['c66x_vs_ref_max_diff']:.6e}")
        if "c7x_vs_ref_max_diff" in comparison:
            passed = comparison["c7x_vs_ref_passed"]
            all_passed = all_passed and passed
            print("\nC7x vs PyTorch:")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")
            print(f"  Max diff: {comparison['c7x_vs_ref_max_diff']:.6e}")
        if "c7x_dload_vs_ref_max_diff" in comparison:
            passed = comparison["c7x_dload_vs_ref_passed"]
            all_passed = all_passed and passed
            print("\nC7x DLOAD vs PyTorch:")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")
            print(f"  Max diff: {comparison['c7x_dload_vs_ref_max_diff']:.6e}")

        # Show sample output
        print("\nSample output (class logits):")
        print(f"  {results['torch_result'][0][:5]}...")

        print("\n" + "=" * 60)
        if all_passed:
            print("TEST PASSED")
        else:
            print("TEST FAILED")
        print("=" * 60)

        if not all_passed:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
