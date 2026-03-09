#!/usr/bin/env python
"""
MLP DSP test.

Tests a simple MLP (Multi-Layer Perceptron) on DSP (host emulation, C66x,
and/or DLOAD) comparing against PyTorch reference.

Memory considerations:
- C66x L2 pool: 64 KB, L3 pool: 1024 KB
- C7x L2 pool: ~1.59 MB, DDR pool: ~55 MB
- Default MLP uses input_size=784, hidden_size=128, output_size=10
- This results in ~100KB weights (784*128 + 128*10 + biases)

Usage:
    # Run with C66x host emulation
    pytest test_mlp_dsp.py -v

    # Run on C66x hardware
    pytest test_mlp_dsp.py -v --dsp-mode=c66x

    # Run via DLOAD pipeline (c7x_compute)
    pytest test_mlp_dsp.py -v --dsp-mode=c7x_dload

    # Run as standalone script
    python test_mlp_dsp.py --dsp-mode c66x_host
    python test_mlp_dsp.py --dsp-mode c66x
    python test_mlp_dsp.py --dsp-mode c7x_dload
"""

import argparse
import logging
import sys
from pathlib import Path

import pytest
import torch

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, compare_results, get_target_string, assert_dsp_comparison  # noqa: E402
from model_utils import create_mlp_model  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_mlp_dsp_test(
    dsp_mode: str,
    timeout_ms: int = 60000,
    profile_layers: bool = False,
    use_cpp_api: bool = False,
    input_size: int = 784,
    hidden_size: int = 128,
    output_size: int = 10,
) -> dict:
    """
    Run MLP model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c66x_host", "c66x", "c7x_host", or "c7x_dload")
        timeout_ms: C66x execution timeout
        profile_layers: Enable per-layer cycle profiling (C66x only)
        use_cpp_api: Enable direct VM builtin calls (bypass FFI)
        input_size: Input feature dimension
        hidden_size: Hidden layer size
        output_size: Output dimension

    Returns:
        Dictionary with test results and comparison metrics
    """
    # Create model
    tvm_mod, torch_model, input_data = create_mlp_model(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    # Run PyTorch reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = torch_model(torch_input).numpy()

    # Build target string
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

    # Compare results
    comparison = compare_results(dsp_results, torch_result, "PyTorch")

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


@pytest.mark.quick
def test_mlp_dsp(dsp_mode, dsp_timeout, use_cpp_api):
    """Test MLP model on DSP comparing against PyTorch reference.

    Uses smaller dimensions to fit in C66x L2 memory (~320KB).
    Architecture: 128 -> 32 -> 10 (~5KB weights)
    """
    results = _run_mlp_dsp_test(
        dsp_mode,
        dsp_timeout,
        use_cpp_api=use_cpp_api,
        input_size=128,
        hidden_size=32,
        output_size=10,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run MLP DSP test as standalone script."""
    parser = argparse.ArgumentParser(description="MLP DSP Test")
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
    parser.add_argument("--input-size", type=int, default=784, help="Input feature dimension")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--output-size", type=int, default=10, help="Output dimension")
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

    # Estimate memory usage
    weight_bytes = (args.input_size * args.hidden_size + args.hidden_size * args.output_size) * 4
    print(f"\n{'=' * 60}")
    print("MLP DSP Test")
    print(f"  Mode: {mode}")
    print(f"  Architecture: {args.input_size} -> {args.hidden_size} -> {args.output_size}")
    print(f"  Estimated weights: {weight_bytes / 1024:.1f} KB")
    print(f"  Timeout: {args.timeout}ms")
    print(f"{'=' * 60}\n")

    try:
        results = _run_mlp_dsp_test(
            dsp_mode=mode,
            timeout_ms=args.timeout,
            profile_layers=args.profile_layers,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
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
