#!/usr/bin/env python
"""
Matmul DSP test.

Tests matrix multiplication on DSP (host emulation, C66x, and/or C7x hardware)
comparing against PyTorch reference.

Usage:
    # Run with C66x host emulation
    pytest test_matmul_dsp.py -v

    # Run on C66x hardware
    pytest test_matmul_dsp.py -v --dsp-mode=c66x

    # Run via DLOAD pipeline (c7x_compute)
    pytest test_matmul_dsp.py -v --dsp-mode=c7x_dload

    # Run as standalone script
    python test_matmul_dsp.py --dsp-mode c66x_host
    python test_matmul_dsp.py --dsp-mode c66x
    python test_matmul_dsp.py --dsp-mode c7x_dload
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
from model_utils import create_matmul_model  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_matmul_dsp_test(
    dsp_mode: str,
    timeout_ms: int = 60000,
    profile_layers: bool = False,
    use_cpp_api: bool = False,
    m: int = 64,
    k: int = 64,
    n: int = 64,
) -> dict:
    """
    Run matmul model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c66x_host", "c66x", "c7x_host", or "c7x_dload")
        timeout_ms: C66x execution timeout
        profile_layers: Enable per-layer cycle profiling (C66x only)
        use_cpp_api: Enable direct VM builtin calls (bypass FFI)
        m: First dimension of input matrix
        k: Second dimension of input / first dimension of weight
        n: Second dimension of weight / output

    Returns:
        Dictionary with test results and comparison metrics
    """
    # Create model
    tvm_mod, torch_model, input_data = create_matmul_model(m=m, k=k, n=n)

    # Run PyTorch reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = torch_model(torch_input).numpy()

    # Build target string
    target_string = get_target_string(dsp_mode, profile_layers=profile_layers, use_cpp_api=True)

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
def test_matmul_dsp(dsp_mode, dsp_timeout, use_cpp_api):
    """Test matmul model on DSP comparing against PyTorch reference."""
    results = _run_matmul_dsp_test(dsp_mode, dsp_timeout, use_cpp_api=use_cpp_api)

    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run matmul DSP test as standalone script."""
    parser = argparse.ArgumentParser(description="Matmul DSP Test")
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
    parser.add_argument("-m", type=int, default=64, help="First dimension of input matrix")
    parser.add_argument("-k", type=int, default=64, help="Second dimension of input")
    parser.add_argument("-n", type=int, default=64, help="Second dimension of weight")
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
    print("Matmul DSP Test")
    print(f"  Mode: {mode}")
    print(f"  Matrix size: ({args.m}, {args.k}) x ({args.k}, {args.n})")
    print(f"  Timeout: {args.timeout}ms")
    print(f"{'=' * 60}\n")

    try:
        results = _run_matmul_dsp_test(
            dsp_mode=mode,
            timeout_ms=args.timeout,
            profile_layers=args.profile_layers,
            m=args.m,
            k=args.k,
            n=args.n,
        )

        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        # Check for execution errors (e.g., timeout)
        dsp_results = results["dsp_results"]
        all_passed = True
        if "c66x_host_error" in dsp_results:
            print(f"\nC66x Host: ERROR - {dsp_results['c66x_host_error']}")
            all_passed = False
        if "c66x_error" in dsp_results:
            print(f"\nC66x: ERROR - {dsp_results['c66x_error']}")
            all_passed = False
        if "c7x_error" in dsp_results:
            print(f"\nC7x: ERROR - {dsp_results['c7x_error']}")
            all_passed = False
        if "c7x_dload_error" in dsp_results:
            print(f"\nC7x DLOAD: ERROR - {dsp_results['c7x_dload_error']}")
            all_passed = False

        comparison = results["comparison"]
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
