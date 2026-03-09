#!/usr/bin/env python
"""
ResNet-18 DSP test.

Tests ResNet-18 model on DSP comparing against PyTorch reference.
Supports host emulation and DLOAD (dynamic loading on C7x hardware).

Usage:
    # Run with C66x host emulation
    pytest test_resnet_dsp.py -v

    # Run with DLOAD on C7x hardware
    pytest test_resnet_dsp.py -v --dsp-mode=c7x_dload --use-cpp-api

    # Run as standalone script
    python test_resnet_dsp.py --dsp-mode c66x_host
    python test_resnet_dsp.py --dsp-mode c7x_dload -v

Note: DLOAD mode requires:
- C++ API codegen (-use-cpp-api=1) for multi-element make_tuple (skip connections)
- 504 MB input buffer (lib0.out with embedded ~46 MB weights)
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, compare_results, get_target_string, assert_dsp_comparison  # noqa: E402

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def create_resnet_model() -> tuple:
    """
    Create ResNet-18 model for DSP testing.

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, 3, 224, 224]
    """
    # Initialize torch model with pre-trained weights
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

    # Create example input for torch.export
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

    # Export to TVM Relax
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(exported_program, keep_params_as_input=True)

    # Detach and bind parameters
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    # Create deterministic test input
    np.random.seed(42)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    return mod, torch_model, input_data


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_resnet_dsp_test(
    dsp_mode: str = "c66x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
    profile_layers: bool = False,
) -> dict:
    """
    Run ResNet-18 model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c66x_host" or "c7x_dload")
        timeout_ms: Execution timeout in milliseconds
        use_cpp_api: Enable C++ API codegen (required for DLOAD)
        profile_layers: Enable per-layer cycle profiling

    Returns:
        Dictionary with test results and comparison metrics
    """
    # Create model
    tvm_mod, torch_model, input_data = create_resnet_model()

    # Run PyTorch reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = torch_model(torch_input).numpy()

    # Build target string
    target_string = get_target_string(dsp_mode, profile_layers=profile_layers,
                                      use_cpp_api=use_cpp_api)

    # Run on DSP
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
        profile_layers=profile_layers,
    )

    # Compare results — wider tolerance for 18 layers of float32 accumulation
    # with C7x math intrinsics (stable max error ~4.7e-02)
    comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=5e-2, atol=5e-2)

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


def test_resnet_dsp(dsp_mode, dsp_timeout, use_cpp_api, profile_layers):
    """Test ResNet-18 model on DSP comparing against PyTorch reference.

    Supports host emulation and DLOAD (dynamic loading on C7x hardware).
    DLOAD requires --use-cpp-api for multi-element make_tuple support.
    """
    results = _run_resnet_dsp_test(
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    parser = argparse.ArgumentParser(description="ResNet-18 DSP Test")
    parser.add_argument(
        "--dsp-mode",
        default=None,
        choices=["c66x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument(
        "--profile-layers",
        action="store_true",
        help="Enable per-layer cycle profiling",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
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
    print(f"ResNet-18 DSP Test (mode: {dsp_mode})")
    print("=" * 70)

    if args.profile_layers:
        print("Layer profiling enabled")

    # Create model
    print("\n[1/3] Creating ResNet-18 model...")
    tvm_mod, torch_model, input_data = create_resnet_model()

    total_params = sum(p.numel() for p in torch_model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Input shape: {input_data.shape}")

    # Run PyTorch reference
    print("\n[2/3] Running PyTorch reference inference...")
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = torch_model(torch_input).numpy()

    print(f"  Output shape: {torch_result.shape}")
    print(f"  Output range: [{torch_result.min():.6f}, {torch_result.max():.6f}]")

    # Build target string
    target_string = get_target_string(
        dsp_mode,
        profile_layers=args.profile_layers,
        use_cpp_api=(dsp_mode == "c7x_dload"),
    )

    # Run on DSP
    print(f"\n[3/3] DSP Compilation and Execution (mode: {dsp_mode})...")
    print(f"  Target: {target_string}")

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        profile_layers=args.profile_layers,
    )

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

        # Extract inference cycles from stdout
        if "c7x_dload_stdout" in dsp_results:
            stdout = dsp_results["c7x_dload_stdout"]
            cycles_match = re.search(r"Inference complete:\s*(\d+)\s*cycles", stdout)
            if cycles_match:
                cycles = int(cycles_match.group(1))
                time_ms = cycles / 1_000_000  # 1 GHz for C7x
                print(f"[C7x DLOAD] Inference cycles: {cycles:,} ({time_ms:.3f} ms at 1 GHz)")

            # Print layer profile if available and profiling was requested
            if args.profile_layers and "TVM Layer Profile" in stdout:
                profile_start = stdout.find("===== TVM Layer Profile =====")
                profile_end = stdout.find("=============================", profile_start + 1)
                if profile_start != -1 and profile_end != -1:
                    profile_section = stdout[profile_start : profile_end + 29]
                    print(f"\n{profile_section}")

    elif "c7x_dload_error" in dsp_results:
        print(f"\n[C7x DLOAD] SKIPPED: {dsp_results['c7x_dload_error']}")

    # Compare results
    print("\n[Comparison] vs PyTorch:")
    compare_results(dsp_results, torch_result, "PyTorch", rtol=5e-2, atol=5e-2)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Model parameters: {total_params:,}")

    passed = True

    if "c66x_host_result" in dsp_results:
        c66x_host_diff = np.max(np.abs(dsp_results["c66x_host_result"] - torch_result))
        c66x_host_passed = np.allclose(dsp_results["c66x_host_result"], torch_result, rtol=5e-2, atol=5e-2)
        status = "PASS" if c66x_host_passed else "FAIL"
        print(f"  C66x Host vs PyTorch:    {c66x_host_diff:.2e} [{status}]")
        passed = passed and c66x_host_passed

    if "c7x_dload_result" in dsp_results:
        c7x_dload_diff = np.max(np.abs(dsp_results["c7x_dload_result"] - torch_result))
        c7x_dload_passed = np.allclose(dsp_results["c7x_dload_result"], torch_result, rtol=5e-2, atol=5e-2)
        status = "PASS" if c7x_dload_passed else "FAIL"
        print(f"  C7x DLOAD vs PyTorch:   {c7x_dload_diff:.2e} [{status}]")
        passed = passed and c7x_dload_passed

    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
