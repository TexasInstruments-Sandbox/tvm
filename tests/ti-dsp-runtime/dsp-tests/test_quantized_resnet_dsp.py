#!/usr/bin/env python
"""
Quantized ResNet-18 DSP test.

Tests INT8 quantized ResNet-18 model on DSP comparing against PyTorch
quantized reference. Uses PT2E static quantization with XNNPACKQuantizer
to produce a QDQ (quantize-dequantize) graph.

Usage:
    # Run with C7x host emulation
    pytest test_quantized_resnet_dsp.py -v --dsp-mode=c7x_host

    # Run with DLOAD on C7x hardware
    pytest test_quantized_resnet_dsp.py -v --dsp-mode=c7x_dload --use-cpp-api

    # Run as standalone script
    python test_quantized_resnet_dsp.py --dsp-mode c7x_host
    python test_quantized_resnet_dsp.py --dsp-mode c7x_dload -v
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


def create_quantized_resnet_model() -> tuple:
    """
    Create INT8 quantized ResNet-18 model for DSP testing.

    Uses PT2E static quantization with XNNPACKQuantizer to produce
    a QDQ graph with per-tensor quantization.

    Returns:
        Tuple of (tvm_mod, quantized_gm, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - quantized_gm: PyTorch quantized GraphModule for reference
        - input_data: numpy array for test input [1, 3, 224, 224]
    """
    import warnings

    # torch.ao.quantization is deprecated in torch 2.10 in favor of
    # torchao, but torchao 0.16 doesn't ship XNNPACKQuantizer yet.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from torch.ao.quantization.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

    # Initialize torch model with pre-trained weights
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

    # Create example input for torch.export
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

    # Step 1: Capture the float model (prepare_pt2e needs a GraphModule)
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
    model_gm = exported_program.module()

    # Step 2: PT2E quantization
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(model_gm, quantizer)

    # Calibrate with random inputs
    with torch.no_grad():
        for _ in range(10):
            prepared(torch.randn(1, 3, 224, 224, dtype=torch.float32))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="erase_node")
        quantized_gm = convert_pt2e(prepared)

    # Step 3: Re-export the quantized model and import to TVM
    with torch.no_grad():
        exported_program_q = export(quantized_gm, example_args)
        mod = from_exported_program(exported_program_q, keep_params_as_input=True)

    # Detach and bind parameters
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    # Create deterministic test input
    np.random.seed(42)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    return mod, quantized_gm, input_data


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_quantized_resnet_dsp_test(
    dsp_mode: str = "c7x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
    profile_layers: bool = False,
    profile: bool = False,
) -> dict:
    """
    Run quantized ResNet-18 model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c7x_host" or "c7x_dload")
        timeout_ms: Execution timeout in milliseconds
        use_cpp_api: Enable C++ API codegen (required for DLOAD)
        profile_layers: Enable per-layer cycle profiling

    Returns:
        Dictionary with test results and comparison metrics
    """
    # Create model
    tvm_mod, quantized_gm, input_data = create_quantized_resnet_model()

    # Run PyTorch quantized reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = quantized_gm(torch_input).numpy()

    # Build target string
    target_string = get_target_string(
        dsp_mode,
        profile_layers=profile_layers,
        use_cpp_api=use_cpp_api,
    )

    # Run on DSP
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
        profile_layers=profile_layers,
        profile=profile,
    )

    # Compare results — after EliminateQDQRoundTrip, TVM keeps full float32
    # precision at residual-add boundaries while PyTorch's quantized model
    # still rounds to int8.  This creates up to ~0.3 max diff on deep
    # networks like ResNet-18 (20 layers with residual connections).
    comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=3e-1)

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


def test_quantized_resnet_dsp(dsp_mode, dsp_timeout, use_cpp_api, profile_layers, profile):
    """Test quantized ResNet-18 model on DSP comparing against PyTorch reference.

    Uses PT2E static quantization (INT8 QDQ graph).
    Supports host emulation and DLOAD (dynamic loading on C7x hardware).
    DLOAD requires --use-cpp-api for multi-element make_tuple support.
    """
    results = _run_quantized_resnet_dsp_test(
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
        profile=profile,
    )

    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    parser = argparse.ArgumentParser(description="Quantized ResNet-18 DSP Test")
    parser.add_argument(
        "--dsp-mode",
        default=None,
        choices=["c7x_host", "c7x_dload"],
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
    print(f"Quantized ResNet-18 DSP Test (mode: {dsp_mode})")
    print("=" * 70)

    if args.profile_layers:
        print("Layer profiling enabled")

    # Create model
    print("\n[1/3] Creating quantized ResNet-18 model (PT2E INT8)...")
    tvm_mod, quantized_gm, input_data = create_quantized_resnet_model()

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
        dsp_mode,
        profile_layers=args.profile_layers,
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
    if "c7x_host_result" in dsp_results:
        c7x_host_result = dsp_results["c7x_host_result"]
        print(f"\n[C7x Host] Output shape: {c7x_host_result.shape}")
        print(
            f"[C7x Host] Output range: [{c7x_host_result.min():.6f}, {c7x_host_result.max():.6f}]"
        )

    # Print DLOAD results
    if "c7x_dload_result" in dsp_results:
        c7x_dload_result = dsp_results["c7x_dload_result"]
        print(f"\n[C7x DLOAD] Output shape: {c7x_dload_result.shape}")
        print(
            f"[C7x DLOAD] Output range: [{c7x_dload_result.min():.6f}, {c7x_dload_result.max():.6f}]"
        )

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
    print("\n[Comparison] vs PyTorch quantized:")
    compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=3e-1)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = True

    if "c7x_host_result" in dsp_results:
        c7x_host_diff = np.max(np.abs(dsp_results["c7x_host_result"] - torch_result))
        c7x_host_passed = np.allclose(
            dsp_results["c7x_host_result"], torch_result, rtol=1e-1, atol=3e-1
        )
        status = "PASS" if c7x_host_passed else "FAIL"
        print(f"  C7x Host vs PyTorch:     {c7x_host_diff:.2e} [{status}]")
        passed = passed and c7x_host_passed

    if "c7x_dload_result" in dsp_results:
        c7x_dload_diff = np.max(np.abs(dsp_results["c7x_dload_result"] - torch_result))
        c7x_dload_passed = np.allclose(
            dsp_results["c7x_dload_result"], torch_result, rtol=1e-1, atol=3e-1
        )
        status = "PASS" if c7x_dload_passed else "FAIL"
        print(f"  C7x DLOAD vs PyTorch:   {c7x_dload_diff:.2e} [{status}]")
        passed = passed and c7x_dload_passed

    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
