#!/usr/bin/env python
"""
Multi-output model DSP test.

Tests multi-output tuple handling on DSP (host emulation and DLOAD).
This validates the DSP runtime's make_tuple implementation and
Model::InferMulti() API for models that return multiple tensors.

The c_static backend supports multi-element make_tuple in C++ API mode,
enabling multi-output models on C7x hardware.

Usage:
    # Run with C66x host emulation
    pytest test_rtmdet_dsp.py -v

    # Run via DLOAD pipeline (c7x_compute)
    pytest test_rtmdet_dsp.py -v --dsp-mode=c7x_dload

    # Run as standalone script
    python test_rtmdet_dsp.py --dsp-mode c66x_host
    python test_rtmdet_dsp.py --dsp-mode c7x_dload
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.export import export

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402

pytestmark = [pytest.mark.c7x_only]

# Configure logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Multi-Output Model
# -----------------------------------------------------------------------------


class TwoOutputModel(nn.Module):
    """Model that returns two outputs (tuple).

    Tests multi-output support in DSP runtime.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning two outputs.

        For 32x32 input:
        - output1: [1, 16, 16, 16] (after conv1 + relu + pool)
        - output2: [1, 8, 16, 16] (after conv2)
        """
        x1 = self.pool(self.relu(self.conv1(x)))  # [1, 16, 16, 16]
        x2 = self.conv2(x1)  # [1, 8, 16, 16]
        return x1, x2


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def create_multi_output_model() -> tuple:
    """
    Create multi-output model for DSP testing.

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, 3, 32, 32]
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    torch_model = TwoOutputModel()
    torch_model.eval()

    # Create example input for torch.export
    example_args = (torch.randn(1, 3, 32, 32, dtype=torch.float32),)

    # Export to TVM Relax
    with torch.no_grad():
        exported_program = export(torch_model, example_args, strict=False)
        mod = from_exported_program(exported_program, keep_params_as_input=True)

    # Detach and bind parameters
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    # Create deterministic test input
    input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)

    return mod, torch_model, input_data


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_multi_output_test(
    dsp_mode: str = "c66x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
) -> dict:
    """
    Run multi-output model on DSP and compare with PyTorch reference.

    Args:
        dsp_mode: Execution mode ("c66x_host", "c66x", "c7x_host", or "c7x_dload")
        timeout_ms: Execution timeout in milliseconds
        use_cpp_api: Enable direct VM builtin calls (bypass FFI)

    Returns:
        Dictionary with test results
    """
    # Create model
    tvm_mod, torch_model, input_data = create_multi_output_model()

    # Run PyTorch reference
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_outputs = torch_model(torch_input)
        torch_outputs = [out.numpy() for out in torch_outputs]

    # Build target string based on mode
    target_string = get_target_string(dsp_mode, use_cpp_api=use_cpp_api)

    # Run on DSP
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
    )

    return {
        "torch_outputs": torch_outputs,
        "dsp_results": dsp_results,
    }


def test_multi_output_count(dsp_mode, dsp_timeout, use_cpp_api):
    """Test that multi-output model returns correct number of outputs."""
    results = _run_multi_output_test(dsp_mode, dsp_timeout, use_cpp_api)
    dsp_results = results["dsp_results"]

    # Check for execution errors
    if "c66x_host_error" in dsp_results:
        pytest.fail(f"Host execution error: {dsp_results['c66x_host_error']}")
    if "c66x_error" in dsp_results:
        pytest.fail(f"C66x execution error: {dsp_results['c66x_error']}")
    if "c7x_dload_error" in dsp_results:
        pytest.fail(f"DLOAD execution error: {dsp_results['c7x_dload_error']}")

    # Check results based on execution mode
    # Determine which result key to check based on execution mode
    _mode_to_key = {"c7x_dload": "c7x_dload_result", "c7x_host": "c7x_host_result"}
    result_key = _mode_to_key.get(dsp_mode, "c66x_host_result")
    if result_key in dsp_results:
        dsp_result = dsp_results[result_key]
        # Multi-output should return a list
        if isinstance(dsp_result, list):
            assert len(dsp_result) == 2, f"Expected 2 outputs, got {len(dsp_result)}"
        else:
            pytest.fail(f"Expected list of outputs, got {type(dsp_result)}")
    else:
        pytest.fail(f"No DSP {result_key} results available")


def test_multi_output_shapes(dsp_mode, dsp_timeout, use_cpp_api):
    """Test that multi-output model produces correct output shapes."""
    results = _run_multi_output_test(dsp_mode, dsp_timeout, use_cpp_api)
    dsp_results = results["dsp_results"]

    # Check for execution errors
    if "c66x_host_error" in dsp_results:
        pytest.fail(f"Host execution error: {dsp_results['c66x_host_error']}")
    if "c66x_error" in dsp_results:
        pytest.fail(f"C66x execution error: {dsp_results['c66x_error']}")
    if "c7x_dload_error" in dsp_results:
        pytest.fail(f"DLOAD execution error: {dsp_results['c7x_dload_error']}")

    expected_shapes = [(1, 16, 16, 16), (1, 8, 16, 16)]

    # Determine which result key to check based on execution mode
    _mode_to_key = {"c7x_dload": "c7x_dload_result", "c7x_host": "c7x_host_result"}
    result_key = _mode_to_key.get(dsp_mode, "c66x_host_result")
    if result_key in dsp_results:
        dsp_result = dsp_results[result_key]
        if isinstance(dsp_result, list):
            assert len(dsp_result) == len(expected_shapes)
            for i, (result, expected_shape) in enumerate(zip(dsp_result, expected_shapes)):
                actual_shape = tuple(result.shape)
                assert actual_shape == expected_shape, (
                    f"Output {i}: expected shape {expected_shape}, got {actual_shape}"
                )
        else:
            pytest.fail(f"Expected list of outputs, got {type(dsp_result)}")
    else:
        pytest.fail(f"No DSP {result_key} results available")


def test_multi_output_correctness(dsp_mode, dsp_timeout, use_cpp_api):
    """Test multi-output model correctness against PyTorch reference."""
    results = _run_multi_output_test(dsp_mode, dsp_timeout, use_cpp_api)
    torch_outputs = results["torch_outputs"]
    dsp_results = results["dsp_results"]

    # Check for execution errors
    if "c66x_host_error" in dsp_results:
        pytest.fail(f"Host execution error: {dsp_results['c66x_host_error']}")
    if "c66x_error" in dsp_results:
        pytest.fail(f"C66x execution error: {dsp_results['c66x_error']}")
    if "c7x_dload_error" in dsp_results:
        pytest.fail(f"DLOAD execution error: {dsp_results['c7x_dload_error']}")

    # Determine which result key to check based on execution mode
    _mode_to_key = {"c7x_dload": "c7x_dload_result", "c7x_host": "c7x_host_result"}
    result_key = _mode_to_key.get(dsp_mode, "c66x_host_result")
    if result_key in dsp_results:
        dsp_result = dsp_results[result_key]
        if isinstance(dsp_result, list):
            for i, (dsp_out, torch_out) in enumerate(zip(dsp_result, torch_outputs)):
                max_diff = np.max(np.abs(dsp_out - torch_out))
                assert np.allclose(dsp_out, torch_out, rtol=1e-4, atol=1e-5), (
                    f"Output {i}: max diff = {max_diff:.2e}"
                )
        else:
            pytest.fail(f"Expected list of outputs, got {type(dsp_result)}")
    else:
        pytest.fail(f"No DSP {result_key} results available")


def test_multi_output_dtype(dsp_mode, dsp_timeout, use_cpp_api):
    """Test that all outputs have float32 dtype."""
    results = _run_multi_output_test(dsp_mode, dsp_timeout, use_cpp_api)
    dsp_results = results["dsp_results"]

    # Check for execution errors
    if "c66x_host_error" in dsp_results:
        pytest.fail(f"Host execution error: {dsp_results['c66x_host_error']}")
    if "c66x_error" in dsp_results:
        pytest.fail(f"C66x execution error: {dsp_results['c66x_error']}")
    if "c7x_dload_error" in dsp_results:
        pytest.fail(f"DLOAD execution error: {dsp_results['c7x_dload_error']}")

    # Determine which result key to check based on execution mode
    _mode_to_key = {"c7x_dload": "c7x_dload_result", "c7x_host": "c7x_host_result"}
    result_key = _mode_to_key.get(dsp_mode, "c66x_host_result")
    if result_key in dsp_results:
        dsp_result = dsp_results[result_key]
        if isinstance(dsp_result, list):
            for i, result in enumerate(dsp_result):
                assert result.dtype == np.float32, (
                    f"Output {i}: expected float32 dtype, got {result.dtype}"
                )
        else:
            pytest.fail(f"Expected list of outputs, got {type(dsp_result)}")
    else:
        pytest.fail(f"No DSP {result_key} results available")


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    parser = argparse.ArgumentParser(description="Multi-Output DSP Test")
    parser.add_argument(
        "--dsp-mode",
        default=None,
        choices=["c66x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    # Determine execution mode
    execution_mode = args.dsp_mode
    if execution_mode is None:
        parser.print_help()
        return 1

    target_string = get_target_string(
        execution_mode, use_cpp_api=(execution_mode == "c7x_dload")
    )
    mode_name = "C7x DLOAD" if execution_mode == "c7x_dload" else "C66x Host Emulation"

    print("=" * 70)
    print(f"Multi-Output DSP Test ({mode_name})")
    print("=" * 70)

    # Create model
    print("\n[1/3] Creating multi-output model...")
    tvm_mod, torch_model, input_data = create_multi_output_model()

    total_params = sum(p.numel() for p in torch_model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Input shape: {input_data.shape}")

    # Run PyTorch reference
    print("\n[2/3] Running PyTorch reference inference...")
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_outputs = torch_model(torch_input)
        torch_outputs = [out.numpy() for out in torch_outputs]

    print(f"  Output 0 shape: {torch_outputs[0].shape}")
    print(f"  Output 1 shape: {torch_outputs[1].shape}")

    # Run on DSP
    print(f"\n[3/3] DSP Compilation and Execution ({mode_name})...")
    print(f"  Target: {target_string}")

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=execution_mode,
    )

    # Print and compare results
    passed = True
    _mode_to_key = {"c7x_dload": "c7x_dload_result"}
    result_key = _mode_to_key.get(execution_mode, "c66x_host_result")
    label = "C7x DLOAD" if execution_mode == "c7x_dload" else "C66x Host"

    if result_key in dsp_results:
        dsp_result = dsp_results[result_key]
        if isinstance(dsp_result, list):
            print(f"\n[DSP {label}] Number of outputs: {len(dsp_result)}")
            for i, (dsp_out, torch_out) in enumerate(zip(dsp_result, torch_outputs)):
                print(f"[DSP {label}] Output {i} shape: {dsp_out.shape}")
                max_diff = np.max(np.abs(dsp_out - torch_out))
                matches = np.allclose(dsp_out, torch_out, rtol=1e-4, atol=1e-5)
                status = "PASS" if matches else "FAIL"
                print(f"[DSP] Output {i} vs PyTorch: max diff = {max_diff:.2e} [{status}]")
                passed = passed and matches
        else:
            print(f"\n[ERROR] Expected list of outputs, got single output: {type(dsp_result)}")
            passed = False
    else:
        print(f"\n[ERROR] No {label} results available")
        passed = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Model parameters: {total_params:,}")
    print("  Number of outputs: 2")
    print(f"  Output 0 shape: {torch_outputs[0].shape}")
    print(f"  Output 1 shape: {torch_outputs[1].shape}")
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
