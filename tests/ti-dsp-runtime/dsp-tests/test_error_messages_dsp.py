#!/usr/bin/env python
"""
Error message and compilation validation test for DSP targets.

Tests that the C Static backend can compile models correctly and validates
basic error handling with DSP target settings.

This test runs on host emulation only.

Usage:
    # Run with pytest
    pytest test_error_messages_dsp.py -v

    # Run as standalone script
    python test_error_messages_dsp.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp  # noqa: E402

pytestmark = [pytest.mark.core, pytest.mark.c66x_only]

# -----------------------------------------------------------------------------
# Test Models
# -----------------------------------------------------------------------------


@I.ir_module
class SimpleModel:
    """Simple model for testing - just multiplies input by 2"""

    @T.prim_func(private=True)
    def multiply_by_two(x: T.handle, z: T.handle) -> None:
        X = T.match_buffer(x, (1, 3, 224, 224))
        Z = T.match_buffer(z, (1, 3, 224, 224))
        for i, j, k, m in T.grid(1, 3, 224, 224):
            with T.block("multiply"):
                vi, vj, vk, vm = T.axis.remap("SSSS", [i, j, k, m])
                Z[vi, vj, vk, vm] = X[vi, vj, vk, vm] * T.float32(2)

    @R.function
    def main(
        x: R.Tensor((1, 3, 224, 224), "float32"),
    ) -> R.Tensor((1, 3, 224, 224), "float32"):
        R.func_attr({"num_input": 1})
        gv0 = R.call_tir(
            SimpleModel.multiply_by_two,
            (x,),
            R.Tensor((1, 3, 224, 224), dtype="float32"),
        )
        return gv0


@I.ir_module
class SmallModel:
    """Smaller model for testing"""

    @T.prim_func(private=True)
    def add_one(x: T.handle, z: T.handle) -> None:
        X = T.match_buffer(x, (1, 10))
        Z = T.match_buffer(z, (1, 10))
        for i, j in T.grid(1, 10):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = X[vi, vj] + T.float32(1)

    @R.function
    def main(x: R.Tensor((1, 10), "float32")) -> R.Tensor((1, 10), "float32"):
        R.func_attr({"num_input": 1})
        gv0 = R.call_tir(SmallModel.add_one, (x,), R.Tensor((1, 10), dtype="float32"))
        return gv0


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


@pytest.mark.dsp_host_only
def test_simple_model_compiles_with_dsp_target(dsp_mode):
    """Test that SimpleModel compiles and runs correctly with DSP target."""
    mod = SimpleModel

    # Create correct input
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # Should compile and run without error
    target_string = "c_static -mcpu=c66x -use-cpp-api=0"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode="c66x_host",
    )

    # Verify we got a result
    assert "c66x_host_result" in results, "Should have c66x_host result"
    host_result = results["c66x_host_result"]

    # Verify output shape matches input shape
    assert host_result.shape == input_data.shape, (
        f"Output shape {host_result.shape} should match input shape {input_data.shape}"
    )

    # Verify the operation (multiply by 2)
    expected = input_data * 2.0
    assert np.allclose(host_result, expected, rtol=1e-5, atol=1e-5), (
        f"Output should be input * 2. Max diff: {np.max(np.abs(host_result - expected))}"
    )


@pytest.mark.dsp_host_only
def test_small_model_compiles_with_dsp_target(dsp_mode):
    """Test that SmallModel compiles and runs correctly with DSP target."""
    mod = SmallModel

    # Create correct input
    input_data = np.random.rand(1, 10).astype(np.float32)

    # Should compile and run without error
    target_string = "c_static -mcpu=c66x -use-cpp-api=0"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode="c66x_host",
    )

    # Verify we got a result
    assert "c66x_host_result" in results, "Should have c66x_host result"
    host_result = results["c66x_host_result"]

    # Verify output shape matches input shape
    assert host_result.shape == input_data.shape, (
        f"Output shape {host_result.shape} should match input shape {input_data.shape}"
    )

    # Verify the operation (add 1)
    expected = input_data + 1.0
    assert np.allclose(host_result, expected, rtol=1e-5, atol=1e-5), (
        f"Output should be input + 1. Max diff: {np.max(np.abs(host_result - expected))}"
    )


@pytest.mark.dsp_host_only
def test_dsp_target_with_skip_runtime_checks(dsp_mode):
    """Test compilation with skip-runtime-checks attribute."""
    mod = SmallModel

    input_data = np.random.rand(1, 10).astype(np.float32)

    # Test with skip-runtime-checks=1
    target_string = "c_static -mcpu=c66x -use-cpp-api=0 -skip-runtime-checks=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode="c66x_host",
    )

    assert "c66x_host_result" in results, "Should have c66x_host result"
    expected = input_data + 1.0
    assert np.allclose(results["c66x_host_result"], expected, rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    print("\n" + "=" * 70)
    print("DSP Compilation Tests")
    print("=" * 70)

    tests = [
        ("Simple Model Compilation", test_simple_model_compiles_with_dsp_target),
        ("Small Model Compilation", test_small_model_compiles_with_dsp_target),
        ("Skip Runtime Checks", test_dsp_target_with_skip_runtime_checks),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            # Mock dsp_mode fixture
            test_func("c66x_host")
            print("  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
