"""
Relax If expression test on c_static / C7x DSP.

Validates that the Relax IR If expression (runtime conditional branch
selection) compiles and executes correctly through the c_static backend
on C7x host emulation and C7x DLOAD hardware.

Model structure
---------------
Inputs:
  x      : float32[4]  - data tensor
  cond_f : float32[]   - scalar condition (0.0 = false, non-zero = true)

Computation:
  cond_bool = astype(cond_f, bool)
  if cond_bool:
      output = x + x          # add branch  (true)
  else:
      output = x * x          # mul branch  (false)

Expected outputs:
  x = [1, 2, 3, 4]
  cond_f = 1.0  →  output = [2, 4, 6, 8]
  cond_f = 0.0  →  output = [1, 4, 9, 16]

Usage
-----
    # C7x host emulation (fast, no hardware)
    pytest tests/ti-dsp-runtime/dynamic-tests/test_if_dsp.py \\
        -m quick --dsp-mode=c7x_host -v

    # C7x hardware via DLOAD
    pytest tests/ti-dsp-runtime/dynamic-tests/test_if_dsp.py \\
        -m quick --dsp-mode=c7x_dload -v

    # Preserve build directory for inspection
    DSP_KEEP_TEMP=1 pytest ... --dsp-mode=c7x_host -v
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm.script import relax as R

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402
    add_board_arg,
    assert_dsp_comparison,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relax module: If expression (static shapes, float32 condition)
# ---------------------------------------------------------------------------

@tvm.script.ir_module
class IfSelectModule:
    """
    Minimal Relax module exercising the If expression.

    The condition is passed as a float32 scalar (0.0 = false, non-zero = true)
    to avoid bool dtype serialization complexity.  It is cast to bool inside
    the model so the actual Relax If node uses a proper bool condition.
    """

    @R.function
    def main(
        x: R.Tensor((4,), dtype="float32"),
        cond_f: R.Tensor((), dtype="float32"),
    ) -> R.Tensor((4,), dtype="float32"):
        R.func_attr({"num_input": 2})
        # Cast float32 condition to bool (0.0 → False, non-zero → True)
        cond_bool: R.Tensor((), dtype="bool") = R.astype(cond_f, "bool")
        # Relax If expression: both branches must assign to the same variable
        if cond_bool:
            out: R.Tensor((4,), dtype="float32") = R.add(x, x)
        else:
            out: R.Tensor((4,), dtype="float32") = R.multiply(x, x)
        return out


# ---------------------------------------------------------------------------
# Reference computation (numpy)
# ---------------------------------------------------------------------------

def reference_if_select(x: np.ndarray, cond: float) -> np.ndarray:
    """Compute reference output matching the Relax If model."""
    if cond != 0.0:
        return x + x      # add branch
    return x * x          # multiply branch


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_if_test(
    dsp_mode: str,
    cond_value: float,
    timeout_ms: int = 60000,
    use_cpp_api: bool = False,
) -> dict:
    """
    Compile and run the If model for a given condition value.

    Args:
        dsp_mode: "c7x_host" or "c7x_dload"
        cond_value: 0.0 for false branch, 1.0 for true branch
        timeout_ms: DSP execution timeout in milliseconds
        use_cpp_api: Enable direct VM builtin calls (bypass FFI)

    Returns:
        dict with keys: dsp_results, reference, comparison
    """
    # Fixed test input: [1, 2, 3, 4]
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    cond_f = np.array(cond_value, dtype=np.float32)  # 0-d array, shape=()

    # Numpy reference
    reference = reference_if_select(x, cond_value)

    # Build target string
    target_string = get_target_string(dsp_mode, use_cpp_api=use_cpp_api)

    # Compile and run: pass (x, cond_f) as multi-input tuple
    dsp_results = compile_and_run_dsp(
        mod=IfSelectModule,
        input_data=(x, cond_f),
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
    )

    comparison = compare_results(dsp_results, reference, "numpy")
    return {
        "dsp_results": dsp_results,
        "reference": reference,
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# Pytest test cases
# ---------------------------------------------------------------------------

@pytest.mark.quick
@pytest.mark.c7x_only
def test_if_true_branch(dsp_mode, dsp_timeout, use_cpp_api):
    """
    If true branch: cond=1.0 → output = x + x = [2, 4, 6, 8].

    Validates that the add branch executes when the condition is non-zero.
    """
    results = _run_if_test(
        dsp_mode=dsp_mode,
        cond_value=1.0,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
    )
    # Log what we expect to see
    logger.info(f"Expected (true branch): {results['reference']}")
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


@pytest.mark.quick
@pytest.mark.c7x_only
def test_if_false_branch(dsp_mode, dsp_timeout, use_cpp_api):
    """
    If false branch: cond=0.0 → output = x * x = [1, 4, 9, 16].

    Validates that the multiply branch executes when the condition is zero.
    """
    results = _run_if_test(
        dsp_mode=dsp_mode,
        cond_value=0.0,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
    )
    logger.info(f"Expected (false branch): {results['reference']}")
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


@pytest.mark.c7x_only
def test_if_both_branches(dsp_mode, dsp_timeout, use_cpp_api):
    """
    Run both branches and verify outputs are distinct.

    The same compiled model is called twice with different conditions to
    confirm that runtime branch selection works correctly.  This is not
    marked quick because it runs two DSP compilations.
    """
    results_true = _run_if_test(
        dsp_mode=dsp_mode,
        cond_value=1.0,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
    )
    results_false = _run_if_test(
        dsp_mode=dsp_mode,
        cond_value=0.0,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
    )

    assert_dsp_comparison(results_true["dsp_results"], results_true["comparison"])
    assert_dsp_comparison(results_false["dsp_results"], results_false["comparison"])

    # The two outputs must be different (branches selected different ops)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(results_true["reference"], x + x)
    np.testing.assert_array_equal(results_false["reference"], x * x)


# ---------------------------------------------------------------------------
# Standalone script mode
# ---------------------------------------------------------------------------

def main():
    """Run If DSP test as standalone script."""
    parser = argparse.ArgumentParser(
        description="Relax If expression DSP test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_if_dsp.py --dsp-mode c7x_host
  python test_if_dsp.py --dsp-mode c7x_dload
  DSP_KEEP_TEMP=1 python test_if_dsp.py --dsp-mode c7x_host
""",
    )
    parser.add_argument(
        "--dsp-mode",
        required=True,
        choices=["c7x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument(
        "--timeout", type=int, default=60000,
        help="DSP execution timeout in ms (default: 60000)",
    )
    add_board_arg(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    all_passed = True

    for cond_val, branch_name, expected in [
        (1.0, "true  (add: x+x)", x + x),
        (0.0, "false (mul: x*x)", x * x),
    ]:
        print(f"\n{'='*60}")
        print(f"If branch: {branch_name}")
        print(f"  Input x:    {x}")
        print(f"  Expected:   {expected}")
        print(f"  Mode:       {args.dsp_mode}")
        print(f"{'='*60}")

        try:
            results = _run_if_test(
                dsp_mode=args.dsp_mode,
                cond_value=cond_val,
                timeout_ms=args.timeout,
            )
            comparison = results["comparison"]
            for key in sorted(comparison):
                if key.endswith("_vs_ref_max_diff"):
                    mode = key.removesuffix("_vs_ref_max_diff")
                    passed = comparison[key.replace("_max_diff", "_passed")]
                    status = "PASS" if passed else "FAIL"
                    diff = comparison[key]
                    print(f"  {mode}: {status}  (max diff: {diff:.2e})")
                    if not passed:
                        all_passed = False
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_passed = False

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
