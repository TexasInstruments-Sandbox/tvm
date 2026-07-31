"""
Dynamic shape tests on c_static / C7x DSP.

Validates that symbolic shape dimensions compile and execute correctly
through the c_static backend on C7x host emulation and C7x DLOAD hardware.

Models
------
DynBatchAdd:    x[batch,4] + y[batch,4]  (element-wise, 1 symbolic dim)
DynBatchMatmul: x[batch,K] @ w[K,N]     (matmul, shape_func computes output size)

The shape heap mechanism stores runtime dimensions at model-entry time
via match_shape, then shape_func computes derived values (storage sizes),
and make_shape constructs output tensor shapes.

Usage
-----
    pytest tests/ti-dsp-runtime/dynamic-tests/test_dynamic_batch_dsp.py \\
        -m quick --dsp-mode=c7x_host -v

    pytest tests/ti-dsp-runtime/dynamic-tests/test_dynamic_batch_dsp.py \\
        -m quick --dsp-mode=c7x_dload -v
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm.script import relax as R
from tvm.script import tir as T

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
# Relax module: dynamic batch dimension
# ---------------------------------------------------------------------------

@tvm.script.ir_module
class DynBatchAdd:
    """
    Element-wise add with symbolic batch dimension.

    The 'batch' dimension is unknown at compile time.  The shape heap
    mechanism (alloc_shape_heap -> match_shape -> shape_func -> make_shape)
    stores the runtime batch value and uses it for output allocation.
    """

    @R.function
    def main(
        x: R.Tensor(("batch", 4), dtype="float32"),
        y: R.Tensor(("batch", 4), dtype="float32"),
    ) -> R.Tensor(("batch", 4), dtype="float32"):
        R.func_attr({"num_input": 2})
        batch = T.int64()
        return R.add(x, y)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_dynamic_batch_test(
    dsp_mode: str,
    batch_size: int,
    timeout_ms: int = 60000,
    use_cpp_api: bool = False,
) -> dict:
    """
    Compile and run the dynamic batch model for a given batch size.

    Returns:
        dict with keys: dsp_results, reference, comparison
    """
    np.random.seed(42)
    x = np.random.randn(batch_size, 4).astype(np.float32)
    y = np.random.randn(batch_size, 4).astype(np.float32)

    reference = x + y

    target_string = get_target_string(dsp_mode, use_cpp_api=use_cpp_api)

    dsp_results = compile_and_run_dsp(
        mod=DynBatchAdd,
        input_data=(x, y),
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
def test_dynamic_batch_1(dsp_mode, dsp_timeout, use_cpp_api):
    """Dynamic batch=1 (degenerate case)."""
    results = _run_dynamic_batch_test(
        dsp_mode=dsp_mode, batch_size=1,
        timeout_ms=dsp_timeout, use_cpp_api=use_cpp_api,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


@pytest.mark.quick
@pytest.mark.c7x_only
def test_dynamic_batch_4(dsp_mode, dsp_timeout, use_cpp_api):
    """Dynamic batch=4."""
    results = _run_dynamic_batch_test(
        dsp_mode=dsp_mode, batch_size=4,
        timeout_ms=dsp_timeout, use_cpp_api=use_cpp_api,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


@pytest.mark.quick
@pytest.mark.c7x_only
def test_dynamic_batch_8(dsp_mode, dsp_timeout, use_cpp_api):
    """Dynamic batch=8."""
    results = _run_dynamic_batch_test(
        dsp_mode=dsp_mode, batch_size=8,
        timeout_ms=dsp_timeout, use_cpp_api=use_cpp_api,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# ---------------------------------------------------------------------------
# Relax module: dynamic batch matmul
# ---------------------------------------------------------------------------

@tvm.script.ir_module
class DynBatchMatmul:
    """
    Matrix multiply with symbolic batch dimension.

    x[batch, K] @ w[K, N] -> out[batch, N]

    More demanding than DynBatchAdd because:
    - shape_func must compute a non-trivial output storage size
      (batch * N * sizeof(float32)) from the runtime batch value
    - The TIR matmul kernel iterates over the dynamic batch dimension
    - Weight tensor w has static shape (no shape heap entry)
    """

    @R.function
    def main(
        x: R.Tensor(("batch", 8), dtype="float32"),
        w: R.Tensor((8, 4), dtype="float32"),
    ) -> R.Tensor(("batch", 4), dtype="float32"):
        R.func_attr({"num_input": 2})
        batch = T.int64()
        return R.matmul(x, w)


def _run_dynamic_matmul_test(
    dsp_mode: str,
    batch_size: int,
    timeout_ms: int = 60000,
    use_cpp_api: bool = False,
) -> dict:
    """Compile and run the dynamic matmul model for a given batch size."""
    np.random.seed(42)
    x = np.random.randn(batch_size, 8).astype(np.float32)
    w = np.random.randn(8, 4).astype(np.float32)

    reference = x @ w

    target_string = get_target_string(dsp_mode, use_cpp_api=use_cpp_api)

    dsp_results = compile_and_run_dsp(
        mod=DynBatchMatmul,
        input_data=(x, w),
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


@pytest.mark.quick
@pytest.mark.c7x_only
def test_dynamic_matmul_batch_1(dsp_mode, dsp_timeout, use_cpp_api):
    """Dynamic matmul batch=1 (single vector-matrix multiply)."""
    results = _run_dynamic_matmul_test(
        dsp_mode=dsp_mode, batch_size=1,
        timeout_ms=dsp_timeout, use_cpp_api=use_cpp_api,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


@pytest.mark.quick
@pytest.mark.c7x_only
def test_dynamic_matmul_batch_4(dsp_mode, dsp_timeout, use_cpp_api):
    """Dynamic matmul batch=4."""
    results = _run_dynamic_matmul_test(
        dsp_mode=dsp_mode, batch_size=4,
        timeout_ms=dsp_timeout, use_cpp_api=use_cpp_api,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


@pytest.mark.quick
@pytest.mark.c7x_only
def test_dynamic_matmul_batch_16(dsp_mode, dsp_timeout, use_cpp_api):
    """Dynamic matmul batch=16 (larger batch, more accumulation)."""
    results = _run_dynamic_matmul_test(
        dsp_mode=dsp_mode, batch_size=16,
        timeout_ms=dsp_timeout, use_cpp_api=use_cpp_api,
    )
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# ---------------------------------------------------------------------------
# Standalone script mode
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dynamic batch DSP test")
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    parser.add_argument("--timeout", type=int, default=60000)
    add_board_arg(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    all_passed = True
    tests = [
        ("DynBatchAdd", _run_dynamic_batch_test, [1, 4, 8]),
        ("DynBatchMatmul", _run_dynamic_matmul_test, [1, 4, 16]),
    ]
    for model_name, runner, batches in tests:
        for batch in batches:
            print(f"\n{'='*60}")
            print(f"{model_name} batch={batch}, mode={args.dsp_mode}")
            print(f"{'='*60}")
            try:
                results = runner(args.dsp_mode, batch, args.timeout)
                for key in sorted(results["comparison"]):
                    if key.endswith("_vs_ref_max_diff"):
                        mode = key.removesuffix("_vs_ref_max_diff")
                        passed = results["comparison"][key.replace("_max_diff", "_passed")]
                        print(f"  {mode}: {'PASS' if passed else 'FAIL'}  "
                              f"(max diff: {results['comparison'][key]:.2e})")
                        if not passed:
                            all_passed = False
            except Exception as exc:
                print(f"  ERROR: {exc}")
                all_passed = False

    print(f"\n{'='*60}")
    print("ALL PASSED" if all_passed else "SOME FAILED")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
