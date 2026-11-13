"""
Quantized ShuffleNetV2 (x0.5) DSP test.

Tests INT8 quantized ShuffleNetV2 on DSP comparing against PyTorch reference.

Runs with --mmalib only (see MMALIB_ONLY_REASON in model_utils.py).

Usage:
    pytest test_quantized_shufflenet_v2.py -v --dsp-mode=c7x_host --mmalib
    pytest test_quantized_shufflenet_v2.py -v --dsp-mode=c7x_dload --mmalib
    python test_quantized_shufflenet_v2.py --dsp-mode c7x_host --mmalib
"""

import argparse
import logging
import sys

import numpy as np
import pytest
import torch
from dsp_utils import (
    add_board_arg,
    assert_dsp_comparison,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)
from model_utils import MMALIB_ONLY_REASON, create_quantized_shufflenet_v2_model

pytestmark = [pytest.mark.c7x_only, pytest.mark.core]

logger = logging.getLogger(__name__)

# Twice the largest |DSP - fake-quant reference| measured on c7x_host and
# BeagleY-AI (with MMALIB), rounded up to a multiple of 0.25. The output is
# deterministic for this input and identical on host and board, so the margin
# only absorbs toolchain and library changes.
_ATOL = 6.75


def _run_test(
    dsp_mode: str = "c7x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = True,
    profile_layers: bool = False,
    profile: bool = False,
    mmalib: bool = False,
) -> dict:
    tvm_mod, quantized_gm, input_data = create_quantized_shufflenet_v2_model()

    with torch.no_grad():
        torch_result = quantized_gm(torch.from_numpy(input_data)).numpy()

    target_string = get_target_string(
        dsp_mode, profile_layers=profile_layers, use_cpp_api=use_cpp_api
    )
    if mmalib:
        target_string += " -mmalib=1"

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
        profile_layers=profile_layers,
        profile=profile,
    )

    comparison = compare_results(
        dsp_results, torch_result, "PyTorch", atol=_ATOL, rtol=0.0, require_top1=True
    )

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


def test_quantized_shufflenet_v2_dsp(
    dsp_mode, dsp_timeout, use_cpp_api, profile_layers, profile, mmalib, record_cycles
):
    """Test quantized ShuffleNetV2 (x0.5) on DSP."""
    if not mmalib:
        pytest.skip(MMALIB_ONLY_REASON)
    results = _run_test(
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
        profile=profile,
        mmalib=mmalib,
    )
    record_cycles("shufflenet_v2_int8", results["dsp_results"].get("c7x_dload_cycles", 0))
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


def main():
    parser = argparse.ArgumentParser(description="Quantized ShuffleNetV2 (x0.5) DSP Test")
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    add_board_arg(parser)
    parser.add_argument("--mmalib", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    print(f"Quantized ShuffleNetV2 (x0.5) DSP Test (mode: {args.dsp_mode})")
    results = _run_test(
        dsp_mode=args.dsp_mode,
        profile_layers=args.profile,
        profile=args.profile,
        mmalib=args.mmalib,
    )

    dsp_results = results["dsp_results"]
    torch_result = results["torch_result"]

    for key in ("c7x_host_result", "c7x_dload_result"):
        if key in dsp_results:
            diff = np.max(np.abs(dsp_results[key] - torch_result))
            passed = results["comparison"].get(key.replace("_result", "_vs_ref_passed"), False)
            status = "PASS" if passed else "FAIL"
            print(f"  {key}: max_diff={diff:.2e} [{status}]")

    return 0 if all(v for k, v in results["comparison"].items() if k.endswith("_passed")) else 1


if __name__ == "__main__":
    sys.exit(main())
