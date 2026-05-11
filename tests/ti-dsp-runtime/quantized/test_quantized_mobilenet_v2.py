"""
Quantized MobileNetV2 DSP test.

Tests INT8 quantized MobileNetV2 on DSP comparing against PyTorch reference.

Usage:
    pytest test_quantized_mobilenet_v2.py -v --dsp-mode=c7x_host
    pytest test_quantized_mobilenet_v2.py -v --dsp-mode=c7x_host --mmalib
    pytest test_quantized_mobilenet_v2.py -v --dsp-mode=c7x_dload
    python test_quantized_mobilenet_v2.py --dsp-mode c7x_host
"""

import argparse
import logging
import sys

import numpy as np
import pytest
import torch
from dsp_utils import (
    assert_dsp_comparison,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)
from model_utils import create_quantized_mobilenet_v2_model

pytestmark = [pytest.mark.c7x_only, pytest.mark.core]

logger = logging.getLogger(__name__)


def _run_test(
    dsp_mode: str = "c7x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = True,
    profile_layers: bool = False,
    profile: bool = False,
    mmalib: bool = False,
) -> dict:
    tvm_mod, quantized_gm, input_data = create_quantized_mobilenet_v2_model()

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

    if mmalib:
        comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=5e-1, atol=25.0)
    else:
        comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=20.0)

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


def test_quantized_mobilenet_v2_dsp(
    dsp_mode, dsp_timeout, use_cpp_api, profile_layers, profile, mmalib, record_cycles
):
    """Test quantized MobileNetV2 on DSP."""
    results = _run_test(
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
        profile=profile,
        mmalib=mmalib,
    )
    record_cycles("mobilenet_v2_int8", results["dsp_results"].get("c7x_dload_cycles", 0))
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


def main():
    parser = argparse.ArgumentParser(description="Quantized MobileNetV2 DSP Test")
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    parser.add_argument("--mmalib", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    print(f"Quantized MobileNetV2 DSP Test (mode: {args.dsp_mode})")
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
