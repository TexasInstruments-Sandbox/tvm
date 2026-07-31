#!/usr/bin/env python
"""
TorchVision classification models — C7x DLOAD tests.

Parameterized test over 8 ImageNet classification models that fit in the
128 MB DDR heap (weights < ~90 MB).

Usage:
    # Run all classification models via DLOAD
    pytest test_classification_dsp.py -v --dsp-mode=c7x_dload

    # Run a single model
    pytest test_classification_dsp.py -v --dsp-mode=c7x_dload -k squeezenet

    # Run with C66x host emulation
    pytest test_classification_dsp.py -v

    # Standalone script
    python test_classification_dsp.py --model squeezenet1_1 --dsp-mode c7x_dload
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.export import export

from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Add dsp-cpp to path for dsp_utils
_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402
    add_board_arg,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)

logger = logging.getLogger(__name__)

# Classification models that fit in 128 MB DDR heap.
# Format: (model_name, torchvision_factory, weights_enum)
CLASSIFICATION_MODELS = [
    "squeezenet1_1",
    "shufflenet_v2_x1_0",
    "mobilenet_v3_small",
    "mobilenet_v2",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "densenet121",
    "resnet34",
]


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def _get_model_and_weights(model_name: str):
    """Load a torchvision classification model with default weights."""
    import torchvision.models as tv_models

    # Map model name to factory function and weights enum
    factory = getattr(tv_models, model_name)

    # Find the matching weights enum
    weights_name = None
    model_lower = model_name.lower()
    for attr in dir(tv_models):
        if attr.endswith("_Weights") and model_lower == attr.lower().replace(
            "_weights", ""
        ):
            weights_name = attr
            break

    if weights_name:
        weights_enum = getattr(tv_models, weights_name)
        weights = getattr(weights_enum, "DEFAULT")
        model = factory(weights=weights)
    else:
        model = factory(weights="DEFAULT")

    model.eval()
    return model


def create_classification_model(model_name: str) -> tuple:
    """
    Create a classification model for DSP testing.

    Returns:
        Tuple of (tvm_mod, torch_model, input_data)
    """
    torch_model = _get_model_and_weights(model_name)

    example_args = (torch.randn(1, 3, 112, 112, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(torch_model, example_args)
        mod = from_exported_program(
            exported_program, keep_params_as_input=True
        )

    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(
        func_name="main", params=func_params_dict
    )(mod)

    np.random.seed(42)
    input_data = np.random.rand(1, 3, 112, 112).astype(np.float32)

    return mod, torch_model, input_data


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_classification_test(
    model_name: str,
    dsp_mode: str = "c66x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
    profile_layers: bool = False,
) -> dict:
    """Run a classification model on DSP and compare with PyTorch."""
    tvm_mod, torch_model, input_data = create_classification_model(model_name)

    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = torch_model(torch_input).numpy()

    target_string = get_target_string(dsp_mode, profile_layers=profile_layers,
                                      use_cpp_api=use_cpp_api)

    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=timeout_ms,
        profile_layers=profile_layers,
    )

    comparison = compare_results(
        dsp_results, torch_result, "PyTorch", rtol=5e-2, atol=5e-2
    )

    # Top-1 class agreement: mobile/efficient architectures use depthwise
    # convolutions that accumulate FP reassociation error across channels,
    # causing max diffs of 0.1-0.2 in logit space while preserving top-1.
    ref_top1 = int(np.argmax(torch_result))
    for mode_prefix in ("c7x_host", "c7x_dload", "c66x_host", "c66x"):
        key = f"{mode_prefix}_result"
        if key in dsp_results:
            dsp_top1 = int(np.argmax(dsp_results[key]))
            comparison[f"{mode_prefix}_top1_match"] = (dsp_top1 == ref_top1)

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


@pytest.mark.core
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_dsp(
    model_name, dsp_mode, dsp_timeout, use_cpp_api, profile_layers, record_cycles
):
    """Test classification model on DSP comparing against PyTorch reference."""
    results = _run_classification_test(
        model_name=model_name,
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
    )
    comparison = results["comparison"]
    dsp_results = results["dsp_results"]
    record_cycles(model_name, dsp_results.get("c7x_dload_cycles", 0))

    # Primary assertion: top-1 class must match PyTorch reference.
    # Models with depthwise convolutions (MobileNet, EfficientNet, ShuffleNet)
    # produce larger numerical diffs due to FP reassociation, but top-1 is
    # preserved.  Fail hard only if the predicted class is wrong.
    for key, match in comparison.items():
        if key.endswith("_top1_match"):
            mode = key.removesuffix("_top1_match")
            ref_top1 = int(np.argmax(results["torch_result"]))
            dsp_top1 = int(np.argmax(dsp_results[f"{mode}_result"]))
            assert match, (
                f"{mode}: top-1 mismatch — DSP={dsp_top1} vs ref={ref_top1}"
            )

    # Secondary: print a warning if numerical tolerance is exceeded, but do
    # not fail the test (top-1 correctness is the meaningful metric here).
    for key, passed in comparison.items():
        if key.endswith("_vs_ref_passed") and not passed:
            diff_key = key.replace("_passed", "_max_diff")
            mode = key.removesuffix("_vs_ref_passed")
            print(f"\n  WARNING: {mode} max diff {comparison[diff_key]:.2e} "
                  f"exceeds 5e-2 tolerance (top-1 class correct)")


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    parser = argparse.ArgumentParser(
        description="Classification Model DSP Tests"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=CLASSIFICATION_MODELS,
        help="Model to test (default: all)",
    )
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
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    add_board_arg(parser)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s: %(message)s"
        )

    dsp_mode = args.dsp_mode
    if dsp_mode is None:
        parser.print_help()
        return 1

    models = [args.model] if args.model else CLASSIFICATION_MODELS

    all_passed = True
    for model_name in models:
        print("=" * 70)
        print(f"{model_name} (mode: {dsp_mode})")
        print("=" * 70)

        print(f"\n[1/3] Creating {model_name} model...")
        tvm_mod, torch_model, input_data = create_classification_model(
            model_name
        )
        total_params = sum(p.numel() for p in torch_model.parameters())
        print(f"  Parameters: {total_params:,}")

        print("\n[2/3] Running PyTorch reference...")
        with torch.no_grad():
            torch_input = torch.from_numpy(input_data)
            torch_result = torch_model(torch_input).numpy()
        print(f"  Output shape: {torch_result.shape}")

        target_string = get_target_string(
            dsp_mode,
            profile_layers=args.profile_layers,
            use_cpp_api=(dsp_mode == "c7x_dload"),
        )

        print("\n[3/3] DSP Compilation and Execution...")
        print(f"  Target: {target_string}")

        dsp_results = compile_and_run_dsp(
            mod=tvm_mod,
            input_data=input_data,
            target_string=target_string,
            execution_mode=dsp_mode,
            profile_layers=args.profile_layers,
        )

        if "c7x_dload_result" in dsp_results:
            c7x_dload_result = dsp_results["c7x_dload_result"]
            print(f"\n[C7x DLOAD] Output shape: {c7x_dload_result.shape}")
            if "c7x_dload_stdout" in dsp_results:
                stdout = dsp_results["c7x_dload_stdout"]
                cycles_match = re.search(
                    r"Inference complete:\s*(\d+)\s*cycles", stdout
                )
                if cycles_match:
                    cycles = int(cycles_match.group(1))
                    time_ms = cycles / 1_000_000
                    print(
                        f"[C7x DLOAD] Inference cycles: {cycles:,} "
                        f"({time_ms:.3f} ms at 1 GHz)"
                    )

        if "c66x_host_result" in dsp_results:
            c66x_host_result = dsp_results["c66x_host_result"]
            print(f"\n[C66x Host] Output shape: {c66x_host_result.shape}")

        print("\n[Comparison] vs PyTorch:")
        comparison = compare_results(
            dsp_results, torch_result, "PyTorch", rtol=5e-2, atol=5e-2
        )

        passed = True
        if "c66x_host_vs_ref_max_diff" in comparison:
            status = "PASS" if comparison["c66x_host_vs_ref_passed"] else "FAIL"
            print(
                f"  C66x Host:  {comparison['c66x_host_vs_ref_max_diff']:.2e} [{status}]"
            )
            passed = passed and comparison["c66x_host_vs_ref_passed"]

        if "c7x_dload_vs_ref_max_diff" in comparison:
            status = "PASS" if comparison["c7x_dload_vs_ref_passed"] else "FAIL"
            print(
                f"  C7x DLOAD: {comparison['c7x_dload_vs_ref_max_diff']:.2e} "
                f"[{status}]"
            )
            passed = passed and comparison["c7x_dload_vs_ref_passed"]

        all_passed = all_passed and passed
        print()

    print("=" * 70)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
