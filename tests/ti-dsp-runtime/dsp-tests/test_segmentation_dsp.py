#!/usr/bin/env python
"""
TorchVision segmentation models — C7x DLOAD tests.

Tests LRASPP MobileNetV3 and DeepLabV3 MobileNetV3 on DSP.
Both models return 2 outputs (main segmentation map + auxiliary head).
Comparison is done on the main output (index 0).

Usage:
    # Run all segmentation models via DLOAD
    pytest test_segmentation_dsp.py -v --dsp-mode=c7x_dload

    # Run only LRASPP
    pytest test_segmentation_dsp.py -v --dsp-mode=c7x_dload -k lraspp

    # Run with C66x host emulation
    pytest test_segmentation_dsp.py -v

    # Standalone script
    python test_segmentation_dsp.py --model lraspp_mobilenet_v3_large --dsp-mode c7x_dload
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

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402

pytestmark = [pytest.mark.c7x_only]

logger = logging.getLogger(__name__)

INPUT_SHAPE = (1, 3, 112, 112)

SEGMENTATION_MODELS = [
    "lraspp_mobilenet_v3_large",
    "deeplabv3_mobilenet_v3_large",
]


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def create_segmentation_model(model_name: str) -> tuple:
    """
    Create a segmentation model for DSP testing.

    Returns:
        Tuple of (tvm_mod, torch_model, input_data)
    """
    from torchvision.models import segmentation as seg_models

    # Load model with default weights
    model_func = getattr(seg_models, model_name)

    # Find weights enum
    weights_enum = None
    model_lower = model_name.lower()
    for attr in dir(seg_models):
        if attr.endswith("_Weights") and model_lower == attr.lower().replace(
            "_weights", ""
        ):
            weights_enum = getattr(seg_models, attr)
            break

    if weights_enum:
        weights = getattr(weights_enum, "DEFAULT")
        torch_model = model_func(weights=weights)
    else:
        torch_model = model_func(weights="DEFAULT")

    torch_model.eval()

    example_args = (torch.randn(*INPUT_SHAPE, dtype=torch.float32),)

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
    input_data = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    return mod, torch_model, input_data


def _get_torch_outputs(torch_model, input_data):
    """Run PyTorch inference and return outputs as list of numpy arrays.

    Segmentation models return OrderedDict with 'out' and optionally 'aux'.
    We convert to a list for consistent comparison with DSP multi-output.
    """
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        result = torch_model(torch_input)

    # PyTorch segmentation models return OrderedDict {'out': ..., 'aux': ...}
    if isinstance(result, dict):
        outputs = []
        if "out" in result:
            outputs.append(result["out"].numpy())
        if "aux" in result:
            outputs.append(result["aux"].numpy())
        return outputs

    # Fallback for tuple/list
    if isinstance(result, (tuple, list)):
        return [r.numpy() for r in result]

    return [result.numpy()]


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_segmentation_test(
    model_name: str,
    dsp_mode: str = "c66x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
    profile_layers: bool = False,
) -> dict:
    """Run a segmentation model on DSP and compare with PyTorch."""
    tvm_mod, torch_model, input_data = create_segmentation_model(model_name)

    torch_outputs = _get_torch_outputs(torch_model, input_data)

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

    return {
        "torch_outputs": torch_outputs,
        "dsp_results": dsp_results,
    }


def _check_main_output(dsp_result, torch_outputs, label, rtol, atol):
    """Compare the main segmentation output (index 0) vs PyTorch."""
    if isinstance(dsp_result, list):
        dsp_main = dsp_result[0]
    else:
        dsp_main = dsp_result

    torch_main = torch_outputs[0]

    assert dsp_main.shape == torch_main.shape, (
        f"{label} main output shape mismatch: "
        f"{dsp_main.shape} vs {torch_main.shape}"
    )

    max_diff = np.max(np.abs(dsp_main - torch_main))
    assert np.allclose(dsp_main, torch_main, rtol=rtol, atol=atol), (
        f"{label} main output: max diff = {max_diff:.2e}"
    )


@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_dsp(
    model_name, dsp_mode, dsp_timeout, use_cpp_api, profile_layers
):
    """Test segmentation model on DSP comparing main output vs PyTorch."""
    results = _run_segmentation_test(
        model_name=model_name,
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
    )

    dsp_results = results["dsp_results"]
    torch_outputs = results["torch_outputs"]

    if "c66x_host_error" in dsp_results:
        raise AssertionError(
            f"C66x Host execution error: {dsp_results['c66x_host_error']}"
        )
    if "c7x_dload_error" in dsp_results:
        raise AssertionError(
            f"C7x DLOAD execution error: {dsp_results['c7x_dload_error']}"
        )

    rtol, atol = 5e-2, 5e-2
    has_results = False

    if "c66x_host_result" in dsp_results:
        _check_main_output(
            dsp_results["c66x_host_result"], torch_outputs, "C66x Host", rtol, atol
        )
        has_results = True

    if "c7x_dload_result" in dsp_results:
        _check_main_output(
            dsp_results["c7x_dload_result"], torch_outputs, "C7x DLOAD", rtol, atol
        )
        has_results = True

    assert has_results, (
        "No DSP results available. Check hardware connection or mode."
    )


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    parser = argparse.ArgumentParser(
        description="Segmentation Model DSP Tests"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=SEGMENTATION_MODELS,
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
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s: %(message)s"
        )

    dsp_mode = args.dsp_mode
    if dsp_mode is None:
        parser.print_help()
        return 1

    models = [args.model] if args.model else SEGMENTATION_MODELS

    all_passed = True
    for model_name in models:
        print("=" * 70)
        print(f"{model_name} (mode: {dsp_mode})")
        print("=" * 70)

        print(f"\n[1/3] Creating {model_name} model...")
        tvm_mod, torch_model, input_data = create_segmentation_model(
            model_name
        )
        total_params = sum(p.numel() for p in torch_model.parameters())
        print(f"  Parameters: {total_params:,}")

        print("\n[2/3] Running PyTorch reference...")
        torch_outputs = _get_torch_outputs(torch_model, input_data)
        for i, out in enumerate(torch_outputs):
            print(f"  Output {i} shape: {out.shape}")

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

        passed = True
        result_key = "c7x_dload_result" if dsp_mode == "c7x_dload" else "c66x_host_result"
        label = "C7x DLOAD" if dsp_mode == "c7x_dload" else "C66x Host"

        if result_key in dsp_results:
            dsp_result = dsp_results[result_key]
            if isinstance(dsp_result, list):
                dsp_main = dsp_result[0]
            else:
                dsp_main = dsp_result

            torch_main = torch_outputs[0]
            print(f"\n[{label}] Main output shape: {dsp_main.shape}")
            max_diff = np.max(np.abs(dsp_main - torch_main))
            matches = np.allclose(
                dsp_main, torch_main, rtol=5e-2, atol=5e-2
            )
            status = "PASS" if matches else "FAIL"
            print(
                f"[{label}] Main output vs PyTorch: "
                f"max diff = {max_diff:.2e} [{status}]"
            )
            passed = passed and matches

            if "c7x_dload_stdout" in dsp_results:
                stdout = dsp_results["c7x_dload_stdout"]
                cycles_match = re.search(
                    r"Inference complete:\s*(\d+)\s*cycles", stdout
                )
                if cycles_match:
                    cycles = int(cycles_match.group(1))
                    time_ms = cycles / 1_000_000
                    print(
                        f"[{label}] Inference cycles: {cycles:,} "
                        f"({time_ms:.3f} ms at 1 GHz)"
                    )
        else:
            error_key = f"{dsp_mode}_error"
            if error_key in dsp_results:
                print(f"\n[{label}] SKIPPED: {dsp_results[error_key]}")
            else:
                print(f"\n[{label}] No results available")
            passed = False

        all_passed = all_passed and passed
        print()

    print("=" * 70)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
