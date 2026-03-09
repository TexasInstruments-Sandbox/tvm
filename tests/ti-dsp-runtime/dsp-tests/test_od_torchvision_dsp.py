#!/usr/bin/env python
"""
SSDLite320 MobileNetV3 object detection — C7x DLOAD test.

Tests the SSDLite320_MobileNet_V3_Large model on DSP. The model wrapper
bypasses the internal GeneralizedRCNNTransform and returns raw
(bbox_regression, cls_logits) tensors. Post-processing (NMS, box
decoding) stays in Python.

Usage:
    # Run via DLOAD on C7x hardware
    pytest test_od_torchvision_dsp.py -v --dsp-mode=c7x_dload

    # Run with C66x host emulation
    pytest test_od_torchvision_dsp.py -v

    # Standalone script
    python test_od_torchvision_dsp.py --dsp-mode c7x_dload
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
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

logger = logging.getLogger(__name__)

MODEL_NAME = "ssdlite320_mobilenet_v3_large"
INPUT_SHAPE = (1, 3, 320, 320)


# -----------------------------------------------------------------------------
# Detection Model Wrapper
# -----------------------------------------------------------------------------


class DetectionModelWrapper(nn.Module):
    """Wrapper to export SSD/SSDLite backbone + head for torch.export.

    Bypasses GeneralizedRCNNTransform and returns raw tuple of
    (bbox_regression, cls_logits).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.backbone(x)
        if isinstance(features, dict):
            features_list = list(features.values())
        else:
            features_list = features

        processed_features = [
            feat.unsqueeze(0) if feat.ndim == 3 else feat
            for feat in features_list
        ]

        outputs = self.head(processed_features)
        bbox_regression = outputs["bbox_regression"]
        cls_logits = outputs["cls_logits"]
        return bbox_regression, cls_logits


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def create_od_model() -> tuple:
    """
    Create SSDLite320 MobileNetV3 model for DSP testing.

    Returns:
        Tuple of (tvm_mod, wrapped_model, torch_model, input_data)
    """
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    torch_model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )
    torch_model.eval()

    wrapped = DetectionModelWrapper(torch_model)
    wrapped.eval()

    example_args = (torch.randn(*INPUT_SHAPE, dtype=torch.float32),)

    with torch.no_grad():
        exported_program = export(wrapped, example_args, strict=False)
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

    return mod, wrapped, torch_model, input_data


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_od_test(
    dsp_mode: str = "c66x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
    profile_layers: bool = False,
) -> dict:
    """Run SSDLite320 on DSP and compare with PyTorch."""
    tvm_mod, wrapped, torch_model, input_data = create_od_model()

    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_outputs = wrapped(torch_input)
        torch_outputs = [out.numpy() for out in torch_outputs]

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


def _check_multi_output(dsp_result, torch_outputs, label, rtol, atol):
    """Validate multi-output DSP result against PyTorch reference."""
    assert isinstance(dsp_result, list), (
        f"Expected list of outputs, got {type(dsp_result)}"
    )
    assert len(dsp_result) == len(torch_outputs), (
        f"Expected {len(torch_outputs)} outputs, got {len(dsp_result)}"
    )
    for i, (dsp_out, torch_out) in enumerate(
        zip(dsp_result, torch_outputs)
    ):
        assert dsp_out.shape == torch_out.shape, (
            f"Output {i} shape mismatch: {dsp_out.shape} vs {torch_out.shape}"
        )
        max_diff = np.max(np.abs(dsp_out - torch_out))
        assert np.allclose(dsp_out, torch_out, rtol=rtol, atol=atol), (
            f"{label} output {i}: max diff = {max_diff:.2e}"
        )


def test_od_torchvision_dsp(
    dsp_mode, dsp_timeout, use_cpp_api, profile_layers
):
    """Test SSDLite320 MobileNetV3 on DSP vs PyTorch reference."""
    results = _run_od_test(
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
    )

    dsp_results = results["dsp_results"]
    torch_outputs = results["torch_outputs"]

    if "c66x_host_error" in dsp_results:
        raise AssertionError(
            f"Host execution error: {dsp_results['c66x_host_error']}"
        )
    if "c7x_dload_error" in dsp_results:
        raise AssertionError(
            f"DLOAD execution error: {dsp_results['c7x_dload_error']}"
        )

    rtol, atol = 0.1, 0.1

    has_results = False

    if "c66x_host_result" in dsp_results:
        _check_multi_output(
            dsp_results["c66x_host_result"], torch_outputs, "Host", rtol, atol
        )
        has_results = True

    if "c7x_dload_result" in dsp_results:
        _check_multi_output(
            dsp_results["c7x_dload_result"], torch_outputs, "DLOAD", rtol, atol
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
        description="SSDLite320 MobileNetV3 DSP Test"
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

    print("=" * 70)
    print(f"SSDLite320 MobileNetV3 (mode: {dsp_mode})")
    print("=" * 70)

    print("\n[1/3] Creating model...")
    tvm_mod, wrapped, torch_model, input_data = create_od_model()
    total_params = sum(p.numel() for p in torch_model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Input shape: {input_data.shape}")

    print("\n[2/3] Running PyTorch reference...")
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_outputs = wrapped(torch_input)
        torch_outputs = [out.numpy() for out in torch_outputs]
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
            print(f"\n[{label}] Outputs: {len(dsp_result)}")
            for i, (dsp_out, torch_out) in enumerate(
                zip(dsp_result, torch_outputs)
            ):
                print(f"[{label}] Output {i} shape: {dsp_out.shape}")
                max_diff = np.max(np.abs(dsp_out - torch_out))
                matches = np.allclose(
                    dsp_out, torch_out, rtol=0.1, atol=0.1
                )
                status = "PASS" if matches else "FAIL"
                print(
                    f"[{label}] Output {i} vs PyTorch: "
                    f"max diff = {max_diff:.2e} [{status}]"
                )
                passed = passed and matches
        else:
            print(f"\n[{label}] Single output shape: {dsp_result.shape}")
            passed = False

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

    print("\n" + "=" * 70)
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
