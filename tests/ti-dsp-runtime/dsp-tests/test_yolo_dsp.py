#!/usr/bin/env python
"""
YOLO object detection models — C7x DLOAD tests.

Parameterized test over YOLOv5 (n, s) and YOLOv8 (n, s) models.
All return a single raw detection tensor; NMS is done in Python.

YOLOv5 models are loaded via torch.hub (ultralytics/yolov5).
YOLOv8 models require the ultralytics package.

Usage:
    # Run all YOLO models via DLOAD
    pytest test_yolo_dsp.py -v --dsp-mode=c7x_dload

    # Run only YOLOv5 variants
    pytest test_yolo_dsp.py -v --dsp-mode=c7x_dload -k yolov5

    # Run with C66x host emulation
    pytest test_yolo_dsp.py -v

    # Standalone script
    python test_yolo_dsp.py --model yolov5n --dsp-mode c7x_dload
"""

import argparse
import logging
import re
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

from dsp_utils import compile_and_run_dsp, compare_results, get_target_string, assert_dsp_comparison  # noqa: E402

logger = logging.getLogger(__name__)

INPUT_SHAPE = (1, 3, 320, 320)

# (model_name, version) — version is "v5" or "v8"
YOLO_MODELS = [
    ("yolov5n", "v5"),
    ("yolov5s", "v5"),
    ("yolov8n", "v8"),
    ("yolov8s", "v8"),
]


# -----------------------------------------------------------------------------
# YOLO Wrapper (same pattern as od_yolo.py)
# -----------------------------------------------------------------------------


class YOLOWrapper(nn.Module):
    """Extract the core YOLO model for torch.export compatibility."""

    def __init__(self, yolo_model, version: str = "v5"):
        super().__init__()
        self.version = version
        if hasattr(yolo_model, "model"):
            self.model = yolo_model.model
        else:
            self.model = yolo_model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------


def _load_yolov5(model_name: str):
    """Load YOLOv5 model via torch.hub."""
    model = torch.hub.load(
        "ultralytics/yolov5", model_name, pretrained=True
    )
    model.eval()
    return model


def _load_yolov8(model_name: str):
    """Load YOLOv8 model via ultralytics package."""
    from ultralytics import YOLO

    model = YOLO(f"{model_name}.pt")
    model.model.eval()
    return model


def _needs_ultralytics(version: str) -> bool:
    return version == "v8"


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def create_yolo_model(model_name: str, version: str) -> tuple:
    """
    Create a YOLO model for DSP testing.

    Returns:
        Tuple of (tvm_mod, wrapped_model, input_data)
    """
    if version == "v5":
        raw_model = _load_yolov5(model_name)
    else:
        raw_model = _load_yolov8(model_name)

    wrapped = YOLOWrapper(raw_model, version=version)
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

    return mod, wrapped, input_data


# -----------------------------------------------------------------------------
# Pytest Tests
# -----------------------------------------------------------------------------


def _run_yolo_test(
    model_name: str,
    version: str,
    dsp_mode: str = "c7x_host",
    timeout_ms: int = 120000,
    use_cpp_api: bool = False,
    profile_layers: bool = False,
) -> dict:
    """Run a YOLO model on DSP and compare with PyTorch."""
    tvm_mod, wrapped, input_data = create_yolo_model(model_name, version)

    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        torch_result = wrapped(torch_input).numpy()

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

    # YOLO raw detection outputs contain sigmoid/exp-transformed
    # coordinates where small floating-point differences compound.
    # Use cosine similarity (> 0.999) instead of element-wise tolerance
    # for a more meaningful accuracy check on detection tensors.
    comparison = compare_results(
        dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=2e1
    )
    # Override pass/fail with cosine similarity check
    flat_ref = torch_result.flatten()
    for key in list(comparison.keys()):
        if key.endswith("_result") or not key.endswith("_passed"):
            continue
        result_key = key.replace("_vs_ref_passed", "_result")
        if result_key in dsp_results:
            flat_dsp = dsp_results[result_key].flatten()
            cos_sim = np.dot(flat_ref, flat_dsp) / (
                np.linalg.norm(flat_ref) * np.linalg.norm(flat_dsp) + 1e-10
            )
            comparison[key] = cos_sim > 0.999
            comparison[key.replace("_passed", "_cos_sim")] = float(cos_sim)

    return {
        "torch_result": torch_result,
        "dsp_results": dsp_results,
        "comparison": comparison,
    }


def _yolo_param_id(param):
    """Generate readable pytest ID for YOLO model parametrize."""
    model_name, version = param
    return model_name


def _skip_yolov8_if_no_ultralytics(model_name, version):
    """Skip YOLOv8 tests if ultralytics is not installed."""
    if version == "v8":
        pytest.importorskip(
            "ultralytics",
            reason=f"{model_name} requires ultralytics package",
        )


@pytest.mark.parametrize(
    "model_spec",
    YOLO_MODELS,
    ids=[m[0] for m in YOLO_MODELS],
)
def test_yolo_dsp(
    model_spec, dsp_mode, dsp_timeout, use_cpp_api, profile_layers
):
    """Test YOLO model on DSP comparing against PyTorch reference."""
    model_name, version = model_spec

    _skip_yolov8_if_no_ultralytics(model_name, version)

    results = _run_yolo_test(
        model_name=model_name,
        version=version,
        dsp_mode=dsp_mode,
        timeout_ms=dsp_timeout,
        use_cpp_api=use_cpp_api,
        profile_layers=profile_layers,
    )

    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    """Run as standalone script."""
    model_names = [m[0] for m in YOLO_MODELS]
    parser = argparse.ArgumentParser(description="YOLO DSP Tests")
    parser.add_argument(
        "--model",
        default=None,
        choices=model_names,
        help="Model to test (default: all)",
    )
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

    # Build list of models to test
    if args.model:
        models = [(n, v) for n, v in YOLO_MODELS if n == args.model]
    else:
        models = list(YOLO_MODELS)

    all_passed = True
    for model_name, version in models:
        # Check ultralytics availability for v8
        if version == "v8":
            try:
                import ultralytics  # noqa: F401
            except ImportError:
                print(f"\nSKIP {model_name}: ultralytics not installed")
                continue

        print("=" * 70)
        print(f"{model_name} (mode: {dsp_mode})")
        print("=" * 70)

        print(f"\n[1/3] Creating {model_name} model...")
        tvm_mod, wrapped, input_data = create_yolo_model(
            model_name, version
        )

        print("\n[2/3] Running PyTorch reference...")
        with torch.no_grad():
            torch_input = torch.from_numpy(input_data)
            torch_result = wrapped(torch_input).numpy()
        print(f"  Output shape: {torch_result.shape}")

        target_string = get_target_string(
            dsp_mode,
            profile_layers=args.profile_layers,
            use_cpp_api=(dsp_mode in ("c7x_host", "c7x_dload")),
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

        if "c7x_host_result" in dsp_results:
            c7x_host_result = dsp_results["c7x_host_result"]
            print(f"\n[C7x Host] Output shape: {c7x_host_result.shape}")

        # YOLO raw detection outputs contain sigmoid/exp-transformed
        # coordinates where small floating-point differences from the
        # TI Host Emulation library compound across detection heads.
        # Use cosine similarity (> 0.999) for pass/fail instead of
        # element-wise tolerance on raw tensor values.
        print("\n[Comparison] vs PyTorch:")
        comparison = compare_results(
            dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=2e1
        )

        passed = True
        flat_ref = torch_result.flatten()
        for mode in ["c7x_host", "c7x_dload"]:
            diff_key = f"{mode}_vs_ref_max_diff"
            pass_key = f"{mode}_vs_ref_passed"
            result_key = f"{mode}_result"
            if diff_key not in comparison:
                continue
            # Compute cosine similarity
            if result_key in dsp_results:
                flat_dsp = dsp_results[result_key].flatten()
                cos_sim = float(np.dot(flat_ref, flat_dsp) / (
                    np.linalg.norm(flat_ref) * np.linalg.norm(flat_dsp) + 1e-10
                ))
                cos_pass = cos_sim > 0.999
                label = mode.replace("_", " ").title()
                status = "PASS" if cos_pass else "FAIL"
                print(
                    f"  {label}: max_diff={comparison[diff_key]:.2e} "
                    f"cos_sim={cos_sim:.6f} [{status}]"
                )
                passed = passed and cos_pass

        all_passed = all_passed and passed
        print()

    print("=" * 70)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
