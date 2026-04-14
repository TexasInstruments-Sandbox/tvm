#!/usr/bin/env python
"""
YOLO object detection models — C7x DLOAD and TIDL offloading tests.

Parameterized test over YOLOv5 (n, s) and YOLOv8 (n, s) models.
All return a single raw detection tensor; NMS is done in Python.

YOLOv5 models are loaded via torch.hub (ultralytics/yolov5).
YOLOv8 models require the ultralytics package.

Pure-TVM DSP tests (test_yolo_dsp):
    # Run all YOLO models via DLOAD
    pytest test_yolo_dsp.py -v --dsp-mode=c7x_dload

    # Run only YOLOv5 variants
    pytest test_yolo_dsp.py -v --dsp-mode=c7x_dload -k yolov5

TIDL offloading tests (TestYOLOTIDL, n-variants only):
    # Build-only (no hardware, requires tidl_model_import_relax.so + C7x compiler)
    pytest test_yolo_dsp.py::TestYOLOTIDL::test_yolo_tidl_build -v

    # Hardware correctness test (requires AM67A with c7x_compute firmware)
    pytest test_yolo_dsp.py::TestYOLOTIDL::test_yolo_tidl_correctness -v

Standalone script:
    python test_yolo_dsp.py --model yolov5n --dsp-mode c7x_dload
    python test_yolo_dsp.py --model yolov5n --visualize partitioning.html
"""

import argparse
import logging
import os
import re
import shutil
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

pytestmark = [pytest.mark.c7x_only]

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# TIDL paths and dependency helpers
# -----------------------------------------------------------------------------

C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH",
    os.path.expanduser("~/ml/c7x-mma-tidl"),
)
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH,
    "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so",
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")


def _has_import_so():
    return os.path.isfile(RELAX_SO_PATH)


def _has_c7x_compiler():
    return os.environ.get("TI_CGT_C7000_PATH") is not None


INPUT_SHAPE = (1, 3, 320, 320)

# (model_name, version) — version is "v5" or "v8"
YOLO_MODELS = [
    ("yolov5n", "v5"),
    ("yolov5s", "v5"),
    ("yolov8n", "v8"),
    ("yolov8s", "v8"),
]

# Subset used for TIDL tests: n-variants only (faster build, ~2-3 min each)
YOLO_TIDL_MODELS = [
    ("yolov5n", "v5"),
    ("yolov8n", "v8"),
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


# -----------------------------------------------------------------------------
# Model Creation
# -----------------------------------------------------------------------------


def _create_yolo_model_unbound(model_name: str, version: str) -> tuple:
    """
    Create a YOLO model with unbound parameters.

    Used by TIDL tests: TIDLOffloadCompiler.build() expects the raw module
    (params still as function arguments) plus a separate param_dict.

    Returns:
        Tuple of (tvm_mod, param_dict, wrapped_model, input_data)
        where param_dict maps Var -> tvm.runtime.NDArray.
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
    param_dict = dict(zip(mod["main"].params[1:], params["main"]))

    np.random.seed(42)
    input_data = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    return mod, param_dict, wrapped, input_data


def create_yolo_model(model_name: str, version: str) -> tuple:
    """
    Create a YOLO model for DSP testing.

    Returns:
        Tuple of (tvm_mod, wrapped_model, input_data)
    """
    mod, param_dict, wrapped, input_data = _create_yolo_model_unbound(
        model_name, version
    )
    mod = relax.transform.BindParams(
        func_name="main", params=param_dict
    )(mod)
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


def _skip_yolo_if_no_ultralytics(model_name):
    """Skip YOLO tests if ultralytics is not installed.

    Both v5 and v8 depend on the ultralytics package: v8 directly imports
    ultralytics.YOLO, and v5 uses the torch.hub entry point which now also
    requires ultralytics (the modern yolov5 hubconf.py imports from it).
    """
    pytest.importorskip(
        "ultralytics",
        reason=f"{model_name} requires the ultralytics package",
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

    _skip_yolo_if_no_ultralytics(model_name)

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
# TIDL Offloading Tests
# -----------------------------------------------------------------------------


class TestYOLOTIDL:
    """YOLO TIDL offloading: build pipeline + hardware validation.

    Covers the n-variants (yolov5n, yolov8n) by default.  All tests require:
      - tidl_model_import_relax.so   (built from c7x-mma-tidl)
      - TI_CGT_C7000_PATH            (C7x cross-compiler)

    Hardware tests (test_yolo_tidl_correctness) additionally require:
      - AM67A board at hostname ``am67a`` with c7x_compute firmware running

    Only the 8 highest-FLOPs subgraphs are offloaded to TIDL (empirical limit
    limit).  Subgraphs that the TIDL optimizer cannot handle are automatically
    skipped via ``skip_failing_subgraphs=True`` and fall back to the TVM C7x
    scalar path.
    """

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not _has_import_so():
            pytest.fail(
                f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}"
            )
        if not _has_c7x_compiler():
            pytest.fail("TI_CGT_C7000_PATH not set")

    @pytest.mark.parametrize(
        "model_spec",
        YOLO_TIDL_MODELS,
        ids=[m[0] for m in YOLO_TIDL_MODELS],
    )
    def test_yolo_tidl_build(self, tmp_path, model_spec):
        """Build YOLO model with TIDL offloading (no hardware needed).

        Validates the full TIDL pipeline: prepare -> partition ->
        tidl_import -> lower -> c_static codegen -> bridge -> dynmod.
        Subgraphs that the TIDL optimizer rejects or that have multiple
        outputs are automatically skipped and compiled by TVM instead.
        """
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        model_name, version = model_spec
        _skip_yolo_if_no_ultralytics(model_name)

        mod, param_dict, _wrapped, _input_data = _create_yolo_model_unbound(
            model_name, version
        )

        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
                "skip_failing_subgraphs": True,
                "max_subgraphs": 8,
            }
        )
        result = compiler.build(
            mod,
            params=param_dict,
            build_dir=str(tmp_path / "build"),
        )

        assert result.module_path.exists(), (
            f"Build failed: {result.module_path}"
        )
        # At least some subgraphs should have been offloaded to TIDL
        assert len(result.artifacts) > 0, "No TIDL artifacts produced"

        size_mb = result.module_path.stat().st_size / (1024 * 1024)
        print(f"\nTIDL module: {result.module_path} ({size_mb:.1f} MB)")
        print(f"TIDL artifacts: {len(result.artifacts)} subgraph(s)")

        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(result.gen_dir), ignore_errors=True)
            shutil.rmtree(str(result.build_dir), ignore_errors=True)

    @pytest.mark.parametrize(
        "model_spec",
        YOLO_TIDL_MODELS,
        ids=[m[0] for m in YOLO_TIDL_MODELS],
    )
    def test_yolo_tidl_correctness(self, tmp_path, model_spec):
        """Deploy TIDL YOLO to AM67A and verify vs PyTorch.

        Requires AM67A hardware with c7x_compute firmware running.
        Uses cosine similarity (> 0.95) to account for int8 quantization
        error from partial TIDL offloading with 2 calibration frames.
        """
        from dsp_utils import run_dsp_dload

        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        model_name, version = model_spec
        _skip_yolo_if_no_ultralytics(model_name)

        mod, param_dict, wrapped, input_data = _create_yolo_model_unbound(
            model_name, version
        )

        # PyTorch reference
        with torch.no_grad():
            torch_out = wrapped(torch.from_numpy(input_data)).numpy()

        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
                "skip_failing_subgraphs": True,
                "max_subgraphs": 8,
            }
        )
        result = compiler.build(
            mod,
            params=param_dict,
            build_dir=str(tmp_path / "build"),
        )

        assert result.module_path.exists(), (
            f"Build failed: {result.module_path}"
        )

        try:
            output, stdout, cycles = run_dsp_dload(
                result.module_path,
                result.weights_path,
                [input_data],
                embedded_weights=True,
            )

            assert output is not None, "No output from DSP"

            flat_ref = torch_out.flatten()
            flat_dsp = output.flatten()
            cos_sim = float(
                np.dot(flat_ref, flat_dsp)
                / (
                    np.linalg.norm(flat_ref) * np.linalg.norm(flat_dsp)
                    + 1e-10
                )
            )
            print(
                f"\nTIDL {model_name}: output shape={output.shape}, "
                f"cos_sim={cos_sim:.6f}"
            )
            if cycles:
                print(
                    f"TIDL cycles: {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)"
                )
            if stdout:
                print(stdout)

            # Partial TIDL offloading with 2 calibration frames:
            # expect ~0.95-0.98 cosine similarity; 0.94 threshold allows for
            # int8 quantization variance across different calibration runs.
            assert cos_sim > 0.94, (
                f"TIDL vs PyTorch cos_sim {cos_sim:.6f} < 0.94 "
                f"for {model_name}"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(result.gen_dir), ignore_errors=True)
                shutil.rmtree(str(result.build_dir), ignore_errors=True)


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
        "--visualize",
        default=None,
        metavar="FILE",
        help="Generate interactive HTML TIDL partitioning visualization "
        "(defaults to yolov5n if --model not given; no hardware needed)",
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

    # --visualize: partition + HTML (no hardware, no .so needed for partition)
    if args.visualize:
        from tvm.relax.backend.tidl import TIDLOffloadCompiler
        from tvm.relax.backend.tidl.visualize import visualize_partitioning

        model_name = args.model or "yolov5n"
        version = next(v for n, v in YOLO_MODELS if n == model_name)

        if version == "v8":
            try:
                import ultralytics  # noqa: F401
            except ImportError:
                print(f"ERROR: {model_name} requires ultralytics package")
                return 1

        print(f"Partitioning {model_name} with TIDL...")
        mod, param_dict, _wrapped, _input_data = _create_yolo_model_unbound(
            model_name, version
        )
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": f"/tmp/tidl_viz_{model_name}",
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
            }
        )
        prepared = compiler.prepare(mod, param_dict)
        partitioned = compiler.partition(prepared)
        visualize_partitioning(
            partitioned,
            args.visualize,
            title=f"{model_name} TIDL Offloading",
        )
        print(f"Visualization: {args.visualize}")
        return 0

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
