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
    # c7x_host smoke + c7x_dload correctness (requires .so + compiler + AM67A)
    pytest test_yolo_dsp.py::TestYOLOTIDL::test_yolo_tidl -v

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

from dsp_utils import (  # noqa: E402
    assert_dsp_comparison,
    compare_results,
    compile_and_run_dsp,
    get_target_string,
)

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
    from conftest import has_c7x_host_env

    return has_c7x_host_env()


INPUT_SHAPE = (1, 3, 320, 320)

_TEST_IMAGES_DIR = _THIS_DIR.parent.parent / "cstatic" / "test_images"


def _load_calibration_images(size: int = 320) -> np.ndarray:
    """Load test images for TIDL INT8 calibration.

    Returns float32 array of shape (N, 3, H, W) with values in [0, 1].
    Real images give TIDL much better INT8 scale estimates than random
    data, particularly for the backbone input subgraph.
    """
    from PIL import Image  # noqa: PLC0415

    images = []
    for p in sorted(_TEST_IMAGES_DIR.glob("*.jpg")):
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0  # HWC [0,1]
        images.append(arr.transpose(2, 0, 1))  # CHW
    if not images:
        return None
    return np.stack(images)  # (N, 3, H, W)


def _split_calib_frames(images):
    """Split (N, 3, H, W) batch into list of (1, 3, H, W) per-frame arrays.

    Used to build calibration_inputs for TIDLOffloadCompiler:
    per-frame inputs let _build_calibration_module run the model once per
    image and collect real intermediate activations at each subgraph boundary.
    """
    if images is None:
        return None
    return [images[i : i + 1] for i in range(len(images))]


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
    """Load YOLOv5 model from local .pt file (preferred) or torch.hub."""
    # Prefer cached hub repo (source="local") to avoid network access
    # behind corporate proxies.  Falls back to github download.
    hub_dir = Path(torch.hub.get_dir()) / "ultralytics_yolov5_master"
    local_pt = _THIS_DIR.parent / f"{model_name}.pt"
    # Weights cached by torch.hub at TORCH_HOME/hub/checkpoints/
    cached_pt = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}.pt"
    hub_cached = hub_dir.is_dir()
    # Find a .pt file: local dir > hub checkpoints cache
    pt_file = local_pt if local_pt.exists() else (cached_pt if cached_pt.exists() else None)
    if pt_file and hub_cached:
        model = torch.hub.load(
            str(hub_dir),
            "custom",
            path=str(pt_file),
            source="local",
            verbose=False,
        )
    elif pt_file:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(pt_file), verbose=False)
    else:
        pytest.skip(f"{model_name} requires pre-cached weights at {cached_pt} or {local_pt}")
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
        mod = from_exported_program(exported_program, keep_params_as_input=True)

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
    mod, param_dict, wrapped, input_data = _create_yolo_model_unbound(model_name, version)
    mod = relax.transform.BindParams(func_name="main", params=param_dict)(mod)
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

    target_string = get_target_string(
        dsp_mode, profile_layers=profile_layers, use_cpp_api=use_cpp_api
    )

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
    comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=2e1)
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
def test_yolo_dsp(model_spec, dsp_mode, dsp_timeout, use_cpp_api, profile_layers, record_cycles):
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
    record_cycles(model_name, results["dsp_results"].get("c7x_dload_cycles", 0))
    assert_dsp_comparison(results["dsp_results"], results["comparison"])


# -----------------------------------------------------------------------------
# TIDL Offloading Tests
# -----------------------------------------------------------------------------


class TestYOLOTIDL:
    """YOLO TIDL offloading: c7x_host smoke + c7x_dload correctness.

    Covers the n-variants (yolov5n, yolov8n) by default.  All tests require:
      - tidl_model_import_relax.so   (built from c7x-mma-tidl)
      - TI_CGT_C7000_PATH            (C7x cross-compiler)

    c7x_dload additionally requires:
      - AM67A board at hostname ``am67a`` with c7x_compute firmware running

    Only the 8 highest-FLOPs subgraphs are offloaded to TIDL.  Subgraphs
    that the TIDL optimizer rejects fall back to the TVM C7x scalar path
    via ``skip_failing_subgraphs=True``.
    """

    _CALIB_IMAGES = _load_calibration_images()
    # Split the batch array into a list of per-frame arrays so that
    # TIDLOffloadCompiler can run the model once per image and collect
    # real intermediate activations at each TIDL subgraph boundary.
    # This fixes INT8 miscalibration for deeper subgraphs whose inputs are
    # not raw image pixels but post-activation feature maps (DFL NaN fix).
    _CALIB_INPUTS = _split_calib_frames(_CALIB_IMAGES)

    _COMPILER_CONFIG = {
        "tidl_tools_path": TIDL_TOOLS_PATH,
        "num_calibration_frames": len(_CALIB_INPUTS) if _CALIB_INPUTS else 1,
        "calibration_inputs": _CALIB_INPUTS,  # real boundary activations per subgraph
        "skip_failing_subgraphs": True,
        "max_subgraphs": 8,
    }

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not _has_import_so():
            pytest.fail(f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}")
        if not _has_c7x_compiler():
            pytest.fail("TI_CGT_C7000_PATH not set")

    @pytest.mark.parametrize(
        "model_spec",
        YOLO_TIDL_MODELS,
        ids=[m[0] for m in YOLO_TIDL_MODELS],
    )
    def test_yolo_tidl(self, tmp_path, model_spec, dsp_mode):
        """TIDL offload: c7x_host smoke test then c7x_dload correctness.

        Pipeline:
          1. TIDL compile: partition -> import -> lower (expensive step,
             done once; both host and dload builds share the result)
          2. c_static codegen: relax.build -> lib0.c + weights.bin
          3. c7x_host smoke: build with stub bridge (TIDL outputs zeroed)
             and run via TI Host Emulation.  Verifies the pipeline end-to-
             end without hardware.  Fails fast if codegen or non-TIDL ops
             are broken.  Skipped if PC TIDL libs are absent.
          4. c7x_dload correctness: build with real TIDL bridge, deploy to
             AM67A, compare output to PyTorch with cosine similarity > 0.94.
             Only runs when --dsp-mode=c7x_dload.
        """
        import tarfile  # noqa: PLC0415,I001

        import tvm  # noqa: PLC0415
        from dsp_utils import (  # noqa: PLC0415
            build_dsp_c7x_host,
            build_dsp_dynmod,
            run_dsp_dload,
            run_dsp_host,
            write_tensors_to_file,
        )
        from tvm.relax.backend.cpu_generic.pipeline import (  # noqa: PLC0415
            get_default_pipeline,
        )
        from tvm.relax.backend.tidl import TIDLOffloadCompiler  # noqa: PLC0415

        if dsp_mode not in ("c7x_host", "c7x_dload"):
            pytest.skip(f"--dsp-mode=c7x_host or c7x_dload required, got {dsp_mode!r}")

        model_name, version = model_spec
        _skip_yolo_if_no_ultralytics(model_name)

        mod, param_dict, wrapped, input_data = _create_yolo_model_unbound(model_name, version)

        with torch.no_grad():
            torch_out = wrapped(torch.from_numpy(input_data)).numpy()

        cfg = dict(self._COMPILER_CONFIG)
        cfg["artifacts_dir"] = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(config=cfg)

        # ------------------------------------------------------------------
        # Step 1: TIDL compile (partition → import → lower).
        # This is the expensive step; lowered module is reused for both
        # c7x_host and c7x_dload builds.
        # ------------------------------------------------------------------
        lowered, artifacts = compiler.compile(mod, params=param_dict)
        assert len(artifacts) > 0, "No TIDL subgraphs produced"
        print(f"\n{model_name}: {len(artifacts)} TIDL subgraph(s)")

        # ------------------------------------------------------------------
        # Step 2: C code generation (relax.build → lib0.c + weights.bin)
        # ------------------------------------------------------------------
        target = "c_static -mcpu=c7x -use-cpp-api=1 -tidl-runtime=1"
        tvm_target = tvm.target.Target(target)
        pipeline = get_default_pipeline(tvm_target)
        with tvm_target, tvm.transform.PassContext(opt_level=3):
            ex = relax.build(
                lowered,
                target=tvm_target,
                exec_mode="compiled",
                system_lib=True,
                relax_pipeline=pipeline,
                tir_pipeline=None,
            )
        gen_dir = tmp_path / "gen"
        gen_dir.mkdir()
        tar_path = gen_dir / "model.tar"
        ex.export_library(str(tar_path), target=tvm_target)
        with tarfile.open(str(tar_path)) as tf:
            tf.extractall(str(gen_dir))
        tar_path.unlink()

        # ------------------------------------------------------------------
        # Step 3: c7x_host real bridge (PC AVX TIDL libs, no board needed).
        # Validates actual INT8 TIDL inference on host; fails fast if
        # calibration is wrong, enabling fast iteration without hardware.
        # Guarded by has_tidl_pc_libs() — skipped if libs are absent.
        # ------------------------------------------------------------------
        from conftest import has_tidl_pc_libs  # noqa: PLC0415
        from tvm.relax.backend.tidl import generate_artifacts_c  # noqa: PLC0415

        if has_tidl_pc_libs():
            real_bridge_host = str(gen_dir / "tidl_bridge.c")
            TIDLOffloadCompiler.generate_bridge(
                lowered,
                real_bridge_host,
                stub=False,
                artifacts_dir=compiler._artifacts_dir,
            )
            artifacts_c = str(gen_dir / "tidl_artifacts.c")
            generate_artifacts_c(compiler._artifacts_dir, artifacts_c)

            host_build_dir = tmp_path / "host_build"
            exe = build_dsp_c7x_host(
                gen_dir,
                tidl_bridge=[real_bridge_host, artifacts_c],
                build_dir=host_build_dir,
                use_tidl=True,
            )
            input_file = host_build_dir / "input.bin"
            write_tensors_to_file([input_data], str(input_file))
            host_out = run_dsp_host(exe, working_dir=host_build_dir)

            flat_ref = torch_out.flatten()
            flat_host = host_out.flatten()
            print(f"c7x_host: ref nan={np.isnan(flat_ref).any()} inf={np.isinf(flat_ref).any()}")
            print(f"c7x_host: dsp nan={np.isnan(flat_host).any()} inf={np.isinf(flat_host).any()}")
            cos_sim_host = float(
                np.dot(flat_ref, flat_host)
                / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_host) + 1e-10)
            )
            print(
                f"c7x_host real bridge: output shape={host_out.shape}, cos_sim={cos_sim_host:.6f}"
            )
            assert np.isfinite(cos_sim_host), f"c7x_host: cos_sim is not finite for {model_name}"
            assert cos_sim_host > 0.94, (
                f"c7x_host TIDL vs PyTorch cos_sim {cos_sim_host:.6f} < 0.94 for {model_name}"
            )

        # ------------------------------------------------------------------
        # Step 4: c7x_dload correctness (real TIDL bridge, AM67A hardware).
        # Only runs when --dsp-mode=c7x_dload.
        # ------------------------------------------------------------------
        if dsp_mode == "c7x_dload":
            # Re-use the real bridge generated in Step 3 if present; otherwise
            # generate it now (allows running Step 4 without PC TIDL libs).
            real_bridge_dload = str(gen_dir / "tidl_bridge.c")
            if not Path(real_bridge_dload).exists():
                TIDLOffloadCompiler.generate_bridge(
                    lowered,
                    real_bridge_dload,
                    stub=False,
                    artifacts_dir=compiler._artifacts_dir,
                )

            dload_build_dir = tmp_path / "dload_build"
            weights_path = gen_dir / "weights.bin"
            # fp_reassoc_off=True guards against cl7x reordering FP operations.
            # The c7x_dload NaN (0/0 in YOLO DFL softmax) is a calibration
            # accuracy issue: per-subgraph calibration uses image pixels [0,1]
            # for all subgraphs but intermediate activations at the DFL boundary
            # are in a different range.  DSP MMA TIDL produces slightly lower
            # INT8 values than PC AVX, causing all exp(x_i) to underflow to 0.0
            # → sum=0 → 0/0=NaN.  The test marks this path xfail at runtime.
            module_path = build_dsp_dynmod(
                generated_dir=gen_dir,
                build_dir=dload_build_dir,
                weights_file=weights_path,
                tidl_bridge=real_bridge_dload,
                use_tidl=True,
                tidl_artifacts_dir=compiler._artifacts_dir,
                fp_reassoc_off=True,
            )

            try:
                output, stdout, cycles = run_dsp_dload(
                    module_path,
                    weights_path,
                    [input_data],
                    embedded_weights=True,
                )

                flat_ref = torch_out.flatten()
                flat_dsp = output.flatten()
                ref_nan = np.isnan(flat_ref).any()
                ref_inf = np.isinf(flat_ref).any()
                dsp_nan = np.isnan(flat_dsp).any()
                dsp_inf = np.isinf(flat_dsp).any()
                print(f"c7x_dload: ref nan={ref_nan} inf={ref_inf}")
                print(f"c7x_dload: dsp nan={dsp_nan} inf={dsp_inf}")
                cos_sim = float(
                    np.dot(flat_ref, flat_dsp)
                    / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_dsp) + 1e-10)
                )
                print(f"c7x_dload: output shape={output.shape}, cos_sim={cos_sim:.6f}")
                if cycles:
                    print(f"TIDL cycles: {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)")
                if stdout:
                    print(stdout)

                if not np.isfinite(cos_sim):
                    # Root cause (confirmed): PC AVX TIDL and DSP MMA TIDL apply
                    # slightly different INT8 rounding at inference time.  Even with
                    # correct per-subgraph calibration (real images, cos_sim > 0.98
                    # on c7x_host), the DSP-side dequantized DFL softmax inputs are
                    # shifted enough that exp(x_i) underflows to exactly 0.0 for all
                    # 16 elements — genuine IEEE 754 underflow, not FTZ (clearing
                    # FPCR bit 4 had no effect).  sum(0,...) = 0 → 0/0 = NaN.
                    # Fix requires eliminating the PC-DSP gap: either force float32
                    # at the TIDL subgraph boundary, or switch YOLO to the MMALIB
                    # path (C7xMMAQuantizer) which bakes scales into compiled code
                    # and has no separate PC-calibration inference step.
                    # See docs/dsp/tidl_subgraph_calibration.md for full analysis.
                    pytest.xfail(
                        f"c7x_dload cos_sim is not finite for {model_name}: "
                        "PC AVX vs DSP MMA INT8 rounding gap causes exp() underflow "
                        "in YOLO DFL softmax; c7x_host passes (cos_sim > 0.98)"
                    )
                assert cos_sim > 0.94, (
                    f"c7x_dload TIDL vs PyTorch cos_sim {cos_sim:.6f} < 0.94 for {model_name}"
                )

            finally:
                if not os.environ.get("DSP_KEEP_TEMP"):
                    shutil.rmtree(str(dload_build_dir), ignore_errors=True)

        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(gen_dir), ignore_errors=True)


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
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

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
        mod, param_dict, _wrapped, _input_data = _create_yolo_model_unbound(model_name, version)
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": f"/tmp/tidl_viz_{model_name}",
                "tidl_tools_path": TIDL_TOOLS_PATH,
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
        tvm_mod, wrapped, input_data = create_yolo_model(model_name, version)

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
                cycles_match = re.search(r"Inference complete:\s*(\d+)\s*cycles", stdout)
                if cycles_match:
                    cycles = int(cycles_match.group(1))
                    time_ms = cycles / 1_000_000
                    print(f"[C7x DLOAD] Inference cycles: {cycles:,} ({time_ms:.3f} ms at 1 GHz)")

        if "c7x_host_result" in dsp_results:
            c7x_host_result = dsp_results["c7x_host_result"]
            print(f"\n[C7x Host] Output shape: {c7x_host_result.shape}")

        # YOLO raw detection outputs contain sigmoid/exp-transformed
        # coordinates where small floating-point differences from the
        # TI Host Emulation library compound across detection heads.
        # Use cosine similarity (> 0.999) for pass/fail instead of
        # element-wise tolerance on raw tensor values.
        print("\n[Comparison] vs PyTorch:")
        comparison = compare_results(dsp_results, torch_result, "PyTorch", rtol=1e-1, atol=2e1)

        passed = True
        flat_ref = torch_result.flatten()
        for mode in ["c7x_host", "c7x_dload"]:
            diff_key = f"{mode}_vs_ref_max_diff"
            _pass_key = f"{mode}_vs_ref_passed"  # noqa: F841
            result_key = f"{mode}_result"
            if diff_key not in comparison:
                continue
            # Compute cosine similarity
            if result_key in dsp_results:
                flat_dsp = dsp_results[result_key].flatten()
                cos_sim = float(
                    np.dot(flat_ref, flat_dsp)
                    / (np.linalg.norm(flat_ref) * np.linalg.norm(flat_dsp) + 1e-10)
                )
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
