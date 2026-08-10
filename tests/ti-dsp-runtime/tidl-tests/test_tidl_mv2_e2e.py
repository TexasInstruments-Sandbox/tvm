#!/usr/bin/env python
"""MobileNetV2 TIDL end-to-end validation — calibration infrastructure check.

Purpose: comparison test against ResNet-18's Bug 4 (calibration stats 400K×
too large at layer4[1]'s second conv).

MobileNetV2 has only depthwise/pointwise convolutions with BN-folded weight
magnitudes (w_max ≤ 0.3, vs ResNet-18's layer4[1] at 3.648).  If MV2
calibrates correctly (stats binary max < 1000), the TIDL Relax calibration
infrastructure is sound and the ResNet-18 overflow is specific to the
unusually large BN-folded weights at layer4[1].  If MV2 also shows inflated
stats, a deeper bug exists in tidl_relaxImport.cpp or the net.bin write path.

Prerequisites (test is skipped without these):
  - tidl_model_import_relax.so  (Relax FFI bridge to TIDL import tool)
  - TI C7x cross-compiler       (TI_CGT_C7000_PATH)
  - torch + torchvision          (for model export)
  - AM67A board at hostname ``am67a`` with c7x_compute firmware running
    (correctness test only; build test has no board requirement)

Set DSP_KEEP_TEMP=1 to preserve build artifacts for debugging.
"""

import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DSP_CPP_DIR = os.path.join(_TESTS_DIR, "dsp-cpp")
_TEST_IMAGES_DIR = Path(_TESTS_DIR).parent / "cstatic" / "test_images"
sys.path.insert(0, _DSP_CPP_DIR)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH",
    os.path.expanduser("~/ml/c7x-mma-tidl"),
)
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH,
    "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so",
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")

# Expected calibration stats ceiling: if max exceeds this the tool is broken.
_STATS_MAX_THRESHOLD = 1000.0


# Pinned explicitly (not globbed) so unrelated additions to the shared
# test_images/ directory can't silently shift TIDL's INT8 calibration
# statistics for this classification test.
_CALIBRATION_IMAGE_NAMES = ("YellowLabradorLooking_new.jpg", "bird_0.jpg", "dog.jpg")


def _load_calibration_images(size: int = 224):
    """Load pinned test images with ImageNet normalization for TIDL INT8 calibration."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None

    images = []
    for name in _CALIBRATION_IMAGE_NAMES:
        img = Image.open(_TEST_IMAGES_DIR / name).convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        images.append(arr[None])
    return images if images else None


_CALIB_INPUTS = _load_calibration_images()


def _has_import_so():
    return os.path.isfile(RELAX_SO_PATH)


def _has_c7x_compiler():
    return os.environ.get("TI_CGT_C7000_PATH") is not None


def _has_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


def _create_mv2():
    """Export MobileNetV2 (pretrained) to Relax.

    Returns (mod, param_dict, torch_model, input_data).
    """
    import torch
    from torch.export import export as torch_export
    from torchvision.models import mobilenet_v2
    from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    torch_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()

    if _CALIB_INPUTS is not None:
        input_data = _CALIB_INPUTS[0]
    else:
        np.random.seed(42)
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    with torch.no_grad():
        exported = torch_export(torch_model, example_args)
        mod = from_exported_program(exported, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    param_dict = dict(zip(mod["main"].params[1:], params["main"]))

    return mod, param_dict, torch_model, input_data


def _check_calibration_stats(artifacts_dir: str) -> dict:
    """Read stats binaries and return per-file summary dicts.

    Raises AssertionError if any stats file has max > _STATS_MAX_THRESHOLD.
    """
    stats_files = sorted(glob.glob(os.path.join(artifacts_dir, "*_stats_tool_out.bin")))
    summaries = {}
    for f in stats_files:
        d = np.fromfile(f, dtype=np.float32)
        summary = {
            "n_floats": len(d),
            "max": float(d.max()),
            "mean": float(d.mean()),
        }
        summaries[os.path.basename(f)] = summary
        print(
            f"  Stats {os.path.basename(f)}: {len(d)} floats, "
            f"max={d.max():.2f}, mean={d.mean():.2f}"
        )
        assert d.max() < _STATS_MAX_THRESHOLD, (
            f"Calibration stats look wrong: {os.path.basename(f)} "
            f"max={d.max():.0f} (expected < {_STATS_MAX_THRESHOLD}). "
            "The TIDL Relax calibration tool may have the same gParams or "
            "net.bin write bug as seen in ResNet-18 layer4[1]."
        )
    return summaries


class TestTIDLMV2E2E:
    """MobileNetV2 TIDL offloading: calibration health check + hardware validation."""

    @pytest.fixture(autouse=True)
    def _check_deps(self, dsp_mode):  # noqa: ARG002
        if not _has_torch():
            pytest.fail("torch/torchvision not installed")
        if not _has_import_so():
            pytest.fail(f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}")
        if not _has_c7x_compiler():
            pytest.fail("TI_CGT_C7000_PATH not set")

    def test_tidl_mv2_build(self, tmp_path):
        """Build MV2 with TIDL offloading and validate calibration stats.

        No hardware needed.  After build, reads the stats binary produced by
        PC_dsp_test_dl_algo.out and asserts per-channel values are < 1000.
        If they are millions here (as in ResNet-18), the bug is not model-specific.
        """
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod, param_dict, _torch_model, _input_data = _create_mv2()

        calib_inputs = _CALIB_INPUTS or [
            np.random.rand(1, 3, 224, 224).astype("float32"),
            np.random.rand(1, 3, 224, 224).astype("float32"),
        ]
        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "num_calibration_frames": len(calib_inputs),
                "calibration_inputs": calib_inputs,
            }
        )
        result = compiler.build(
            mod,
            params=param_dict,
            build_dir=str(tmp_path / "build"),
        )

        assert result.module_path.exists(), f"Build failed: {result.module_path}"
        assert len(result.artifacts) > 0, "No TIDL artifacts produced"

        size_mb = result.module_path.stat().st_size / (1024 * 1024)
        print(f"\nTIDL module: {result.module_path} ({size_mb:.1f} MB)")
        print(f"TIDL artifacts: {len(result.artifacts)} subgraph(s)")
        print(f"Calib images: {len(calib_inputs)} ({'real' if _CALIB_INPUTS else 'random'})")

        print("\nCalibration stats check:")
        stats = _check_calibration_stats(artifacts_dir)
        if not stats:
            print("  (no stats files found — calibration tool may not have run)")

        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(result.gen_dir), ignore_errors=True)
            shutil.rmtree(str(result.build_dir), ignore_errors=True)

    def test_tidl_mv2_correctness(self, tmp_path, dsp_mode):
        """Verify TIDL MV2 output matches PyTorch within INT8 tolerance.

        If this passes but test_tidl_resnet18_correctness fails, the calibration
        infrastructure is correct and the ResNet-18 failure is caused by
        layer4[1]'s unusually large BN-folded weights (w_max=3.648).
        """
        if dsp_mode not in ("c7x_host", "c7x_dload"):
            pytest.skip("requires --dsp-mode=c7x_host or c7x_dload")
        if dsp_mode == "c7x_host":
            from conftest import has_tidl_pc_libs  # noqa: PLC0415

            if not has_tidl_pc_libs():
                pytest.skip("PC TIDL algo libs not found (required for c7x_host TIDL bridge)")
        import torch

        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod, param_dict, torch_model, input_data = _create_mv2()

        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(input_data)).numpy()

        calib_inputs = _CALIB_INPUTS or [
            np.random.rand(1, 3, 224, 224).astype("float32"),
            np.random.rand(1, 3, 224, 224).astype("float32"),
        ]
        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "num_calibration_frames": len(calib_inputs),
                "calibration_inputs": calib_inputs,
                "profile_layers": True,
            }
        )
        result = compiler.build(
            mod,
            params=param_dict,
            exec_mode=dsp_mode,
            build_dir=str(tmp_path / "build"),
        )

        assert result.module_path.exists(), f"Build failed: {result.module_path}"
        assert len(result.artifacts) > 0, "No TIDL artifacts produced"

        size_mb = result.module_path.stat().st_size / (1024 * 1024)
        print(f"\nTIDL module: {result.module_path} ({size_mb:.1f} MB)")
        print(f"TIDL artifacts: {len(result.artifacts)} subgraph(s)")
        print(f"Calib images: {len(calib_inputs)} ({'real' if _CALIB_INPUTS else 'random'})")

        print("\nCalibration stats check:")
        _check_calibration_stats(artifacts_dir)

        try:
            if dsp_mode == "c7x_host":
                from dsp_utils import (  # noqa: PLC0415
                    INPUT_BIN_FILE,
                    run_dsp_host,
                    write_tensors_to_file,
                )

                write_tensors_to_file([input_data], str(result.build_dir / INPUT_BIN_FILE))
                output = run_dsp_host(result.module_path)
                cycles = None
            else:
                from dsp_utils import run_dsp_dload  # noqa: PLC0415

                output, stdout, cycles = run_dsp_dload(
                    result.module_path,
                    result.weights_path,
                    [input_data],
                    embedded_weights=True,
                    profile=True,
                )
                if stdout:
                    print(f"--- DSP profile output ({len(stdout)} chars) ---")
                    print(stdout)
                    print("--- end DSP profile output ---")

            assert output is not None, "No output from DSP"
            assert output.shape == (1, 1000), f"Unexpected shape: {output.shape}"
            assert not np.any(np.isnan(output)), "TIDL output contains NaN"
            assert not np.any(np.isinf(output)), "TIDL output contains Inf"

            diff = np.abs(output - torch_out)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            tidl_top1 = int(np.argmax(output))
            torch_top1 = int(np.argmax(torch_out))
            tidl_top5 = np.argsort(output[0])[-5:][::-1].tolist()
            torch_top5 = np.argsort(torch_out[0])[-5:][::-1].tolist()

            print(f"TIDL output: shape={output.shape}")
            print(f"  max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}")
            print(f"  top-1: TIDL={tidl_top1}  PyTorch={torch_top1}  match={tidl_top1==torch_top1}")
            print(f"  top-5: TIDL={tidl_top5}")
            print(f"         PyTorch={torch_top5}  match={tidl_top5==torch_top5}")
            if cycles:
                print(f"TIDL cycles (steady-state): {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)")

            # When FC runs on TVM scalar (float32), mean_diff is ~0.11 (conv INT8 only).
            # When FC is fully offloaded to TIDL INT8 (the current path), INT8 quantization
            # of 1280 input features accumulates more error; mean_diff ~0.4 is expected.
            # Top-1 accuracy is the primary check; mean_diff < 0.5 catches catastrophic
            # failures (e.g. the original Bug 3 where diff was ~10 million).
            assert mean_diff < 0.5, (
                f"TIDL vs PyTorch mean diff {mean_diff:.4f} exceeds 0.5"
            )
            assert tidl_top1 == torch_top1, (
                f"Top-1 mismatch: TIDL={tidl_top1} PyTorch={torch_top1}"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(result.gen_dir), ignore_errors=True)
                shutil.rmtree(str(result.build_dir), ignore_errors=True)


def main():
    """Standalone: build TIDL-partitioned MobileNetV2."""
    import argparse

    parser = argparse.ArgumentParser(description="MobileNetV2 TIDL E2E")
    parser.add_argument("--test", action="store_true", help="Run pytest instead")
    args = parser.parse_args()

    if args.test:
        pytest.main([__file__, "-v", "-s"])
        return

    from tvm.relax.backend.tidl import TIDLOffloadCompiler

    mod, param_dict, _torch_model, _input_data = _create_mv2()

    calib = _CALIB_INPUTS or [np.random.rand(1, 3, 224, 224).astype("float32")]
    print(f"Calibration: {len(calib)} frame(s) ({'real' if _CALIB_INPUTS else 'random'})")

    artifacts_dir = "/tmp/tidl_mv2_artifacts"
    compiler = TIDLOffloadCompiler(
        config={
            "artifacts_dir": artifacts_dir,
            "tidl_tools_path": TIDL_TOOLS_PATH,
            "num_calibration_frames": len(calib),
            "calibration_inputs": calib,
        }
    )
    result = compiler.build(mod, params=param_dict, build_dir="/tmp/tidl_mv2_build")

    print(f"Module: {result.module_path}")
    print(f"Subgraphs: {len(result.artifacts)}")
    print("\nCalibration stats:")
    _check_calibration_stats(artifacts_dir)


if __name__ == "__main__":
    main()
