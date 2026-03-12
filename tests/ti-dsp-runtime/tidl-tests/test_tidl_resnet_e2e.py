#!/usr/bin/env python
"""ResNet-18 TIDL end-to-end validation on AM67A with cycle comparison.

Builds ResNet-18 two ways and runs both on AM67A C7x hardware:
  1. TIDL-offloaded  -- conv/bn/relu subgraphs run on MMA via TIDL (int8)
  2. Pure c_static   -- entire model runs as float32 on C7x scalar pipeline

Both outputs are compared against PyTorch for correctness, and cycle
counts from the DSP's TSC counter are printed side-by-side.

Prerequisites (test is skipped without these):
  - tidl_model_import_relax.so  (Relax FFI bridge to TIDL import tool)
  - TI C7x cross-compiler       (TI_CGT_C7000_PATH)
  - torch + torchvision          (for model export)
  - AM67A board at hostname ``am67a`` with c7x_compute firmware running

Known limitation: the deploy+infer step may timeout on AM67A if the
TIDL runtime exceeds available DSP heap memory for large models.  The
build step alone validates the TIDL import/codegen/linking pipeline.

Set DSP_KEEP_TEMP=1 to preserve build artifacts for debugging.
"""

import os
import shutil
import sys

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DSP_CPP_DIR = os.path.join(_TESTS_DIR, "dsp-cpp")
sys.path.insert(0, _DSP_CPP_DIR)

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


def _has_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except ImportError:
        return False


def _create_resnet18():
    """Create ResNet-18, export to Relax, and return everything needed.

    Returns (mod, param_dict, torch_model, input_data) where:
      - mod is an unbound IRModule (params still as function args)
      - param_dict maps Var -> tvm.runtime.NDArray
      - torch_model is the eval-mode PyTorch model (same random weights)
      - input_data is a (1,3,224,224) float32 numpy array
    """
    import torch
    from torch.export import export as torch_export
    from torchvision.models.resnet import resnet18

    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    torch_model = resnet18(weights=None).eval()

    np.random.seed(42)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    with torch.no_grad():
        exported = torch_export(torch_model, example_args)
        mod = from_exported_program(exported, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    param_dict = dict(zip(mod["main"].params[1:], params["main"]))

    return mod, param_dict, torch_model, input_data


@pytest.mark.skipif(not _has_torch(), reason="torch/torchvision not installed")
@pytest.mark.skipif(
    not _has_import_so(),
    reason=f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}",
)
@pytest.mark.skipif(not _has_c7x_compiler(), reason="TI_CGT_C7000_PATH not set")
class TestTIDLResNetE2E:
    """ResNet-18 TIDL offloading: build pipeline + hardware validation."""

    def test_tidl_resnet18_build(self, tmp_path):
        """Build ResNet-18 with TIDL offloading (no hardware needed).

        Validates the full TIDL pipeline: prepare -> partition ->
        tidl_import -> lower -> c_static codegen -> bridge -> dynmod.
        """
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod, param_dict, _torch_model, _input_data = _create_resnet18()

        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
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
        for sg_name, paths in result.artifacts.items():
            print(f"  {sg_name}: {paths}")

        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(result.gen_dir), ignore_errors=True)
            shutil.rmtree(str(result.build_dir), ignore_errors=True)

    def test_tidl_resnet18_correctness(self, tmp_path):
        """Deploy TIDL ResNet-18 to AM67A and verify vs PyTorch.

        Requires AM67A hardware with c7x_compute firmware running.
        May timeout if TIDL runtime exceeds DSP heap for large models.
        """
        import torch
        from dsp_utils import run_dsp_dload

        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod, param_dict, torch_model, input_data = _create_resnet18()

        # PyTorch reference
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(input_data)).numpy()

        # Build with TIDL + layer profiling
        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
                "profile_layers": True,
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

        # Deploy and run on AM67A with profile mode (repeat=2):
        # iteration 1 includes TIDL init, iteration 2 is steady-state.
        try:
            output, stdout, cycles = run_dsp_dload(
                result.module_path,
                result.weights_path,
                [input_data],
                embedded_weights=True,
                profile=True,
            )

            assert output is not None, "No output from DSP"
            assert output.shape == (1, 1000), f"Unexpected shape: {output.shape}"

            max_diff = np.max(np.abs(output - torch_out))
            print(f"TIDL output: shape={output.shape}, max_diff_vs_pytorch={max_diff:.4f}")
            print(f"TIDL cycles (steady-state): {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)")
            if stdout:
                print(f"--- DSP profile output ({len(stdout)} chars) ---")
                print(stdout)
                print("--- end DSP profile output ---")

            # TIDL uses int8 internally so allow wider tolerance
            assert max_diff < 0.15, (
                f"TIDL vs PyTorch max diff {max_diff:.4f} exceeds threshold 0.15"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(result.gen_dir), ignore_errors=True)
                shutil.rmtree(str(result.build_dir), ignore_errors=True)

    def test_tidl_vs_tvm_cycles(self, tmp_path):
        """Compare TIDL-offloaded vs pure c_static cycle counts.

        Requires AM67A hardware with c7x_compute firmware running.
        """
        import torch
        from dsp_utils import compile_and_run_dsp, get_target_string, run_dsp_dload

        from tvm import relax
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod, param_dict, torch_model, input_data = _create_resnet18()

        # PyTorch reference
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(input_data)).numpy()

        # --- TIDL path ---
        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
            }
        )
        tidl_result = compiler.build(
            mod,
            params=param_dict,
            build_dir=str(tmp_path / "build_tidl"),
        )

        try:
            tidl_output, _stdout, tidl_cycles = run_dsp_dload(
                tidl_result.module_path,
                tidl_result.weights_path,
                [input_data],
                embedded_weights=True,
            )
        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(tidl_result.gen_dir), ignore_errors=True)
                shutil.rmtree(str(tidl_result.build_dir), ignore_errors=True)

        # --- Non-TIDL (pure c_static) path ---
        mod_bound = relax.transform.BindParams(
            func_name="main", params=param_dict
        )(mod)
        target_string = get_target_string("c7x_dload", use_cpp_api=True)
        tvm_results = compile_and_run_dsp(
            mod=mod_bound,
            input_data=input_data,
            target_string=target_string,
            execution_mode="c7x_dload",
        )
        tvm_output = tvm_results["c7x_dload_result"]
        tvm_cycles = tvm_results.get("c7x_dload_cycles", 0)

        # --- Correctness checks ---
        tidl_diff = np.max(np.abs(tidl_output - torch_out))
        tvm_diff = np.max(np.abs(tvm_output - torch_out))

        print("\n" + "=" * 60)
        print("ResNet-18 TIDL vs Pure-TVM Comparison")
        print("=" * 60)
        print(f"{'':20s} {'TIDL':>15s} {'Pure TVM':>15s}")
        print(f"{'-'*20} {'-'*15} {'-'*15}")
        print(f"{'Max diff vs PyTorch':20s} {tidl_diff:15.4f} {tvm_diff:15.4f}")
        print(f"{'Cycles':20s} {tidl_cycles:15,} {tvm_cycles:15,}")
        if tidl_cycles > 0:
            print(f"{'Time @ 1GHz (ms)':20s} {tidl_cycles/1e6:15.2f} {tvm_cycles/1e6:15.2f}")
        if tidl_cycles > 0 and tvm_cycles > 0:
            speedup = tvm_cycles / tidl_cycles
            print(f"{'Speedup (TIDL)':20s} {speedup:15.1f}x")
        print("=" * 60)

        assert tidl_diff < 0.15, (
            f"TIDL vs PyTorch max diff {tidl_diff:.4f} exceeds 0.15"
        )
        assert tvm_diff < 0.05, (
            f"TVM vs PyTorch max diff {tvm_diff:.4f} exceeds 0.05"
        )

        # TIDL (MMA int8) should be faster than scalar float32
        if tidl_cycles > 0 and tvm_cycles > 0:
            assert tidl_cycles < tvm_cycles, (
                f"Expected TIDL ({tidl_cycles:,}) to be faster than "
                f"pure TVM ({tvm_cycles:,})"
            )


def main():
    """Standalone: build TIDL-partitioned ResNet-18 and visualize."""
    import argparse

    parser = argparse.ArgumentParser(description="ResNet-18 TIDL E2E")
    parser.add_argument(
        "--visualize",
        default=None,
        metavar="FILE",
        help="Generate interactive HTML visualization",
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        metavar="FILE",
        help="JSON file with layer profile data (from parse_layer_profile)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run pytest instead of standalone mode",
    )
    args = parser.parse_args()

    if args.test or args.visualize is None:
        pytest.main([__file__, "-v", "-s"])
        return

    import json

    from tvm.relax.backend.tidl import TIDLOffloadCompiler
    from tvm.relax.backend.tidl.visualize import visualize_partitioning

    mod, param_dict, _torch_model, _input_data = _create_resnet18()

    print("Partitioning ResNet-18 with TIDL...")
    compiler = TIDLOffloadCompiler(
        config={
            "artifacts_dir": "/tmp/tidl_viz_artifacts",
            "tidl_tools_path": TIDL_TOOLS_PATH,
            "tidl_relax_so_path": RELAX_SO_PATH,
            "num_calibration_frames": 2,
        }
    )
    prepared = compiler.prepare(mod, param_dict)
    partitioned = compiler.partition(prepared)

    profile = None
    if args.profile_json:
        with open(args.profile_json) as f:
            profile = json.load(f)
        print(f"Loaded {len(profile)} layer profiles from {args.profile_json}")

    visualize_partitioning(
        partitioned,
        args.visualize,
        title="ResNet-18 TIDL Offloading",
        profile_data=profile,
    )
    print(f"Visualization: {args.visualize}")


if __name__ == "__main__":
    main()
