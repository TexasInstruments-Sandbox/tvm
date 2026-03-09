#!/usr/bin/env python
"""End-to-end TIDL hardware test on AM67A (c7x_dload).

Tests the full pipeline:
  partition -> tidl_import() -> lower -> codegen -> bridge -> build -> deploy -> run

Uses tidl_import() (Relax FFI) to generate TIDL artifacts on-the-fly,
then compiles, deploys, and runs on AM67A hardware.

Requires:
  - tidl_model_import_relax.so (from c7x-mma-tidl)
  - TI C7x cross-compiler (TI_CGT_C7000_PATH)
  - PSDK with TIDL+MMALIB libraries
  - AM67A board with c7x_compute firmware
"""

import os
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.tidl import TIDLOffloadCompiler
from tvm.relax.frontend import nn

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


class ConvReluSoftmaxModel(nn.Module):
    """Mixed model: conv+relu to TIDL, softmax stays in TVM."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 16, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.softmax(x, axis=1)
        return x


def _compile_c_static(mod, target_str="c_static -mcpu=c7x"):
    from pathlib import Path

    target = tvm.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=0):
        ex = relax.build(mod, target=target, exec_mode="compiled", system_lib=True)
    td = tempfile.mkdtemp(prefix="tidl_e2e_")
    tar_path = os.path.join(td, "model.tar")
    ex.export_library(tar_path, target=target)
    with tarfile.open(tar_path) as tf:
        tf.extractall(td)
    os.remove(tar_path)
    return Path(td)


@pytest.mark.skipif(
    not _has_import_so(),
    reason=f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}",
)
@pytest.mark.skipif(
    not _has_c7x_compiler(),
    reason="TI_CGT_C7000_PATH not set",
)
class TestTIDLImportE2E:
    """End-to-end: tidl_import() -> build -> run on AM67A."""

    def test_import_build_run(self, tmp_path):
        """Full pipeline: partition -> import -> lower -> build -> run."""
        from pathlib import Path

        from dsp_utils import build_dsp_dynmod, run_dsp_dload

        # 1. Export and bind params
        model = ConvReluSoftmaxModel()
        mod, param_spec = model.export_tvm(
            spec={"main": {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")}}
        )
        np.random.seed(42)
        params = [
            tvm.runtime.tensor(
                np.random.rand(*p.shape).astype("float32"),
                device=tvm.cpu(),
            )
            for _, p in param_spec
        ]
        param_dict = dict(zip(mod["main"].params[1:], params))

        # 2. Prepare + partition + import
        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
            }
        )
        mod = compiler.prepare(mod, param_dict)
        mod = compiler.partition(mod)

        # Verify we have TIDL subgraphs
        tidl_funcs = [
            name
            for name, f in mod.functions.items()
            if hasattr(f, "attrs") and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        assert len(tidl_funcs) > 0, "No TIDL subgraphs found after partition"

        mod, artifacts = compiler.tidl_import(mod)
        assert len(artifacts) > 0, "tidl_import() returned no artifacts"

        # Verify artifact files exist
        for sg_name, paths in artifacts.items():
            assert os.path.isfile(paths["net_bin"]), (
                f"Missing net_bin for {sg_name}: {paths['net_bin']}"
            )
            assert os.path.isfile(paths["io_bin"]), (
                f"Missing io_bin for {sg_name}: {paths['io_bin']}"
            )
            net_size = os.path.getsize(paths["net_bin"])
            io_size = os.path.getsize(paths["io_bin"])
            print(f"  {sg_name}: net={net_size / 1024:.0f}KB, io={io_size / 1024:.0f}KB")

        # 3. Lower TIDL subgraphs to TIR extern calls
        lowered = compiler.lower_tidl(mod, artifacts)

        # 4. Compile to C static
        gen_dir = _compile_c_static(lowered)

        build_dir = None
        try:
            # 5. Generate real bridge (TIDL API with embedded artifacts)
            bridge_path = str(gen_dir / "tidl_bridge.c")
            TIDLOffloadCompiler.generate_bridge(lowered, bridge_path, stub=False)

            # 6. Build c7x-dynmod with TIDL artifacts from import
            build_dir = Path(tempfile.mkdtemp(prefix="tidl_e2e_build_"))
            module_path = build_dsp_dynmod(
                generated_dir=gen_dir,
                build_dir=build_dir,
                weights_file=gen_dir / "weights.bin",
                tidl_bridge=bridge_path,
                use_tidl=True,
                tidl_artifacts_dir=artifacts_dir,
            )

            assert module_path.exists(), f"Build failed: {module_path}"
            size_mb = module_path.stat().st_size / (1024 * 1024)
            print(f"Module: {module_path} ({size_mb:.1f} MB)")

            # 7. Deploy and run on AM67A
            input_data = np.random.randn(1, 3, 32, 32).astype("float32")
            result, stdout = run_dsp_dload(
                module_path,
                gen_dir / "weights.bin",
                [input_data],
                embedded_weights=True,
            )

            # 8. Verify output
            assert result is not None, "No output from DSP"
            assert result.shape == (1, 16, 32, 32), f"Unexpected shape: {result.shape}"
            print(
                f"Output: shape={result.shape}, "
                f"min={result.min():.4f}, max={result.max():.4f}, "
                f"mean={result.mean():.4f}"
            )

            # Softmax output should sum to ~1 along channel axis
            sums = result.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=0.1), (
                f"Softmax channel sums should be ~1.0, got mean={sums.mean():.4f}"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(gen_dir), ignore_errors=True)
                if build_dir:
                    shutil.rmtree(str(build_dir), ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
