#!/usr/bin/env python
"""End-to-end TIDL hardware test on AM67A (c7x_dload).

Exercises the full automated build pipeline via a single
``TIDLOffloadCompiler.build()`` call, then deploys the resulting
DLOAD module to an AM67A board and verifies inference output.

Pipeline under test
-------------------
build() internally runs every stage of the TIDL offloading flow:

  1. prepare    -- bind params, fold constants, normalize
  2. partition  -- FuseOpsByPattern with TIDL patterns, merge composites
  3. tidl_import -- call TIDL import .so via Relax FFI to produce
                    net.bin + io.bin artifacts for each subgraph
  4. lower_tidl -- replace Codegen="tidl" functions with TIR extern
                   stubs (call_extern -> tidl_subgraph_N_process)
  5. relax.build + export -- c_static codegen emits lib0.c + weights.bin
  6. generate_bridge -- emit tidl_bridge.c with real TIDL API calls and
                        per-subgraph embedded artifact symbols
  7. _build_dynmod  -- cmake cross-compile via TI C7x toolchain,
                       two-stage link producing lib0.out

The test then deploys lib0.out to the AM67A via SSH/SCP and runs
inference through the c7x_compute DLOAD CLI.

Model
-----
ConvReluSoftmaxModel: Conv2D(3->16, 3x3) + ReLU + Softmax.
Conv+ReLU is offloaded to TIDL (MMA accelerator, int8 internally).
Softmax is not in the TIDL pattern table and stays as TVM-generated
C code executing on the C7x scalar pipeline.  This mixed model
ensures the VM register-file / call_tir plumbing works correctly
when TIDL and non-TIDL ops coexist.

Assertions
----------
- build() produces a lib0.out that exists on disk
- At least one TIDL artifact pair (net.bin + io.bin) is generated
- DSP output shape matches expected (1, 16, 32, 32)
- Softmax invariant: channel sums ~= 1.0 (atol=0.1), confirming
  both the TIDL subgraph and the TVM softmax executed correctly

Environment
-----------
Requires all of the following (test is skipped otherwise):

- tidl_model_import_relax.so  -- Relax FFI bridge to TIDL import tool.
  Built from c7x-mma-tidl repo (see tidl README for build steps).
  Located via TIDL_RELAX_SO_PATH or auto-detected from C7X_MMA_TIDL_PATH.

- TI C7x cross-compiler (TI_CGT_C7000_PATH) -- cl7x / lnk7x used by
  the cmake dynmod build to produce the relocatable ELF module.

- PSDK with TIDL + MMALIB libraries -- linked into the c7x_compute
  firmware (not the model module).  The firmware must already be
  deployed and running on the AM67A.

- AM67A board at hostname ``am67a`` with c7x_compute firmware running.
  The firmware provides DLOAD, TIDL algo libs, and shared UDMA/DMA
  resources.  See src/runtime/ti_dsp/firmware/c7x/ for firmware docs.

Set DSP_KEEP_TEMP=1 to preserve build artifacts for debugging.
"""

import os
import shutil
import sys

import numpy as np
import pytest

import tvm
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
        """Full pipeline via compiler.build()."""
        from dsp_utils import run_dsp_dload

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

        # 2. Build via single API call
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

        assert result.module_path.exists(), (
            f"Build failed: {result.module_path}"
        )
        assert len(result.artifacts) > 0, "No TIDL artifacts produced"

        size_mb = result.module_path.stat().st_size / (1024 * 1024)
        print(f"Module: {result.module_path} ({size_mb:.1f} MB)")

        # 3. Deploy and run on AM67A
        try:
            input_data = np.random.randn(1, 3, 32, 32).astype("float32")
            output, stdout = run_dsp_dload(
                result.module_path,
                result.weights_path,
                [input_data],
                embedded_weights=True,
            )

            # 4. Verify output
            assert output is not None, "No output from DSP"
            assert output.shape == (1, 16, 32, 32), (
                f"Unexpected shape: {output.shape}"
            )
            print(
                f"Output: shape={output.shape}, "
                f"min={output.min():.4f}, max={output.max():.4f}, "
                f"mean={output.mean():.4f}"
            )

            # Softmax output should sum to ~1 along channel axis
            sums = output.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=0.1), (
                f"Softmax channel sums should be ~1.0, got mean={sums.mean():.4f}"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(result.gen_dir), ignore_errors=True)
                shutil.rmtree(str(result.build_dir), ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
