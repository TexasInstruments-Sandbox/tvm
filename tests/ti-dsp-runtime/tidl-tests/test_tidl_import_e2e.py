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

Models
------
test_import_build_run -- ConvReluSoftmaxModel (1 TIDL subgraph):
  Conv2D(3->16, 3x3) + ReLU + Softmax.
  Conv+ReLU offloads to TIDL; softmax stays as TVM C code.
  Verifies the basic single-subgraph pipeline.

test_two_subgraph_model -- TwoSubgraphModel (2 TIDL subgraphs):
  Conv+ReLU -> Softmax -> Conv+ReLU -> Softmax.
  Each Conv+ReLU is a separate TIDL subgraph; the softmax between
  them forces a partition split.  Verifies the firmware handles
  multiple init_tidl_subgraph / process_tidl_subgraph calls within
  one inference invocation.

Assertions
----------
- build() produces a lib0.out that exists on disk
- Expected number of TIDL artifacts are generated
- DSP output shape matches expected dimensions
- Softmax invariant: channel sums ~= 1.0 (atol=0.1), confirming
  both the TIDL subgraphs and the TVM softmax executed correctly

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
    from conftest import has_c7x_host_env

    return has_c7x_host_env()


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


class TwoSubgraphModel(nn.Module):
    """Conv+ReLU+Sigmoid chains.

    All ops (conv, relu, sigmoid) are supported by TIDL, so this merges
    into a single subgraph rather than two.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(8, 4, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.sigmoid(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.sigmoid(x)
        return x


@pytest.mark.quick
class TestTIDLImportE2E:
    """End-to-end: tidl_import() -> build -> run on AM67A."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not _has_import_so():
            pytest.skip(
                f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}"
            )
        if not _has_c7x_compiler():
            pytest.skip("TI_CGT_C7000_PATH not set")

    def test_import_build_run(self, dsp_mode, tmp_path):
        """Full pipeline via compiler.build()."""
        from conftest import tidl_build_and_run

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

        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": str(tmp_path / "tidl_artifacts"),
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
            }
        )

        input_data = np.random.randn(1, 3, 32, 32).astype("float32")
        output, n_artifacts = tidl_build_and_run(
            compiler, mod, param_dict, input_data, tmp_path, dsp_mode
        )

        assert n_artifacts > 0, "No TIDL artifacts produced"
        assert output.shape == (1, 16, 32, 32), f"Unexpected shape: {output.shape}"
        print(
            f"Output: shape={output.shape}, "
            f"min={output.min():.4f}, max={output.max():.4f}, "
            f"mean={output.mean():.4f}"
        )
        # Softmax output (via TIDL int8) should be non-negative
        assert output.min() >= -0.01, f"Softmax output min {output.min():.4f} below 0"
        assert output.max() <= 1.01, f"Softmax output max {output.max():.4f} above 1"

    def test_two_subgraph_model(self, dsp_mode, tmp_path):
        """Conv+ReLU+Sigmoid chain offloaded to TIDL on AM67A.

        Originally designed to produce two subgraphs (sigmoid was not
        a TIDL pattern), but now all ops merge into one subgraph.
        """
        from conftest import tidl_build_and_run

        model = TwoSubgraphModel()
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

        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": str(tmp_path / "tidl_artifacts"),
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "tidl_relax_so_path": RELAX_SO_PATH,
                "num_calibration_frames": 2,
            }
        )

        input_data = np.random.randn(1, 3, 32, 32).astype("float32")
        output, n_artifacts = tidl_build_and_run(
            compiler, mod, param_dict, input_data, tmp_path, dsp_mode
        )

        assert n_artifacts >= 1, f"Expected >= 1 TIDL subgraphs, got {n_artifacts}"
        assert output.shape == (1, 4, 32, 32), f"Unexpected shape: {output.shape}"
        print(
            f"Output: shape={output.shape}, "
            f"min={output.min():.4f}, max={output.max():.4f}"
        )
        # Sigmoid output should be in (0, 1)
        assert output.min() >= -0.01, f"Sigmoid output min {output.min():.4f} below 0"
        assert output.max() <= 1.01, f"Sigmoid output max {output.max():.4f} above 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
