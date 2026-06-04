#!/usr/bin/env python
"""End-to-end pipeline test with stub bridge on c7x_host.

Validates the full TIDL offloading pipeline without requiring TIDL
libraries, artifacts, or AM67A hardware.  Uses a stub bridge that
zero-fills TIDL subgraph outputs, allowing the rest of the pipeline
(partitioning, lowering, c_static codegen, cross-compile, execution)
to be verified on the host.

Pipeline under test
-------------------
  1. partition_for_tidl -- pattern match and merge TIDL subgraphs
  2. LowerTIDLToTIR    -- replace Codegen="tidl" funcs with TIR stubs
  3. relax.build       -- c_static codegen (lib0.c + weights.bin)
  4. generate_bridge   -- stub bridge (memset output to zero)
  5. build_dsp_c7x_host -- compile with g++ + TI Host Emulation library
  6. run_dsp_host      -- execute the binary on the host PC

Model
-----
ConvReluModel: Conv2D(3->16, 3x3) + ReLU.
Both ops match TIDL patterns so the entire model is one TIDL subgraph.
The stub bridge zero-fills the subgraph output, so the result is all
zeros.

Assertions
----------
- Build produces an executable
- Output shape is (1, 16, 32, 32)
- All output values are 0.0 (stub bridge zero-filled the subgraph),
  confirming the pipeline built and the stub bridge executed

Requirements
------------
- TI_CGT_C7000_PATH -- TI C7000 CGT with Host Emulation library
- TVM DSP runtime built for c7x_host target
- No TIDL .so, no hardware, no TIDL artifacts needed

See test_tidl_import_e2e.py for the full hardware test with real
TIDL artifacts on AM67A.
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
from tvm.relax.backend.tidl import (
    LowerTIDLToTIR,
    TIDLOffloadCompiler,
    partition_for_tidl,
    generate_artifacts_c,
)
from tvm.relax.frontend import nn

# Add dsp-cpp to path for dsp_utils
_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DSP_CPP_DIR = os.path.join(_TESTS_DIR, "dsp-cpp")
sys.path.insert(0, _DSP_CPP_DIR)


_C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH",
    os.path.expanduser("~/ml/c7x-mma-tidl"),
)
_RELAX_SO_PATH = os.path.join(
    _C7X_MMA_TIDL_PATH,
    "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so",
)
_TIDL_TOOLS_PATH = os.path.join(_C7X_MMA_TIDL_PATH, "tidl_tools")
_TIDL_PC_LIB_DIR = os.path.join(_C7X_MMA_TIDL_PATH, "ti_dl/lib/J722S/PC/algo/release")


def _has_c7x_host_env():
    """Check if c7x_host build environment is available."""
    return os.environ.get("TI_CGT_C7000_PATH") is not None


def _has_import_so():
    """Check if the TIDL import .so is present."""
    return os.path.isfile(_RELAX_SO_PATH)


def _has_tidl_pc_libs():
    """Check if PC (x86-64) TIDL algo libs are present."""
    return os.path.isfile(os.path.join(_TIDL_PC_LIB_DIR, "libtidl_algo.a"))


def _export_and_bind(model_cls, input_spec):
    """Export model and bind random params as constants."""
    model = model_cls()
    mod, param_spec = model.export_tvm(spec={"main": input_spec})
    device = tvm.cpu()
    params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
    params = [tvm.runtime.tensor(param, device=device) for param in params]
    func_params_dict = dict(zip(mod["main"].params[1:], params))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)
    return mod


def _compile_c_static(mod, target_str="c_static -mcpu=c7x"):
    """Compile a Relax module with c_static and extract to a temp dir.

    Uses exec_mode="compiled" and system_lib=True to generate the
    __vmtir__main TIR function and cg_main_dsp DSP wrapper, matching
    the DSP test pipeline in dsp_utils.py.
    """
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


class ConvReluModel(nn.Module):
    """Conv2D + ReLU model, fully offloaded to a single TIDL subgraph."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        return x


@pytest.mark.skipif(
    not _has_c7x_host_env(),
    reason="TI_CGT_C7000_PATH not set (c7x_host build environment required)",
)
class TestTIDLPipelineHost:
    """Level 1: Full pipeline validation with stub bridge on c7x_host."""

    def test_tidl_conv_relu_stub(self):
        """Partition + lower + codegen + stub bridge + c7x_host build + run."""
        from pathlib import Path

        from dsp_utils import (
            INPUT_BIN_FILE,
            build_dsp_c7x_host,
            run_dsp_host,
            write_tensors_to_file,
        )

        # 1. Create and partition model
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        partitioned = partition_for_tidl(mod)
        lowered = LowerTIDLToTIR()(partitioned)

        # 2. Compile to C
        gen_dir = _compile_c_static(lowered)

        build_dir = None
        try:
            # 3. Generate stub bridge
            bridge_path = str(gen_dir / "tidl_bridge.c")
            TIDLOffloadCompiler.generate_bridge(lowered, bridge_path, stub=True)

            assert os.path.exists(bridge_path)
            assert "tidl_subgraph_0_process" in open(bridge_path).read()

            # 4. Build with c7x_host + bridge
            build_dir = Path(tempfile.mkdtemp(prefix="tidl_build_"))
            exe_path = build_dsp_c7x_host(
                generated_dir=gen_dir,
                build_dir=build_dir,
                tidl_bridge=bridge_path,
            )
            assert exe_path.exists(), f"Build failed: {exe_path} not found"

            # 5. Write input and run
            input_data = np.random.randn(1, 3, 32, 32).astype("float32")
            input_file = build_dir / INPUT_BIN_FILE
            write_tensors_to_file([input_data], str(input_file))

            result = run_dsp_host(exe_path)

            # 6. Verify output
            assert result is not None, "Execution returned None"
            assert result.shape == (1, 16, 32, 32), f"Unexpected output shape: {result.shape}"
            # conv+relu are both TIDL patterns, so the entire model lands in
            # one TIDL subgraph.  The stub bridge zero-fills all subgraph
            # outputs, so the result should be all zeros.
            assert np.all(result == 0.0), (
                f"Expected all-zero stub output, got min={result.min():.6f} max={result.max():.6f}"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(gen_dir), ignore_errors=True)
                if build_dir:
                    shutil.rmtree(str(build_dir), ignore_errors=True)


@pytest.mark.skipif(
    not _has_c7x_host_env(),
    reason="TI_CGT_C7000_PATH not set (c7x_host build environment required)",
)
@pytest.mark.skipif(
    not _has_import_so(),
    reason=(
        f"tidl_model_import_relax.so not found at {_RELAX_SO_PATH} (build from c7x-mma-tidl repo)"
    ),
)
class TestTIDLPipelineHostRealBridge:
    """Level 2: Full TIDL pipeline with real bridge on c7x_host (AVX emulation).

    Unlike TestTIDLPipelineHost (stub bridge), this class runs the actual TIDL
    import to produce net.bin / io.bin artifacts, then compiles tidl_api.c with
    the PC x86-64 TIDL algo libs (AVX reference path).  TIDL outputs are real
    conv+relu values, confirming end-to-end inference executed.

    PC TIDL algo libs are located at:
      ~/ml/c7x-mma-tidl/ti_dl/lib/J722S/PC/algo/release/libtidl_algo.a
    These are the same AVX-accelerated reference implementation used by the
    TIDL import tool when running on PC (HOST_EMULATION path in tidl_api.c).

    Pipeline
    --------
    1. Export + bind params (ConvReluModel)
    2. partition_for_tidl  -- mark conv+relu as TIDL subgraph
    3. TIDLOffloadCompiler.tidl_import()  -- produce net.bin + io.bin
    4. LowerTIDLToTIR     -- replace Codegen="tidl" funcs with call_extern
    5. relax.build          -- c_static codegen -> lib0.c + weights.bin
                              (-tidl-runtime=1 so cg_main_dsp calls init)
    6. generate_bridge      -- real TIDL API calls (stub=False)
    7. generate_artifacts_c -- embed net.bin/io.bin as _binary_ C arrays
    8. build_dsp_c7x_host   -- g++ + TI Host Emu + PC TIDL libs (USE_TIDL=ON)
    9. run_dsp_host         -- execute binary on host
    10. Assert output is non-zero (TIDL ran for real, not stub)

    Skip conditions
    ---------------
    - TI_CGT_C7000_PATH not set
    - tidl_model_import_relax.so not found
    - PC TIDL algo libs not found (skip inside test with clear message)
    """

    def test_tidl_conv_relu_real_bridge(self, tmp_path):
        """Partition + TIDL import + real bridge + c7x_host build + run.

        Conv+ReLU runs on the TIDL PC (AVX reference) emulator.  Unlike
        the stub test, TIDL outputs are real (non-zero), confirming the
        full inference pipeline executed end-to-end.
        """
        from pathlib import Path

        from dsp_utils import (
            INPUT_BIN_FILE,
            build_dsp_c7x_host,
            run_dsp_host,
            write_tensors_to_file,
        )

        # TODO: When a c7x_host-native TIDL emulation library that provides
        # the C7x MMA accelerated path becomes available (e.g. a host-emulation
        # variant of the DSP tidl_algo.lib), update this test to use it.
        # For now we use the PC x86-64 AVX reference path which does not
        # exercise the MMA co-processor emulation but does validate the real
        # TIDL inference pipeline end-to-end on the host.
        if not _has_tidl_pc_libs():
            pytest.skip(
                f"PC TIDL algo libs not found at {_TIDL_PC_LIB_DIR}/libtidl_algo.a. "
                "Build them from the c7x-mma-tidl repo (PC target)."
            )

        # 1. Create and partition model
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        partitioned = partition_for_tidl(mod)

        # 2. Run TIDL import to produce net.bin + io.bin artifacts
        artifacts_dir = str(tmp_path / "tidl_artifacts")
        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": artifacts_dir,
                "tidl_tools_path": _TIDL_TOOLS_PATH,
                "num_calibration_frames": 1,
                # Pipeline test — accuracy not checked; one random frame is enough.
                "calibration_inputs": [np.random.randn(1, 3, 32, 32).astype("float32")],
            }
        )
        lowered_mod, artifacts = compiler.tidl_import(partitioned)
        assert artifacts, "TIDL import produced no artifacts"

        # 3. Lower TIDL subgraphs to call_extern TIR stubs
        lowered_mod = LowerTIDLToTIR()(lowered_mod)

        # 4. Compile to C (lib0.c + weights.bin).
        # -tidl-runtime=1 causes cg_main_dsp to call tidl_bridge_init_all()
        # before the first inference, which is required for real TIDL bridges.
        gen_dir = _compile_c_static(lowered_mod, "c_static -mcpu=c7x -tidl-runtime=1")

        build_dir = None
        try:
            # 5. Generate real TIDL bridge
            bridge_path = str(gen_dir / "tidl_bridge.c")
            TIDLOffloadCompiler.generate_bridge(
                lowered_mod,
                bridge_path,
                stub=False,
                artifacts_dir=artifacts_dir,
            )
            assert os.path.exists(bridge_path)
            # Real bridge should contain init_tidl_subgraph calls
            bridge_src = open(bridge_path).read()
            assert "init_tidl_subgraph" in bridge_src, "Real bridge missing init_tidl_subgraph call"

            # 6. Generate C source with _binary_tidl_net_N_start[] symbols.
            # Passed alongside the bridge .c so the linker resolves the externs
            # in tidl_bridge.c without any CMakeLists changes.
            artifacts_c = str(gen_dir / "tidl_artifacts.c")
            generate_artifacts_c(artifacts_dir, artifacts_c)
            assert os.path.exists(artifacts_c), "generate_artifacts_c produced no output"

            # 7. Build with c7x_host + USE_TIDL (PC AVX libs) + real bridge
            build_dir = Path(tempfile.mkdtemp(prefix="tidl_real_build_"))
            exe_path = build_dsp_c7x_host(
                generated_dir=gen_dir,
                build_dir=build_dir,
                tidl_bridge=[bridge_path, artifacts_c],
                use_tidl=True,
            )
            assert exe_path.exists(), f"Build failed: {exe_path} not found"

            # 8. Write input and run
            np.random.seed(0)
            input_data = np.random.randn(1, 3, 32, 32).astype("float32")
            input_file = build_dir / INPUT_BIN_FILE
            write_tensors_to_file([input_data], str(input_file))

            result = run_dsp_host(exe_path)

            # 9. Verify output
            assert result is not None, "Execution returned None"
            assert result.shape == (1, 16, 32, 32), f"Unexpected output shape: {result.shape}"
            # Real TIDL ran: output must be non-zero (stub would give all zeros).
            # Note: INT8 quantization with random calibration data can produce
            # small negative values even after ReLU; only finiteness and
            # non-zero max are checked here.
            assert np.isfinite(result).all(), "Output contains non-finite values"
            assert result.max() > 0, "Output is all zeros: TIDL did not run (stub-like output)"

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(gen_dir), ignore_errors=True)
                if build_dir:
                    shutil.rmtree(str(build_dir), ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
