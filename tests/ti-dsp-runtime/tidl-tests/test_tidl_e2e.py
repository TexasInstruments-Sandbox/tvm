#!/usr/bin/env python
"""End-to-end pipeline tests for TIDL subgraph offloading.

Level 1: Pipeline validation with stub bridge (c7x_host).
Verifies the full partition -> lower -> codegen -> bridge -> build -> run
pipeline without requiring actual TIDL libraries or artifacts.
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
)
from tvm.relax.frontend import nn

# Add dsp-cpp to path for dsp_utils
_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DSP_CPP_DIR = os.path.join(_TESTS_DIR, "dsp-cpp")
sys.path.insert(0, _DSP_CPP_DIR)


def _has_c7x_host_env():
    """Check if c7x_host build environment is available."""
    return os.environ.get("TI_CGT_C7000_PATH") is not None


def _export_and_bind(model_cls, input_spec):
    """Export model and bind random params as constants."""
    model = model_cls()
    mod, param_spec = model.export_tvm(spec={"main": input_spec})
    device = tvm.cpu()
    params = [
        np.random.rand(*param.shape).astype("float32") for _, param in param_spec
    ]
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
        ex = relax.build(
            mod, target=target, exec_mode="compiled", system_lib=True
        )

    td = tempfile.mkdtemp(prefix="tidl_e2e_")
    tar_path = os.path.join(td, "model.tar")
    ex.export_library(tar_path, target=target)
    with tarfile.open(tar_path) as tf:
        tf.extractall(td)
    os.remove(tar_path)

    return Path(td)


class ConvReluSoftmaxModel(nn.Module):
    """Mixed model: conv+relu offloaded to TIDL, softmax stays in TVM.

    Uses a mixed model because when ALL ops are offloaded, the c_static
    codegen produces a degenerate wrapper without cg_main_dsp (the DSP
    runtime entry point).  A mixed model ensures proper VM builtin usage.
    """

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
        x = nn.softmax(x, axis=1)  # not supported by TIDL patterns
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
            ConvReluSoftmaxModel,
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
            assert "tidl_subgraph_0_process" in open(str(gen_dir / "lib0.c")).read()

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
            assert result.shape == (1, 16, 32, 32), (
                f"Unexpected output shape: {result.shape}"
            )
            # With stub bridge, TIDL outputs zeros, then softmax
            # produces uniform distribution (1/16 per channel for axis=1)
            expected_val = 1.0 / 16.0
            assert np.allclose(result, expected_val, atol=1e-5), (
                f"Expected uniform softmax output (~{expected_val}), "
                f"got min={result.min():.6f} max={result.max():.6f}"
            )

        finally:
            if not os.environ.get("DSP_KEEP_TEMP"):
                shutil.rmtree(str(gen_dir), ignore_errors=True)
                if build_dir:
                    shutil.rmtree(str(build_dir), ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
