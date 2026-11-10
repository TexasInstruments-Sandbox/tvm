"""Tests for the tvm-ti-c7x-compile (x86) wheel.

Verifies that the installed wheel contains the expected artifacts
and that core TVM functionality works through the wheel-installed
package.  Artifact tests are skipped when running from a source tree
(artifacts only exist in the wheel).
"""

import pytest

pytestmark = [pytest.mark.quick]


def _has_bundled_artifacts():
    """True when running from an installed wheel with DSP artifacts."""
    try:
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        return (get_ti_dsp_path() / "lib").is_dir()
    except ImportError:
        return False


_skip_source_tree = pytest.mark.skipif(
    not _has_bundled_artifacts(),
    reason="DSP artifacts only available in installed wheel",
)


class TestCompileWheelContents:
    """Verify bundled artifacts are present and discoverable."""

    def test_tvm_imports(self):
        """Core TVM modules must be importable."""
        import tvm
        from tvm import relax
        assert tvm.__version__

    def test_llvm_enabled(self):
        """LLVM codegen must be available for CPU reference builds."""
        import tvm
        assert tvm.runtime.enabled("llvm"), (
            "LLVM not enabled in libtvm.so — rebuild with USE_LLVM=ON"
        )

    def test_c_static_lib_target(self):
        """c_static_lib target must be registered."""
        import tvm
        target = tvm.target.Target("c_static_lib -mcpu=c7x")
        assert target.kind.name == "c_static_lib"

    def test_dsp_data_paths(self):
        """Bundled DSP data directory must exist with paths module."""
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        data_dir = get_ti_dsp_path()
        assert data_dir.is_dir()

    @_skip_source_tree
    def test_dsp_runtime_libs(self):
        """Cross-compiled DSP runtime libraries must be bundled."""
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        lib_dir = get_ti_dsp_path() / "lib"
        assert (lib_dir / "libtvm_dsp_runtime_c7x.a").exists()
        assert (lib_dir / "libtvm_dsp_runtime_c7x_host.a").exists()

    @_skip_source_tree
    def test_firmware_bundled(self):
        """Firmware binaries must be bundled."""
        from tvm.data.ti_dsp.paths import get_ti_dsp_path
        fw_dir = get_ti_dsp_path() / "firmware"
        assert (fw_dir / "c7x_compute.out").exists()
        assert (fw_dir / "c7x_compute").exists()
        assert (fw_dir / "libc7x_arm_runtime.so").exists()

    @_skip_source_tree
    def test_tidl_so_bundled(self):
        """TIDL import .so must be bundled."""
        from tvm.data.ti_dsp.paths import find_tidl_relax_so
        path = find_tidl_relax_so()
        assert path is not None, "tidl_model_import_relax.so not found"

    @_skip_source_tree
    def test_tidl_tools_bundled(self):
        """The tools the TIDL import .so runs must be bundled, and used when
        neither tidl_tools_path nor C7X_MMA_TIDL_PATH is set."""
        import os
        from pathlib import Path
        from unittest import mock
        from tvm.data.ti_dsp.paths import find_tidl_tools_dir
        from tvm.relax.backend.tidl._gate import resolve_tools_path
        tools = find_tidl_tools_dir()
        assert tools is not None, "tidl_tools not bundled"
        for name in ("PC_dsp_test_dl_algo.out", "ti_cnnperfsim.out"):
            assert os.access(Path(tools) / name, os.X_OK), f"{name} missing or not executable"
        env = {k: v for k, v in os.environ.items() if k != "C7X_MMA_TIDL_PATH"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert resolve_tools_path({}) == tools

    @_skip_source_tree
    def test_tidl_iobufdesc_layout_bundled(self):
        """The IOBufDesc layout table must be bundled and valid, so TIDL
        output padding works without itidl_io.h."""
        from tvm.data.ti_dsp.paths import find_tidl_iobufdesc_layout
        from tvm.relax.backend.tidl.io_buf_desc import load_layout
        path = find_tidl_iobufdesc_layout()
        assert path is not None, "sTIDL_IOBufDesc_t.layout.json not found"
        layout, source = load_layout()
        assert source == path
        assert layout["size"] > 0 and "outBufSize" in layout["fields"]

    @_skip_source_tree
    def test_tidl_host_bundle(self):
        """c7x_host TIDL must build from the wheel: the glue archive, the
        TIDL/MMALIB x86 archives and the dsp-cpp harness are bundled."""
        from tvm.data.ti_dsp.paths import find_host_harness_dir, get_ti_dsp_path
        harness = find_host_harness_dir()
        assert harness is not None, "c7x_host harness not bundled"
        assert (harness / "main_dsp.cpp").exists()
        host = get_ti_dsp_path() / "lib" / "host"
        for name in ("tidl_api_host", "tidl_algo", "tidl_ref", "tidl_priv", "tidl_kernels",
                     "tidl_avx_kernels", "tidl_custom", "mmalib_x86_64", "mmalib_cn_x86_64",
                     "common_x86_64"):
            assert (host / f"lib{name}.a").exists(), f"lib{name}.a not bundled"

    @_skip_source_tree
    def test_dynmod_infra_bundled(self):
        """Dynmod build infrastructure must be bundled."""
        from tvm.data.ti_dsp.paths import find_dsp_runtime_dir
        dsp_dir = find_dsp_runtime_dir()
        assert dsp_dir is not None
        assert (dsp_dir / "dynmod" / "CMakeLists.txt").exists()
        assert (dsp_dir / "cmake" / "toolchain-c7x.cmake").exists()

    def test_tidl_patterns_available(self):
        """TIDL partitioning patterns must load without the import .so."""
        from tvm.relax.backend.tidl import get_tidl_patterns
        patterns = get_tidl_patterns()
        assert len(patterns) > 50


class TestCompileWheelFunctionality:
    """Verify compilation works through the wheel."""

    def test_torch_to_relax(self):
        """PyTorch model export to Relax IR must work."""
        import torch
        from tvm.relax.frontend.torch import from_exported_program

        model = torch.nn.Linear(16, 4)
        model.eval()
        example = torch.randn(1, 16)
        exported = torch.export.export(model, (example,))
        mod = from_exported_program(exported, keep_params_as_input=True)
        assert mod is not None

    def test_relax_build_llvm_reference(self):
        """relax.build with LLVM target must work for CPU reference."""
        import numpy as np
        import torch

        import tvm
        from tvm import relax
        from tvm.relax.frontend.torch import from_exported_program

        model = torch.nn.Linear(8, 4, bias=False)
        model.eval()
        example = torch.randn(1, 8)
        exported = torch.export.export(model, (example,))
        mod = from_exported_program(exported, keep_params_as_input=True)

        ex = relax.build(mod, target=tvm.target.Target("llvm"))
        vm = relax.VirtualMachine(ex, tvm.cpu())
        params = [v.detach().numpy() for v in model.parameters()]
        result = vm["main"](example.numpy(), *params)
        if hasattr(result, "numpy"):
            out = result.numpy()
        else:
            out = result[0].numpy()
        assert out.shape == (1, 4)
