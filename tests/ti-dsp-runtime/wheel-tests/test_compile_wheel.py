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

    def test_c_static_target(self):
        """c_static target must be registered."""
        import tvm
        target = tvm.target.Target("c_static -mcpu=c7x")
        assert target.kind.name == "c_static"

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
    def test_dynmod_infra_bundled(self):
        """Dynmod build infrastructure must be bundled."""
        from tvm.data.ti_dsp.paths import find_dsp_runtime_dir
        dsp_dir = find_dsp_runtime_dir()
        assert dsp_dir is not None
        assert (dsp_dir / "dynmod" / "CMakeLists.txt").exists()
        assert (dsp_dir / "cmake" / "toolchain-j722s-c7x.cmake").exists()

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
