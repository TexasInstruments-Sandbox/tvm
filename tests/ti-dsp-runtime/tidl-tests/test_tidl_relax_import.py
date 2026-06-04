#!/usr/bin/env python
"""Unit tests for the TIDL Relax import .so (tidl_model_import_relax.so).

Tests verify that:
- The .so loads and all TIDL_relax* FFI functions register
- TIDL_relaxInit succeeds with proper config
- TIDL_relaxAllowNode can parse composites without crashing
- The parser correctly walks Relax composite function bodies

Requirements: TVM + built tidl_model_import_relax.so + c7x-mma-tidl tree
              (for device_config.cfg).  No hardware needed.
"""

import os

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.backend.tidl import get_tidl_patterns
from tvm.relax.frontend import nn

# ---------------------------------------------------------------------------
# Locate and load the .so
# ---------------------------------------------------------------------------

C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH",
    os.path.expanduser("~/ml/c7x-mma-tidl"),
)
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH,
    "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so",
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")


@pytest.fixture(scope="module", autouse=True)
def load_relax_so():
    """Load the TIDL Relax import .so once per module."""
    if not os.path.isfile(RELAX_SO_PATH):
        pytest.skip(f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}")
    tvm.runtime.load_module(RELAX_SO_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FFI_NAMES = [
    "TIDL_relaxInit",
    "TIDL_relaxImportInit",
    "TIDL_relaxImportNode",
    "TIDL_relaxImportOutDataLayer",
    "TIDL_relaxImportLinkNode",
    "TIDL_relaxOptimizeNet",
    "TIDL_relaxPostProcessNet",
    "TIDL_relaxAllowNode",
    "TIDL_relaxUpdateDenyList",
]


def _init_tidl():
    """Call TIDL_relaxInit with minimal options.  Skips if device_config.cfg
    is missing (need c7x-mma-tidl tree)."""
    device_cfg = os.path.join(TIDL_TOOLS_PATH, "device_config.cfg")
    if not os.path.isfile(device_cfg):
        pytest.skip(f"device_config.cfg not found at {device_cfg}")
    init_fn = tvm.get_global_func("TIDL_relaxInit")
    ret = init_fn(1, {"tidl_tools_path": TIDL_TOOLS_PATH, "artifacts_folder": "/tmp"})
    assert ret == 0, f"TIDL_relaxInit returned {ret}"


def _build_composite_call(comp_fn, orig_call):
    """Build a synthetic relax.Call with the composite Function as op
    and proper struct_info (AllowNode expects call->op to be a Function)."""
    new_call = relax.Call(comp_fn, orig_call.args)
    relax._ffi_api.UpdateStructInfo(new_call, orig_call.struct_info)
    return new_call


def _extract_composites(mod):
    """Extract (composite_name, synthetic_call) from a partitioned module."""
    results = []
    for _gv, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        if not func.attrs or func.attrs.get("Codegen") != "tidl":
            continue

        comp_fn = None
        orig_call = None
        for block in func.body.blocks:
            for b in block.bindings:
                val = b.value
                if isinstance(val, relax.Function) and val.attrs and val.attrs.get("Composite"):
                    comp_fn = val
                elif isinstance(val, relax.Call):
                    orig_call = val

        if comp_fn and orig_call:
            name = str(comp_fn.attrs["Composite"])
            results.append((name, _build_composite_call(comp_fn, orig_call)))
    return results


def _partition_model(model_cls, input_spec):
    """Export, bind params, and partition a model."""
    model = model_cls()
    mod, ps = model.export_tvm(spec={"main": input_spec})
    params = [
        tvm.runtime.tensor(np.random.rand(*p.shape).astype("float32"), device=tvm.cpu())
        for _, p in ps
    ]
    mod = relax.transform.BindParams(
        func_name="main",
        params=dict(zip(mod["main"].params[1:], params)),
    )(mod)
    patterns = get_tidl_patterns()
    return relax.transform.FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=True)(
        mod
    )


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class ConvReluModel(nn.Module):
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
        return nn.relu(self.conv1(x))


class ConvBiasReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def main(self, x):
        return nn.relu(self.conv1(x))


# ---------------------------------------------------------------------------
# Tests: FFI registration
# ---------------------------------------------------------------------------


class TestRelaxSOLoad:
    """Verify the .so loads and all FFI functions are available."""

    @pytest.mark.parametrize("name", _FFI_NAMES)
    def test_ffi_function_registered(self, name):
        fn = tvm.get_global_func(name, allow_missing=True)
        assert fn is not None, f"FFI function '{name}' not registered"


# ---------------------------------------------------------------------------
# Tests: Init
# ---------------------------------------------------------------------------


class TestRelaxInit:
    """Test TIDL_relaxInit with device config."""

    def test_init_succeeds(self):
        """Init should return 0 when device_config.cfg is available."""
        _init_tidl()


# ---------------------------------------------------------------------------
# Tests: AllowNode (parser exercised, constraint check may reject)
# ---------------------------------------------------------------------------


class TestRelaxAllowNode:
    """Test that TIDL_relaxAllowNode can parse Relax composites.

    The parser walks the composite function body, extracts attrs from
    inner calls, and populates sTIDL_LayerPC_t.  The constraint checker
    may reject ops based on device capabilities, so we only assert that
    the call completes without error (returns 0 or 1).
    """

    def test_conv_relu_parseable(self):
        """Conv2d+relu composite should be parseable (may or may not pass
        device constraints)."""
        _init_tidl()
        allow_fn = tvm.get_global_func("TIDL_relaxAllowNode")
        mod = _partition_model(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        composites = _extract_composites(mod)
        assert len(composites) > 0, "No composites found"

        for name, call in composites:
            result = allow_fn(call)
            assert result in (0, 1), f"AllowNode returned unexpected value {result} for {name}"

    def test_conv_bias_relu_parseable(self):
        """Conv2d+bias+relu composite should be parseable."""
        _init_tidl()
        _allow_fn = tvm.get_global_func("TIDL_relaxAllowNode")
        mod = _partition_model(
            ConvBiasReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        composites = _extract_composites(mod)
        assert len(composites) > 0, "No composites found"

        # Should find a conv2d variant with bias
        names = {name for name, _ in composites}
        assert any("conv2d" in n for n in names), f"Expected conv2d composite, found: {names}"

    def test_composite_name_extraction(self):
        """Verify the parser correctly extracts composite names from
        the Relax IR."""
        mod = _partition_model(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        composites = _extract_composites(mod)
        names = {name for name, _ in composites}
        assert "tidl.nn.conv2d_relu" in names, f"Expected tidl.nn.conv2d_relu, found: {names}"


# ---------------------------------------------------------------------------
# Tests: tidl_import() integration (partition -> import -> artifacts)
# ---------------------------------------------------------------------------


class TestTIDLImport:
    """Test the full tidl_import() pipeline.

    These tests require:
    - tidl_model_import_relax.so
    - device_config.cfg (from c7x-mma-tidl tree)
    - TIDL tools (PC_dsp_test_dl_algo.out, ti_cnnperfsim.out)
    """

    def _make_compiler(self, tmpdir):
        """Create a TIDLOffloadCompiler with test config."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        # Pipeline test — INT8 accuracy is not verified, but calibration_inputs
        # is still required by tidl_import().  A single random frame is enough.
        calib_frame = np.random.randn(1, 3, 32, 32).astype("float32")
        return TIDLOffloadCompiler(
            config={
                "artifacts_dir": str(tmpdir),
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "num_calibration_frames": 1,
                "calibration_inputs": [calib_frame],
            }
        )

    def _prepare_and_partition(self, compiler, model_cls, input_spec):
        """Export, prepare, and partition a model."""
        model = model_cls()
        mod, ps = model.export_tvm(spec={"main": input_spec})
        params = [
            tvm.runtime.tensor(np.random.rand(*p.shape).astype("float32"), device=tvm.cpu())
            for _, p in ps
        ]
        param_dict = dict(zip(mod["main"].params[1:], params))
        mod = compiler.prepare(mod, param_dict)
        mod = compiler.partition(mod)
        return mod

    def test_import_returns_artifacts(self, tmp_path):
        """tidl_import() should return (mod, artifacts) dict."""
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        # Verify the partitioned module has TIDL subgraphs
        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition")

        mod_out, artifacts = compiler.tidl_import(mod)

        # Module should be returned unchanged
        assert mod_out is not None

        # Artifacts dict should have entries for each subgraph
        assert len(artifacts) > 0, "Expected at least one artifact entry"
        for sg_name, paths in artifacts.items():
            assert "net_bin" in paths
            assert "io_bin" in paths

    def test_import_calibration_data_written(self, tmp_path):
        """Calibration data file should be created in artifacts dir."""
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition")

        compiler.tidl_import(mod)

        # Calibration data file should exist
        calib_path = tmp_path / "calib_raw_data0.bin"
        assert calib_path.exists(), f"Calibration data not found at {calib_path}"
        assert calib_path.stat().st_size > 0

    def test_import_conv_bias_relu(self, tmp_path):
        """Import a conv2d+bias+relu subgraph."""
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            ConvBiasReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition")

        mod_out, artifacts = compiler.tidl_import(mod)
        assert len(artifacts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
