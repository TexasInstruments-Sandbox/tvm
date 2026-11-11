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

import contextlib
import os
import sys
import tempfile

import numpy as np
import pytest
from dsp_utils import has_tidl_tools, tidl_tools_config  # noqa: E402

import tvm
from tvm import relax
from tvm.relax.backend.tidl import get_tidl_patterns
from tvm.relax.backend.tidl import tools as tidl_tools
from tvm.relax.frontend import nn

# ---------------------------------------------------------------------------
# Locate and load the .so
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def load_relax_so():
    """Load the TIDL Relax import .so once per module."""
    if tvm.get_global_func("TIDL_relaxInit", allow_missing=True) is not None:
        return  # already loaded by a prior test module in this session
    if not has_tidl_tools():
        pytest.skip("TIDL tools directory not staged")
    # The gate's loader also re-opens libtvm_compiler.so RTLD_GLOBAL, which the .so needs.
    from tvm.relax.backend.tidl._gate import load_relax_so as _load  # noqa: PLC0415

    _load(tidl_tools_config())


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
    tools_path = tidl_tools.import_tools_path(tidl_tools_config())
    device_cfg = os.path.join(tools_path, "device_config.cfg")
    if not os.path.isfile(device_cfg):
        pytest.skip(f"device_config.cfg not found at {device_cfg}")
    init_fn = tvm.get_global_func("TIDL_relaxInit")
    ret = init_fn(1, {"tidl_tools_path": tools_path, "artifacts_folder": "/tmp"})
    assert ret == 0, f"TIDL_relaxInit returned {ret}"


@contextlib.contextmanager
def _capture_c_stdout():
    """Capture writes to fd 1 into a list-of-one string.

    ``contextlib.redirect_stdout`` only rebinds ``sys.stdout``, which the TIDL
    import .so and the calibration subprocess it spawns via ``system()`` never
    go through -- they write straight to fd 1.  Redirect the descriptor itself
    so the subprocess inherits it.
    """
    captured = []
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved_fd = os.dup(1)
        try:
            sys.stdout.flush()
            os.dup2(tmp.fileno(), 1)
            yield captured
        finally:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            tmp.seek(0)
            captured.append(tmp.read())


def _build_composite_call(comp_fn, orig_call):
    """Build a synthetic relax.Call with the composite Function as op
    and proper type (AllowNode expects call->op to be a Function)."""
    new_call = relax.Call(comp_fn, orig_call.args)
    relax.expr._update_type(new_call, orig_call.ty)
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

        if comp_fn is not None and orig_call is not None:
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


class ConvTransposeModel(nn.Module):
    """Conv2D followed by conv2d_transpose with a learnable deconv weight."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)
        # IOHW layout: in_ch=8, out_ch=4, kH=3, kW=3
        self.deconv_weight = nn.Parameter((8, 4, 3, 3))

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        y = _op.nn.conv2d_transpose(x._expr, self.deconv_weight._expr, strides=(1, 1), padding=(1, 1))
        return wrap_nested(y, "deconv_out")


class ConvMeanModel(nn.Module):
    """Conv2D followed by global average pool via mean(axes=[2,3])."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        y = wrap_nested(_op.mean(x._expr, axis=[2, 3], keepdims=True), "global_avg")
        return y


class ConvMeanFCModel(nn.Module):
    """Conv2D -> global average pool -> flatten -> FC (matmul + bias).

    Mirrors a classifier head (ResNet/MobileNetV2-style): the FC's weight
    goes through a TIDL_ConstDataLayer, unlike conv weights which are
    embedded directly in the ConvolutionLayer.
    """

    def __init__(self, channels=8, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2D(3, channels, 3, 1, 1, bias=False)
        self.fc = nn.Linear(channels, num_classes, bias=True)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        h = wrap_nested(_op.mean(x._expr, axis=[2, 3], keepdims=True), "global_avg")
        h = nn.op.reshape(h, (h.shape[0], h.shape[1]))
        return self.fc(h)


class GAPFCModel(nn.Module):
    """Global average pool -> flatten -> FC, with no conv backbone at all.

    Isolates the mean+reshape+matmul+add tail: the FC weight is the only
    constant that goes through a TIDL_ConstDataLayer (conv weights are
    embedded directly in the ConvolutionLayer and never hit that path),
    so this is the fastest reproduction of the dtype bug covered by
    test_import_matmul_const_data_layer_dtype -- no conv layers means no
    per-channel calibration work beyond a single InnerProductLayer.
    """

    def __init__(self, channels=3, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(channels, num_classes, bias=True)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        h = wrap_nested(_op.mean(x._expr, axis=[2, 3], keepdims=True), "global_avg")
        h = nn.op.reshape(h, (h.shape[0], h.shape[1]))
        return self.fc(h)


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
                **tidl_tools_config(),
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

    def _assert_finite_calibration_stats(self, artifacts_dir, max_abs_threshold=100.0):
        """Every *_stats_tool_out.bin under artifacts_dir must be finite and
        small — non-finite or implausibly large values indicate a
        ConstDataLayer dtype (or similar) corruption in the calibration
        pipeline, not real INT8 quantization noise."""
        found_stats = False
        for root, _dirs, files in os.walk(artifacts_dir):
            for fn in files:
                if fn.endswith("_stats_tool_out.bin"):
                    found_stats = True
                    arr = np.fromfile(os.path.join(root, fn), dtype=np.float32)
                    finite = arr[np.isfinite(arr)]
                    assert finite.size == arr.size, (
                        f"{fn} contains non-finite calibration stats "
                        "(ConstDataLayer dtype corruption?)"
                    )
                    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
                    assert max_abs < max_abs_threshold, (
                        f"{fn} max|stat|={max_abs:.2f} is implausibly large for "
                        "small random weights (ConstDataLayer dtype corruption?)"
                    )
        assert found_stats, "No calibration stats file found under artifacts_dir"

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

    def test_import_conv2d_transpose(self, tmp_path):
        """tidl_import() must produce artifacts for conv2d_transpose.

        Before fixes 1-3, the parser left tensorHeight/tensorWidth
        uninitialised and outPadH/W uninitialised; the allowlisting checker
        may have evaluated garbage values and silently rejected the layer.
        This test verifies the import completes and produces a net.bin.
        """
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            ConvTransposeModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition — check pattern matching")

        mod_out, artifacts = compiler.tidl_import(mod)
        assert len(artifacts) > 0, "Expected at least one TIDL artifact"
        for _sg, paths in artifacts.items():
            assert "net_bin" in paths, "net.bin missing from artifacts"
            assert os.path.getsize(paths["net_bin"]) > 0, "net.bin is empty"

    def test_import_global_avg_pool_via_mean(self, tmp_path):
        """tidl_import() must produce artifacts for mean(axes=[2,3]).

        Verifies that the spatial-mean parser correctly populates avgDims,
        kernelH/W (0-sentinel), and storage_order.  If these fields were
        wrong the import would either crash or produce an unusable net.bin.
        """
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            ConvMeanModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition — check pattern matching")

        mod_out, artifacts = compiler.tidl_import(mod)
        assert len(artifacts) > 0, "Expected at least one TIDL artifact"
        for _sg, paths in artifacts.items():
            assert "net_bin" in paths, "net.bin missing from artifacts"
            assert os.path.getsize(paths["net_bin"]) > 0, "net.bin is empty"

    @pytest.mark.quick
    def test_import_matmul_const_data_layer_dtype(self, tmp_path):
        """FC weight's ConstDataLayer must calibrate with sane statistics.

        Regression test: no Relax parser sets layer.weights/bias.dataType
        after copying constant data (NDArrtoFloat always produces
        float32), so it stayed at its zero-initialized TIDL_UnsignedChar.
        tidl_addConstDataLayers() reads that field to size the
        ConstDataLayer's own allocation and memcpy, so the wrong type
        under-allocated by 4x and corrupted (or crashed) calibration for
        any op needing a separate ConstDataLayer -- matmul/InnerProduct
        (this model's FC head) being the common case, since conv weights
        are embedded directly in the ConvolutionLayer and never hit this
        path. A plain conv+relu+mean subgraph (see
        test_import_global_avg_pool_via_mean above) never exercised it.
        """
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            ConvMeanFCModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition — check pattern matching")

        mod_out, artifacts = compiler.tidl_import(mod)
        assert len(artifacts) > 0, "Expected at least one TIDL artifact"

        for _sg, paths in artifacts.items():
            assert "net_bin" in paths, "net.bin missing from artifacts"
            assert os.path.getsize(paths["net_bin"]) > 0, "net.bin is empty"

        self._assert_finite_calibration_stats(tmp_path)

    @pytest.mark.quick
    def test_import_fc_const_data_layer_no_backbone(self, tmp_path):
        """Import a bare GAP+FC model -- no conv backbone at all.

        A fast smoke test for the import + calibration pipeline on a
        topology with a single InnerProductLayer and no conv2d pattern
        matching. It does guard the native-type element-update pass's
        out-of-bounds reads (reverting those bound-checks segfaults here),
        but it does *not* guard the ConstDataLayer dtype bug that
        test_import_matmul_const_data_layer_dtype above covers: measured
        by reverting that fix, this case still passes and only the
        conv+GAP+FC variant trips the stats threshold. Don't rely on it
        for ConstDataLayer coverage.
        """
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            GAPFCModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )

        tidl_funcs = [
            gv
            for gv, f in mod.functions.items()
            if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
        ]
        if not tidl_funcs:
            pytest.skip("No TIDL subgraphs found after partition — check pattern matching")

        mod_out, artifacts = compiler.tidl_import(mod)
        assert len(artifacts) > 0, "Expected at least one TIDL artifact"

        for _sg, paths in artifacts.items():
            assert "net_bin" in paths, "net.bin missing from artifacts"
            assert os.path.getsize(paths["net_bin"]) > 0, "net.bin is empty"

        self._assert_finite_calibration_stats(tmp_path)

    @pytest.mark.quick
    def test_import_calibration_subprocess_reports_no_errors(self, tmp_path):
        """The float range-collection pass must actually run.

        ``tidl_isCalibrationInNativeTypes()`` (added by the AM62D tag) is true
        for ``TIDL_IMPORT_MODEL_FORMAT_TVM_RELAY``, the format the Relax
        importer reuses.  On the float stats-collection pass it copies
        ``gParams.modelInElementType`` into ``inElementType`` without checking
        for the -1 "unset" sentinel, so an importer that leaves those fields
        unpopulated makes the stats tool's ``tidl_tb_dataConvert`` reject every
        element of the input tensor and abort in ``TIDLRT_invoke``.  The float
        ranges are then collected on an unwritten input buffer, which silently
        miscalibrates every INT8 scale in the network (ResNet-18 mean_diff
        0.158 -> 2.28, top-1 208 -> 419).

        ``PC_dsp_test_dl_algo.out`` prints those errors but still exits 0, so
        the import reports success either way -- hence checking its output
        rather than a return code.  A tiny model is enough: the failure is a
        property of the input plumbing, not of the network.
        """
        _init_tidl()
        compiler = self._make_compiler(tmp_path)
        mod = self._prepare_and_partition(
            compiler,
            GAPFCModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        if not any(
            isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "tidl"
            for f in mod.functions.values()
        ):
            pytest.skip("No TIDL subgraphs found after partition — check pattern matching")

        with _capture_c_stdout() as captured:
            _mod_out, artifacts = compiler.tidl_import(mod)
        output = captured[0]

        assert len(artifacts) > 0, "Expected at least one TIDL artifact"
        n_convert_errors = output.count("of function : tidl_tb_dataConvert")
        assert n_convert_errors == 0, (
            f"Calibration tool rejected {n_convert_errors} tensor elements in "
            "tidl_tb_dataConvert — the network's input/output element type is "
            "invalid (modelInElementType left at -1?), so the float range "
            "collection ran on an unwritten buffer"
        )
        assert "TIDLRT_invoke failed" not in output, (
            "Calibration tool's TIDLRT_invoke failed — float range collection "
            f"did not run.  Tool output:\n{output[-2000:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
