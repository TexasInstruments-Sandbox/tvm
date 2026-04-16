#!/usr/bin/env python
"""Unit tests for TIDL lowering and c_static code generation (Phase 4).

Tests verify that:
- The LowerTIDLToTIR pass correctly replaces Codegen='tidl' functions
  with TIR PrimFuncs containing call_extern
- The generated C code contains TIDL extern function calls
- The lowered module can be built through the c_static pipeline
"""

import os
import tarfile
import tempfile

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.tidl import LowerTIDLToTIR, partition_for_tidl
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R

# All tests in this file are quick (pure codegen, no hardware, no .so)
pytestmark = [pytest.mark.quick, pytest.mark.core]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _partition_and_lower(mod):
    """Partition for TIDL, then lower TIDL functions to TIR."""
    partitioned = partition_for_tidl(mod)
    lowered = LowerTIDLToTIR()(partitioned)
    return lowered


def _build_and_get_source(mod, target_str="c_static -mcpu=c7x"):
    """Build a module through c_static and return the generated lib0.c source."""
    target = tvm.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=0):
        ex = relax.build(mod, target=target)

    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "model.tar")
        ex.export_library(tar_path, target=target)
        with tarfile.open(tar_path) as tf:
            tf.extractall(td)

        lib0_path = os.path.join(td, "lib0.c")
        if not os.path.exists(lib0_path):
            return ""
        with open(lib0_path) as f:
            return f.read()


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class ConvReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        return x


class TwoConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2D(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False
        )

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        return x


# ---------------------------------------------------------------------------
# Tests: LowerTIDLToTIR pass
# ---------------------------------------------------------------------------


class TestLowerTIDLToTIR:
    """Test the TIDL lowering pass."""

    def test_tidl_functions_removed(self):
        """After lowering, no Codegen='tidl' functions should remain."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        for gv, func in lowered.functions.items():
            if isinstance(func, relax.Function) and func.attrs:
                assert func.attrs.get("Codegen") != "tidl", (
                    f"Found residual Codegen='tidl' function: {gv.name_hint}"
                )

    def test_tir_stub_created(self):
        """A TIR PrimFunc named tidl_subgraph_0 should exist."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        found_stub = False
        for gv, func in lowered.functions.items():
            if isinstance(func, tvm.tir.PrimFunc):
                if "tidl_subgraph" in gv.name_hint:
                    found_stub = True
        assert found_stub, "Expected a TIR PrimFunc named tidl_subgraph_*"

    def test_main_calls_tir_stub(self):
        """Main function should use call_tir to invoke the TIR stub."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        main_script = lowered["main"].script()
        assert "call_tir" in main_script, "Expected call_tir in main function"
        assert "tidl_subgraph" in main_script, "Expected tidl_subgraph reference in main function"

    def test_no_tidl_noop(self):
        """LowerTIDLToTIR on a module with no TIDL functions is a no-op."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        # Don't partition — no TIDL functions
        lowered = LowerTIDLToTIR()(mod)
        # Should be identical (no changes)
        assert "main" in {gv.name_hint for gv in lowered.functions}


# ---------------------------------------------------------------------------
# Tests: c_static code generation
# ---------------------------------------------------------------------------


class TestTIDLCodegen:
    """Test that the c_static codegen produces TIDL extern calls."""

    def test_extern_call_in_generated_code(self):
        """Generated C code should contain tidl_subgraph_0_process call."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)
        source = _build_and_get_source(lowered)

        assert "tidl_subgraph_0_process" in source, (
            "Expected tidl_subgraph_0_process in generated C code"
        )

    def test_builds_without_error(self):
        """Partition + lower + c_static build should succeed."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        target = tvm.target.Target("c_static -mcpu=c7x")
        with tvm.transform.PassContext(opt_level=0):
            # Should not raise
            ex = relax.build(lowered, target=target)
        assert ex is not None

    def test_two_conv_model_codegen(self):
        """Two-conv model should produce TIDL extern calls."""
        mod = _export_and_bind(
            TwoConvModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)
        source = _build_and_get_source(lowered)

        # Should have at least one tidl_subgraph process call
        assert "tidl_subgraph" in source, "Expected tidl_subgraph in generated C code"
        assert "_process" in source, "Expected _process extern call in generated C code"

    def test_mixed_model_codegen(self):
        """Model with TIDL ops + unsupported ops should have both
        TIDL extern calls and regular TVM compute in generated code."""

        @I.ir_module
        class ConvGeluModel:
            @R.function
            def main(x: R.Tensor((1, 3, 32, 32), "float32")):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                    y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                    y = R.nn.relu(y)
                    y = R.nn.gelu(y)
                    R.output(y)
                return y

        lowered = _partition_and_lower(ConvGeluModel)
        source = _build_and_get_source(lowered)

        # TIDL subgraph for conv+relu
        assert "tidl_subgraph_0_process" in source, "Expected TIDL extern call for conv+relu"
        # GELU should be lowered as regular TVM compute
        assert len(source) > 500, "Expected substantial generated code (gelu compute)"


# ---------------------------------------------------------------------------
# Tests: Bridge generation (multi-subgraph)
# ---------------------------------------------------------------------------


class ConvGeluConvModel(nn.Module):
    """Two TIDL subgraphs separated by an unsupported op.

    conv+relu -> gelu (not TIDL) -> conv+relu
    This produces two separate tidl_subgraph TIR stubs.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 16, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(16, 32, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.gelu(x)  # breaks the TIDL subgraph
        x = self.conv2(x)
        x = nn.relu(x)
        return x


class TestTIDLBridgeGeneration:
    """Test bridge code generation including multi-subgraph support."""

    def test_stub_bridge_single_subgraph(self):
        """Stub bridge for a single subgraph zeros the output."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            bridge_path = f.name
        try:
            TIDLOffloadCompiler.generate_bridge(lowered, bridge_path, stub=True)
            with open(bridge_path) as f:
                code = f.read()

            assert "tidl_subgraph_0_process" in code
            assert "memset(out0, 0," in code
        finally:
            os.unlink(bridge_path)

    def test_stub_bridge_multi_subgraph(self):
        """Stub bridge produces separate functions per subgraph."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod = _export_and_bind(
            ConvGeluConvModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        # Count TIR stubs — should be 2 for the two TIDL subgraphs
        stubs = [
            gv.name_hint
            for gv, func in lowered.functions.items()
            if isinstance(func, tvm.tir.PrimFunc) and "tidl_subgraph" in gv.name_hint
        ]
        assert len(stubs) == 2, f"Expected 2 TIR stubs, got {len(stubs)}: {stubs}"

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            bridge_path = f.name
        try:
            TIDLOffloadCompiler.generate_bridge(lowered, bridge_path, stub=True)
            with open(bridge_path) as f:
                code = f.read()

            assert "tidl_subgraph_0_process" in code
            assert "tidl_subgraph_1_process" in code
            # Both should have memset (stub mode)
            assert code.count("memset(out0, 0,") == 2, (
                "Expected 2 memset calls (one per subgraph stub)"
            )
        finally:
            os.unlink(bridge_path)

    def test_real_bridge_multi_subgraph_symbols(self):
        """Real bridge produces per-subgraph artifact symbols."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod = _export_and_bind(
            ConvGeluConvModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            bridge_path = f.name
        try:
            TIDLOffloadCompiler.generate_bridge(lowered, bridge_path, stub=False)
            with open(bridge_path) as f:
                code = f.read()

            # Per-subgraph artifact symbols
            assert "_binary_tidl_net_0_start" in code
            assert "_binary_tidl_net_0_size" in code
            assert "_binary_tidl_io_0_start" in code
            assert "_binary_tidl_net_1_start" in code
            assert "_binary_tidl_net_1_size" in code
            assert "_binary_tidl_io_1_start" in code

            # Both process functions
            assert "tidl_subgraph_0_process" in code
            assert "tidl_subgraph_1_process" in code

            # Each calls init_tidl_subgraph with its own artifacts
            assert code.count("init_tidl_subgraph(") == 2

            # Shared includes emitted once
            assert code.count('#include "tidl_api.h"') == 1
            assert code.count('#include "dlpack/dlpack.h"') == 1
        finally:
            os.unlink(bridge_path)

    def test_bridge_header_generated(self):
        """Bridge generation also produces a .h header."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        lowered = _partition_and_lower(mod)

        with tempfile.TemporaryDirectory() as td:
            bridge_path = os.path.join(td, "tidl_bridge.c")
            TIDLOffloadCompiler.generate_bridge(lowered, bridge_path, stub=True)

            header_path = os.path.join(td, "tidl_bridge.h")
            assert os.path.exists(header_path), "Bridge header not generated"
            with open(header_path) as f:
                header = f.read()
            assert "tidl_subgraph_0_process" in header


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
