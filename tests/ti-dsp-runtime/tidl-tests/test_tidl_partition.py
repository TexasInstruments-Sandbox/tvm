#!/usr/bin/env python
"""Unit tests for TIDL pattern matching and partitioning (Phase 1).

Tests verify that:
- TIDL-supported ops are correctly matched and grouped into composite
  functions annotated with Composite="tidl.*"
- Adjacent TIDL composites are merged into subgraph functions annotated
  with Codegen="tidl"
- Unsupported ops (or ops violating constraints) remain in the main
  function
"""

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.backend.tidl import get_tidl_patterns, partition_for_tidl
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R

# ---------------------------------------------------------------------------
# Helper: inspect partitioned module
# ---------------------------------------------------------------------------


def _count_tidl_subgraphs(mod):
    """Count functions with Codegen='tidl'."""
    return len([
        gv for gv, func in mod.functions.items()
        if isinstance(func, relax.Function)
        and func.attrs
        and func.attrs.get("Codegen") == "tidl"
    ])


def _find_composites_in_module(mod, prefix="tidl."):
    """Find all Composite names in the module, including nested inner functions.

    After MergeCompositeFunctions, composites become inner functions of the
    Codegen-annotated outer function.  We walk the IR to find them.
    """
    composites = set()

    def _walk_expr(expr):
        if isinstance(expr, relax.Function):
            if expr.attrs:
                comp = expr.attrs.get("Composite")
                if comp and str(comp).startswith(prefix):
                    composites.add(str(comp))
            _walk_expr(expr.body)
        elif isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    _walk_expr(binding.value)
        elif isinstance(expr, relax.Call):
            if isinstance(expr.op, relax.Function):
                _walk_expr(expr.op)
            for arg in expr.args:
                _walk_expr(arg)

    for _gv, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _walk_expr(func)

    return composites


def _has_composite(mod, composite_name):
    """Check if a specific composite function exists in the module."""
    return composite_name in _find_composites_in_module(mod)


def _has_main_function(mod):
    """Check that the module still has a main function."""
    for gv in mod.functions:
        if gv.name_hint == "main":
            return True
    return False


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SimpleConv2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=16, kernel_size=3,
            stride=1, padding=1, bias=False
        )

    def main(self, x):
        return self.conv1(x)


class ConvReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=16, kernel_size=3,
            stride=1, padding=1, bias=False
        )

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        return x


class TwoConvModel(nn.Module):
    """Two consecutive conv layers — should merge into one TIDL subgraph."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3, out_channels=16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2D(
            in_channels=16, out_channels=32, kernel_size=3,
            stride=1, padding=1, bias=False
        )

    def main(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        return x


def _export_and_bind(model_cls, input_spec):
    """Export model and bind random params as constants."""
    model = model_cls()
    mod, param_spec = model.export_tvm(spec={"main": input_spec})

    device = tvm.cpu()
    params = [
        np.random.rand(*param.shape).astype("float32")
        for _, param in param_spec
    ]
    params = [tvm.runtime.tensor(param, device=device) for param in params]

    func_params_dict = dict(zip(mod["main"].params[1:], params))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTIDLPatternRegistration:
    """Test that TIDL patterns are properly registered."""

    def test_patterns_exist(self):
        """Verify get_tidl_patterns returns non-empty list."""
        patterns = get_tidl_patterns()
        assert len(patterns) > 0

    def test_pattern_names_prefixed(self):
        """All pattern names should start with 'tidl.'."""
        patterns = get_tidl_patterns()
        for pat in patterns:
            assert pat.name.startswith("tidl."), (
                f"Pattern {pat.name} does not have 'tidl.' prefix"
            )

    def test_expected_patterns_present(self):
        """Check that key patterns exist."""
        patterns = get_tidl_patterns()
        names = {p.name for p in patterns}
        expected = {
            "tidl.nn.conv2d",
            "tidl.nn.conv2d_bias",
            "tidl.nn.conv2d_relu",
            "tidl.nn.conv2d_bias_relu",
            "tidl.nn.max_pool2d",
            "tidl.nn.avg_pool2d",
            "tidl.nn.batch_norm",
            "tidl.nn.relu",
            "tidl.add",
        }
        for exp in expected:
            assert exp in names, f"Expected pattern '{exp}' not found"


class TestTIDLPartitionSimple:
    """Test partitioning on simple models."""

    def test_conv2d_partitioned(self):
        """A single conv2d should be partitioned into a TIDL subgraph."""
        mod = _export_and_bind(
            SimpleConv2dModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        partitioned = partition_for_tidl(mod)

        assert _has_main_function(partitioned)
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl >= 1, (
            f"Expected at least 1 TIDL subgraph, got {n_tidl}"
        )

    def test_conv_relu_fused(self):
        """Conv2d + relu should be fused into a single conv2d_relu composite."""
        mod = _export_and_bind(
            ConvReluModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        partitioned = partition_for_tidl(mod)

        assert _has_main_function(partitioned)
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl >= 1

        # Check that conv2d_relu composite exists (nested inside Codegen fn)
        assert _has_composite(partitioned, "tidl.nn.conv2d_relu"), (
            f"Expected conv2d_relu composite. Found composites: "
            f"{_find_composites_in_module(partitioned)}"
        )

    def test_conv_relu_pool_partitioned(self):
        """Conv + relu + pool chain (via TVMScript) should produce TIDL subgraph(s)."""

        @I.ir_module
        class ConvReluPoolModel:
            @R.function
            def main(x: R.Tensor((1, 3, 32, 32), "float32")):
                with R.dataflow():
                    w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                    y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                    y = R.nn.relu(y)
                    y = R.nn.max_pool2d(y, pool_size=[2, 2], strides=[2, 2])
                    R.output(y)
                return y

        partitioned = partition_for_tidl(ConvReluPoolModel)

        assert _has_main_function(partitioned)
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl >= 1, (
            f"Expected TIDL subgraph(s), got {n_tidl}"
        )

        # Should have both conv2d_relu and max_pool2d composites
        composites = _find_composites_in_module(partitioned)
        assert "tidl.nn.conv2d_relu" in composites, (
            f"Expected conv2d_relu composite. Found: {composites}"
        )
        assert "tidl.nn.max_pool2d" in composites, (
            f"Expected max_pool2d composite. Found: {composites}"
        )

    def test_two_conv_merged(self):
        """Two consecutive conv-relu layers should be in TIDL subgraphs."""
        mod = _export_and_bind(
            TwoConvModel,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        partitioned = partition_for_tidl(mod)

        assert _has_main_function(partitioned)
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl >= 1, (
            f"Expected TIDL subgraph(s), got {n_tidl}"
        )


class TestTIDLConstraintChecks:
    """Test that constraint check functions correctly reject unsupported ops."""

    def test_large_kernel_rejected(self):
        """Conv2d with kernel > 7 should NOT be partitioned to TIDL."""

        class LargeKernelConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2D(
                    in_channels=3, out_channels=16, kernel_size=9,
                    stride=1, padding=4, bias=False
                )

            def main(self, x):
                return self.conv1(x)

        mod = _export_and_bind(
            LargeKernelConv,
            {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")},
        )
        partitioned = partition_for_tidl(mod)

        # The conv2d with kernel=9 should NOT be in a TIDL subgraph
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl == 0, (
            f"Large-kernel conv should not be offloaded, but found "
            f"{n_tidl} TIDL subgraphs"
        )

    def test_unequal_strides_rejected(self):
        """Conv2d with unequal H/W strides should be rejected."""

        @I.ir_module
        class UnequalStrideModel:
            @R.function
            def main(x: R.Tensor((1, 3, 32, 32), "float32")):
                with R.dataflow():
                    w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                    y = R.nn.conv2d(
                        x, w,
                        strides=[1, 2],
                        padding=[1, 1, 1, 1],
                    )
                    R.output(y)
                return y

        partitioned = partition_for_tidl(UnequalStrideModel)

        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl == 0, (
            f"Unequal-stride conv should not be offloaded, but found "
            f"{n_tidl} TIDL subgraphs"
        )


class TestTIDLPartitionMixed:
    """Test partitioning with a mix of supported and unsupported ops."""

    def test_unsupported_op_not_in_tidl(self):
        """Softmax (not a TIDL pattern) should remain in main function."""

        @I.ir_module
        class ConvSoftmaxModel:
            @R.function
            def main(x: R.Tensor((1, 3, 32, 32), "float32")):
                with R.dataflow():
                    w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                    y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                    y = R.nn.relu(y)
                    y = R.nn.softmax(y, axis=1)
                    R.output(y)
                return y

        partitioned = partition_for_tidl(ConvSoftmaxModel)

        # TIDL should have captured conv+relu
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl >= 1, "Conv+relu should be in a TIDL subgraph"

        # main function should still exist (softmax stays there)
        assert _has_main_function(partitioned)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
