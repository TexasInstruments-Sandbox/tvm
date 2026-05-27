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

# All tests in this file are quick (pure IR, no hardware, no .so)
pytestmark = [pytest.mark.quick, pytest.mark.core]

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
            "tidl.nn.relu",
            "tidl.nn.softmax",
            "tidl.add",
            "tidl.multiply",
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


class TestTIDLBatchNormFolding:
    """Test that batch_norm is folded into conv2d before partitioning."""

    @pytest.fixture()
    def resnet18_mod(self):
        """Import torchvision ResNet-18 into Relax with params bound."""
        torch = pytest.importorskip("torch")
        tv_resnet = pytest.importorskip("torchvision.models.resnet")
        from_exported_program = pytest.importorskip(
            "tvm.relax.frontend.torch"
        ).from_exported_program

        model = tv_resnet.resnet18(weights=None).eval()
        example = (torch.randn(1, 3, 224, 224),)
        with torch.no_grad():
            exported = torch.export.export(model, example)
            mod = from_exported_program(exported, keep_params_as_input=True)

        mod, params = relax.frontend.detach_params(mod)
        func_params = dict(zip(mod["main"].params[1:], params["main"]))
        mod = relax.transform.BindParams("main", func_params)(mod)
        return mod

    @staticmethod
    def _count_ops(mod, op_name):
        """Count occurrences of a Relax op in the module."""
        count = 0

        def _walk(expr):
            nonlocal count
            if isinstance(expr, relax.Call) and hasattr(expr.op, "name"):
                if expr.op.name == op_name:
                    count += 1
                for arg in expr.args:
                    _walk(arg)
            elif isinstance(expr, relax.SeqExpr):
                for block in expr.blocks:
                    for binding in block.bindings:
                        _walk(binding.value)
            elif isinstance(expr, relax.Function):
                _walk(expr.body)
            elif isinstance(expr, relax.Tuple):
                for f in expr.fields:
                    _walk(f)
            elif isinstance(expr, relax.TupleGetItem):
                _walk(expr.tuple_value)

        for _gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                _walk(func)
        return count

    def test_bn_folded_after_prepare(self, resnet18_mod):
        """After prepare(), no batch_norm ops should remain."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        compiler = TIDLOffloadCompiler()
        prepared = compiler.prepare(resnet18_mod)

        bn_count = self._count_ops(prepared, "relax.nn.batch_norm")
        assert bn_count == 0, (
            f"Expected 0 batch_norm ops after prepare(), found {bn_count}"
        )

    def test_resnet18_partition_has_conv2d_bias(self, resnet18_mod):
        """Partitioned ResNet-18 should have conv2d_bias* composites."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        compiler = TIDLOffloadCompiler()
        prepared = compiler.prepare(resnet18_mod)
        partitioned = compiler.partition(prepared)

        composites = _find_composites_in_module(partitioned)
        has_bias = any("conv2d_bias" in c for c in composites)
        assert has_bias, (
            f"Expected conv2d_bias* composites (from folded BN). "
            f"Found: {composites}"
        )

    def test_resnet18_few_subgraphs(self, resnet18_mod):
        """ResNet-18 with BN folding should produce few TIDL subgraphs."""
        from tvm.relax.backend.tidl import TIDLOffloadCompiler

        compiler = TIDLOffloadCompiler()
        prepared = compiler.prepare(resnet18_mod)
        partitioned = compiler.partition(prepared)

        n_tidl = _count_tidl_subgraphs(partitioned)
        # ResNet-18 has ~20 conv layers; without BN folding we'd get ~16
        # tiny subgraphs. With folding, conv+add+relu fuse together and
        # adjacent composites merge, so we expect far fewer subgraphs.
        assert n_tidl <= 5, (
            f"Expected <= 5 TIDL subgraphs with BN folding, got {n_tidl}"
        )
        assert n_tidl >= 1, (
            f"Expected at least 1 TIDL subgraph, got {n_tidl}"
        )


def _count_codegen_subgraphs(mod, codegen="tidl"):
    """Count functions with a specific Codegen attribute."""
    return sum(
        1 for _gv, func in mod.functions.items()
        if isinstance(func, relax.Function)
        and func.attrs
        and func.attrs.get("Codegen") == codegen
    )


class TestNonCompositeBridgeCycle:
    """Regression tests for MergeCompositeFunctions cycle via non-composite bridge.

    Tests use hand-crafted composite functions (Composite="tidl.*") fed directly
    into MergeCompositeFunctions, avoiding TIDL pattern-matching complexity.

    Topology under test:
        backbone_composite (tidl)
              |              |
        bridge (non-comp)   skip (direct arg)
              |              |
        detection_composite (tidl, takes bridge output AND backbone skip directly)

    Without the fix in UpdateGroupDependencies, detection merges into backbone,
    creating a cycle when the bridge later feeds back. With the fix, the two
    groups remain separate.
    """

    @staticmethod
    def _make_bridge_module():
        """Build a pre-fused module with the SPPF bridge topology.

        After FuseOpsByPattern the graph looks like:
          main:
            lv  = backbone_composite(x, w1)     # Composite="tidl.nn.conv2d"
            lv2 = nn.relu(lv)                   # non-composite bridge
            gv  = detection_composite(lv2, lv)  # Composite="tidl.nn.conv2d_relu"
                                                # takes lv (backbone) as direct arg
        """

        @I.ir_module
        class BridgeModule:
            @R.function(private=True)
            def backbone_comp(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                R.func_attr({"Composite": "tidl.nn.conv2d", "Primitive": True})
                with R.dataflow():
                    gv: R.Tensor((1, 16, 8, 8), "float32") = R.nn.conv2d(
                        x, w, strides=[1, 1], padding=[1, 1, 1, 1]
                    )
                    R.output(gv)
                return gv

            @R.function(private=True)
            def detection_comp(
                bridge: R.Tensor((1, 16, 8, 8), "float32"),
                skip: R.Tensor((1, 16, 8, 8), "float32"),
                w: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                R.func_attr({"Composite": "tidl.nn.conv2d_relu", "Primitive": True})
                with R.dataflow():
                    tmp: R.Tensor((1, 16, 8, 8), "float32") = R.nn.conv2d(
                        bridge, w, strides=[1, 1], padding=[1, 1, 1, 1]
                    )
                    gv: R.Tensor((1, 16, 8, 8), "float32") = R.add(tmp, skip)
                    R.output(gv)
                return gv

            @R.function
            def main(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w1: R.Tensor((16, 16, 3, 3), "float32"),
                w2: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                cls = BridgeModule
                with R.dataflow():
                    # backbone composite
                    lv: R.Tensor((1, 16, 8, 8), "float32") = cls.backbone_comp(x, w1)
                    # non-composite bridge
                    lv2: R.Tensor((1, 16, 8, 8), "float32") = R.nn.relu(lv)
                    # detection composite: args are (bridge_output, backbone_output_direct, w2)
                    gv: R.Tensor((1, 16, 8, 8), "float32") = cls.detection_comp(lv2, lv, w2)
                    R.output(gv)
                return gv

        return BridgeModule

    @staticmethod
    def _make_linear_module():
        """Build a pre-fused module with a linear chain (no bridge)."""

        @I.ir_module
        class LinearModule:
            @R.function(private=True)
            def conv_a(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                R.func_attr({"Composite": "tidl.nn.conv2d", "Primitive": True})
                with R.dataflow():
                    gv: R.Tensor((1, 16, 8, 8), "float32") = R.nn.conv2d(
                        x, w, strides=[1, 1], padding=[1, 1, 1, 1]
                    )
                    R.output(gv)
                return gv

            @R.function(private=True)
            def conv_b(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                R.func_attr({"Composite": "tidl.nn.conv2d", "Primitive": True})
                with R.dataflow():
                    gv: R.Tensor((1, 16, 8, 8), "float32") = R.nn.conv2d(
                        x, w, strides=[1, 1], padding=[1, 1, 1, 1]
                    )
                    R.output(gv)
                return gv

            @R.function
            def main(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w1: R.Tensor((16, 16, 3, 3), "float32"),
                w2: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                cls = LinearModule
                with R.dataflow():
                    lv: R.Tensor((1, 16, 8, 8), "float32") = cls.conv_a(x, w1)
                    gv: R.Tensor((1, 16, 8, 8), "float32") = cls.conv_b(lv, w2)
                    R.output(gv)
                return gv

        return LinearModule

    def test_bridge_keeps_subgraphs_separate(self):
        """Non-composite bridge must prevent backbone and detection from merging."""
        mod = self._make_bridge_module()
        merged = relax.transform.MergeCompositeFunctions()(mod)

        n = _count_codegen_subgraphs(merged, "tidl")
        assert n == 2, (
            f"Expected 2 TIDL subgraphs (backbone + detection), got {n}. "
            "If got 1, UpdateGroupDependencies reverse-propagation fix is missing."
        )

    def test_bridge_module_has_main(self):
        """Merged module must still have a main function."""
        mod = self._make_bridge_module()
        merged = relax.transform.MergeCompositeFunctions()(mod)
        assert _has_main_function(merged)

    def test_linear_chain_still_merges(self):
        """Direct chain (no bridge) must still collapse into 1 subgraph.

        The fix must not over-restrict: adjacent composites with no non-composite
        op between them should still merge.
        """
        mod = self._make_linear_module()
        merged = relax.transform.MergeCompositeFunctions()(mod)

        n = _count_codegen_subgraphs(merged, "tidl")
        assert n == 1, (
            f"Linear chain (no bridge) should merge into 1 TIDL subgraph, got {n}. "
            "Fix over-restricts merging of directly chained composites."
        )

    def test_multi_bridge_sppf_pattern(self):
        """SPPF-like topology: backbone → 3 non-composite pools → detection skip."""

        @I.ir_module
        class SPPFModule:
            @R.function(private=True)
            def backbone_comp(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                R.func_attr({"Composite": "tidl.nn.conv2d", "Primitive": True})
                with R.dataflow():
                    gv: R.Tensor((1, 16, 8, 8), "float32") = R.nn.conv2d(
                        x, w, strides=[1, 1], padding=[1, 1, 1, 1]
                    )
                    R.output(gv)
                return gv

            @R.function(private=True)
            def detection_comp(
                bridge: R.Tensor((1, 16, 8, 8), "float32"),
                skip: R.Tensor((1, 16, 8, 8), "float32"),
                w: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                R.func_attr({"Composite": "tidl.nn.conv2d_relu", "Primitive": True})
                with R.dataflow():
                    tmp: R.Tensor((1, 16, 8, 8), "float32") = R.nn.conv2d(
                        bridge, w, strides=[1, 1], padding=[1, 1, 1, 1]
                    )
                    gv: R.Tensor((1, 16, 8, 8), "float32") = R.add(tmp, skip)
                    R.output(gv)
                return gv

            @R.function
            def main(
                x: R.Tensor((1, 16, 8, 8), "float32"),
                w1: R.Tensor((16, 16, 3, 3), "float32"),
                w2: R.Tensor((16, 16, 3, 3), "float32"),
            ) -> R.Tensor((1, 16, 8, 8), "float32"):
                cls = SPPFModule
                with R.dataflow():
                    backbone: R.Tensor((1, 16, 8, 8), "float32") = cls.backbone_comp(x, w1)
                    # Three non-composite bridge ops (SPPF maxpool chain)
                    p1: R.Tensor((1, 16, 8, 8), "float32") = R.nn.relu(backbone)
                    p2: R.Tensor((1, 16, 8, 8), "float32") = R.nn.relu(p1)
                    p3: R.Tensor((1, 16, 8, 8), "float32") = R.add(p2, p1)
                    # detection: takes p3 (through bridge) AND backbone (direct skip)
                    gv: R.Tensor((1, 16, 8, 8), "float32") = cls.detection_comp(p3, backbone, w2)
                    R.output(gv)
                return gv

        merged = relax.transform.MergeCompositeFunctions()(SPPFModule)

        n = _count_codegen_subgraphs(merged, "tidl")
        assert n == 2, (
            f"SPPF multi-bridge pattern must produce 2 TIDL subgraphs, got {n}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
