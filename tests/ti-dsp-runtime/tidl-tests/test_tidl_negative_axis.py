#!/usr/bin/env python
"""Test for Class 8: Negative axis indexing in split.

Relay fix 02f41e3e7: "Fix Split operator parsing for negative axis indexing"
- Normalize negative axis values before metadata access to prevent UB
- Initialize input/output tensor dimensions from TVM metadata
- Calculate equal-section split slicePoints correctly using normalized axis

Relax's _check_split (patterns.py) rejects >16 outputs but doesn't explicitly
handle negative axis in the split attribute. The underlying relax.op.split
may normalize at op-construction time; need to verify."""

import pytest
from tvm import relax
from tvm.relax.backend.tidl import partition_for_tidl


pytestmark = [pytest.mark.quick, pytest.mark.core]


def _count_tidl_subgraphs(mod):
    """Count functions with Codegen='tidl'."""
    return len(
        [
            gv
            for gv, func in mod.functions.items()
            if isinstance(func, relax.Function)
            and func.attrs
            and func.attrs.get("Codegen") == "tidl"
        ]
    )


def _build_split_model(indices_or_sections, axis=1, channel_dim=64, dtype="float32"):
    x = relax.Var("x", relax.TensorType((1, channel_dim, 16, 16), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.split(x, indices_or_sections, axis=axis))
            gv = bb.emit_output(out[0])
        bb.emit_func_output(gv)
    return bb.get()


class TestNegativeAxis:
    """Test split with negative axis indexing."""

    def test_split_axis_minus_1_partitioned(self):
        """split(axis=-1) on 4D tensor (axis=3, width) should partition.

        -1 maps to the last dimension (width in NCHW).
        """
        mod = _build_split_model(indices_or_sections=4, axis=-1, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(axis=-1) on 4D should offload"
        )

    def test_split_axis_minus_2_partitioned(self):
        """split(axis=-2) on 4D tensor (axis=2, height) should partition.

        -2 maps to height in NCHW.
        """
        mod = _build_split_model(indices_or_sections=4, axis=-2, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(axis=-2) on 4D should offload"
        )

    def test_split_axis_minus_3_partitioned(self):
        """split(axis=-3) on 4D tensor (axis=1, channel) should partition.

        -3 maps to channel in NCHW.
        """
        mod = _build_split_model(indices_or_sections=4, axis=-3, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(axis=-3) on 4D should offload"
        )

    def test_split_axis_minus_4_partitioned(self):
        """split(axis=-4) on 4D tensor (axis=0, batch) -- currently offloads.

        -4 maps to batch. Unlike concat (Task 2.3's _check_concat rejects a
        batch-axis concat on a >3-D tensor, matching the generic
        tidlConstraintConcat backstop), split has no batch-axis-specific
        decline anywhere in patterns.py or the generic allowlisting
        backstop, so this currently offloads. Pins down the current
        behavior; tightening this is a product decision (offload coverage
        vs. safety), not something to change silently if it regresses.
        """
        mod = _build_split_model(indices_or_sections=2, axis=-4, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(axis=-4) currently offloads (no batch-axis decline exists for split)"
        )

    def test_split_positive_axis_1_partitioned(self):
        """split(axis=1) -- positive axis control test."""
        mod = _build_split_model(indices_or_sections=4, axis=1, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(axis=1) should offload"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])