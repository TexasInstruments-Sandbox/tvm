#!/usr/bin/env python
"""Test for Class 12: Split single-index parsing.

Relay fix 5e1c79395: "Fix split parsing for the case where there is a single
split index".  In Relay, `indices_or_sections` as `Box<int64_t>` (single
integer) means equal sections; as `Array<Box<int64_t>>` with one element means
explicit single cut index.

Relax's _check_split handles both forms and caps total outputs at 16.
Need to verify single-index list case works correctly."""

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


class TestSplitSingleIndex:
    """Test split with single-index indices_or_sections."""

    def test_split_equal_sections_int_partitioned(self):
        """split(sections=4) -- single int = 4 equal sections, should partition."""
        mod = _build_split_model(indices_or_sections=4, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(sections=4) should offload"
        )

    def test_split_single_explicit_index_partitioned(self):
        """split(indices_or_sections=[32]) -- single cut at 32, produces 2 outputs.

        This is the case Relay fixed: Array with one element = explicit index.
        Should produce 2 outputs (0..32, 32..64).
        """
        mod = _build_split_model(indices_or_sections=[32], channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(indices_or_sections=[32]) should offload (2 outputs)"
        )

    def test_split_two_explicit_indices_partitioned(self):
        """split(indices_or_sections=[16, 48]) -- two cuts, 3 outputs."""
        mod = _build_split_model(indices_or_sections=[16, 48], channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split(indices_or_sections=[16, 48]) should offload (3 outputs)"
        )

    def test_split_sections_17_declined(self):
        """split(sections=17) -- 17 outputs exceeds TIDL_NUM_OUT_BUFS=16, declined."""
        mod = _build_split_model(indices_or_sections=17, channel_dim=68)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "split into 17 sections should not be offloaded"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])