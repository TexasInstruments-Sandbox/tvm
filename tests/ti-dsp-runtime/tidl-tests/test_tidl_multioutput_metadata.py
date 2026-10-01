#!/usr/bin/env python
"""Test for Class 13: Slice/tuple output metadata extraction.

Relay fix 3523e53df part 3: "Fix tuple output metadata extraction to check
call->checked_type() as TupleTypeNode and extract shapes from tuple fields
directly."

Relax's multi-output ops: split, take, topk (all produce Tuple output).
Need to verify the C++ parsers handle tuple checked_type correctly for
output naming/indexing. The Python _check_* functions don't directly
control this; it's a C++ parser behavior.

These tests verify that multi-output ops partition and the output naming
works (i.e., no crash/assertion in C++ parser)."""

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
    x = relax.Var("x", relax.TensorStructInfo((1, channel_dim, 16, 16), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.split(x, indices_or_sections, axis=axis))
            # Use all outputs to ensure tuple is preserved
            gv = bb.emit_output(relax.Tuple(out))
        bb.emit_func_output(gv)
    return bb.get()


def _build_take_model(data, indices_shape, axis, indices_dtype="int32"):
    bb = relax.BlockBuilder()
    inputs = [v for v in (data,) if isinstance(v, relax.Var)]
    indices = relax.Var("indices", relax.TensorStructInfo(indices_shape, indices_dtype))
    inputs.append(indices)
    with bb.function("main", inputs):
        with bb.dataflow():
            out = bb.emit(relax.op.take(data, indices, axis=axis))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _build_topk_model(shape, axis, k=1, ret_type="values", dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.topk(x, k=k, axis=axis, ret_type=ret_type))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


class TestMultiOutputMetadata:
    """Test multi-output ops (split, take, topk) tuple metadata handling."""

    def test_split_multi_output_partitioned(self):
        """split into 4 outputs produces Tuple -- should partition without crash."""
        mod = _build_split_model(indices_or_sections=4, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split with tuple output should offload"
        )

    def test_split_2_outputs_partitioned(self):
        """split into 2 outputs (single explicit index) -- should partition."""
        mod = _build_split_model(indices_or_sections=[32], channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "split 2 outputs should offload"
        )

    def test_take_multi_output_partitioned(self):
        """take with ret_type='both' (values + indices) produces Tuple -- test ret_type='values' only for now.

        relax.op.take only returns values. topk has ret_type option.
        """
        data = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        mod = _build_take_model(data, indices_shape=(4,), axis=3)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "take should offload"
        )

    def test_topk_values_only_partitioned(self):
        """topk with ret_type='values' (single output) -- should partition."""
        mod = _build_topk_model(shape=(1, 8, 16, 16), axis=-1, ret_type="values")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "topk values-only should offload"
        )

    def test_topk_both_outputs_partitioned(self):
        """topk with ret_type='both' produces Tuple(values, indices) -- offloads.

        Confirms the C++ parser's tuple checked_type handling (the subject
        of Relay fix 3523e53df) carries over correctly to Relax's two-output
        topk: no crash and the composite partitions as one subgraph.
        """
        mod = _build_topk_model(shape=(1, 8, 16, 16), axis=-1, ret_type="both")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "topk with ret_type='both' (two outputs) currently offloads"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])