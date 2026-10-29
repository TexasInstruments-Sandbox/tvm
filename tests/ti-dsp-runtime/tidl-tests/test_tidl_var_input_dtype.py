#!/usr/bin/env python
"""Test for Class 4: Variable/dynamic input dtype allowlist.

Relay's TidlParseTVM calls isLayerIODataTypesSupported() after parsing every
layer, which rejects unknown types (-1) and enforces per-layer allowlists for
unsupported integer types (int32/int64/uint32/uint64).  Gather data cannot be
int64; TopK values/output cannot be unsupported; EltWise Mod allows int32/int64
only; Cast/Slice/Reshape/Transpose/Concat/Split/Shape/Size/Tile/LogicalOp/
NonZero are permitted.

Relax's TIDL_relaxAllowNode has NO equivalent check -- it only calls
TIDL_relaxIsLayerIONumSupported (buffer count) and tidlCheckAllowlistingConstraints.
The generic tidlConstraint* backstops do not cover this class (they're
layer-attribute checks, not data-type allowlists).

These tests assert that such cases are declined at partition time (Python gate)
or at least at import time (C++ gate)."""

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


class TestVariableInputDtype:
    """Test that unsupported integer dtypes on variable inputs are declined."""

    def test_take_int64_data_declined(self):
        """take with int64 data tensor should be declined.

        Relay's isLayerIODataTypesSupported explicitly rejects int64/uint64
        on the data input (index 0) for GatherLayer/GatherNDLayer/
        GatherElementsLayer.
        """
        # Use a Var for data (variable input), not a Constant
        data = relax.Var("x", relax.TensorType((1, 8, 16, 16), "int64"))
        indices = relax.Var("indices", relax.TensorType((4,), "int32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [data, indices]):
            with bb.dataflow():
                out = bb.emit(relax.op.take(data, indices, axis=1))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        # Should NOT offload -- int64 data is unsupported for Gather
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "take with int64 data tensor should not be offloaded"
        )

    def test_take_uint64_data_declined(self):
        """take with uint64 data tensor should be declined."""
        data = relax.Var("x", relax.TensorType((1, 8, 16, 16), "uint64"))
        indices = relax.Var("indices", relax.TensorType((4,), "int32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [data, indices]):
            with bb.dataflow():
                out = bb.emit(relax.op.take(data, indices, axis=1))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "take with uint64 data tensor should not be offloaded"
        )

    def test_take_int32_data_declined(self):
        """take with int32 data tensor is declined -- Relax is stricter here.

        Relay's isLayerIODataTypesSupported allowlists int32 specifically
        for GatherLayer's data input. Relax has no per-op allowlist of that
        kind: every _check_* in patterns.py gates data dtype through the
        single blanket _check_dtype, backed by
        _TIDL_SUPPORTED_DTYPES = {float32, int8, int16, uint8}, which does
        not include int32. So this is declined -- more conservative than
        Relay, not a parity bug (matches Task 3.2's recorded conclusion for
        this defect class: no fix needed, the Python dtype gate subsumes
        the per-op allowlist Relay needed).
        """
        data = relax.Var("x", relax.TensorType((1, 8, 16, 16), "int32"))
        indices = relax.Var("indices", relax.TensorType((4,), "int32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [data, indices]):
            with bb.dataflow():
                out = bb.emit(relax.op.take(data, indices, axis=1))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "take with int32 data is declined by the blanket dtype allowlist"
        )

    def test_topk_values_output_cannot_be_int64(self):
        """An int64-output topk cannot be constructed via ret_type='values'.

        relax.op.topk's values output dtype always mirrors the input dtype
        (there is no separate out_dtype parameter), so int64 input is the
        only way to get an int64 values output -- and that case is already
        covered by test_topk_int64_input_declined below (declined, because
        the int64 *input* fails _check_dtype). This test instead confirms
        the float32 baseline is unaffected: ret_type='values' with a
        float32 input still offloads.
        """
        x = relax.Var("x", relax.TensorType((1, 8, 16, 16), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit(relax.op.topk(x, k=1, axis=-1, ret_type="values"))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "topk(ret_type='values') with float32 input still offloads"
        )

    def test_topk_int64_input_declined(self):
        """topk with int64 input should be declined (values output would be int64).

        Relay rejects unsupported dtype on TopK input (varTensorIndices[0]).
        """
        x = relax.Var("x", relax.TensorType((1, 8, 16, 16), "int64"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit(relax.op.topk(x, k=1, axis=-1, ret_type="values"))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "topk with int64 input should not be offloaded"
        )

    def test_eltwise_mod_int32_declined(self):
        """EltWise Mod with int32 inputs is declined -- Relax is stricter here.

        Relay permits int32/int64 specifically for Mod via its op-specific
        allowlist. Relax's _check_eltwise (patterns.py) routes through the
        same blanket _check_dtype / _TIDL_SUPPORTED_DTYPES = {float32, int8,
        int16, uint8} as every other op, which excludes int32, so this is
        declined. Same conclusion as test_take_int32_data_declined above:
        more conservative than Relay, not a parity gap.
        """
        x = relax.Var("x", relax.TensorType((1, 8, 16, 16), "int32"))
        y = relax.Var("y", relax.TensorType((1, 8, 16, 16), "int32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x, y]):
            with bb.dataflow():
                out = bb.emit(relax.op.mod(x, y))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "eltwise mod with int32 inputs is declined by the blanket dtype allowlist"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])