#!/usr/bin/env python
"""Test for Class 7: Cast dtype validation (roundMode/saturate) -- FINDING: NOT APPLICABLE.

Relay commits:
- 34d7b98a2: Added default values for roundMode and saturate in Cast parser
- b842fe4bd: Moving cast dtype check to operator-specific parser
- 7eb991d4c: Removing cast dtype specific check from parser code

Relay's cast parser validates dtype AND sets defaults for roundMode/saturate:
  roundMode = TIDL_CastRoundUp (default)
  saturate = 1 (required by constraint)

FINDING: Relax's relax.op.astype has NO round_mode or saturate parameters
(signature: astype(x, dtype)). These are TIDL-internal CastLayer fields that
Relay's parser defaulted, but Relax's astype op doesn't expose them.

Therefore, this class does NOT apply to Relax -- there are no attributes to
validate. The C++ parser tidl_parse_relax_cast.cpp unconditionally passes,
and the Python _check_cast validates dtype only (which is correct).

This test file documents the finding and verifies basic cast still works."""

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


class TestCastAttributes:
    """Test cast (astype) -- round_mode/saturate not applicable to Relax."""

    def test_cast_basic_partitioned(self):
        """Basic astype float32 -> int8 should partition."""
        x = relax.Var("x", relax.TensorType((1, 8, 16, 16), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit(relax.op.astype(x, "int8"))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "Basic astype float32->int8 should offload"
        )

    def test_cast_int8_to_float32_partitioned(self):
        """astype int8 -> float32 should partition."""
        x = relax.Var("x", relax.TensorType((1, 8, 16, 16), "int8"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit(relax.op.astype(x, "float32"))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "astype int8->float32 should offload"
        )

    def test_cast_uint8_to_int16_partitioned(self):
        """astype uint8 -> int16 should partition."""
        x = relax.Var("x", relax.TensorType((1, 8, 16, 16), "uint8"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit(relax.op.astype(x, "int16"))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "astype uint8->int16 should offload"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])