#!/usr/bin/env python
"""Test for Class 6: Layout assumptions / pseudo-3D conv.

Relay fix 1980204aa: "Fix TVM parser to support pseudo-3D convolutions
(3D conv with depth=1)".  The ONNX/TVM parser handles kernel_shape=[1, K, K]
as a 2D convolution.  Need to verify Relax's tidl_parse_relax_conv.cpp handles
this the same way, or at least declines cleanly if not supported.

Note: relax.nn.conv2d infers kernel size from weight tensor shape (last 2 dims).
There's no explicit kernel_size parameter. The "pseudo-3D" case in Relay
meant the ONNX parser receiving a 5D weight (OIDHW) with depth=1 and treating
it as 2D. Relax's conv2d only accepts 4D weight (OIHW)."""

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


class TestPseudo3DConv:
    """Test pseudo-3D convolution handling."""

    def test_conv2d_standard_2d_partitioned(self):
        """Standard 2D conv2d with 4D weight (OIHW) should partition."""
        x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        weight = relax.Var("w", relax.TensorStructInfo((16, 8, 3, 3), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x, weight]):
            with bb.dataflow():
                out = bb.emit(relax.op.nn.conv2d(x, weight))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "Standard 2D conv2d should offload"
        )

    def test_conv2d_with_dilation_partitioned(self):
        """Standard (non-depthwise) conv2d with dilation=(2, 2) offloads.

        tidl_parse_relax_conv.cpp reads dilation from Conv2DAttrs into
        convParams.dilationH/W unconditionally (no decline on a non-1
        value). The generic tidlConstraintConv backstop's dilation checks
        (tidl_constraint_conv.h) are part of the depthwise kernel/stride
        whitelist, which this standard conv (in_channels=8 != groups) never
        enters, so nothing declines it either.
        """
        x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        weight = relax.Var("w", relax.TensorStructInfo((16, 8, 3, 3), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x, weight]):
            with bb.dataflow():
                out = bb.emit(relax.op.nn.conv2d(x, weight, dilation=(2, 2)))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()

        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1, (
            "standard conv2d with dilation=(2, 2) currently offloads"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])