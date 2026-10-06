#!/usr/bin/env python
"""Regression tests for the deny-by-default pattern checks.

``tidl.quantize``, ``tidl.dequantize`` and ``tidl.cast`` match with bare
wildcards, and the C++ parsers behind them
(``tidlParseRelax/tidl_parse_relax_{quantize,dequantize,cast}.cpp``) have no
denial path of their own -- whatever the Python ``_check_*`` callback in
``patterns.py`` accepts is offloaded unconditionally.  ``_check_quantize``,
``_check_dequantize`` and ``_check_cast`` used to be unconditional
``return True`` stubs; they now validate both the input and output dtype
against TIDL's supported set.  These tests assert the decline side of that
change: an unsupported dtype on either end must leave the op unpartitioned.

Each also has an accept-side sibling using a supported dtype, so a failure
here points at the dtype check rather than at unrelated pattern-matching
breakage.
"""

import subprocess
import sys

import pytest

from tvm import relax, tir
from tvm.relax.backend.tidl import partition_for_tidl

# All tests in this file are quick (pure IR, no hardware, no .so)
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


# ---------------------------------------------------------------------------
# Model builders
#
# quantize/dequantize out_dtype and input dtype are both attribute-checked
# by the op itself (src/relax/op/tensor/qdq.cc), so "unsupported dtype" here
# means a dtype the op accepts but TIDL's Python check does not: uint16 and
# int32 for quantize, float16 and int32 for dequantize.
# ---------------------------------------------------------------------------


def _build_quantize_model(data_dtype="float32", out_dtype="int8"):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), data_dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            scale = relax.const(0.1, "float32")
            zp = relax.const(0, "int8")
            out = bb.emit(relax.op.quantize(x, scale, zp, axis=1, out_dtype=out_dtype))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _build_dequantize_model(data_dtype="int8", out_dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), data_dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            scale = relax.const(0.1, "float32")
            zp = relax.const(0, "int8")
            out = bb.emit(relax.op.dequantize(x, scale, zp, axis=1, out_dtype=out_dtype))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _build_cast_model(data_dtype="float32", out_dtype="int8"):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), data_dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.astype(x, out_dtype))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuantizeDenyByDefault:
    def test_supported_dtype_partitioned(self):
        """float32 -> int8 quantize: both ends supported, should offload."""
        mod = _build_quantize_model(data_dtype="float32", out_dtype="int8")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_unsupported_out_dtype_declined(self):
        """uint16 output is valid for relax.quantize but not TIDL's dtype set."""
        mod = _build_quantize_model(data_dtype="float32", out_dtype="uint16")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "quantize with unsupported out_dtype='uint16' should not be offloaded"
        )

    def test_unsupported_input_dtype_declined(self):
        """int32 input is valid for relax.quantize but not TIDL's dtype set."""
        mod = _build_quantize_model(data_dtype="int32", out_dtype="int8")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "quantize with unsupported input dtype='int32' should not be offloaded"
        )


class TestDequantizeDenyByDefault:
    def test_supported_dtype_partitioned(self):
        """int8 -> float32 dequantize: both ends supported, should offload."""
        mod = _build_dequantize_model(data_dtype="int8", out_dtype="float32")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_unsupported_out_dtype_declined(self):
        """float16 output is valid for relax.dequantize but not TIDL's dtype set."""
        mod = _build_dequantize_model(data_dtype="int8", out_dtype="float16")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "dequantize with unsupported out_dtype='float16' should not be offloaded"
        )

    def test_unsupported_input_dtype_declined(self):
        """int32 input is valid for relax.dequantize but not TIDL's dtype set."""
        mod = _build_dequantize_model(data_dtype="int32", out_dtype="float32")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "dequantize with unsupported input dtype='int32' should not be offloaded"
        )


class TestCastDenyByDefault:
    def test_supported_dtype_partitioned(self):
        """float32 -> int8 cast: both ends supported, should offload."""
        mod = _build_cast_model(data_dtype="float32", out_dtype="int8")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_unsupported_out_dtype_declined(self):
        """astype accepts int64, but it is outside TIDL's dtype set."""
        mod = _build_cast_model(data_dtype="float32", out_dtype="int64")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "cast with unsupported out_dtype='int64' should not be offloaded"
        )

    def test_unsupported_input_dtype_declined(self):
        """astype accepts an int32 source, but it is outside TIDL's dtype set."""
        mod = _build_cast_model(data_dtype="int32", out_dtype="float32")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "cast with unsupported input dtype='int32' should not be offloaded"
        )


# ---------------------------------------------------------------------------
# Task 2.3: attribute validation for strided_slice / pad / concat.
#
# These three checks used to validate dtype only.  Each now rejects a case the
# C++ Relax parser mis-parses (wrong output, not a decline) rather than an
# outright missing feature -- see the docstrings on _check_strided_slice,
# _check_pad and _check_concat in patterns.py for the parser-level evidence.
# ---------------------------------------------------------------------------


def _build_strided_slice_model(axes, begin, end, strides=None, dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.strided_slice(x, axes, begin, end, strides))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _build_pad_model(pad_width, pad_mode="constant", dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.nn.pad(x, pad_width, pad_mode=pad_mode))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _build_concat_model(axis, dtype="float32"):
    shape = (1, 8, 16, 16)
    x1 = relax.Var("x1", relax.TensorStructInfo(shape, dtype))
    x2 = relax.Var("x2", relax.TensorStructInfo(shape, dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x1, x2]):
        with bb.dataflow():
            out = bb.emit(relax.op.concat([x1, x2], axis=axis))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


class TestStridedSliceDenyByDefault:
    def test_single_axis_partitioned(self):
        """axes=[1], stride=1: the one shape TIDL's SliceLayer supports."""
        mod = _build_strided_slice_model(axes=[1], begin=[0], end=[4])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_multi_axis_declined(self):
        """Two axes at once: the C++ parser reads only field 0 of each tuple
        and would silently slice just the first axis instead of declining."""
        mod = _build_strided_slice_model(axes=[1, 2], begin=[0, 0], end=[4, 8])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "multi-axis strided_slice should not be offloaded"
        )

    def test_negative_stride_declined(self):
        """A reversed slice: the C++ parser only declines stride > 1, so a
        negative stride would pass through and be executed as stride=1."""
        mod = _build_strided_slice_model(axes=[1], begin=[0], end=[4], strides=[-1])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "negative-stride strided_slice should not be offloaded"
        )

    def test_stride_gt_one_declined(self):
        """stride=2: still declined (matches the C++ parser's own check)."""
        mod = _build_strided_slice_model(axes=[1], begin=[0], end=[4], strides=[2])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "stride > 1 strided_slice should not be offloaded"
        )


class TestPadDenyByDefault:
    def test_hw_only_pad_partitioned(self):
        """Padding only H and W (N,C before/after all zero): what TIDL supports."""
        mod = _build_pad_model(pad_width=[0, 0, 0, 0, 1, 1, 2, 2])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_channel_pad_declined(self):
        """Non-zero C padding: the Relax parser never inspects N/C at all, so
        this would be silently dropped rather than declined or applied."""
        mod = _build_pad_model(pad_width=[0, 0, 1, 1, 0, 0, 0, 0])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "pad with non-zero channel padding should not be offloaded"
        )

    def test_batch_pad_declined(self):
        """Non-zero N padding: same silent-drop gap as the channel case."""
        mod = _build_pad_model(pad_width=[1, 1, 0, 0, 0, 0, 0, 0])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "pad with non-zero batch padding should not be offloaded"
        )

    def test_non_constant_mode_declined(self):
        """"edge" mode: valid for relax.nn.pad, unsupported by TIDL's PadLayer."""
        mod = _build_pad_model(pad_width=[0, 0, 0, 0, 1, 1, 1, 1], pad_mode="edge")
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "non-constant pad_mode should not be offloaded"
        )


class TestConcatDenyByDefault:
    def test_channel_axis_partitioned(self):
        """axis=1 (channel): the common case, e.g. multi-head assembly."""
        mod = _build_concat_model(axis=1)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_none_axis_on_1d_partitioned(self):
        """axis=None requires 1-D inputs (InferStructInfoConcat enforces this),
        where "flatten-then-concat" and "concat along axis 0" are the same
        operation -- so this is not the mis-parse it might look like and
        should still offload."""
        x1 = relax.Var("x1", relax.TensorStructInfo((8,), "float32"))
        x2 = relax.Var("x2", relax.TensorStructInfo((8,), "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x1, x2]):
            with bb.dataflow():
                out = bb.emit(relax.op.concat([x1, x2], axis=None))
                gv = bb.emit_output(out)
            bb.emit_func_output(gv)
        mod = bb.get()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_batch_axis_declined(self):
        """axis=0 on a 4-D tensor: TIDL's generic ConcatLayer constraint
        declines batch-axis concat above rank 3; matched here for an early exit."""
        mod = _build_concat_model(axis=0)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "batch-axis concat on a 4D tensor should not be offloaded"
        )


# ---------------------------------------------------------------------------
# Task 2.3 continued: split / reshape / permute_dims.
#
# Unlike strided_slice/pad, split's gap is a real out-of-bounds write (not
# just a mis-parse): TIDL's fixed-size slicePoints[TIDL_NUM_OUT_BUFS + 1]
# array overflows past 16 outputs, in both the Relax and Relay C++ parsers,
# with no generic backstop protecting the Relax path (see _check_split's
# docstring in patterns.py). reshape and permute_dims turned up no mis-parse
# or memory-safety gap on inspection -- only an early-exit rank <= 6 ceiling
# that TIDL's generic backstop already enforces unconditionally.
# ---------------------------------------------------------------------------


def _build_split_model(indices_or_sections, axis=1, channel_dim=64, dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo((1, channel_dim, 16, 16), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.split(x, indices_or_sections, axis=axis))
            gv = bb.emit_output(out[0])
        bb.emit_func_output(gv)
    return bb.get()


def _build_reshape_model(target_shape, dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.reshape(x, target_shape))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _build_permute_dims_model(shape, axes=None, dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.permute_dims(x, axes=axes))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


class TestSplitDenyByDefault:
    def test_few_sections_partitioned(self):
        """4-way split: well within TIDL_NUM_OUT_BUFS=16."""
        mod = _build_split_model(indices_or_sections=4, channel_dim=64)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_too_many_sections_declined(self):
        """17-way equal split: one more output than slicePoints[17] can hold
        without an out-of-bounds write in the C++ importer."""
        mod = _build_split_model(indices_or_sections=17, channel_dim=68)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "split into 17 sections should not be offloaded"
        )

    def test_too_many_explicit_indices_declined(self):
        """19 explicit cut points -> 20 outputs, same overflow via the other
        indices_or_sections branch of the C++ parser."""
        mod = _build_split_model(
            indices_or_sections=list(range(5, 100, 5)), channel_dim=100
        )
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "split into 20 outputs via explicit indices should not be offloaded"
        )


class TestReshapeDenyByDefault:
    def test_low_rank_target_partitioned(self):
        """Flatten to 2D: the common pre-FC-layer case."""
        mod = _build_reshape_model(target_shape=[1, 2048])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_high_rank_target_declined(self):
        """7-D target shape: above TIDL's rank-6 allowlisting ceiling."""
        mod = _build_reshape_model(target_shape=[1, 1, 1, 1, 1, 1, -1])
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "reshape to a 7-D target shape should not be offloaded"
        )


def _build_unary_model(struct_info, op):
    x = relax.Var("x", struct_info)
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(op(x))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def _symbolic_batch():
    return relax.TensorStructInfo([tir.Var("n", "int64"), 8, 16, 16], "float32")


def _rank_only():
    return relax.TensorStructInfo(dtype="float32", ndim=4)


class TestUnknownShapeFailsClosed:
    """_check_rank / the exact-rank guards decline when the shape cannot be
    established, instead of deferring to a C++ importer that has no denial
    path.  A symbolic dimension used to raise TypeError out of the callback
    (``int()`` on a tir.Var)."""

    OPS = {
        "permute_dims": lambda x: relax.op.permute_dims(x, axes=[0, 2, 3, 1]),
        "max_pool2d": lambda x: relax.op.nn.max_pool2d(x, (2, 2), (2, 2)),
        "relu": relax.op.nn.relu,
        "softmax": lambda x: relax.op.nn.softmax(x, axis=1),
    }

    @pytest.mark.parametrize("op_name", sorted(OPS))
    def test_static_shape_partitioned(self, op_name):
        """Accept-side sibling: same op, fully static shape."""
        static = relax.TensorStructInfo((1, 8, 16, 16), "float32")
        mod = _build_unary_model(static, self.OPS[op_name])
        assert _count_tidl_subgraphs(partition_for_tidl(mod)) == 1

    @pytest.mark.parametrize("op_name", sorted(OPS))
    def test_symbolic_dim_declined(self, op_name):
        mod = _build_unary_model(_symbolic_batch(), self.OPS[op_name])
        assert _count_tidl_subgraphs(partition_for_tidl(mod)) == 0, (
            f"{op_name} with a symbolic batch dim should not be offloaded"
        )

    @pytest.mark.parametrize("op_name", sorted(OPS))
    def test_unknown_shape_declined(self, op_name):
        mod = _build_unary_model(_rank_only(), self.OPS[op_name])
        assert _count_tidl_subgraphs(partition_for_tidl(mod)) == 0, (
            f"{op_name} with a known rank but no shape should not be offloaded"
        )


class TestPermuteDimsDenyByDefault:
    def test_4d_partitioned(self):
        """Ordinary NCHW transpose."""
        mod = _build_permute_dims_model(shape=(1, 8, 16, 16))
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_7d_declined(self):
        """7-D input: above TIDL's rank-6 allowlisting ceiling."""
        mod = _build_permute_dims_model(shape=(1, 1, 1, 1, 2, 3, 4))
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "permute_dims on a 7-D tensor should not be offloaded"
        )


# ---------------------------------------------------------------------------
# Task 2.3 continued: take / topk.
#
# take's gap is a mis-attribution bug (constant DATA read as constant
# INDICES, an embedding-table-like pattern), analogous to strided_slice/pad
# rather than to split's memory-safety class. topk's gap is a real,
# unconditional (not modelType-gated) hardware axis restriction that the
# dtype-only check ignored entirely. See _check_take / _check_topk in
# patterns.py for the C++ parser and constraint-file evidence.
# ---------------------------------------------------------------------------


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


def _build_topk_model(shape, axis, dtype="float32"):
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit(relax.op.topk(x, k=1, axis=axis, ret_type="values"))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


class TestTakeDenyByDefault:
    def test_last_axis_partitioned(self):
        """Gather along the last (W) axis: dims after it are trivially empty."""
        data = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        mod = _build_take_model(data, indices_shape=(4,), axis=3)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_constant_data_declined(self):
        """Embedding-table pattern: data is a Constant, indices is variable.
        tidl_parse_relax_take.cpp would read this constant as if it were the
        indices, since it takes whichever Constant it finds first."""
        data = relax.const([[0.0] * 16] * 8, "float32")
        mod = _build_take_model(data, indices_shape=(4,), axis=0)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "take with a constant data operand should not be offloaded"
        )

    def test_channel_axis_partitioned(self):
        """Gather along the channel axis of a 4-D tensor with H,W > 1 after
        it: NOT declined. (An earlier version of this check misread
        tidl_constraint_gather.h as requiring dims after the axis to be
        singleton; the real, much narrower constraint checks dims *before*
        the axis and is gated behind conditions this check doesn't attempt
        to replicate.) This mirrors test_take in test_tidl_layer_offload.py,
        which is the regression this test guards against."""
        data = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        mod = _build_take_model(data, indices_shape=(4,), axis=1)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_non_1d_indices_declined(self):
        """2-D indices with a non-singleton leading dim: TIDL's GatherLayer
        only supports effectively-1D indices."""
        data = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
        mod = _build_take_model(data, indices_shape=(2, 4), axis=3)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "take with non-1D indices should not be offloaded"
        )


class TestTopKDenyByDefault:
    def test_width_axis_partitioned(self):
        """axis=-1 (width): the one axis family TIDL's TopKLayer supports."""
        mod = _build_topk_model(shape=(1, 8, 16, 16), axis=-1)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 1

    def test_channel_axis_declined(self):
        """axis=1 (channel): TIDL's TopKLayer only supports width/height."""
        mod = _build_topk_model(shape=(1, 8, 16, 16), axis=1)
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) == 0, (
            "topk along the channel axis should not be offloaded"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_patterns_module_import_exits_cleanly():
    """Holding a reference to ``patterns`` used to segfault the interpreter at
    exit: register_patterns did not record FusionPattern names, so the atexit
    cleanup left 74 patterns with Python check callbacks in libtvm's static
    table, released after Python had shut down."""
    proc = subprocess.run(
        [sys.executable, "-c", "from tvm.relax.backend.tidl import patterns"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"exit code {proc.returncode}\n{proc.stderr[-500:]}"
