"""Unit tests for EliminateQDQTransparent pass.

Verifies that dequantize→op→quantize patterns with matching scales
are correctly eliminated, letting ops run directly on int8 data.
No hardware needed — tests the pass at the Relax IR level.
"""

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo


def _build_qdq_model(op_fn, input_shape, scale, zp, **op_kwargs):
    """Build a minimal model: dequantize(x) → op → quantize(out)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo(input_shape, "int8"))
    s = relax.Constant(np.array(scale, dtype=np.float32))
    z = relax.Constant(np.array(zp, dtype=np.int8))

    with bb.function("main", [x]):
        with bb.dataflow():
            dq = bb.emit(relax.op.dequantize(x, s, z))
            op_out = bb.emit(op_fn(dq, **op_kwargs))
            result = bb.emit(relax.op.quantize(op_out, s, z, out_dtype="int8"))
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize()


def _run_pass(mod):
    """Run the EliminateQDQTransparent pass."""
    return tvm.relax.transform.EliminateQDQTransparent()(mod)


def _has_dequantize(mod):
    """Check if any function in the module contains a dequantize op."""
    text = mod.script()
    return "dequantize" in text


def _has_quantize(mod):
    """Check if any function in the module contains a quantize op."""
    text = mod.script()
    return "quantize" in text


class TestMaxPool2d:
    def test_eliminates_qdq(self):
        mod = _build_qdq_model(
            lambda x: relax.op.nn.max_pool2d(
                x, pool_size=(3, 3), strides=(2, 2), padding=(1, 1)
            ),
            input_shape=(1, 64, 112, 112),
            scale=0.00935,
            zp=0,
        )
        new_mod = _run_pass(mod)
        assert not _has_dequantize(new_mod)
        assert not _has_quantize(new_mod)
        assert "max_pool2d" in new_mod.script()

    def test_preserves_when_scales_differ(self):
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 64, 112, 112), "int8"))
        s_in = relax.Constant(np.array(0.01, dtype=np.float32))
        s_out = relax.Constant(np.array(0.02, dtype=np.float32))
        z = relax.Constant(np.array(0, dtype=np.int8))

        with bb.function("main", [x]):
            with bb.dataflow():
                dq = bb.emit(relax.op.dequantize(x, s_in, z))
                pool = bb.emit(
                    relax.op.nn.max_pool2d(
                        dq, pool_size=(3, 3), strides=(2, 2), padding=(1, 1)
                    )
                )
                result = bb.emit(relax.op.quantize(pool, s_out, z, out_dtype="int8"))
                bb.emit_output(result)
            bb.emit_func_output(result)
        mod = bb.finalize()

        new_mod = _run_pass(mod)
        assert _has_dequantize(new_mod)
        assert _has_quantize(new_mod)


class TestReshape:
    def test_eliminates_qdq(self):
        mod = _build_qdq_model(
            lambda x: relax.op.reshape(x, (1, 3136)),
            input_shape=(1, 64, 7, 7),
            scale=0.05,
            zp=0,
        )
        new_mod = _run_pass(mod)
        assert not _has_dequantize(new_mod)
        assert not _has_quantize(new_mod)
        assert "reshape" in new_mod.script()


class TestPermuteDims:
    def test_eliminates_qdq(self):
        mod = _build_qdq_model(
            lambda x: relax.op.permute_dims(x, axes=[0, 2, 3, 1]),
            input_shape=(1, 64, 56, 56),
            scale=0.03,
            zp=0,
        )
        new_mod = _run_pass(mod)
        assert not _has_dequantize(new_mod)
        assert not _has_quantize(new_mod)
        assert "permute_dims" in new_mod.script()


class TestRelu:
    def test_eliminates_qdq_symmetric(self):
        """Relu with zp=0 (symmetric) should be eliminated."""
        mod = _build_qdq_model(
            lambda x: relax.op.nn.relu(x),
            input_shape=(1, 64, 56, 56),
            scale=0.05,
            zp=0,
        )
        new_mod = _run_pass(mod)
        assert not _has_dequantize(new_mod)
        assert not _has_quantize(new_mod)
        assert "relu" in new_mod.script()

    def test_preserves_qdq_asymmetric(self):
        """Relu with zp!=0 (asymmetric) should NOT be eliminated."""
        mod = _build_qdq_model(
            lambda x: relax.op.nn.relu(x),
            input_shape=(1, 64, 56, 56),
            scale=0.05,
            zp=-128,
        )
        new_mod = _run_pass(mod)
        assert _has_dequantize(new_mod)
        assert _has_quantize(new_mod)


class TestConcat:
    @pytest.mark.skip(reason="DPL can't capture variable-arity dequantize inputs in tuple")
    def test_eliminates_qdq_same_scales(self):
        """Concat with same scale/zp on all inputs should be eliminated."""
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 32, 56, 56), "int8"))
        y = relax.Var("y", TensorStructInfo((1, 32, 56, 56), "int8"))
        s = relax.Constant(np.array(0.05, dtype=np.float32))
        z = relax.Constant(np.array(0, dtype=np.int8))

        with bb.function("main", [x, y]):
            with bb.dataflow():
                dq_x = bb.emit(relax.op.dequantize(x, s, z))
                dq_y = bb.emit(relax.op.dequantize(y, s, z))
                cat = bb.emit(relax.op.concat([dq_x, dq_y], axis=1))
                result = bb.emit(relax.op.quantize(cat, s, z, out_dtype="int8"))
                bb.emit_output(result)
            bb.emit_func_output(result)
        mod = bb.finalize()

        new_mod = _run_pass(mod)
        assert not _has_dequantize(new_mod)
        assert not _has_quantize(new_mod)
        assert "concat" in new_mod.script()

    def test_preserves_qdq_different_scales(self):
        """Concat with different scales should NOT be eliminated."""
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 32, 56, 56), "int8"))
        y = relax.Var("y", TensorStructInfo((1, 32, 56, 56), "int8"))
        s1 = relax.Constant(np.array(0.05, dtype=np.float32))
        s2 = relax.Constant(np.array(0.10, dtype=np.float32))
        s_out = relax.Constant(np.array(0.05, dtype=np.float32))
        z = relax.Constant(np.array(0, dtype=np.int8))

        with bb.function("main", [x, y]):
            with bb.dataflow():
                dq_x = bb.emit(relax.op.dequantize(x, s1, z))
                dq_y = bb.emit(relax.op.dequantize(y, s2, z))
                cat = bb.emit(relax.op.concat([dq_x, dq_y], axis=1))
                result = bb.emit(relax.op.quantize(cat, s_out, z, out_dtype="int8"))
                bb.emit_output(result)
            bb.emit_func_output(result)
        mod = bb.finalize()

        new_mod = _run_pass(mod)
        assert _has_dequantize(new_mod)
