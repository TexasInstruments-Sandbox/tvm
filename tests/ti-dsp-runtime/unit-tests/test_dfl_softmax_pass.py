"""Unit tests for FuseQDQToC7xActivation's dfl_softmax composite.

Pure Relax IR-level tests, no hardware or DSP build needed -- mirrors
test_movement_pass.py's convention.

Covers the pattern:
  dq(x[B,A,K,N]) -> permute_dims(axes=[0,2,1,3]) -> softmax(axis=1) -> q
      -> c7x_int8_dfl_softmax, output [B,K,A,N]

This is YOLOv8's DFL head shape; see ti_fuse_qdq_c7x_activation.py's
_make_dfl_softmax_pattern docstring for why the permute is folded into the
kernel rather than matched as a separate op.
"""

import re

import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo


def _run_pass(mod):
    return tvm.relax.transform.FuseQDQToC7xActivation()(mod)


def _build_dfl_model(B, A, K, N, d_scale, d_zp, o_scale, o_zp, axes=(0, 2, 1, 3), axis=1):
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((B, A, K, N), "int8"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            dq = bb.emit(
                relax.op.dequantize(x, relax.const(d_scale, "float32"), relax.const(d_zp, "int8"))
            )
            perm = bb.emit(relax.op.permute_dims(dq, axes=list(axes)))
            sm = bb.emit(relax.op.nn.softmax(perm, axis=axis))
            q = bb.emit(
                relax.op.quantize(
                    sm, relax.const(o_scale, "float32"), relax.const(o_zp, "int8"), out_dtype="int8"
                )
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestDFLSoftmax:
    def test_fires_and_emits_call_extern(self):
        mod = _build_dfl_model(1, 4, 16, 32, d_scale=0.1, d_zp=0, o_scale=0.0078125, o_zp=0)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_dfl_softmax" in text
        assert "R.permute_dims" not in text
        assert "R.nn.softmax" not in text
        assert "R.dequantize" not in text
        assert "R.quantize" not in text

    def test_params_match_formula(self):
        mod = _build_dfl_model(1, 3, 8, 20, d_scale=0.07, d_zp=2, o_scale=0.02, o_zp=-1)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        m = re.search(
            r'T\.call_extern\("int32", "c7x_int8_dfl_softmax", [^,]+, [^,]+, '
            r"(\d+), (\d+), (\d+), (\d+), (-?\d+), T\.float32\(([-\d.eE]+)\), "
            r"(-?\d+), T\.float32\(([-\d.eE]+)\)\)",
            text,
        )
        assert m is not None, f"call_extern args not found in:\n{text}"
        b, a, k, n, zx, sx, zy, sy = m.groups()
        assert (int(b), int(a), int(k), int(n)) == (1, 3, 8, 20)
        assert int(zx) == 2
        assert float(sx) == pytest.approx(0.07)
        assert int(zy) == -1
        assert float(sy) == pytest.approx(0.02)

    def test_output_shape_is_permuted(self):
        """Output shape must be [B,K,A,N], matching the real post-permute
        layout the downstream conv (DFL's integral) expects."""
        mod = _build_dfl_model(1, 4, 16, 32, d_scale=0.1, d_zp=0, o_scale=0.0078125, o_zp=0)
        new_mod = _run_pass(mod)
        main = new_mod["main"]
        out_sinfo = main.ret_struct_info
        assert [int(s) for s in out_sinfo.shape] == [1, 16, 4, 32]

    def test_wrong_permute_axes_declines_to_fuse(self):
        """A permute that doesn't swap axes 1/2 must fall through unchanged
        -- the kernel only implements this one axis mapping."""
        mod = _build_dfl_model(
            1, 4, 16, 32, d_scale=0.1, d_zp=0, o_scale=0.0078125, o_zp=0, axes=(0, 1, 3, 2)
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_dfl_softmax" not in text

    def test_wrong_softmax_axis_declines_to_fuse(self):
        """A softmax over a different axis than the permuted K dim must
        fall through unchanged."""
        mod = _build_dfl_model(
            1, 4, 16, 32, d_scale=0.1, d_zp=0, o_scale=0.0078125, o_zp=0, axis=2
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_dfl_softmax" not in text

    def test_non_constant_scale_declines_to_fuse(self):
        """Regression guard: non-compile-time scale must decline, not mis-fuse."""
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 4, 16, 32), "int8"))
        s_in = relax.Var("s_in", TensorStructInfo((), "float32"))
        with bb.function("main", [x, s_in], attrs={"num_input": 2}):
            with bb.dataflow():
                dq = bb.emit(
                    relax.op.dequantize(x, relax.const(0.1, "float32"), relax.const(0, "int8"))
                )
                perm = bb.emit(relax.op.permute_dims(dq, axes=[0, 2, 1, 3]))
                sm = bb.emit(relax.op.nn.softmax(perm, axis=1))
                o_scale = bb.emit(relax.op.multiply(s_in, relax.const(1.0, "float32")))
                q = bb.emit(
                    relax.op.quantize(sm, o_scale, relax.const(0, "int8"), out_dtype="int8")
                )
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_dfl_softmax" not in text
