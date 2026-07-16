"""Unit tests for FuseInputNormalizeQuantize pass.

Verifies that the traced torchvision transform_input chain
(take/expand_dims/multiply/add x3 -> concat -> quantize) folds into a
single call_extern to c7x_int8_quantize_rgb with correctly-derived
per-channel (inv_scale, offset) parameters. Pure Relax IR-level test, no
hardware or DSP build needed -- mirrors test_qdq_transparent_ops.py's
convention (build a small model, run the pass, inspect mod.script()).
"""

import re

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo

_KERNEL = "c7x_int8_quantize_rgb"


def _build_transform_input_model(shape, affine, scale, zp, channel_order=(0, 1, 2)):
    """dq-free float32 model: 3x [take(x,c)->expand_dims->mul(a_c)->add(b_c)]
    -> concat(axis=1) -> quantize(scale, zp).

    affine: list of (a_c, b_c) pairs indexed by channel.
    channel_order: the order branches are built/concatenated in, as a
    permutation of channel indices -- lets tests confirm the pass doesn't
    assume branch position == channel index.
    """
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo(shape, "float32"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            branches = []
            for c in channel_order:
                a_c, b_c = affine[c]
                idx = relax.const(np.array(c, dtype="int64"))
                taken = bb.emit(relax.op.take(x, idx, axis=1))
                expanded = bb.emit(relax.op.expand_dims(taken, axis=1))
                mul = bb.emit(relax.op.multiply(expanded, relax.const(a_c, "float32")))
                add = bb.emit(relax.op.add(mul, relax.const(b_c, "float32")))
                branches.append(add)
            cat = bb.emit(relax.op.concat(branches, axis=1))
            s = relax.const(scale, "float32")
            z = relax.const(zp, "int8")
            q = bb.emit(relax.op.quantize(cat, s, z, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_pass(mod):
    return tvm.relax.transform.FuseInputNormalizeQuantize()(mod)


def _extract_call_extern_args(mod):
    """Pull the c7x_int8_quantize_rgb call_extern's literal args out of
    mod.script() text -- avoids depending on TIR AST internals for a
    simple param-value check."""
    text = mod.script()
    m = re.search(rf'T\.call_extern\("int32", "{_KERNEL}", [^,]+, [^,]+, (.+)\)', text)
    assert m is not None, f"{_KERNEL} call_extern not found in:\n{text}"
    arg_strs = [a.strip() for a in m.group(1).split(",")]

    def _val(s):
        s = s.strip()
        m2 = re.match(r"T\.(?:int32|int64|float32)\(([-\d.eE]+)\)$", s)
        return float(m2.group(1)) if m2 else float(s)

    return [_val(a) for a in arg_strs]


class TestFuseInputNormalizeQuantize:
    def test_folds_to_single_call_extern(self):
        affine = [(0.458, -0.03), (0.448, -0.088), (0.45, -0.188)]
        mod = _build_transform_input_model((1, 3, 8, 8), affine, scale=0.01, zp=-3)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert _KERNEL in text
        for op in ("take", "expand_dims", "concat"):
            assert f"R.{op}" not in text and f"T.{op}" not in text

    @pytest.mark.parametrize(
        "channel_order",
        [(0, 1, 2), (2, 1, 0)],
        ids=["in_order", "reverse_order"],
    )
    def test_derived_params_match_formula(self, channel_order):
        """inv_scale_c = a_c/scale, offset_c = b_c/scale + zp, exactly --
        regardless of the order branches are built/concatenated in (must
        map each branch's actual take-index to the right (a_c, b_c), not
        positional order)."""
        affine = [(0.458, -0.03), (0.448, -0.088), (0.45, -0.188)]
        scale, zp = 0.01, -3
        mod = _build_transform_input_model(
            (1, 3, 8, 8), affine, scale, zp, channel_order=channel_order
        )
        new_mod = _run_pass(mod)
        args = _extract_call_extern_args(new_mod)
        # [N, HW, is0, off0, is1, off1, is2, off2]
        assert args[0] == 1
        assert args[1] == 64
        inv_scale = 1.0 / scale
        for c, (a_c, b_c) in enumerate(affine):
            is_c, off_c = args[2 + 2 * c], args[3 + 2 * c]
            assert is_c == pytest.approx(a_c * inv_scale, rel=1e-5)
            assert off_c == pytest.approx(b_c * inv_scale + zp, rel=1e-5)

    def test_does_not_fire_on_intermediate_quantize(self):
        """The quantize's operand must be the raw model input Var, not an
        intermediate activation -- same Var-boundary guard as
        FuseInputQuantize._check_quantize."""
        affine = [(0.458, -0.03), (0.448, -0.088), (0.45, -0.188)]
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 3, 8, 8), "float32"))
        with bb.function("main", [x], attrs={"num_input": 1}):
            with bb.dataflow():
                # An upstream op stands between the function input and the
                # transform_input chain -- x is no longer the raw input Var
                # by the time it reaches `take`.
                pre = bb.emit(relax.op.add(x, relax.const(0.0, "float32")))
                branches = []
                for c, (a_c, b_c) in enumerate(affine):
                    idx = relax.const(np.array(c, dtype="int64"))
                    taken = bb.emit(relax.op.take(pre, idx, axis=1))
                    expanded = bb.emit(relax.op.expand_dims(taken, axis=1))
                    mul = bb.emit(relax.op.multiply(expanded, relax.const(a_c, "float32")))
                    add = bb.emit(relax.op.add(mul, relax.const(b_c, "float32")))
                    branches.append(add)
                cat = bb.emit(relax.op.concat(branches, axis=1))
                q = bb.emit(
                    relax.op.quantize(
                        cat,
                        relax.const(0.01, "float32"),
                        relax.const(-3, "int8"),
                        out_dtype="int8",
                    )
                )
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        assert _KERNEL not in new_mod.script()

    def test_does_not_fire_on_non_scalar_output_scale(self):
        """Per-channel output scale is not handled by this pass, same
        guard as FuseInputQuantize."""
        affine = [(0.458, -0.03), (0.448, -0.088), (0.45, -0.188)]
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 3, 8, 8), "float32"))
        with bb.function("main", [x], attrs={"num_input": 1}):
            with bb.dataflow():
                branches = []
                for c, (a_c, b_c) in enumerate(affine):
                    idx = relax.const(np.array(c, dtype="int64"))
                    taken = bb.emit(relax.op.take(x, idx, axis=1))
                    expanded = bb.emit(relax.op.expand_dims(taken, axis=1))
                    mul = bb.emit(relax.op.multiply(expanded, relax.const(a_c, "float32")))
                    add = bb.emit(relax.op.add(mul, relax.const(b_c, "float32")))
                    branches.append(add)
                cat = bb.emit(relax.op.concat(branches, axis=1))
                s = relax.const(np.array([0.01, 0.02, 0.03], dtype="float32"))
                z = relax.const(np.array([-3, -3, -3], dtype="int8"))
                q = bb.emit(relax.op.quantize(cat, s, z, axis=1, out_dtype="int8"))
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        assert _KERNEL not in new_mod.script()
