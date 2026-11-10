"""Unit tests for FuseQDQToC7xActivation's handling of shared activation outputs.

Pure Relax IR-level tests, no hardware or DSP build needed (the convention of
test_silu_f32out_pass.py: build a small graph, run the pass, inspect the script).

An activation whose float result is read both by its own quantize and by an op
left in float. The quantized PyTorch graph gives that reader the float
activation, rounded once at the reader's output scale. The pass must not hand it
dequantize(int8 activation), which rounds, and clips, at the other reader's scale:

  concat -> quantize(s_c):   a second c7x_int8_<act> call at s_c feeds the concat
  multiply(gate) -> quantize: c7x_int8_<act>_gate_multiply from the int8 input
"""

import re

import pytest

import tvm
from tvm import relax

T = relax.TensorType


def _c(v, dtype):
    return relax.const(v, dtype)


def _dq(bb, x, s, z=0):
    return bb.emit(relax.op.dequantize(x, _c(s, "float32"), _c(z, "int8")))


def _q(bb, x, s, z=0):
    return bb.emit(relax.op.quantize(x, _c(s, "float32"), _c(z, "int8"), out_dtype="int8"))


def _silu(bb, x, s=0.05, z=0):
    dq = _dq(bb, x, s, z)
    return bb.emit(relax.op.multiply(dq, bb.emit(relax.op.sigmoid(dq))))


def _hardswish(bb, x, commuted=False, s=0.05, z=0):
    """The shape from_exported_program gives aten.hardswish."""
    dq = _dq(bb, x, s, z)
    added = bb.emit(relax.op.add(dq, _c(3.0, "float32")))
    clip_lo = bb.emit(relax.op.clip(added, 0.0, float("inf")))
    clip_hi = bb.emit(relax.op.clip(clip_lo, float("-inf"), 6.0))
    mul = bb.emit(relax.op.multiply(clip_hi, dq) if commuted else relax.op.multiply(dq, clip_hi))
    return bb.emit(relax.op.divide(mul, _c(6.0, "float32")))


def _module(params, body):
    bb = relax.BlockBuilder()
    with bb.function("main", params, attrs={"num_input": len(params)}):
        with bb.dataflow():
            out = bb.emit_output(body(bb))
        bb.emit_func_output(out)
    return bb.finalize()


def _calls(mod):
    """[(extern name, argument text)] for every c7x_int8_* call_extern after the pass."""
    text = tvm.relax.transform.FuseQDQToC7xActivation()(mod).script()
    return re.findall(r'T\.call_extern\("int32", "(c7x_int8_\w+)",([^\n]*)\)', text), text


def _floats(args):
    return [float(v) for v in re.findall(r"T\.float32\(([-\d.eE]+)\)", args)]


def _ints(args):
    return [int(v) for v in re.findall(r",\s*(-?\d+)(?=,|$)", args)]


# ---------------------------------------------------------------------------
# concat reader
# ---------------------------------------------------------------------------


def _silu_with_concat(concat_readers=1):
    x = relax.Var("x", T((1, 8, 4, 4), "int8"))
    y = relax.Var("y", T((1, 4, 4, 4), "int8"))

    def body(bb):
        act = _silu(bb, x)
        own = _q(bb, act, 0.01)  # e.g. a conv's input, whose scale clips the SiLU
        cat = bb.emit(relax.op.concat([act, _dq(bb, y, 0.04)], axis=1))
        outs = [own, _q(bb, cat, 0.03)]
        if concat_readers > 1:
            outs.append(bb.emit(relax.op.sum(cat)))
        return relax.Tuple(outs)

    return _module([x, y], body)


class TestConcatReader:
    def test_concat_gets_its_own_silu_at_its_scale(self):
        calls, text = _calls(_silu_with_concat())
        silu = [args for name, args in calls if name == "c7x_int8_silu"]
        assert len(silu) == 2
        # (in scale, out scale) of each call: the own reader's 0.01 and the concat's 0.03.
        assert sorted(_floats(a)[1] for a in silu) == pytest.approx([0.01, 0.03])
        assert "R.sigmoid" not in text
        # The concat reads the new call's output dequantized at the concat's 0.03;
        # nothing dequantizes the 0.01-scale result any more.
        dequantizes = [line for line in text.splitlines() if "R.dequantize(" in line]
        assert not any("0.0099999" in line for line in dequantizes)
        assert any("0.0299999" in line for line in dequantizes)
        assert "R.concat" in text

    def test_concat_with_other_readers_keeps_the_old_path(self):
        """A concat whose output is not only quantized isn't rewritten."""
        calls, _ = _calls(_silu_with_concat(concat_readers=2))
        assert [name for name, _ in calls].count("c7x_int8_silu") == 1

    def test_concat_of_two_shared_activations_copies_both(self):
        x1 = relax.Var("x1", T((1, 8, 4, 4), "int8"))
        x2 = relax.Var("x2", T((1, 4, 4, 4), "int8"))

        def body(bb):
            a1, a2 = _silu(bb, x1), _silu(bb, x2, s=0.04)
            cat = bb.emit(relax.op.concat([a1, a2], axis=1))
            return relax.Tuple([_q(bb, a1, 0.01), _q(bb, a2, 0.02), _q(bb, cat, 0.03)])

        calls, text = _calls(_module([x1, x2], body))
        silu = [args for name, args in calls if name == "c7x_int8_silu"]
        assert sorted(_floats(a)[1] for a in silu) == pytest.approx([0.01, 0.02, 0.03, 0.03])
        assert "R.sigmoid" not in text

    def test_per_channel_concat_quantize_keeps_the_old_path(self):
        """The copy would take the concat's qparams, which must then be per-tensor."""
        x = relax.Var("x", T((1, 8, 4, 4), "int8"))
        y = relax.Var("y", T((1, 4, 4, 4), "int8"))

        def body(bb):
            act = _silu(bb, x)
            cat = bb.emit(relax.op.concat([act, _dq(bb, y, 0.04)], axis=1))
            scales = relax.const([0.03] * 12, "float32")
            zps = relax.const([0] * 12, "int8")
            per_channel = bb.emit(relax.op.quantize(cat, scales, zps, axis=1, out_dtype="int8"))
            return relax.Tuple([_q(bb, act, 0.01), per_channel])

        calls, _ = _calls(_module([x, y], body))
        assert [name for name, _ in calls].count("c7x_int8_silu") == 1


# ---------------------------------------------------------------------------
# squeeze-excite multiply reader
# ---------------------------------------------------------------------------


def _se(
    act,
    gate_first=False,
    commuted_hardswish=False,
    gate_kind="sigmoid",
    own_reader=True,
    gate_shape=None,
):
    x = relax.Var("x", T((1, 8, 5, 5), "int8"))
    g = relax.Var("g", T(gate_shape or (1, 8, 1, 1), "int8"))

    def body(bb):
        a = _silu(bb, x) if act == "silu" else _hardswish(bb, x, commuted_hardswish)
        if gate_kind == "sigmoid":  # EfficientNet: a float sigmoid of the excitation
            gate = bb.emit(relax.op.sigmoid(_dq(bb, g, 0.1)))
        else:  # MobileNetV3: a dequantized int8 hardsigmoid output
            gate = _dq(bb, g, 1.0 / 255)
        mul = bb.emit(relax.op.multiply(gate, a) if gate_first else relax.op.multiply(a, gate))
        out = _q(bb, mul, 0.02)
        return relax.Tuple([_q(bb, a, 0.01), out]) if own_reader else out

    return _module([x, g], body)


class TestSqueezeExciteReader:
    @pytest.mark.parametrize("act", ["silu", "hardswish"])
    @pytest.mark.parametrize("gate_first", [False, True])
    def test_fused_kernel_replaces_the_requantized_multiply(self, act, gate_first):
        gate_kind = "sigmoid" if act == "silu" else "dequantize"
        calls, text = _calls(_se(act, gate_first=gate_first, gate_kind=gate_kind))
        names = [name for name, _ in calls]
        assert names.count(f"c7x_int8_{act}_gate_multiply") == 1
        assert names.count(f"c7x_int8_{act}") == 1  # the activation's own reader
        assert "c7x_int8_channel_scale_multiply" not in names
        # No full-size float multiply is left; only the [1,C,1,1] gate is computed in float.
        assert not re.search(r"R\.multiply\([^\n]*\n?[^\n]*R\.Tensor\(\(1, 8, 5, 5\)", text)

    def test_commuted_hardswish_chain(self):
        calls, _ = _calls(_se("hardswish", commuted_hardswish=True, gate_kind="dequantize"))
        assert [n for n, _ in calls].count("c7x_int8_hardswish_gate_multiply") == 1

    def test_kernel_arguments(self):
        calls, _ = _calls(_se("silu"))
        (args,) = [a for n, a in calls if n == "c7x_int8_silu_gate_multiply"]
        assert _floats(args) == pytest.approx([0.05, 0.02])  # input, output scale
        ints = [int(v) for v in re.findall(r"\b(\d+)\b", args.split("T.float32")[0])]
        assert ints[-3:-1] == [8, 25]  # C, H*W

    def test_unshared_activation_feeding_a_gate_also_fuses(self):
        """No own quantize reader: the composite matches without the rewrite."""
        calls, _ = _calls(_se("silu", own_reader=False))
        assert [n for n, _ in calls] == ["c7x_int8_silu_gate_multiply"]

    def test_full_size_multiply_is_not_a_gate(self):
        calls, _ = _calls(_se("silu", gate_shape=(1, 8, 5, 5)))
        assert "c7x_int8_silu_gate_multiply" not in [n for n, _ in calls]


def test_non_constant_scale_declines():
    """A runtime input scale can't be folded into the kernel call."""
    x = relax.Var("x", T((1, 8, 5, 5), "int8"))
    s = relax.Var("s", T((), "float32"))
    g = relax.Var("g", T((1, 8, 1, 1), "float32"))

    def body(bb):
        dq = bb.emit(relax.op.dequantize(x, s, _c(0, "int8")))
        a = bb.emit(relax.op.multiply(dq, bb.emit(relax.op.sigmoid(dq))))
        return relax.Tuple([_q(bb, a, 0.01), _q(bb, bb.emit(relax.op.multiply(a, g)), 0.02)])

    calls, _ = _calls(_module([x, s, g], body))
    assert calls == []
