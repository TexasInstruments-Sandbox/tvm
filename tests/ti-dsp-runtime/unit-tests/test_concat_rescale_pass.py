"""Unit tests for FuseQDQToC7xConcat's channel-axis rescale composite.

Pure Relax IR-level tests, no hardware or DSP build needed -- the convention of
test_concat_sigmoid_pass.py.

  dq(x1,s1,z1)
  dq(x2,s2,z2)   -> concat(axis=1) -> q(so,zo)   -> c7x_int8_concat_rescale
  dq(x3,s3,z3)

The kernel writes slot i at channel offset C0+...+C(i-1), so its slots must
follow the concat's tuple, whatever order the dequantizes are bound in.
"""

import re

import pytest

import tvm
from tvm import relax

T = relax.TensorType

_CHANNELS = (4, 8, 16)
_SCALES = (0.125, 0.25, 0.5)


def _build(dq_order):
    """Concat x0, x1, x2 in that order, binding their dequantizes in dq_order."""
    xs = [relax.Var(f"x{i}", T((1, c, 2, 2), "int8")) for i, c in enumerate(_CHANNELS)]
    bb = relax.BlockBuilder()
    with bb.function("main", xs, attrs={"num_input": len(xs)}):
        with bb.dataflow():
            dqs = {}
            for i in dq_order:
                dqs[i] = bb.emit(
                    relax.op.dequantize(
                        xs[i], relax.const(_SCALES[i], "float32"), relax.const(0, "int8")
                    )
                )
            cat = bb.emit(relax.op.concat([dqs[0], dqs[1], dqs[2]], axis=1))
            q = bb.emit(
                relax.op.quantize(
                    cat, relax.const(0.75, "float32"), relax.const(0, "int8"), out_dtype="int8"
                )
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


@pytest.mark.parametrize("dq_order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_slots_follow_the_concat(dq_order):
    mod = tvm.relax.transform.FuseQDQToC7xConcat()(_build(dq_order))
    text = mod.script()
    (args,) = re.findall(r'T\.call_extern\("int32", "c7x_int8_concat_rescale",([^\n]*)\)', text)
    # Per slot: data, C_i, s_i, z_i. The output scale is the last float.
    floats = [float(v) for v in re.findall(r"T\.float32\(([-\d.eE]+)\)", args)]
    assert floats[:3] == pytest.approx(_SCALES)
    channels = [int(v) for v in re.findall(r"\.data, (\d+), T\.float32", args)]
    assert channels[:3] == list(_CHANNELS)
    (call,) = re.findall(r"R\.call_tir\(\w*concat_rescale\w*, \(([^)]*)\)", mod["main"].script())
    assert [a.strip() for a in call.split(",")][:3] == ["x0", "x1", "x2"]
