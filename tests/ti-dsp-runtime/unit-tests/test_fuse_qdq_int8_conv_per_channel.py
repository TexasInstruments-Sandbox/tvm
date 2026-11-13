"""FuseQDQToInt8Conv2D with per-channel activation qparams.

Pure Relax IR-level tests, no hardware or DSP build needed. FuseQDQToInt8Conv2D is
the scalar int8 path for a quantized conv MMALIB declines (e.g. a stride-2
depthwise conv whose padded height is odd). C7xMMAQuantizer gives a depthwise
conv's input, and a conv output read by a depthwise block, one scale per channel.
The rewrite must fold those scales into the right output channels or leave the conv
alone; each rewritten graph is run on LLVM against the graph before the pass.
"""

import numpy as np
import pytest

import tvm
from tvm import relax

T = relax.TensorType


def _module(
    c_in, c_out, groups, d_scale, d_zp, o_scale, o_zp, *, bias=False, relu=False, seed=0
):
    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, (c_out, c_in // groups, 3, 3)).astype("int8")
    w_scale = rng.uniform(0.002, 0.01, c_out).astype("float32")
    x = relax.Var("x", T((1, c_in, 7, 7), "int8"))

    def qparams(scale, zp):
        scale = np.asarray(scale, "float32")
        zp = np.broadcast_to(np.asarray(zp, "int8"), scale.shape).copy()
        return relax.const(scale), relax.const(zp), 1 if scale.ndim else -1

    bb = relax.BlockBuilder()
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            ds, dz, d_axis = qparams(d_scale, d_zp)
            dq = bb.emit(relax.op.dequantize(x, ds, dz, d_axis))
            w_dq = bb.emit(
                relax.op.dequantize(
                    relax.const(w), relax.const(w_scale), relax.const(np.zeros(c_out, "int8")), 0
                )
            )
            y = bb.emit(relax.op.nn.conv2d(dq, w_dq, strides=2, padding=1, groups=groups))
            if bias:
                b = rng.uniform(-0.5, 0.5, (1, c_out, 1, 1)).astype("float32")
                y = bb.emit(relax.op.add(y, relax.const(b)))
            if relu:
                y = bb.emit(relax.op.nn.relu(y))
            os_, oz, o_axis = qparams(o_scale, o_zp)
            out = bb.emit_output(relax.op.quantize(y, os_, oz, o_axis, out_dtype="int8"))
        bb.emit_func_output(out)
    return bb.finalize()


def _run(mod, x):
    ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"](tvm.runtime.tensor(x)).numpy()


def _fused(mod):
    return "int32" in tvm.relax.transform.FuseQDQToInt8Conv2D()(mod).script()


def _check_matches(mod, c_in):
    """The rewritten graph gives the codes of the graph before the pass. The two
    compute the requantize in a different float order, so a value next to a
    rounding boundary may land one code away."""
    fused = tvm.relax.transform.FuseQDQToInt8Conv2D()(mod)
    assert "int32" in fused.script()
    x = np.random.default_rng(1).integers(-128, 128, (1, c_in, 7, 7)).astype("int8")
    ref, got = _run(mod, x).astype(int), _run(fused, x).astype(int)
    assert np.abs(ref - got).max() <= 1
    assert (ref != got).mean() < 0.01


C = 6
PER_CHANNEL = np.linspace(0.01, 0.2, C)  # channel scales 20x apart


@pytest.mark.parametrize("bias,relu", [(False, False), (True, False), (True, True)])
def test_depthwise_per_channel_input(bias, relu):
    mod = _module(C, C, C, PER_CHANNEL, 0, 0.05, 0, bias=bias, relu=relu)
    _check_matches(mod, C)


def test_depthwise_per_channel_input_and_output():
    _check_matches(_module(C, C, C, PER_CHANNEL, 3, PER_CHANNEL * 2, -2, bias=True), C)


def test_per_channel_output():
    """A conv output read by a depthwise block: per-channel output scales."""
    _check_matches(_module(4, C, 1, 0.05, 0, PER_CHANNEL, 0, bias=True), 4)


def test_per_tensor_still_fuses():
    _check_matches(_module(C, C, C, 0.05, 0, 0.05, 0), C)


def test_per_channel_input_of_a_regular_conv_declines():
    """Each output channel sums over input channels of different scales, which one
    output-channel scale can't express."""
    assert not _fused(_module(C, C, 1, PER_CHANNEL, 0, 0.05, 0))


def test_channel_multiplier_above_one_declines():
    """groups == C_in, but two output channels per input channel."""
    assert not _fused(_module(C, 2 * C, C, PER_CHANNEL, 0, 0.05, 0))


def test_differing_zero_points_decline():
    zps = np.arange(C) - 3
    assert not _fused(_module(C, C, C, PER_CHANNEL, zps, 0.05, 0))
