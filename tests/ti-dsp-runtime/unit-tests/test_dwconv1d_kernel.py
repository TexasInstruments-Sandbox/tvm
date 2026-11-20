"""c7x_int8_dwconv1d and FuseQDQToC7xDwConv1d.

A symmetric QDQ depthwise conv with a 1xK kernel on a [1, C, 1, T] tensor
(QuartzNet's time-channel separable convs) is lowered by FuseQDQToInt8Conv2D
to an int32 group conv and a requantize, then by FuseQDQToC7xDwConv1d to one
c7x_int8_dwconv1d call. The kernel must give exactly what the generic path
gives: int32 accumulation, (float)acc times the float32 scale
(d_scale * w_scale) / o_scale, round half to even, clip.

The source checks need no DSP. The numeric checks run on c7x_host and c7x_dload.

Usage:
    pytest test_dwconv1d_kernel.py -v --dsp-mode=c7x_host
    pytest test_dwconv1d_kernel.py -v --dsp-mode=c7x_dload --board beagley-ai
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tvm import relax

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, compile_for_dsp, get_target_string  # noqa: E402

_TARGET = "c_static_lib -mcpu=c7x -use-cpp-api=1"

# (C, T_in, K, stride, dilation, pad_left, pad_right, per-channel input scale)
_CASES = {
    "k3": (8, 64, 3, 1, 1, 1, 1, False),
    "k33_partial_block": (12, 100, 33, 1, 1, 16, 16, False),
    "k9_stride2": (16, 70, 9, 2, 1, 4, 4, False),
    "k7_dilation2": (8, 50, 7, 1, 2, 6, 6, False),
    "k5_stride2_dilation2": (4, 41, 5, 2, 2, 4, 4, False),
    "uneven_pad": (6, 37, 4, 1, 1, 3, 0, False),
    "per_channel_input_scale": (10, 96, 11, 1, 1, 5, 5, True),
}


def _params(case, seed=0):
    C, T, K, stride, dil, pl, pr, per_channel = case
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (1, C, 1, T), dtype=np.int8)
    w = rng.integers(-128, 128, (C, 1, 1, K), dtype=np.int8)
    w[0, 0, 0, :] = 127  # one channel at the accumulator's largest magnitudes
    x_scale = (
        rng.uniform(0.01, 0.05, C).astype(np.float32)
        if per_channel
        else np.float32(rng.uniform(0.01, 0.05))
    )
    w_scale = rng.uniform(0.001, 0.01, C).astype(np.float32)
    o_scale = np.float32(rng.uniform(0.05, 0.2))
    return x, w, x_scale, w_scale, o_scale


def _build_module(case, w, x_scale, w_scale, o_scale, kernel_h=1, h=1, relu=None):
    """dequantize -> depthwise conv2d (weights dequantized) -> quantize.

    relu="input" puts a transparent ReLU (dequantize -> relu -> quantize, same
    scale) on the conv's input; relu="shared" does too and also returns the
    ReLU's output, giving it a second reader.
    """
    C, T, K, stride, dil, pl, pr, _ = case
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([1, C, h, T], "int8"))
    zero = relax.const(0, "int8")
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            xs = relax.const(x_scale, "float32")
            xz = relax.const(np.zeros(C, np.int8), "int8") if np.ndim(x_scale) else zero
            xd = bb.emit(relax.op.dequantize(x, xs, xz, axis=1))
            if relu is not None:
                r = bb.emit(relax.op.nn.relu(xd))
                rq = bb.emit(relax.op.quantize(r, xs, xz, axis=1))
                xd = bb.emit(relax.op.dequantize(rq, xs, xz, axis=1))
            wq = np.repeat(w, kernel_h, axis=2)
            wd = bb.emit(
                relax.op.dequantize(
                    relax.const(wq, "int8"),
                    relax.const(w_scale, "float32"),
                    relax.const(np.zeros(C, np.int8), "int8"),
                    axis=0,
                )
            )
            conv = bb.emit(
                relax.op.nn.conv2d(
                    xd,
                    wd,
                    strides=(1, stride),
                    padding=(0, pl, 0, pr),
                    dilation=(1, dil),
                    groups=C,
                )
            )
            out = bb.emit(relax.op.quantize(conv, relax.const(o_scale, "float32"), zero, axis=1))
            if relu == "shared":
                out = bb.emit_output(relax.Tuple([out, rq]))
            else:
                out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _reference(case, x, w, x_scale, w_scale, o_scale, relu=False):
    """The generic path's arithmetic, in numpy."""
    C, T, K, stride, dil, pl, pr, _ = case
    if relu:
        x = np.maximum(x, 0)
    xp = np.pad(x[0, :, 0, :].astype(np.int32), ((0, 0), (pl, pr)))
    t_out = (T + pl + pr - (K - 1) * dil - 1) // stride + 1
    acc = np.zeros((C, t_out), np.int32)
    for k in range(K):
        taps = xp[:, k * dil : k * dil + stride * t_out : stride]
        acc += w[:, 0, 0, k : k + 1].astype(np.int32) * taps
    d_scale = np.broadcast_to(np.asarray(x_scale, np.float32), (C,))
    scale = (d_scale * w_scale) / o_scale  # float32, as FoldConstant computes it
    y = np.clip(np.rint(acc.astype(np.float32) * scale[:, None]), -128, 127).astype(np.int8)
    return y[None, :, None, :]


def _generated_source(mod):
    with tempfile.TemporaryDirectory() as tmp:
        gen = compile_for_dsp(mod, _TARGET, output_dir=Path(tmp))
        return "\n".join(p.read_text() for p in sorted(gen.glob("lib*.c")))


@pytest.mark.quick
@pytest.mark.parametrize("name", list(_CASES))
def test_depthwise_1d_lowered_to_kernel(name):
    case = _CASES[name]
    _, w, xs, ws, os_ = _params(case)
    src = _generated_source(_build_module(case, w, xs, ws, os_))
    assert "c7x_int8_dwconv1d(" in src
    assert "group_conv2d" not in src


@pytest.mark.quick
@pytest.mark.parametrize("kernel_h,h", [(3, 8), (1, 4)], ids=["kxk", "h4"])
def test_other_depthwise_left_alone(kernel_h, h):
    """A KxK kernel, or a 1xK kernel on H > 1, stays on the generic path."""
    case = _CASES["k3"]
    _, w, xs, ws, os_ = _params(case)
    src = _generated_source(_build_module(case, w, xs, ws, os_, kernel_h=kernel_h, h=h))
    assert "c7x_int8_dwconv1d(" not in src


@pytest.mark.quick
@pytest.mark.parametrize("name", list(_CASES))
def test_dwconv1d_exact(dsp_mode, name):
    """The kernel's output equals the generic path's arithmetic exactly."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    case = _CASES[name]
    x, w, xs, ws, os_ = _params(case)
    ref = _reference(case, x, w, xs, ws, os_)
    results = compile_and_run_dsp(
        mod=_build_module(case, w, xs, ws, os_),
        input_data=x,
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].reshape(ref.shape)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.quick
@pytest.mark.parametrize("name", ["k33_partial_block", "k9_stride2", "per_channel_input_scale"])
def test_input_relu_folded(name):
    """A transparent ReLU read only by the conv folds into c7x_int8_dwconv1d_relu."""
    case = _CASES[name]
    _, w, xs, ws, os_ = _params(case)
    src = _generated_source(_build_module(case, w, xs, ws, os_, relu="input"))
    assert "c7x_int8_dwconv1d_relu(" in src
    assert "c7x_int8_relu(" not in src


@pytest.mark.quick
def test_shared_relu_not_folded():
    """A ReLU with another reader stays a separate kernel."""
    case = _CASES["k3"]
    _, w, xs, ws, os_ = _params(case)
    src = _generated_source(_build_module(case, w, xs, ws, os_, relu="shared"))
    assert "c7x_int8_relu(" in src
    assert "c7x_int8_dwconv1d(" in src and "c7x_int8_dwconv1d_relu(" not in src


@pytest.mark.quick
@pytest.mark.parametrize("name", ["k33_partial_block", "k9_stride2", "k5_stride2_dilation2"])
def test_dwconv1d_relu_exact(dsp_mode, name):
    """With the ReLU folded in, the output equals relu-then-conv exactly."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    case = _CASES[name]
    x, w, xs, ws, os_ = _params(case)
    ref = _reference(case, x, w, xs, ws, os_, relu=True)
    results = compile_and_run_dsp(
        mod=_build_module(case, w, xs, ws, os_, relu="input"),
        input_data=x,
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].reshape(ref.shape)
    np.testing.assert_array_equal(out, ref)
