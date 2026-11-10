"""Unit tests for c7x_int8_silu_gate_multiply and c7x_int8_hardswish_gate_multiply.

Invokes each kernel via call_extern with known inputs and checks the output
against a numpy model of the kernel's own arithmetic, bit for bit, and against
the float formula it implements, within one code. Independent of
FuseQDQToC7xActivation.

  out[c, j] = quantize(act(dequantize(x[c, j])) * gate[c])

x and out are int8 [C, H_W], gate is float32 [C]. The kernels loop over channels
(as c7x_int8_channel_scale_multiply does), so the vectorized bulk and the scalar
tail are per channel: H_W // 8 * 8 bulk elements and H_W % 8 tail elements in
each. The bulk rounds ties to even (__float_to_int); the tail calls rq_f, which
rounds half away from zero.

Usage:
    pytest test_gate_multiply_kernels.py -v --dsp-mode=c7x_host
    pytest test_gate_multiply_kernels.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tirx

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent / "dsp-cpp"))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import (  # noqa: E402
    bulk_tail_split,
    compile_and_run_dsp,
    get_target_string,
    round_half_away,
)
from test_silu_kernel import _exp_taylor, _vec_recip  # noqa: E402

_F32 = np.float32


def _silu_bulk(x):
    return x * _vec_recip(_exp_taylor(-x) + _F32(1.0))


def _hardswish_bulk(x):
    return x * np.clip(x * _F32(1.0 / 6.0) + _F32(0.5), _F32(0.0), _F32(1.0)).astype(_F32)


def _silu_exact(x):
    return x / (1.0 + np.exp(-x))


def _hardswish_exact(x):
    return x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)


_ACTS = {"silu": (_silu_bulk, _silu_exact), "hardswish": (_hardswish_bulk, _hardswish_exact)}


def _numpy_gate_multiply(act, x, gate, n_ch, hw, zx, sx, zy, sy):
    """The kernel's arithmetic: float32 bulk, float64-approximated rq_f tail."""
    bulk_act, exact_act = _ACTS[act]
    x = x.reshape(n_ch, hw)
    nvec8 = bulk_tail_split(hw, vec_width=8)
    zx32, sx32, zy32, invsy = _F32(zx), _F32(sx), _F32(zy), _F32(1.0) / _F32(sy)
    out = np.empty((n_ch, hw), dtype=np.int32)
    for c in range(n_ch):
        g = _F32(gate[c])
        xf = (x[c, :nvec8].astype(_F32) - zx32) * sx32
        y = bulk_act(xf) * g
        out[c, :nvec8] = np.round(np.clip(y * invsy + zy32, _F32(-128), _F32(127))).astype(np.int32)
        xt = (x[c, nvec8:].astype(np.float64) - zx) * sx
        yt = exact_act(xt) * float(g)
        out[c, nvec8:] = round_half_away(yt / float(sy)).astype(np.int32) + zy
    return np.clip(out, -128, 127).astype(np.int8).reshape(-1)


def _exact_gate_multiply(act, x, gate, n_ch, hw, zx, sx, zy, sy):
    """The formula in float64, rounding half to even as PyTorch's quantize does."""
    _, exact_act = _ACTS[act]
    xf = (x.reshape(n_ch, hw).astype(np.float64) - zx) * sx
    y = exact_act(xf) * gate.astype(np.float64)[:, None]
    return np.clip(np.round(y / sy) + zy, -128, 127).astype(np.int8).reshape(-1)


def _build_module(act, n_ch, hw, zx, sx, zy, sy):
    extern_name = f"c7x_int8_{act}_gate_multiply"

    def te_kernel(x_t, g_t):
        def fcompute(ins, outs):
            return tirx.call_extern(
                "int32",
                extern_name,
                ins[0].data,
                ins[1].data,
                outs[0].data,
                tirx.IntImm("int32", n_ch),
                tirx.IntImm("int32", hw),
                tirx.IntImm("int32", int(zx)),
                tirx.FloatImm("float32", float(sx)),
                tirx.IntImm("int32", int(zy)),
                tirx.FloatImm("float32", float(sy)),
            )

        return te.extern([n_ch * hw], [x_t, g_t], fcompute, name="gate_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorType([n_ch * hw], "int8"))
    g_var = relax.Var("g", relax.TensorType([n_ch], "float32"))
    with bb.function("main", [x_var, g_var], attrs={"num_input": 2}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, g_var, primfunc_name_hint=extern_name)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run(dsp_mode, act, x, gate, n_ch, hw, zx, sx, zy, sy):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    results = compile_and_run_dsp(
        mod=_build_module(act, n_ch, hw, zx, sx, zy, sy),
        input_data=[x, gate.astype(np.float32)],
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"].flatten(), results.get("c7x_dload_cycles", 0)


def _check(dsp_mode, act, n_ch, hw, zx, sx, zy, sy, seed, record_cycles=None):
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, n_ch * hw, dtype=np.int8)
    # Squeeze-excite gates are a sigmoid or hardsigmoid output.
    gate = rng.uniform(0.0, 1.0, n_ch).astype(np.float32)
    out, cycles = _run(dsp_mode, act, x, gate, n_ch, hw, zx, sx, zy, sy)
    model = _numpy_gate_multiply(act, x, gate, n_ch, hw, zx, sx, zy, sy)
    exact = _exact_gate_multiply(act, x, gate, n_ch, hw, zx, sx, zy, sy)
    mismatch = np.flatnonzero(out != model)
    assert mismatch.size == 0, (
        f"{mismatch.size} of {out.size} differ from the kernel model, first at {mismatch[:5]}"
    )
    assert np.abs(out.astype(int) - exact.astype(int)).max() <= 1
    if record_cycles is not None and cycles:
        record_cycles(f"{act}_gate_multiply_c{n_ch}_hw{hw}", cycles)
    return out


_ACT_NAMES = ["silu", "hardswish"]


@pytest.mark.quick
@pytest.mark.parametrize("act", _ACT_NAMES)
@pytest.mark.parametrize(
    ("n_ch", "hw"),
    [
        (3, 37),  # 4 unrolled vectors + 5-element tail per channel
        (8, 49),  # 7x7 map: one unrolled group, two single vectors, one tail element
        (16, 8),  # exactly one vector per channel, no unrolled group, no tail
        (5, 5),  # smaller than a vector: tail only, no stream opened
    ],
    ids=["tail", "7x7", "one-vector", "tail-only"],
)
def test_gate_multiply_shapes(dsp_mode, act, n_ch, hw):
    _check(dsp_mode, act, n_ch, hw, zx=0, sx=0.05, zy=0, sy=0.02, seed=n_ch * 10 + len(act))


@pytest.mark.quick
@pytest.mark.parametrize("act", _ACT_NAMES)
def test_gate_multiply_zero_points(dsp_mode, act):
    _check(dsp_mode, act, 6, 29, zx=-10, sx=0.04, zy=5, sy=0.03, seed=21)


@pytest.mark.quick
@pytest.mark.parametrize("act", _ACT_NAMES)
def test_gate_multiply_saturates(dsp_mode, act):
    """An output scale far below the product's range: both ends clamp."""
    out = _check(dsp_mode, act, 4, 64, zx=0, sx=0.2, zy=0, sy=0.005, seed=22)
    assert (out == 127).any()


@pytest.mark.quick
@pytest.mark.parametrize("act", _ACT_NAMES)
def test_gate_multiply_zero_gate(dsp_mode, act):
    """A closed gate gives the output zero point, whatever the activation."""
    rng = np.random.default_rng(23)
    n_ch, hw, zy = 4, 40, 3
    x = rng.integers(-128, 128, n_ch * hw, dtype=np.int8)
    out, _ = _run(dsp_mode, act, x, np.zeros(n_ch, np.float32), n_ch, hw, 0, 0.05, zy, 0.02)
    assert (out == zy).all()


@pytest.mark.core
@pytest.mark.parametrize(
    ("act", "n_ch", "hw"),
    [
        ("silu", 32, 112 * 112),  # EfficientNet-B0's first squeeze-excite block
        ("silu", 1152, 7 * 7),  # its last
        ("hardswish", 960, 7 * 7),  # MobileNetV3-Large's last
        ("hardswish", 480, 14 * 14),
    ],
)
def test_gate_multiply_model_shapes(dsp_mode, act, n_ch, hw, record_cycles):
    _check(
        dsp_mode, act, n_ch, hw, zx=0, sx=0.06, zy=0, sy=0.03, seed=24, record_cycles=record_cycles
    )
