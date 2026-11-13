"""Unit tests for c7x_int8_silu_per_channel and c7x_int8_hardswish_per_channel.

Invokes each kernel via call_extern and checks it bit for bit against a numpy model
of its arithmetic, and within one code of the float formula. Channel c is the
per-tensor kernel on that channel's values with scales sx[c] and sy[c], so the model
is test_gate_multiply_kernels's per-channel model with a gate of 1.0 (multiplying by
1.0 is exact): the vectorized bulk rounds ties to even, the scalar tail half away
from zero.

Usage:
    pytest test_per_channel_activation_kernels.py -v --dsp-mode=c7x_host
    pytest test_per_channel_activation_kernels.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tirx

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent / "dsp-cpp"))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from test_gate_multiply_kernels import _exact_gate_multiply, _numpy_gate_multiply  # noqa: E402


def _model(act, x, n_ch, hw, zx, sx, zy, sy):
    x = x.reshape(n_ch, hw)
    one = np.ones(1, np.float32)
    return np.concatenate(
        [_numpy_gate_multiply(act, x[c], one, 1, hw, zx, sx[c], zy, sy[c]) for c in range(n_ch)]
    )


def _exact(act, x, n_ch, hw, zx, sx, zy, sy):
    x = x.reshape(n_ch, hw)
    one = np.ones(1, np.float32)
    return np.concatenate(
        [_exact_gate_multiply(act, x[c], one, 1, hw, zx, sx[c], zy, sy[c]) for c in range(n_ch)]
    )


def _build_module(act, n_ch, hw, zx, zy):
    extern_name = f"c7x_int8_{act}_per_channel"

    def te_kernel(x_t, sx_t, sy_t):
        def fcompute(ins, outs):
            return tirx.call_extern(
                "int32",
                extern_name,
                ins[0].data,
                ins[1].data,
                ins[2].data,
                outs[0].data,
                tirx.IntImm("int32", n_ch),
                tirx.IntImm("int32", hw),
                tirx.IntImm("int32", int(zx)),
                tirx.IntImm("int32", int(zy)),
            )

        return te.extern([n_ch * hw], [x_t, sx_t, sy_t], fcompute, name="pc_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorType([n_ch * hw], "int8"))
    sx_var = relax.Var("sx", relax.TensorType([n_ch], "float32"))
    sy_var = relax.Var("sy", relax.TensorType([n_ch], "float32"))
    with bb.function("main", [x_var, sx_var, sy_var], attrs={"num_input": 3}):
        with bb.dataflow():
            out = bb.emit_output(
                bb.emit_te(te_kernel, x_var, sx_var, sy_var, primfunc_name_hint=extern_name)
            )
        bb.emit_func_output(out)
    return bb.finalize()


def _check(dsp_mode, act, n_ch, hw, zx, zy, seed):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, n_ch * hw, dtype=np.int8)
    # Channels whose ranges differ by up to 100x, as in a depthwise block's input.
    sx = (0.002 * 10 ** rng.uniform(0, 2, n_ch)).astype(np.float32)
    sy = (sx * rng.uniform(0.3, 1.5, n_ch)).astype(np.float32)
    results = compile_and_run_dsp(
        mod=_build_module(act, n_ch, hw, zx, zy),
        input_data=[x, sx, sy],
        target_string=get_target_string(dsp_mode, use_cpp_api=True),
        execution_mode=dsp_mode,
    )
    out = results[f"{dsp_mode}_result"].flatten()
    model = _model(act, x, n_ch, hw, zx, sx, zy, sy)
    mismatch = np.flatnonzero(out != model)
    assert mismatch.size == 0, (
        f"{mismatch.size} of {out.size} differ from the model, first at {mismatch[:5]}"
    )
    assert np.abs(out.astype(int) - _exact(act, x, n_ch, hw, zx, sx, zy, sy).astype(int)).max() <= 1


_ACTS = ["silu", "hardswish"]


@pytest.mark.quick
@pytest.mark.parametrize("act", _ACTS)
@pytest.mark.parametrize(
    ("n_ch", "hw"),
    [(3, 37), (8, 49), (16, 8), (5, 5)],
    ids=["tail", "7x7", "one-vector", "tail-only"],
)
def test_per_channel_shapes(dsp_mode, act, n_ch, hw):
    _check(dsp_mode, act, n_ch, hw, zx=0, zy=0, seed=n_ch * 10 + len(act))


@pytest.mark.quick
@pytest.mark.parametrize("act", _ACTS)
def test_per_channel_zero_points(dsp_mode, act):
    _check(dsp_mode, act, 6, 29, zx=-10, zy=5, seed=31)


@pytest.mark.core
@pytest.mark.parametrize(
    ("act", "n_ch", "hw"),
    [("hardswish", 16, 112 * 112), ("silu", 32, 112 * 112), ("silu", 144, 56 * 56)],
    ids=["mobilenet_v3-stem", "efficientnet-stem", "efficientnet-block2"],
)
def test_per_channel_model_shapes(dsp_mode, act, n_ch, hw):
    _check(dsp_mode, act, n_ch, hw, zx=0, zy=0, seed=32)
