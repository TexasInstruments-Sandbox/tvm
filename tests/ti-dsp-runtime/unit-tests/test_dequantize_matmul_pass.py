"""Unit tests for FuseDequantizeMatmul's bias handling.

Pure Relax IR-level tests, no hardware or DSP build needed. torch.export turns a
linear into addmm, which the frontend imports as add(bias, matmul), bias first,
and the pass's pattern matches add in either operand order: the fused result must
keep the bias whichever side it is on.
"""

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorType as TensorStructInfo

M, K, N = 4, 64, 32


def _weights():
    rng = np.random.default_rng(0)
    w = rng.integers(-128, 128, (N, K)).astype("int8")
    scale = (rng.random(N) * 0.01 + 0.001).astype("float32")
    bias = rng.standard_normal(N).astype("float32")
    return w, scale, bias


def _linear_model(bias_first):
    """dequantize(w) -> permute_dims -> matmul(act, _) -> add with the bias."""
    w, scale, bias = _weights()
    act = relax.Var("act", TensorStructInfo((M, K), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [act]):
        with bb.dataflow():
            w_dq = bb.emit(
                relax.op.dequantize(
                    relax.const(w), relax.const(scale), relax.const(np.zeros(N, "int8")), axis=0
                )
            )
            w_t = bb.emit(relax.op.permute_dims(w_dq))
            mm = bb.emit(relax.op.matmul(act, w_t))
            b = relax.const(bias)
            add = relax.op.add(b, mm) if bias_first else relax.op.add(mm, b)
            out = bb.emit_output(bb.emit(add))
        bb.emit_func_output(out)
    return bb.finalize()


@pytest.mark.parametrize("bias_first", [True, False])
def test_bias_kept_in_either_operand_order(bias_first):
    mod = relax.transform.FuseDequantizeMatmul()(_linear_model(bias_first))
    assert "dequantize_matmul" in mod.script(show_meta=False)

    w, scale, bias = _weights()
    act = np.random.default_rng(1).standard_normal((M, K)).astype("float32")
    expected = act @ (w.astype("float32") * scale[:, None]).T + bias

    ex = relax.build(mod, tvm.target.Target("llvm", host="llvm"))
    out = relax.VirtualMachine(ex, tvm.cpu())["main"](tvm.runtime.tensor(act)).numpy()
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bias_first", [True, False])
def test_c7x_extern_path_keeps_bias(bias_first):
    """On a C7x target the matmul becomes a call_extern kernel; the bias add stays."""
    with tvm.target.Target("c_static_lib -mcpu=c7x"):
        mod = relax.transform.FuseDequantizeMatmul()(_linear_model(bias_first))
    script = mod["main"].script(show_meta=False)
    assert "dequantize_vecmatmul" in mod.script(show_meta=False)
    assert script.count("R.add(") == 1
