"""Unit tests for the c7x_int8_dfl_softmax kernel.

Invokes the kernel via call_extern with known inputs and verifies output
against a numpy reference. Tests are independent of the
FuseQDQToC7xActivation pass.

c7x_int8_dfl_softmax: fused dequantize -> transpose -> softmax -> quantize
for YOLOv8's DFL head.

  in:  int8[B][A][K][N], single scalar (zx, sx) dequant
  out: int8[B][K][A][N], single scalar (zy, sy) requant

  out[b,k,a,n] = quant(softmax_k(dq(in[b,a,:,n]))[k])

Scalar-only kernel (no vectorized path -- see c7x_softmax.cpp's module
docstring for why), so c7x_host and c7x_dload run identical code; both
modes are expected to match the reference exactly.

Usage:
    pytest test_dfl_softmax_kernel.py -v --dsp-mode=c7x_host
    pytest test_dfl_softmax_kernel.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


def _numpy_dfl_softmax(inp_bakn, zx, sx, zy, sy):
    """inp_bakn: [B,A,K,N] int8. Returns [B,K,A,N] int8."""
    f64 = np.float64
    x = (inp_bakn.astype(f64) - zx) * sx  # [B,A,K,N]
    m = x.max(axis=2, keepdims=True)
    e = np.exp(x - m)
    s = e.sum(axis=2, keepdims=True)
    sm = e / s  # [B,A,K,N]
    v = np.trunc(sm / sy + 0.5).astype(np.int64) + zy
    q = np.clip(v, -128, 127).astype(np.int8)
    return np.transpose(q, (0, 2, 1, 3))  # -> [B,K,A,N]


def _build_dfl_softmax_module(B, A, K, N, zx, sx, zy, sy):
    B_v, A_v, K_v, N_v = int(B), int(A), int(K), int(N)
    zx_v, sx_v, zy_v, sy_v = int(zx), float(sx), int(zy), float(sy)

    def te_kernel(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                "c7x_int8_dfl_softmax",
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", B_v),
                tir.IntImm("int32", A_v),
                tir.IntImm("int32", K_v),
                tir.IntImm("int32", N_v),
                tir.IntImm("int32", zx_v),
                tir.FloatImm("float32", sx_v),
                tir.IntImm("int32", zy_v),
                tir.FloatImm("float32", sy_v),
            )

        return te.extern(
            [B_v, K_v, A_v, N_v], [x_t], fcompute, name="dfl_softmax_out", dtype="int8"
        )

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([B_v, A_v, K_v, N_v], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, primfunc_name_hint="c7x_int8_dfl_softmax")
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_dfl_softmax(dsp_mode, inp, zx, sx, zy, sy):
    B, A, K, N = inp.shape
    mod = _build_dfl_softmax_module(B, A, K, N, zx, sx, zy, sy)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[inp],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_dfl_softmax_small(dsp_mode):
    """B=1, A=2, K=4, N=5 -- small, easy to hand-verify shape correctness."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    B, A, K, N = 1, 2, 4, 5
    inp = rng.integers(-128, 127, (B, A, K, N), dtype=np.int8)
    zx, sx, zy, sy = 0, 0.1, 0, 0.0078125
    ref = _numpy_dfl_softmax(inp, zx, sx, zy, sy)
    out, _ = _run_dfl_softmax(dsp_mode, inp, zx, sx, zy, sy)
    out = out.reshape(B, K, A, N)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


@pytest.mark.quick
def test_dfl_softmax_asymmetric_zp(dsp_mode):
    """Non-zero zero-points on input and output."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    B, A, K, N = 1, 3, 8, 7
    inp = rng.integers(-128, 127, (B, A, K, N), dtype=np.int8)
    zx, sx, zy, sy = -5, 0.08, 3, 0.0078125
    ref = _numpy_dfl_softmax(inp, zx, sx, zy, sy)
    out, _ = _run_dfl_softmax(dsp_mode, inp, zx, sx, zy, sy)
    out = out.reshape(B, K, A, N)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


@pytest.mark.quick
def test_dfl_softmax_batch2(dsp_mode):
    """B=2 -- exercises the outer batch loop."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    B, A, K, N = 2, 4, 16, 10
    inp = rng.integers(-128, 127, (B, A, K, N), dtype=np.int8)
    zx, sx, zy, sy = 0, 0.05, 0, 0.0078125
    ref = _numpy_dfl_softmax(inp, zx, sx, zy, sy)
    out, _ = _run_dfl_softmax(dsp_mode, inp, zx, sx, zy, sy)
    out = out.reshape(B, K, A, N)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


@pytest.mark.core
def test_dfl_softmax_yolov8n_shape(dsp_mode, record_cycles):
    """B=1, A=4, K=16, N=2100 -- yolov8n's actual DFL head shape (320x320
    input, 3 detection scales -> 2100 anchors)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    B, A, K, N = 1, 4, 16, 2100
    inp = rng.integers(-128, 127, (B, A, K, N), dtype=np.int8)
    zx, sx, zy, sy = 0, 0.1123003289103508, 0, 0.0071706650778651237
    ref = _numpy_dfl_softmax(inp, zx, sx, zy, sy)
    out, cycles = _run_dfl_softmax(dsp_mode, inp, zx, sx, zy, sy)
    record_cycles("dfl_softmax_yolov8n", cycles)
    out = out.reshape(B, K, A, N)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"
    if cycles:
        n = B * K * A * N
        print(
            f"\n  c7x_int8_dfl_softmax B={B} A={A} K={K} N={N}: "
            f"{cycles:,} cycles ({cycles / n:.2f} cycles/output-element)"
        )
