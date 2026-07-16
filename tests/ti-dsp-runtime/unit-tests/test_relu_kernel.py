"""Unit tests for c7x_int8_relu kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference. Tests are independent of the
ti_fuse_qdq_tidl_relu.py fusion pass.

The kernel computes: out[i] = max(in[i], clip_lo), a plain per-element
clamp with no rescale (relu is only lowered when d_zp == o_zp, so there is
no zero-point/scale math here, unlike c7x_int8_requantize_clamp).

Usage:
    pytest test_relu_kernel.py -v --dsp-mode=c7x_host
    pytest test_relu_kernel.py -v --dsp-mode=c7x_dload
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

_KERNEL = "c7x_int8_relu"

# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------


def _numpy_relu(x, clip_lo):
    return np.maximum(x.astype(np.int32), clip_lo).clip(-128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_relu_module(n, clip_lo):
    n_val = int(n)
    clip_lo_val = int(clip_lo)

    def te_relu(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", n_val),
                tir.IntImm("int32", clip_lo_val),
            )

        return te.extern([n_val], [x_t], fcompute, name="relu_out", dtype="int8")

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([n_val], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_relu, x_var, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_relu(dsp_mode, x, clip_lo):
    n = len(x)
    mod = _build_relu_module(n, clip_lo)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=x,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    out = results[f"{dsp_mode}_result"]
    cycles = results.get("c7x_dload_cycles", 0)
    return out, cycles


def _check(dsp_mode, x, clip_lo):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    ref = _numpy_relu(x, clip_lo)
    out, _ = _run_relu(dsp_mode, x, clip_lo)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_relu_zero_threshold(dsp_mode):
    """Standard ReLU (clip_lo=0) — exercises the 4x-unrolled vector loop."""
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 127, 256, dtype=np.int8)
    _check(dsp_mode, x, 0)


@pytest.mark.quick
def test_relu_nonzero_threshold(dsp_mode):
    """Non-zero clip_lo (asymmetric quant zero point folded into relu)."""
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 127, 256, dtype=np.int8)
    _check(dsp_mode, x, -12)


@pytest.mark.quick
@pytest.mark.parametrize("n", [1, 5, 7, 9, 15, 17, 63])
def test_relu_non_multiple_of_8(dsp_mode, n):
    """n % 8 != 0 — exercises the scalar tail path in the vectorized kernel."""
    rng = np.random.default_rng(2)
    x = rng.integers(-128, 127, n, dtype=np.int8)
    _check(dsp_mode, x, 3)


@pytest.mark.quick
def test_relu_below_vector_width(dsp_mode):
    """n < 8 — no vector iterations at all, pure scalar tail."""
    rng = np.random.default_rng(3)
    x = rng.integers(-128, 127, 5, dtype=np.int8)
    _check(dsp_mode, x, 0)


@pytest.mark.quick
def test_relu_extremes(dsp_mode):
    """INT8_MIN/MAX inputs around the threshold — boundary correctness."""
    x = np.array([-128, -1, 0, 1, 127, -128, 127, 0], dtype=np.int8)
    _check(dsp_mode, x, 0)


@pytest.mark.core
def test_relu_large_tensor(dsp_mode, record_cycles):
    """Large tensor (ResNeXt101-sized relu call) — confirms the vectorized
    path scales; ResNeXt101's relu is the #1 profiled bottleneck this step
    targets (docs/dsp/quantized_model_optimization.md Step 15)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(4)
    n = 64 * 56 * 56
    x = rng.integers(-128, 127, n, dtype=np.int8)
    ref = _numpy_relu(x, 0)
    out, cycles = _run_relu(dsp_mode, x, 0)
    record_cycles("relu_i8_200k", cycles)
    assert np.array_equal(out.flatten(), ref)
    if cycles:
        print(f"\n  c7x_int8_relu n={n}: {cycles:,} cycles ({cycles / n:.2f} cycles/element)")
