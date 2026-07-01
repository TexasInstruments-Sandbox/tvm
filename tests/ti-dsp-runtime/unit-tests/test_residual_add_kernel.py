"""Unit tests for tvm_int8_residual_add_relu kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference.  Tests the kernel in isolation, independent
of the FuseInt8ResidualAdd pass.

The kernel computes:
  out[i] = sat_i8(((x[i]-zp_x)*M_x + (skip[i]-zp_skip)*M_skip) >> shift + zp_out)

Test coverage:
  - Symmetric quantization (all zero-points = 0), with and without relu
  - Asymmetric quantization (non-zero zero-points)
  - n % 8 != 0: exercises the scalar tail path
  - n < 8: no vector iterations at all, pure scalar tail
  - Large tensor: exercises the 4x-unrolled vector loop
  - Saturation: values that would overflow without clamping
  - int8 vs int16 variant correctness

Usage:
    pytest test_residual_add_kernel.py -v --dsp-mode=c7x_host
    pytest test_residual_add_kernel.py -v --dsp-mode=c7x_dload
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _pack_params_i8(M_x, M_skip, shift, zp_x=0, zp_skip=0, zp_out=0):
    """Pack requantization params into the 16-byte layout expected by the kernel."""
    buf = bytearray(16)
    struct.pack_into("<iii", buf, 0, M_x, M_skip, shift)
    struct.pack_into("bbb", buf, 12, zp_x, zp_skip, zp_out)
    return np.frombuffer(bytes(buf), dtype=np.int8)


def _numpy_ref_i8(x, skip, M_x, M_skip, shift, zp_x, zp_skip, zp_out, has_relu):
    """Exact integer arithmetic reference matching the kernel's operation."""
    x    = x.astype(np.int32)
    skip = skip.astype(np.int32)
    acc  = (x - zp_x) * M_x + (skip - zp_skip) * M_skip
    r    = (acc >> shift) + zp_out  # arithmetic right shift
    if has_relu:
        r = np.maximum(r, 0)
    return np.clip(r, -128, 127).astype(np.int8)


def _numpy_ref_i16(x, skip, M_x, M_skip, shift, zp_x, zp_skip, zp_out, has_relu):
    """int16 reference with int64 accumulator (matches the scalar kernel)."""
    x    = x.astype(np.int64)
    skip = skip.astype(np.int64)
    acc  = (x - zp_x) * M_x + (skip - zp_skip) * M_skip
    r    = (acc >> shift).astype(np.int32) + zp_out
    if has_relu:
        r = np.maximum(r, 0)
    return np.clip(r, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def _build_residual_add_module(n, params_np, kernel_name, in_dtype, out_dtype,
                                has_relu):
    """Build a minimal Relax module that calls the residual add kernel directly."""
    n_val     = int(n)
    has_relu_val = int(has_relu)

    def te_kernel(x_t, skip_t, params_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32", kernel_name,
                ins[0].data, ins[1].data, ins[2].data, outs[0].data,
                tir.IntImm("int32", n_val),
                tir.IntImm("int32", has_relu_val),
            )
        return te.extern([n_val], [x_t, skip_t, params_t], fcompute,
                         name="residual_add_out", dtype=out_dtype)

    bb = relax.BlockBuilder()
    x_var      = relax.Var("x",      relax.TensorStructInfo([n_val], in_dtype))
    skip_var   = relax.Var("skip",   relax.TensorStructInfo([n_val], in_dtype))
    params_var = relax.Var("params", relax.TensorStructInfo([16],    "int8"))

    with bb.function("main", [x_var, skip_var, params_var],
                     attrs={"num_input": 2}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, x_var, skip_var, params_var,
                                primfunc_name_hint=kernel_name)
            out = bb.emit_output(result)
        bb.emit_func_output(out)

    mod = bb.finalize()
    # Bind params constant; x and skip remain runtime inputs.
    mod = relax.transform.BindParams(
        func_name="main",
        params={mod["main"].params[2]: params_np},
    )(mod)
    return mod


def _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift, zp_x, zp_skip,
                       zp_out, has_relu, kernel="tvm_int8_residual_add_relu",
                       in_dtype="int8", out_dtype="int8"):
    n = len(x)
    params_np = _pack_params_i8(M_x, M_skip, shift, zp_x, zp_skip, zp_out)
    mod = _build_residual_add_module(n, params_np, kernel, in_dtype, out_dtype,
                                      has_relu)
    target  = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[x, skip],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    out = results[f"{dsp_mode}_result"]
    cycles = results.get("c7x_dload_cycles", 0)
    return out, cycles


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_i8_symmetric_no_relu(dsp_mode, record_cycles):
    """Symmetric quant (zp=0), no relu — exercises the basic vectorized path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    n   = 256
    x    = rng.integers(-100, 100, n, dtype=np.int8)
    skip = rng.integers(-100, 100, n, dtype=np.int8)
    # Typical ResNet-18 residual add scale: x_scale=0.02, skip_scale=0.03, o_scale=0.04
    shift = 16
    M_x   = int(round(0.02 / 0.04 * (1 << shift)))
    M_skip = int(round(0.03 / 0.04 * (1 << shift)))
    ref = _numpy_ref_i8(x, skip, M_x, M_skip, shift, 0, 0, 0, False)
    out, cycles = _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift,
                                     0, 0, 0, False)
    record_cycles("residual_add_i8_n256", cycles)
    assert np.array_equal(out.flatten(), ref), \
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"


@pytest.mark.quick
def test_i8_symmetric_with_relu(dsp_mode):
    """Symmetric quant with relu — checks that negative results are zeroed."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    n   = 128
    x    = rng.integers(-127, 0, n, dtype=np.int8)   # all negative → relu clips to 0
    skip = rng.integers(-127, 0, n, dtype=np.int8)
    shift, M_x, M_skip = 16, 32768, 32768             # scale ratio = 1.0
    ref = _numpy_ref_i8(x, skip, M_x, M_skip, shift, 0, 0, 0, True)
    out, _ = _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift,
                                0, 0, 0, True)
    assert np.all(out.flatten() >= 0), "relu should clip negatives to 0"
    assert np.array_equal(out.flatten(), ref)


@pytest.mark.quick
def test_i8_asymmetric(dsp_mode):
    """Asymmetric quantization (non-zero zero-points) — tests zp subtraction."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng   = np.random.default_rng(2)
    n     = 64
    x     = rng.integers(-128, 127, n, dtype=np.int8)
    skip  = rng.integers(-128, 127, n, dtype=np.int8)
    shift = 14
    M_x   = int(round(0.05 / 0.06 * (1 << shift)))
    M_skip = int(round(0.04 / 0.06 * (1 << shift)))
    # Non-zero zero-points typical of asymmetric quant
    zp_x, zp_skip, zp_out = 5, -3, 2
    ref = _numpy_ref_i8(x, skip, M_x, M_skip, shift, zp_x, zp_skip, zp_out, False)
    out, _ = _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift,
                                zp_x, zp_skip, zp_out, False)
    assert np.array_equal(out.flatten(), ref)


@pytest.mark.quick
def test_i8_non_multiple_of_8(dsp_mode):
    """n % 8 != 0 — exercises the scalar tail path in the vectorized kernel."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    for n in (1, 5, 7, 9, 15, 17, 63):
        x    = rng.integers(-100, 100, n, dtype=np.int8)
        skip = rng.integers(-100, 100, n, dtype=np.int8)
        shift, M_x, M_skip = 15, 16384, 16384
        ref = _numpy_ref_i8(x, skip, M_x, M_skip, shift, 0, 0, 0, False)
        out, _ = _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift,
                                    0, 0, 0, False)
        assert np.array_equal(out.flatten(), ref), \
            f"n={n}: max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"


@pytest.mark.quick
def test_i8_saturation(dsp_mode):
    """Values that saturate to INT8_MIN/MAX — checks clamp correctness."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    # Use large multipliers so intermediate products push well beyond ±127
    x    = np.array([127,  127, -128, -128,   0,   0,  64,  64], dtype=np.int8)
    skip = np.array([127,  127, -128, -128,   0,   0,  64,  64], dtype=np.int8)
    # shift=0 so no right-shift: raw sum after multiply lands far outside int8 range
    shift, M_x, M_skip = 0, 1, 1
    ref = _numpy_ref_i8(x, skip, M_x, M_skip, shift, 0, 0, 0, False)
    out, _ = _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift,
                                0, 0, 0, False)
    assert np.array_equal(out.flatten(), ref)
    # Extremes must be exactly ±127 (saturated)
    assert out.flatten()[0] == 127
    assert out.flatten()[2] == -128


@pytest.mark.quick
def test_i8_large_tensor(dsp_mode, record_cycles):
    """Large tensor (ResNet-18 residual add size) — confirms 4x-unrolled loop."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    # ResNet-18 layer1: 1×64×56×56 = 200,704 elements
    rng = np.random.default_rng(4)
    n   = 64 * 56 * 56
    x    = rng.integers(-100, 100, n, dtype=np.int8)
    skip = rng.integers(-100, 100, n, dtype=np.int8)
    shift = 16
    M_x   = int(round(0.02 / 0.04 * (1 << shift)))
    M_skip = int(round(0.03 / 0.04 * (1 << shift)))
    ref = _numpy_ref_i8(x, skip, M_x, M_skip, shift, 0, 0, 0, False)
    out, cycles = _run_residual_add(dsp_mode, x, skip, M_x, M_skip, shift,
                                     0, 0, 0, False)
    record_cycles("residual_add_i8_200k", cycles)
    assert np.array_equal(out.flatten(), ref)
    if cycles:
        print(f"\n  residual_add n={n}: {cycles:,} cycles "
              f"({cycles / n:.1f} cycles/element)")


@pytest.mark.quick
def test_i16_scalar_correctness(dsp_mode):
    """int16 variant (scalar path) — confirms int64 accumulator handles overflow."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(5)
    n   = 128
    x    = rng.integers(-32768, 32767, n, dtype=np.int16)
    skip = rng.integers(-32768, 32767, n, dtype=np.int16)
    # Large M values that would overflow int32 accumulation:
    # 32767 * M_x > INT32_MAX when M_x > 65537, hence int64 is required.
    shift = 20
    M_x   = int(round(0.001 / 0.002 * (1 << shift)))  # ~524288, > 2^15
    M_skip = int(round(0.0015 / 0.002 * (1 << shift)))
    ref = _numpy_ref_i16(x, skip, M_x, M_skip, shift, 0, 0, 0, False)
    out, _ = _run_residual_add(
        dsp_mode, x, skip, M_x, M_skip, shift, 0, 0, 0, False,
        kernel="tvm_int16_residual_add_relu",
        in_dtype="int16", out_dtype="int16",
    )
    assert np.array_equal(out.flatten(), ref)
