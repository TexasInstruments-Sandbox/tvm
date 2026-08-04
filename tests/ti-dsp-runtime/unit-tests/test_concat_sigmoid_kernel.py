"""Unit tests for c7x_int8_concat_sigmoid kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference. Tests are independent of
FuseQDQToC7xConcat (see test_concat_sigmoid_pass.py for pass-level coverage).

The kernel concatenates up to 4 int8 [C, n_i] tensors along the last (n)
axis, dequantizing and applying sigmoid per element (float32 output, no
requantize). Every branch shares the same leading channel count C; only the
trailing width n_i varies per branch (e.g. per-detection-scale anchor
counts in the YOLO multi-scale class-score glue). Unused slots are
indicated by n_i=0.

Same Taylor-exp + reciprocal-refinement vectorized path as
c7x_int8_silu_f32out (see test_silu_f32out_kernel.py); only the final store
differs (no self-gate multiply). Because there is no requantize step to
absorb 1-ULP float32 differences, comparisons use np.allclose rather than
exact equality.

Usage:
    pytest test_concat_sigmoid_kernel.py -v --dsp-mode=c7x_host
    pytest test_concat_sigmoid_kernel.py -v --dsp-mode=c7x_dload
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvm import relax, te, tir

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import bulk_tail_split, compile_and_run_dsp, get_target_string  # noqa: E402

# ---------------------------------------------------------------------------
# Reference implementation (Taylor exp / reciprocal copied from
# test_silu_f32out_kernel.py -- same kernel-side math, minus the self-gate
# multiply).
# ---------------------------------------------------------------------------

_KERNEL = "c7x_int8_concat_sigmoid"
_FLT_MAX = np.finfo(np.float32).max


def _exp_taylor(x):
    f32 = np.float32
    x = x.astype(f32)
    ln2 = f32(0.693147180559945)
    one_by_ln2 = f32(1.44269504090)
    one_by_6 = f32(0.1666667)
    one_by_24 = f32(0.0416667)
    pkd_one_by_65536 = f32(0.0000152587890625)

    y = one_by_ln2 * x
    yI = np.round(y).astype(np.int32)
    yf = y - yI.astype(f32)

    r1 = yf * ln2
    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r2 * r2
    two_pw_f = f32(1.0) + r1 + r2 * f32(0.5) + r3 * one_by_6 + r4 * one_by_24

    def _pow2_shift16(amt):
        pos = amt > 0
        shift_l = np.left_shift(np.int32(1 << 16), np.clip(amt, 0, 30))
        shift_r = np.right_shift(np.int32(1 << 16), np.clip(-amt, 0, 30))
        return np.where(pos, shift_l, shift_r)

    # Two chained safe rings (each clamped to +/-14, the widest single
    # shift that can't overflow int32) extend the exact range to yI in
    # [-28,28] instead of one unclamped ring -- see c7x_qdq_common.h's
    # exp_taylor for why.
    yI_lo = np.clip(yI, -14, 14)
    excess = yI - yI_lo
    excess_lo = np.clip(excess, -14, 14)

    e_pw_x = two_pw_f * _pow2_shift16(yI_lo).astype(f32) * pkd_one_by_65536
    e_pw_x = e_pw_x * _pow2_shift16(excess_lo).astype(f32) * pkd_one_by_65536
    e_pw_x = np.where(yI < -28, f32(0.0), e_pw_x)
    e_pw_x = np.where(yI > 28, f32(_FLT_MAX), e_pw_x)
    return e_pw_x


def _vec_recip(v):
    f32 = np.float32
    return (f32(1.0) / v.astype(np.float64)).astype(f32)


def _numpy_sigmoid_row(inp, zx, sx):
    """Dequantize + sigmoid for one flat row, matching the kernel's
    vectorized-bulk / scalar-tail split (see bulk_tail_split)."""
    f32 = np.float32
    n = len(inp)
    nvec8 = bulk_tail_split(n, vec_width=8)
    zx32, sx32 = f32(zx), f32(sx)
    one = f32(1.0)

    x = (inp.astype(f32) - zx32) * sx32
    sig_bulk = _vec_recip(_exp_taylor(-x) + one)[:nvec8]

    x_tail = (inp[nvec8:].astype(np.float64) - zx) * sx
    sig_tail = 1.0 / (1.0 + np.exp(-x_tail))

    out = np.empty(n, dtype=np.float32)
    out[:nvec8] = sig_bulk
    out[nvec8:] = sig_tail.astype(np.float32)
    return out


def _numpy_concat_sigmoid(branches, C):
    """branches: list of (data_flat[C*n_i], n_i, s_i, z_i) for active slots.

    Mirrors the kernel's per-channel interleave: output row c is
    [branch0 row c][branch1 row c]... (see c7x_concat.cpp's
    process_branch_sigmoid).
    """
    n_total = sum(n_i for _, n_i, _, _ in branches)
    out = np.zeros((C, n_total), dtype=np.float32)
    offset = 0
    for data_flat, n_i, s_i, z_i in branches:
        d = data_flat.reshape(C, n_i)
        for c in range(C):
            out[c, offset : offset + n_i] = _numpy_sigmoid_row(d[c], z_i, s_i)
        offset += n_i
    return out.flatten()


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------

_DUMMY = np.zeros([1], dtype=np.int8)


def _build_concat_sigmoid_module(slots, C):
    """Build a Relax module that calls c7x_int8_concat_sigmoid.

    slots: list of exactly 4 (data_np, n_i, s_i, z_i) tuples; pad unused
    slots with (_DUMMY, 0, 1.0, 0).
    """
    assert len(slots) == 4, "expect 4 slots, padded with (dummy, 0, 1.0, 0)"

    C_v = int(C)

    def _shapes(slot):
        """Full flat buffer length (C_v * n_i, maxed with 1 for disabled
        slots -- matches _DUMMY's own length of 1, not C_v * 1)."""
        data, n_i, s, z = slot
        n_full = max(C_v * n_i, 1)
        return n_full, float(s), int(z)

    n0_full, s0, z0 = _shapes(slots[0])
    n1_full, s1, z1 = _shapes(slots[1])
    n2_full, s2, z2 = _shapes(slots[2])
    n3_full, s3, z3 = _shapes(slots[3])
    n0_v, n1_v, n2_v, n3_v = (int(slots[i][1]) for i in range(4))
    n_total = n0_v + n1_v + n2_v + n3_v

    def te_kernel(t0, t1, t2, t3):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                tir.IntImm("int32", n0_v),
                tir.FloatImm("float32", s0),
                tir.IntImm("int32", z0),
                ins[1].data,
                tir.IntImm("int32", n1_v),
                tir.FloatImm("float32", s1),
                tir.IntImm("int32", z1),
                ins[2].data,
                tir.IntImm("int32", n2_v),
                tir.FloatImm("float32", s2),
                tir.IntImm("int32", z2),
                ins[3].data,
                tir.IntImm("int32", n3_v),
                tir.FloatImm("float32", s3),
                tir.IntImm("int32", z3),
                outs[0].data,
                tir.IntImm("int32", C_v),
            )

        return te.extern(
            [C_v * n_total],
            [t0, t1, t2, t3],
            fcompute,
            name="concat_sigmoid_out",
            dtype="float32",
        )

    bb = relax.BlockBuilder()
    v0 = relax.Var("in0", relax.TensorStructInfo([n0_full], "int8"))
    v1 = relax.Var("in1", relax.TensorStructInfo([n1_full], "int8"))
    v2 = relax.Var("in2", relax.TensorStructInfo([n2_full], "int8"))
    v3 = relax.Var("in3", relax.TensorStructInfo([n3_full], "int8"))
    with bb.function("main", [v0, v1, v2, v3], attrs={"num_input": 4}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, v0, v1, v2, v3, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_concat_sigmoid(dsp_mode, slots, C):
    mod = _build_concat_sigmoid_module(slots, C)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    inputs = [slot[0] for slot in slots]
    results = compile_and_run_dsp(
        mod=mod,
        input_data=inputs,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


def _make_slots(branches):
    """Build the padded 4-slot list for _run_concat_sigmoid /
    _build_concat_sigmoid_module from a list of (data_flat, n_i, s, z)."""
    slots = list(branches)
    dummy_slot = (_DUMMY, 0, 1.0, 0)
    while len(slots) < 4:
        slots.append(dummy_slot)
    return slots


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_concat_sigmoid_2branch(dsp_mode):
    """2 branches, different scales -- basic arity-2 case."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    C, n0, n1 = 4, 16, 24
    d0 = rng.integers(-128, 127, C * n0, dtype=np.int8)
    d1 = rng.integers(-128, 127, C * n1, dtype=np.int8)
    s0, z0, s1, z1 = 0.05, 3, 0.03, -2
    branches = [(d0, n0, s0, z0), (d1, n1, s1, z1)]
    ref = _numpy_concat_sigmoid(branches, C)
    slots = _make_slots(branches)
    out, _ = _run_concat_sigmoid(dsp_mode, slots, C)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.quick
def test_concat_sigmoid_scalar_tail(dsp_mode):
    """Row widths not multiples of 8 -- exercises the scalar tail per row."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    C, n0, n1, n2 = 3, 13, 7, 5
    d0 = rng.integers(-128, 127, C * n0, dtype=np.int8)
    d1 = rng.integers(-128, 127, C * n1, dtype=np.int8)
    d2 = rng.integers(-128, 127, C * n2, dtype=np.int8)
    s0, z0, s1, z1, s2, z2 = 0.04, 3, 0.02, -1, 0.06, 0
    branches = [(d0, n0, s0, z0), (d1, n1, s1, z1), (d2, n2, s2, z2)]
    ref = _numpy_concat_sigmoid(branches, C)
    slots = _make_slots(branches)
    out, _ = _run_concat_sigmoid(dsp_mode, slots, C)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.quick
def test_concat_sigmoid_asymmetric_zp(dsp_mode):
    """Non-zero zero-point per branch -- tests zp subtraction per branch."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    C, n0, n1 = 6, 40, 32
    d0 = rng.integers(-128, 127, C * n0, dtype=np.int8)
    d1 = rng.integers(-128, 127, C * n1, dtype=np.int8)
    s0, z0, s1, z1 = 0.045, -10, 0.021, 12
    branches = [(d0, n0, s0, z0), (d1, n1, s1, z1)]
    ref = _numpy_concat_sigmoid(branches, C)
    slots = _make_slots(branches)
    out, _ = _run_concat_sigmoid(dsp_mode, slots, C)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.core
def test_concat_sigmoid_yolo26n_shape(dsp_mode, record_cycles):
    """C=80, n=[1600,400,100] -- the real yolo26n/yolov8n multi-scale
    class-score shape (P3/P4/P5 at 40x40/20x20/10x10; see
    yolo_head_qdq_movement_fusion.md's Step 4 class-score path)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    C = 80
    n0, n1, n2 = 1600, 400, 100
    d0 = rng.integers(-128, 127, C * n0, dtype=np.int8)
    d1 = rng.integers(-128, 127, C * n1, dtype=np.int8)
    d2 = rng.integers(-128, 127, C * n2, dtype=np.int8)
    s0, z0 = 0.22449895739555359, 0
    s1, z1 = 0.54564881324768066, 0
    s2, z2 = 0.92429441213607788, 0
    branches = [(d0, n0, s0, z0), (d1, n1, s1, z1), (d2, n2, s2, z2)]
    ref = _numpy_concat_sigmoid(branches, C)
    slots = _make_slots(branches)
    out, cycles = _run_concat_sigmoid(dsp_mode, slots, C)
    record_cycles("concat_sigmoid_yolo26n_C80_n2100", cycles)
    np.testing.assert_allclose(out.flatten(), ref, rtol=1e-5, atol=1e-6)
    if cycles:
        n = C * (n0 + n1 + n2)
        print(
            f"\n  c7x_int8_concat_sigmoid C=80 n=2100: {cycles:,} cycles "
            f"({cycles / n:.2f} cycles/element)"
        )
