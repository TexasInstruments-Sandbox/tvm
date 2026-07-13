"""Unit tests for c7x_int8_concat_rescale kernel.

Directly invokes the kernel via call_extern with known inputs and verifies
output against a numpy reference.  Tests are independent of FuseQDQToC7xConcat.

The kernel concatenates up to 4 int8 NCHW tensors along the channel axis with
per-input requantization.  Unused slots are indicated by C_i=0.

Operation per input i, element x:
  output = sat_i8(trunc((x - z_i) * s_i / s_out + 0.5) + z_out)

Transparent fast path fires when s_i == s_out and z_i == z_out; uses memcpy.

Usage:
    pytest test_concat_kernel.py -v --dsp-mode=c7x_host
    pytest test_concat_kernel.py -v --dsp-mode=c7x_dload
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

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

_KERNEL = "c7x_int8_concat_rescale"


_SHIFT = 13


def _numpy_concat_rescale(inputs_zp_s, s_out, z_out):
    """Reference using the kernel's Q13 fixed-point arithmetic for exact match.

    Kernel computes:
      scale_q = round(s_i / s_out * 2^13)
      offset  = z_out - ((z_in * scale_q) >> 13)   [arithmetic right shift]
      out[j]  = clip((in[j] * scale_q >> 13) + offset, -128, 127)

    Transparent slot (s_i == s_out, z_i == z_out): kernel uses memcpy, which is
    equivalent to Q13 with scale_q=8192, offset=0.
    """
    parts = []
    for inp, s_i, z_i in inputs_zp_s:
        scale_q = np.int32(int(s_i / s_out * (1 << _SHIFT) + 0.5))
        offset = int(z_out) - int(np.int64(z_i) * int(scale_q) >> _SHIFT)
        result = np.clip(
            (inp.astype(np.int32) * int(scale_q) >> _SHIFT) + offset, -128, 127
        ).astype(np.int8)
        parts.append(result)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------

_DUMMY = np.zeros([1], dtype=np.int8)


def _build_concat_module(slots):
    """Build a Relax module that calls c7x_int8_concat_rescale.

    slots: list of (data_np, C, s_i, z_i) for up to 4 slots.
           Append (dummy, 0, 1.0, 0) tuples to pad to 4 entries.
    hw: H * W (spatial dims, same for all real slots)
    s_out, z_out: output QDQ params
    """
    assert len(slots) == 6, "expect (slot0..3, (s_out,z_out), HW)"
    slot0, slot1, slot2, slot3, (s_out, z_out), HW = slots

    def _shapes(slot):
        data, C, s, z = slot
        n = max(C * HW, 1)
        return n, C, float(s), int(z)

    n0, C0, s0, z0 = _shapes(slot0)
    n1, C1, s1, z1 = _shapes(slot1)
    n2, C2, s2, z2 = _shapes(slot2)
    n3, C3, s3, z3 = _shapes(slot3)
    s_out_v, z_out_v = float(s_out), int(z_out)
    HW_v = int(HW)
    C_total = C0 + C1 + C2 + C3

    def te_kernel(t0, t1, t2, t3):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                tir.IntImm("int32", C0),
                tir.FloatImm("float32", s0),
                tir.IntImm("int32", z0),
                ins[1].data,
                tir.IntImm("int32", C1),
                tir.FloatImm("float32", s1),
                tir.IntImm("int32", z1),
                ins[2].data,
                tir.IntImm("int32", C2),
                tir.FloatImm("float32", s2),
                tir.IntImm("int32", z2),
                ins[3].data,
                tir.IntImm("int32", C3),
                tir.FloatImm("float32", s3),
                tir.IntImm("int32", z3),
                outs[0].data,
                tir.IntImm("int32", HW_v),
                tir.FloatImm("float32", s_out_v),
                tir.IntImm("int32", z_out_v),
            )

        return te.extern(
            [C_total * HW_v],
            [t0, t1, t2, t3],
            fcompute,
            name="concat_out",
            dtype="int8",
        )

    bb = relax.BlockBuilder()
    v0 = relax.Var("in0", relax.TensorStructInfo([n0], "int8"))
    v1 = relax.Var("in1", relax.TensorStructInfo([n1], "int8"))
    v2 = relax.Var("in2", relax.TensorStructInfo([n2], "int8"))
    v3 = relax.Var("in3", relax.TensorStructInfo([n3], "int8"))
    with bb.function("main", [v0, v1, v2, v3], attrs={"num_input": 4}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, v0, v1, v2, v3, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_concat(dsp_mode, slots):
    mod = _build_concat_module(slots)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    inputs = [slot[0] for slot in slots[:4]]
    results = compile_and_run_dsp(
        mod=mod,
        input_data=inputs,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


def _make_slots(inputs_zp_s, HW, s_out, z_out):
    """Build the 6-tuple for _run_concat / _build_concat_module."""
    slots = []
    for data, s, z in inputs_zp_s:
        C = len(data) // HW
        slots.append((data, C, s, z))
    dummy_slot = (_DUMMY, 0, 1.0, 0)
    while len(slots) < 4:
        slots.append(dummy_slot)
    return slots + [(s_out, z_out), HW]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_concat2_transparent(dsp_mode):
    """2 inputs with matching scales — exercises the memcpy fast path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    HW, C0, C1 = 16, 32, 64
    d0 = rng.integers(-128, 127, C0 * HW, dtype=np.int8)
    d1 = rng.integers(-128, 127, C1 * HW, dtype=np.int8)
    s, z = 0.04, -3
    ref = _numpy_concat_rescale([(d0, s, z), (d1, s, z)], s_out=s, z_out=z)
    slots = _make_slots([(d0, s, z), (d1, s, z)], HW=HW, s_out=s, z_out=z)
    out, _ = _run_concat(dsp_mode, slots)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_concat2_rescale(dsp_mode):
    """2 inputs with different scales — exercises the Q13 rescale path."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    HW, C0, C1 = 16, 32, 48
    d0 = rng.integers(-128, 127, C0 * HW, dtype=np.int8)
    d1 = rng.integers(-128, 127, C1 * HW, dtype=np.int8)
    s0, z0, s1, z1, s_out, z_out = 0.03, 5, 0.05, -2, 0.04, 0
    ref = _numpy_concat_rescale([(d0, s0, z0), (d1, s1, z1)], s_out=s_out, z_out=z_out)
    slots = _make_slots([(d0, s0, z0), (d1, s1, z1)], HW=HW, s_out=s_out, z_out=z_out)
    out, _ = _run_concat(dsp_mode, slots)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_concat4_rescale(dsp_mode):
    """4 inputs with different scales — covers the Inception 4-branch pattern."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    HW = 16
    channels = [64, 128, 32, 32]
    scales = [0.02, 0.035, 0.018, 0.025]
    zps = [0, -3, 2, -1]
    inputs = [rng.integers(-128, 127, C * HW, dtype=np.int8) for C in channels]
    s_out, z_out = 0.03, 0
    entries = [(d, s, z) for d, s, z in zip(inputs, scales, zps)]
    ref = _numpy_concat_rescale(entries, s_out=s_out, z_out=z_out)
    slots = _make_slots(entries, HW=HW, s_out=s_out, z_out=z_out)
    out, _ = _run_concat(dsp_mode, slots)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.quick
def test_concat_scalar_tail(dsp_mode):
    """C*HW % 8 != 0 — exercises the scalar tail path in the vectorized kernel."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    HW = 11  # 11 * C is not divisible by 8 for most C
    C0, C1 = 5, 7
    d0 = rng.integers(-128, 127, C0 * HW, dtype=np.int8)
    d1 = rng.integers(-128, 127, C1 * HW, dtype=np.int8)
    s0, z0, s1, z1, s_out, z_out = 0.04, 3, 0.02, -1, 0.03, 0
    ref = _numpy_concat_rescale([(d0, s0, z0), (d1, s1, z1)], s_out=s_out, z_out=z_out)
    slots = _make_slots([(d0, s0, z0), (d1, s1, z1)], HW=HW, s_out=s_out, z_out=z_out)
    out, _ = _run_concat(dsp_mode, slots)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )


@pytest.mark.core
def test_concat_googlenet_size(dsp_mode, record_cycles):
    """GoogleNet layer [39]: C=[64,128,32,32], HW=28×28=784 (largest concat)."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(4)
    HW = 28 * 28
    channels = [64, 128, 32, 32]
    scales = [0.021, 0.034, 0.019, 0.027]
    zps = [0, -2, 1, 0]
    inputs = [rng.integers(-128, 127, C * HW, dtype=np.int8) for C in channels]
    s_out, z_out = 0.025, 0
    entries = list(zip(inputs, scales, zps))
    ref = _numpy_concat_rescale(entries, s_out=s_out, z_out=z_out)
    slots = _make_slots(entries, HW=HW, s_out=s_out, z_out=z_out)
    out, cycles = _run_concat(dsp_mode, slots)
    record_cycles("concat_googlenet_256ch_28x28", cycles)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        n = sum(channels) * HW
        print(
            f"\n  c7x_int8_concat_rescale C=256 HW=784: {cycles:,} cycles "
            f"({cycles / n:.2f} cycles/element)"
        )


@pytest.mark.core
def test_concat_inceptionv3_size(dsp_mode, record_cycles):
    """InceptionV3 layers [47],[63]: C=[192,256,64,96], HW=35×35=1225."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(5)
    HW = 35 * 35
    channels = [192, 256, 64, 96]
    scales = [0.018, 0.022, 0.031, 0.015]
    zps = [-1, 0, 2, -3]
    inputs = [rng.integers(-128, 127, C * HW, dtype=np.int8) for C in channels]
    s_out, z_out = 0.020, 0
    entries = list(zip(inputs, scales, zps))
    ref = _numpy_concat_rescale(entries, s_out=s_out, z_out=z_out)
    slots = _make_slots(entries, HW=HW, s_out=s_out, z_out=z_out)
    out, cycles = _run_concat(dsp_mode, slots)
    record_cycles("concat_inceptionv3_608ch_35x35", cycles)
    assert np.array_equal(out.flatten(), ref), (
        f"max_err={np.abs(out.flatten().astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        n = sum(channels) * HW
        print(
            f"\n  c7x_int8_concat_rescale C=608 HW=1225: {cycles:,} cycles "
            f"({cycles / n:.2f} cycles/element)"
        )
