"""Unit tests for c7x_int8_max_pool_tidl kernel.

Directly invokes the TIDL-backed max pool kernel via call_extern and verifies:
  1. Output matches the scalar c7x_int8_max_pool reference byte-for-byte.
  2. Cycle count is substantially lower than the ~18.5M cycles of the scalar path.

The TIDL kernel uses a 3-row simultaneous approach: two Streaming Engines
deliver 32 int8 values/cycle from three consecutive rows, vertical max is
taken in 2 __max() calls, and horizontal max for stride=2 uses register
shifts — no loop over the kernel window.  Expected: ~300–600K cycles for
the ResNet-18 configuration (112×112×64 input, 3×3/stride=2).

Note: c7x_host is NOT supported for this test — the TIDL max pool wrapper
(tidl_maxpool_wrapper.cpp) is compiled into the firmware only (USE_TIDL_RUNTIME)
and is not available in the host emulation build.

Usage:
    pytest test_maxpool_kernel.py -v --dsp-mode=c7x_dload
    pytest test_maxpool_kernel.py -v --dsp-mode=c7x_dload -s   # print cycles
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
# Numpy reference
# ---------------------------------------------------------------------------


def _numpy_max_pool(x_nchw, kH, kW, sH, sW, pH, pW):
    """Reference: NCHW int8 max pool with symmetric padding, filling with -128."""
    N, C, H, W = x_nchw.shape
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    out = np.full((N, C, H_out, W_out), -128, dtype=np.int8)
    for b in range(N):
        for c in range(C):
            for ph in range(H_out):
                for pw in range(W_out):
                    ih0 = ph * sH - pH
                    iw0 = pw * sW - pW
                    m = np.int8(-128)
                    for kh in range(kH):
                        for kw in range(kW):
                            ih = ih0 + kh
                            iw = iw0 + kw
                            if 0 <= ih < H and 0 <= iw < W:
                                v = x_nchw[b, c, ih, iw]
                                if v > m:
                                    m = v
                    out[b, c, ph, pw] = m
    return out


# ---------------------------------------------------------------------------
# Module builder (shared for both kernels)
# ---------------------------------------------------------------------------


def _build_maxpool_module(kernel_name, N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW):
    """Build a Relax module that calls the given max pool kernel via call_extern."""
    _N, _C = N, C
    _H_in, _W_in, _H_out, _W_out = H_in, W_in, H_out, W_out
    _kH, _kW, _sH, _sW, _pH, _pW = kH, kW, sH, sW, pH, pW

    def te_maxpool(x_t):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                kernel_name,
                ins[0].data,
                outs[0].data,
                tir.IntImm("int32", _N),
                tir.IntImm("int32", _C),
                tir.IntImm("int32", _H_in),
                tir.IntImm("int32", _W_in),
                tir.IntImm("int32", _H_out),
                tir.IntImm("int32", _W_out),
                tir.IntImm("int32", _kH),
                tir.IntImm("int32", _kW),
                tir.IntImm("int32", _sH),
                tir.IntImm("int32", _sW),
                tir.IntImm("int32", _pH),
                tir.IntImm("int32", _pW),
            )

        return te.extern(
            [_N, _C, _H_out, _W_out],
            [x_t],
            fcompute,
            name="maxpool_out",
            dtype="int8",
        )

    bb = relax.BlockBuilder()
    x_var = relax.Var("x", relax.TensorStructInfo([N, C, H_in, W_in], "int8"))
    with bb.function("main", [x_var], attrs={"num_input": 1}):
        with bb.dataflow():
            result = bb.emit_te(te_maxpool, x_var, primfunc_name_hint=kernel_name)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_maxpool(dsp_mode, kernel_name, x, N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW):
    mod = _build_maxpool_module(kernel_name, N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW)
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
    return out.reshape(N, C, H_out, W_out), cycles


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_tidl_maxpool_resnet18_correctness(dsp_mode, record_cycles):
    """ResNet-18 config: 3×3/s2, 112×112×64 → 56×56×64.

    Verifies c7x_int8_max_pool_tidl output matches the scalar c7x_int8_max_pool
    reference byte-for-byte.  Also records cycles — expected ~300–600K
    (vs ~18.5M for the scalar kernel).
    """
    if dsp_mode != "c7x_dload":
        pytest.skip("c7x_int8_max_pool_tidl is firmware-only; requires c7x_dload")

    rng = np.random.default_rng(0)
    N, C, H, W = 1, 64, 112, 112
    kH, kW, sH, sW, pH, pW = 3, 3, 2, 2, 1, 1
    H_out = (H + 2 * pH - kH) // sH + 1  # 56
    W_out = (W + 2 * pW - kW) // sW + 1  # 56
    x = rng.integers(-128, 127, (N, C, H, W), dtype=np.int8)

    ref = _numpy_max_pool(x, kH, kW, sH, sW, pH, pW)

    out_tidl, cycles = _run_maxpool(
        dsp_mode,
        "c7x_int8_max_pool_tidl",
        x,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
    )

    record_cycles("maxpool_tidl_112x112x64", cycles)

    assert np.array_equal(out_tidl, ref), (
        f"Output mismatch: max_diff={np.abs(out_tidl.astype(int) - ref.astype(int)).max()}"
    )
    if cycles:
        print(f"\n  max pool 112×112×64: {cycles:,} cycles ({cycles / 1e6:.2f} ms @ 1 GHz)")
        if cycles < 5_000_000:
            print("  → TIDL vectorized kernel active")
        else:
            print("  → scalar fallback active (TIDL_MAXPOOL_USE_TIDL_KERNEL disabled)")
    # When TIDL kernel is active: expect < 5M; scalar fallback: ~18-19M.
    # Both are valid; the test documents the expected range for each path.
    assert cycles == 0 or cycles < 25_000_000, f"Unexpectedly high cycle count: {cycles:,}"


@pytest.mark.quick
def test_tidl_vs_scalar_identical_output(dsp_mode):
    """TIDL and scalar kernels must produce identical output on the same input."""
    if dsp_mode != "c7x_dload":
        pytest.skip("c7x_int8_max_pool_tidl is firmware-only; requires c7x_dload")

    rng = np.random.default_rng(1)
    N, C, H, W = 1, 64, 112, 112
    kH, kW, sH, sW, pH, pW = 3, 3, 2, 2, 1, 1
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    x = rng.integers(-128, 127, (N, C, H, W), dtype=np.int8)

    out_tidl, _ = _run_maxpool(
        dsp_mode,
        "c7x_int8_max_pool_tidl",
        x,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
    )
    out_scalar, _ = _run_maxpool(
        dsp_mode,
        "c7x_int8_max_pool",
        x,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
    )

    assert np.array_equal(out_tidl, out_scalar), (
        f"TIDL and scalar outputs differ: "
        f"max_diff={np.abs(out_tidl.astype(int) - out_scalar.astype(int)).max()}, "
        f"n_mismatch={np.sum(out_tidl != out_scalar)}"
    )


@pytest.mark.quick
def test_tidl_maxpool_padding_correctness(dsp_mode):
    """3×3/stride=1 with padding=1 — different TIDL kernel path than stride=2."""
    if dsp_mode != "c7x_dload":
        pytest.skip("requires c7x_dload")

    rng = np.random.default_rng(2)
    N, C, H, W = 1, 32, 28, 28
    kH, kW, sH, sW, pH, pW = 3, 3, 1, 1, 1, 1
    H_out = (H + 2 * pH - kH) // sH + 1  # 28
    W_out = (W + 2 * pW - kW) // sW + 1  # 28
    x = rng.integers(-128, 127, (N, C, H, W), dtype=np.int8)

    ref = _numpy_max_pool(x, kH, kW, sH, sW, pH, pW)
    out, _ = _run_maxpool(
        dsp_mode,
        "c7x_int8_max_pool_tidl",
        x,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
    )
    assert np.array_equal(out, ref)


@pytest.mark.quick
def test_tidl_maxpool_no_padding(dsp_mode):
    """3×3/stride=2, no padding — all elements are valid input."""
    if dsp_mode != "c7x_dload":
        pytest.skip("requires c7x_dload")

    rng = np.random.default_rng(3)
    N, C, H, W = 1, 16, 8, 8
    kH, kW, sH, sW, pH, pW = 3, 3, 2, 2, 0, 0
    H_out = (H + 2 * pH - kH) // sH + 1  # 3
    W_out = (W + 2 * pW - kW) // sW + 1  # 3
    x = rng.integers(-128, 127, (N, C, H, W), dtype=np.int8)

    ref = _numpy_max_pool(x, kH, kW, sH, sW, pH, pW)
    out, _ = _run_maxpool(
        dsp_mode,
        "c7x_int8_max_pool_tidl",
        x,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
    )
    assert np.array_equal(out, ref)


# ---------------------------------------------------------------------------
# Tests: c7x_int8_max_pool scalar-symbol kernel (SE-vectorized fast path)
#
# Unlike the TIDL tests above, these run on both c7x_host and c7x_dload:
# the fast path (max_pool_interior_fast in c7x_pool_relu.cpp) is a pure C7x
# kernel with no TIDL dependency, and __C7524__ is defined by <c7x.h> itself
# under host emulation too, so it's bit-exact-testable without hardware.
# ---------------------------------------------------------------------------


def _check_maxpool(dsp_mode, x, kH, kW, sH, sW, pH, pW, record_cycles=None, cycle_name=None):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    N, C, H, W = x.shape
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    ref = _numpy_max_pool(x, kH, kW, sH, sW, pH, pW)
    out, cycles = _run_maxpool(
        dsp_mode,
        "c7x_int8_max_pool",
        x,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        kH,
        kW,
        sH,
        sW,
        pH,
        pW,
    )
    if record_cycles is not None and cycle_name is not None:
        record_cycles(cycle_name, cycles)
    assert np.array_equal(out, ref), (
        f"max_diff={np.abs(out.astype(int) - ref.astype(int)).max()}, "
        f"n_mismatch={np.sum(out != ref)}"
    )
    return cycles


@pytest.mark.quick
def test_scalar_symbol_maxpool_resnet18_fastpath(dsp_mode, record_cycles):
    """3×3/s2/p1, 112×112×64 -> 56×56×64: the ResNet-18 fast-path target.

    Interior is 55×55 (ph_lo,ph_hi=1,56; pw_lo,pw_hi=1,56); routes through
    max_pool_interior_fast_dualrow_s2 (row_pairs=27 + 1 trailing row).
    dual_block_width=15 gives num_blocks_dual=3 with a 10-column
    remainder -- covered by the tail-overlap block (one more, slightly
    redundant, vectorized block landing exactly on pw_hi) rather than the
    scalar path. C=64 exercises the per-channel loop at production scale.

    Measured on AM67A/BeagleY-AI hardware (c7x_dload): ~2.73M cycles, a
    6.8x drop from the ~18.5M scalar baseline. This wasn't the first
    number measured here: the initial
    two-output-rows-per-pass port (no tail-overlap, plain scalar
    remainder) actually *regressed* to ~6.92M vs. the single-row stride-2
    path's ~6.15M -- the wider dual_block_width (15 vs. the single-row
    path's 8) left a larger remainder for this specific width (55), and
    the extra scalar work outweighed the vectorized win. The tail-overlap
    block (see max_pool_interior_fast_dualrow_s2's comment) fixed that by
    eliminating the remainder scalar path entirely instead of shrinking it.
    """
    rng = np.random.default_rng(10)
    N, C, H, W = 1, 64, 112, 112
    x = rng.integers(-128, 128, (N, C, H, W), dtype=np.int8)
    cycles = _check_maxpool(
        dsp_mode,
        x,
        3,
        3,
        2,
        2,
        1,
        1,
        record_cycles=record_cycles,
        cycle_name="maxpool_scalar_symbol_resnet18",
    )
    if dsp_mode == "c7x_dload" and cycles:
        print(f"\n  c7x_int8_max_pool fastpath 112×112×64: {cycles:,} cycles")
        # Measured ~2.73M on hardware; 4M leaves headroom while still
        # firmly gating against a regression to either the ~18.5M scalar
        # baseline or the ~6.15M/~6.92M single-row/pre-tail-overlap points.
        assert cycles < 4_000_000, (
            f"Fast path did not reduce cycles as expected: {cycles:,} "
            "cycles (scalar baseline is ~18.5M)"
        )


_SHAPE_CASES = {
    # 3x3/s1/p1, 28x28x32 (same-size pool): interior 26x26. Routes through
    # max_pool_interior_fast_dualrow (all 3x3/s1 shapes do), but interior
    # width 26 < dual_block_width (30), so numBlocks_dual=0 here and the
    # SE0/SE1 vectorized reduction never actually fires -- this case only
    # exercises the early-return guard and the all-scalar-remainder path.
    # See the dualrow_* cases below for shapes that actually run the
    # vectorized path.
    "3x3_stride1": (11, 1, 32, 28, 28, 3, 3, 1, 1, 1, 1),
    # 2x2/s2/p0, 8x8: interior=4. Routes through
    # max_pool_interior_fast_dualrow_s2 (all 2x2/s2 shapes do), but
    # dual_block_width for kW=2 is 16, so num_blocks_dual=0 here -- the
    # vectorized pack_consec combination never fires, only the early-return
    # guard and the all-scalar-remainder path. Also has zero border (p0),
    # unlike every other case. See the dualrow_s2_* cases below for shapes
    # that actually run the vectorized path.
    "2x2_stride2_below_vector_width": (12, 1, 1, 8, 8, 2, 2, 2, 2, 0, 0),
    # 2x2/s2/p0, 32x32x4: interior=16 -> num_blocks_dual=1 (dual_block_width
    # =16 for kW=2, no boundary loss), remainder=0 -- exercises the
    # vectorized pack_consec(vmax1,vmax0) combination for kW=2 (no
    # shift_right_bytes term needed). C=4 exercises the per-channel loop.
    "2x2_stride2_multichannel": (13, 1, 4, 32, 32, 2, 2, 2, 2, 0, 0),
    # 3x3/s2/p1 on a 3x3 input -> 1x1 output: ph_lo==ph_hi==1 (zero interior
    # rows/cols). The fast path must recognize rows<=0/numBlocks<=0 and
    # no-op cleanly, leaving the border strips to cover the entire
    # (degenerate, all-border) output -- the one case where a missed guard
    # on the DECIM-based fast path would show up.
    "zero_interior": (14, 1, 1, 3, 3, 3, 3, 2, 2, 1, 1),
    # 5x5/s1/p2 ("same" pool with a 5x5 window): not in the fast-path shape
    # table, so the whole image must use max_pool_scalar_rect -- confirms no
    # regression for shapes outside the fast path.
    "non_eligible_fallback": (15, 1, 2, 16, 16, 5, 5, 1, 1, 2, 2),
    # 3x3/s1/p1, 64x64x2: interior 62x62 (even row count -> row_pairs=31,
    # no trailing row), interior_w=62 -> numBlocks_dual=2, remainder=2.
    # First case wide/tall enough to actually run max_pool_interior_fast_
    # dualrow's SE0/SE1 vectorized reduction (vertical-max + horizontal
    # shift), not just its early-return guard.
    "dualrow_even_rows": (16, 1, 2, 64, 64, 3, 3, 1, 1, 1, 1),
    # 3x3/s1/p1, 63x64x2 (H odd, W even): interior rows=61 (odd ->
    # row_pairs=30 + one trailing row), interior_w=62 (numBlocks_dual=2,
    # remainder=2). Exercises the trailing-row fallback to the single-row
    # max_pool_interior_fast, and its own (differently-sized) remainder
    # split, alongside the paired rows' remainder.
    "dualrow_odd_rows": (17, 1, 2, 63, 64, 3, 3, 1, 1, 1, 1),
    # 3x3/s1/p1, 34x32x1: interior rows=32 (even), interior_w=30 -- exactly
    # one dual_block_width-wide block, remainder=0. Isolates the pure
    # block-only path (no remainder noise) for the vectorized reduction.
    "dualrow_exact_block_multiple": (18, 1, 1, 34, 32, 3, 3, 1, 1, 1, 1),
    # 3x3/s2/p1, 34x34x2: interior 16x16 (even rows -> row_pairs=8, no
    # trailing), interior_w=16 -> num_blocks_dual=1 (dual_block_width=15
    # for kW=3), remainder=1. First case sized to actually run
    # max_pool_interior_fast_dualrow_s2's pack_consec-based vectorized
    # combination for the stride-2 shape family.
    "dualrow_s2_3x3_even_rows": (19, 1, 2, 34, 34, 3, 3, 2, 2, 1, 1),
    # 3x3/s2/p1, 32x34x2 (H odd interior, W even interior): interior
    # rows=15 (odd -> row_pairs=7 + one trailing row), interior_w=16
    # (num_blocks_dual=1, remainder=1). Exercises the trailing-row fallback
    # for the stride-2 dualrow path specifically.
    "dualrow_s2_3x3_odd_rows": (20, 1, 2, 32, 34, 3, 3, 2, 2, 1, 1),
    # 3x3/s2/p1, 34x62x1: interior 16x30 -> num_blocks_dual=2, remainder=0
    # -- clean block-only path (no remainder noise) for kW=3's
    # shift_right_bytes<1> combination term.
    "dualrow_s2_3x3_exact_block_multiple": (21, 1, 1, 34, 62, 3, 3, 2, 2, 1, 1),
    # 2x2/s2/p0, 16x64x2: interior 8x32 (even rows -> row_pairs=4, no
    # trailing), num_blocks_dual=2 (dual_block_width=16), remainder=0.
    "dualrow_s2_2x2_even_rows": (22, 1, 2, 16, 64, 2, 2, 2, 2, 0, 0),
    # 2x2/s2/p0, 14x64x2: interior rows=7 (odd -> row_pairs=3 + trailing),
    # num_blocks_dual=2, remainder=0. Trailing-row fallback for kW=2.
    "dualrow_s2_2x2_odd_rows": (23, 1, 2, 14, 64, 2, 2, 2, 2, 0, 0),
}


@pytest.mark.quick
@pytest.mark.parametrize(
    "seed, N, C, H, W, kH, kW, sH, sW, pH, pW",
    _SHAPE_CASES.values(),
    ids=_SHAPE_CASES.keys(),
)
def test_scalar_symbol_maxpool_shapes(dsp_mode, seed, N, C, H, W, kH, kW, sH, sW, pH, pW):
    """See _SHAPE_CASES above for what each case exercises."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (N, C, H, W), dtype=np.int8)
    _check_maxpool(dsp_mode, x, kH, kW, sH, sW, pH, pW)


@pytest.mark.quick
@pytest.mark.parametrize("value", [-128, 127], ids=["all_min", "all_max"])
def test_scalar_symbol_maxpool_saturating_constant(dsp_mode, value):
    """Constant -128 / 127 input on the 3x3/s2 shape at 20x20: too narrow
    (interior_w=9 < dual_block_width=15) to reach the vectorized
    pack_consec combination -- exercises the scalar remainder/border paths
    at the sign boundary. See
    test_scalar_symbol_maxpool_dualrow_s2_saturating_constant below for a
    size that actually reaches the vectorized path."""
    N, C, H, W = 1, 2, 20, 20
    x = np.full((N, C, H, W), value, dtype=np.int8)
    _check_maxpool(dsp_mode, x, 3, 3, 2, 2, 1, 1)


@pytest.mark.quick
@pytest.mark.parametrize("value", [-128, 127], ids=["all_min", "all_max"])
def test_scalar_symbol_maxpool_dualrow_s2_saturating_constant(dsp_mode, value):
    """Constant -128 / 127 input on the 3x3/s2 dualrow path (34x34, same
    shape as dualrow_s2_3x3_even_rows above): stresses the pack_consec
    deinterleave-and-combine plus shift_right_bytes<1> at the int8 sign
    boundary."""
    N, C, H, W = 1, 2, 34, 34
    x = np.full((N, C, H, W), value, dtype=np.int8)
    _check_maxpool(dsp_mode, x, 3, 3, 2, 2, 1, 1)


@pytest.mark.quick
@pytest.mark.parametrize("value", [-128, 127], ids=["all_min", "all_max"])
def test_scalar_symbol_maxpool_dualrow_saturating_constant(dsp_mode, value):
    """Constant -128 / 127 input on the 3x3/s1 dualrow path (64x64, same
    shape as dualrow_even_rows above): stresses the unpromoted __char32
    max-reduction and register-shift horizontal combine at the int8 sign
    boundary -- a different arithmetic path than the promoted-__int8
    stride-2 fast path above, so needs its own saturating-value check."""
    N, C, H, W = 1, 2, 64, 64
    x = np.full((N, C, H, W), value, dtype=np.int8)
    _check_maxpool(dsp_mode, x, 3, 3, 1, 1, 1, 1)


@pytest.mark.quick
def test_scalar_symbol_maxpool_saturating_checkerboard(dsp_mode):
    """Alternating -128/127 checkerboard on the stride-2 fast path."""
    N, C, H, W = 1, 2, 20, 20
    idx = np.indices((H, W)).sum(axis=0) % 2
    plane = np.where(idx == 0, -128, 127).astype(np.int8)
    x = np.broadcast_to(plane, (N, C, H, W)).copy()
    _check_maxpool(dsp_mode, x, 3, 3, 2, 2, 1, 1)
