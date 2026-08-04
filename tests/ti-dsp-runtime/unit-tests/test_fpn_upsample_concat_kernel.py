"""Unit tests for the c7x_int8_fpn_upsample_concat kernel.

Invokes the kernel via call_extern with known inputs and verifies output
against a numpy reference.  Tests are independent of the
FuseQDQToC7xMovement pass.

c7x_int8_fpn_upsample_concat: SiLU(in1) upsampled 2x nearest, concatenated
with a plain rescale of in2 (NOT SiLU -- see c7x_rescale.h for why: in2 is
already the output of an independently-lowered c7x_int8_silu call by the
time this kernel's caller runs), along the channel axis (in1's channels
first):

  out[0:C1, 2h+dh, 2w+dw] = quant(silu(dequant(in1[c,h,w])))   dh,dw in {0,1}
  out[C1:C1+C2, h2, w2]   = quant(dequant(in2[c,h2,w2]))

Both branches target (s_out, z_out) directly. in2 must already be at the
upsampled (2H, 2W) spatial size.

Usage:
    pytest test_fpn_upsample_concat_kernel.py -v --dsp-mode=c7x_host
    pytest test_fpn_upsample_concat_kernel.py -v --dsp-mode=c7x_dload
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

_KERNEL = "c7x_int8_fpn_upsample_concat"


def _numpy_silu_requant(x_i8, zx, sx, zy, sy):
    """Plain (non-vectorized) scalar SiLU + requantize, matching the
    kernel's scalar dq_f/rq_f + expf-based implementation exactly (this
    kernel has no SE-vectorized path -- see c7x_rescale.h)."""
    xf = (x_i8.astype(np.float64) - zx) * sx
    yf = xf / (1.0 + np.exp(-xf))
    v = np.trunc(yf / sy + 0.5).astype(np.int64) + zy
    return np.clip(v, -128, 127).astype(np.int8)


def _numpy_rescale_requant(x_i8, zx, sx, zy, sy):
    """Plain affine rescale, no SiLU -- matches branch 2's kernel math.
    dq_f/rq_f are pure float32 in the kernel (C `float`), so this must
    stay in float32 throughout to match exactly at rounding boundaries
    (same rationale as test_silu_kernel.py's bulk-path note)."""
    f32 = np.float32
    xf = (x_i8.astype(f32) - f32(zx)) * f32(sx)
    v = np.trunc(xf / f32(sy) + f32(0.5)).astype(np.int64) + zy
    return np.clip(v, -128, 127).astype(np.int8)


def _numpy_fpn_upsample_concat(in1_chw, z1, s1, in2_chw, z2, s2, s_out, z_out):
    """in1_chw: [C1, H, W] int8. in2_chw: [C2, 2H, 2W] int8."""
    branch1 = _numpy_silu_requant(in1_chw, z1, s1, z_out, s_out)
    branch1_up = np.repeat(np.repeat(branch1, 2, axis=1), 2, axis=2)
    branch2 = _numpy_rescale_requant(in2_chw, z2, s2, z_out, s_out)
    return np.concatenate([branch1_up, branch2], axis=0)


def _build_fpn_module(C1, H, W, z1, s1, C2, z2, s2, s_out, z_out):
    C1_v, H_v, W_v = int(C1), int(H), int(W)
    C2_v = int(C2)
    z1_v, s1_v, z2_v, s2_v = int(z1), float(s1), int(z2), float(s2)
    s_out_v, z_out_v = float(s_out), int(z_out)
    H2_v, W2_v = 2 * H_v, 2 * W_v

    def te_kernel(t1, t2):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL,
                ins[0].data,
                tir.IntImm("int32", C1_v),
                tir.IntImm("int32", H_v),
                tir.IntImm("int32", W_v),
                tir.IntImm("int32", z1_v),
                tir.FloatImm("float32", s1_v),
                ins[1].data,
                tir.IntImm("int32", C2_v),
                tir.IntImm("int32", z2_v),
                tir.FloatImm("float32", s2_v),
                outs[0].data,
                tir.FloatImm("float32", s_out_v),
                tir.IntImm("int32", z_out_v),
            )

        return te.extern(
            [C1_v + C2_v, H2_v, W2_v], [t1, t2], fcompute, name="fpn_out", dtype="int8"
        )

    bb = relax.BlockBuilder()
    v1 = relax.Var("in1", relax.TensorStructInfo([C1_v, H_v, W_v], "int8"))
    v2 = relax.Var("in2", relax.TensorStructInfo([C2_v, H2_v, W2_v], "int8"))
    with bb.function("main", [v1, v2], attrs={"num_input": 2}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, v1, v2, primfunc_name_hint=_KERNEL)
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_fpn(dsp_mode, in1_chw, z1, s1, in2_chw, z2, s2, s_out, z_out):
    C1, H, W = in1_chw.shape
    C2 = in2_chw.shape[0]
    mod = _build_fpn_module(C1, H, W, z1, s1, C2, z2, s2, s_out, z_out)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[in1_chw, in2_chw],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_fpn_small(dsp_mode):
    """C1=2, C2=1, H=W=4 -- small, easy to hand-verify shape correctness."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(0)
    C1, C2, H, W = 2, 1, 4, 4
    in1 = rng.integers(-128, 127, (C1, H, W), dtype=np.int8)
    in2 = rng.integers(-128, 127, (C2, 2 * H, 2 * W), dtype=np.int8)
    z1, s1, z2, s2, s_out, z_out = 0, 0.03, 0, 0.04, 0.05, 0
    ref = _numpy_fpn_upsample_concat(in1, z1, s1, in2, z2, s2, s_out, z_out)
    out, _ = _run_fpn(dsp_mode, in1, z1, s1, in2, z2, s2, s_out, z_out)
    out = out.reshape(C1 + C2, 2 * H, 2 * W)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


@pytest.mark.quick
def test_fpn_asymmetric_zp(dsp_mode):
    """Non-zero zero-points on both branches and the output."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(1)
    C1, C2, H, W = 3, 2, 5, 5
    in1 = rng.integers(-128, 127, (C1, H, W), dtype=np.int8)
    in2 = rng.integers(-128, 127, (C2, 2 * H, 2 * W), dtype=np.int8)
    z1, s1, z2, s2, s_out, z_out = -5, 0.02, 3, 0.045, 0.03, 2
    ref = _numpy_fpn_upsample_concat(in1, z1, s1, in2, z2, s2, s_out, z_out)
    out, _ = _run_fpn(dsp_mode, in1, z1, s1, in2, z2, s2, s_out, z_out)
    out = out.reshape(C1 + C2, 2 * H, 2 * W)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"


@pytest.mark.core
def test_fpn_yolo26_10to20(dsp_mode, record_cycles):
    """C1=256, C2=128, 10x10 -> 20x20 -- yolo26n's actual P5->P4 FPN shape."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(2)
    C1, C2, H, W = 256, 128, 10, 10
    in1 = rng.integers(-128, 127, (C1, H, W), dtype=np.int8)
    in2 = rng.integers(-128, 127, (C2, 2 * H, 2 * W), dtype=np.int8)
    z1, s1, z2, s2, s_out, z_out = 0, 0.058583375066518784, 0, 0.04, 0.058583375066518784, 0
    ref = _numpy_fpn_upsample_concat(in1, z1, s1, in2, z2, s2, s_out, z_out)
    out, cycles = _run_fpn(dsp_mode, in1, z1, s1, in2, z2, s2, s_out, z_out)
    record_cycles("fpn_upsample_concat_yolo26_10to20", cycles)
    out = out.reshape(C1 + C2, 2 * H, 2 * W)
    assert np.array_equal(out, ref), f"max_err={np.abs(out.astype(int) - ref.astype(int)).max()}"
    if cycles:
        n = (C1 + C2) * (2 * H) * (2 * W)
        print(
            f"\n  c7x_int8_fpn_upsample_concat C1={C1} C2={C2} 10->20: "
            f"{cycles:,} cycles ({cycles / n:.2f} cycles/output-element)"
        )


# ---------------------------------------------------------------------------
# c7x_int8_fpn_upsample_concat_ex: same, plus branch 1's pre-upsample
# quantized SiLU value as a second output (the "is_tuple_out" case -- see
# c7x_rescale.h and ti_fuse_qdq_c7x_movement.py's pattern-2 docstring).
# ---------------------------------------------------------------------------

_KERNEL_EX = "c7x_int8_fpn_upsample_concat_ex"


def _numpy_presize_silu_f32(in1_chw, z1, s1):
    """Branch 1's SiLU value at the pre-upsample [C1,H,W] size -- the
    second output. This is the EXACT float32 SiLU value (dq -> x*sigmoid(x)),
    NOT requantized to the output scale: the _ex kernel emits it as float32
    so a downstream consumer gets it losslessly. Matches the kernel's
    float32 dq_f + expf math (float32 throughout, same as the branch-2
    rescale reference's rationale)."""
    f32 = np.float32
    xf = (in1_chw.astype(f32) - f32(z1)) * f32(s1)
    yf = xf / (f32(1.0) + np.exp(-xf).astype(f32))
    return yf.astype(f32)


def _build_fpn_module_ex(C1, H, W, z1, s1, C2, z2, s2, s_out, z_out):
    C1_v, H_v, W_v = int(C1), int(H), int(W)
    C2_v = int(C2)
    z1_v, s1_v, z2_v, s2_v = int(z1), float(s1), int(z2), float(s2)
    s_out_v, z_out_v = float(s_out), int(z_out)
    H2_v, W2_v = 2 * H_v, 2 * W_v

    def te_kernel(t1, t2):
        def fcompute(ins, outs):
            return tir.call_extern(
                "int32",
                _KERNEL_EX,
                ins[0].data,
                tir.IntImm("int32", C1_v),
                tir.IntImm("int32", H_v),
                tir.IntImm("int32", W_v),
                tir.IntImm("int32", z1_v),
                tir.FloatImm("float32", s1_v),
                ins[1].data,
                tir.IntImm("int32", C2_v),
                tir.IntImm("int32", z2_v),
                tir.FloatImm("float32", s2_v),
                outs[0].data,
                tir.FloatImm("float32", s_out_v),
                tir.IntImm("int32", z_out_v),
                outs[1].data,
            )

        out_buf0 = tir.decl_buffer(
            [C1_v + C2_v, H2_v, W2_v], "int8", "fpn_out_ex_main", data_alignment=8
        )
        out_buf1 = tir.decl_buffer(
            [C1_v, H_v, W_v], "float32", "fpn_out_ex_presize", data_alignment=8
        )
        return te.extern(
            [[C1_v + C2_v, H2_v, W2_v], [C1_v, H_v, W_v]],
            [t1, t2],
            fcompute,
            out_buffers=[out_buf0, out_buf1],
            name="fpn_out_ex",
            dtype=["int8", "float32"],
        )

    bb = relax.BlockBuilder()
    v1 = relax.Var("in1", relax.TensorStructInfo([C1_v, H_v, W_v], "int8"))
    v2 = relax.Var("in2", relax.TensorStructInfo([C2_v, H2_v, W2_v], "int8"))
    with bb.function("main", [v1, v2], attrs={"num_input": 2}):
        with bb.dataflow():
            result = bb.emit_te(te_kernel, v1, v2, primfunc_name_hint=_KERNEL_EX)
            # Route both tuple fields through TupleGetItem + reshape + concat
            # into ONE flat single-tensor output, matching how the real pass
            # (ti_fuse_qdq_c7x_movement.py) actually uses this kernel: the
            # tuple is always an *internal* Relax value immediately consumed
            # via TupleGetItem by separate downstream ops within a much
            # larger function, never itself a compiled function's own
            # top-level return value. Returning the raw 2-different-shape
            # tuple directly as this test module's own output hits an
            # unrelated c_static/DLOAD limitation with multi-shape top-level
            # function outputs (confirmed via direct testing -- not
            # reproducible in the real yolov8n/yolo26n models, where this
            # exact kernel + tuple-output path is proven correct on real
            # hardware) that has nothing to do with the kernel itself; this
            # concat sidesteps it while still exercising both kernel outputs.
            main_field = bb.emit(relax.TupleGetItem(result, 0))
            presize_field = bb.emit(relax.TupleGetItem(result, 1))
            main_flat = bb.emit(relax.op.reshape(main_field, [(C1_v + C2_v) * H2_v * W2_v]))
            presize_flat = bb.emit(relax.op.reshape(presize_field, [C1_v * H_v * W_v]))
            # main is int8, presize is now float32. Promote main to float32
            # (lossless -- every int8 value is exactly representable) so both
            # fields share ONE flat output tensor, keeping this test's
            # single-tensor top-level return (which sidesteps the DLOAD
            # multi-shape/dtype tuple-return limitation described above) while
            # still exercising both kernel outputs.
            main_flat_f = bb.emit(relax.op.astype(main_flat, "float32"))
            combined = bb.emit(relax.op.concat([main_flat_f, presize_flat], axis=0))
            out = bb.emit_output(combined)
        bb.emit_func_output(out)
    return bb.finalize()


def _run_fpn_ex(dsp_mode, in1_chw, z1, s1, in2_chw, z2, s2, s_out, z_out):
    C1, H, W = in1_chw.shape
    C2 = in2_chw.shape[0]
    mod = _build_fpn_module_ex(C1, H, W, z1, s1, C2, z2, s2, s_out, z_out)
    target = get_target_string(dsp_mode, use_cpp_api=True)
    results = compile_and_run_dsp(
        mod=mod,
        input_data=[in1_chw, in2_chw],
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )
    return results[f"{dsp_mode}_result"], results.get("c7x_dload_cycles", 0)


def _split_combined(combined, C1, C2, H, W):
    n_main = (C1 + C2) * 2 * H * 2 * W
    flat = np.asarray(combined).flatten()
    # main was promoted int8 -> float32 (lossless); round back to int8.
    out_main = np.rint(flat[:n_main]).astype(np.int8).reshape(C1 + C2, 2 * H, 2 * W)
    out_presize = flat[n_main:].astype(np.float32).reshape(C1, H, W)
    return out_main, out_presize


@pytest.mark.quick
def test_fpn_ex_small(dsp_mode):
    """Both outputs must be correct: the concat result (same as the plain
    kernel) and the pre-upsample branch-1-only SiLU value."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(3)
    C1, C2, H, W = 2, 1, 4, 4
    in1 = rng.integers(-128, 127, (C1, H, W), dtype=np.int8)
    in2 = rng.integers(-128, 127, (C2, 2 * H, 2 * W), dtype=np.int8)
    z1, s1, z2, s2, s_out, z_out = 0, 0.03, 0, 0.04, 0.05, 0
    ref_main = _numpy_fpn_upsample_concat(in1, z1, s1, in2, z2, s2, s_out, z_out)
    ref_presize = _numpy_presize_silu_f32(in1, z1, s1)
    combined, _ = _run_fpn_ex(dsp_mode, in1, z1, s1, in2, z2, s2, s_out, z_out)
    out_main, out_presize = _split_combined(combined, C1, C2, H, W)
    assert np.array_equal(out_main, ref_main), (
        f"main output mismatch, max_err={np.abs(out_main.astype(int) - ref_main.astype(int)).max()}"
    )
    # presize is exact float32 SiLU -- compare with a small tolerance for the
    # difference between the kernel's expf() and numpy's exp().
    assert np.allclose(out_presize, ref_presize, rtol=1e-5, atol=1e-5), (
        f"presize output mismatch, max_err={np.abs(out_presize - ref_presize).max()}"
    )


@pytest.mark.quick
def test_fpn_ex_asymmetric_zp(dsp_mode):
    """Non-zero zero-points on both branches and the output, both outputs checked."""
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    rng = np.random.default_rng(4)
    C1, C2, H, W = 3, 2, 5, 5
    in1 = rng.integers(-128, 127, (C1, H, W), dtype=np.int8)
    in2 = rng.integers(-128, 127, (C2, 2 * H, 2 * W), dtype=np.int8)
    z1, s1, z2, s2, s_out, z_out = -5, 0.02, 3, 0.045, 0.03, 2
    ref_main = _numpy_fpn_upsample_concat(in1, z1, s1, in2, z2, s2, s_out, z_out)
    ref_presize = _numpy_presize_silu_f32(in1, z1, s1)
    combined, _ = _run_fpn_ex(dsp_mode, in1, z1, s1, in2, z2, s2, s_out, z_out)
    out_main, out_presize = _split_combined(combined, C1, C2, H, W)
    assert np.array_equal(out_main, ref_main)
    assert np.allclose(out_presize, ref_presize, rtol=1e-5, atol=1e-5), (
        f"presize output mismatch, max_err={np.abs(out_presize - ref_presize).max()}"
    )
