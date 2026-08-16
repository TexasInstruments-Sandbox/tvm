"""Unit tests for FuseQDQToC7xMovement.

Pure Relax IR-level tests, no hardware or DSP build needed -- mirrors
test_input_normalize_quantize_pass.py's convention (build a small model,
run the pass, inspect mod.script()).

Covers the pass's two patterns:
  1. dq(x) -> reshape -> q                         -> c7x_int8_rescale
  2. dq(x1)->sigmoid->multiply->resize2d(2x)->A ;
     dq(x2)->B (bare dequantize, NOT its own SiLU
     diamond -- see the pass module's docstring for
     why the real compiled graphs look like this) ;
     concat([A,B],axis=1)->q                        -> c7x_int8_fpn_upsample_concat
                                                       (one call, not chained --
                                                       see the pass module's
                                                       docstring for why)

There is also a transitively-constant-input regression guard: the
FuseQDQToC7x* cluster shares a systemic const-reachability caveat (see
project memory / ti_c7x_const_reachability.py) where a scale/zp Constant
reached only *through* another op (not passed directly) can fail to be
recognized as compile-time-constant by the pattern's check callback. Both
tests below pass scale/zp as direct relax.const(...) args to dequantize/
quantize (the common, supported case); a dedicated non-constant-input test
confirms the pass correctly declines to fuse rather than mishandling it.
"""

import re

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo


def _run_pass(mod):
    return tvm.relax.transform.FuseQDQToC7xMovement()(mod)


# ---------------------------------------------------------------------------
# Pattern 1: dq -> reshape -> q
# ---------------------------------------------------------------------------


def _build_reshape_model(in_shape, out_shape, d_scale, d_zp, o_scale, o_zp):
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo(in_shape, "int8"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            dq = bb.emit(
                relax.op.dequantize(x, relax.const(d_scale, "float32"), relax.const(d_zp, "int8"))
            )
            reshaped = bb.emit(relax.op.reshape(dq, out_shape))
            q = bb.emit(
                relax.op.quantize(
                    reshaped,
                    relax.const(o_scale, "float32"),
                    relax.const(o_zp, "int8"),
                    out_dtype="int8",
                )
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestReshapeRescale:
    def test_fires_and_emits_call_extern(self):
        mod = _build_reshape_model(
            (1, 8, 4, 4), (1, 128), d_scale=0.03, d_zp=2, o_scale=0.05, o_zp=-1
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_rescale" in text
        assert "R.reshape" not in text
        assert "R.dequantize" not in text
        assert "R.quantize" not in text

    def test_params_match_formula(self):
        mod = _build_reshape_model((1, 4, 2, 2), (16,), d_scale=0.04, d_zp=3, o_scale=0.02, o_zp=-2)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        m = re.search(
            r'T\.call_extern\("int32", "c7x_int8_rescale", [^,]+, [^,]+, '
            r"(\d+), (-?\d+), T\.float32\(([-\d.eE]+)\), "
            r"(-?\d+), T\.float32\(([-\d.eE]+)\)\)",
            text,
        )
        assert m is not None, f"call_extern args not found in:\n{text}"
        n, zx, sx, zy, sy = m.groups()
        assert int(n) == 16
        assert int(zx) == 3
        assert float(sx) == pytest.approx(0.04)
        assert int(zy) == -2
        assert float(sy) == pytest.approx(0.02)

    def test_transparent_still_fuses(self):
        """Matching scale/zp: EliminateQDQTransparent normally removes this
        upstream, but if it reaches this pass anyway (e.g. a standalone
        unit test), the pass should still fuse it -- the kernel's own
        memcpy fast path handles the transparent case correctly."""
        mod = _build_reshape_model((1, 4, 2, 2), (16,), d_scale=0.03, d_zp=0, o_scale=0.03, o_zp=0)
        new_mod = _run_pass(mod)
        assert "c7x_int8_rescale" in new_mod.script()

    def test_non_constant_scale_declines_to_fuse(self):
        """Regression guard: when the output quantize's scale is NOT a
        compile-time Constant (e.g. it flows through another op instead of
        being passed directly), the check callback must reject the match --
        the pass has no way to derive scale_q/offset at compile time.
        Confirms decline-to-fuse, not a mis-fuse with fabricated values."""
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        s_in = relax.Var("s_in", TensorStructInfo((), "float32"))
        with bb.function("main", [x, s_in], attrs={"num_input": 2}):
            with bb.dataflow():
                dq = bb.emit(
                    relax.op.dequantize(x, relax.const(0.03, "float32"), relax.const(0, "int8"))
                )
                reshaped = bb.emit(relax.op.reshape(dq, (16,)))
                # o_scale is a non-constant Var (e.g. derived at runtime),
                # not a relax.Constant -- the check callback must decline.
                o_scale = bb.emit(relax.op.multiply(s_in, relax.const(1.0, "float32")))
                q = bb.emit(
                    relax.op.quantize(reshaped, o_scale, relax.const(0, "int8"), out_dtype="int8")
                )
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_rescale" not in text

    def test_constant_tensor_input_declines_and_still_compiles(self):
        """When x is itself a literal relax.Constant tensor -- a real,
        already-supported match per _check_single_input's isinstance(x,
        relax.Constant) branch, not merely reachable from constants -- the
        ConstReachability guard declines to fuse. Confirms both the decline
        AND that the module still compiles all the way through LegalizeOps
        -> FoldConstant -> FuseOps -> FuseTIR.

        This exercises the _decline path (verified: one _decline hit,
        composite c7x_movement.reshape, composite gone after the pass). It is
        NOT a regression guard for the inline-on-decline fix: with
        inline_declined_composite neutered so the decline leaves the
        composite call in place, this module still compiles through FuseTIR
        without raising. FuseOpsByPattern(bind_constants=False) lifts the
        Constant to a composite parameter rather than embedding it in the
        body, so the "Relax.Constant is not supported in primitive
        functions" path is not reached here. Do not cite this test as
        evidence that leaving a declined composite in place is unsafe."""
        const_x = relax.const(np.arange(16, dtype="int8").reshape(1, 4, 2, 2))
        bb = relax.BlockBuilder()
        with bb.function("main", [], attrs={"num_input": 0}):
            with bb.dataflow():
                dq = bb.emit(
                    relax.op.dequantize(
                        const_x, relax.const(0.03, "float32"), relax.const(0, "int8")
                    )
                )
                reshaped = bb.emit(relax.op.reshape(dq, (16,)))
                q = bb.emit(
                    relax.op.quantize(
                        reshaped,
                        relax.const(0.05, "float32"),
                        relax.const(-1, "int8"),
                        out_dtype="int8",
                    )
                )
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()

        new_mod = _run_pass(mod)
        assert "c7x_int8_rescale" not in new_mod.script()

        new_mod = relax.transform.LegalizeOps()(new_mod)
        new_mod = relax.transform.FoldConstant()(new_mod)
        new_mod = relax.transform.FuseOps()(new_mod)
        relax.transform.FuseTIR()(new_mod)  # must not raise


# ---------------------------------------------------------------------------
# Pattern 2: FPN upsample-concat
# ---------------------------------------------------------------------------


def _build_fpn_model(C1, C2, H, W, s1, z1, s2, z2, o_scale, o_zp, resize_first=True):
    bb = relax.BlockBuilder()
    x1 = relax.Var("x1", TensorStructInfo((1, C1, H, W), "int8"))
    x2 = relax.Var("x2", TensorStructInfo((1, C2, 2 * H, 2 * W), "int8"))
    with bb.function("main", [x1, x2], attrs={"num_input": 2}):
        with bb.dataflow():
            dq1 = bb.emit(
                relax.op.dequantize(x1, relax.const(s1, "float32"), relax.const(z1, "int8"))
            )
            sig1 = bb.emit(relax.op.sigmoid(dq1))
            mul1 = bb.emit(relax.op.multiply(dq1, sig1))
            resized = bb.emit(
                relax.op.image.resize2d(
                    mul1,
                    (2 * H, 2 * W),
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                )
            )

            # Branch 2 is a BARE dequantize, not its own sigmoid->multiply
            # SiLU diamond -- matches what the real compiled graphs actually
            # look like by the time this pass runs (see the pass module's
            # pattern-2 docstring for why).
            dq2 = bb.emit(
                relax.op.dequantize(x2, relax.const(s2, "float32"), relax.const(z2, "int8"))
            )

            fields = [resized, dq2] if resize_first else [dq2, resized]
            cat = bb.emit(relax.op.concat(fields, axis=1))
            q = bb.emit(
                relax.op.quantize(
                    cat,
                    relax.const(o_scale, "float32"),
                    relax.const(o_zp, "int8"),
                    out_dtype="int8",
                )
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_fpn_model_shared_branch1(
    C1, C2, H, W, s1, z1, s2, z2, o_scale, o_zp, extra_scale, extra_zp
):
    """Same FPN structure as _build_fpn_model, but branch 1's SiLU output
    (mul1) has a SECOND consumer outside the concat -- forces
    FuseOpsByPattern to promote it to an extra tuple output of the matched
    composite (the "is_tuple_out" case), matching what happens on the real
    compiled yolov8n/yolo26n graphs (see the pass module's pattern-2
    docstring)."""
    bb = relax.BlockBuilder()
    x1 = relax.Var("x1", TensorStructInfo((1, C1, H, W), "int8"))
    x2 = relax.Var("x2", TensorStructInfo((1, C2, 2 * H, 2 * W), "int8"))
    with bb.function("main", [x1, x2], attrs={"num_input": 2}):
        with bb.dataflow():
            dq1 = bb.emit(
                relax.op.dequantize(x1, relax.const(s1, "float32"), relax.const(z1, "int8"))
            )
            sig1 = bb.emit(relax.op.sigmoid(dq1))
            mul1 = bb.emit(relax.op.multiply(dq1, sig1))
            resized = bb.emit(
                relax.op.image.resize2d(
                    mul1,
                    (2 * H, 2 * W),
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                )
            )
            dq2 = bb.emit(
                relax.op.dequantize(x2, relax.const(s2, "float32"), relax.const(z2, "int8"))
            )
            cat = bb.emit(relax.op.concat([resized, dq2], axis=1))
            q = bb.emit(
                relax.op.quantize(
                    cat,
                    relax.const(o_scale, "float32"),
                    relax.const(o_zp, "int8"),
                    out_dtype="int8",
                )
            )
            # Second consumer of mul1, outside the concat -- gives it fan-out 2.
            q_extra = bb.emit(
                relax.op.quantize(
                    mul1,
                    relax.const(extra_scale, "float32"),
                    relax.const(extra_zp, "int8"),
                    out_dtype="int8",
                )
            )
            out = bb.emit_output(relax.Tuple([q, q_extra]))
        bb.emit_func_output(out)
    return bb.finalize()


class TestFPNUpsampleConcat:
    def test_fires_and_emits_call_extern(self):
        mod = _build_fpn_model(
            C1=8, C2=4, H=4, W=4, s1=0.03, z1=0, s2=0.04, z2=0, o_scale=0.05, o_zp=0
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert text.count('T.call_extern("int32", "c7x_int8_fpn_upsample_concat"') == 1
        assert "R.image.resize2d" not in text
        assert "R.concat" not in text

    def test_shared_branch1_uses_ex_kernel(self):
        """Branch 1's SiLU output also feeds a second, unrelated quantize
        outside the concat -- forces the "is_tuple_out" path
        (c7x_int8_fpn_upsample_concat_ex + a reconstructed float32
        companion for the other consumer), matching the real compiled
        yolov8n/yolo26n graphs rather than the simpler single-consumer case
        the other test above covers."""
        mod = _build_fpn_model_shared_branch1(
            C1=8,
            C2=4,
            H=4,
            W=4,
            s1=0.03,
            z1=0,
            s2=0.04,
            z2=0,
            o_scale=0.05,
            o_zp=0,
            extra_scale=0.02,
            extra_zp=1,
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert 'T.call_extern("int32", "c7x_int8_fpn_upsample_concat_ex"' in text
        assert "R.image.resize2d" not in text
        assert "R.sigmoid" not in text
        # The second quantize (of the reconstructed float32 companion)
        # must still be present -- its consumer wasn't touched, only its
        # producer chain changed.
        assert text.count("R.quantize") == 1  # the reconstructed-companion one; main uses call_tir
        assert "R.sigmoid" not in text

    def test_field_order_reversed_declines_to_fuse(self):
        """concat([skip, resized]) -- the resize2d branch is field 1, not
        field 0. Only the canonical (resize2d-first) tuple order is
        supported (the kernel always places branch 1 first in the output);
        a commuted pattern variant was tried and reverted because
        registering it alongside the canonical pattern crashes TVM's
        FuseOpsByPattern grouping on real (many-concat) graphs like
        yolov8n/yolo26n, even though it works on this kind of small
        isolated graph -- see the pass module's comment above
        _PATTERN_REGISTRY. Both confirmed real FPN upsample sites always
        put resize2d first, so declining to fuse the reversed order (falling
        through to the generic path, still correct) is the safe trade-off."""
        mod = _build_fpn_model(
            C1=8,
            C2=4,
            H=4,
            W=4,
            s1=0.03,
            z1=0,
            s2=0.04,
            z2=0,
            o_scale=0.05,
            o_zp=0,
            resize_first=False,
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_fpn_upsample_concat" not in text

    def test_non_2x_resize_declines_to_fuse(self):
        """A resize2d that ISN'T an exact 2x upsample must fall through
        unchanged -- the kernel only implements the 2x case."""
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", TensorStructInfo((1, 8, 4, 4), "int8"))
        x2 = relax.Var("x2", TensorStructInfo((1, 4, 10, 10), "int8"))
        with bb.function("main", [x1, x2], attrs={"num_input": 2}):
            with bb.dataflow():
                dq1 = bb.emit(
                    relax.op.dequantize(x1, relax.const(0.03, "float32"), relax.const(0, "int8"))
                )
                sig1 = bb.emit(relax.op.sigmoid(dq1))
                mul1 = bb.emit(relax.op.multiply(dq1, sig1))
                resized = bb.emit(
                    relax.op.image.resize2d(
                        mul1,
                        (10, 10),  # 2.5x, not 2x
                        layout="NCHW",
                        method="nearest_neighbor",
                        coordinate_transformation_mode="half_pixel",
                        rounding_method="round",
                    )
                )
                dq2 = bb.emit(
                    relax.op.dequantize(x2, relax.const(0.04, "float32"), relax.const(0, "int8"))
                )
                cat = bb.emit(relax.op.concat([resized, dq2], axis=1))
                q = bb.emit(
                    relax.op.quantize(
                        cat,
                        relax.const(0.05, "float32"),
                        relax.const(0, "int8"),
                        out_dtype="int8",
                    )
                )
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_fpn_upsample_concat" not in text

    def test_constant_tensor_branch_declines_and_still_compiles(self):
        """Same decline check as TestReshapeRescale's version above, for the
        FPN upsample-concat pattern: branch 2's data operand is itself a
        literal relax.Constant tensor. See that test's docstring, including
        its caveat that this is a decline-path + still-compiles check, not a
        regression guard for the inline-on-decline fix."""
        C1, C2, H, W = 8, 4, 4, 4
        const_x2 = relax.const(
            np.arange(C2 * 2 * H * 2 * W, dtype="int8").reshape(1, C2, 2 * H, 2 * W)
        )
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", TensorStructInfo((1, C1, H, W), "int8"))
        with bb.function("main", [x1], attrs={"num_input": 1}):
            with bb.dataflow():
                dq1 = bb.emit(
                    relax.op.dequantize(x1, relax.const(0.03, "float32"), relax.const(0, "int8"))
                )
                sig1 = bb.emit(relax.op.sigmoid(dq1))
                mul1 = bb.emit(relax.op.multiply(dq1, sig1))
                resized = bb.emit(
                    relax.op.image.resize2d(
                        mul1,
                        (2 * H, 2 * W),
                        layout="NCHW",
                        method="nearest_neighbor",
                        coordinate_transformation_mode="half_pixel",
                        rounding_method="round",
                    )
                )
                dq2 = bb.emit(
                    relax.op.dequantize(
                        const_x2, relax.const(0.04, "float32"), relax.const(0, "int8")
                    )
                )
                cat = bb.emit(relax.op.concat([resized, dq2], axis=1))
                q = bb.emit(
                    relax.op.quantize(
                        cat,
                        relax.const(0.05, "float32"),
                        relax.const(0, "int8"),
                        out_dtype="int8",
                    )
                )
                out = bb.emit_output(q)
            bb.emit_func_output(out)
        mod = bb.finalize()

        new_mod = _run_pass(mod)
        assert "c7x_int8_fpn_upsample_concat" not in new_mod.script()

        new_mod = relax.transform.LegalizeOps()(new_mod)
        new_mod = relax.transform.FoldConstant()(new_mod)
        new_mod = relax.transform.FuseOps()(new_mod)
        relax.transform.FuseTIR()(new_mod)  # must not raise
