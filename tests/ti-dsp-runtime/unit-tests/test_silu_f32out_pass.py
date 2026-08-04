"""Unit tests for FuseQDQToC7xSiluF32Out.

Pure Relax IR-level tests, no hardware or DSP build needed -- mirrors
test_movement_pass.py's convention (build a small model, run the pass,
inspect mod.script()).

Covers the pattern:
  dq(x) -> sigmoid -> multiply(self)   (no trailing quantize)
      -> split / concat                -> c7x_int8_silu_f32out

This is the C2f-block shape: a self-gated SiLU whose float32 result feeds
further movement instead of a quantize. See ti_fuse_qdq_c7x_activation.py's
FuseQDQToC7xSiluF32Out docstring for why this pass must run standalone,
positioned after FuseQDQToC7xMovement in the pipeline, rather than as part
of FuseQDQToC7xActivation's own Round 1: Movement's FPN upsample-concat
pattern matches this exact dq->sigmoid->multiply shape directly when it
feeds a resize2d, and needs to see it before this pass would otherwise
consume it.
"""

import re

import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo


def _run_pass(mod):
    return tvm.relax.transform.FuseQDQToC7xSiluF32Out()(mod)


def _build_silu_split_model(in_shape, d_scale, d_zp, split_index):
    """dq(x) -> sigmoid -> multiply(self) -> split -> two quantized outputs.

    Mirrors the real C2f-block shape: the SiLU'd float32 value is split
    into two channel halves, each independently quantized (as if each half
    fed a different downstream branch)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo(in_shape, "int8"))
    with bb.function("main", [x], attrs={"num_input": 1}):
        with bb.dataflow():
            dq = bb.emit(
                relax.op.dequantize(x, relax.const(d_scale, "float32"), relax.const(d_zp, "int8"))
            )
            sig = bb.emit(relax.op.sigmoid(dq))
            mul = bb.emit(relax.op.multiply(dq, sig))
            split = bb.emit(relax.op.split(mul, indices_or_sections=[split_index], axis=1))
            half0 = bb.emit(relax.TupleGetItem(split, 0))
            half1 = bb.emit(relax.TupleGetItem(split, 1))
            q0 = bb.emit(
                relax.op.quantize(
                    half0, relax.const(0.05, "float32"), relax.const(0, "int8"), out_dtype="int8"
                )
            )
            q1 = bb.emit(
                relax.op.quantize(
                    half1, relax.const(0.06, "float32"), relax.const(1, "int8"), out_dtype="int8"
                )
            )
            out = bb.emit_output(relax.Tuple([q0, q1]))
        bb.emit_func_output(out)
    return bb.finalize()


def _build_silu_concat_model(shape_a, shape_b, d_scale, d_zp):
    """dq(x) -> sigmoid -> multiply(self) -> concat(with another float
    branch) -> quantize. The OTHER concat branch is a bare dequantize,
    matching the real graphs' canonical shape (see
    ti_fuse_qdq_c7x_movement.py's pattern-2 docstring for the analogous
    convention in the FPN case)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo(shape_a, "int8"))
    y = relax.Var("y", TensorStructInfo(shape_b, "int8"))
    with bb.function("main", [x, y], attrs={"num_input": 2}):
        with bb.dataflow():
            dq = bb.emit(
                relax.op.dequantize(x, relax.const(d_scale, "float32"), relax.const(d_zp, "int8"))
            )
            sig = bb.emit(relax.op.sigmoid(dq))
            mul = bb.emit(relax.op.multiply(dq, sig))
            dq_other = bb.emit(
                relax.op.dequantize(y, relax.const(0.04, "float32"), relax.const(0, "int8"))
            )
            cat = bb.emit(relax.op.concat([mul, dq_other], axis=1))
            q = bb.emit(
                relax.op.quantize(
                    cat, relax.const(0.05, "float32"), relax.const(0, "int8"), out_dtype="int8"
                )
            )
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestSiluF32OutSplit:
    def test_fires_and_emits_call_extern(self):
        mod = _build_silu_split_model((1, 32, 20, 20), d_scale=0.03, d_zp=0, split_index=16)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_silu_f32out" in text
        assert "R.sigmoid" not in text
        assert "R.multiply" not in text

    def test_params_match_formula(self):
        mod = _build_silu_split_model((1, 8, 4, 4), d_scale=0.04, d_zp=3, split_index=4)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        m = re.search(
            r'T\.call_extern\("int32", "c7x_int8_silu_f32out", [^,]+, [^,]+, '
            r"(\d+), (-?\d+), T\.float32\(([-\d.eE]+)\)\)",
            text,
        )
        assert m is not None, f"call_extern args not found in:\n{text}"
        n, zx, sx = m.groups()
        assert int(n) == 8 * 4 * 4
        assert int(zx) == 3
        assert float(sx) == pytest.approx(0.04)

    def test_output_is_float32(self):
        """The kernel's own te.extern output must be float32 -- no
        requantize params, unlike c7x_int8_silu's int8 output."""
        mod = _build_silu_split_model((1, 16, 10, 10), d_scale=0.02, d_zp=0, split_index=8)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert 'dtype="float32"' in text

    def test_non_constant_scale_declines_to_fuse(self):
        """Regression guard: when the dequantize's scale is NOT a
        compile-time Constant, the check callback must reject the match."""
        bb = relax.BlockBuilder()
        x = relax.Var("x", TensorStructInfo((1, 8, 4, 4), "int8"))
        s_in = relax.Var("s_in", TensorStructInfo((), "float32"))
        with bb.function("main", [x, s_in], attrs={"num_input": 2}):
            with bb.dataflow():
                # d_scale is a non-constant Var (e.g. derived at runtime),
                # not a relax.Constant -- the check callback must decline.
                d_scale = bb.emit(relax.op.multiply(s_in, relax.const(1.0, "float32")))
                dq = bb.emit(relax.op.dequantize(x, d_scale, relax.const(0, "int8")))
                sig = bb.emit(relax.op.sigmoid(dq))
                mul = bb.emit(relax.op.multiply(dq, sig))
                split = bb.emit(relax.op.split(mul, indices_or_sections=[4], axis=1))
                half0 = bb.emit(relax.TupleGetItem(split, 0))
                half1 = bb.emit(relax.TupleGetItem(split, 1))
                q0 = bb.emit(
                    relax.op.quantize(
                        half0,
                        relax.const(0.05, "float32"),
                        relax.const(0, "int8"),
                        out_dtype="int8",
                    )
                )
                q1 = bb.emit(
                    relax.op.quantize(
                        half1,
                        relax.const(0.06, "float32"),
                        relax.const(0, "int8"),
                        out_dtype="int8",
                    )
                )
                out = bb.emit_output(relax.Tuple([q0, q1]))
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_silu_f32out" not in text


class TestSiluF32OutConcat:
    def test_fires_and_emits_call_extern(self):
        mod = _build_silu_concat_model((1, 8, 4, 4), (1, 4, 4, 4), d_scale=0.03, d_zp=0)
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_silu_f32out" in text
        assert "R.sigmoid" not in text
        # The concat/quantize downstream of the SiLU'd branch is untouched --
        # only its producer chain changed, matching how FuseQDQToC7xMovement's
        # analogous branch-2 handling leaves the surrounding ops in place.
        assert "R.concat" in text
        assert "R.quantize" in text


class TestSiluF32OutMovementOrdering:
    """Regression guard for the ordering bug found during implementation:
    this pattern must not consume a dq->sigmoid->multiply chain that
    FuseQDQToC7xMovement's FPN pattern needs to see raw (feeding a
    resize2d). Confirmed via direct IR inspection of compiled
    yolo26n/yolov8n that running this pattern before Movement makes
    c7x_int8_fpn_upsample_concat stop appearing entirely."""

    def test_movement_runs_first_in_the_real_pipeline(self):
        """Structural check: FuseQDQToC7xSiluF32Out must appear after
        FuseQDQToC7xMovement in the default C7x legalize pass list."""
        import tvm.target
        from tvm.relax.backend.cpu_generic.pipeline import legalize_passes

        target = tvm.target.Target("c_static -mcpu=c7x -mmalib=1")
        passes = legalize_passes(target)
        names = [type(p).__name__ for p in passes]
        # module_pass-decorated classes report their pass name via .info.name,
        # not type(p).__name__ (which is the opaque wrapper) -- use that.
        names = [p.info.name for p in passes]
        assert "FuseQDQToC7xMovement" in names
        assert "FuseQDQToC7xSiluF32Out" in names
        assert names.index("FuseQDQToC7xMovement") < names.index("FuseQDQToC7xSiluF32Out")
