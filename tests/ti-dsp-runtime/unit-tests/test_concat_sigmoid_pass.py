"""Unit tests for FuseQDQToC7xConcat's concat_sigmoid composite.

Pure Relax IR-level tests, no hardware or DSP build needed -- mirrors
test_movement_pass.py's / test_dfl_softmax_pass.py's convention.

Covers the pattern:
  dq(reshape(x1,_),s1,z1)
  dq(reshape(x2,_),s2,z2)   -> concat(axis=-1) -> sigmoid
  dq(reshape(x3,_),s3,z3)                          -> c7x_int8_concat_sigmoid

This is the YOLO multi-scale class-score glue (per-detection-scale int8
NCHW conv outputs, flattened and concatenated along the anchor axis, then
a bare sigmoid with no self-multiply and no trailing quantize); see
ti_fuse_qdq_c7x_concat.py's _make_concat_sigmoid_pattern docstring and
yolo_head_qdq_movement_fusion.md's Step 4.

There is also a transitively-constant-input regression guard (see
test_movement_pass.py's module docstring / project memory /
ti_c7x_const_reachability.py): a dedicated non-constant-input test confirms
the pass declines to fuse rather than mishandling it.
"""

import re

import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo


def _run_pass(mod):
    return tvm.relax.transform.FuseQDQToC7xConcat()(mod)


def _build_concat_sigmoid_model(branch_shapes, d_scales, d_zps, concat_axis=-1, use_relu=False):
    """branch_shapes: list of (C, H, W) raw NCHW shapes, one per branch --
    each is reshaped to [1, C, H*W] before dequantize+concat+sigmoid.

    use_relu swaps the composite root from sigmoid to relu, to test that
    the pattern requires a bare sigmoid specifically.
    """
    bb = relax.BlockBuilder()
    xs = [
        relax.Var(f"x{i}", TensorStructInfo((1, C, H, W), "int8"))
        for i, (C, H, W) in enumerate(branch_shapes)
    ]
    with bb.function("main", xs, attrs={"num_input": len(xs)}):
        with bb.dataflow():
            dqs = []
            for i, (x, (C, H, W)) in enumerate(zip(xs, branch_shapes)):
                r = bb.emit(relax.op.reshape(x, (1, C, H * W)))
                dq = bb.emit(
                    relax.op.dequantize(
                        r, relax.const(d_scales[i], "float32"), relax.const(d_zps[i], "int8")
                    )
                )
                dqs.append(dq)
            cat = bb.emit(relax.op.concat(dqs, axis=concat_axis))
            result = bb.emit(relax.op.nn.relu(cat)) if use_relu else bb.emit(relax.op.sigmoid(cat))
            out = bb.emit_output(result)
        bb.emit_func_output(out)
    return bb.finalize()


class TestConcatSigmoid:
    def test_fires_and_emits_call_extern(self):
        mod = _build_concat_sigmoid_model(
            [(4, 4, 4), (4, 2, 2), (4, 1, 1)],
            d_scales=[0.1, 0.2, 0.3],
            d_zps=[0, 0, 0],
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_concat_sigmoid" in text
        assert "R.reshape" not in text
        assert "R.dequantize" not in text
        assert "R.concat" not in text
        assert "R.sigmoid" not in text

    def test_params_match_formula(self):
        mod = _build_concat_sigmoid_model(
            [(3, 2, 5), (3, 1, 4)],
            d_scales=[0.07, 0.11],
            d_zps=[2, -3],
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        m = re.search(
            r'T\.call_extern\("int32", "c7x_int8_concat_sigmoid", [^,]+, '
            r"(\d+), T\.float32\(([-\d.eE]+)\), (-?\d+), "
            r"[^,]+, (\d+), T\.float32\(([-\d.eE]+)\), (-?\d+), "
            r"[^,]+, (\d+), T\.float32\(([-\d.eE]+)\), (-?\d+), "
            r"[^,]+, (\d+), T\.float32\(([-\d.eE]+)\), (-?\d+), "
            r"[^,]+, (\d+)\)",
            text,
        )
        assert m is not None, f"call_extern args not found in:\n{text}"
        (
            n0,
            s0,
            z0,
            n1,
            s1,
            z1,
            n2,
            _s2,
            _z2,
            n3,
            _s3,
            _z3,
            C,
        ) = m.groups()
        assert (int(n0), int(n1)) == (10, 4)
        assert float(s0) == pytest.approx(0.07)
        assert int(z0) == 2
        assert float(s1) == pytest.approx(0.11)
        assert int(z1) == -3
        # Padded slots (n_i=0) for the unused 3rd/4th input.
        assert (int(n2), int(n3)) == (0, 0)
        assert int(C) == 3

    def test_two_branch_arity(self):
        mod = _build_concat_sigmoid_model(
            [(5, 3, 3), (5, 2, 2)],
            d_scales=[0.05, 0.09],
            d_zps=[0, 1],
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_concat_sigmoid" in text

    def test_yolo26n_real_shape(self):
        """The real yolo26n/yolov8n multi-scale class-score shape: C=80,
        n=[1600,400,100] from P3/P4/P5 at 40x40/20x20/10x10."""
        mod = _build_concat_sigmoid_model(
            [(80, 40, 40), (80, 20, 20), (80, 10, 10)],
            d_scales=[0.22449895739555359, 0.54564881324768066, 0.92429441213607788],
            d_zps=[0, 0, 0],
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_concat_sigmoid" in text
        main = new_mod["main"]
        out_sinfo = main.ret_struct_info
        assert [int(s) for s in out_sinfo.shape] == [1, 80, 2100]
        assert str(out_sinfo.dtype) == "float32"

    def test_wrong_axis_declines_to_fuse(self):
        """Channel-axis (axis=1) concat feeding a bare sigmoid isn't the
        anchor-axis glue this kernel implements -- must fall through."""
        mod = _build_concat_sigmoid_model(
            [(4, 2, 2), (6, 2, 2)],
            d_scales=[0.1, 0.2],
            d_zps=[0, 0],
            concat_axis=1,
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_concat_sigmoid" not in text

    def test_no_sigmoid_declines_to_fuse(self):
        """A composite root of relu (not sigmoid) must not match -- the
        pattern requires the bare sigmoid activation specifically."""
        mod = _build_concat_sigmoid_model(
            [(4, 2, 2), (4, 1, 1)],
            d_scales=[0.1, 0.2],
            d_zps=[0, 0],
            use_relu=True,
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_concat_sigmoid" not in text

    def test_non_constant_scale_declines_to_fuse(self):
        """Regression guard: non-compile-time scale must decline, not mis-fuse."""
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", TensorStructInfo((1, 4, 2, 2), "int8"))
        x2 = relax.Var("x2", TensorStructInfo((1, 4, 1, 1), "int8"))
        s_in = relax.Var("s_in", TensorStructInfo((), "float32"))
        with bb.function("main", [x1, x2, s_in], attrs={"num_input": 3}):
            with bb.dataflow():
                r1 = bb.emit(relax.op.reshape(x1, (1, 4, 4)))
                s1 = bb.emit(relax.op.multiply(s_in, relax.const(1.0, "float32")))
                dq1 = bb.emit(relax.op.dequantize(r1, s1, relax.const(0, "int8")))
                r2 = bb.emit(relax.op.reshape(x2, (1, 4, 1)))
                dq2 = bb.emit(
                    relax.op.dequantize(r2, relax.const(0.2, "float32"), relax.const(0, "int8"))
                )
                cat = bb.emit(relax.op.concat([dq1, dq2], axis=-1))
                sig = bb.emit(relax.op.sigmoid(cat))
                out = bb.emit_output(sig)
            bb.emit_func_output(out)
        mod = bb.finalize()
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_concat_sigmoid" not in text
