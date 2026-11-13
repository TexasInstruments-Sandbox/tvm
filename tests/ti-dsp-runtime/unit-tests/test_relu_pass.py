"""Unit tests for FuseQDQToC7xRelu.

Pure Relax IR-level tests, no hardware or DSP build needed -- mirrors
test_movement_pass.py's convention (build a small model, run the pass,
inspect mod.script()). No IR-level test file previously existed for this
pass (only the heavier DSP-compile-and-run test_relu_kernel.py).

Covers the pass's three patterns:
  1. dq(x) -> relax.nn.relu -> q, transparent
     (d_scale ~= o_scale, per tensor or per channel, AND d_zp == o_zp) -> c7x_int8_relu
  2. dq(x) -> relax.clip -> q, transparent
     (d_scale ~= o_scale AND d_zp == o_zp)                     -> c7x_int8_clamp
  3. dq(x) -> relax.clip -> q, non-transparent
     (d_scale != o_scale, both zp == 0)               -> c7x_int8_requantize_clamp

There is also a regression guard mirroring test_movement_pass.py's: when
the composite's data operand x is itself a literal tvm.ir.GenericConst tensor
(a real, already-supported match per each pattern's isinstance(x, relax.
Constant) branch, not merely reachable from constants), the
ConstReachability guard declines to fuse. Before the inline-on-decline fix
(same root cause as the yolov5n crash on Jenkins build 230, fixed for
FuseQDQToC7xMovement's analogous guards), the un-consumed Composite/
Primitive function -- which embeds that Constant tensor directly, since
FuseOpsByPattern here also uses bind_constants=False -- crashed
relax.transform.FuseTIR downstream with "Relax.Constant is not supported
in primitive functions." A scalar embedded constant (e.g. the scale/zp) is
NOT enough to reproduce this: LegalizeOps bakes scalars into TIR literals,
so only a tensor-shaped Constant survives to reach FuseTIR.
"""

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorType as TensorStructInfo


def _run_pass(mod):
    return tvm.relax.transform.FuseQDQToC7xRelu()(mod)


def _assert_compiles(new_mod):
    """The rest of the pipeline's own FuseTIR-facing sequence, run directly
    (see legalize_passes in python/tvm/relax/backend/cpu_generic/pipeline.py)
    -- must not raise."""
    new_mod = relax.transform.LegalizeOps()(new_mod)
    new_mod = relax.transform.FoldConstant()(new_mod)
    new_mod = relax.transform.FuseOps()(new_mod)
    relax.transform.FuseTIR()(new_mod)


# ---------------------------------------------------------------------------
# Pattern 1: relu
# ---------------------------------------------------------------------------


def _qparams(scale, zp):
    """Scale and zero-point constants and the axis; a list scale is per channel on
    axis 1."""
    if isinstance(scale, list):
        zp = zp if isinstance(zp, list) else [zp] * len(scale)
        return relax.const(scale, "float32"), relax.const(zp, "int8"), 1
    return relax.const(scale, "float32"), relax.const(zp, "int8"), -1


def _build_relu_model(x, d_scale, d_zp, o_scale, o_zp, params=None, attrs=None):
    bb = relax.BlockBuilder()
    with bb.function("main", params or [], attrs=attrs or {"num_input": 0}):
        with bb.dataflow():
            ds, dz, d_axis = _qparams(d_scale, d_zp)
            dq = bb.emit(relax.op.dequantize(x, ds, dz, d_axis))
            relu_out = bb.emit(relax.op.nn.relu(dq))
            os_, oz, o_axis = _qparams(o_scale, o_zp)
            q = bb.emit(relax.op.quantize(relu_out, os_, oz, o_axis, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestRelu:
    def test_fires_and_emits_call_extern(self):
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        mod = _build_relu_model(x, d_scale=0.03, d_zp=2, o_scale=0.03, o_zp=2, params=[x])
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_relu" in text
        assert "R.nn.relu" not in text

    def test_constant_tensor_input_declines_and_still_compiles(self):
        """Regression guard: x is itself a literal tvm.ir.GenericConst tensor.
        See this module's docstring for the full crash-mode explanation."""
        const_x = relax.const(np.arange(16, dtype="int8").reshape(1, 4, 2, 2))
        mod = _build_relu_model(const_x, d_scale=0.03, d_zp=2, o_scale=0.03, o_zp=2)

        new_mod = _run_pass(mod)
        assert "c7x_int8_relu" not in new_mod.script()
        _assert_compiles(new_mod)

    def test_per_channel_scales_fuse(self):
        """A ReLU between a conv and a depthwise conv quantized per channel: the
        kernel copies codes, which is right when each channel keeps its scale."""
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        s = [0.01, 0.02, 0.05, 0.3]
        mod = _build_relu_model(x, d_scale=s, d_zp=0, o_scale=list(s), o_zp=0, params=[x])
        text = _run_pass(mod).script()
        assert "c7x_int8_relu" in text
        assert "R.nn.relu" not in text

    @pytest.mark.parametrize(
        "d_scale,o_scale",
        [(0.03, 0.05), ([0.01, 0.02, 0.05, 0.3], [0.01, 0.02, 0.05, 0.2])],
        ids=["per-tensor", "per-channel"],
    )
    def test_differing_scales_decline_and_compile(self, d_scale, o_scale):
        """Copying codes would rescale the output. The ReLU's output can have an
        observer of its own when its producer isn't annotated."""
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        mod = _build_relu_model(x, d_scale, d_zp=0, o_scale=o_scale, o_zp=0, params=[x])
        new_mod = _run_pass(mod)
        assert "c7x_int8_relu" not in new_mod.script()
        _assert_compiles(new_mod)

    def test_differing_per_channel_zero_points_decline(self):
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        s = [0.01, 0.02, 0.05, 0.3]
        zps = [0, 1, 2, 3]
        mod = _build_relu_model(x, s, d_zp=zps, o_scale=list(s), o_zp=list(zps), params=[x])
        assert "c7x_int8_relu" not in _run_pass(mod).script()


# ---------------------------------------------------------------------------
# Pattern 2: transparent clamp
# ---------------------------------------------------------------------------


def _build_clamp_model(x, d_scale, d_zp, a_min, a_max, o_scale, o_zp, params=None, attrs=None):
    bb = relax.BlockBuilder()
    with bb.function("main", params or [], attrs=attrs or {"num_input": 0}):
        with bb.dataflow():
            ds, dz, d_axis = _qparams(d_scale, d_zp)
            dq = bb.emit(relax.op.dequantize(x, ds, dz, d_axis))
            clipped = bb.emit(relax.op.clip(dq, a_min, a_max))
            os_, oz, o_axis = _qparams(o_scale, o_zp)
            q = bb.emit(relax.op.quantize(clipped, os_, oz, o_axis, out_dtype="int8"))
            out = bb.emit_output(q)
        bb.emit_func_output(out)
    return bb.finalize()


class TestClamp:
    def test_fires_and_emits_call_extern(self):
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        mod = _build_clamp_model(
            x, d_scale=0.03, d_zp=0, a_min=0.0, a_max=6.0, o_scale=0.03, o_zp=0, params=[x]
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_clamp" in text
        assert "R.clip" not in text

    def test_constant_tensor_input_declines_and_still_compiles(self):
        """Regression guard: x is itself a literal tvm.ir.GenericConst tensor.
        See this module's docstring for the full crash-mode explanation."""
        const_x = relax.const(np.arange(16, dtype="int8").reshape(1, 4, 2, 2))
        mod = _build_clamp_model(
            const_x, d_scale=0.03, d_zp=0, a_min=0.0, a_max=6.0, o_scale=0.03, o_zp=0
        )

        new_mod = _run_pass(mod)
        assert "c7x_int8_clamp" not in new_mod.script()
        _assert_compiles(new_mod)

    @pytest.mark.parametrize("o_scale", [0.03, [0.03, 0.03, 0.05, 0.05]], ids=["same", "other"])
    def test_per_channel_scales_decline_and_compile(self, o_scale):
        """The clamp kernels take one scale; neither clamp pattern may take a
        per-channel q/dq (the transparent one nor the requantizing one)."""
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        s = [0.03, 0.03, 0.05, 0.05]
        o = list(s) if isinstance(o_scale, list) else o_scale
        mod = _build_clamp_model(x, s, d_zp=0, a_min=0.0, a_max=6.0, o_scale=o, o_zp=0, params=[x])
        new_mod = _run_pass(mod)
        assert "c7x_int8_" not in new_mod.script()
        _assert_compiles(new_mod)


# ---------------------------------------------------------------------------
# Pattern 3: non-transparent requantize-clamp
# ---------------------------------------------------------------------------


class TestReqClamp:
    def test_fires_and_emits_call_extern(self):
        x = relax.Var("x", TensorStructInfo((1, 4, 2, 2), "int8"))
        # d_scale != o_scale, both zero-points 0 -- the non-transparent case.
        mod = _build_clamp_model(
            x, d_scale=0.03, d_zp=0, a_min=0.0, a_max=6.0, o_scale=0.05, o_zp=0, params=[x]
        )
        new_mod = _run_pass(mod)
        text = new_mod.script()
        assert "c7x_int8_requantize_clamp" in text
        assert "R.clip" not in text

    def test_constant_tensor_input_declines_and_still_compiles(self):
        """Regression guard: x is itself a literal tvm.ir.GenericConst tensor.
        See this module's docstring for the full crash-mode explanation."""
        const_x = relax.const(np.arange(16, dtype="int8").reshape(1, 4, 2, 2))
        mod = _build_clamp_model(
            const_x, d_scale=0.03, d_zp=0, a_min=0.0, a_max=6.0, o_scale=0.05, o_zp=0
        )

        new_mod = _run_pass(mod)
        assert "c7x_int8_requantize_clamp" not in new_mod.script()
        _assert_compiles(new_mod)
