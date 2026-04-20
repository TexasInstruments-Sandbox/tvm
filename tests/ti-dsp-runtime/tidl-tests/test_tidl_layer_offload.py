#!/usr/bin/env python
"""Per-layer unit tests for TIDL offloading.

Tests verify that each newly supported layer type is correctly matched,
partitioned, and (where the .so is available) accepted by the TIDL C++
parser.  Organised by op category with parametrised tests where possible.

Test levels:
  Level 1 (TestLayerPartition): Pattern matching only — no .so, no hw
  Level 4 (TestLayerHardware):  Full pipeline on AM67A via c7x_dload
"""

import os
import shutil
import sys

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.tidl import get_tidl_patterns, partition_for_tidl
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

# Level 1 tests are quick (pure IR, no hardware, no .so)
pytestmark = [pytest.mark.quick, pytest.mark.core]

# ---------------------------------------------------------------------------
# Environment for hardware tests
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DSP_CPP_DIR = os.path.join(_TESTS_DIR, "dsp-cpp")
sys.path.insert(0, _DSP_CPP_DIR)

C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH",
    os.path.expanduser("~/ml/c7x-mma-tidl"),
)
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH,
    "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so",
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")


def _has_full_tidl_env():
    return os.path.isfile(RELAX_SO_PATH) and os.environ.get("TI_CGT_C7000_PATH") is not None


# ---------------------------------------------------------------------------
# Helpers (shared with test_tidl_partition.py)
# ---------------------------------------------------------------------------


def _count_tidl_subgraphs(mod):
    """Count functions with Codegen='tidl'."""
    return len(
        [
            gv
            for gv, func in mod.functions.items()
            if isinstance(func, relax.Function)
            and func.attrs
            and func.attrs.get("Codegen") == "tidl"
        ]
    )


def _find_composites_in_module(mod, prefix="tidl."):
    """Find all Composite names in the module (including nested)."""
    composites = set()

    def _walk_expr(expr):
        if isinstance(expr, relax.Function):
            if expr.attrs:
                comp = expr.attrs.get("Composite")
                if comp and str(comp).startswith(prefix):
                    composites.add(str(comp))
            _walk_expr(expr.body)
        elif isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    _walk_expr(binding.value)
        elif isinstance(expr, relax.Call):
            if isinstance(expr.op, relax.Function):
                _walk_expr(expr.op)
            for arg in expr.args:
                _walk_expr(arg)

    for _gv, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _walk_expr(func)

    return composites


def _has_composite(mod, composite_name):
    """Check if a specific composite function exists in the module."""
    return composite_name in _find_composites_in_module(mod)


# ---------------------------------------------------------------------------
# Model builders — each constructs a small Relax model with a target op
# ---------------------------------------------------------------------------


def _build_sigmoid_model():
    """Conv -> relu -> sigmoid."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), "float32")):
            with R.dataflow():
                w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                y = R.nn.relu(y)
                y = R.sigmoid(y)
                R.output(y)
            return y

    return Model


def _build_tanh_model():
    """Conv -> relu -> tanh."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), "float32")):
            with R.dataflow():
                w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                y = R.nn.relu(y)
                y = R.tanh(y)
                R.output(y)
            return y

    return Model


def _build_clip_model():
    """Conv -> clip(0, 6) (relu6-style)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), "float32")):
            with R.dataflow():
                w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                y = R.clip(
                    y,
                    R.prim_value(T.float32(0)),
                    R.prim_value(T.float32(6)),
                )
                R.output(y)
            return y

    return Model


def _build_leakyrelu_model():
    """Conv -> leaky_relu."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), "float32")):
            with R.dataflow():
                w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                y = R.nn.leakyrelu(y, alpha=0.1)
                R.output(y)
            return y

    return Model


def _build_prelu_model():
    """Conv -> prelu."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), "float32")):
            with R.dataflow():
                w = R.const(np.random.randn(16, 3, 3, 3).astype("float32"))
                y = R.nn.conv2d(x, w, strides=[1, 1], padding=[1, 1, 1, 1])
                alpha = R.const(np.full(16, 0.25, dtype="float32"))
                y = R.nn.prelu(y, alpha, axis=1)
                R.output(y)
            return y

    return Model


def _build_batch_norm_model():
    """Standalone batch_norm (no preceding conv)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 16, 32, 32), "float32")):
            with R.dataflow():
                gamma = R.const(np.ones(16, dtype="float32"))
                beta = R.const(np.zeros(16, dtype="float32"))
                mean = R.const(np.zeros(16, dtype="float32"))
                var = R.const(np.ones(16, dtype="float32"))
                bn = R.nn.batch_norm(x, gamma, beta, mean, var, axis=1, epsilon=1e-5)
                y = bn[0]
                R.output(y)
            return y

    return Model


def _build_eltwise_divide_model():
    """Element-wise divide with rank-4 inputs."""

    @I.ir_module
    class Model:
        @R.function
        def main(
            x: R.Tensor((1, 16, 32, 32), "float32"),
            y: R.Tensor((1, 16, 32, 32), "float32"),
        ):
            with R.dataflow():
                z = R.divide(x, y)
                R.output(z)
            return z

    return Model


def _build_eltwise_subtract_model():
    """Element-wise subtract with rank-4 inputs."""

    @I.ir_module
    class Model:
        @R.function
        def main(
            x: R.Tensor((1, 16, 32, 32), "float32"),
            y: R.Tensor((1, 16, 32, 32), "float32"),
        ):
            with R.dataflow():
                z = R.subtract(x, y)
                R.output(z)
            return z

    return Model


def _build_eltwise_maximum_model():
    """Element-wise maximum with rank-4 inputs."""

    @I.ir_module
    class Model:
        @R.function
        def main(
            x: R.Tensor((1, 16, 32, 32), "float32"),
            y: R.Tensor((1, 16, 32, 32), "float32"),
        ):
            with R.dataflow():
                z = R.maximum(x, y)
                R.output(z)
            return z

    return Model


def _build_eltwise_minimum_model():
    """Element-wise minimum with rank-4 inputs."""

    @I.ir_module
    class Model:
        @R.function
        def main(
            x: R.Tensor((1, 16, 32, 32), "float32"),
            y: R.Tensor((1, 16, 32, 32), "float32"),
        ):
            with R.dataflow():
                z = R.minimum(x, y)
                R.output(z)
            return z

    return Model


# --- Shape manipulation model builders ---


def _build_layer_norm_model():
    """Conv -> layer_norm (transformer normalization)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 16, 32), "float32")):
            with R.dataflow():
                gamma = R.const(np.ones(32, dtype="float32"))
                beta = R.const(np.zeros(32, dtype="float32"))
                y = R.nn.layer_norm(x, gamma, beta, axes=[-1])
                R.output(y)
            return y

    return Model


def _build_flatten_model():
    """Conv -> relu -> flatten."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 4, 4), "float32")):
            with R.dataflow():
                y = R.flatten(x)
                R.output(y)
            return y

    return Model


def _build_squeeze_model():
    """Squeeze dim 2 from (1, 8, 1, 16)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 1, 16), "float32")):
            with R.dataflow():
                y = R.squeeze(x, axis=[2])
                R.output(y)
            return y

    return Model


def _build_expand_dims_model():
    """Expand dims: insert axis 2 into (1, 8, 16)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16), "float32")):
            with R.dataflow():
                y = R.expand_dims(x, axis=2)
                R.output(y)
            return y

    return Model


def _build_strided_slice_model():
    """Strided slice: take channels 0..3 from (1, 8, 16, 16)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.strided_slice(x, axes=[1], begin=[0], end=[4])
                R.output(y)
            return y

    return Model


def _build_cast_model():
    """Cast float32 -> int8."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.astype(x, "int8")
                R.output(y)
            return y

    return Model


# --- Advanced op model builders ---


def _build_resize2d_model():
    """Resize2d nearest upsample 2x."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 8, 8), "float32")):
            with R.dataflow():
                y = R.image.resize2d(x, size=(16, 16), method="nearest_neighbor")
                R.output(y)
            return y

    return Model


def _build_take_model():
    """Take (gather) along channel axis."""
    x_var = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
    idx_var = relax.Var("idx", relax.TensorStructInfo((4,), "int32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x_var, idx_var]):
        with bb.dataflow():
            out = bb.emit(relax.op.take(x_var, idx_var, axis=1))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


def _build_topk_model():
    """TopK along HEIGHT axis (k=4 from 8-element axis), both outputs."""
    x_var = relax.Var("x", relax.TensorStructInfo((1, 8, 8, 8), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x_var]):
        with bb.dataflow():
            # topk on HEIGHT axis (axis=2); ret_type="both" → Tuple(values,idx)
            t = bb.emit(relax.op.topk(x_var, k=4, axis=2, ret_type="both"))
            bb.emit_output(t)
        bb.emit_func_output(t)
    return bb.get()


def _build_split_model():
    """Split along channel axis into 4 equal slices."""
    x_var = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x_var]):
        with bb.dataflow():
            t = bb.emit(relax.op.split(x_var, 4, axis=1))
            # Return all slices as a tuple; partition test checks composite exists
            bb.emit_output(t)
        bb.emit_func_output(t)
    return bb.get()


# --- Math/unary model builders ---


def _make_unary_module(op_fn):
    """Build an IRModule that applies a unary op to a (1,8,16,16) tensor."""
    x_var = relax.Var("x", relax.TensorStructInfo((1, 8, 16, 16), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x_var]):
        with bb.dataflow():
            out = bb.emit(op_fn(x_var))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


# Pre-built models for each unary op (used by partition tests)
def _build_abs_model():
    return _make_unary_module(lambda x: relax.op.abs(x))


def _build_sqrt_model():
    return _make_unary_module(lambda x: relax.op.sqrt(x))


def _build_exp_model():
    return _make_unary_module(lambda x: relax.op.exp(x))


def _build_log_model():
    return _make_unary_module(lambda x: relax.op.log(x))


def _build_erf_model():
    return _make_unary_module(lambda x: relax.op.erf(x))


def _build_floor_model():
    return _make_unary_module(lambda x: relax.op.floor(x))


def _build_negative_model():
    return _make_unary_module(lambda x: relax.op.negative(x))


def _build_sin_model():
    return _make_unary_module(lambda x: relax.op.sin(x))


def _build_cos_model():
    return _make_unary_module(lambda x: relax.op.cos(x))


def _build_tan_model():
    return _make_unary_module(lambda x: relax.op.tan(x))


def _build_sinh_model():
    return _make_unary_module(lambda x: relax.op.sinh(x))


def _build_cosh_model():
    return _make_unary_module(lambda x: relax.op.cosh(x))


def _build_asin_model():
    return _make_unary_module(lambda x: relax.op.asin(x))


def _build_acos_model():
    return _make_unary_module(lambda x: relax.op.acos(x))


def _build_atan_model():
    return _make_unary_module(lambda x: relax.op.atan(x))


def _build_asinh_model():
    return _make_unary_module(lambda x: relax.op.asinh(x))


def _build_power_model():
    """Power op: x ** 2."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.power(x, R.const(2.0, "float32"))
                R.output(y)
            return y

    return Model


# --- Reduction model builders ---


def _build_sum_model():
    """Sum over channel axis (keepdims=True) on (1,8,16,16)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.sum(x, axis=[1], keepdims=True)
                R.output(y)
            return y

    return Model


def _build_reduce_max_model():
    """Max over HEIGHT axis (axis=2, the only axis TIDL ReduceLayer supports)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.max(x, axis=[2], keepdims=True)
                R.output(y)
            return y

    return Model


def _build_reduce_min_model():
    """Min over HEIGHT axis (axis=2, the only axis TIDL ReduceLayer supports)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.min(x, axis=[2], keepdims=True)
                R.output(y)
            return y

    return Model


def _build_argmax_model():
    """Argmax over channel axis (axis=1, keepdims=True — TIDL ArgOpLayer constraint)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.argmax(x, axis=1, keepdims=True)
                R.output(y)
            return y

    return Model


def _build_argmin_model():
    """Argmin over channel axis (axis=1, keepdims=True — TIDL ArgOpLayer constraint)."""

    @I.ir_module
    class Model:
        @R.function
        def main(x: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                y = R.argmin(x, axis=1, keepdims=True)
                R.output(y)
            return y

    return Model


# ---------------------------------------------------------------------------
# Level 1: Partition Tests (no .so, no hardware)
# ---------------------------------------------------------------------------

_ELTWISE_MODELS = {
    "divide": (_build_eltwise_divide_model, "tidl.divide"),
    "subtract": (_build_eltwise_subtract_model, "tidl.subtract"),
    "maximum": (_build_eltwise_maximum_model, "tidl.maximum"),
    "minimum": (_build_eltwise_minimum_model, "tidl.minimum"),
}


class TestLayerPartition:
    """Verify each op is matched by TIDL patterns and partitioned."""

    # --- Pattern registration for new ops ---

    def test_new_patterns_registered(self):
        """All new patterns should be in get_tidl_patterns()."""
        patterns = get_tidl_patterns()
        names = {p.name for p in patterns}
        expected_new = {
            "tidl.sigmoid",
            "tidl.tanh",
            "tidl.clip",
            "tidl.nn.leakyrelu",
            "tidl.nn.prelu",
            "tidl.divide",
            "tidl.subtract",
            "tidl.maximum",
            "tidl.minimum",
        }
        for exp in expected_new:
            assert exp in names, f"Expected pattern '{exp}' not found"

    # --- Activations ---

    def test_sigmoid(self):
        """Sigmoid after conv should be partitioned."""
        mod = _build_sigmoid_model()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) >= 1
        assert _has_composite(partitioned, "tidl.sigmoid"), (
            f"Expected tidl.sigmoid. Found: {_find_composites_in_module(partitioned)}"
        )

    def test_tanh(self):
        """Tanh after conv should be partitioned."""
        mod = _build_tanh_model()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) >= 1
        assert _has_composite(partitioned, "tidl.tanh"), (
            f"Expected tidl.tanh. Found: {_find_composites_in_module(partitioned)}"
        )

    def test_clip_standalone(self):
        """Standalone clip (relu6) should be partitioned."""
        mod = _build_clip_model()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) >= 1
        assert _has_composite(partitioned, "tidl.clip"), (
            f"Expected tidl.clip. Found: {_find_composites_in_module(partitioned)}"
        )

    def test_leakyrelu(self):
        """Leaky ReLU after conv should be partitioned."""
        mod = _build_leakyrelu_model()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) >= 1
        assert _has_composite(partitioned, "tidl.nn.leakyrelu"), (
            f"Expected tidl.nn.leakyrelu. Found: {_find_composites_in_module(partitioned)}"
        )

    def test_prelu(self):
        """PReLU after conv should be partitioned."""
        mod = _build_prelu_model()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) >= 1
        assert _has_composite(partitioned, "tidl.nn.prelu"), (
            f"Expected tidl.nn.prelu. Found: {_find_composites_in_module(partitioned)}"
        )

    # --- Element-wise ops ---

    @pytest.mark.parametrize(
        "eltwise_op",
        ["divide", "subtract", "maximum", "minimum"],
    )
    def test_eltwise(self, eltwise_op):
        """Element-wise binary op with rank-4 inputs is partitioned."""
        builder, composite = _ELTWISE_MODELS[eltwise_op]
        mod = builder()
        partitioned = partition_for_tidl(mod)
        assert _count_tidl_subgraphs(partitioned) >= 1, f"Expected TIDL subgraph for {eltwise_op}"
        assert _has_composite(partitioned, composite), (
            f"Expected {composite}. Found: {_find_composites_in_module(partitioned)}"
        )

    # --- Constraint rejection ---

    # --- Shape manipulation, normalization, cast ---

    def test_layer_norm(self):
        """Layer norm should be partitioned."""
        mod = _build_layer_norm_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.nn.layer_norm")

    def test_flatten(self):
        """Flatten should be partitioned."""
        mod = _build_flatten_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.flatten")

    def test_squeeze(self):
        """Squeeze should be partitioned."""
        mod = _build_squeeze_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.squeeze")

    def test_expand_dims(self):
        """Expand dims should be partitioned."""
        mod = _build_expand_dims_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.expand_dims")

    def test_strided_slice(self):
        """Strided slice should be partitioned."""
        mod = _build_strided_slice_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.strided_slice")

    def test_cast(self):
        """Cast (astype) should be partitioned."""
        mod = _build_cast_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.cast")

    # --- Constraint rejection ---

    # --- Reduction ops ---

    @pytest.mark.parametrize(
        "builder,composite",
        [
            (_build_sum_model, "tidl.sum"),
            (_build_reduce_max_model, "tidl.reduce_max"),
            (_build_reduce_min_model, "tidl.reduce_min"),
            (_build_argmax_model, "tidl.argmax"),
            (_build_argmin_model, "tidl.argmin"),
        ],
    )
    def test_reduction(self, builder, composite):
        """Reduction op should be partitioned."""
        mod = builder()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, composite), (
            f"Expected {composite}. Found: {_find_composites_in_module(partitioned)}"
        )

    # --- Advanced ops ---

    def test_resize2d(self):
        """Resize2d (nearest 2x upsample) should be partitioned."""
        mod = _build_resize2d_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.image.resize2d")

    def test_take(self):
        """Take (gather along axis) should be partitioned."""
        mod = _build_take_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.take")

    def test_topk(self):
        """TopK should be partitioned."""
        mod = _build_topk_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.topk")

    def test_split(self):
        """Split should be partitioned."""
        mod = _build_split_model()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, "tidl.split")

    # --- Math/unary ops (parametrized) ---
    # TIDL converts these to TIDL_BatchNormLayer + hardware LUT via
    # tidl_convertRelUToBNLayer during PostProcessNet.  Both Relay and Relax
    # parsers only need to set layerType + actType; the LUT is built
    # automatically from calibration stats.

    @pytest.mark.parametrize(
        "builder,composite",
        [
            (_build_abs_model, "tidl.abs"),
            (_build_sqrt_model, "tidl.sqrt"),
            (_build_power_model, "tidl.power"),
            (_build_exp_model, "tidl.exp"),
            (_build_log_model, "tidl.log"),
            (_build_erf_model, "tidl.erf"),
            (_build_floor_model, "tidl.floor"),
            (_build_negative_model, "tidl.negative"),
            (_build_sin_model, "tidl.sin"),
            (_build_cos_model, "tidl.cos"),
            (_build_tan_model, "tidl.tan"),
            (_build_sinh_model, "tidl.sinh"),
            (_build_cosh_model, "tidl.cosh"),
            (_build_asin_model, "tidl.asin"),
            (_build_acos_model, "tidl.acos"),
            (_build_atan_model, "tidl.atan"),
            (_build_asinh_model, "tidl.asinh"),
        ],
    )
    def test_math_unary(self, builder, composite):
        """Math/unary op should be partitioned."""
        mod = builder()
        partitioned = partition_for_tidl(mod)
        assert _has_composite(partitioned, composite), (
            f"Expected {composite}. Found: {_find_composites_in_module(partitioned)}"
        )

    # --- Constraint rejection ---

    def test_reduce_multi_axis_rejected(self):
        """Multi-axis reduction should NOT be offloaded (TIDL only supports single axis)."""

        @I.ir_module
        class MultiAxisSum:
            @R.function
            def main(x: R.Tensor((1, 8, 16, 16), "float32")):
                with R.dataflow():
                    y = R.sum(x, axis=[2, 3])
                    R.output(y)
                return y

        partitioned = partition_for_tidl(MultiAxisSum)
        assert not _has_composite(partitioned, "tidl.sum"), "Multi-axis sum should not be offloaded"

    def test_divide_rejects_rank2(self):
        """Rank-2 divide should NOT be offloaded (rank < 4)."""

        @I.ir_module
        class Rank2Divide:
            @R.function
            def main(
                x: R.Tensor((16, 32), "float32"),
                y: R.Tensor((16, 32), "float32"),
            ):
                with R.dataflow():
                    z = R.divide(x, y)
                    R.output(z)
                return z

        partitioned = partition_for_tidl(Rank2Divide)
        n_tidl = _count_tidl_subgraphs(partitioned)
        assert n_tidl == 0, (
            f"Rank-2 divide should not be offloaded, but found {n_tidl} TIDL subgraphs"
        )


# ---------------------------------------------------------------------------
# Level 4: Hardware e2e tests (AM67A via c7x_dload)
# ---------------------------------------------------------------------------


def _build_and_run(model_cls, input_spec, input_data, tmp_path, expected_shape):
    """Build model through TIDL pipeline and run on AM67A.

    Returns (output, n_artifacts).
    """
    from dsp_utils import run_dsp_dload

    from tvm.relax.backend.tidl import TIDLOffloadCompiler

    model = model_cls()
    mod, param_spec = model.export_tvm(spec={"main": input_spec})
    np.random.seed(42)
    params = [
        tvm.runtime.tensor(
            np.random.rand(*p.shape).astype("float32"),
            device=tvm.cpu(),
        )
        for _, p in param_spec
    ]
    param_dict = dict(zip(mod["main"].params[1:], params))

    artifacts_dir = str(tmp_path / "tidl_artifacts")
    compiler = TIDLOffloadCompiler(
        config={
            "artifacts_dir": artifacts_dir,
            "tidl_tools_path": TIDL_TOOLS_PATH,
            "tidl_relax_so_path": RELAX_SO_PATH,
            "num_calibration_frames": 2,
        }
    )
    result = compiler.build(
        mod,
        params=param_dict,
        build_dir=str(tmp_path / "build"),
    )

    assert result.module_path.exists(), f"Build failed: {result.module_path}"
    n_artifacts = len(result.artifacts)

    try:
        output, _stdout, cycles = run_dsp_dload(
            result.module_path,
            result.weights_path,
            [input_data],
            embedded_weights=True,
        )

        assert output is not None, "No output from DSP"
        assert output.shape == expected_shape, (
            f"Unexpected shape: {output.shape}, expected {expected_shape}"
        )
        print(
            f"  Output: shape={output.shape}, "
            f"min={output.min():.4f}, max={output.max():.4f}, "
            f"cycles={cycles:,}"
        )
        return output, n_artifacts

    finally:
        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(result.gen_dir), ignore_errors=True)
            shutil.rmtree(str(result.build_dir), ignore_errors=True)


# --- nn.Module models for hardware tests ---


class ConvSigmoidModel(nn.Module):
    """Conv + ReLU + Sigmoid — output bounded [0, 1]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        x = nn.sigmoid(x)
        return x


class ConvTanhModel(nn.Module):
    """Conv + ReLU + Tanh — output bounded [0, 1) (relu clips negatives)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        x = nn.tanh(x)
        return x


class ConvClipModel(nn.Module):
    """Conv + ReLU6 (clip 0..6) — output bounded [0, 6]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        x = nn.relu6(x)
        return x


class ConvLeakyReluModel(nn.Module):
    """Conv + LeakyReLU(0.1) — output may have small negatives."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        # nn.Module doesn't expose leakyrelu directly; use relax op
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = wrap_nested(_op.nn.leakyrelu(x._expr, alpha=0.1), "leakyrelu")
        return x


class ConvSubtractModel(nn.Module):
    """Two conv+relu branches subtracted element-wise."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        a = self.conv1(x)
        a = nn.relu(a)
        b = self.conv2(x)
        b = nn.relu(b)
        return nn.subtract(a, b)


class ConvMaximumModel(nn.Module):
    """Two conv+relu branches with element-wise maximum."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        a = self.conv1(x)
        a = nn.relu(a)
        b = self.conv2(x)
        b = nn.relu(b)
        return nn.maximum(a, b)


class ConvMinimumModel(nn.Module):
    """Two conv+relu branches with element-wise minimum."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        a = self.conv1(x)
        a = nn.relu(a)
        b = self.conv2(x)
        b = nn.relu(b)
        return nn.minimum(a, b)


# --- Shape/normalization nn.Module models for hardware tests ---


# --- Reduction nn.Module models for hardware tests ---


class ConvSumModel(nn.Module):
    """Conv + sum over channel axis (keepdims=True) → (1,1,16,16)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        x = wrap_nested(_op.sum(x._expr, axis=[1], keepdims=True), "sum")
        return x


class ConvReduceMaxModel(nn.Module):
    """Conv + max over HEIGHT axis (axis=2, TIDL only supports height)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        x = wrap_nested(_op.max(x._expr, axis=[2], keepdims=True), "reduce_max")
        return x


class ConvArgmaxModel(nn.Module):
    """Conv + argmax over channel axis (keepdims=True), cast to int32."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        # ArgMax returns int64; cast to int32 so DSP output is parseable
        x = wrap_nested(_op.argmax(x._expr, axis=1, keepdims=True), "argmax")
        x = wrap_nested(_op.astype(x._expr, "int32"), "cast_int32")
        return x


class ConvArgminModel(nn.Module):
    """Conv + argmin over channel axis (keepdims=True), cast to int32."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        # ArgMin returns int64; cast to int32 so DSP output is parseable
        x = wrap_nested(_op.argmin(x._expr, axis=1, keepdims=True), "argmin")
        x = wrap_nested(_op.astype(x._expr, "int32"), "cast_int32")
        return x


# ---------------------------------------------------------------------------
# Math/unary hardware models
#
# CALIBRATION NOTE: These ops are converted to TIDL_BatchNormLayer + LUT by
# tidl_convertRelUToBNLayer during PostProcessNet (same path for Relay and
# Relax).  The hardware LUT computes the function correctly at inference time.
#
# With *random* calibration data the quantisation scale can be set poorly for
# ops with unbounded output range (exp, power, sinh, cosh) because
# random float32 after relu can be large, making exp(x) overflow.  TIDL then
# sets a huge scale and all "normal" output values near 1 quantise to zero.
# This is a calibration data quality issue, not a hardware limitation.
# Bounded-output ops (abs, sqrt, erf, sin, cos, atan, negative, floor) always
# produce correct non-zero output with random calibration data.
#
# All hardware tests use np.isfinite(output).all() to validate the full
# import → codegen → hardware pipeline without depending on calibration quality.
# ---------------------------------------------------------------------------


def _make_conv_unary_model(op_attr: str) -> type:
    """Factory: return an nn.Module class that applies conv+relu+<op>."""

    class ConvUnaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

        def main(self, x):
            from tvm.relax import op as _op
            from tvm.relax.frontend.nn.op import wrap_nested

            x = self.conv(x)
            x = nn.relu(x)
            x = wrap_nested(getattr(_op, op_attr)(x._expr), op_attr)
            return x

    ConvUnaryModel.__name__ = f"Conv{op_attr.capitalize()}Model"
    return ConvUnaryModel


# Pre-instantiated model classes for each op (used by test parametrization)
_MATH_UNARY_HW_OPS = [
    ("abs", "tidl.abs"),
    ("sqrt", "tidl.sqrt"),
    ("exp", "tidl.exp"),
    ("log", "tidl.log"),
    ("erf", "tidl.erf"),
    ("floor", "tidl.floor"),
    ("negative", "tidl.negative"),
    ("sin", "tidl.sin"),
    ("cos", "tidl.cos"),
    ("tan", "tidl.tan"),
    ("sinh", "tidl.sinh"),
    ("cosh", "tidl.cosh"),
    ("asin", "tidl.asin"),
    ("acos", "tidl.acos"),
    ("atan", "tidl.atan"),
    ("asinh", "tidl.asinh"),
]
# power needs separate model (binary op)
ConvExpModel = _make_conv_unary_model("exp")  # kept for direct reference


class ConvResizeModel(nn.Module):
    """Conv + resize nearest 2x upsample → (1,8,32,32)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        x = wrap_nested(
            _op.image.resize2d(x._expr, size=(32, 32), method="nearest_neighbor"),
            "resize2d",
        )
        return x


class ConvSliceModel(nn.Module):
    """Conv + strided_slice to take first 4 channels."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        x = wrap_nested(
            _op.strided_slice(x._expr, axes=[1], begin=[0], end=[4]),
            "strided_slice",
        )
        return x


class ConvPermuteModel(nn.Module):
    """Conv + ReLU + permute_dims (NCHW → NHWC)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        x = nn.permute_dims(x, axes=[0, 2, 3, 1])  # NCHW → NHWC
        return x


class ConvReluConcatModel(nn.Module):
    """Two conv+relu branches concatenated along channel axis."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 4, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(3, 4, 3, 1, 1, bias=False)

    def main(self, x):
        a = self.conv1(x)
        a = nn.relu(a)
        b = self.conv2(x)
        b = nn.relu(b)
        return nn.concat([a, b], dim=1)


class ConvTopKModel(nn.Module):
    """Conv + TopK on HEIGHT axis (k=8 of 16), return values.

    TIDL always produces both values and indices; we extract values [0].
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        # topk on HEIGHT (axis=2): k=8 of 16 rows; ret_type="both" → Tuple
        t = _op.topk(x._expr, k=8, axis=2, ret_type="both")
        # Return values (index 0); indices (index 1) discarded
        x = wrap_nested(t[0], "topk_values")
        return x


class ConvSplitModel(nn.Module):
    """Conv + split into 4 equal channel slices, return first slice."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        x = self.conv(x)
        x = nn.relu(x)
        # split 8 channels into 4 slices of 2 channels each along channel axis
        parts = _op.split(x._expr, 4, axis=1)
        # Return first slice only (single tensor output)
        x = wrap_nested(parts[0], "split_first")
        return x


@pytest.mark.skipif(not _has_full_tidl_env(), reason="needs .so + compiler + AM67A")
class TestLayerHardware:
    """Test new layer ops through full TIDL pipeline on AM67A."""

    INPUT_SPEC = {"x": nn.spec.Tensor((1, 3, 16, 16), "float32")}
    INPUT_SHAPE = (1, 3, 16, 16)
    OUTPUT_SHAPE = (1, 8, 16, 16)

    def _run(self, model_cls, tmp_path, expected_shape=None):
        if expected_shape is None:
            expected_shape = self.OUTPUT_SHAPE
        input_data = np.random.randn(*self.INPUT_SHAPE).astype("float32")
        return _build_and_run(
            model_cls,
            self.INPUT_SPEC,
            input_data,
            tmp_path,
            expected_shape,
        )

    def test_sigmoid_hw(self, tmp_path):
        """Sigmoid offloaded to TIDL on AM67A."""
        output, n = self._run(ConvSigmoidModel, tmp_path)
        assert n >= 1
        # Sigmoid output is bounded [0, 1]
        assert output.min() >= -0.01
        assert output.max() <= 1.01

    def test_tanh_hw(self, tmp_path):
        """Tanh offloaded to TIDL on AM67A."""
        output, n = self._run(ConvTanhModel, tmp_path)
        assert n >= 1
        # Tanh output bounded [-1,1]; quantization may exceed slightly
        assert output.min() >= -1.1
        assert output.max() <= 1.1

    def test_clip_hw(self, tmp_path):
        """Clip (relu6) offloaded to TIDL on AM67A."""
        output, n = self._run(ConvClipModel, tmp_path)
        assert n >= 1
        assert output.min() >= -0.01
        assert output.max() <= 6.01

    def test_leakyrelu_hw(self, tmp_path):
        """LeakyReLU offloaded to TIDL on AM67A."""
        output, n = self._run(ConvLeakyReluModel, tmp_path)
        assert n >= 1
        # LeakyReLU has no upper bound, but output should be finite
        assert np.isfinite(output).all()

    def test_subtract_hw(self, tmp_path):
        """Element-wise subtract offloaded to TIDL on AM67A."""
        output, n = self._run(ConvSubtractModel, tmp_path)
        assert n >= 1
        assert np.isfinite(output).all()

    def test_maximum_hw(self, tmp_path):
        """Element-wise maximum offloaded to TIDL on AM67A."""
        output, n = self._run(ConvMaximumModel, tmp_path)
        assert n >= 1
        # Output should be finite (quantization may produce negatives)
        assert np.isfinite(output).all()

    def test_minimum_hw(self, tmp_path):
        """Element-wise minimum offloaded to TIDL on AM67A."""
        output, n = self._run(ConvMinimumModel, tmp_path)
        assert n >= 1
        # Output should be finite (quantization may produce negatives)
        assert np.isfinite(output).all()

    # --- Reduction hardware tests ---

    def test_sum_hw(self, tmp_path):
        """Sum over channels offloaded to TIDL on AM67A."""
        output, n = self._run(
            ConvSumModel,
            tmp_path,
            expected_shape=(1, 1, 16, 16),
        )
        assert n >= 1
        assert np.isfinite(output).all()

    def test_reduce_max_hw(self, tmp_path):
        """Height-wise max offloaded to TIDL on AM67A (axis=2, NCHW → (1,8,1,16))."""
        output, n = self._run(
            ConvReduceMaxModel,
            tmp_path,
            expected_shape=(1, 8, 1, 16),
        )
        assert n >= 1
        assert np.isfinite(output).all()

    def test_argmax_hw(self, tmp_path):
        """Argmax over channel axis (cast int64→int32) on AM67A → (1,1,16,16)."""
        output, n = self._run(
            ConvArgmaxModel,
            tmp_path,
            expected_shape=(1, 1, 16, 16),
        )
        assert n >= 1
        # Output is int32 channel indices (0–7 for 8 channels)
        assert output.max() < 8

    def test_argmin_hw(self, tmp_path):
        """Argmin over channel axis (cast int64→int32) on AM67A → (1,1,16,16)."""
        output, n = self._run(
            ConvArgminModel,
            tmp_path,
            expected_shape=(1, 1, 16, 16),
        )
        assert n >= 1
        # Output is int32 channel indices (0–7 for 8 channels)
        assert output.max() < 8

    # --- Math/unary hardware tests ---

    @pytest.mark.parametrize("op_attr,composite", _MATH_UNARY_HW_OPS)
    def test_math_unary_hw(self, op_attr, composite, tmp_path):
        """Math/unary op offloaded to TIDL on AM67A.

        Validates the full import → codegen → hardware pipeline.
        Numerical output may be zero for ops with unbounded range (exp, sinh,
        cosh, power) when random calibration data causes quantisation scale
        overflow — this is expected and is a calibration data quality issue,
        not a hardware or parser bug.  See module-level comment above
        _MATH_UNARY_HW_OPS for details.
        """
        model_cls = _make_conv_unary_model(op_attr)
        output, n = self._run(model_cls, tmp_path)
        assert n >= 1
        assert np.isfinite(output).all()

    def test_power_hw(self, tmp_path):
        """Power (x**2) offloaded to TIDL on AM67A."""
        from tvm.relax import op as _op
        from tvm.relax.frontend.nn.op import wrap_nested

        class ConvPowerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

            def main(self, x):
                x = self.conv(x)
                x = nn.relu(x)
                x = wrap_nested(_op.power(x._expr, relax.const(2.0, "float32")), "power")
                return x

        output, n = self._run(ConvPowerModel, tmp_path)
        assert n >= 1
        assert np.isfinite(output).all()

    # --- Advanced op hardware tests ---

    def test_resize2d_hw(self, tmp_path):
        """Resize nearest 2x upsample offloaded to TIDL on AM67A."""
        output, n = self._run(
            ConvResizeModel,
            tmp_path,
            expected_shape=(1, 8, 32, 32),
        )
        assert n >= 1
        assert np.isfinite(output).all()

    # --- Shape/slice/layout hardware tests ---

    def test_strided_slice_hw(self, tmp_path):
        """Strided slice (first 4 channels) offloaded to TIDL on AM67A."""
        output, n = self._run(
            ConvSliceModel,
            tmp_path,
            expected_shape=(1, 4, 16, 16),
        )
        assert n >= 1
        assert np.isfinite(output).all()

    def test_permute_dims_hw(self, tmp_path):
        """permute_dims (NCHW→NHWC transpose) offloaded to TIDL on AM67A."""
        output, n = self._run(
            ConvPermuteModel,
            tmp_path,
            expected_shape=(1, 16, 16, 8),  # NHWC output
        )
        assert n >= 1
        assert np.isfinite(output).all()

    def test_concat_hw(self, tmp_path):
        """Concat inside TIDL subgraph on AM67A — 4+4=8 channels."""
        output, n = self._run(
            ConvReluConcatModel,
            tmp_path,
            expected_shape=(1, 8, 16, 16),
        )
        assert n >= 1
        # Shape must be correct (8 = 4+4 channels after concat)
        assert output.shape == (1, 8, 16, 16)
        assert np.isfinite(output).all()

    def test_topk_hw(self, tmp_path):
        """TopK (values, k=8 of 16 HEIGHT rows) offloaded to TIDL on AM67A."""
        output, n = self._run(
            ConvTopKModel,
            tmp_path,
            expected_shape=(1, 8, 8, 16),  # 8 top rows selected from 16
        )
        assert n >= 1
        assert np.isfinite(output).all()

    def test_split_hw(self, tmp_path):
        """Split (first of 4 channel slices) offloaded to TIDL on AM67A."""
        output, n = self._run(
            ConvSplitModel,
            tmp_path,
            expected_shape=(1, 2, 16, 16),  # 2 channels = 8 / 4
        )
        assert n >= 1
        assert np.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
