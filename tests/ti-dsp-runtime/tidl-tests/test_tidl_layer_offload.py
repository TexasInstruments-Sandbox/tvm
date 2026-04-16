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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
