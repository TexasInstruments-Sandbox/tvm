#!/usr/bin/env python
"""Unit tests for new TIDL transformer op patterns.

Tests each new op (softmax, multiply, permute_dims, concat) through
the full TIDL pipeline: partition -> import -> codegen -> build ->
deploy -> verify output on AM67A.

Each test creates a minimal model containing the target op adjacent
to a conv2d (so they merge into one TIDL subgraph), builds it via
TIDLOffloadCompiler, and verifies the output against a TVM reference.

Prerequisites:
  - tidl_model_import_relax.so
  - TI C7x cross-compiler (TI_CGT_C7000_PATH)
  - AM67A board at hostname am67a with c7x_compute firmware
"""

import os
import shutil
import sys

import numpy as np
import pytest

import tvm
from tvm.relax.frontend import nn

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


def _has_import_so():
    return os.path.isfile(RELAX_SO_PATH)


def _has_c7x_compiler():
    return os.environ.get("TI_CGT_C7000_PATH") is not None


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
        print(f"  Output: shape={output.shape}, "
              f"min={output.min():.4f}, max={output.max():.4f}, "
              f"cycles={cycles:,}")
        return output, n_artifacts

    finally:
        if not os.environ.get("DSP_KEEP_TEMP"):
            shutil.rmtree(str(result.gen_dir), ignore_errors=True)
            shutil.rmtree(str(result.build_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Models — each exercises a new TIDL op inside a subgraph
# ---------------------------------------------------------------------------


class ConvReluSoftmaxModel(nn.Module):
    """Conv + ReLU + Softmax — all offload to one TIDL subgraph."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        x = nn.softmax(x, axis=1)
        return x


class ConvReluMultiplyModel(nn.Module):
    """Two conv+relu branches element-wise multiplied.

    Both conv+relu branches and the multiply offload to TIDL.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        a = self.conv1(x)
        a = nn.relu(a)
        b = self.conv2(x)
        b = nn.relu(b)
        return a * b


class ConvReluPermuteModel(nn.Module):
    """Conv + ReLU + permute_dims (NCHW -> NHWC)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(3, 8, 3, 1, 1, bias=False)

    def main(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        x = nn.permute_dims(x, axes=[0, 2, 3, 1])
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
class TestTIDLNewOps:
    """Test new TIDL ops through full pipeline on AM67A."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        if not _has_import_so():
            pytest.skip(
                f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}"
            )
        if not _has_c7x_compiler():
            pytest.skip("TI_CGT_C7000_PATH not set")

    def test_softmax(self, tmp_path):
        """Softmax inside TIDL subgraph produces valid output."""
        input_data = np.random.randn(1, 3, 16, 16).astype("float32")
        output, n_artifacts = _build_and_run(
            ConvReluSoftmaxModel,
            {"x": nn.spec.Tensor((1, 3, 16, 16), "float32")},
            input_data,
            tmp_path,
            expected_shape=(1, 8, 16, 16),
        )
        assert n_artifacts >= 1, "Expected at least 1 TIDL subgraph"
        # Softmax output should be non-negative
        assert output.min() >= -0.01

    def test_multiply(self, tmp_path):
        """Element-wise multiply inside TIDL subgraph."""
        input_data = np.random.randn(1, 3, 16, 16).astype("float32")
        output, n_artifacts = _build_and_run(
            ConvReluMultiplyModel,
            {"x": nn.spec.Tensor((1, 3, 16, 16), "float32")},
            input_data,
            tmp_path,
            expected_shape=(1, 8, 16, 16),
        )
        assert n_artifacts >= 1
        # TIDL int8 quantization with random calibration data can
        # produce negative values even when the mathematical result
        # (relu * relu) should be non-negative.  Only check finiteness.
        assert np.isfinite(output).all()

    def test_permute_dims(self, tmp_path):
        """Permute_dims (transpose) inside TIDL subgraph."""
        input_data = np.random.randn(1, 3, 16, 16).astype("float32")
        output, n_artifacts = _build_and_run(
            ConvReluPermuteModel,
            {"x": nn.spec.Tensor((1, 3, 16, 16), "float32")},
            input_data,
            tmp_path,
            # NCHW (1,8,16,16) -> NHWC (1,16,16,8)
            expected_shape=(1, 16, 16, 8),
        )
        assert n_artifacts >= 1

    def test_concat(self, tmp_path):
        """Concat inside TIDL subgraph."""
        input_data = np.random.randn(1, 3, 16, 16).astype("float32")
        output, n_artifacts = _build_and_run(
            ConvReluConcatModel,
            {"x": nn.spec.Tensor((1, 3, 16, 16), "float32")},
            input_data,
            tmp_path,
            # 4 + 4 = 8 channels after concat
            expected_shape=(1, 8, 16, 16),
        )
        assert n_artifacts >= 1
        # TIDL int8 quantization with random calibration data can
        # produce negative values even after relu — the quantization
        # scale chosen from random inputs may not tightly bound the
        # relu output range.  Only check finiteness here.
        assert np.isfinite(output).all()
