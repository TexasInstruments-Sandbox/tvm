"""Phase 1 diagnostic for Bug 4: does TIDL calibration overflow for
large BN-folded weights even WITHOUT a residual shortcut?

ResNet-18's layer4[1] second conv has BN-folded w_max=3.648 (4-10x larger
than every other layer).  TIDL's float32 calibration forward pass produces
stats max=16.5M there (should be ~17.5).  This test isolates whether the
overflow is purely weight-magnitude-driven or requires the specific shortcut
topology of the identity block.

Two tests:
  test_stats_sane_small_weights  — control: w_max=0.3 (like MV2), expect max < 50
  test_stats_overflow_large_weights — probe: w_max=3.648 (like layer4[1])

Interpretation:
  Both sane  → overflow needs the shortcut topology → fix in tensor wiring
              (Phase 2B: TIDL_quantStatsFixedOrFloat / tensor linking)
  Large overflows, small sane → weight magnitude alone triggers it
              (Phase 2A: per-layer weight normalization before calibration)

Requirements: tidl_model_import_relax.so (no hardware, no TI C7x toolchain).
"""

import glob
import os

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.frontend import nn

C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH",
    os.path.expanduser("~/ml/c7x-mma-tidl"),
)
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH,
    "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so",
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(RELAX_SO_PATH),
    reason=f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}",
)


@pytest.fixture(scope="module", autouse=True)
def _load_so():
    if tvm.get_global_func("TIDL_relaxInit", allow_missing=True) is not None:
        return
    tvm.runtime.load_module(RELAX_SO_PATH)


# ---------------------------------------------------------------------------
# Model: single conv2d_bias on 512-channel 7×7 feature map
# (same shape as ResNet-18 layer4[1]'s second conv, no shortcut)
# ---------------------------------------------------------------------------


class SingleConvBias(nn.Module):
    """512→512 conv2d with bias, 3×3 kernel, same padding — no residual."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(512, 512, 3, padding=1, bias=True)

    def main(self, x: nn.spec.Tensor((1, 512, 7, 7), "float32")):  # type: ignore[override]
        return self.conv(x)


def _build_and_import(tmp_path, w_max: float, label: str) -> float:
    """Export, bind scaled weights, run tidl_import, return stats max.

    Weights are scaled so that np.abs(W).max() == w_max exactly, matching
    the BN-folded weight magnitude of the target layer.
    """
    from tvm.relax.backend.tidl import TIDLOffloadCompiler

    model = SingleConvBias()
    input_spec = {"x": nn.spec.Tensor((1, 512, 7, 7), "float32")}
    mod, param_spec = model.export_tvm(spec={"main": input_spec})

    device = tvm.cpu()
    bound_params = []
    for name, param in param_spec:
        arr = np.random.randn(*param.shape).astype("float32")
        if arr.ndim == 4:  # conv weight
            arr = arr * (w_max / np.abs(arr).max())
        else:              # bias: keep small
            arr = np.zeros(param.shape, dtype=np.float32)
        bound_params.append(tvm.runtime.tensor(arr, device=device))

    param_dict = dict(zip(mod["main"].params[1:], bound_params))
    mod = relax.transform.BindParams(func_name="main", params=param_dict)(mod)

    artifacts_dir = str(tmp_path / f"artifacts_{label}")
    calib = [np.random.randn(1, 512, 7, 7).astype("float32") for _ in range(3)]

    compiler = TIDLOffloadCompiler(
        config={
            "artifacts_dir": artifacts_dir,
            "tidl_tools_path": TIDL_TOOLS_PATH,
            "num_calibration_frames": len(calib),
            "calibration_inputs": calib,
        }
    )

    prepared = compiler.prepare(mod, {})
    partitioned = compiler.partition(prepared)

    tidl_funcs = [
        gv
        for gv, f in partitioned.functions.items()
        if isinstance(f, relax.Function)
        and f.attrs
        and f.attrs.get("Codegen") == "tidl"
    ]
    if not tidl_funcs:
        pytest.skip("No TIDL subgraphs found — conv2d_bias pattern not matched")

    compiler.tidl_import(partitioned)

    stats_files = sorted(glob.glob(os.path.join(artifacts_dir, "*_stats_tool_out.bin")))
    assert stats_files, f"No stats binary found in {artifacts_dir}"

    values = np.fromfile(stats_files[0], dtype=np.float32)
    stats_max = float(np.abs(values).max())
    print(f"\n[{label}] w_max={w_max:.3f}  stats: {len(values)} floats  "
          f"max={stats_max:.2f}  mean={values.mean():.4f}")
    return stats_max


class TestLargeWeightCalibIsolation:
    """Phase 1 diagnostic: standalone conv with large vs small weights."""

    def test_stats_sane_small_weights(self, tmp_path):
        """Control: w_max=0.3 (typical MV2 conv) — stats must be sane."""
        stats_max = _build_and_import(tmp_path, w_max=0.3, label="small")
        assert stats_max < 500, (
            f"Control failed: w_max=0.3 produced stats_max={stats_max:.1f} "
            "(expected < 500 for normal weights)"
        )

    def test_stats_large_weights(self, tmp_path):
        """Probe: w_max=3.648 (ResNet-18 layer4[1] second conv).

        Passes unconditionally — records the stats max so the test output
        shows whether the overflow is topology-independent.

        If stats_max > 10_000: overflow fires WITHOUT the shortcut.
            → Fix in Phase 2A: weight normalization in TIDL_quantStatsFixedOrFloat.
        If stats_max < 500: overflow REQUIRES the shortcut topology.
            → Fix in Phase 2B: tensor wiring in tidl_copyPCNetToDeviceNet.
        """
        stats_max = _build_and_import(tmp_path, w_max=3.648, label="large")
        # Record conclusion in the test output rather than asserting a threshold.
        if stats_max > 10_000:
            print(f"\nCONCLUSION: overflow is topology-INDEPENDENT "
                  f"(stats_max={stats_max:.0f} >> 10000). "
                  "Fix target: Phase 2A (weight normalization).")
        else:
            print(f"\nCONCLUSION: overflow requires the shortcut "
                  f"(stats_max={stats_max:.1f} < 10000). "
                  "Fix target: Phase 2B (tensor wiring).")
