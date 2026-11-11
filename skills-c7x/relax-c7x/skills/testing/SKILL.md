---
name: testing
description: "Writing and running TVM C7x DSP tests. Use when: adding new pytest test cases, structuring tests for c7x_host/c7x_dload, getting cycle counts, layer-level profiling, debugging test failures (DSP_KEEP_TEMP), recording performance metrics (record_cycles/cycles.csv), writing SmolLM e2e tests (compile/deploy/board inference), or understanding test markers (quick/core/c7x_only), fixtures (dsp_mode/dsp_config/profile), and the dsp_utils helper API. NOT for build/deploy steps (see build) or operator implementation (see dsp-ops)."
---

# Testing for TVM C7x DSP

## Test Directory Layout

```
tests/ti-dsp-runtime/
├── dsp-tests/               <- Main integration tests (73 tests)
│   ├── conftest.py          <- Fixtures, markers, CLI options, cycle tracking
│   ├── model_utils.py       <- Model creation helpers
│   └── test_*.py
├── mmalib-tests/            <- MMALIB kernel unit tests (int8 + int16)
├── pt2e-tests/              <- C7xMMAQuantizer + int8/int16 PT2E pipeline tests
│   ├── pt2e_utils.py        <- quantize_pt2e / e2e_quantize_and_import / run_and_check
│   ├── test_c7x_mma_quantizer.py      <- Annotator unit tests (pure Python)
│   ├── test_c7x_mma_quantizer_i16.py  <- Int16 import + fusion tests (pure Python)
│   ├── test_c7x_mma_quantizer_e2e_dsp.py  <- Int8 + int16 e2e DSP tests
│   └── test_mobilenet_v2_pt2e_dsp.py  <- MobileNetV2 integration test
├── tidl-tests/              <- TIDL partition + e2e tests
├── quantized/               <- Full quantized model tests
├── SmolLM/                  <- LLM e2e pipeline
│   ├── smollm_c7x.py       <- compile/infer/test CLI
│   └── smollm_board.py     <- Board-side inference (no TVM/torch)
├── dsp-cpp/
│   └── dsp_utils.py         <- Core helpers (compile, build, run)
└── results/                 <- cycles.csv output
```

## Writing a New Test

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "dsp-cpp"))
from dsp_utils import compile_and_run_dsp, get_target_string, assert_dsp_comparison

@pytest.mark.quick
@pytest.mark.core
def test_my_op_dsp(dsp_mode, dsp_timeout, use_cpp_api, record_cycles):
    """Test my_op on DSP comparing against PyTorch reference."""
    tvm_mod, torch_model, input_data = create_my_model()

    with torch.no_grad():
        torch_result = torch_model(torch.from_numpy(input_data)).numpy()

    target = get_target_string(dsp_mode, use_cpp_api=use_cpp_api)
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod, input_data=input_data,
        target_string=target, execution_mode=dsp_mode,
        timeout_ms=dsp_timeout,
    )
    record_cycles("my_op", dsp_results.get("c7x_dload_cycles", 0))
    assert_dsp_comparison(dsp_results, {"max_diff": ..., ...})
```

### Key Patterns

1. **Always dual-mode**: Tests must work on both `c7x_host` and `c7x_dload`. Use `dsp_mode` fixture, never hardcode.
2. **PyTorch reference**: Compare DSP output against PyTorch. Use `np.allclose(rtol=1e-4, atol=1e-5)` for float32.
3. **Standalone script mode**: Add `if __name__ == "__main__":` with argparse for manual debugging.
4. **Markers**: `@pytest.mark.quick` for <30s tests, `@pytest.mark.core` for post-merge gate.

## Markers

| Marker | Purpose |
|--------|---------|
| `quick` | Small model, fast compile+run (~20s host, ~5min hw) |
| `core` | All core ops + classification + accuracy (post-merge) |
| `c7x_only` | C7x-specific feature or model too large for C66x |
| `dsp_host_only` | Too large for C66x hardware |
| `requires_c7x_firmware` | Needs live firmware on the target board (am67a or beagley-ai) |

## Fixtures (dsp-tests/conftest.py)

| Fixture | Type | Source |
|---------|------|--------|
| `dsp_mode` | str | `--dsp-mode` |
| `dsp_timeout` | int | `--dsp-timeout` (default 60000) |
| `use_cpp_api` | bool | `--use-cpp-api` |
| `profile` | bool | `--profile` (compile + repeat=2) |
| `mmalib` | bool | `--mmalib` |
| `save_artifacts` | str | `--save-artifacts DIR` |
| `board_target` | str | `--board-target HOST` — **only** un-skips/redirects the `test_c7x_vm_dsp` SSH integration tests. Does not affect the general `c7x_dload` path below. |
| `record_cycles` | func | `record_cycles(name, cycles)` → cycles.csv |

Every `c7x_dload` run, and every TIDL test, needs `--board j722s-evm|beagley-ai|am62d` (root
`tests/ti-dsp-runtime/conftest.py`; no default). It picks the SSH host
(`j722s-evm` -> `am67a`, `beagley-ai` -> `beagley-ai`, `am62d` -> `am62d`; a lab host under another
name is an SSH-config alias) and, through `get_current_board()`, the compiler device and TIDL board
config. No environment variable overrides the host; the deploy scripts use the same mapping.
```bash
pytest --rootdir=. dsp-tests/test_matmul_dsp.py -m quick --dsp-mode=c7x_dload --board beagley-ai -v
```
This assumes firmware + the `c7x_compute`/`libc7x_arm_runtime.so` ARM client
are already deployed on that board, at the same service version
(see `relax-c7x:build`/`:firmware`). `--board-target` is separate: it only
un-skips the `test_c7x_vm_dsp` SSH integration tests.

## dsp_utils API

| Function | Purpose |
|----------|---------|
| `get_target_string(mode, profile_layers, use_cpp_api)` | Build the c_static_lib target (a legacy `c_static_lib -opt=val` string, which the fork converts to the 0.27 dict form) |
| `compile_and_run_dsp(mod, input_data, target_string, execution_mode, timeout_ms)` | Full pipeline → results dict |
| `compile_for_dsp(mod, input_data, target_string)` | Compile only (returns workspace) |
| `build_dsp_dynmod(workspace, mode)` | Build DLOAD ELF |
| `run_dsp_dload(module_path, weights_path, inputs, embedded_weights=...)` | Run on the board; `target_host` defaults to the host for `--board` |
| `assert_dsp_comparison(dsp_results, comparison)` | Assert pass/fail |

## Running Tests

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Quick regression
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_host -v                     # ~20s
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_dload --board j722s-evm -v  # ~5min

# Full suite
pytest --rootdir=. dsp-tests/ --dsp-mode=c7x_host -v                              # 73 tests, ~5min
pytest --rootdir=. dsp-tests/ --dsp-mode=c7x_dload --board j722s-evm -v           # 73 tests, ~2-3h

# With profiling (c7x_dload only)
pytest --rootdir=. dsp-tests/test_conv2d_dsp.py -v \
    --dsp-mode=c7x_dload --board j722s-evm --use-cpp-api --profile

# Against BeagleY-AI instead of am67a
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_dload --board beagley-ai -v
```

Invocation rules that bite on this tree:
- **One `tests/ti-dsp-runtime` subdirectory per pytest run.** Several directories in one run load two
  `conftest.py` files that both add `--dsp-mode`, and pytest aborts.
- Run from the repo root with `-p no:tvm.testing.plugin --rootdir=tests/ti-dsp-runtime`, or from
  `tests/ti-dsp-runtime` with `--rootdir=.`.
- A file that selects 0 tests under `-m quick` exits 5, not 0: check its markers before treating it as a failure.
- The c7x_host runtime library must exist first: `cd src/runtime/ti_dsp && bash build_runtime.sh c7x_host`.
- Keep the pinned tvm-ffi (see `relax-c7x:build`); don't run tests through `uv run`.
- Never run two `c7x_dload` sessions on the same board. One session per board, in parallel, is fine.
- TIDL tests read the TIDL tools directory: the packaged `python/tvm/data/ti_dsp/tidl` (stage it in a
  source tree with `cd src/runtime/ti_dsp && bash build_runtime.sh tidl-tools`; an installed
  `--tidl ON` wheel already has it), or `--tidl-tools-dir=<dir>`. No env var selects it; the old
  `C7X_MMA_TIDL_PATH`/`tidl_tools_path` test plumbing is gone. Unstaged → library-gated tests skip.
  Re-stage after editing `tidl_api.c`/`tidl_host_stubs.c` for `c7x_host`: the harness links the
  staged `host/libtidl_api_host.a`. Pass `**tidl_tools_config()` (from `dsp_utils`) into a test's
  compiler config; gate with `has_tidl_tools(need)`.
- Testing against an **installed wheel** (not the source tree): TIDL builds need no `TVM_HOME`,
  `C7X_MMA_TIDL_PATH` or `PSDK_INSTALL_PATH` -- the wheel carries the TIDL tools directory and the
  `c7x_host` harness, and `tidl_api.c` is in the firmware for `c7x_dload`.
- TIDL imports write fixed `/tmp` files (`perfsimPrivData.bin`, `nc_infofile.txt`, `netbufinfo.bin`). If another user
  (Jenkins `sdomcbuild`) owns them, the import fails with "Memory Planning Failed", and a local TIDL run leaves files
  that later break Jenkins. Don't run TIDL tests locally while Jenkins is in a TIDL stage; remove the files afterwards.
- The wheel must ship `libtvm_runtime_extra.so` (sort kernels such as `tvm.contrib.sort.topk`). The backend autoloader
  silently skips a missing library, so the symptom is a late `Cannot find function tvm.contrib.sort.topk`.

### Quantized MMALIB smoke test (beagley-ai)

9-model INT8-quantized TorchVision sweep with MMALIB offload -- the board
regression check for quantization/MMALIB/codegen changes. Run from
`tests/ti-dsp-runtime`, on a power-cycled board with firmware deployed, via
`docker/bash.sh --net=host` (see `relax-c7x:build`):

```bash
python -m pytest --rootdir=. \
  quantized/test_quantized_resnet.py \
  quantized/test_quantized_mobilenet_v2.py \
  quantized/test_quantized_mobilenet_v3.py \
  quantized/test_quantized_googlenet.py \
  quantized/test_quantized_shufflenet_v2.py \
  quantized/test_quantized_inception_v3.py \
  quantized/test_quantized_resnext101.py \
  'quantized/test_quantized_torchvision.py::test_quantized_torchvision_dsp[resnet50]' \
  'quantized/test_quantized_torchvision.py::test_quantized_torchvision_dsp[densenet121]' \
  --dsp-mode=c7x_dload --mmalib --board beagley-ai -v
```

Covers ResNet-18, MobileNetV2, MobileNetV3-Large, GoogLeNet, ShuffleNetV2,
InceptionV3, ResNeXt-101, ResNet-50, DenseNet-121 (~20 min). Prereqs:
pretrained weights cached under the container's `~/.cache/torch/hub/checkpoints`
(network works with `--net=host`), `requests` + `pillow` in the venv (the
torchvision sweep imports them), firmware deployed.

## Cycle Counts and Profiling

**Total cycles**: `dsp_results["c7x_dload_cycles"]` — extracted from `c7x_compute` stdout. Use `record_cycles(name, cycles)` → writes `results/cycles.csv`.

**Per-layer profiling** (`--profile`): compiles with `-profile-layers=1`, runs with `repeat=2`. Output via DSP printf:
```
===== TVM Layer Profile =====
Layer  0 (conv2d):     1,234,567 cycles
Layer  1 (relu):          12,345 cycles
Total:                 1,246,912 cycles
=============================
```

**Cycle breakdown tests**: `test_conv2d_cycle_breakdown.py` injects TSC reads to attribute cycles to regions (pad_setup, zero_fill, reduction, post_conv).

## Debugging and SmolLM

- **Debugging failures**: See [references/debugging.md](references/debugging.md) for DSP_KEEP_TEMP, failure mode table, hardware recovery.
- **SmolLM e2e pipeline**: See [references/smollm-e2e.md](references/smollm-e2e.md) for compile-chat/deploy/board workflow.

## Related Skills

- `relax-c7x:build` — Environment setup and build commands
- `relax-c7x:dsp-ops` — Writing new operator implementations
- `relax-c7x:firmware` — Firmware recovery when tests cause hangs
