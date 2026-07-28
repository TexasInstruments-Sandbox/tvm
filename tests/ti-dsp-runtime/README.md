# TI DSP Runtime Tests

Tests for the TVM TI DSP runtime, covering DSP model execution and
TIDL subgraph offloading on C66x (AWRL6844) and C7x (J722S/AM67A)
targets.

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `dsp-cpp/` | Build infrastructure: CMakeLists.txt, dsp_utils.py, c7x_dynmod linker scripts, and C++ integration examples |
| `dsp-tests/` | Pytest-based DSP model tests (conv2d, resnet, YOLO, etc.) for c66x_host, c7x_host, and c7x_dload modes |
| `dynamic-tests/` | Dynamic shape and control-flow tests (Relax `If`, dynamic batch) on the c_static/C7x backend |
| `mmalib-tests/` | MMALIB direct-offload tests (int8/int16 conv2d, matmul, depthwise) on the C7x MMA accelerator |
| `pt2e-tests/` | `C7xMMAQuantizer` PT2E quantization and activation/pool/norm fusion tests |
| `quantized/` | INT8/INT16 quantized end-to-end model tests (ResNet, MobileNet, GoogLeNet, Inception) |
| `SmolLM/` | SmolLM-135M LLM chat e2e tests: KV cache, IPC protocol, board inference |
| `tidl-tests/` | TIDL subgraph offloading tests: partitioning, import, codegen, and end-to-end hardware inference |
| `unit-tests/` | Standalone kernel unit tests (activation, avg-pool, clamp, concat, dequantize) |
| `wheel-tests/` | Wheel packaging tests: compile + inference e2e via installed `tvm_ti_c7x_*` wheels |

## Test Tiers

Three depth markers control how much of the suite runs at each stage:

| Marker | Purpose | DSP tests | TIDL tests | Time (host) | Time (board) |
|--------|---------|-----------|-----------|-------------|-------------|
| `quick` | PR gate | 37 tests | 138 tests | ~20 s | ~5 min |
| `core` | Post-merge | 61 tests | 132 tests | ~10 min | ~25 min |
| *(none)* | Nightly | 73 tests | 169 tests (all) | 40+ min | 2–3 h |

`core` is a superset of `quick` for almost all tests; a handful of
hardware smoke tests run only under `quick` and are excluded from
`core` (TIDL `test_tidl_new_ops.py`/`test_tidl_import_e2e.py`; DSP
`test_mmalib_oc_tile_consistency.py`).

## Jenkins Pipeline Commands

### Required environment setup

```bash
export TVM_HOME=$(pwd)          # or wherever the tvm repo is
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
```

`TI_CGT_C7000_PATH` is required for all DSP and TIDL e2e tests.

> **Note:** the current `Jenkinsfile` only runs `c7x_host` and
> `c7x_dload` stages (plus `unit-tests/`, `mmalib-tests/`, `pt2e-tests/`,
> `quantized/`, `wheel-tests/`, and `SmolLM/`, not shown below). There
> is no c66x stage in CI today; the commands below remain valid for
> anyone running the C66x (AWRL6844) path manually.

### c66x host stage (no hardware)

```bash
cd $TVM_HOME

# PR gate — codegen unit tests only, ~10 s:
pytest tests/ti-dsp-runtime/tidl-tests/ -m quick -v
pytest tests/ti-dsp-runtime/dsp-tests/ -m "quick and not c7x_only" --dsp-mode=c66x_host -v

# Full — all tests valid for c66x, ~35 min:
pytest tests/ti-dsp-runtime/tidl-tests/ -m "not core or core" -v   # all tidl (same as no filter)
pytest tests/ti-dsp-runtime/dsp-tests/ -m "not c7x_only" --dsp-mode=c66x_host -v
```

### c7x host stage (no hardware, needs TI_CGT_C7000_PATH)

```bash
cd $TVM_HOME

# PR gate — ~25 s:
pytest tests/ti-dsp-runtime/tidl-tests/ -m quick -v
pytest tests/ti-dsp-runtime/dsp-tests/ -m quick --dsp-mode=c7x_host -v

# Post-merge gate — ~10 min:
pytest tests/ti-dsp-runtime/tidl-tests/ -m core -v
pytest tests/ti-dsp-runtime/dsp-tests/ -m core --dsp-mode=c7x_host -v
```

### c7x board stage (AM67A required; never run in background or in parallel)

```bash
cd $TVM_HOME

# PR gate — ~5 min:
pytest tests/ti-dsp-runtime/tidl-tests/ -m quick -v
pytest tests/ti-dsp-runtime/dsp-tests/ -m quick --dsp-mode=c7x_dload -v

# Post-merge gate — ~25 min:
pytest tests/ti-dsp-runtime/tidl-tests/ -m core -v
pytest tests/ti-dsp-runtime/dsp-tests/ -m core --dsp-mode=c7x_dload -v

# Nightly full regression — 2–3 h:
pytest tests/ti-dsp-runtime/tidl-tests/ -v                          # all TIDL
pytest tests/ti-dsp-runtime/dsp-tests/ --dsp-mode=c7x_dload -v     # all DSP
```

> **Important:** c7x_dload tests must not be run in parallel or in the
> background.  The AM67A board has a single DSP core; concurrent sessions
> cause DMA-BUF exhaustion and firmware hangs.

## TIDL Tests

TIDL tests are always c7x-only and have a separate dependency axis:

| File | Tier | Dependencies | Time |
|------|------|-------------|------|
| `test_tidl_partition.py` (17) | **quick + core** | TVM only | ~4 s |
| `test_tidl_codegen.py` (12) | **quick + core** | TVM only | ~4 s |
| `test_tidl_layer_offload.py` (103) | **quick + core** | Level 1 (61, pattern-matching): TVM only; Level 4 (42, hardware): `.so` + TI_CGT + AM67A, `skipif`-gated | ~1 s (Level 1) / ~10 min (Level 4, board) |
| `test_tidl_e2e.py` (2) | nightly | TI_CGT_C7000_PATH only | ~30 s |
| `test_tidl_new_ops.py` (4) | **quick** | tidl_model_import_relax.so + AM67A | ~5 min |
| `test_tidl_import_e2e.py` (2) | **quick** | .so + TI_CGT + AM67A | ~90 s |
| `test_tidl_mv2_e2e.py` (2) | nightly | .so + TI_CGT + AM67A | ~2 min |
| `test_tidl_relax_import.py` (18) | nightly | tidl_model_import_relax.so | ~60 s |
| `test_tidl_resnet_e2e.py` (3) | nightly | .so + TI_CGT + AM67A | 2–10 min |
| `test_yolo_dsp.py` (6) | nightly (`c7x_only`) | torch.hub/ultralytics weights + c7x_dload AM67A; `TestYOLOTIDL` variants also need `.so` + TI_CGT | ~5–10 min |

**Note on TIDL `quick`**: The 138 TIDL `quick` tests split into three groups:
- **90 dependency-free** (partition + codegen + layer-offload Level 1 pattern-matching): always
  pass — these are also `core`
- **42 layer-offload hardware tests** (Level 4): require `.so` + TI_CGT + AM67A — `skipif`-gated
  at runtime rather than deselected, and are still part of `core`
- **6 hardware e2e** (new_ops + import_e2e): require `.so` + AM67A — skip on host stages,
  `quick`-only (not `core`), and provide the full hardware gate on the c7x_board stage

**ResNet-18 TIDL tests are nightly-only** — they require
`tidl_model_import_relax.so` (external build artifact), take 2–10 minutes
each, and two of the three tests need AM67A hardware. Not in `core`.

## DSP Tests

DSP tests require `--dsp-mode` to select the execution backend.  Tests
are further organised by architecture marker:

| Marker | Meaning | Jenkins filter |
|--------|---------|---------------|
| `c7x_only` | Model too large for C66x, or c7x-specific API | `-m "not c7x_only"` on c66x stages |
| *(none)* | Works on both c66x and c7x | no filter needed |

See `dsp-tests/README.md` for the full test catalogue, per-file timing,
and standalone script usage.

## History

These tests were originally developed in a separate `tests/ti-dsp-runtime`
repository and moved into the tvm repo for version consistency.  Key
development milestones:

- DSP runtime integration tests with golden verification
- C66x (AWRL6844) hardware support via JTAG/CCS
- pytest-based test infrastructure (dsp-tests/)
- C7x (J722S/AM67A) support: MMU, cache coherency, DLOAD dynamic
  module loading, remoteproc firmware
- DMA tiling with UDMA/DRU subsystem
- TIDL subgraph offloading: pattern matching, Relax FFI import,
  code generation, bridge generation, hardware inference
- IOBufDesc struct fix (TIDL_IO_MAX_NUM_CORES=4 vs TIDL_MAX_NUM_CORES=2)
