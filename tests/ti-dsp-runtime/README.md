# TI DSP Runtime Tests

Tests for the TVM TI DSP runtime, covering DSP model execution and
TIDL subgraph offloading on C66x (AWRL6844) and C7x (J722S/AM67A)
targets.

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `dsp-cpp/` | Build infrastructure: CMakeLists.txt, dsp_utils.py, c7x_dynmod linker scripts, and C++ integration examples |
| `dsp-tests/` | Pytest-based DSP model tests (conv2d, resnet, YOLO, etc.) for c66x_host, c7x_host, and c7x_dload modes |
| `tidl-tests/` | TIDL subgraph offloading tests: partitioning, import, codegen, and end-to-end hardware inference |

## Test Tiers

Three depth markers control how much of the suite runs at each stage:

| Marker | Purpose | DSP tests | TIDL tests | Time (host) | Time (board) |
|--------|---------|-----------|-----------|-------------|-------------|
| `quick` | PR gate | 6 core models | 25 partition+codegen | ~20 s | ~5 min |
| `core` | Post-merge | ~38 tests | +1 stub e2e | ~10 min | ~25 min |
| *(none)* | Nightly | ~80 tests | all | 40+ min | 2–3 h |

`core` is a strict superset of `quick`.

## Jenkins Pipeline Commands

### Required environment setup

```bash
export TVM_HOME=$(pwd)          # or wherever the tvm repo is
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
```

`TI_CGT_C7000_PATH` is required for all DSP and TIDL e2e tests.

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
| `test_tidl_partition.py` (13) | **quick + core** | TVM only | ~4 s |
| `test_tidl_codegen.py` (12) | **quick + core** | TVM only | ~4 s |
| `test_tidl_e2e.py` (1) | nightly | TI_CGT_C7000_PATH only | ~30 s |
| `test_tidl_new_ops.py` (4) | **quick** | tidl_model_import_relax.so + AM67A | ~5 min |
| `test_tidl_import_e2e.py` (2) | **quick** | .so + TI_CGT + AM67A | ~90 s |
| `test_tidl_relax_import.py` (8) | nightly | tidl_model_import_relax.so | ~60 s |
| `test_tidl_resnet_e2e.py` (3) | nightly | .so + TI_CGT + AM67A | 2–10 min |

**Note on TIDL `quick`**: The 31 TIDL `quick` tests split into two groups:
- **25 dependency-free** (partition + codegen): always pass — these are also `core`
- **6 hardware e2e** (new_ops + import_e2e): require `.so` + AM67A — skip on host stages but
  provide the full hardware gate on the c7x_board stage

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
- DMA tiling with EDMA/DRU subsystem
- TIDL subgraph offloading: pattern matching, Relax FFI import,
  code generation, bridge generation, hardware inference
- IOBufDesc struct fix (TIDL_IO_MAX_NUM_CORES=4 vs TIDL_MAX_NUM_CORES=2)
