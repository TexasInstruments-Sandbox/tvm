# Testing Overview

Tests for the TVM TI DSP runtime, covering DSP model execution on C66x
(AWRL6844) and C7x (J722S/AM67A) targets. Located at
`tests/ti-dsp-runtime/`.

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
| `unit-tests/` | Standalone kernel unit tests (activation, avg-pool, clamp, concat, dequantize) |
| `wheel-tests/` | Wheel packaging tests: compile + inference e2e via installed `tvm_ti_c7x_*` wheels |

## Test Tiers

Three depth markers control how much of the DSP test suite (`dsp-tests/`)
runs at each stage:

| Marker | Purpose | DSP tests | Time (host) | Time (board) |
|--------|---------|-----------|-------------|-------------|
| `quick` | PR gate | 37 tests | ~20 s | ~2 min |
| `core` | Post-merge | 61 tests | ~10 min | ~25 min |
| *(none)* | Nightly | 73 tests | not tracked separately | not tracked separately |

`core` is a superset of `quick` for almost all tests; the exception is
`test_mmalib_oc_tile_consistency.py`, which is `quick`-only. See
[DSP Test Suite](dsp-suite.md) for the full test catalogue and
per-file timing.

## Jenkins Pipeline Commands

### Required environment setup

```bash
export TVM_HOME=$(pwd)          # or wherever the tvm repo is
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
```

`TI_CGT_C7000_PATH` is required for all DSP e2e tests.

> **Note:** the current `Jenkinsfile` runs a much richer pipeline than
> the manual commands below — separate stages for `unit-tests/`,
> `mmalib-tests/`, `pt2e-tests/`, `quantized/`, `wheel-tests/`, and
> `SmolLM/` in addition to `dsp-tests/`. The commands here are a manual
> reference for running the DSP suite by hand; there is no c66x stage in
> CI today, but the commands below remain valid for anyone running the
> C66x (AWRL6844) path manually.

### c66x host stage (no hardware)

```bash
cd $TVM_HOME

# PR gate:
pytest tests/ti-dsp-runtime/dsp-tests/ -m "quick and not c7x_only" --dsp-mode=c66x_host -v

# Full — all tests valid for c66x:
pytest tests/ti-dsp-runtime/dsp-tests/ -m "not c7x_only" --dsp-mode=c66x_host -v
```

### c7x host stage (no hardware, needs TI_CGT_C7000_PATH)

```bash
cd $TVM_HOME

# PR gate:
pytest tests/ti-dsp-runtime/dsp-tests/ -m quick --dsp-mode=c7x_host -v

# Post-merge gate:
pytest tests/ti-dsp-runtime/dsp-tests/ -m core --dsp-mode=c7x_host -v
```

### c7x board stage (AM67A required; never run in background or in parallel)

```bash
cd $TVM_HOME

# PR gate:
pytest tests/ti-dsp-runtime/dsp-tests/ -m quick --dsp-mode=c7x_dload -v

# Post-merge gate:
pytest tests/ti-dsp-runtime/dsp-tests/ -m core --dsp-mode=c7x_dload -v

# Nightly full regression:
pytest tests/ti-dsp-runtime/dsp-tests/ --dsp-mode=c7x_dload -v
```

> **Important:** c7x_dload tests must not be run in parallel or in the
> background.  The AM67A board has a single DSP core; concurrent sessions
> cause DMA-BUF exhaustion and firmware hangs.

## DSP Tests

DSP tests require `--dsp-mode` to select the execution backend.  Tests
are further organised by architecture marker:

| Marker | Meaning | Jenkins filter |
|--------|---------|---------------|
| `c7x_only` | Model too large for C66x, or c7x-specific API | `-m "not c7x_only"` on c66x stages |
| *(none)* | Works on both c66x and c7x | no filter needed |

See [DSP Test Suite](dsp-suite.md) for the full test catalogue,
per-file timing, and standalone script usage.

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
