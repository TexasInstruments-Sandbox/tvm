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

## Quick Regression Tests

All quick tests are marked with `@pytest.mark.quick`.  Run them as a
fast regression check before pushing changes.

### Required environment setup

```bash
export TVM_HOME=$(pwd)          # or wherever the tvm repo is
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
```

`TI_CGT_C7000_PATH` is required for all DSP and TIDL e2e tests.
Tests will **fail** (not skip) if it is missing.

### TIDL quick tests (27 tests, ~90 s)

Partition, codegen, and small-model hardware e2e.  No `--dsp-mode`
needed.  The 2 hardware e2e tests require the TIDL import .so and
C7x compiler — they fail with a clear message if either is missing.

```bash
pytest tests/ti-dsp-runtime/tidl-tests/ -m quick -v
```

| File | Tests | Time | Dependencies |
|------|------:|-----:|--------------|
| `test_tidl_partition.py` | 14 | ~4 s | None (torch optional for ResNet BN tests) |
| `test_tidl_codegen.py` | 12 | ~4 s | None |
| `test_tidl_import_e2e.py` | 2 | ~80 s | tidl_model_import_relax.so, TI_CGT_C7000_PATH, AM67A board |

### DSP quick tests (6 tests)

Small models: conv2d, matmul, MLP, CLISTA, conv2d-stack, quantized
conv2d-stack.  Requires `--dsp-mode` to select the execution backend.

```bash
# C7x host emulation -- no hardware, ~20 s:
cd tests/ti-dsp-runtime
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_host -v

# C7x DLOAD on AM67A hardware -- ~5 min:
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_dload -v
```

| File | Tests | c7x_host | c7x_dload |
|------|------:|---------:|----------:|
| `test_clista_dsp.py` | 1 | ~3 s | ~20 s |
| `test_conv2d_dsp.py` | 1 | ~2 s | ~8 s |
| `test_conv2d_stack_dsp.py` | 1 | ~3 s | ~90 s |
| `test_matmul_dsp.py` | 1 | ~2 s | ~8 s |
| `test_mlp_dsp.py` | 1 | ~3 s | ~20 s |
| `test_quantized_conv2d_stack_dsp.py` | 1 | ~15 s | ~3 min |

**Important:** c7x_dload tests must not be run in parallel or in the
background.  The AM67A board has a single DSP core; concurrent
sessions cause DMA-BUF exhaustion and firmware hangs.

### All quick tests in one shot

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS

# 1. TIDL (no --dsp-mode needed):
pytest tests/ti-dsp-runtime/tidl-tests/ -m quick -v

# 2. DSP host emulation:
cd tests/ti-dsp-runtime
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_host -v

# 3. DSP hardware (only if AM67A is available):
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_dload -v
```

Expected total wall time: ~2 min (host-only) or ~8 min (with AM67A).

## Full Test Catalog

Beyond the quick tests, these suites exercise larger models and
longer-running hardware flows:

| Suite | Command | Notes |
|-------|---------|-------|
| TIDL import tests | `pytest tidl-tests/test_tidl_relax_import.py -v` | Needs .so + TIDL tools, ~60 s |
| TIDL ResNet-18 build | `pytest tidl-tests/test_tidl_resnet_e2e.py::TestTIDLResNetE2E::test_tidl_resnet18_build -v` | Needs .so + compiler, ~2 min |
| TIDL ResNet-18 hardware | `pytest tidl-tests/test_tidl_resnet_e2e.py::TestTIDLResNetE2E::test_tidl_resnet18_correctness -v -s` | Needs AM67A, ~5 min |
| DSP classification | `pytest --rootdir=. dsp-tests/test_classification_dsp.py -v --dsp-mode=c7x_dload` | SqueezeNet/MobileNet on AM67A |
| DSP ResNet-18 | `pytest --rootdir=. dsp-tests/test_resnet_dsp.py -v --dsp-mode=c7x_dload` | Full ResNet-18 on AM67A |

See `CLAUDE.md` for full build instructions, environment setup, and
hardware deployment.

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
