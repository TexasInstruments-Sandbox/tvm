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

## Quick Start

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# TIDL partition + codegen tests (no hardware needed):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_codegen.py -v

# DSP smoke test (host emulation):
pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c7x_host -m quick

# TIDL hardware e2e test (needs AM67A + tidl_model_import_relax.so):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_import_e2e.py -v -s
```

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
