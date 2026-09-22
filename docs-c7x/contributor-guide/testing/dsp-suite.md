# DSP Test Suite

Pytest-based tests for running TVM-compiled models on TI DSP targets.
Tests compile models with the `c_static_lib` backend, build for the selected
execution mode, run inference, and compare results against a PyTorch
reference. Located at `tests/ti-dsp-runtime/dsp-tests/`.

## Directory Structure

```
dsp-tests/
├── conftest.py                          # Pytest fixtures for DSP configuration
├── model_utils.py                       # Shared model creation utilities
├── test_c66x_pragmas_dsp.py             # C66x/C7x pragma generation tests
├── test_c7x_vm_dsp.py                   # C7xVirtualMachine Python API
├── test_classification_dsp.py           # TorchVision classification models
├── test_clista_dsp.py                   # CLISTA-DoA radar model
├── test_conv2d_cycle_breakdown.py       # Conv2D O2 vs O3 cycle breakdown benchmark
├── test_conv2d_dsp.py                   # Conv2D model
├── test_conv2d_stack_dsp.py             # Conv2D + BN + ReLU stack (4 layers)
├── test_error_messages_dsp.py           # Compilation and error handling tests
├── test_lenet_dsp.py                    # LeNet-5 MNIST classifier
├── test_matmul_dsp.py                   # Matrix multiplication
├── test_mlp_dsp.py                      # Multi-layer perceptron
├── test_mmalib_oc_tile_consistency.py   # MMALIB conv2d_i8 OC-tiling consistency
├── test_od_torchvision_dsp.py           # SSDLite320 object detection
├── test_quantized_conv2d_stack_dsp.py   # INT8 quantized conv2d stack
├── test_resnet_dsp.py                   # ResNet-18 image classifier
├── test_rtmdet_dsp.py                   # Multi-output tuple handling
└── test_segmentation_dsp.py             # TorchVision segmentation models
```

## MMALIB Test Suite (merged from mmalib-suite.md)

End-to-end tests for TVM c_static_lib backend calling MMALIB functions
directly on the C7x MMA accelerator (AM67A / J722S). Located at
`tests/ti-dsp-runtime/mmalib-tests/`.

### Running MMALIB Tests

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# Quick smoke tests (~2 min host emulation)
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# Full suite (all markers)
pytest --rootdir=. mmalib-tests/ -v --dsp-mode=c7x_host

# Hardware (AM67A board)
pytest --rootdir=. mmalib-tests/ -m quick --dsp-mode=c7x_dload -v
```

### MMALIB Test Files

#### Kernel unit tests (execution required)

| File | Op | Dtype | Path | Description |
|------|----|-------|------|-------------|
| `test_mmalib_matmul_dsp.py` | matmul | int8 | legalize | Direct legalization via `LegalizeOps`, exact match |
| `test_mmalib_matmul_i16_dsp.py` | matmul | int16 | legalize | Float→int16 dynamic quant + shift-based overflow prevention; used by SmolLM MLP offload |
| `test_mmalib_conv2d_dsp.py` | conv2d | int16 | legalize | Direct int16 conv2d legalization, exact match |
| `test_mmalib_conv2d_i8_dsp.py` | conv2d | int8 | QDQ | `FuseMMALIBQDQConv2d` — PT2E pattern with per-channel bias/scale/shift, ±2 tolerance |
| `test_mmalib_conv2d_i16_dsp.py` | conv2d | int16 | QDQ | `FuseMMALIBQDQConv2dI16` — same PT2E pattern but int16, ±10 tolerance (Phase 2b) |
| `test_mmalib_dwconv2d_i8_dsp.py` | depthwise conv2d | int8 | QDQ | `FuseMMALIBQDQDwConv2d` — depthwise (groups=C), 3×3/5×5/7×7, ±2 tolerance |
| `test_mmalib_dwconv_i16_dsp.py` | depthwise conv2d | int16 | QDQ | `FuseMMALIBQDQDwConv2dI16` — int16 depthwise, **3×3 only** (MMALIB-882), ±5 tolerance (Phase 2c) |
| `test_mmalib_fc_i8_dsp.py` | FC / linear | int8 | QDQ | `FuseMMALIBQDQFC` — matmul_bias_i8, per-channel scale/shift; 2D and 3D reshape variants |
| `test_mmalib_fc_i16_dsp.py` | FC / linear | int16 | QDQ + direct | Direct `mmalib_matmul_bias_i16` wrapper tests (SmolLM dims) plus `FuseMMALIBQDQFCI16` PT2E QDQ fusion (Phase 2b) |
| `test_mmalib_residual_add_i8_dsp.py` | residual add | int8 | QDQ | `FuseInt8ResidualAdd` — both `add(x,skip)` and `add(skip,x)` operand orders (Phase 2a) |
| `test_mmalib_residual_add_i16_dsp.py` | residual add | int16 | QDQ | `FuseInt16ResidualAdd` — symmetric only (zp=0), both operand orders (Phase 2c) |
| `test_mmalib_conv2d_i8_grouped_loop_dsp.py` | conv2d (grouped) | int8 | direct | Direct `call_extern` to `mmalib_conv2d_i8_grouped_loop` for ResNeXt101-32x8d's four stage shapes (stride 1/2), exact match (Step 13) |
| `test_mmalib_qdq_grouped_conv2d_i8_dsp.py` | conv2d (grouped) | int8 | QDQ | `FuseMMALIBQDQConv2d` groups>1 path — PT2E pattern end-to-end via `-mmalib=1`, ±2 tolerance (Step 13) |
| `test_mmalib_loop_only_chain_dsp.py` | conv2d (grouped) | int8 | direct | Regression: chains of `mmalib_conv2d_i8_grouped_loop` calls within one inference (2/3-call quick test, 16-call stress test), exact match (Step 13) |

#### Pass-level unit tests (pure Python, no DSP required)

| File | What it tests |
|------|---------------|
| `test_mmalib_inject_dma.py` | `InjectMMALIBDMA` guard bytes: verifies `pad_top` (not `stride_h`) is read from args[15] for i8 and i16 conv2d; fallback to 128 bytes when `pad_top == 0` |
| `test_mmalib_fc_i16_dsp.py` *(guard test)* | `test_fuse_fc_i16_rejects_nonzero_o_zp` — verifies the i16 FC check function rejects patterns with non-zero output zero-point |

### MMALIB Test Execution Paths

**Legalize path** (`test_mmalib_matmul_dsp.py`, `test_mmalib_conv2d_dsp.py`,
`test_mmalib_matmul_i16_dsp.py`): The `LegalizeOps` pass with a custom
`legalize_map` replaces eligible float ops with `call_extern` to MMALIB
wrappers. No quantization nodes in the graph.

**QDQ fusion path** (files marked `QDQ` in the table above): The
`FuseMMALIBQDQ*` passes run *before* `FuseQDQToInt8Conv2D` and match the
intact PT2E QDQ pattern:
```
dequantize(data_int8/16) → op(_, dequantize(weight)) → [bias] → [relu] → quantize
```
The fused kernel receives compile-time-computed integer bias/scale/shift
derived from the quantization parameters.

**Direct call_extern path** (`test_mmalib_conv2d_i8_grouped_loop_dsp.py`,
`test_mmalib_loop_only_chain_dsp.py`): builds `te.extern`/`tir.call_extern`
calls to `mmalib_conv2d_i8_grouped_loop` by hand, bypassing both
`LegalizeOps` and the QDQ fusion passes, to isolate ResNeXt101's
grouped-conv kernel from the rest of the compiler pipeline.

### MMALIB Tolerances

| Dtype | Tolerance | Reason |
|-------|-----------|--------|
| int8 | ≤ 2 | uint8 scale/shift approximation; small K |
| int16 | ≤ 5–10 | wider uint8 scale/shift approximation error for larger K |
| int16 direct | ≤ 1 | per-row L1-norm shift, no requantization |

### MMALIB Data Layout

All ops use NCHW (planar channel-first). The pipeline skips NHWC conversion
when `-mmalib=1` is set.

### MMALIB Known Limitations

- **INT16 depthwise**: only 3×3 kernels supported (`mmalib_depthwise_conv2d_i16`);
  5×5 and 7×7 return `MMALIB_ERR_NOT_IMPLEMENTED` (tracked as MMALIB-882).
- **INT16 QDQ activation quantization**: always symmetric (d_zp = 0 required).
  Asymmetric activation quant (`d_zp ≠ 0`) is rejected by the i16 check
  functions and falls through to float computation.

---

## C Static Lib Test Suite (merged from c-static-lib-suite.md)

Validation suite for the TVM C Static Lib backend (`c_static_lib` target).
Compiles models for both LLVM (reference) and c_static_lib, then compares
outputs within tolerance (rtol=1e-3, atol=1e-5). Located at `tests/cstatic/`.

### Quick Start

```bash
# From repo root
export TVM_HOME=$(pwd)
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Run quick tests in parallel (excludes slow and model_zoo)
cd tests/cstatic
pytest --rootdir=. unit-tests/ -m "not slow and not model_zoo" -n auto -v

# Run all tests (including slow: ViT-B/16, segmentation; excludes model_zoo)
pytest --rootdir=. unit-tests/ -m "not model_zoo" -v

# Debug a failed test (preserve temp workspace)
CSTATIC_KEEP_TEMP=1 pytest --rootdir=. unit-tests/test_resnet.py -v
```

### Prerequisites

1. **TVM build (2-pass)**:
   ```bash
   mkdir -p build && cp cmake/config.cmake build/
   cd build
   cmake -G Ninja .. && ninja           # Pass 1: shared libs (for Python)
   cmake -DBUILD_STATIC_RUNTIME=ON ..
   ninja tvm_runtime                    # Pass 2: libtvm_runtime.a (for c_static_lib)
   cd ..
   ```

2. **cnpy** (NumPy I/O for the C++ test harness):
   ```bash
   cd 3rdparty/cnpy
   mkdir -p build && cd build && cmake .. && make -j$(nproc)
   ```

3. **Python dependencies**:
   ```bash
   uv pip install numpy pytest pytest-xdist torch torchvision onnx Pillow tqdm
   uv pip install -e 3rdparty/tvm-ffi
   ```

### Unit Tests

All automated tests are in `unit-tests/`:

| File | What it tests | Marker |
|------|---------------|--------|
| `test_conv2d.py` | 2D convolution | quick |
| `test_matmul.py` | Matrix multiplication (16x16) | quick |
| `test_mlp.py` | Fully connected layers (784-256-10) | quick |
| `test_resnet.py` | ResNet-18 (torchvision, ImageNet) | quick |
| `test_rtmdet_tvm_minimal.py` | Multi-output (6-tensor tuple) | quick |
| `test_error_messages.py` | Shape mismatch error handling | quick |
| `test_use_cpp_api_codegen.py` | C++ API codegen flag verification | quick |
| `test_vitb16.py` | Vision Transformer ViT-B/16 | slow |
| `test_segmentation.py` | FCN ResNet-50 (dynamic shapes) | slow |
| `test_model_zoo.py` | 111 TorchVision/YOLO models (classification, detection, segmentation, YOLO) | model_zoo |

#### Running by category

```bash
pytest --rootdir=. unit-tests/ -m "not slow and not model_zoo"   # Quick only (~30s)
pytest --rootdir=. unit-tests/ -m "slow and not model_zoo"       # Slow only (~5min)
pytest --rootdir=. unit-tests/test_model_zoo.py                  # Model zoo only (111 models)
pytest --rootdir=. unit-tests/ -m "not model_zoo" -n auto        # Quick + slow, parallel
```

### Standalone Model Scripts

Interactive scripts for broader model coverage (not run by CI):

| Script | Domain | Models |
|--------|--------|--------|
| `cl_torchvision.py` | Classification | ResNet, MobileNet, EfficientNet, ViT |
| `od_torchvision.py` | Detection (COCO) | Faster R-CNN, RetinaNet, FCOS, SSD |
| `od_yolo.py` | Detection (YOLO) | YOLOv5, YOLOv8, YOLOv11 |
| `od_rtmdet.py` | Detection (RTMDet) | RTMDet via MMDetection (Docker) |
| `od_rtmdet_pure.py` | Detection (RTMDet) | RTMDet via rtmdet package |
| `od_rt_detr.py` | Detection (RT-DETR) | RT-DETR transformer detector |
| `seg_torchvision.py` | Segmentation | FCN, DeepLabV3, LRASPP |

Common options: `--tvm`, `--compare`, `--test-all`, `--parallel`.

### How Tests Work

Each test:
1. Creates or loads a model (PyTorch or TVM IR)
2. Compiles for **LLVM** (reference) and **c_static_lib** (target under test)
3. For c_static_lib: exports to C, builds with CMake in an isolated temp dir,
   runs the binary, loads outputs from NPZ
4. Asserts numerical match between LLVM and c_static_lib outputs

The C++ build template is in `cpp/` (CMakeLists.txt + main.cpp).
Each test gets its own `/tmp/cpp_cstatic_XXXXX/` workspace for safe
parallel execution.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `TVM_HOME` | Used by `cpp/CMakeLists.txt` to find TVM headers and libs |
| `CSTATIC_KEEP_TEMP` | Set to `1` to preserve temp workspaces for debugging |

### CI

The Jenkinsfile in this directory runs the full suite:
- 2-pass TVM build (shared + static runtime)
- cnpy build
- Quick tests in parallel (`-n auto`)
- Slow tests (ViT, segmentation) unless `SKIP_SLOW_TESTS` is set
- Model zoo tests (`test_model_zoo.py`) unless `SKIP_MODEL_ZOO` is set, with
  per-category skips (`SKIP_CLASSIFICATION`, `SKIP_DETECTION`,
  `SKIP_SEGMENTATION`, `SKIP_YOLO`)

---

## DSP Test Suite (Original dsp-tests/)

### Test Descriptions

| Test File | Description | Model Type |
|-----------|-------------|------------|
| `test_c66x_pragmas_dsp.py` | C66x/C7x pragma generation and TI compiler directives | TIR codegen |
| `test_c7x_vm_dsp.py` | C7xVirtualMachine Python API (struct layout, inference, context manager) | VM API |
| `test_classification_dsp.py` | 8 ImageNet classifiers (SqueezeNet to ResNet-34) | Conv2D, various |
| `test_clista_dsp.py` | CLISTA-DoA radar signal processing | Conv1D, Linear |
| `test_conv2d_cycle_breakdown.py` | Conv2D O2 vs O3 compiler cycle breakdown benchmark | Conv2D, profiling |
| `test_conv2d_dsp.py` | Single 2D convolution | Conv2D |
| `test_conv2d_stack_dsp.py` | 4-layer conv2d + batch_norm + relu stack | Conv2D, BN, ReLU |
| `test_error_messages_dsp.py` | Compilation validation and error handling | TIR simple ops |
| `test_lenet_dsp.py` | LeNet-5 MNIST classifier | Conv2D, Linear |
| `test_matmul_dsp.py` | Matrix multiplication | Matmul |
| `test_mlp_dsp.py` | Multi-layer perceptron | Linear, ReLU |
| `test_mmalib_oc_tile_consistency.py` | MMALIB `mmalib_conv2d_i8` output-channel tiling consistency | Conv2D, MMALIB |
| `test_od_torchvision_dsp.py` | SSDLite320 MobileNetV3 object detection | Conv2D, multi-output |
| `test_quantized_conv2d_stack_dsp.py` | INT8 quantized conv2d stack (PT2E QDQ) | Conv2D, quantized |
| `test_resnet_dsp.py` | ResNet-18 image classifier | Conv2D, BN, skip |
| `test_rtmdet_dsp.py` | Multi-output tuple handling validation | Conv2D (2 outputs) |
| `test_segmentation_dsp.py` | LRASPP and DeepLabV3 MobileNetV3 segmentation | Conv2D, multi-output |

### Execution Modes

All tests require `--dsp-mode` to select the execution target. There is
no default.

| Mode | Description |
|------|-------------|
| `c66x_host` | C66x host emulation — builds with system gcc, runs on PC |
| `c66x` | C66x hardware — cross-compiles with TI C6000, runs on AWRL6844 via JTAG |
| `c7x_host` | C7x host emulation — builds with system g++ + TI Host Emu library |
| `c7x_dload` | C7x DLOAD — cross-compiles relocatable module, loads on AM67A via c7x_compute |

Not all modes are available for every test. Larger models that exceed
C66x memory are restricted to `c66x_host` and `c7x_dload`.

### Running Tests

#### Via pytest

```bash
# Set environment
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Run a test with C66x host emulation
pytest test_conv2d_dsp.py -v --dsp-mode=c66x_host

# Run on C66x hardware
pytest test_conv2d_dsp.py -v --dsp-mode=c66x

# Run with C7x host emulation
pytest test_conv2d_dsp.py -v --dsp-mode=c7x_host

# Run via C7x DLOAD on AM67A hardware
pytest test_conv2d_dsp.py -v --dsp-mode=c7x_dload

# Run multiple tests
pytest test_conv2d_dsp.py test_mlp_dsp.py test_lenet_dsp.py \
    -v --dsp-mode=c66x_host

# Run quick tests only — PR gate (~20s host, ~2 min board)
pytest -v --dsp-mode=c7x_dload -m quick

# Run core tests — post-merge gate (~10 min host, ~25 min board)
pytest -v --dsp-mode=c7x_dload -m core

# Run all tests valid for c66x — full no-hardware regression
pytest -v --dsp-mode=c66x_host -m "not c7x_only"
```

#### Test depth tiers

Three markers control which tests run at each pipeline stage:

| Marker | Tests | When to use |
|--------|-------|-------------|
| `quick` | 37 tests | PR gate — fast compile + run |
| `core` | 61 tests | Post-merge gate — all ops, classification, detection |
| *(none)* | 73 tests | Nightly full regression |

`core` is a superset of `quick`, with the exception of the 2 unit
tests in `test_mmalib_oc_tile_consistency.py`, which are `quick`-only.

##### `quick` tests (both c66x and c7x, unless noted)

| Test | Model |
|------|-------|
| `test_conv2d_dsp` | Single Conv2D |
| `test_conv2d_stack_dsp` | 4-layer Conv2D + BN + ReLU |
| `test_clista_dsp` | CLISTA-DoA radar |
| `test_matmul_dsp` | Matrix multiplication |
| `test_mlp_dsp` | Multi-layer perceptron |
| `test_quantized_conv2d_stack_dsp` | INT8 quantized Conv2D stack |
| `test_mmalib_oc_tile_consistency` | MMALIB conv2d_i8 OC-tiling consistency |
| `test_c7x_vm_dsp` (all) | c7x only |

##### `core` tests added beyond `quick` (24 additional)

| Test | Architecture |
|------|-------------|
| `test_c66x_pragmas_dsp` (all) | both |
| `test_error_messages_dsp` (all) | both |
| `test_lenet_dsp` | both |
| `test_resnet_dsp` | both |
| `test_classification_dsp` (8 models) | both |

##### `c7x_only` tests (excluded from c66x stages)

Tests marked `c7x_only` use models too large for C66x memory or exercise
c7x-specific features. Jenkins c66x stages filter with `-m "not c7x_only"`:

| File | Reason |
|------|--------|
| `test_c7x_vm_dsp.py` | C7xVirtualMachine API |
| `test_od_torchvision_dsp.py` | SSDLite320 (c7x_dload only) |
| `test_rtmdet_dsp.py` | RTMDet (c7x_dload only) |
| `test_segmentation_dsp.py` | LRASPP / DeepLabV3 |
| `test_conv2d_cycle_breakdown.py` | Cycle profiling benchmark |

#### Jenkins pipeline commands

```bash
cd tests/ti-dsp-runtime

# ── c66x host (no hardware) ──────────────────────────────────────────────
# PR gate
pytest --rootdir=. dsp-tests/ -m quick             --dsp-mode=c66x_host -v
# Full (no hardware)
pytest --rootdir=. dsp-tests/ -m "not c7x_only"   --dsp-mode=c66x_host -v

# ── c7x host (no hardware, needs TI_CGT_C7000_PATH) ─────────────────────
# PR gate
pytest --rootdir=. dsp-tests/ -m quick             --dsp-mode=c7x_host  -v
# Post-merge
pytest --rootdir=. dsp-tests/ -m core              --dsp-mode=c7x_host  -v

# ── c7x board (AM67A, never run in background) ───────────────────────────
# PR gate
pytest --rootdir=. dsp-tests/ -m quick             --dsp-mode=c7x_dload -v
# Post-merge
pytest --rootdir=. dsp-tests/ -m core              --dsp-mode=c7x_dload -v
# Nightly
pytest --rootdir=. dsp-tests/                      --dsp-mode=c7x_dload -v
```

#### Via standalone script

Each test file can also be run directly:

```bash
python test_conv2d_dsp.py --dsp-mode c66x_host
python test_conv2d_dsp.py --dsp-mode c7x_dload -v
python test_resnet_dsp.py --dsp-mode c7x_dload --profile-layers

# Save build artifacts for inspection
python test_clista_dsp.py --dsp-mode c66x_host --save-artifacts /tmp/artifacts
```

#### Command-line options

| Option | Description |
|--------|-------------|
| `--dsp-mode=MODE` | Execution mode (required): `c66x_host`, `c66x`, `c7x_host`, `c7x_dload` |
| `--dsp-timeout=N` | Hardware execution timeout in ms (default: 60000) |
| `--dsp-verbose` | Enable verbose DSP logging |
| `--save-artifacts=DIR` | Copy build artifacts (lib0.c, weights.bin, devc.c) to DIR |
| `--profile` | Per-layer cycle counters + repeat=2 init/steady-state split (c7x_dload only) |
| `--profile-layers` | Deprecated alias for `--profile` |
| `--use-cpp-api` | Enable direct VM builtin calls (bypass FFI dispatch) |
| `--mmalib` | Enable MMALIB acceleration for eligible conv2d/matmul ops |
| `--board-target=HOST` | AM67A hostname for remote `test_c7x_vm_dsp` tests via SSH |

### Key Components

#### `conftest.py`

Pytest fixtures and configuration:
- `dsp_mode`: Execution mode from `--dsp-mode` option (required)
- `dsp_timeout`: Timeout from `--dsp-timeout` option
- `dsp_verbose`: Verbose flag from `--dsp-verbose` option
- `save_artifacts`: Artifact directory from `--save-artifacts` option
- `profile`: Profiling flag from `--profile` (or deprecated `--profile-layers`) option
- `profile_layers`: Alias of `profile`, kept for backward compatibility
- `use_cpp_api`: C++ API flag from `--use-cpp-api` option
- `mmalib`: MMALIB acceleration flag from `--mmalib` option
- `board_target`: AM67A hostname from `--board-target` option (for remote c7x_vm tests)
- `dsp_config`: Combined configuration dictionary

#### `model_utils.py`

Shared model creation functions:
- `torch_to_relax_with_params()`: Convert PyTorch model to TVM with bound parameters
- `create_conv2d_model()`: Single Conv2D layer
- `create_conv2d_stack_model()`: 4-layer conv2d + BN + ReLU stack
- `create_mlp_model()`: Multi-layer perceptron
- `create_matmul_model()`: Matrix multiplication
- `create_clista_model()`: CLISTA-DoA radar model
- `create_lenet_model()`: LeNet-5 CNN
- `create_quantized_conv2d_stack_model()`: INT8 quantized conv2d stack

#### `dsp_utils.py` (in `../dsp-cpp/`)

DSP compilation and execution utilities:
- `get_target_string()`: Map mode to c_static_lib target string
- `assert_dsp_comparison()`: Assert DSP results match reference
- `compile_and_run_dsp()`: End-to-end compile, build, and run
- `compare_results()`: Compare DSP output against reference
- `compile_for_dsp()`: Compile TVM module to C code
- `build_dsp_host()`: Build for C66x host emulation
- `build_dsp_c66x()`: Cross-compile for C66x hardware
- `build_dsp_c7x_host()`: Build for C7x host emulation
- `build_dsp_dynmod()`: Build DLOAD-compatible C7x relocatable module
- `run_dsp_host()`: Run host emulation executable
- `run_dsp_c66x()`: Run on C66x hardware via CCS
- `run_dsp_dload()`: Run on AM67A via c7x_compute CLI

### Adding New Tests

Use `get_target_string()` and `assert_dsp_comparison()` to avoid
per-mode boilerplate:

```python
from dsp_utils import (
    compile_and_run_dsp, compare_results,
    get_target_string, assert_dsp_comparison,
)
from model_utils import create_my_model

def test_my_model(dsp_mode, dsp_timeout, use_cpp_api):
    tvm_mod, torch_model, input_data = create_my_model()

    # PyTorch reference
    with torch.no_grad():
        torch_result = torch_model(torch.from_numpy(input_data)).numpy()

    # Compile and run on DSP
    target_string = get_target_string(dsp_mode, use_cpp_api=use_cpp_api)
    dsp_results = compile_and_run_dsp(
        mod=tvm_mod,
        input_data=input_data,
        target_string=target_string,
        execution_mode=dsp_mode,
        timeout_ms=dsp_timeout,
    )

    # Compare and assert
    comparison = compare_results(dsp_results, torch_result, "PyTorch")
    assert_dsp_comparison(dsp_results, comparison)
```

### Multi-Output Support

The DSP runtime supports models that return multiple outputs (tuples).
`test_rtmdet_dsp.py` validates this:

- Maximum 128 outputs supported (`Model::kMaxOutputs`, compile-time check)
- `Model::InferMulti()` API returns all outputs
- Output tensors written to `output.bin` in order

### Debugging with DSP_KEEP_TEMP

Set `DSP_KEEP_TEMP=1` to preserve the temporary workspace after each test.
The workspace is named after the test and timestamped for easy correlation:

```bash
DSP_KEEP_TEMP=1 pytest test_conv2d_dsp.py -v --dsp-mode=c7x_host
ls /tmp/dsp_test_conv2d_dsp_20260505_154303/
```

Each workspace contains both the TVM-generated code and the native build
artifacts in a single directory:

```
/tmp/dsp_test_conv2d_dsp_20260505_154303/
├── lib0.c                  # TVM-generated C code
├── devc.c                  # Device constants
├── weights.bin             # Model weights
├── model_library.tar       # Exported TVM library
└── build-c7x_host/         # Native build (named after execution mode)
    ├── cg_dsp              # Compiled executable
    ├── cmake.log           # Build log
    ├── input.bin           # Input tensors
    └── output.bin          # Output tensors
```

The build subdirectory is named after the execution mode: `build-c7x_host`,
`build-c7x_dload`, `build-c66x_host`, or `build-c66x`.

### Requirements

- TVM with c_static_lib backend
- PyTorch and torchvision for model creation and reference inference
- **c66x_host**: No additional requirements (system gcc)
- **c66x**: AWRL6844 board with XDS110 debug probe, TI C6000 compiler
  (`TI_CGT_C6000_PATH`)
- **c7x_host**: TI C7000 CGT with Host Emulation library
  (`TI_CGT_C7000_PATH`)
- **c7x_dload**: AM67A (J722S) board at hostname `am67a`, TI C7000
  compiler (`TI_CGT_C7000_PATH`), c7x_compute firmware deployed
