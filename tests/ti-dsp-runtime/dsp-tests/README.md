# DSP Tests

Pytest-based tests for running TVM-compiled models on TI DSP targets.
Tests compile models with the `c_static` backend, build for the selected
execution mode, run inference, and compare results against a PyTorch
reference.

## Directory Structure

```
dsp-tests/
├── conftest.py                          # Pytest fixtures for DSP configuration
├── model_utils.py                       # Shared model creation utilities
├── test_c66x_pragmas_dsp.py             # C66x/C7x pragma generation tests
├── test_classification_dsp.py           # TorchVision classification models
├── test_clista_dsp.py                   # CLISTA-DoA radar model
├── test_conv2d_dsp.py                   # Conv2D model
├── test_conv2d_stack_dsp.py             # Conv2D + BN + ReLU stack (4 layers)
├── test_error_messages_dsp.py           # Compilation and error handling tests
├── test_lenet_dsp.py                    # LeNet-5 MNIST classifier
├── test_matmul_dsp.py                   # Matrix multiplication
├── test_mlp_dsp.py                      # Multi-layer perceptron
├── test_od_torchvision_dsp.py           # SSDLite320 object detection
├── test_quantized_conv2d_stack_dsp.py   # INT8 quantized conv2d stack
├── test_quantized_resnet_dsp.py         # INT8 quantized ResNet-18
├── test_resnet_dsp.py                   # ResNet-18 image classifier
├── test_rtmdet_dsp.py                   # Multi-output tuple handling
├── test_segmentation_dsp.py             # TorchVision segmentation models
├── test_yolo_dsp.py                     # YOLOv5/YOLOv8 object detection
└── README.md                            # This file
```

## Test Descriptions

| Test File | Description | Model Type |
|-----------|-------------|------------|
| `test_c66x_pragmas_dsp.py` | C66x/C7x pragma generation and TI compiler directives | TIR codegen |
| `test_classification_dsp.py` | 8 ImageNet classifiers (SqueezeNet to ResNet-34) | Conv2D, various |
| `test_clista_dsp.py` | CLISTA-DoA radar signal processing | Conv1D, Linear |
| `test_conv2d_dsp.py` | Single 2D convolution | Conv2D |
| `test_conv2d_stack_dsp.py` | 4-layer conv2d + batch_norm + relu stack | Conv2D, BN, ReLU |
| `test_error_messages_dsp.py` | Compilation validation and error handling | TIR simple ops |
| `test_lenet_dsp.py` | LeNet-5 MNIST classifier | Conv2D, Linear |
| `test_matmul_dsp.py` | Matrix multiplication | Matmul |
| `test_mlp_dsp.py` | Multi-layer perceptron | Linear, ReLU |
| `test_od_torchvision_dsp.py` | SSDLite320 MobileNetV3 object detection | Conv2D, multi-output |
| `test_quantized_conv2d_stack_dsp.py` | INT8 quantized conv2d stack (PT2E QDQ) | Conv2D, quantized |
| `test_quantized_resnet_dsp.py` | INT8 quantized ResNet-18 (PT2E QDQ) | Conv2D, quantized |
| `test_resnet_dsp.py` | ResNet-18 image classifier | Conv2D, BN, skip |
| `test_rtmdet_dsp.py` | Multi-output tuple handling validation | Conv2D (2 outputs) |
| `test_segmentation_dsp.py` | LRASPP and DeepLabV3 MobileNetV3 segmentation | Conv2D, multi-output |
| `test_yolo_dsp.py` | YOLOv5 (n, s) and YOLOv8 (n, s) detection | Conv2D, various |

## Execution Modes

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

## Running Tests

### Via pytest

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

### Test depth tiers

Three markers control which tests run at each pipeline stage:

| Marker | Tests | When to use |
|--------|-------|-------------|
| `quick` | 6 small models | PR gate — fast compile + run |
| `core` | ~20 tests | Post-merge gate — all ops, classification, detection |
| *(none)* | ~50 tests | Nightly full regression |

`core` is a superset of `quick`. All `quick` tests are also `core`.

#### `quick` tests (6 tests, both c66x and c7x)

| Test | Model |
|------|-------|
| `test_conv2d_dsp` | Single Conv2D |
| `test_conv2d_stack_dsp` | 4-layer Conv2D + BN + ReLU |
| `test_clista_dsp` | CLISTA-DoA radar |
| `test_matmul_dsp` | Matrix multiplication |
| `test_mlp_dsp` | Multi-layer perceptron |
| `test_quantized_conv2d_stack_dsp` | INT8 quantized Conv2D stack |

#### `core` tests added beyond `quick` (~14 additional)

| Test | Architecture |
|------|-------------|
| `test_c66x_pragmas_dsp` (all) | both |
| `test_error_messages_dsp` (all) | both |
| `test_lenet_dsp` | both |
| `test_resnet_dsp` | both |
| `test_classification_dsp` (8 models) | both |
| `test_quantized_resnet_dsp` | c7x only |

#### `c7x_only` tests (excluded from c66x stages)

Tests marked `c7x_only` use models too large for C66x memory or exercise
c7x-specific features. Jenkins c66x stages filter with `-m "not c7x_only"`:

| File | Reason |
|------|--------|
| `test_c7x_vm_dsp.py` | C7xVirtualMachine API |
| `test_quantized_resnet_dsp.py` | INT8 ResNet-18 (~47 MB) |
| `test_od_torchvision_dsp.py` | SSDLite320 (c7x_dload only) |
| `test_rtmdet_dsp.py` | RTMDet (c7x_dload only) |
| `test_segmentation_dsp.py` | LRASPP / DeepLabV3 |
| `test_yolo_dsp.py` | YOLOv5/v8 + TIDL |
| `test_conv2d_cycle_breakdown.py` | Cycle profiling benchmark |

### Jenkins pipeline commands

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

### Via standalone script

Each test file can also be run directly:

```bash
python test_conv2d_dsp.py --dsp-mode c66x_host
python test_conv2d_dsp.py --dsp-mode c7x_dload -v
python test_resnet_dsp.py --dsp-mode c7x_dload --profile-layers

# Save build artifacts for inspection
python test_clista_dsp.py --dsp-mode c66x_host --save-artifacts /tmp/artifacts
```

### Command-line options

| Option | Description |
|--------|-------------|
| `--dsp-mode=MODE` | Execution mode (required): `c66x_host`, `c66x`, `c7x_host`, `c7x_dload` |
| `--dsp-timeout=N` | Hardware execution timeout in ms (default: 60000) |
| `--dsp-verbose` | Enable verbose DSP logging |
| `--save-artifacts=DIR` | Copy build artifacts (lib0.c, weights.bin, devc.c) to DIR |
| `--profile-layers` | Enable per-layer cycle profiling (hardware modes) |
| `--use-cpp-api` | Enable direct VM builtin calls (bypass FFI dispatch) |

## Key Components

### `conftest.py`
Pytest fixtures and configuration:
- `dsp_mode`: Execution mode from `--dsp-mode` option (required)
- `dsp_timeout`: Timeout from `--dsp-timeout` option
- `dsp_verbose`: Verbose flag from `--dsp-verbose` option
- `save_artifacts`: Artifact directory from `--save-artifacts` option
- `profile_layers`: Profiling flag from `--profile-layers` option
- `use_cpp_api`: C++ API flag from `--use-cpp-api` option
- `dsp_config`: Combined configuration dictionary

### `model_utils.py`
Shared model creation functions:
- `torch_to_relax_with_params()`: Convert PyTorch model to TVM with bound parameters
- `create_conv2d_model()`: Single Conv2D layer
- `create_conv2d_stack_model()`: 4-layer conv2d + BN + ReLU stack
- `create_mlp_model()`: Multi-layer perceptron
- `create_matmul_model()`: Matrix multiplication
- `create_clista_model()`: CLISTA-DoA radar model
- `create_lenet_model()`: LeNet-5 CNN
- `create_quantized_conv2d_stack_model()`: INT8 quantized conv2d stack

### `dsp_utils.py` (in `../dsp-cpp/`)
DSP compilation and execution utilities:
- `get_target_string()`: Map mode to c_static target string
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

## Adding New Tests

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

## Multi-Output Support

The DSP runtime supports models that return multiple outputs (tuples).
`test_rtmdet_dsp.py` validates this:

- Maximum 8 outputs supported (compile-time check)
- `Model::InferMulti()` API returns all outputs
- Output tensors written to `output.bin` in order

## Requirements

- TVM with c_static backend
- PyTorch and torchvision for model creation and reference inference
- **c66x_host**: No additional requirements (system gcc)
- **c66x**: AWRL6844 board with XDS110 debug probe, TI C6000 compiler
  (`TI_CGT_C6000_PATH`)
- **c7x_host**: TI C7000 CGT with Host Emulation library
  (`TI_CGT_C7000_PATH`)
- **c7x_dload**: AM67A (J722S) board at hostname `am67a`, TI C7000
  compiler (`TI_CGT_C7000_PATH`), c7x_compute firmware deployed
