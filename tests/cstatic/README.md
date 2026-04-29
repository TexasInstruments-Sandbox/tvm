# CStatic Backend Tests

Validation suite for the TVM C Static backend (`c_static` target).
Compiles models for both LLVM (reference) and c_static, then compares
outputs within tolerance (rtol=1e-3, atol=1e-5).

## Quick Start

```bash
# From repo root
export TVM_HOME=$(pwd)
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Run quick tests in parallel
cd tests/cstatic
pytest --rootdir=. unit-tests/ -m "not slow" -n auto -v

# Run all tests (including slow: ViT-B/16, segmentation)
pytest --rootdir=. unit-tests/ -v

# Debug a failed test (preserve temp workspace)
CSTATIC_KEEP_TEMP=1 pytest --rootdir=. unit-tests/test_resnet.py -v
```

## Prerequisites

1. **TVM build (2-pass)**:
   ```bash
   mkdir -p build && cp cmake/config.cmake build/
   cd build
   cmake -G Ninja .. && ninja           # Pass 1: shared libs (for Python)
   cmake -DBUILD_STATIC_RUNTIME=ON ..
   ninja tvm_runtime                    # Pass 2: libtvm_runtime.a (for c_static)
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

## Unit Tests

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

### Running by category

```bash
pytest --rootdir=. unit-tests/ -m "not slow"   # Quick only (~30s)
pytest --rootdir=. unit-tests/ -m slow          # Slow only (~5min)
pytest --rootdir=. unit-tests/ -n auto          # All, parallel
```

## Standalone Model Scripts

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

## How Tests Work

Each test:
1. Creates or loads a model (PyTorch or TVM IR)
2. Compiles for **LLVM** (reference) and **c_static** (target under test)
3. For c_static: exports to C, builds with CMake in an isolated temp dir,
   runs the binary, loads outputs from NPZ
4. Asserts numerical match between LLVM and c_static outputs

The C++ build template is in `cpp/` (CMakeLists.txt + main.cpp).
Each test gets its own `/tmp/cpp_cstatic_XXXXX/` workspace for safe
parallel execution.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `TVM_HOME` | Used by `cpp/CMakeLists.txt` to find TVM headers and libs |
| `CSTATIC_KEEP_TEMP` | Set to `1` to preserve temp workspaces for debugging |

## CI

The Jenkinsfile in this directory runs the full suite:
- 2-pass TVM build (shared + static runtime)
- cnpy build
- Quick tests in parallel (`-n auto`)
- Slow tests (ViT, segmentation) unless `SKIP_SLOW_TESTS` is set
