# TIDL Offloading Tests

Tests for TIDL subgraph offloading in the TVM/Relax c_static backend.

## Tests

| Test | Count | What it tests | Requirements |
|------|-------|---------------|--------------|
| `test_tidl_partition.py` | 10 | Pattern matching, partitioning, constraints | TVM only |
| `test_tidl_codegen.py` | 12 | Lowering pass, TIR stubs, c_static codegen, bridge generation (single + multi-subgraph, stub + real) | TVM only |
| `test_tidl_relax_import.py` | 16 | FFI load, init, AllowNode, tidl_import() pipeline | `tidl_model_import_relax.so` + c7x-mma-tidl tree |
| `test_tidl_layer_offload.py` | 94 | Per-layer offload validation: pattern matching (Level 1, 55 tests) and hardware inference (Level 4, 39 tests) covering all supported layer types | Level 1: TVM only; Level 4: `.so` + `TI_CGT_C7000_PATH` + AM67A |
| `test_tidl_new_ops.py` | 4 | Newer composite ops (softmax, multiply, permute_dims, concat) through full build + AM67A pipeline | `.so` + `TI_CGT_C7000_PATH` + AM67A |
| `test_tidl_e2e.py` | 1 | Full pipeline with stub bridge on c7x_host (no TIDL libs needed) | `TI_CGT_C7000_PATH` |
| `test_tidl_import_e2e.py` | 2 | `compiler.build()` one-call pipeline -> deploy -> run on AM67A (single-subgraph ConvReluSoftmax + multi-subgraph 2-conv model) | `.so` + `TI_CGT_C7000_PATH` + AM67A |
| `test_tidl_resnet_e2e.py` | 3 | ResNet-18 TIDL build pipeline validation, hardware correctness, cycle comparison | `.so` + `TI_CGT_C7000_PATH` (build test); + AM67A (hardware tests) |
| `diag_tidl_levels.py` | -- | Standalone multi-level TIDL init debug script | `TI_CGT_C7000_PATH` + artifacts + AM67A |

```bash
# Run partition + codegen tests (no hardware needed):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_codegen.py -v

# Per-layer partition tests — no .so or hardware (42 tests, ~0.3s):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_layer_offload.py \
       -k TestLayerPartition -v

# Import tests (needs tidl_model_import_relax.so):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_relax_import.py -v

# Per-layer hardware tests on AM67A (~10min, all layer types):
TI_CGT_C7000_PATH=~/ti/.../ti-cgt-c7000_5.0.1.LTS \
  pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_layer_offload.py \
         -k TestLayerHardware -v

# Hardware e2e tests (needs .so + C7x compiler + AM67A):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_import_e2e.py -v -s
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_resnet_e2e.py -v -s

# ResNet-18 build-only test (no AM67A needed, validates full pipeline):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_resnet_e2e.py::TestTIDLResNetE2E::test_tidl_resnet18_build -v

# Generate TIDL offloading visualization:
python tests/ti-dsp-runtime/tidl-tests/test_tidl_resnet_e2e.py \
    --visualize resnet18_tidl.html

# Standalone TIDL diagnostic (levels 0/1/1b/2/3):
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 3
```

## Per-layer Tests (`test_tidl_layer_offload.py`)

Isolated validation for each supported layer type, organized into two
test classes.  For constraint details (axis restrictions, calibration
behavior for math ops, etc.) see
`python/tvm/relax/backend/tidl/README.md`.

### Level 1 — Partition (`TestLayerPartition`, 55 tests)

Pure Python pattern-matching tests.  No `.so`, no hardware, no TI
compiler required.  Runs in under a second.

Each test builds a minimal Relax IRModule with the target op, calls
`partition_for_tidl()`, and asserts the expected composite name appears.

Includes **constraint rejection** tests:
- `test_reduce_multi_axis_rejected` — multi-axis reduction rejected
- `test_divide_rejects_rank2` — sub-4D element-wise ops rejected

### Level 4 — Hardware (`TestLayerHardware`, 39 tests)

End-to-end tests running the full TIDL pipeline on AM67A:
`partition → tidl_import → lower → codegen → bridge → build → run_dsp_dload`

Requirements: `tidl_model_import_relax.so`, `TI_CGT_C7000_PATH`, AM67A.

| Test group | Tests |
|---|---|
| Activations | sigmoid, tanh, clip, leakyrelu, elu, hard_sigmoid, hard_swish, mish |
| Element-wise | subtract, maximum, minimum |
| Reductions | sum, reduce_max, argmax, argmin |
| Advanced | resize2d, strided_slice, permute_dims, concat, topk, split, depth_to_space |
| Math/unary | abs, sqrt, exp, log, erf, floor, negative, sin, cos, tan, sinh, cosh, asin, acos, atan, asinh, power |

All hardware tests assert `np.isfinite(output).all()`.  Math/unary ops
with unbounded output range (`exp`, `sinh`, `cosh`, `power`) may produce
all-zero output with random calibration data — this validates the
import/codegen pipeline without depending on calibration quality.

## TIDL Artifacts

TIDL subgraphs need compiled binary artifacts (`net.bin` +
`params_1.bin`).  Artifacts are generated at runtime by
`tidl_import()` (the Relax FFI import path) using
`tidl_model_import_relax.so`.

### Artifact naming

TIDL's default naming convention (used by both the standalone import
tool and the Relax FFI):

| File | Description |
|------|-------------|
| `subgraph{id}_net.bin` | Network topology + weights (~1.3 MB with MMA code) |
| `subgraph{id}_params_1.bin` | I/O buffer descriptors (~378 KB) |

### Version compatibility

The import `.so`, calibration tool, network compiler, and firmware
algo libs must all be built from the same `c7x-mma-tidl` source tree
with `TARGET_SOC=J722S` to ensure struct layout and version
consistency.

### Device config (J722S/AM67A)

The device config is at
`c7x-mma-tidl/ti_dl/test/testvecs/config/import/device_config.cfg`
(symlinked to `device_configs/j722s_config.cfg`):

```
L2MEMSIZE_KB    = 224
MSMCSIZE_KB     = 240
DEVICE_NAME     = 4
```

`MSMCSIZE_KB` controls the L3 scratch buffer size baked into the
network binary.  J722S has no MSMC SRAM; we map 240 KB of auxiliary
L2 as the L3 pool.  Setting this to 2048 (the default) causes TIDL
to request 2 MB of SRAM that does not exist.

### Known issues

- **IOBufDesc header mismatch**: The c7x-mma-tidl source header
  (`itidl_io.h`) uses `TIDL_IO_MAX_NUM_CORES=4` for the IOBufDesc
  struct layout.  The PSDK header uses SOC-dependent
  `TIDL_MAX_NUM_CORES` (2 for J722S).  The DLOAD module
  (`CMakeLists.txt`) must include the c7x-mma-tidl source header,
  not the PSDK header, to match the import tool's artifact layout.

- **CWD sensitivity**: The import `.so` uses hardcoded relative paths
  (`../../test/`, `../../utils/perfsim/`).  The `tidl_import()` FFI
  handles this by setting the correct CWD.

- **72 MB artifacts**: If `ti_cnnperfsim.out` is not built, artifacts
  are ~72 MB (generic code) instead of ~1.3 MB (MMA optimized).
  Build with `make nc` (see "Building TIDL Dependencies" below).

- **device_config.cfg symlink**: Must exist at
  `ti_dl/test/testvecs/config/import/device_config.cfg` pointing to
  `device_configs/j722s_config.cfg`.  Create with:
  ```bash
  cd ~/ml/c7x-mma-tidl/ti_dl/test/testvecs/config/import
  ln -sf device_configs/j722s_config.cfg device_config.cfg
  ```

## Diagnostic Script

`diag_tidl_levels.py` is a standalone script for debugging TIDL init
on AM67A hardware.  It provides multiple diagnostic levels:

| Level | What it does |
|-------|-------------|
| `0` | Stub bridge only -- no TIDL calls, verifies module load/run |
| `1` | Calls `algNumAlloc` + dumps IALG function table addresses |
| `1b` | Calls `algAlloc` directly with TIDL headers and NULL callbacks |
| `2` | Full `init_tidl_subgraph` via `tidl_api.c` |
| `3` | Full `process_tidl_subgraph` (real int8 inference on MMA) |

```bash
export TI_CGT_C7000_PATH=~/ti/.../ti-cgt-c7000_5.0.1.LTS
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Start with level 0 and work up
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 0
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 1
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 1b
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 2
```

If the DSP hangs, read the remoteproc trace buffer from A53:
```bash
ssh root@am67a cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
```

## Building TIDL Dependencies from Source

For full version consistency, all TIDL tools and libraries should be
built from the same `c7x-mma-tidl` source tree.  This section
documents the complete build process.

### Prerequisites

| Component | Location | Notes |
|---|---|---|
| PSDK RTOS | `~/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06` | J722S PSDK 11_00 |
| c7x-mma-tidl | `~/ml/c7x-mma-tidl` | TIDL source (Bitbucket) |
| neo-tvm | `~/ml/neo-tvm` | TI's TVM fork (for Relay headers) |
| MMALIB | `~/ml/am67a/mmalib_11_02_00_06` | Prebuilt MMALIB 11.02 with C7524 libs |
| OpenCV 4.1.0 | `/cgnas/tvm/deps/opencv-4.1.0` | Prebuilt static libs |
| TI C7x compiler | `~/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS` | CGT C7000 5.0.1 |

### One-time setup

```bash
PSDK=~/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06

# 1. mcu_plus_sdk symlink (build system expects this name)
ln -sf mcu_plus_sdk_j722s_11_00_00_12 $PSDK/mcu_plus_sdk

# 2. MMALIB symlink (c7x-mma-tidl expects version 11_02_00_06)
ln -sf ~/ml/am67a/mmalib_11_02_00_06 $PSDK/mmalib_11_02_00_06

# 3. OpenCV symlink
ln -sf /cgnas/tvm/deps/opencv-4.1.0 $PSDK/opencv-4.1.0

# 4. TI compiler symlink (PSDK_TOOLS_PATH convention)
ln -sf ~/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS \
       ~/ti/ti-cgt-c7000_5.0.1.LTS

# 5. flatbuffers 1.12.0 (download + build)
cd $PSDK
wget -q https://github.com/google/flatbuffers/archive/v1.12.0.zip
unzip -q v1.12.0.zip && rm v1.12.0.zip
cd flatbuffers-1.12.0
cmake -G "Unix Makefiles" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DFLATBUFFERS_BUILD_TESTS=OFF . > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1

# 6. TFLite C headers (for custom layer import)
TF_INC=$PSDK/targetfs/usr/include/tensorflow
mkdir -p $TF_INC/tensorflow/lite/core/c
for h in common.h builtin_op_data.h c_api_types.h; do
    wget -q "https://raw.githubusercontent.com/tensorflow/tensorflow/\
v2.12.0/tensorflow/lite/core/c/$h" -O $TF_INC/tensorflow/lite/core/c/$h
done

# 7. neo-tvm submodules + build
cd ~/ml/neo-tvm
git submodule update --init 3rdparty/dmlc-core 3rdparty/dlpack \
                            3rdparty/rang 3rdparty/libbacktrace
mkdir -p build && cp cmake/config.cmake build/
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' build/config.cmake
cd build && cmake -G Ninja .. > /dev/null 2>&1 && ninja

# 8. neo-tvm wrapper dir (build system expects TVM_HOME/tvm/include)
mkdir -p ~/ml/neo-tvm-wrapper
ln -sf ~/ml/neo-tvm ~/ml/neo-tvm-wrapper/tvm

# 9. c7x-mma-tidl config: update compiler version
sed -i 's/CGT_C7X_VERSION := 5.0.0.LTS/CGT_C7X_VERSION := 5.0.1.LTS/' \
    ~/ml/c7x-mma-tidl/makerules/config.mk

# 10. dmautils host emulation lib (from MCU+ SDK source)
cd $PSDK/mcu_plus_sdk/source/drivers/dmautils
CGT7X_ROOT=~/ti/ccs2040/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
CGT_TI_C7X_HOSTEMU_PATH=/usr \
CGT_TI_C7000_PATH=$CGT7X_ROOT \
make -f makefile.j722s.c75ssx-0.ti-c7x-hostemu PROFILE=release \
  CC="/usr/bin/g++-13 -c" AR="/usr/bin/gcc-ar-13" \
  INCLUDES_common="-I$CGT7X_ROOT/host_emulation/include/C7524-MMA2_256 \
                    -I$PSDK/mcu_plus_sdk/source"
```

### Build c7x-mma-tidl

Use the build script for a one-command build of all PC tools:

```bash
cd ~/ml/c7x-mma-tidl
bash build_j722s.sh          # incremental build
bash build_j722s.sh clean    # clean + full rebuild
```

Or build manually with these common environment variables:

```bash
export PSDK_INSTALL_PATH=~/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06
export TVM_HOME=~/ml/neo-tvm-wrapper
export TF_REPO_PATH=$PSDK_INSTALL_PATH/targetfs/usr/include/tensorflow
export ENABLE_SDK_11_0_COMPATIBILITY=1
cd ~/ml/c7x-mma-tidl
```

#### TIDL algo libs (C7x target -- linked into firmware)

```bash
make tidl_lib TARGET_PLATFORM=TI_DEVICE TARGET_SOC=J722S \
  TARGET_CPU=C71 TARGET_BUILD=release \
  RTOS_SDK=mcu_plus_sdk RTOS=FREERTOS -j$(nproc)
# Output: ti_dl/lib/J722S/dsp/algo/release/
#   tidl_algo.lib, tidl_obj_algo.lib, tidl_priv_algo.lib, tidl_custom.lib
```

#### TIDL algo libs (PC emulation -- used by calibration tool)

```bash
make tidl_lib TARGET_PLATFORM=PC TARGET_SOC=J722S \
  TARGET_BUILD=release RTOS_SDK=mcu_plus_sdk RTOS=FREERTOS -j$(nproc)
# Output: ti_dl/lib/J722S/PC/algo/release/
#   libtidl_algo.a, libtidl_obj_algo.a, etc.
```

#### Calibration stats tool (PC_dsp_test_dl_algo.out)

```bash
make -C ./ti_dl/test -f makefile final_install \
  TARGET_PLATFORM=PC TARGET_SOC=J722S TARGET_BUILD=release \
  RTOS_SDK=mcu_plus_sdk RTOS=FREERTOS BUILD_WITH_OPENCV=0 -j$(nproc)
# Output: ti_dl/test/PC_dsp_test_dl_algo.out
```

#### Network compiler (ti_cnnperfsim.out)

```bash
make nc TARGET_PLATFORM=PC TARGET_SOC=J722S TARGET_BUILD=release \
  RTOS_SDK=mcu_plus_sdk RTOS=FREERTOS BUILD_WITH_OPENCV=0 -j$(nproc)
# Output: ti_dl/utils/perfsim/ti_cnnperfsim.out
```

#### Import tool and .so libraries

**Important**: Pass `TARGET_SOC=J722S` to all import tool builds.
This ensures the `sTIDL_IOBufDesc_t` struct layout matches the
firmware and runtime.

```bash
cd ti_dl/utils/tidlModelImport
make -f makefile_lib TARGET_SOC=J722S -j$(nproc)                    # protobuf stubs
make -f makefile_shared_custom TARGET_SOC=J722S -j$(nproc)          # custom layer .so
make -f makefile_shared TARGET_SOC=J722S -j$(nproc)                 # tidl_model_import.so
make -f makefile_shared relax TARGET_SOC=J722S -j$(nproc)           # tidl_model_import_relax.so
make -f makefile_bin TARGET_SOC=J722S -j$(nproc)                    # tidl_model_import.out
# Output: out/tidl_model_import.out, out/*.so
```

The `relax` target compiles against our TVM 0.23 fork (`RELAX_TVM_HOME`,
defaults to `~/ml/tvm`).  It does NOT compile `tidl_relayImport.cpp`
or `tidlParseTVM/*.cpp` -- the Relax .so is fully self-contained with
no Relay/neo-tvm dependencies.

### Wire up device_config.cfg

The import tool resolves `device_config.cfg` via a hardcoded relative
path from CWD.  Create the symlink if not already present:

```bash
cd ~/ml/c7x-mma-tidl/ti_dl/test/testvecs/config/import
ln -sf device_configs/j722s_config.cfg device_config.cfg
```

Verify `MSMCSIZE_KB = 240` in `j722s_config.cfg` (see "Device config"
above).

### Wire up firmware

The firmware CMakeLists (`src/runtime/ti_dsp/firmware/c7x/dsp/CMakeLists.txt`)
hardcodes `PSDK_PATH` and resolves TIDL algo libs from
`$PSDK_PATH/c7x-mma-tidl/ti_dl/lib/J722S/dsp/algo/release/`.
MMALIB is found via `file(GLOB)` on `$PSDK_PATH/mmalib*/lib/C7524/release/`.

Replace the PSDK's prebuilt libs with a symlink to the source-built ones:

```bash
PSDK=~/ml/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06
mv $PSDK/c7x-mma-tidl/ti_dl/lib/J722S/dsp/algo/release \
   $PSDK/c7x-mma-tidl/ti_dl/lib/J722S/dsp/algo/release.psdk_prebuilt
ln -sf ~/ml/c7x-mma-tidl/ti_dl/lib/J722S/dsp/algo/release \
       $PSDK/c7x-mma-tidl/ti_dl/lib/J722S/dsp/algo/release
```

Then rebuild and deploy the firmware:

```bash
cd $TVM_HOME/src/runtime/ti_dsp/firmware/c7x/dsp/build
cmake --build .
cd ../.. && ./deploy-c7x.sh dsp/build/c7x_compute.out
```

### Verify

```bash
# Run partition + codegen + import tests
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_codegen.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_relax_import.py -v

# Run hardware e2e test (needs AM67A with c7x_compute firmware)
TI_CGT_C7000_PATH=~/ti/.../ti-cgt-c7000_5.0.1.LTS \
  pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_import_e2e.py -v -s
```
