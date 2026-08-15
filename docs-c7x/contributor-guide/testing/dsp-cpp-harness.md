# DSP C++ Harness

Infrastructure for running TVM-compiled models on TI DSP hardware (C66x
and C7x) using the minimal DSP runtime. Located at
`tests/ti-dsp-runtime/dsp-cpp/`.

## Overview

The integration demonstrates:
- Loading TVM-generated code (`lib0.c`) with the DSP runtime
- Parsing model weights from `weights.bin`
- Running inference on PC (host emulation), C66x, and C7x hardware
- Using the RAII-based Model API (`model.h`) to call the generated
  `cg_main_dsp` entry point directly, instead of using the full TVM VM

## Supported Platforms

| Platform | Device | L2 SRAM | Main Memory | Weights | Status |
|----------|--------|---------|-------------|---------|--------|
| Host | PC | 4MB (emulated) | 64MB (emulated) | N/A | ✅ Tested |
| C66x | AWRL6844 | 64KB | 1MB L3 | Embedded | ✅ Tested |
| C7x | J722S | 1.59MB | 128MB DDR | 55.7MB DDR | ✅ Tested |

## Quick Start (Python)

For most users, the `dsp_utils.py` module provides a simple Python interface:

```python
import sys
sys.path.insert(0, "path/to/dsp-cpp")

from dsp_utils import compile_and_run_dsp, compare_results

# Compile and run on DSP (mod is a TVM IRModule with parameters bound)
results = compile_and_run_dsp(
    mod=mod,
    input_data=input_data,  # numpy array or tuple of arrays
    target_string="c_static -mcpu=c66x",
    execution_mode="both",  # "host", "c66x", or "both"
    build_type="Release",   # "Release" or "Debug"
)
```

The workflow uses file-based I/O:
1. Python writes input tensors to `input.bin`
2. DSP executable reads `input.bin`, runs inference, writes `output.bin`
3. Python reads `output.bin` and compares against reference

See `../dsp-tests/test_clista_dsp.py` for a complete example.

## Files

| File | Description |
|------|--------------|
| `main_dsp.cpp` | Main entry point - reads input.bin, runs inference, writes output.bin |
| `CMakeLists.txt` | Build configuration for host, C66x, and C7x targets |
| `dsp_utils.py` | Python utilities for DSP compilation and execution |
| `io/tensor_file.cpp`, `io/tensor_file.h` | Tensor file I/O (`input.bin`/`output.bin`) |
| `io/weights_loader.cpp`, `io/weights_loader.h` | Weights source (filesystem or linker-embedded) |
| `io/tensor_file_format.md` | Binary tensor file format specification |

Note: Constants loading (`TVMDSPParseConstants`, `TVMGetConstants`) is now provided
by the TVM DSP runtime library (`constants/constants_loader.cpp`).

## Prerequisites

### For Host Emulation
- CMake 3.16+
- C++11 compiler (GCC, Clang, or AppleClang)
- TVM DSP runtime library (`libtvm_dsp_runtime_host.a`)

### For C66x Hardware
- TI C6000 Compiler v8.5.0+ (part of CCS)
- MMWAVE-L-SDK-6 v6.1.0.05
- AWRL6844 evaluation board with XDS110 debug probe
- TVM DSP runtime library (`libtvm_dsp_runtime_c66x.a`)

### For C7x Hardware
- TI C7000 Compiler v5.0.0+ (part of CCS)
- J722S evaluation board with XDS110 debug probe
- Code Composer Studio 12.0+
- TVM DSP runtime library (`libtvm_dsp_runtime_c7x.a`)

## Building

### Step 1: Build the DSP Runtime

```bash
# Host emulation runtime
cd $TVM_HOME/src/runtime/ti_dsp
mkdir -p build && cd build
cmake ..
cmake --build .

# C66x runtime (optional, for hardware deployment)
cd $TVM_HOME/src/runtime/ti_dsp
mkdir -p build-c66x && cd build-c66x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-awrl6844.cmake ..
cmake --build .

# C7x runtime (optional, for J722S hardware deployment)
cd $TVM_HOME/src/runtime/ti_dsp
mkdir -p build-c7x && cd build-c7x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake ..
cmake --build .
```

### Step 2: Generate TVM Model Code

Use TVM's C static backend to compile your model:

```python
import tvm
from tvm import relax

# Load/compile your model to get an IRModule
mod = ...

# Build with c_static target for C66x
target = tvm.target.Target("c_static -mcpu=c66x")
with tvm.transform.PassContext(opt_level=3):
    ex = relax.build(mod, target, exec_mode="compiled", system_lib=True)

# Export generated code to a directory
ex.export_library("model_dir/model_library.tar", target=target)
# Extract to get lib0.c and weights.bin
```

Or use `dsp_utils.py` which handles this automatically:

```python
from dsp_utils import compile_for_dsp
generated_dir = compile_for_dsp(mod, "c_static -mcpu=c66x")
```

### Step 3: Build Host Emulation

```bash
cd tests/ti-dsp-runtime/dsp-cpp

# Debug build (default)
mkdir -p build-debug && cd build-debug
cmake -DGENERATED_CODE_DIR=/path/to/model_dir ..
cmake --build .

# Release build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGENERATED_CODE_DIR=/path/to/model_dir ..
cmake --build .
```

### Step 4: Build for C66x Hardware

```bash
cd tests/ti-dsp-runtime/dsp-cpp

# Debug build (default)
mkdir -p build-awrl6844-debug && cd build-awrl6844-debug
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-awrl6844.cmake \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .

# Release build
mkdir -p build-awrl6844 && cd build-awrl6844
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-awrl6844.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .
```

### Step 5: Build for C7x Hardware (J722S)

```bash
cd tests/ti-dsp-runtime/dsp-cpp

# Debug build (default)
mkdir -p build-j722s-debug && cd build-j722s-debug
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-j722s-c7x.cmake \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .

# Release build
mkdir -p build-j722s && cd build-j722s
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-j722s-c7x.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGENERATED_CODE_DIR=/path/to/model_dir \
  ..
cmake --build .
```

## Running Inference

The DSP executable uses file-based I/O:
- **Input**: Reads tensor(s) from `input.bin` (must exist in working directory)
- **Output**: Writes tensor(s) to `output.bin`

The binary tensor file format is self-describing with magic number validation.
See `io/tensor_file_format.md` for the full specification.

### Host Emulation

```bash
cd build
# First, create input.bin (typically done by Python)
./cg_dsp
```

Expected output:
```
INFO: Loaded weights from /path/to/model_dir/weights.bin (123456 bytes)
TVM DSP Runtime initialized on Host (PC Emulation)
  Fast pool: 4096 KB
  Main pool: 65536 KB
Loaded 185 constants
Input[0]: shape=[1,2,16], dtype=2.32
Cycles: 42000
Num outputs: 1
Output[0]: shape=[1,128,1], dtype=2.32
Memory: L2 peak=12345, L3 peak=0
Done
```

### C66x Hardware

```bash
# Using the deployment script
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh build-awrl6844/cg_dsp_c66x.out

# Or with extended timeout
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c66x.sh build-awrl6844/cg_dsp_c66x.out --timeout 120000
```

Expected output (similar to host, with C66x memory configuration):
```
INFO: Using embedded weights (123456 bytes)
TVM DSP Runtime initialized on C66x (AWRL6844)
  L2 pool: 0x00850000 - 0x00860000 (64 KB)
  L3 pool: 0x88050400 - 0x88150400 (1024 KB)
...
```

### C7x Hardware (J722S)

```bash
# Using the deployment script
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c75x.sh build-j722s/cg_dsp_c7x.out

# Or with extended timeout
$TVM_HOME/src/runtime/ti_dsp/scripts/run_on_c75x.sh build-j722s/cg_dsp_c7x.out --timeout 120000
```

Expected output:
```
INFO: Using embedded weights (340 bytes)
C7x security mode: CXM=3 (RootSupervisor)
TVM DSP Runtime initialized on C7x (J722S_C75)
  L2 pool: 0x7e069000 - 0x7e200000 (1628 KB)
  DDR pool: 0x108000000 - 0x110000000 (128 MB)
...
```

## Configuration Options

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type: `Debug` or `Release` |
| `TVM_DSP_TARGET` | `host` | Target: `host`, `c66x`, `c7x_host`, or `c7x-dynmod` |
| `TVM_DSP_DEVICE` | (none) | Device variant: `awrl6844` or `j722s` |
| `GENERATED_CODE_DIR` | `../cstatic-tests` | Directory containing lib0.c and weights.bin |
| `TVM_HOME` | Auto-detect | Path to TVM repository |
| `WEIGHTS_FILE` | `${GENERATED_CODE_DIR}/weights.bin` | Path to weights file |

### Build Types

| Build Type | Host Flags | C66x Flags |
|------------|------------|------------|
| Debug | `-g -O0` | `-g --opt_level=0` |
| Release | `-O3 -DNDEBUG` | `-O3 --opt_for_speed=5` |

### Model Configuration

Set via CMake, compiled in as preprocessor definitions consumed by the
Model API:

```bash
cmake -DMODEL_ENTRY_FUNCTION=main -DMODEL_NUM_INPUTS=1 \
      -DMODEL_RETURNS_TUPLE=ON -DGENERATED_CODE_DIR=/path/to/model_dir ..
```

Input shape and data are provided via `input.bin` at runtime (no recompilation needed).

## Generated Code Notes

When using `-mcpu=c66x` target, the generated `lib0.c`:
- Includes DSP-specific headers directly (`ffi_types.h`, etc.)
- Uses `TVM_DSP_SKIP_CG_MAIN` to exclude exception-based wrapper
- Is compiled as C++ to support the DSP runtime API

The CMakeLists.txt handles this automatically:
```cmake
set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)
```

## Troubleshooting

### "Failed to load weights"
- Verify `weights.bin` exists at the configured path
- Check file permissions
- For C66x: ensure weights are properly embedded

### "TVMFFIFunctionCall with NULL function"
- Ensure `InitVMBuiltins()` is called before `__vmtir__main`
- Verify VM builtins are registered with `TVMDSPRegisterVMBuiltins()`

### Inference timeout on C66x
- Check memory usage in the `.map` file
- Verify L2/L3 pool sizes are sufficient
- Try increasing timeout: `--timeout 300000`

### Linker errors about LLVM paths
- The toolchain file should clear host linker flags
- If not, manually set: `-DCMAKE_EXE_LINKER_FLAGS=""`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (dsp_utils.py)                    │
│  - write_tensors_to_file() -> input.bin                     │
│  - Run DSP executable                                       │
│  - read_tensors_from_file() <- output.bin                   │
│  - Compare against PyTorch reference                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       main_dsp.cpp                          │
│  - GetWeightsData() - loads weights.bin                     │
│  - model.Load() - parses constants                          │
│  - Read input tensor(s) from input.bin                      │
│  - model.InferMulti() - calls generated cg_main_dsp()       │
│  - Write output tensor(s) to output.bin                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         lib0.c                              │
│  - Generated TIR code                                       │
│  - cg_main_dsp(inputs, num_inputs, outputs, num_outputs)    │
│  - Operator implementations (conv1d, add, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TVM DSP Runtime                          │
│  - TVMDSPParseConstants() / TVMGetConstants()               │
│  - vm.builtin.alloc_storage, alloc_tensor, reshape, copy    │
│  - Tensor file I/O (input.bin, output.bin)                  │
│  - Memory pools (L2/L3 SRAM)                                │
│  - NDArray management                                       │
│  - weights.bin parsing (zero-copy)                          │
└─────────────────────────────────────────────────────────────┘
```

See [C7x Memory Map Reference](../dsp-runtime/memory-map.md) for the
J722S memory layout and MMU configuration this harness relies on for
standalone JTAG execution.
