# Verifying Your Deployment

Two tools confirm a deployment works before you run a full model: the
firmware's own hardware test suite (`test_dynmod.sh`, checks firmware
boot and the DLOAD/IPC path directly) and `test_c7x_runtime` (checks the
`c7x::Module` C++ API on top of a working firmware).

## Firmware Hardware Test Suite

```bash
# From the firmware/c7x directory:
test/test_dynmod.sh --deploy

# Or with options:
test/test_dynmod.sh --target root@am67a --module /path/to/lib0.out
```

The test script covers 6 milestones:
1. Firmware boots via remoteproc
2. Basic IPC (ping, status)
3. Dynamic module load via DLOAD
4. Inference execution (cg_main_dsp with trace verification)
5. Module unload
6. Load-infer-unload stability cycle (5 iterations)

## test_c7x_runtime (C++ API)

`test_c7x_runtime` is a C++ test binary for the `c7x::Module` ARM inference
API. It exercises `libc7x_arm_runtime.so` end-to-end on an AM67A/BeagleY-AI
ARM board against a live C7x DSP firmware instance -- a good way to confirm
your deployment actually works before running a full model.

This is the C++ counterpart to the Python integration tests in
`tests/ti-dsp-runtime/dsp-tests/test_c7x_vm_dsp.py`. Both test the
same underlying `c7x_compute_client` IPC path; this binary tests the
C++ `c7x::Module` wrapper directly, without Python or TVM overhead.

### Test cases

| # | Name | What it checks |
|---|------|----------------|
| 1 | LOAD/CLOSE | `Module::Load()` connects to the DSP and loads the ELF; `Close()` is idempotent (safe to call twice) |
| 2 | INFERENCE | `Run()` returns a non-degenerate output tensor (`ndim > 0`, `data_size > 0`); prints shape and dtype |
| 3 | REFERENCE | If `--ref` is supplied: `max|out - ref| < atol`; skipped otherwise |
| 4 | CREATE_INPUT | `CreateInput()` pointer lies within `[StagingBuffer(), StagingBuffer()+size)` (pre-staged DDR); inference result matches standard-path result |
| 5 | REPEATED_INFER | Three consecutive `Run()` calls with identical input produce bit-identical output |

Test 1 is a prerequisite: if the module cannot be loaded, the remaining
tests are skipped and the binary exits immediately.

### Building

`test_c7x_runtime` is built automatically alongside `libc7x_arm_runtime.so`
when the `arm/` CMakeLists detects the source file:

```bash
cd src/runtime/ti_dsp/firmware/c7x/arm

# Cross-compile for ARM64 (default)
./build.sh --board j722s-evm

# Or natively on the AM67A board
./build.sh --board j722s-evm native
```

Outputs written to `arm/build/`:
```
libc7x_arm_runtime.so   — shared library (required at runtime)
c7x_compute             — CLI tool
test_c7x_runtime        — this test binary
```

### Deploying to AM67A

```bash
cd src/runtime/ti_dsp/firmware/c7x/arm
./build.sh --board j722s-evm deploy
```

This SCPs all three binaries plus `c7x_runtime.h` to the board and
runs `ldconfig` so the shared library is found by the dynamic linker:

```
am67a:/usr/local/bin/c7x_compute
am67a:/usr/local/bin/test_c7x_runtime
am67a:/usr/local/lib/libc7x_arm_runtime.so
am67a:/usr/local/include/c7x_runtime.h
```

The deploy hostname is a deterministic function of `--board`
(`beagley-ai` -> `beagley-ai`, else `am67a`) -- add an SSH-config alias if
your board is reachable under a different name.

### Preparing test inputs

The test binary takes a raw flat binary `input.bin` (contiguous, row-major).
Use Python/numpy on the development machine to generate it alongside a
CPU reference output `ref.bin`:

```python
import numpy as np

# Generate a random input matching your model's expected shape
inp = np.random.randn(1, 64).astype("float32")
inp.tofile("/tmp/input.bin")

# Run on CPU to get the reference output
import sys
sys.path.insert(0, "tests/ti-dsp-runtime/dsp-tests")
from test_c7x_vm_dsp import _cpu_reference_mlp
ref = _cpu_reference_mlp(inp)
ref.tofile("/tmp/ref.bin")
```

Or use the pytest fixture directly — `test_c7x_vm_dsp.py::TestC7xCpp`
generates and transfers the files automatically.

### Running on AM67A

SSH into the board and run:

```bash
# Minimal: load + infer + repeatability only (no reference check)
test_c7x_runtime /path/to/lib0.out /path/to/input.bin \
    --shape 1,64 --dtype float32

# With reference comparison (max |out - ref| < 1e-3)
test_c7x_runtime /path/to/lib0.out /path/to/input.bin \
    --shape 1,64 --dtype float32 \
    --ref /path/to/ref.bin --atol 1e-3

# Classification model example (ResNet-18 style, 1000 classes)
test_c7x_runtime resnet18.out input_1x3x224x224.bin \
    --shape 1,3,224,224 --dtype float32 \
    --ref cpu_output_1x1000.bin --atol 5e-3
```

### CLI reference

```
test_c7x_runtime <lib0.out> <input.bin>
                 [--shape D0,D1,...] [--dtype TYPE]
                 [--ref ref.bin] [--atol TOL]

Required:
  lib0.out      TVM c_static_lib DLOAD module (output of build_dsp_dynmod())
  input.bin     Raw binary input tensor, flat row-major, no header

Optional:
  --shape       Comma-separated dimensions matching the model input
                (default: 1,64)
  --dtype       Element type: float32 float16 int32 int8 uint8
                (default: float32)
  --ref         CPU reference output for numerical comparison
  --atol        Absolute tolerance for --ref comparison (default: 1e-3)

Exit code: 0 = all run tests passed; N = N failures
```

### Expected output (all tests pass)

```
test_c7x_runtime: /tmp/mlp_lib0.out
  input: /tmp/input.bin  shape: 1,64  dtype: float32 (256 bytes)

--- Test 1: LOAD/CLOSE
  PASS  load_close

--- Test 2: INFERENCE
  Output: ndim=2  data_size=32  dtype=2.32
    shape[0]=1
    shape[1]=8
  PASS  inference

--- Test 3: REFERENCE COMPARISON
  max|out - ref| = 1.23e-07  (atol=1.00e-03)
  PASS  reference

--- Test 4: CREATE_INPUT
  PASS  create_input_range
  PASS  create_input_result

--- Test 5: REPEATED_INFER
  PASS  repeated_infer

Results: 5 passed, 0 failed
```

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| AM67A (J722S) board | Running Linux (Yocto/Ubuntu) |
| c7x_compute firmware | Running on DSP; check with `c7x_compute ping` |
| `libc7x_arm_runtime.so` | Installed via `./build.sh deploy` |
| `lib0.out` | TVM c_static_lib module for C7x (DLOAD-compatible ELF) |
| `aarch64-linux-gnu-g++` | Cross-compiler, for building on dev PC |

## DSP C++ Harness Build Reference

For developing custom DSP executables that run TVM-generated code directly
via the DSP runtime (bypassing the firmware/IPC layer), see the
`tests/ti-dsp-runtime/dsp-cpp/` infrastructure.

### Quick Start (Python)

```python
import sys
sys.path.insert(0, "path/to/dsp-cpp")

from dsp_utils import compile_and_run_dsp, compare_results

# Compile and run on DSP (mod is a TVM IRModule with parameters bound)
results = compile_and_run_dsp(
    mod=mod,
    input_data=input_data,  # numpy array or tuple of arrays
    target_string="c_static_lib -mcpu=c66x",
    execution_mode="both",  # "host", "c66x", or "both"
    build_type="Release",   # "Release" or "Debug"
)
```

The workflow uses file-based I/O:
1. Python writes input tensors to `input.bin`
2. DSP executable reads `input.bin`, runs inference, writes `output.bin`
3. Python reads `output.bin` and compares against reference

See `../dsp-tests/test_clista_dsp.py` for a complete example.

### Key Files

| File | Description |
|------|-------------|
| `main_dsp.cpp` | Main entry point - reads input.bin, runs inference, writes output.bin |
| `CMakeLists.txt` | Build configuration for host, C66x, and C7x targets |
| `dsp_utils.py` | Python utilities for DSP compilation and execution |
| `io/tensor_file.cpp`, `io/tensor_file.h` | Tensor file I/O (`input.bin`/`output.bin`) |
| `io/weights_loader.cpp`, `io/weights_loader.h` | Weights source (filesystem or linker-embedded) |
| `io/tensor_file_format.md` | Binary tensor file format specification |

### Prerequisites

**For Host Emulation:**
- CMake 3.16+
- C++11 compiler (GCC, Clang, or AppleClang)
- TVM DSP runtime library (`libtvm_dsp_runtime_host.a`)

**For C66x Hardware:**
- TI C6000 Compiler v8.5.0+ (part of CCS)
- MMWAVE-L-SDK-6 v6.1.0.05
- AWRL6844 evaluation board with XDS110 debug probe
- TVM DSP runtime library (`libtvm_dsp_runtime_c66x.a`)

**For C7x Hardware:**
- TI C7000 Compiler v5.0.0+ (part of CCS)
- J722S evaluation board with XDS110 debug probe
- Code Composer Studio 12.0+
- TVM DSP runtime library (`libtvm_dsp_runtime_c7x.a`)

### Building

#### Step 1: Build the DSP Runtime

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

#### Step 2: Generate TVM Model Code

```python
import tvm
from tvm import relax

# Load/compile your model to get an IRModule
mod = ...

# Build with c_static_lib target for C66x
target = tvm.target.Target("c_static_lib -mcpu=c66x")
with tvm.transform.PassContext(opt_level=3):
    ex = relax.build(mod, target, exec_mode="compiled", system_lib=True)

# Export generated code to a directory
ex.export_library("model_dir/model_library.tar", target=target)
# Extract to get lib0.c and weights.bin
```

Or use `dsp_utils.py` which handles this automatically:

```python
from dsp_utils import compile_for_dsp
generated_dir = compile_for_dsp(mod, "c_static_lib -mcpu=c66x")
```

#### Step 3: Build Host Emulation

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

#### Step 4: Build for C66x Hardware

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

#### Step 5: Build for C7x Hardware (J722S)

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

### Configuration Options

**CMake Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type: `Debug` or `Release` |
| `TVM_DSP_TARGET` | `host` | Target: `host`, `c66x`, `c7x_host`, or `c7x-dynmod` |
| `TVM_DSP_DEVICE` | (none) | Device variant: `awrl6844` or `j722s` |
| `GENERATED_CODE_DIR` | `../cstatic-tests` | Directory containing lib0.c and weights.bin |
| `TVM_HOME` | Auto-detect | Path to TVM repository |
| `WEIGHTS_FILE` | `${GENERATED_CODE_DIR}/weights.bin` | Path to weights file |

**Model Configuration:**

Set via CMake, compiled in as preprocessor definitions consumed by the
Model API:

```bash
cmake -DMODEL_ENTRY_FUNCTION=main -DMODEL_NUM_INPUTS=1 \
      -DMODEL_RETURNS_TUPLE=ON -DGENERATED_CODE_DIR=/path/to/model_dir ..
```

Input shape and data are provided via `input.bin` at runtime (no recompilation needed).

### Generated Code Notes

When using `-mcpu=c66x` target, the generated `lib0.c`:
- Includes DSP-specific headers directly (`ffi_types.h`, etc.)
- Uses `TVM_DSP_SKIP_CG_MAIN` to exclude exception-based wrapper
- Is compiled as C++ to support the DSP runtime API

The CMakeLists.txt handles this automatically:
```cmake
set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)
```

### Troubleshooting

**"Failed to load weights"**
- Verify `weights.bin` exists at the configured path
- Check file permissions
- For C66x: ensure weights are properly embedded

**"TVMFFIFunctionCall with NULL function"**
- Ensure `InitVMBuiltins()` is called before `__vmtir__main`
- Verify VM builtins are registered with `TVMDSPRegisterVMBuiltins()`

**Inference timeout on C66x**
- Check memory usage in the `.map` file
- Verify L2/L3 pool sizes are sufficient
- Try increasing timeout: `--timeout 300000`

**Linker errors about LLVM paths**
- The toolchain file should clear host linker flags
- If not, manually set: `-DCMAKE_EXE_LINKER_FLAGS=""`

## Dynamic Shape Limitation

The `c_static_lib` backend does not support Relax loops (tail-recursive
functions requiring `vm.builtin.invoke_closure` with a runtime function
dispatch table). The static C codegen generates standalone C functions
with no inter-function call mechanism. Models needing iterative computation
(autoregressive LLMs, RNNs, iterative refinement) must unroll the loop at
the graph level or use the VM backend instead.

See `tests/ti-dsp-runtime/dynamic-tests/` for tests exercising supported
dynamic features (`If` expressions, symbolic batch dimensions).
