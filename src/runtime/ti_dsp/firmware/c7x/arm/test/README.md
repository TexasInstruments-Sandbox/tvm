# test_c7x_runtime

C++ test binary for the `c7x::Module` ARM inference API. Exercises
`libc7x_arm_runtime.so` end-to-end on an AM67A ARM board against a
live C7x DSP firmware instance.

This is the C++ counterpart to the Python integration tests in
`tests/ti-dsp-runtime/dsp-tests/test_c7x_vm_dsp.py`. Both test the
same underlying `c7x_compute_client` IPC path; this binary tests the
C++ `c7x::Module` wrapper directly, without Python or TVM overhead.

## Test cases

| # | Name | What it checks |
|---|------|----------------|
| 1 | LOAD/CLOSE | `Module::Load()` connects to the DSP and loads the ELF; `Close()` is idempotent (safe to call twice) |
| 2 | INFERENCE | `Run()` returns a non-degenerate output tensor (`ndim > 0`, `data_size > 0`); prints shape and dtype |
| 3 | REFERENCE | If `--ref` is supplied: `max|out - ref| < atol`; skipped otherwise |
| 4 | CREATE\_INPUT | `CreateInput()` pointer lies within `[StagingBuffer(), StagingBuffer()+size)` (pre-staged DDR); inference result matches standard-path result |
| 5 | REPEATED\_INFER | Three consecutive `Run()` calls with identical input produce bit-identical output |

Test 1 is a prerequisite: if the module cannot be loaded, the remaining
tests are skipped and the binary exits immediately.

## Building

`test_c7x_runtime` is built automatically alongside `libc7x_arm_runtime.so`
when the `arm/` CMakeLists detects the source file:

```bash
cd src/runtime/ti_dsp/firmware/c7x/arm

# Cross-compile for ARM64 (default)
./build.sh

# Or natively on the AM67A board
./build.sh native
```

Outputs written to `arm/build/`:

```
libc7x_arm_runtime.so   — shared library (required at runtime)
c7x_compute             — CLI tool
test_c7x_runtime        — this test binary
```

## Deploying to AM67A

```bash
cd src/runtime/ti_dsp/firmware/c7x/arm
./build.sh deploy
```

This SCPs all three binaries plus `c7x_runtime.h` to the board and
runs `ldconfig` so the shared library is found by the dynamic linker:

```
am67a:/usr/local/bin/c7x_compute
am67a:/usr/local/bin/test_c7x_runtime
am67a:/usr/local/lib/libc7x_arm_runtime.so
am67a:/usr/local/include/c7x_runtime.h
```

The AM67A hostname is taken from the `AM67A_TARGET` environment variable
(default: `am67a`).

## Preparing test inputs

The test binary takes a raw flat binary `input.bin` (contiguous, row-major).
Use Python/numpy on the development machine to generate it alongside a
CPU reference output `ref.bin`:

```python
import numpy as np
from tvm.relax.backend.tidl import TIDLBuildResult  # or any TVM build path

# Generate a random input matching your model's expected shape
inp = np.random.randn(1, 64).astype("float32")
inp.tofile("/tmp/input.bin")

# Run on CPU to get the reference output
import tvm
from tvm import relax
from tests.ti-dsp-runtime.dsp_tests.test_c7x_vm_dsp import _cpu_reference_mlp
ref = _cpu_reference_mlp(inp)
ref.tofile("/tmp/ref.bin")
```

Or use the pytest fixture directly — `test_c7x_vm_dsp.py::TestC7xCpp`
generates and transfers the files automatically.

## Running on AM67A

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
  lib0.out      TVM c_static DLOAD module (output of build_dsp_dynmod()
                or TIDLOffloadCompiler.build())
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

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| AM67A (J722S) board | Running Linux (Yocto/Ubuntu) |
| c7x_compute firmware | Running on DSP; check with `c7x_compute ping` |
| `libc7x_arm_runtime.so` | Installed via `./build.sh deploy` |
| `lib0.out` | TVM c_static module for C7x (DLOAD-compatible ELF) |
| `aarch64-linux-gnu-g++` | Cross-compiler, for building on dev PC |
