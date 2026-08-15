# Python / C++ API Reference: `C7xVirtualMachine` / `c7x::Module`

Arm-side inference API for TVM `c_static` modules running on the C7x DSP.
Provides a `relax.VirtualMachine`-compatible interface in both Python and
C++, routing inference to the DSP via the `c7x_compute` IPC service.

The C++ API is a direct mirror of the Python one, so this single document
covers both.

**Runs on the board, not the dev host.** Both APIs talk to the DSP over the
board's local rpmsg IPC channel, so your Python/C++ process must run on the
ARM Linux side of the target board (AM67A / BeagleY-AI) itself — it cannot
connect to a board remotely over the network. Compile/quantize the model on
your dev host, then copy the compiled artifact (`lib0.out`) to the board to
run inference.

**Prerequisites:**
- `c7x_compute` firmware already running on the DSP (check with
  `c7x_compute ping` on the board)
- `libc7x_arm_runtime.so` installed on the board — see [Getting
  Started](getting-started.md) for the wheel-based build+deploy flow, or
  [Deploying Firmware](deploying-firmware.md) for the native
  cross-compile/scp flow. [Locating Library and Header
  Paths](#locating-library-and-header-paths) below covers resolving where
  either flow put the `.so`/header, for building your own application
  against these APIs.

---

## Python API

**Import:**

```python
from tvm.contrib.c7x import C7xVirtualMachine
```

### Standard usage (copy-based inputs)

Identical syntax to `relax.VirtualMachine` on CPU:

```python
import tvm, numpy as np
from tvm.contrib.c7x import C7xVirtualMachine

vm = C7xVirtualMachine("/models/resnet18.out")
inp = tvm.nd.array(np.random.randn(1, 3, 224, 224).astype("float32"))
out = vm["main"](inp)        # returns tvm.nd.NDArray
print(out.numpy().argmax())

vm.close()
```

Context manager form (recommended for scripts):

```python
with C7xVirtualMachine("/models/resnet18.out") as vm:
    out = vm["main"](inp)
```

### Zero-copy outputs: `run_nocopy()`

`vm["main"]()` copies outputs from result DDR to new memory (safe across
multiple calls).  `run_nocopy()` returns numpy views directly into the mmap'd
result buffer — no copy — but they are **only valid until the next
`run_nocopy()` call**:

```python
out_np = vm.run_nocopy(data)   # numpy array, zero-copy
process(out_np)                # must finish before the next run_nocopy() call
```

### Zero-copy inputs: `create_input()`

Pre-allocate an input tensor inside the staging DDR buffer.  Writing to it
skips the staging `memcpy` on the next inference call:

```python
staging = vm.create_input((1, 3, 224, 224), "float32")  # backed by staging DDR
staging.copyfrom(frame)     # writes directly to staging buffer
out = vm["main"](staging)   # no input copy
```

The tensor is valid until `close()` is called.  Multiple inputs can be
pre-staged; each call to `create_input()` advances the allocation offset by the
tensor size (aligned to 64 bytes).

### Properties

```python
vm.last_cycles   # DSP TSC cycle count from the most recent inference (int)
vm.is_loaded     # True if the module is currently loaded on the DSP (bool)
```

### Constructor

```python
C7xVirtualMachine(module_path, so_path="libc7x_arm_runtime.so")
```

- `module_path`: path to `lib0.out` (the TVM c_static dynmod)
- `so_path`: name or path of `libc7x_arm_runtime.so`. A bare name (the
  default) first checks the wheel's own bundled copy
  (`tvm.data.ti_dsp.paths.find_c7x_arm_runtime_so()`), then falls back to
  `ctypes.util.find_library` + `LD_LIBRARY_PATH` -- so a `tvm-ti-c7x-inference`
  wheel install needs no override at all; only a native/source-tree deploy
  does (or pass an explicit absolute path to skip the lookup).

Connection to the DSP is established lazily on the first call to `vm["main"]`
or `create_input()`.

---

## C++ API

**Header:**
`c7x_runtime.h` (`src/runtime/ti_dsp/firmware/c7x/arm/include/c7x_runtime.h`
in the source tree)
— not deployed to `/usr/local/include/` by `./build.sh deploy`; resolve it
from the `tvm-ti-c7x-inference` wheel or the source tree instead, see
[Locating Library and Header Paths](#locating-library-and-header-paths)

No TVM runtime dependency — only DLPack is required, so this header can be
used from any C++ application on the board.

### Standard usage

```cpp
#include "c7x_runtime.h"

// Load module — throws std::runtime_error on failure
auto vm = c7x::Module::Load("/models/resnet18.out");

// Single-input convenience
c7x::OutputTensor out = vm.Run(&input_dl_tensor);
// out.dl.data → pointer into result DDR, valid until next Run() or Close()

// Multi-input convenience
std::vector<c7x::OutputTensor> outs = vm.Run({&in0, &in1});
```

### Function dispatch (mirrors TVM C++ Module)

```cpp
c7x::OutputTensor outputs[8];
int num_outputs = 0;
const DLTensor* inputs[] = { &dl_tensor };

auto fn = vm["main"];
int rc = fn(inputs, 1, outputs, &num_outputs);
// rc == 0 on success; outputs[0..num_outputs-1] are valid
```

### Zero-copy inputs

```cpp
int64_t shape[] = {1, 3, 224, 224};
DLDataType float32 = {kDLFloat, 32, 1};
DLTensor* inp = vm.CreateInput(shape, 4, float32);  // data in staging DDR
memcpy(inp->data, my_data, nbytes);
auto out = vm.Run(inp);  // no staging memcpy
```

`CreateInput` returns `nullptr` if the staging buffer is full or the module is
not loaded.  Supports up to `kMaxInputs` (128) pre-staged tensors.

### Class reference

```cpp
namespace c7x {

struct OutputTensor {
    DLTensor dl;        // data → result DDR (zero-copy); valid until next Run/Close
    int64_t  _shape[6]; // shape storage (dl.shape → this)
    size_t   data_size; // byte size of the output
};

class Module {
public:
    static Module Load(const std::string& lib0_path);  // throws on failure

    ~Module();
    Module(Module&&) noexcept;
    Module& operator=(Module&&) noexcept;
    Module(const Module&) = delete;

    struct Function {
        int operator()(const DLTensor* const* inputs, int num_inputs,
                       OutputTensor* outputs, int* num_outputs) const;
    };

    Function                       operator[](const std::string& name);
    OutputTensor                   Run(const DLTensor* input);
    std::vector<OutputTensor>      Run(const std::vector<const DLTensor*>& inputs);

    DLTensor* CreateInput(const int64_t* shape, int ndim, DLDataType dtype);
    void*     StagingBuffer(size_t* size_out = nullptr) const;
    void      Close();
};

} // namespace c7x
```

---

## Memory and lifetime rules

Getting these wrong is the most common source of bugs — outputs and
pre-staged inputs are views into shared DDR, not independent allocations:

| API | Valid until |
|-----|-------------|
| `vm["main"](...)` / C++ `Run(...)` | Not time-limited — output is copied to new memory before return. |
| `vm.run_nocopy(...)` output | The **next** `run_nocopy()` call (no copy — a numpy view of `result_buf`). |
| C++ `OutputTensor.dl.data` | The **next** `Run()` call, or `Close()`. |
| `vm.create_input(...)` / C++ `CreateInput(...)` tensor | `vm.close()` / `Close()`. |

If you need output data to outlive the next inference call, copy it
yourself (e.g. `memcpy(my_buf, out.dl.data, out.dl.data_size)` in C++, or use
`vm["main"](...)` instead of `run_nocopy()` in Python).

---

## Locating Library and Header Paths

`libc7x_arm_runtime.so` and `c7x_runtime.h` (used by both APIs above) are
built and deployed to the board by [Getting Started](getting-started.md)'s
wheel-based flow, or natively by [Deploying Firmware](deploying-firmware.md)
(`arm/build.sh deploy`, i.e. cross-compile + scp + `ldconfig`) -- see
[Deploying Firmware -- Testing](deploying-firmware.md#testing) for how to
confirm either deploy actually works before building against it. This
section is only about resolving *where those files ended up* afterwards,
for building your own application against the API -- not about the
build/deploy step itself.

**Wheel install:** `pip install tvm-ti-c7x-inference` on the board unpacks
`libc7x_arm_runtime.so`, `c7x_runtime.h`, and this Python module into
site-packages. The Python API's `so_path` already resolves this
automatically (see Constructor above); a C++ build, or anything linking
against the `.so` directly, has to resolve both paths itself instead of
assuming a system install:

```bash
python3 -c "from tvm.data.ti_dsp.paths import find_c7x_include_dir; \
    print(find_c7x_include_dir())"
python3 -c "from tvm.data.ti_dsp.paths import find_c7x_arm_runtime_so; \
    print(find_c7x_arm_runtime_so())"
```

`ctypes.CDLL()` (used by the Python API) loads the `.so` by absolute path,
so no `ldconfig` step is needed either way. A C++ build should pass the
include dir as `-I` and either link against the resolved `.so` path
directly or `dlopen()` it at runtime. `find_c7x_include_dir()` returns
`None` outside a wheel install (e.g. in a source checkout, where the
header and DLPack live in two separate source-tree directories instead
of one merged `include/`) — callers should fall back to those paths in
that case, as `examples/run_resnet18_classification.py`
(`tests/ti-dsp-runtime/examples/run_resnet18_classification.py` in the
source tree) does.

---

## Examples

[Examples: YOLO26 & ResNet-18](examples.md) has full runnable examples
for both APIs:

- Python: YOLO26 object detection on BeagleY-AI/AM67A --
  `run_yolo26_detection.py` compiles and deploys the model from the dev
  host, and `yolo26_board_runner.py` is the board-side script that
  actually calls `C7xVirtualMachine`.
- C++: ResNet-18 classification -- `run_resnet18_classification.py`
  compiles and cross-compiles from the dev host, and
  `resnet18_board_runner.cpp` is the board-side program that calls
  `c7x::Module` directly, built on the small shared
  `c7x_infer.h` (`tests/ti-dsp-runtime/examples/common/c7x_infer.h` in
  the source tree) helper library.
