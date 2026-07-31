# C7x Arm Runtime: `C7xVirtualMachine` / `c7x::Module`

Arm-side inference API for TVM `c_static` modules running on the C7x DSP.
Provides a `relax.VirtualMachine`-compatible interface in both Python and
C++, routing inference to the DSP via the `c7x_compute` IPC service. See
[`README.md`](../../../../../../README.md) for how this fits into the
overall compile/deploy pipeline.

---

## Python API

**Import:**

```python
from tvm.contrib.c7x import C7xVirtualMachine
```

Or, via a TIDL build result:

```python
result = TIDLOffloadCompiler(config).build(mod, params)
vm = result.as_vm()
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
- `so_path`: name or path of `libc7x_arm_runtime.so`; if a bare name, resolved
  via `ctypes.util.find_library` and `LD_LIBRARY_PATH`

Connection to the DSP is established lazily on the first call to `vm["main"]`
or `create_input()`.

---

## C++ API

**Header:** [`include/c7x_runtime.h`](include/c7x_runtime.h) (installed to
`/usr/local/include/` by `./build.sh deploy`)

No TVM runtime dependency — only DLPack is required.

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
not loaded.  Returns up to `kMaxInputs` (128) pre-staged tensors.

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

## Build and Deploy

The Arm shared library is cross-compiled for aarch64 on the dev host.

```bash
cd src/runtime/ti_dsp/firmware/c7x/arm

# Cross-compile for aarch64 (default):
./build.sh
# Produces: build/libc7x_arm_runtime.so, build/c7x_compute, build/test_c7x_runtime

# Deploy to AM67A (scp + ldconfig):
./build.sh deploy
# Installs:
#   /usr/local/lib/libc7x_arm_runtime.so
#   /usr/local/bin/c7x_compute
#   /usr/local/include/c7x_runtime.h
#   /usr/local/bin/test_c7x_runtime  (if built)

# Native build (run on the AM67A itself):
./build.sh native
```

`BOARD_HOSTNAME` (default `am67a`) and `CROSS_COMPILE` (default
`aarch64-linux-gnu-`) are configurable via environment variables. `--board
<j722s-evm|beagley-ai>` / `--ddr <4gb|8gb>` are also accepted, forwarded
purely for build-dir naming consistency with `build_runtime.sh` and
`dsp/build.sh` — they don't change what's compiled here.

See [`test/README.md`](test/README.md) for the standalone C++ test binary
that exercises this API end-to-end against live DSP firmware.

---

## Integration with the TIDL Build Pipeline

`TIDLBuildResult.as_vm()` wraps the build output in a `C7xVirtualMachine`:

```python
from tvm.relax.backend.tidl import TIDLOffloadCompiler

result = TIDLOffloadCompiler(config).build(mod, params)
vm = result.as_vm()                    # C7xVirtualMachine(result.module_path)
out = vm["main"](inp)

# Custom .so path:
vm = result.as_vm(so_path="/opt/lib/libc7x_arm_runtime.so")
```

`C7xVirtualMachine` is also re-exported from `tvm.relax.backend.tidl`:

```python
from tvm.relax.backend.tidl import C7xVirtualMachine
```

---

## Design

### Component layout

```
python/tvm/contrib/c7x/
├── __init__.py             — exports C7xVirtualMachine
└── c7x_runtime.py          — Python ctypes wrapper

src/runtime/ti_dsp/firmware/c7x/arm/
├── CMakeLists.txt          — builds libc7x_arm_runtime.so + c7x_compute CLI + test binary
├── build.sh                — cross-compile + deploy script
├── include/
│   └── c7x_runtime.h       — C++ c7x::Module API (DLPack-only dependency)
└── src/
    ├── c7x_runtime.cc      — c7x::Module implementation
    ├── c7x_compute_client.cpp  — IPC client (rpmsg, staging/result DDR)
    └── c7x_compute_cli.cpp — CLI executable (links libc7x_arm_runtime.so)
```

`libc7x_arm_runtime.so` contains `c7x_compute_client`, `rpmsg_wrapper`, and
`c7x_runtime`.  The CLI binary (`c7x_compute`) links this library unchanged.

### Zero-copy strategy

| Path | Mechanism |
|------|-----------|
| **Output** | `c7x_client_infer()` returns `data` pointers directly into the mmap'd `result_buf` (shared DDR).  `vm["main"]()` does one `np.copy()` for safety.  `run_nocopy()` skips the copy and returns numpy views of `result_buf`. |
| **Input (standard)** | `c7x_compute_client` copies the user buffer into `staging_buf` via `memcpy`. |
| **Input (zero-copy)** | `create_input()` / `CreateInput()` allocates a tensor **inside** `staging_buf`.  On the next `Run()`, the client detects that the input pointer is already within `[staging_buf, staging_buf + staging_size)` and skips the `memcpy`. |

### Staging buffer layout

```
staging_buf (mmap'd shared DDR, C7X_STAGING_SIZE bytes)
┌──────────────────────────┬──────────────────────────────┐
│   ELF image (lib0.out)   │  create_input() allocations  │
│   [0 .. elf_size)        │  [elf_size .. staging_size)  │
└──────────────────────────┴──────────────────────────────┘
```

`c7x_client_get_input_data_offset()` returns `elf_size` after `dyn_load`,
preventing `create_input()` from overlapping the loaded ELF.

### ctypes struct (`_C7xTensorDesc`)

Python-side struct matching `c7x_tensor_desc_t` in `c7x_compute_client.h`:

```
Offset  Field        Type      Bytes
0       data         void*     8
8       data_size    size_t    8
16      ndim         int32     4
20      dtype_code   int32     4   (DLPack: 0=Int 1=UInt 2=Float)
24      dtype_bits   int32     4
28      _pad         int32     4   (alignment)
32      shape[6]     int64[6]  48
Total                          80
```

Validated at import time: `assert ctypes.sizeof(_C7xTensorDesc) == 80`.

### Output lifetime

- `vm["main"]()`: outputs are copied to new memory before return — safe to
  hold across multiple inference calls.
- `run_nocopy()`: outputs are numpy views of `result_buf` — valid only until
  the next `run_nocopy()` call (next inference overwrites the buffer).
- C++ `OutputTensor.dl.data`: valid until the next `Module::Run()` or
  `Module::Close()`.
