# C7x Arm Runtime — Build, Deploy, and Internals

For the `C7xVirtualMachine` (Python) / `c7x::Module` (C++) API reference, see
[`python/tvm/contrib/c7x/README.md`](../../../../../../python/tvm/contrib/c7x/README.md).
This README covers building and deploying `libc7x_arm_runtime.so` and the
internal design of the IPC client. See the top-level
[`README.md`](../../../../../../README.md) for how this fits into the overall
compile/deploy pipeline.

---

## Build and Deploy

The Arm shared library is cross-compiled for aarch64 on the dev host.

```bash
cd src/runtime/ti_dsp/firmware/c7x/arm

# Cross-compile for aarch64:
./build.sh --board j722s-evm
# Produces: build/libc7x_arm_runtime.so, build/c7x_compute, build/test_c7x_runtime

# Deploy to AM67A (scp + ldconfig):
./build.sh --board j722s-evm deploy
# Installs:
#   /usr/local/lib/libc7x_arm_runtime.so
#   /usr/local/bin/c7x_compute
#   /usr/local/bin/test_c7x_runtime  (if built)
# c7x_runtime.h is not deployed here -- C++ consumers get it from the
# tvm-ti-c7x-inference wheel or the source tree; see
# python/tvm/contrib/c7x/README.md.

# Native build (run on the AM67A itself):
./build.sh --board j722s-evm native
```

`--board <j722s-evm|beagley-ai>` is required (`--ddr <4gb|8gb>` stays
optional, default per-board): besides build-dir naming consistency with
`build_runtime.sh` and `dsp/build.sh`, `--board` also picks the `deploy`
subcommand's default SSH host (`beagley-ai` -> `beagley-ai`, else
`am67a`). `CROSS_COMPILE` (default `aarch64-linux-gnu-`) is still
configurable via an environment variable; there's no `BOARD_HOSTNAME`
override for the deploy host — add an SSH-config alias if your board is
reachable under a different name.

See [`test/README.md`](test/README.md) for the standalone C++ test binary
that exercises this API end-to-end against live DSP firmware.

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
