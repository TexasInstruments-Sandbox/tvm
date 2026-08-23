# ARM Host Client Internals

Internal design of the ARM-side IPC client (`libc7x_arm_runtime.so`) that
backs the `C7xVirtualMachine` (Python) / `c7x::Module` (C++) APIs -- see
[Python / C++ API Reference](../../user-guide/python-api.md) for the public API
surface, and [Deploying Firmware](../../user-guide/deploying-firmware.md)
for build/deploy instructions.

## Component layout

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

## Per-buffer dmabufs

See [Per-Buffer dmabuf Design](dmabuf-design.md) for why this replaced a
single 512 MB dmabuf (measured sync cost, decision record); this section is
the resulting mechanics and API.

Each purpose gets its own `DmaBuf` (fd + own `/dev/remoteprocN` attachment +
mmap + DSP address), synced independently via `DMA_BUF_IOCTL_SYNC` -- the
ioctl has no offset/length, so per-buffer granularity is the only lever for
not syncing far more than actually moved on every inference:

| Buffer | Allocated | Size | Notes |
|--------|-----------|------|-------|
| `client->staging` | `c7x_client_open()` | 480 MB (`C7X_STAGING_SIZE + C7X_KV_SIZE`), fixed | ELF (DLOAD) + weights (MODEL_LOAD) only -- **not** inference input tensors. Allocated first so `gen_pool_first_fit` packs it at the carveout's base, keeping `C7X_STAGING_ADDR`/`C7X_KV_ADDR` valid as fixed DSP addresses without the host needing to steer the allocator. |
| `client->printf_buf` | `c7x_client_open()` | 64 KB, fixed | DSP printf output. Rebinds the DSP's boot-time `shm_printf` target via `C7X_MSG_SET_PRINTF_BUF`, sent once right after open. |
| `client->input_buf` | `c7x_client_dyn_load()` | Declared, from `tvm_dsp_io_meta` (or `c7x_client_reserve_io()`) | Inference input tensors + a descriptor region at the front (see below). Grows (never shrinks) if a later `DYN_LOAD` needs more. |
| `client->output_buf` | `c7x_client_dyn_load()` | Declared, from `tvm_dsp_io_meta` (or `c7x_client_reserve_io()`) | Inference output tensors. Same growth rule as `input_buf`. |

KV cache tensors (`C7X_INFER_FLAG_KV_RESIDENT`) still live at the fixed
`C7X_KV_ADDR`, written directly by the DSP -- unaffected by any of the above,
since `client->staging`'s 480 MB reservation already covers that range.

## Zero-copy strategy

| Path | Mechanism |
|------|-----------|
| **Output** | `c7x_client_infer()` returns `data` pointers directly into the mmap'd `output_buf` dmabuf.  `vm["main"]()` does one `np.copy()` for safety.  `run_nocopy()` skips the copy and returns numpy views of `output_buf`. |
| **Input (standard)** | `c7x_compute_client` copies the user buffer into `input_buf` via `memcpy`. |
| **Input (zero-copy)** | `create_input()` / `CreateInput()` allocates a tensor **inside** `input_buf`.  On the next `Run()`, the client detects that the input pointer is already within `input_buf`'s range and skips the `memcpy`. |

## input_buf layout

```
input_buf (mmap'd dmabuf, declared capacity)
┌──────────────────────────┬──────────────────────────────┐
│  Descriptor region       │  create_input() allocations  │
│  [0 .. input_data_offset)│  [input_data_offset .. size) │
└──────────────────────────┴──────────────────────────────┘
```

The descriptor region holds the `c7x_tensor_desc` array for
`C7X_MSG_INFER_LARGE` (used when too many tensors to fit inline in the
512-byte rpmsg buffer) -- sized from the declared input count, at the front
of `input_buf` rather than "wherever there's room before the tensor data"
(the pre-per-buffer design squeezed it in just below where the ELF ended).
`c7x_client_get_input_data_offset()` returns this region's size, preventing
`create_input()` from overlapping it.

## Capacity: declared, not grown on demand

`input_buf`/`output_buf` are sized once from the loaded module's
`tvm_dsp_io_meta` (see `tvm.contrib.c7x.io_meta`), or via
`c7x_client_reserve_io()` for a module built without one -- or one whose
table over-declares for the calling convention in use (see below). Exceeding
the declared capacity is always an error -- never silently absorbed:

- **Input** over capacity: caught host-side, before staging anything
  (`-EFBIG`).
- **Output** over capacity: only the firmware can detect it, mid-write,
  against the transmitted `result_size` -- `C7X_STATUS_ERR_SIZE` with
  `result_required` naming the capacity needed through the tensor that
  overflowed. It is not the total for the whole output set: the firmware
  stops at the first tensor that doesn't fit, so for a multi-output model
  a reserve-and-retry cycle can report a larger figure each round until
  every tensor fits.

`c7x_client_reserve_io()` is rejected once capacity is locked -- which
happens on the *first* successful use (a `CreateInput()` call, or a
completed inference that handed back output pointers), not merely on
attempting one, so a too-small reservation can still be grown and retried
on the same client without corruption.

### When the table over-declares

`tvm_dsp_io_meta` is derived from the entry signature, which bounds capacity
over *calling conventions* rather than describing one call. The same compiled
module has two footprints: with `C7X_INFER_FLAG_KV_RESIDENT` the firmware
diverts every output past the first to `C7X_KV_ADDR`, so only the first
reaches `output_buf`.

SmolLM-135M prefill returns 61 tensors (logits + 60 KV) and declares
24,384,320 bytes of output. Driven kv-resident it needs 12,583,040. Its input
side genuinely needs 11,802,496 -- prefill must push a zeroed cache, so it
can't use the synthetic-descriptor path decode uses -- and the carveout has
33,488,896 bytes above the staging reservation. The declaration doesn't fit;
the actual use does, with ~9 MB spare.

So `c7x_client_dyn_load()` treats an unmeetable declaration as advisory:
whatever partial sizing succeeded is kept, a warning names declared vs
allocated, and the load reports success. The caller then declares its real
per-call figure with `c7x_client_reserve_io()` before the first inference.
`tests/ti-dsp-runtime/SmolLM/` does this from `metadata.json`'s `io_reserve`,
computed at compile time by `tvm.contrib.c7x.io_meta.kv_resident_output_bytes()`
and sent over the session protocol's `reserve_io` op.

This is why the buffer table above lists `c7x_client_reserve_io()` as a size
source alongside the table, rather than only as a fallback for modules built
without one.

## Known limitation: single client only

The firmware tracks the *current* session's `output_buf`/`printf_buf`
base+size as process-global state (`g_result_dsp_addr`/`g_result_size` in
`compute_service.c`, rebindable printf target), not per-connection. A second
concurrently-connected client would silently redirect the first's output
and printf buffers. Acceptable today (one DSP core, access already
serialised by the "never two `c7x_dload` sessions on the same board"
testing rule) but worth knowing if that assumption ever changes.

## ctypes struct (`_C7xTensorDesc`)

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

## Output lifetime

- `vm["main"]()`: outputs are copied to new memory before return — safe to
  hold across multiple inference calls.
- `run_nocopy()`: outputs are numpy views of `output_buf` — valid only until
  the next `run_nocopy()` call (next inference overwrites the buffer).
- C++ `OutputTensor.dl.data`: valid until the next `Module::Run()` or
  `Module::Close()`.
