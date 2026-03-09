# C7x Firmware - Design Document

## Overview

The c7x-firmware is a host-DSP compute service for the TI AM67A (J722S)
that enables Linux applications running on the ARM A53 to offload ML
inference and data processing to the C7x DSP. It communicates over
RPMessage IPC and transfers data through shared DDR memory allocated from
a DMA heap carveout.

The primary use case is running TVM-compiled neural network models on the
C7x DSP. The TVM C static backend produces a relocatable C7x ELF module
(`lib0.out`) with model weights embedded in a `.rodata.weights` section.
The firmware's dynamic loader (DLOAD) loads this ELF at runtime, resolves
symbols against the firmware's export table, and makes the model's entry
point callable via an INFER command from the host.

For build instructions, deployment, CLI usage, and troubleshooting, see
[README.md](README.md).

## System Architecture

```
+------------------------------------------------------------------+
|                    Linux/ARM (A53)                                |
|  +------------------------------------------------------------+  |
|  |  c7x_compute CLI / Library                                 |  |
|  |  - Allocates shared DDR via /dev/dma_heap                  |  |
|  |  - Sends commands via RPMessage (/dev/rpmsg*)              |  |
|  |  - Discovers rpmsg_ctrl by device tree address             |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                          |
                ====================== RPMessage (virtio/rpmsg_char)
                          |
+------------------------------------------------------------------+
|                    C7x DSP (7e000000.dsp)                        |
|  +------------------------------------------------------------+  |
|  |  Compute Service (FreeRTOS)                                |  |
|  |  - RPMessage endpoint 20 ("rpmsg_chrdev")                  |  |
|  |  - DLOAD dynamic loader for C7x ELF modules               |  |
|  |  - TVM model manager for weights/constants                 |  |
|  |  - INFER pipeline: resolve entry, call cg_main_dsp         |  |
|  |  - Clean shutdown with remoteproc ACK                      |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### Component Roles

**Host CLI / Library** (`host/`): User-space Linux C++14 application
that allocates a 512 MB shared buffer from the DMA heap, maps it into
user space, stages data (ELF binaries, input tensors) into the buffer,
and sends IPC commands to the DSP over RPMessage. The library provides
a C API (`c7x_compute_client.h` with `extern "C"` linkage) and the
CLI wraps it for interactive use. Resource management uses RAII
wrappers (`raii.h`) for file descriptors, mmap regions, and FILE
handles.

**Compute Service** (`dsp/src/compute_service.c`): FreeRTOS task that
blocks on `RPMessage_recv()`, dispatches messages by type, and sends
responses. Handles PING, STATUS, DYN_LOAD, INFER, and DYN_UNLOAD
commands. On DYN_UNLOAD, orchestrates TVM runtime cleanup
(register file, model slot, constants) before freeing module memory.

**Dynamic Loader** (`dsp/src/dyn_loader.c` + `dsp/src/dload/`): Wraps
TI's DLOAD library to load relocatable C7x ELF modules into DDR at
runtime. Provides a symbol export table (61 symbols) so loaded modules
can call firmware-provided functions (C library, TVM runtime, math).

**TVM Model Manager** (`dsp/src/tvm_model.c`): Manages model
weights/constants that are either embedded in the ELF `.rodata.weights`
section or loaded separately via the MODEL_LOAD command. Parses the
TVM weights binary format and constructs DLTensor descriptors.

**Shared Protocol** (`common/c7x_compute_protocol.h`): Header shared
between ARM and DSP defining all message structures, types, status
codes, and memory layout constants.

## Message Protocol

All communication uses a binary message protocol over RPMessage with a
maximum message size of 512 bytes. Every message starts with a 16-byte
header:

```c
struct c7x_msg_hdr {
    uint32_t type;      // Message type (C7X_MSG_*)
    uint32_t seq;       // Sequence number for correlation
    uint32_t len;       // Total message length including header
    int32_t  status;    // Response status (0 = success)
};
```

### Message Types

| Type | Code | Direction | Purpose |
|------|------|-----------|---------|
| PING | 0x0001 | Host -> DSP | Connectivity test |
| GET_STATUS | 0x0003 | Host -> DSP | Get service status |
| DYN_LOAD | 0x0010 | Host -> DSP | Load ELF module |
| DYN_UNLOAD | 0x0012 | Host -> DSP | Unload module |
| MODEL_LOAD | 0x0020 | Host -> DSP | Load weights/constants |
| INFER | 0x0021 | Host -> DSP | Run inference |
| MODEL_UNLOAD | 0x0022 | Host -> DSP | Unload model weights |

Response types are `0x1000 | request_type`.

### INFER Message

The INFER message carries tensor descriptors inline:

```c
struct c7x_msg_infer {
    struct c7x_msg_hdr hdr;
    uint32_t module_handle;     // From DYN_LOAD response
    uint32_t model_id;          // From MODEL_LOAD response
    uint32_t num_inputs;
    uint32_t flags;
    struct c7x_tensor_desc inputs[1]; // Variable-length
};
```

Each tensor descriptor (80 bytes) contains the DSP virtual address of
the data in shared memory, data size, dtype, and shape (up to 6
dimensions). The INFER response includes output tensor descriptors
written by the DSP, cycle count, and `printf_size` indicating the
number of bytes of printf output in the shared memory printf buffer.

### Inference Pipeline

The end-to-end flow for running a TVM model:

1. Host writes ELF binary to input buffer at offset 0
2. Host sends DYN_LOAD with ELF size
3. DSP DLOAD parses ELF, allocates segments in TVM DDR heap,
   resolves symbols, applies relocations -> returns module handle
4. Host writes input tensor to input buffer
5. Host sends INFER with module handle, model_id=0 (embedded
   weights), and input tensor descriptor
6. DSP resets printf buffer, resolves `cg_main_dsp` in loaded
   module, constructs DLTensor arguments, calls entry point, writes
   output to output buffer. Any `printf` calls during inference
   accumulate in the SHM printf buffer (see "Shared Memory Printf")
7. DSP flushes printf buffer via `CacheP_wb`, sets `printf_size`
   in the INFER response
8. Host reads output tensor from output buffer, then reads and
   displays any printf data from the printf buffer region
9. Host sends DYN_UNLOAD; DSP cleans up TVM runtime state (register
   file objects, model slot, constants pools) then frees module
   memory via DLOAD (see "Module Unload Lifecycle" below)

## Memory Architecture

### Physical DDR Layout

The AM67A has DDR starting at 0x80000000. All addresses below 4 GB are
identity-mapped on the DSP unless otherwise noted.

#### DSP Firmware Regions (reserved by device tree)

| Region | Physical/Virtual | Size | Cache | Purpose |
|--------|-----------------|------|-------|---------|
| IPC/DMA | 0xAD000000 | 1 MB | Non-cached (MAIR4) | Linux IPC area |
| Resource table | 0xAD100000 | 1 KB | Non-cached | remoteproc resource table |
| IPC trace | 0xAD100400 | ~1023 KB | Non-cached | DebugP_log trace buffer (`trace0`) |
| Boot code | 0xAD200000 | 1 KB | Cached (MAIR7) | C7x boot vector |
| Vectors | 0xAD400000 | 16 KB | Cached | Interrupt vectors |
| Secure vectors | 0xAD600000 | 16 KB | Cached | Secure interrupt vectors |
| Code/Data | 0xAD604000 | ~34 MB | Cached (MAIR7) | Firmware `.text`, `.data`, `.bss` |
| IPC VRing | 0xAF800000 | 8 MB | Non-cached (MAIR4) | RPMessage virtio rings |

#### Shared Compute Buffer (DMA heap carveout)

| Region | DSP Virtual | Physical | Size | Cache | Purpose |
|--------|-------------|----------|------|-------|---------|
| Input buffer | 0xC0000000 | 0x900000000 | 504 MB | Cached (MAIR7) | ELF modules + input tensors |
| Output buffer | 0xDF800000 | 0x91F800000 | 8 MB | Cached (MAIR7) | Inference output tensors |
| Printf buffer | 0xE07F0000 | 0x91FFF0000 | 64 KB | Cached (MAIR7) | DSP printf output (last 64 KB of output buffer) |

This 512 MB region is the `vision_apps_shared-memories` DMA heap
carveout, exclusively for host-DSP communication. No video codecs,
display, or other subsystems use it. The host allocates it via
`/dev/dma_heap/carveout_vision_apps_shared-memories`, and the DSP MMU
maps it at 0xC0000000 with write-back cached attributes (MAIR7, Outer
Shareable).

#### Extended DDR (above 4 GB, MMU-translated)

| Region | DSP Virtual | Physical | Size | Cache | Purpose |
|--------|-------------|----------|------|-------|---------|
| Non-cacheable heap | 0x100000000 | 0x880000000 | 32 MB | Non-cached (MAIR4) | DMA-accessible allocations |
| TVM DDR heap | 0x102000000 | 0x882000000 | 128 MB | Cached (MAIR7) | DLOAD segments + TVM workspace |

The TVM DDR heap is where DLOAD allocates code and data segments for
loaded modules, and where the TVM runtime allocates workspace tensors
during inference.

#### L2 SRAM (C7x-local, not DDR)

| Region | Address | Size | Purpose |
|--------|---------|------|---------|
| L2 main | 0x7E000000 | 2 MB | Runtime stack, FreeRTOS heap |
| L2 aux | 0x7F000000 | 240 KB | Fast scratch memory |
| L1 alias | 0x7F03C000 | 16 KB | L1 cache as SRAM |

### Host-Side Allocation

The host allocates the shared buffer at runtime from the DMA heap:

```
1. open("/dev/dma_heap/carveout_vision_apps_shared-memories")
2. ioctl(fd, DMA_HEAP_IOCTL_ALLOC, {len=512MB}) -> dma_buf_fd
3. mmap(NULL, 512MB, PROT_READ|PROT_WRITE, MAP_SHARED, dma_buf_fd, 0) -> userspace ptr
4. ioctl(rproc_fd, RPROC_IOC_DMA_BUF_ATTACH) -> physical address (0x900000000)
```

### Cache Coherency

All cached shared regions use MAIR7 (Write-Back Read/Write-Allocate)
with Outer Shareable for hardware cache coherency between ARM and DSP.
The host still performs explicit DMA_BUF_SYNC ioctls:
- Before DSP reads: `DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE` (flush)
- Before host reads: `DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ` (invalidate)

### MMU Configuration

Defined in `dsp/configs/c75ss0.syscfg` (16 regions). Key MAIR attribute
indices:

| MAIR | Encoding | Meaning | Used For |
|------|----------|---------|----------|
| MAIR0 | 0x00 | Device-nGnRnE (strongly ordered) | Peripherals, CLEC, DRU, L2SRAM |
| MAIR4 | 0x29 | Normal non-cacheable | IPC VRing, DMA buffers |
| MAIR7 | 0x3D | Write-Back Read/Write-Allocate | Code, data, shared buffer, TVM heap |

All cached regions (MAIR7) use Outer Shareable for hardware cache
coherency between ARM and DSP.

### Memory Budget Example (ResNet-18)

| Resource | Size | Pool | Limit |
|----------|------|------|-------|
| lib0.out ELF (code + 46 MB weights) | ~47 MB | Input buffer | 504 MB |
| DLOAD segments (relocated code+data) | ~47 MB | TVM DDR heap | 128 MB |
| Inference workspace (intermediate tensors) | ~10-20 MB | TVM DDR heap | ~80 MB remaining |
| Input tensor (1,3,224,224 float32) | 0.6 MB | Input buffer | 504 MB |
| Output tensor (1,1000 float32) | 4 KB | Output buffer | 8 MB |

## Dynamic Module Loading (DLOAD)

The firmware embeds TI's DLOAD dynamic linker, which loads standard
C7x ELF relocatable objects at runtime. This is the mechanism by which
TVM-compiled models are deployed without reflashing firmware.

### DLOAD Integration

The `dyn_loader.c` module provides:
- **DLIF callbacks**: Firmware-side implementations of the DLOAD loader
  interface (`DLIF_allocate`, `DLIF_copy`, `DLIF_read`, etc.) that
  allocate from the TVM DDR heap via `tvm_dsp_alloc`
- **Symbol export table**: 61 symbols (C library, TVM runtime, VM
  builtins, math) made available to loaded modules at relocation time
- **Load/unload API**: `dyn_loader_load()` takes an ELF from the shared
  input buffer, and `dyn_loader_unload()` frees all segments

### Supported Relocations

DLOAD handles 22 C7x-specific relocation types (defined in
`dload/C70_DLOAD_REL/c70_reloc.c`), covering all relocations produced
by the TI CGT C7000 compiler for position-dependent code with external
symbol references.

### Symbol Export Table

The export table in `dyn_loader.c` provides these categories of symbols
to loaded modules:

- **C library**: `printf` (redirected to `shm_printf` for shared
  memory output), `memcpy`, `memset`, `malloc`, `free`, `calloc`,
  `__c7xabi_cmpd`, etc.
- **TVM runtime**: `TVMBackendAllocWorkspace`,
  `TVMBackendFreeWorkspace`, `TVMFuncCall`, `TVMArgs_Create`, etc.
- **VM builtins (packed)**: `vm_builtin_*` functions using TVMArgs
- **VM builtins (direct C++ API)**: `tvm_dsp_*` direct-call variants
  for the C7x C++ API backend
- **Math**: `expf`, `logf`, `sqrtf`, `powf`, `floorf`, `fmaxf`, etc.

### Module Unload Lifecycle

When the host sends DYN_UNLOAD, `handle_dyn_unload()` in
`compute_service.c` must clean up TVM runtime state **before** calling
`dyn_loader_unload()`. This ordering is critical because the loaded
module's `.bss` section contains the static register file used by the
generated `cg_main_dsp` wrapper, and `dyn_loader_unload()` frees all
ELF segments including `.bss`.

The cleanup sequence for the currently loaded module is:

1. **`TVMDSPRegFileCleanup()`** -- Iterates the static register file
   (in the module's `.bss`), decrements reference counts on any
   heap-allocated NDArray/storage objects from the last inference, and
   calls their deleters to free TVM DDR heap memory. The generated DSP
   wrapper calls `TVMDSPRegFileInit()` at the start of each inference
   to register the static register file pointer with the TVM runtime.

2. **`tvm_model_unload(model_id)`** -- Frees the model slot in the
   `g_models[]` table (up to `MAX_MODELS=4`). This must happen before
   constants cleanup because `TVMDSPConstantsCleanup()` resets the
   constants subsystem state that the model slot references.

3. **`TVMDSPConstantsCleanup()`** -- Frees all constants memory pools
   allocated by `TVMDSPLoadConstants()` during the INFER setup phase,
   and resets the constants subsystem (`g_initialized = 0`).

4. **`dyn_loader_unload(handle)`** -- Calls `tracked_free_all()` to
   free all ELF segments (`.text`, `.data`, `.bss`, `.rodata.weights`)
   allocated from the TVM DDR heap during `dyn_loader_load()`.

5. **Clear state** -- Reset `g_loaded_module_handle`,
   `g_cg_main_dsp`, and `g_embedded_model_id` to zero.

If steps 1-3 are skipped or performed after step 4, the register file
memory is already freed (use-after-free), model slots leak preventing
new models from loading after 4 iterations, and constants memory pools
leak reducing the available TVM DDR heap.

### Symbol Table Synchronization

The DLOAD symbol table is defined in `dsp/src/dyn_loader.c` (extern
declarations, name table, and address table for runtime resolution).
When building TVM-compiled modules as DLOAD-loadable ELFs, the module's
linker script must import matching symbol names so that DLOAD can resolve
them at load time.

## IPC Details

### RPMessage Configuration

| Parameter | Value |
|-----------|-------|
| Service Name | "rpmsg_chrdev" (announced) |
| Endpoint | 20 |
| Max Message Size | 512 bytes |
| VRing Location | 0xAF800000 |

### Initialization Sequence

1. `RPMessage_waitForLinuxReady()` -- polls resource table until Linux
   initializes virtio vrings
2. `IpcNotify_registerClient(IPC_NOTIFY_CLIENT_ID_RP_MBOX, ...)` --
   register shutdown callback
3. `RPMessage_construct()` -- create endpoint 20
4. `RPMessage_announce("rpmsg_chrdev")` -- announce service to Linux
5. `compute_service_run()` -- blocking message loop

### Clean Shutdown

1. Linux sends `IPC_NOTIFY_RP_MBOX_SHUTDOWN` via mailbox
2. ISR callback calls `compute_service_stop()` which calls
   `RPMessage_unblock`
3. Service loop exits, sends `IPC_NOTIFY_RP_MBOX_SHUTDOWN_ACK` from
   task context
4. Tears down RPMessage endpoint and drivers
5. Disables interrupts and halts with `IDLE` instruction

The ACK must be sent before `RPMessage_destruct()` because that
disrupts the virtio transport needed for the mailbox ACK to reach the
kernel.

### Device Discovery

Both the deploy script and host library discover hardware dynamically
by matching the device tree address `7e000000.dsp` in sysfs. The host
scans `/sys/class/rpmsg/rpmsg_ctrlN/device` paths for a symlink
resolving to this address. This approach is robust across reboots where
remoteproc/rpmsg indices may change.

## Shared Memory Printf

The DSP's `printf` output is redirected to a 64 KB region at the end
of the output buffer in shared DDR. This replaces the previous approach
of writing to the DebugP trace buffer (`trace0`), which was limited to
~2 KB and could not hold profile output for models with many layers
(e.g. CLISTA-DoA produces ~12 KB of profile data for 156 layers).

### Architecture

The `shm_printf` module (`dsp/src/shm_printf.c`) uses the TI C7000
compiler's `add_device()` RTS mechanism (Section 7.2.4 of the C7000
Compiler User's Guide) to register a custom I/O device named "shmout"
that writes to shared memory. At init time, `freopen()` redirects
stdout through this device, so all standard output functions (`printf`,
`fprintf(stdout, ...)`, `fputs`, `puts`) write to the shared buffer.

Two output paths coexist:

1. **Direct path** (`shm_printf`): The DLOAD symbol alias maps the
   loaded module's `printf` calls directly to `shm_printf()`, which
   does `vsnprintf` into the buffer. This is the fast path that avoids
   FILE* overhead for the generated model code.

2. **Device driver path** (`SHM_write`): The `add_device` write
   callback handles any output that goes through the stdio FILE*
   machinery (e.g. `fprintf(stdout, ...)` from runtime code).

### Buffer Layout

The printf buffer occupies the last 64 KB of the output buffer
(`C7X_PRINTF_BUF_ADDR` = output buffer end - 64 KB):

```
Offset  Size    Field
0       4       magic (0x50524E54 = "PRNT")
4       4       wr_index (bytes written since last reset)
8       4       buf_size (usable text area = 64K - 16)
12      4       reserved
16      ...     text data
```

### Data Flow

No RPMsg is sent per printf call. The flow during inference is:

1. **Before inference**: `shm_printf_reset()` sets `wr_index = 0`
2. **During inference**: DSP `printf` calls write directly to the
   SHM buffer via `memcpy` -- no IPC, no RPMsg, no synchronization
3. **After inference**: `shm_printf_finish()` calls `CacheP_wb()`
   on the header + written data, returns the byte count
4. **INFER response**: `resp->printf_size` carries the byte count
   back to the host in the single RPMsg response
5. **Host reads**: After the existing `sync_output_from_device()`
   (which does `DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ` on the
   entire dmabuf), the host reads `printf_size` bytes from the
   buffer starting at offset 16 (past the header) and writes them
   to stdout

Buffer overflow is handled by silent truncation -- if output exceeds
the 64 KB text area, excess data is dropped without error.

### Integration Points

- **Init**: `shm_printf_init()` called from `compute_service_init()`
  after DLOAD and TVM model manager initialization
- **Reset**: `shm_printf_reset()` called in `handle_infer()` before
  `cg_main_dsp()` entry
- **Finish**: `shm_printf_finish()` called after output extraction,
  return value stored in `resp->printf_size`
- **Symbol alias**: `dyn_loader.c` maps `printf` to `shm_printf` so
  loaded modules' printf calls go directly to the shared buffer
- **Host read**: `c7x_client_infer()` reads and displays the printf
  data after extracting output tensors

## Build System Internals

### DSP Firmware

CMake-based cross-compilation using the TI CGT C7000 toolchain. The
toolchain file (`cmake/toolchain-c7000.cmake`) sets paths to the MCU+
SDK, CGT compiler, and SysConfig tool. The linker script
(`configs/linker_c75_freertos.cmd`) defines all memory sections
including the DLOAD code/data placement.

SDK Dependencies:
- TI MCU+ SDK 11_00_00_06
- TI CGT C7000 5.0.1 LTS
- TI SysConfig 1.26.0

Output: `dsp/build/c7x_compute.out` (~6.9 MB)

### Host Application

C++14 CMake build with `aarch64-linux-gnu-g++` cross-compilation.
Links against pthread. Uses RAII wrappers (UniqueFd, MmapRegion,
UniqueFile) for automatic resource cleanup. The build produces a
single `c7x_compute` binary that serves as both CLI tool and library
test harness. The public C API header (`c7x_compute_client.h`) has
`extern "C"` guards and remains usable from C code.

Dependencies:
- `gcc-aarch64-linux-gnu` and `g++-aarch64-linux-gnu`
  (Ubuntu/Debian: `apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`)

Output: `host/build/c7x_compute`

## Relationship to TVM

This firmware is the runtime target for TVM's C static backend when
compiling for C7x DSP. The workflow is:

1. **TVM compilation** (on development host): Compile a neural network
   model using the TVM C static backend, producing `lib0.c` and
   `weights.bin`
2. **C7x ELF build**: Compile `lib0.c` with the TI CGT C7000 compiler
   into a relocatable ELF (`lib0.out`) with weights embedded in the
   `.rodata.weights` section
3. **Deployment**: Copy `lib0.out` to the AM67A target
4. **Execution**: Use the host CLI or library to load the ELF onto the
   DSP via DLOAD, run inference, and retrieve results

The DSP-side tests in `tests/ti-dsp-runtime/dsp-tests/` automate this full
pipeline using pytest, including TVM compilation, C7x ELF building,
firmware deployment, and inference verification.

## File Organization

```
c7x-firmware/
+-- README.md                     # Usage and reference documentation
+-- design_doc.md                 # This file
+-- deploy-c7x.sh                # SSH-based firmware deployment to AM67A
+-- common/
|   +-- c7x_compute_protocol.h    # Shared protocol definitions (ARM + DSP)
+-- dsp/
|   +-- build.sh                  # DSP firmware build script
|   +-- CMakeLists.txt
|   +-- cmake/
|   |   +-- toolchain-c7000.cmake # C7x cross-compilation toolchain
|   +-- configs/
|   |   +-- c75ss0.syscfg         # SysConfig (IPC, MMU, 16 regions)
|   |   +-- linker_c75_freertos.cmd
|   +-- src/
|       +-- main.c                # FreeRTOS entry, IPC init, shutdown
|       +-- compute_service.c     # RPMessage handler, message dispatch
|       +-- compute_service.h
|       +-- dyn_loader.c          # DLOAD wrapper, symbol table, DLIF callbacks
|       +-- dyn_loader.h
|       +-- tvm_model.c           # TVM model/constants manager
|       +-- tvm_model.h
|       +-- shm_printf.c          # SHM printf: add_device driver + stdout redirect
|       +-- shm_printf.h
|       +-- dload/                # TI DLOAD dynamic linker source
|           +-- DLOAD/            #   ELF parser, segment loader
|           +-- DLOAD_API/        #   Public API header
|           +-- DLOAD_SYM/        #   Symbol table implementation
|           +-- C70_DLOAD_DYN/    #   C7x dynamic linking
|           +-- C70_DLOAD_REL/    #   C7x relocation handling
+-- host/
|   +-- build.sh                  # Host application build script
|   +-- CMakeLists.txt
|   +-- include/
|   |   +-- c7x_compute_client.h  # Client library API
|   +-- src/
|       +-- raii.h                 # RAII wrappers (UniqueFd, MmapRegion, UniqueFile)
|       +-- c7x_compute_client.cpp # Client library implementation (C++14)
|       +-- rpmsg_wrapper.cpp      # rpmsg_ctrl discovery + endpoint mgmt (C++14)
|       +-- rpmsg_wrapper.h
|       +-- c7x_compute_cli.cpp    # CLI tool (C++14)
+-- test/
    +-- test_dynmod.sh            # Automated hardware test suite (19 tests)
```
