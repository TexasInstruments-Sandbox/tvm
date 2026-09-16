---
name: firmware
description: "C7x firmware (c7x_compute) for AM67A/J722S and BeagleY-AI (also J722S, 4gb DDR variant). Use when working on: firmware build/deploy, DLOAD dynamic linker, RPMessage IPC, c7x_compute CLI (ping/status/load/infer/unload/trace), remoteproc lifecycle, DMA (EDMA/DRU/DmaUtilsAutoInc3d), DMA debug tracing, L2 SRAM allocation, DDR heap, memory layout (unified DLOAD/TVM DDR pool, 12 MB KV region), DSP printf/profiling internals, firmware symbol export table, deploy-c7x.sh, board selection (--board/--ddr), TIDL/MMALIB linkage (--tidl/--mmalib, tidl-kernels codegen attr), the BeagleY-AI Linux DTB overlay, or firmware recovery/troubleshooting. NOT for DSP runtime library code (see dsp-runtime) or build environment setup (see build)."
---

# C7x Firmware (c7x_compute)

FreeRTOS-based firmware running on the C7x DSP core of the J722S SoC.
Provides a host-DSP compute service for ML inference offload from ARM A53
Linux. Two boards share this firmware: **am67a** (J722S EVM, 8gb DDR,
default) and **beagley-ai** (BeagleY-AI, 4gb DDR) — selected via `--board`.

## Location

`src/runtime/ti_dsp/firmware/c7x/`

## Build and Deploy

```bash
# Build firmware (am67a/j722s-evm default, TIDL ON; PSDK_INSTALL_PATH has no
# default for either board -- see relax-c7x:build Prerequisites)
cd src/runtime/ti_dsp/firmware/c7x/dsp
PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06 \
C7X_MMA_TIDL_PATH=$HOME/ml/c7x-mma-tidl ./build.sh
# Output: build/c7x_compute.out

# BeagleY-AI — project convention is --tidl OFF --mmalib ON (see
# "TIDL/MMALIB Linkage" below); no C7X_MMA_TIDL_PATH needed since TIDL is off
PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03 \
TI_SYSCONFIG_ROOT=/opt/ti/sysconfig-1.28 \
./build.sh --board beagley-ai --tidl OFF --mmalib ON
# Output: build-beagley-ai-4gb-tidl-OFF-mmalib-ON/c7x_compute.out

# Deploy via remoteproc, then REBOOT the board (not --restart) to activate
# a newly-copied image -- see "Deploy Script Commands" below
cd src/runtime/ti_dsp/firmware/c7x
./deploy-c7x.sh dsp/build/c7x_compute.out                                                      # am67a
./deploy-c7x.sh --board beagley-ai dsp/build-beagley-ai-4gb-tidl-OFF-mmalib-ON/c7x_compute.out  # beagley-ai
ssh root@beagley-ai reboot

# Build ARM client
cd arm && ./build.sh && ./build.sh deploy
```

## Board Selection (`--board`/`--ddr`)

`--board <j722s-evm|beagley-ai>` (default `j722s-evm`) and `--ddr <8gb|4gb>`
(per-board default) resolve SDK paths and the shared-DMA carveout physical
base via `src/runtime/ti_dsp/cmake/boards.cmake` — see `relax-c7x:build`
for the full mechanism. The **only** memory-map difference between boards is
the shared carveout physical base: `0x900000000` (8gb) vs `0x8a0000000`
(4gb). Everything else in "Memory Layout" below — local heap base, region
sizes, DSP virtual addresses — is identical across boards.

**The runtime lib and firmware must be built with the same `--board`/`--ddr`**
(built separately, statically linked; a mismatch silently corrupts DMA, no
build error).

## TIDL/MMALIB Linkage (`--tidl`/`--mmalib`)

Orthogonal to `--board`/`--ddr`: `dsp/build.sh` also takes `--tidl <ON|OFF>`
(default `ON`) and `--mmalib <ON|OFF>` (default `OFF`, forced `ON` whenever
`--tidl ON` — TIDL's own algo lib has unresolved `MMALIB_CNN_*`/
`MMALIB_LINALG_*` symbols at link time). Full flag reference and rationale:
`relax-c7x:build`'s "TIDL/MMALIB linkage" section.

**Project convention: am67a/j722s-evm links TIDL; beagley-ai does not.**
beagley-ai firmware must be built `--tidl OFF --mmalib ON` — MMALIB direct
offload only, no TIDL subgraph offload on that board.

This changes what the firmware actually exports: `dyn_loader.c` guards both
the `c7x_int8_max_pool_tidl` `extern` declaration and its `SYM()`
export-table entry behind `#ifdef USE_TIDL_RUNTIME`, so a `--tidl OFF` build
genuinely has no such symbol (confirm with `nm7x c7x_compute.out | grep
max_pool`). The c_static_lib target's `tidl-kernels` attr defaults to `true`
independent of what any given firmware binary links — codegen has no way to
introspect the deployed firmware — so **every model compiled for beagley-ai
needs `-tidl-kernels=0` passed explicitly**, or `FuseQDQToTIDLMaxPool` emits
a call to `c7x_int8_max_pool_tidl` that resolves fine at compile time (it's
just a `call_extern` by name) and then fails only when DLOAD tries to load
it on the board. This mismatch class — a build-time flag with no
compile-time-checkable link to a separate codegen-time flag — is worth
remembering if a future kernel gets a similar TIDL-only/MMALIB-only split.

### BeagleY-AI also needs a Linux-side DTB overlay (not a CMake/build issue)

Discovered during first hardware bring-up: a fresh BeagleY-AI image may have
the overlay file already built and present at `/boot/dtb/ti/k3-j722s-edgeai-apps.dtbo`,
but **not enabled**. Symptom — firmware deploy succeeds but boot fails:
```
remoteproc remoteproc1: bad phdr da 0xad100000 mem ...
remoteproc remoteproc1: Failed to load program segments: -22
remoteproc remoteproc1: Boot failed: -22
```
Root cause: the stock kernel/DTB reserves only ~16 MB for the C7x DSP's own
boot-time carveout (IPC/code/data ELF segments, `0xad000000` range — this is
a *different* memory pool from the `0x8a0000000` shared-DMA carveout above),
but this firmware's linker command file needs ~64 MB there. The ARM host
also has no `/dev/dma_heap/carveout_vision_apps_shared-memories` device for
the same underlying reason.

Fix — check `/boot/uEnv.txt` on the board; if it's missing a `name_overlays=`
line, add one (back up the file first):
```
name_overlays=ti/k3-j722s-edgeai-apps.dtbo
```
(`dorprocboot=0` should already be present — that's correct, it tells U-Boot
to skip its own rproc init and let Linux's remoteproc drivers load firmware
at runtime instead.) Reboot. Confirm via `dmesg | grep -i "assigned reserved
memory node"` — should show `vision-apps-c71-dma-memory@ad000000`, and
`ls /dev/dma_heap/` should now list `carveout_vision_apps_shared-memories`.
This is a one-time, per-board-image fix, unrelated to anything `--board`/
`--ddr` control in this repo. See `docs/dsp/beagley_ai_enablement.md`
"Hardware bring-up results" for the full investigation.

## ARM-Side Binaries

The `arm/` directory produces two artifacts from the same source:

| Output | Purpose |
|--------|---------|
| `c7x_compute` | CLI executable (load/infer/unload commands) |
| `libc7x_arm_runtime.so` | Shared library implementing the C++ and Python offload APIs |

Both are deployed to AM67A via `./build.sh deploy` (installs to `/usr/local/{bin,lib,include}/`).

### CLI (c7x_compute)

```bash
c7x_compute ping                          # Verify connectivity
c7x_compute status                        # Firmware version + stats
c7x_compute load /path/to/lib0.out        # Load module → handle
c7x_compute infer <handle> <model_id> \
    --input in.bin --output out.bin        # Run inference
c7x_compute unload <handle>               # Free module memory
c7x_compute trace                         # View DSP debug output
```

### Arm Runtime Library (libc7x_arm_runtime.so)

Provides `C7xVirtualMachine` — a `relax.VirtualMachine`-compatible API for ARM-side inference with the DSP as a compute backend. No TVM runtime dependency on the target; only DLPack is required.

**Python** (`tvm.contrib.c7x.C7xVirtualMachine`):
```python
from tvm.contrib.c7x import C7xVirtualMachine
with C7xVirtualMachine("/models/resnet18.out") as vm:
    out = vm["main"](inp)          # copy-based (safe across calls)
    out_np = vm.run_nocopy(inp)    # zero-copy (valid until next call)
    staging = vm.create_input(shape, "float32")  # zero-copy input
```

**C++** (`c7x::Module` in `c7x_runtime.h`):
```cpp
auto vm = c7x::Module::Load("/models/resnet18.out");
auto out = vm.Run(&input_dl_tensor);
DLTensor* inp = vm.CreateInput(shape, ndim, dtype);  // staging DDR
```

Key properties: `vm.last_cycles` (DSP TSC from last inference), `vm.is_loaded`.

Zero-copy strategy: outputs point into mmap'd result DDR; `create_input()` allocates inside the staging buffer so the IPC client skips memcpy when the pointer is already in `[staging_buf, staging_buf + staging_size)`.

Source: `src/runtime/ti_dsp/firmware/c7x/arm/src/c7x_runtime.cc`
Python wrapper: `python/tvm/contrib/c7x/c7x_runtime.py`
Full API docs: `docs/dsp/tvm_arm_offload.md`

## DLOAD Dynamic Linker

Runtime ELF loader enabling hot-swap model deployment without firmware rebuild:
- Parses C7x ELF with dynamic relocation entries
- Allocates module segments in the DDR local heap (see Memory Layout — size has
  changed more than once, always check `linker_c75_freertos.cmd` for the
  current `DDR_C7X_1_LOCAL_HEAP` region size, not this doc)
- Resolves all imported symbols against firmware export table (116 symbols)
- Applies C7x-specific relocations to bind to actual load address

## Memory Layout (J722S)

Two distinct DDR pools: a 512 MB shared carveout for host↔DSP data
exchange, and a separate DSP-local pool for DLOAD segments + TVM workspace
(not visible to the host).

**The DSP-local pool's size has drifted (128 MB → 256 MB → 352 MB as of
2026-07-27) as it got extended to reclaim previously-unclaimed-but-already
MMU-mapped address space.** Source of truth: `DDR_C7X_1_LOCAL_HEAP` in
`src/runtime/ti_dsp/firmware/c7x/dsp/configs/linker_c75_freertos.cmd`
(`ORIGIN`/`LENGTH`), cross-checked against `mmu_armv8_r13` in
`c75ss0.syscfg` in the same directory (must match exactly, or DMA address
translation in `tvm_dsp_dma.c`'s `virt_to_phys()` silently mistranslates
addresses outside its own hardcoded bounds — see DMA Subsystem below).
Numbers below are current as of the 352 MB extension; re-verify before
trusting them.

```
C7x Internal
┌───────────────────────┐
│  L2 SRAM (128 KB)     │  Fast tensor storage (≤64 KB allocs)
└───────────────────────┘

DDR shared carveout (512 MB DMA-BUF, shared host↔DSP)  0xC0000000
┌───────────────────────┐
│  Staging buffer       │  468 MB — ELF modules + weights + input tensors
│  (468 MB)             │
├───────────────────────┤
│  KV region (12 MB)    │  0xDD400000 — DSP-resident KV cache
├───────────────────────┤
│  Result buffer        │  32 MB — inference output tensors
│  (32 MB, incl. printf)│  Last 64 KB = DSP printf buffer
└───────────────────────┘

DDR local heap (352 MB, DSP-only, not shared with host)  0x102000000
┌───────────────────────┐
│  TVM DDR heap          │  DLOAD code/data segments + TVM workspace
│  (352 MB, unified)     │  tensors (single pool, not split)
└───────────────────────┘
```

| Region | Size | Purpose |
|--------|------|---------|
| L2 SRAM | 128 KB | Fast intermediate tensor storage |
| DDR staging buffer | 468 MB | ELF modules + weights + input tensors |
| DDR KV region | 12 MB @ 0xDD400000 | KV-resident cache (SmolLM) |
| DDR result buffer | 32 MB | Inference output tensors + 64 KB printf buffer |
| DDR shared carveout (total) | 512 MB | Staging + KV + result (host↔DSP) |
| DDR local TVM/DLOAD heap | 352 MB (verify — see note above) | Unified pool: DLOAD segments + TVM workspace (DSP-only) |

## DMA Subsystem

| Component | Purpose |
|-----------|---------|
| EDMA/DRU | Hardware DMA engine, 16 channels total |
| `DmaUtilsAutoInc3d` | TI library for 2D/3D transfer descriptors |
| `tvm_dsp_dma.c` | TVM runtime DMA wrapper (virt-to-phys) |
| UDMA instance ID 5 | C7x-local UDMA (not main NAVSS) |

DMA is used for L2 prefetch (staging DDR→L2 before compute). Per-module init/deinit in firmware compute service.

`compute_service.c` calls `tvm_dsp_dma_init(N)` once at startup with a **fixed**
channel count (currently `N=1`), independent of `MAX_DMA_CHANNELS` (2) in
`tvm_dsp_dma.c`. Generated code calling `tvm_dsp_dma_copy`/`tvm_dsp_dma_wait`
with a `queue_id >= N` fails with a silent `-1` (see DMA Debug Tracing below —
this was investigated as a hypothesis for a hardware failure and ruled out for
that specific case by confirming via trace that all calls used `queue_id=0`,
but the mismatch between the init count and `MAX_DMA_CHANNELS` is still a real
footgun worth knowing about).

### DMA Debug Tracing

`tvm_dsp_dma.c`'s `DMA_TRACE(...)` macro (used on every copy/wait/prepare/
configure/trigger step) is a no-op unless `TVM_DMA_DEBUG` is defined, so DMA
activity is invisible by default — a silent DMA-layer failure looks identical
to a silent failure anywhere else in generated code. To see it:

1. Temporarily add `#define TVM_DMA_DEBUG 1` near the top of
   `src/runtime/ti_dsp/dma/tvm_dsp_dma.c` (before the `#ifdef TVM_DMA_DEBUG`
   block).
2. Rebuild the **runtime library** first (`tvm_dsp_dma.c` is part of
   `libtvm_dsp_runtime_c7x.a`, not the firmware itself):
   `cd src/runtime/ti_dsp && bash build_runtime.sh c7x`
3. Rebuild + redeploy the firmware as usual (`dsp/build.sh`, `deploy-c7x.sh`).
4. Read `deploy-c7x.sh --trace` — `DMA_TRACE` uses `DebugP_log`, so it shows up
   in the same remoteproc trace buffer as `[COMPUTE]`/`[DLOAD]` lines, with
   per-transfer detail: `dst`/`src`/`size`/`q` (queue id), physical addresses
   (`virt_to_phys()` output), block/tile decomposition, and each
   prepare/convert/configure/trigger/wait step.
5. **Revert the `#define` before shipping** — this is a diagnostic-only,
   verbose macro, not meant to run enabled in normal builds.

## IPC (RPMessage)

ARM↔C7x communication over shared DDR:
- Commands: load, infer, unload (binary messages)
- Data: input/output tensors via shared DDR buffers (zero-copy)
- Printf: firmware debug output to 64 KB shared buffer (last 64 KB of output buffer)

### DSP Printf/Profiling Pipeline — how it actually works

Non-obvious: `printf()` calls compiled into a **DLOAD module** (e.g. TVM's
codegen'd `TVMPrintLayerProfile`) do **not** go through libc stdio/stdout at
all. `dyn_loader.c` has `SYM_ALIAS("printf", shm_printf)` — the DLOAD import
for the literal symbol name `"printf"` resolves directly to `shm_printf()` in
`shm_printf.c`, a lightweight `vsnprintf`-into-a-shared-buffer function. The
`freopen("shmout:stdout", ...)`/`add_device()` machinery in
`shm_printf_init()` only redirects the **firmware's own** stdout (used by
`compute_service.c`'s own `printf`/`fprintf` calls, e.g. the `repeat > 1`
iteration header) — it is unrelated to how DLOAD-module printf calls resolve.

Full path for a DLOAD module's `printf()` call reaching the ARM host:
1. DLOAD module calls `printf(...)` → resolves to `shm_printf()` → writes
   into the shared printf buffer, advancing `g_hdr->wr_index`
   (`struct shm_printf_hdr` at the start of the buffer: magic/wr_index/
   buf_size/reserved, 16-byte header).
2. `compute_service.c` calls `shm_printf_finish()` (both on success, and on
   the `ret != 0` failure branch — this is intentional: `-profile-layers`
   records the in-flight layer's name before invoking it, so on failure this
   is often the only way to see which kernel call actually failed instead of
   a generic `-1`). This reads `g_hdr->wr_index`, does a `CacheP_wb()`, and
   the byte count becomes `resp->printf_size` in the IPC response message.
3. The ARM client (`c7x_compute_client.cpp`) calls `sync_output_from_device()`
   then, if `resp->printf_size > 0`, `fwrite()`s that many bytes from
   `client->result_buf` (at a fixed printf-region offset) to **stderr** (not
   stdout, so it doesn't interfere with the CLI's JSON stdout output).

**Debugging when profile/printf output goes missing on a failure path** (seen
zero bytes reach the client despite a model that should be profiling): add a
`shm_printf_debug_dump(tag)` helper to `shm_printf.c`/`.h` that
`DebugP_log()`s `g_hdr->magic`/`wr_index`/`buf_size` directly (bypasses the
suspect channel by using the trace buffer instead), and call it at each hop
above (before/after `shm_printf_reset()`, after `cg_main_dsp` returns, after
`print_profile()` runs) — this isolates which of the three hops is dropping
the data. In one investigation this proved hops 1-2 were fine (DSP-side
`resp->printf_size` was a correct, sane nonzero value) while the ARM client
still showed nothing — pointing at `sync_output_from_device()` in
`c7x_compute_client.cpp` as the likely culprit (worth checking whether its
sync range/size depends on `resp->num_outputs` or another field that's zeroed
on the failure path, rather than unconditionally covering the printf region).
Not yet confirmed/fixed as of this writing — see
`relax-c7x:testing` debugging reference for the fuller incident writeup.

Also note: `_tvm_layer_count`/`_tvm_layer_names[]` (the profiling globals in
codegen'd code) are plain statics, not `TVM_DSP_EXPORT`-marked — they are
**not** queryable via `dyn_loader_query_symbol()` from firmware code, unlike
`TVMPrintLayerProfile` itself (which is exported). Don't waste time trying to
peek at them directly from `compute_service.c`; go through the profile
printer or add trace calls inside the generated code's own call sites.

## Firmware Recovery

In order of escalation (add `--board beagley-ai` to the deploy-c7x.sh
commands, and swap the hostname, if not on am67a):
1. `./deploy-c7x.sh --stop && ./deploy-c7x.sh --start`
2. `ssh root@<board-host> reboot`
3. Network power cycle, same PDU, different outlet per board:
   - am67a: `wget --no-proxy --http-user=admin --http-password="" "http://10.219.15.103/outlet.cgi?outlet=1&command=3"`
   - beagley-ai: `wget --no-proxy --http-user=admin --http-password="" "http://10.219.15.103/outlet.cgi?outlet=2&command=3"`

**A genuinely wedged DSP looks recovered after step 2 but isn't.** SSH comes
back (network/Linux side is fine), but the DSP core itself never
re-attaches: `cat /sys/class/remoteproc/remoteproc*/state` for `7e000000.dsp`
reports `running` even though `/dev/rpmsg_ctrl*` never appears, and
`dmesg` shows repeated `k3_rproc_stop: timedout waiting for rproc completion
event` / `can't stop rproc: -16` -- step 1's `--stop` keeps failing with
`EBUSY` against this same state. Don't keep retrying steps 1-2 once you see
this signature; `c7x_compute status` hanging for minutes after a fresh
reboot (rather than connecting or failing fast) is the same tell. Go
straight to step 3 -- the outlet power cycle reliably clears it, since it
resets state a soft `reboot` can leave alone. This is a real DSP-side hang,
not a timing race in how long to wait after reboot before probing it.

## Symbol Export Table

The firmware exports 116 symbols for DLOAD modules to import (`dsp_syms[]`):
- ~90 TVM runtime + stdlib symbols (VM builtins, memory allocators, platform services, C7x kernels)
- 10 MMALIB wrapper symbols (when built with MMALIB)
- 16 TIDL support symbols: `appUdmaGetObj`, `appMemAlloc`, `appMemFree`, `getUDMADrvObjPtr`, L1/L2/L3 memory globals, cache/interrupt helpers

The 116 count is for a `--tidl ON` build. Some entries are conditional on
build flags — e.g. `c7x_int8_max_pool_tidl`'s `extern` + `SYM()` entry are
both wrapped in `#ifdef USE_TIDL_RUNTIME`, so a `--tidl OFF` build (the
beagley-ai convention — see "TIDL/MMALIB Linkage" above) exports fewer
symbols and codegen must be told via `-tidl-kernels=0`. Don't assume the
symbol table is identical across boards; check with `nm7x` if unsure.

Managed in: `src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c`

## How to Add a New Callable Function to the Firmware

A function defined in the DSP runtime (e.g., a new kernel in `src/runtime/ti_dsp/kernels/`)
or a new MMALIB wrapper must be exposed through **four files** before DLOAD modules can
call it. Missing any one causes an unresolved symbol linker error on `c7x_dload`.

### Step 1 — Implement and declare the function

- Write the C implementation in the appropriate source file (e.g., `kernels/tvm_int8_residual_add.c` for DSP runtime kernels, `mmalib/mmalib_wrappers.cpp` for MMALIB wrappers).
- Add the declaration to the corresponding header (e.g., `kernels/tvm_int8_residual_add.h`, `mmalib/mmalib_wrappers.h`).
- Rebuild the DSP runtime library **for the c7x target** (needed for firmware linking):
  ```bash
  export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
  export MCU_PLUS_SDK_PATH=/opt/ti/am67a/.../mcu_plus_sdk_j722s_11_00_00_12
  cd src/runtime/ti_dsp && bash build_runtime.sh c7x
  ```

### Step 2 — Export from the firmware (`dyn_loader.c`)

In `src/runtime/ti_dsp/firmware/c7x/dsp/src/dyn_loader.c`:
1. Add an `extern` declaration near similar functions.
2. Add `SYM(my_new_function)` to the `dsp_syms[]` array.

Example (for a new MMALIB wrapper):
```c
/* dyn_loader.c */
extern int32_t mmalib_my_new_kernel(void* a, void* b, void* c);

// ... in the SYM table:
    SYM(mmalib_my_new_kernel),
```

### Step 3 — Add a link-time stub (`dsp_syms.c`)

In `src/runtime/ti_dsp/dynmod/c7x_dynmod/dsp_syms.c`, add a stub that satisfies the
linker during DLOAD module build. The real function resolves at runtime from the firmware.
```c
__declspec(dllexport) int32_t mmalib_my_new_kernel() { return 0; }
```
Use the actual return type, not `void`, so the linker is satisfied without type errors.

### Step 4 — Declare as an import (`c7x_dynmod.cmd`)

In `src/runtime/ti_dsp/dynmod/c7x_dynmod/c7x_dynmod.cmd`, add an `--import` line so
the TI cl7x linker marks the symbol as an external reference to be resolved at load time:
```
--import=mmalib_my_new_kernel
```

### Step 5 — Rebuild firmware and redeploy

```bash
cd src/runtime/ti_dsp/firmware/c7x/dsp
PSDK_INSTALL_PATH=... C7X_MMA_TIDL_PATH=... MMALIB_PATH=... ./build.sh
cd .. && ./deploy-c7x.sh dsp/build/c7x_compute.out && ./deploy-c7x.sh --restart
```

### Checklist

| Step | File | What to add |
|------|------|-------------|
| 1 | `kernels/*.c` or `mmalib/mmalib_wrappers.cpp` | Implementation |
| 1 | Corresponding `.h` | Declaration |
| 1 | Run `build_runtime.sh c7x` | Rebuild runtime lib |
| 2 | `firmware/.../dyn_loader.c` | `extern` decl + `SYM(...)` |
| 3 | `dynmod/c7x_dynmod/dsp_syms.c` | `__declspec(dllexport) ret_type fn() { return 0; }` |
| 4 | `dynmod/c7x_dynmod/c7x_dynmod.cmd` | `--import=fn` |
| 5 | Rebuild + redeploy firmware | `build.sh` + `deploy-c7x.sh` |

> **Note on `c7x_host` mode:** Steps 2–5 are only required for `c7x_dload`.
> Host emulation uses the pre-built runtime library directly; only step 1
> (implement + rebuild `build_runtime.sh c7x_host`) is needed for `c7x_host` tests.

## Key Files

| File | Purpose |
|------|---------|
| `dsp/build.sh` | Firmware build script |
| `dsp/src/dyn_loader.c` | DLOAD symbol export table |
| `dsp/src/compute_service.c` | IPC command handler |
| `dsp/CMakeLists.txt` | Firmware build system |
| `arm/build.sh` | ARM client build + deploy script |
| `arm/src/c7x_runtime.cc` | c7x::Module C++ implementation |
| `arm/src/c7x_compute_client.cpp` | IPC client (rpmsg, staging/result DDR) |
| `arm/include/c7x_runtime.h` | C++ API header (DLPack-only dependency) |
| `python/tvm/contrib/c7x/` | Python C7xVirtualMachine wrapper |
| `deploy-c7x.sh` | Deployment script (remoteproc) |
| `design_doc.md` | Full architecture documentation |
| `docs/dsp/tvm_arm_offload.md` | Arm runtime API docs (Python + C++) |
| `test/test_dynmod.sh` | 19-test hardware integration suite |

## Deploy Script Commands

```bash
./deploy-c7x.sh dsp/build/c7x_compute.out   # Copy new firmware (am67a default)
./deploy-c7x.sh --status                     # Show remoteproc state
./deploy-c7x.sh --stop                       # Stop DSP
./deploy-c7x.sh --start                      # Start DSP
./deploy-c7x.sh --restart                    # Stop + start
./deploy-c7x.sh --trace                      # Show trace buffer

# Any of the above, targeting BeagleY-AI instead:
./deploy-c7x.sh --board beagley-ai --status
```

**A plain `deploy-c7x.sh <firmware>` only copies the file — it does not
restart anything.** It deliberately avoids calling `--stop`/`--start`
itself: right after the board's own autostart, the kernel returns EBUSY
while virtio vdev negotiation is still settling, with no reliable signal
for when that window has passed. The script prints "Firmware staged --
reboot the board to activate" and means it literally: run
`ssh root@<host> reboot`, wait ~30-60s for the network to come back plus a
few more seconds for SSH/services, then confirm with `c7x_compute status`
— uptime and jobs-completed both reset on a genuinely fresh boot, which is
how you tell the new image actually loaded rather than the old one still
running. `--restart` (stop+start) is for cycling the *same*
already-loaded image (see Firmware Recovery below), not for activating a
newly-deployed one.

`--board beagley-ai` defaults the deploy host to `beagley-ai`; `BOARD_HOSTNAME`
or `TARGET` env vars always override (e.g. if the board resolves to a raw IP
instead of a hostname).

Hardware discovery: matches device tree address `7e000000.dsp` in sysfs (works regardless of remoteproc index changes across reboots).

## Related Skills

- `relax-c7x:dsp-runtime` — Runtime library linked into firmware (116 exported symbols)
- `relax-c7x:build` — Full build chain including firmware
- `relax-c7x:dsp-ops` — Adding new kernel exports to the symbol table
- `relax-c7x:testing` — Hardware test rules and firmware recovery
