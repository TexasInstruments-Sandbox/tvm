# Firmware Architecture

Host-DSP compute service for TI J722S/AM67A that enables Linux applications
to offload data processing and ML inference to the C7x DSP via RPMessage IPC
and shared DDR memory. Includes a dynamic module loader (DLOAD) for loading
and executing TVM-compiled C7x ELF modules at runtime without reflashing
firmware. Part of the TVM DSP runtime (`src/runtime/ti_dsp/`).

The host communicates with the DSP over a binary message protocol defined
in `common/c7x_compute_protocol.h`. Data is transferred through a 512 MB
shared DDR buffer allocated from a DMA heap carveout. See
[Firmware Design Deep-Dive](design-deep-dive.md) for the system
architecture, protocol specification, memory layout, and DLOAD internals.
For build/deploy/usage instructions, see
[Deploying Firmware](../../user-guide/deploying-firmware.md).

## Architecture Notes

### Memory Layout

- 512 MB shared DDR from DMA heap carveout (`vision_apps_shared-memories`)
  - 468 MB staging buffer (model ELF + weights, up to ~47 MB for ResNet-18)
  - 12 MB KV cache region (persistent across inferences, used when
    `C7X_INFER_FLAG_KV_RESIDENT` is set)
  - 32 MB result buffer (inference results + 64 KB printf buffer at end)
- 352 MiB cacheable DDR pool at 0x102000000 for TVM inference workspace
  and DLOAD segment allocation (unified from formerly separate heap/scratch)
- L2 SRAM: 1.25 MB (L2RAM_C7x_1_MAIN per TRM) for scratch/fast buffers

### Dynamic Module Loading (DLOAD)

TI's C70 dynamic ELF loader handles relocation and symbol resolution at
runtime. The firmware exports ~116 symbols to loaded modules (TVM runtime
API, platform functions, DMA operations, C7x kernels, MMALIB wrappers).
Key design decisions:

- DLOAD segment allocation uses `tvm_dsp_alloc`/`tvm_dsp_free` from the
  TVM DDR pool (not `memalign`) so DLOAD and inference share the same
  memory pool
- Weights can be embedded in the DLOAD module (auto-detected via
  `_binary_weights_bin_*` symbols) or loaded separately via `model-load`
- Memory pools are reset on unload to avoid fragmentation across
  back-to-back load cycles
- Module unload sequence: TIDL bridge cleanup (if present), register
  file cleanup, unload constants, then free ELF segments (ordering
  matters because the register file lives in the module's .bss and
  TIDL instances must release DMA channels before pool reset)

### UDMA Subsystem

Async DMA via DmaUtilsAutoInc3d with standalone UDMA driver in DRU
direct TR mode. The standalone UDMA driver is compiled from MCU+ SDK
source (not the prebuilt `dmautils.lib`) because the prebuilt version
uses the full UDMA driver which accesses NAVSS registers not available
in remoteproc mode.

- CLEC events 128-143 mapped to C7x events 32-47 at firmware boot
  (required for DmaUtilsAutoInc3d completion polling)
- UDMA driver (`g_udma_drv`) initialized once at boot and **shared**
  between TVM DMA tiling and TIDL.  TIDL obtains the handle via
  `appUdmaGetObj()` -> `tvm_dsp_dma_get_udma_handle()` and allocates
  its own channels from the same driver during `algInit`.
- DmaUtils context and TR memory use **static buffers** (not pool-
  allocated) so they survive `tvm_dsp_reset_pools()` between module
  load/unload cycles.
- `tvm_dsp_dma_deinit()` is NOT called at module unload.  TIDL's
  `algFree` releases its channels from the shared driver, leaving
  internal state inconsistent for `Udma_deinit`.  This matches the
  neo-tvm/PSDK approach of init-once-never-deinit.

### IPC and Host Communication

- RPMessage over virtio vrings (Linux remoteproc framework)
- Host CLI discovers correct remoteproc/rpmsg indices dynamically by
  matching device tree address `7e000000.dsp` in sysfs symlinks
- rpmsg endpoint discovery uses binary search for high device indices
  (TI rpmsg_char assigns monotonically increasing minor numbers)
- 64-bit TSC cycle counter for inference timing (32-bit wraps at ~4.3s)
- Host uses DMA-BUF with `RPROC_IOC_DMA_BUF_ATTACH` for shared memory;
  the rproc fd must stay open for the lifetime of the mapping
- `c7x_compute run` performs load+infer+unload atomically in a single
  RPMessage/DMA connection, returning JSON for deterministic parsing

## Change History

Moved from `src/runtime/ti_dsp/firmware/c7x/` to
`src/runtime/ti_dsp/firmware/c7x/` for co-location with the TVM DSP
runtime it depends on. Key milestones from original development:

- **Initial framework**: FreeRTOS compute service with RPMessage IPC,
  shared DDR data plane, and basic compute kernels (copy, scale, invert)
- **IPC fixes**: RPMessage_waitForLinuxReady, rpmsg_chrdev announce,
  service loop in main task context (avoiding TaskSupport_setupTaskStack
  assertion on C7x)
- **DLOAD integration**: TI C70 dynamic ELF loader with 61 firmware
  symbol exports, segment allocation, cache coherency on loaded code
- **DMA heap shared memory**: /dev/mem replaced with DMA heap carveout
  (vision_apps_shared-memories) with proper dmabuf cache sync
- **Unified DDR pool**: Merged separate heap/scratch into 128 MB
  contiguous cacheable region, DLOAD allocations routed through
  tvm_dsp_alloc
- **512 MB shared buffer**: Expanded for large models (ResNet-18
  at ~47 MB ELF with embedded weights)
- **Module lifecycle**: Proper cleanup ordering (register file ->
  constants -> ELF segments), pool reset on unload for back-to-back
  cycles
- **Host C++ rewrite**: C99 host code converted to C++14 with RAII
  wrappers for fd, mmap, rpmsg resources
- **SHM printf**: DSP printf redirected to 64 KB shared memory buffer
  via TI RTS add_device(), replacing ~2 KB DebugP trace buffer
- **Composite run command**: Single-shot load+infer+unload with JSON
  output, eliminating multi-SSH-call fragility
- **64-bit cycle counter**: Direct __TSC reads replacing
  CycleCounterP_getCount32 (which wraps at ~4.3s)
- **Robust device discovery**: Binary search for rpmsg endpoint
  indices, sysfs-based remoteproc matching
- **UDMA subsystem**: DmaUtilsAutoInc3d with standalone UDMA/DRU,
  per-module DMA lifecycle, L2 symbol exports
- **DLOAD in-place rodata**: Read-only ELF segments (weights, TIDL
  artifacts) mapped in-place from the staging buffer, eliminating
  DDR pool copies (~13 MB saved for ResNet-18 TIDL)

(Note: the "Unified DDR pool" milestone above describes the pool size
*at the time of that merge* -- 128 MB was correct then; the pool was
later extended to its current 352 MiB, see "Memory Layout" above.)

## Per-buffer dmabufs (done)

The 512 MB shared DDR carveout was originally one dmabuf, synced whole on
every inference regardless of how much data actually moved (`struct
dma_buf_sync` has no offset/length, so sync granularity is allocation
granularity). It's now split into independently-allocated, independently-synced
dmabufs: `client->staging` (ELF + weights only, still fixed at
`C7X_STAGING_ADDR`/468 MB since `gen_pool_first_fit` packs the first
allocation at the pool base), `client->printf_buf` (64 KB, rebindable via
`C7X_MSG_SET_PRINTF_BUF`), and `client->input_buf`/`output_buf` (sized per
session from the compiled module's `tvm_dsp_io_meta`, or
`c7x_client_reserve_io()` for a module with no table). `C7X_KV_ADDR` stays
fixed too, folded into `client->staging`'s 480 MB reservation.

See [ARM Host Client Internals](arm-client-internals.md) for the buffer
table and capacity-declaration semantics, and `c7x_compute_protocol.h` for
the wire format (`c7x_msg_infer`'s `input_dsp_addr`/`result_dsp_addr` fields
replace the old fixed `C7X_RESULT_ADDR` constant).

**Known limitation carried over from this change:** the firmware tracks the
current session's `output_buf`/`printf_buf` base as process-global state,
not per-connection -- a second concurrently-connected client would silently
redirect the first's buffers. Acceptable today (one DSP core, access already
serialised), but worth revisiting if that assumption changes.
