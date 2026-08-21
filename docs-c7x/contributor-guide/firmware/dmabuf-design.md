# Per-Buffer dmabuf Design

Design record for the change that replaced the single 512 MB shared-memory
dmabuf with separate, independently-synced dmabufs. See
[ARM Host Client Internals](arm-client-internals.md) for the resulting
buffer table, API, and code pointers -- this page is the *why*, written
from the as-built result rather than the planning narrative it was derived
from.

## Problem

Every inference paid two `DMA_BUF_IOCTL_SYNC` calls (input, output) on the
single 512 MB dmabuf allocated at `c7x_client_open()`. `struct
dma_buf_sync` carries only a `flags` field -- no offset, no length -- so
each call did cache maintenance over the whole 512 MB regardless of
whether a few KB or a few MB actually moved. Measured on BeagleY-AI: ~77
us/MB, linear in buffer size (confirmed non-driver-no-op by measuring
64 KB / 1 MB / 16 MB / 512 MB separately), giving ~80 ms/inference
recoverable -- for a single conv2d call (91,604 DSP cycles) this was
roughly 900x the DSP compute it was paired with. The only lever available
is the size of the buffer being synced, since the ioctl has no
offset/length to exploit.

## Decisions

**Per-purpose dmabufs, each synced whole.** The kernel ioctl gives no other
choice -- sync granularity is allocation granularity, so buffer boundaries
have to be drawn by *sync lifetime*, not by logical tensor. This is also
the platform's own pattern: `vision_apps`' `app_utils` allocates one dmabuf
per buffer from the same carveout heap and syncs each one whole.

**Capacity is declared, never grown on demand.** Sizes come from the
compiled module's Relax `StructInfo`, computed at export time by
`tvm.contrib.c7x.io_meta` and embedded next to `weights.bin` as
`tvm_dsp_io_meta`. Growing `output_buf` on demand is not
implementable -- its required size is only known *after* the DSP writes --
and growing `input_buf` on demand would invalidate any `CreateInput()`
pointer already handed to the caller. So capacity is fixed per session
(bumped only by loading a bigger module, or by `c7x_client_reserve_io()`),
and exceeding it is always a clean, reported error: `-EFBIG` for input
(caught host-side before staging anything, since the host knows every
input's size before it stages), `C7X_STATUS_ERR_SIZE` + `result_required`
for output (only the firmware can detect it, mid-write, against the
transmitted capacity).

**Capacity locks on first *use*, not first *attempt*.** A rejected
undersized `CreateInput()`/inference must still allow the caller to call
`c7x_client_reserve_io()` again and retry -- that's the only supported path
back from "declared too small." So the lock is set where a buffer pointer
is actually handed to a caller that might hold onto it (a successful
`CreateInput()`, or an inference that completed and returned output
pointers), not merely on attempting one. An early design pass locked
capacity unconditionally at the top of the inference call and at the top of
the two buffer-pointer accessors; the accessor case was necessary (those
hand out a pointer the caller keeps), but the inference-entry case made
every `reserve_io()` retry fail with "already in use" even after a clean,
no-op-on-the-buffer failure. Locking is not needed for correctness anywhere
state wasn't actually exposed to the caller.

**ELF/weights staging is allocated first, at `staging + KV` size, and
untouched otherwise.** The carveout's heap allocator
(`gen_pool_first_fit`) packs from the low end and gives userspace no way to
request a specific offset. Allocating the combined 480 MB
(`C7X_STAGING_SIZE + C7X_KV_SIZE`) region first guarantees it lands at the
pool base, so the DSP's compile-time-fixed `C7X_STAGING_ADDR` and
`C7X_KV_ADDR` stay valid without any change to the DSP-side KV logic --
that reservation's only job is to hold the physical range against the
allocator. This buys minimal scope: no KV-per-request protocol field, no
per-model weights lifetime tracking (both deferred -- see "Out of scope"
below).

**Printf gets its own small dmabuf, rebound after boot.** DSP printf must
work before any host has connected (boot logs, early failures), so
`shm_printf_init()` still targets a fixed scratch address at boot. Once a
client opens and allocates its own `printf_buf`, a new
`C7X_MSG_SET_PRINTF_BUF` message rebinds the DSP's target to it --
`add_device()`/`freopen()` bind the device by name, not by address, so
only the buffer-pointer bookkeeping needs to be redone, not the whole
`shm_printf_init()` registration.

**Single client only, by construction, not by choice.** `output_buf` and
`printf_buf` no longer have fixed addresses, so the firmware has to be told
which one is current -- and it does that with process-global state
(`g_result_dsp_addr`/`g_result_size`, rebindable printf target), not
per-connection. A second concurrently-connected client would silently
redirect the first's buffers. Acceptable today (one DSP core, access
already serialized by the test harness's "never two `c7x_dload` sessions on
the same board" rule) but a real, documented narrowing from the old
fixed-address design, where two clients could coexist by accident.

## Measured result

Re-measured on BeagleY-AI, same conv2d scenario used to establish the
problem: per-inference sync drops from ~80 ms to ~20-25 us (input sync ~6
us for a 4 KB `input_buf`, output sync ~7-9 us for a 3.5 KB `output_buf`,
printf sync ~9 us and now conditional on `printf_size > 0`). The one
remaining whole-buffer sync (the 480 MB staging region) moved to module
load time, where it already was implicitly paid once per model rather than
once per inference.

## Out of scope (deferred)

- **KV-per-request addressing.** KV cache tensors still live at the fixed,
  compile-time `C7X_KV_ADDR` -- retiring that in favor of a per-request
  `kv_dsp_addr` field is a larger protocol change with no payoff for the
  per-inference sync cost this change targeted (KV residency only matters
  for `C7X_MSG_INFER_LARGE`, and the fixed address already works).
- **Exact-sized `model_buf`/`weights_buf` per model.** ELF and weights
  still share one 480 MB staging allocation, sized for the worst case
  rather than the loaded model. Only the ELF/weights *staging* region is
  unaffected by the per-inference sync cost (it syncs once per load, not
  once per inference), so splitting it carries cost (per-model lifetime
  tracking, `MAX_MODELS`-sized bookkeeping) without addressing the
  measured problem.
- **Symbolic entry shapes.** Every model in the current regression set has
  a fully static entry shape, so `tvm_dsp_io_meta` is always exact today.
  A model with a symbolic shape would need either an upper-bound flag in
  the metadata or no table at all (falling back to
  `c7x_client_reserve_io()`) -- not yet exercised by anything real.

## See also

- [ARM Host Client Internals](arm-client-internals.md) -- the resulting
  buffer table, `DmaBuf` API, and capacity-declaration mechanics.
- [DSP Memory Map](../dsp-runtime/memory-map.md) -- the standalone JTAG
  harness's memory layout (a separate, unrelated map from this carveout).
