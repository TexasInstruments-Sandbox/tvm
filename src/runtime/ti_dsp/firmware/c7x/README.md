# C7x Firmware

Host-DSP compute service for TI J722S/AM67A that enables Linux applications
to offload data processing and ML inference to the C7x DSP via RPMessage IPC
and shared DDR memory. Includes a dynamic module loader (DLOAD) for loading
and executing TVM-compiled C7x ELF modules at runtime without reflashing
firmware. Part of the TVM DSP runtime (`src/runtime/ti_dsp/`).

The host communicates with the DSP over a binary message protocol defined
in `common/c7x_compute_protocol.h`. Data is transferred through a 512 MB
shared DDR buffer allocated from a DMA heap carveout. See
[design_doc.md](design_doc.md) for the system architecture, protocol
specification, memory layout, and DLOAD internals.

## Status

Verified end-to-end on AM67A (J722S) with MCU+ SDK 11_00_00_06:

| Feature | Status | Notes |
|---------|--------|-------|
| `ping` | Working | Version + uptime, ~90ms Linux ready time |
| `status` | Working | Version, uptime, job counts |
| `load` (DLOAD) | Working | ELF parse, relocate, 61 exported symbols |
| `infer` | Working | Call `cg_main_dsp` in loaded module, cycle count returned |
| `unload` | Working | Free module memory, reset pools for back-to-back cycles |
| `run` | Working | Composite load+infer+unload in single command, JSON output |
| SHM printf | Working | DSP printf output via shared memory, 64 KB buffer |
| EDMA DMA | Working | Async DMA via DmaUtilsAutoInc3d with standalone UDMA/DRU |
| Clean shutdown | Working | `remoteproc stop` completes without timeout |
| Device discovery | Working | Robust across reboot/stop-start cycles |

Automated test script: `test/test_dynmod.sh` (19/19 tests pass).

## Building

### DSP Firmware

```bash
cd src/runtime/ti_dsp/firmware/c7x/dsp
./build.sh
# Output: build/c7x_compute.out (~6.9 MB with DLOAD + TVM runtime)
```

Requires:
- TI MCU+ SDK 11_00_00_06 (path set in cmake/toolchain-c7000.cmake)
- TI CGT C7000 5.0.1 LTS
- TI SysConfig 1.26.0

The firmware CMakeLists.txt references the TVM DSP runtime library
(`libtvm_dsp_runtime_c7x.a`) via relative path. Build the runtime
first:

```bash
cd src/runtime/ti_dsp
mkdir -p build-c7x && cd build-c7x
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-j722s-c7x.cmake ..
cmake --build .
```

### Host Application

```bash
cd src/runtime/ti_dsp/firmware/c7x/host
./build.sh                  # Cross-compile with aarch64-linux-gnu-g++
```

Requires:
- `gcc-aarch64-linux-gnu` and `g++-aarch64-linux-gnu` packages
  (on Ubuntu/Debian: `apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`)

## Deployment

All deployment commands below are run from the **Linux development host**.
The deploy script SSHs into the AM67A target (default hostname: `am67a`,
override with `AM67A_TARGET` env var) and uses remoteproc to manage the
DSP firmware.

```bash
# Deploy firmware (stop -> copy -> start -> verify)
./deploy-c7x.sh dsp/build/c7x_compute.out

# Deploy with trace buffer dump
./deploy-c7x.sh dsp/build/c7x_compute.out --trace

# Check status
./deploy-c7x.sh --status

# Stop DSP cleanly
./deploy-c7x.sh --stop

# Deploy host CLI to the AM67A board
cd host && ./build.sh deploy
```

The deploy script and host CLI both discover hardware by matching the device
tree address `7e000000.dsp` in sysfs, so they work correctly even if
remoteproc/rpmsg indices change across reboots.

## Usage

All commands below are run **on the AM67A board** (via SSH or local
terminal) and require root access (`/dev/mem` and `/dev/rpmsg*`).

```bash
# Test connectivity
c7x_compute ping

# Get service status
c7x_compute status

# Single-shot run (load + infer + unload, JSON output)
c7x_compute run /path/to/module.out --input in.bin --output out.bin --dtype int8
# Returns: {"status":"ok","cycles":N,"num_outputs":1,"outputs":[...]}

# Load a dynamic module
c7x_compute load /path/to/module.out
# Returns: "Loaded module, handle=1"

# Run inference on loaded module
c7x_compute infer <handle> <model_id> --input in.bin --output out.bin --dtype int8

# Unload module
c7x_compute unload <handle>

# Manage TVM model constants
c7x_compute model-load /path/to/constants.bin
c7x_compute model-unload <model_id>

# View DSP trace buffer
c7x_compute trace
```

### DSP Printf Output

When a TVM model is compiled with `-profile-layers`, the generated code
calls `printf` during inference to emit per-layer cycle counts. On the
DSP, `printf` is redirected to a 64 KB shared memory buffer (the last
64 KB of the output buffer) via the TI RTS `add_device()` mechanism.
The host CLI reads and displays this output after each inference
completes -- no special flags or trace buffer polling required.

```bash
# Layer profile output appears automatically in inference stdout
c7x_compute infer <handle> <model_id> --input in.bin --output out.bin

# From pytest:
pytest tvm-relax-tests/dsp-tests/test_clista_dsp.py \
    -v --dsp-mode=c7x_dload --use-cpp-api --profile-layers
```

The DSP's DebugP trace buffer (`trace0`) is still available for
firmware-level debug messages via `c7x_compute trace`.

### C Library API

```c
#include "c7x_compute_client.h"

c7x_client_t *client = c7x_client_open();

// Dynamic module loading and inference
uint32_t handle;
c7x_client_dyn_load(client, "lib0.out", &handle);

c7x_tensor_desc_t input = { .data = my_data, .data_size = size, ... };
c7x_tensor_desc_t output;
int num_outputs;
uint32_t cycles;
c7x_client_infer(client, handle, model_id,
                 &input, 1, &output, &num_outputs, &cycles);

c7x_client_dyn_unload(client, handle);
c7x_client_close(client);
```

## Testing

### Hardware Test Suite

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

### DSP Model Tests (pytest)

The `tvm-relax-tests/dsp-tests/` directory contains pytest-based tests
that automate the full TVM compilation, C7x ELF build, firmware deployment,
and inference verification pipeline:

```bash
cd $TVM_HOME
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Run on C7x hardware
pytest tvm-relax-tests/dsp-tests/ -v --dsp-mode=c7x_dload
```

## Architecture Notes

### Memory Layout

- 512 MB shared DDR from DMA heap carveout (`vision_apps_shared-memories`)
  - 504 MB input buffer (model ELF + weights, up to ~47 MB for ResNet-18)
  - 8 MB output buffer (inference results + 64 KB printf buffer at end)
- 128 MB cacheable DDR pool at 0x108000000 for TVM inference workspace
  and DLOAD segment allocation (unified from formerly separate heap/scratch)
- L2 SRAM: 1.25 MB (L2RAM_C7x_1_MAIN per TRM) for scratch/fast buffers

### Dynamic Module Loading (DLOAD)

TI's C70 dynamic ELF loader handles relocation and symbol resolution at
runtime. The firmware exports ~61 symbols to loaded modules (TVM runtime
API, platform functions, DMA operations). Key design decisions:

- DLOAD segment allocation uses `tvm_dsp_alloc`/`tvm_dsp_free` from the
  TVM DDR pool (not `memalign`) so DLOAD and inference share the same
  memory pool
- Weights can be embedded in the DLOAD module (auto-detected via
  `_binary_weights_bin_*` symbols) or loaded separately via `model-load`
- Memory pools are reset on unload to avoid fragmentation across
  back-to-back load cycles
- Module unload sequence: cleanup register file, unload constants,
  cleanup constants state, then free ELF segments (ordering matters
  because register file lives in the module's .bss)

### EDMA Subsystem

Async DMA via DmaUtilsAutoInc3d with standalone UDMA driver in DRU
direct TR mode. The standalone UDMA driver is compiled from MCU+ SDK
source (not the prebuilt `dmautils.lib`) because the prebuilt version
uses the full UDMA driver which accesses NAVSS registers not available
in remoteproc mode.

- CLEC events 128-143 mapped to C7x events 32-47 (required for
  DmaUtilsAutoInc3d completion polling)
- DMA init/deinit is per-module-load (not at boot) to avoid DRU
  resource conflicts with host DMA-BUF cleanup between invocations

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

## Troubleshooting

### "Failed to open /dev/mem"
Run as root. The shared memory buffers are accessed via `/dev/mem` mmap.

### "Failed to find rpmsg_ctrl for device 7e000000.dsp"
1. Check DSP is running: `cat /sys/class/remoteproc/remoteproc*/state`
2. Find which remoteproc is the C7x: `ls -l /sys/class/remoteproc/remoteproc*/device | grep 7e000000`
3. Check rpmsg channel exists: `ls /sys/class/rpmsg/rpmsg_ctrl*`
4. The host CLI scans all `/sys/class/rpmsg/rpmsg_ctrlN/device` paths looking for `7e000000.dsp` in the resolved symlink.

### "Response timeout"
1. Check DSP trace: `c7x_compute trace` or `cat /sys/kernel/debug/remoteproc/remoteproc*/trace0`
2. Verify DSP shows "Service loop started, endpoint 20"
3. Verify DSP shows "Announced rpmsg_chrdev"

### "remoteproc stop" times out
The firmware must send `SHUTDOWN_ACK` before tearing down IPC. If using old
firmware that doesn't handle shutdown, a reboot is required. Deploy the
latest firmware which handles clean shutdown.

### Dynamic module load fails
1. Check trace for DLOAD errors: `c7x_compute trace | grep DLOAD`
2. Verify module was built with DLOAD-compatible flags (relocatable ELF, exported `cg_main_dsp`)
3. Check that module's imported symbols are in the firmware's export table (~61 symbols)

### DMA-BUF exhaustion
If repeated c7x_compute invocations fail with DMA-BUF errors, the host
may be leaking DMA-BUF attachments. Ensure:
1. The host CLI discovers the correct remoteproc index (matching
   `7e000000.dsp`), not a hardcoded `/dev/remoteproc0`
2. CLEC/UDMA init happens per-module-load, not at firmware boot

### Recovery

If the C7x becomes unresponsive:

1. **Remoteproc restart**: `./deploy-c7x.sh --stop && ./deploy-c7x.sh --start`
2. **Reboot board**: `ssh root@am67a reboot`
3. **Power cycle** (when SSH unreachable):
   ```bash
   wget --no-proxy --http-user=admin --http-password="" \
     "http://10.219.15.103/outlet.cgi?outlet=1&command=3"
   ```

## Change History

Moved from `tvm-relax-tests/c7x-firmware/` to
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
- **EDMA subsystem**: DmaUtilsAutoInc3d with standalone UDMA/DRU,
  per-module DMA lifecycle, L2 symbol exports
