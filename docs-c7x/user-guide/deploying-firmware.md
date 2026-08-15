# Deploying Firmware

Host-DSP compute service for TI J722S/AM67A that enables Linux applications
to offload data processing and ML inference to the C7x DSP via RPMessage IPC
and shared DDR memory. Includes a dynamic module loader (DLOAD) for loading
and executing TVM-compiled C7x ELF modules at runtime without reflashing
firmware. See
[Firmware Architecture](../contributor-guide/firmware/architecture.md) and
[Firmware Design Deep-Dive](../contributor-guide/firmware/design-deep-dive.md)
for the system architecture, protocol specification, memory layout, and
DLOAD internals.

| Feature | Notes |
|---------|-------|
| `ping` | Version + uptime, ~90ms Linux ready time |
| `status` | Version, uptime, job counts |
| `load` (DLOAD) | ELF parse, relocate, 116 exported symbols |
| `infer` | `cg_main_dsp` in loaded module, cycle count returned |
| `unload` | module memory, reset pools for back-to-back cycles |
| `run` | load+infer+unload in single command, JSON output |

!!! note "CLI scope"
    `c7x_compute` is the test harness's command-line front end, not an
    end-user API. Application code should use the `C7xVirtualMachine`
    (Python) or `c7x::Module` (C++) API instead (see
    [Python / C++ API Reference](python-api.md)), which link directly against
    `libc7x_arm_runtime.so`. Of the commands above, only `ping`, `status`,
    and `trace` are useful for manually checking a deployment; the rest
    exist to support the automated test script
    (`test/test_dynmod.sh` -- see
    [Verifying Your Deployment](../contributor-guide/testing/verifying-deployment.md))
    and the pytest suite.

## Building the DSP Firmware

Refer to `src/runtime/ti_dsp/build_all.sh` for commands to build the firmware.

## Building and Deploying the ARM Host Client

For the `C7xVirtualMachine` (Python) / `c7x::Module` (C++) API reference, see
[Python / C++ API Reference](python-api.md). This section covers building and
deploying `libc7x_arm_runtime.so` (which includes the `c7x_compute` CLI used
throughout this page) -- see
[Architecture Overview](../contributor-guide/architecture-overview.md) for
how this fits into the overall compile/deploy pipeline.

The Arm shared library is cross-compiled for aarch64 on the dev host.

Refer to `src/runtime/ti_dsp/build_all.sh` for commands to build the Arm shared library.

`--board <j722s-evm|beagley-ai>` is required (`--ddr <4gb|8gb>` stays
optional, default per-board): besides build-dir naming consistency with
`build_runtime.sh` and `dsp/build.sh`, `--board` also picks the `deploy`
subcommand's default SSH host (`beagley-ai` -> `beagley-ai`, else
`am67a`). `CROSS_COMPILE` (default `aarch64-linux-gnu-`) is still
configurable via an environment variable; there's no `BOARD_HOSTNAME`
override for the deploy host — add an SSH-config alias if your board is
reachable under a different name.

Requires `gcc-aarch64-linux-gnu` and `g++-aarch64-linux-gnu` packages
(on Ubuntu/Debian: `apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu`).

See [Verifying Your
Deployment](../contributor-guide/testing/verifying-deployment.md) for the
standalone C++ test binary that exercises this API end-to-end against
live DSP firmware.

## Deployment

!!! note "Board hostname"
    All deployment commands below are run from the **Linux development host**.

    `--board` is required: the deploy script SSHs into the hostname it maps to
    (`beagley-ai` -> `beagley-ai`, else `am67a`) and uses remoteproc to manage
    the DSP firmware. Add an SSH-config alias if your board answers to a
    different name.

```bash
# Deploy firmware (stop -> copy -> start -> verify)
./deploy-c7x.sh --board beagley-ai dsp/build/c7x_compute.out

# Deploy with trace buffer dump
./deploy-c7x.sh --board beagley-ai dsp/build/c7x_compute.out --trace

# Check status
./deploy-c7x.sh --board beagley-ai --status

# Stop DSP cleanly
./deploy-c7x.sh --board beagley-ai --stop

# Deploy host CLI to the board
cd arm && ./build.sh --board beagley-ai deploy
```

The deploy script and host CLI both discover hardware by matching the device
tree address `7e000000.dsp` in sysfs, so they work correctly even if
remoteproc/rpmsg indices change across reboots.

Refer to `src/runtime/ti_dsp/validate_all.sh` for an example of using the deploy script.

## Usage

All commands below are run **on the board** (via SSH or local
terminal) and require root access (`/dev/mem` and `/dev/rpmsg*`).

```bash
# Test connectivity
c7x_compute ping

# Get service status
c7x_compute status

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
# From pytest:
pytest tests/ti-dsp-runtime/dsp-tests/test_clista_dsp.py \
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
uint64_t cycles;
c7x_client_infer(client, handle, model_id,
                 &input, 1, &output, &num_outputs, &cycles);

c7x_client_dyn_unload(client, handle);
c7x_client_close(client);
```

## Testing

See [Verifying Your
Deployment](../contributor-guide/testing/verifying-deployment.md) for the
firmware's own hardware test suite (`test/test_dynmod.sh`), and
[DSP Test Suite](../contributor-guide/testing/dsp-suite.md) for the
pytest-based suite that automates the full TVM compilation, C7x ELF
build, firmware deployment, and inference verification pipeline
(`pytest tests/ti-dsp-runtime/dsp-tests/ -v --dsp-mode=c7x_dload`).

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
3. Check that module's imported symbols are in the firmware's export table (~116 symbols)

### DMA-BUF exhaustion
If repeated c7x_compute invocations fail with DMA-BUF errors, the host
may be leaking DMA-BUF attachments. Ensure:
1. The host CLI discovers the correct remoteproc index (matching
   `7e000000.dsp`), not a hardcoded `/dev/remoteproc0`
2. DMA-BUF is allocated/freed by the host (`c7x_client_open/close`),
   not by UDMA init/deinit on the DSP.  The DSP-side UDMA driver
   manages DRU channels, not DMA-BUF CMA allocations.

### Recovery

If the C7x becomes unresponsive:

1. **Remoteproc restart**: `./deploy-c7x.sh --board beagley-ai --stop && ./deploy-c7x.sh --board beagley-ai --start`
2. **Reboot board**: `ssh root@beagley-ai reboot`
3. **Power cycle** the board.
