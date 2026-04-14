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

---

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

**ARM Client / Library** (`arm/`): User-space Linux C++14 application
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

---

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

Response types are `0x1000 | request_type`.

### Message Types

| Type | Code | Direction | Purpose |
|------|------|-----------|---------|
| PING | 0x0001 | Host -> DSP | Connectivity test |
| GET_STATUS | 0x0003 | Host -> DSP | Get service status |
| DYN_LOAD | 0x0010 | Host -> DSP | Load ELF module |
| DYN_UNLOAD | 0x0012 | Host -> DSP | Unload module |
| MODEL_LOAD | 0x0020 | Host -> DSP | Load weights/constants |
| INFER | 0x0021 | Host -> DSP | Run inference |
| INFER_LARGE | 0x0023 | Host -> DSP | Run inference with >4 inputs (descriptors in DDR) |
| MODEL_UNLOAD | 0x0022 | Host -> DSP | Unload model weights |

### INFER Message Structure

The INFER message carries tensor descriptors inline (up to 4 inputs):

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

For >4 inputs (e.g. KV-cache LLMs), descriptors are staged in DDR
and `C7X_MSG_INFER_LARGE` is used instead — the message carries only
`descs_addr` and `descs_size` pointers.

### Wire Format Examples

**DYN_LOAD request** (32 bytes):

```
Offset  Field           Value
  0     type            0x00000010  (C7X_MSG_DYN_LOAD)
  4     seq             <n>
  8     len             32
 12     status          0
 16     elf_size        <bytes>
 20     flags           0
 24-31  reserved        0
```

**DYN_LOAD_RESP** (32 bytes):

```
Offset  Field           Value
  0     type            0x00001010  (C7X_MSG_DYN_LOAD_RESP)
  4     seq             <n>  (matches request)
  8     len             32
 12     status          0   (C7X_STATUS_SUCCESS)
 16     module_handle   1   (opaque integer)
 20     text_size       0
 24     data_size       0
 28     reserved        0
```

**INFER request** (single input = 112 bytes):

```
Offset  Field               Value
  0     type                0x00000021  (C7X_MSG_INFER)
  4     seq                 <n>
  8     len                 112
 12     status              0
 16     module_handle       1
 20     model_id            0  (0 = use embedded weights)
 24     num_inputs          1
 28     flags               0
 32     inputs[0].data_addr 0xC0000000 + elf_size   (DSP virtual)
 40     inputs[0].data_size <bytes>
 48     inputs[0].ndim      4
 52     inputs[0].dtype_code 2  (kDLFloat)
 56     inputs[0].dtype_bits 32
 60     inputs[0].reserved  0
 64     inputs[0].shape[0]  1
 72     inputs[0].shape[1]  3
 80     inputs[0].shape[2]  224
 88     inputs[0].shape[3]  224
 96-111 (remaining shape)   0
```

**INFER_RESP** (single output = 152 bytes):

```
Offset  Field                   Value
  0     type                    0x00001021  (C7X_MSG_INFER_RESP)
  4     seq                     <n>  (matches request)
  8     len                     152
 12     status                  0   (C7X_STATUS_SUCCESS)
 16     return_value            0   (cg_main_dsp return value)
 20     cycles                  <64-bit TSC delta>   (8 bytes)
 28     num_outputs             1
 32     printf_size             0   (or N if -profile-layers was set)
 36     descs_addr              0   (0 = inline; non-zero = out-of-band)
 44     descs_size              0
 48     reserved                0
 52     outputs[0].data_addr    0xDE000000  (C7X_RESULT_ADDR)
 60     outputs[0].data_size    <bytes>
 68     outputs[0].ndim         2
 72     outputs[0].dtype_code   2   (kDLFloat)
 76     outputs[0].dtype_bits   32
 80     outputs[0].reserved     0
 84     outputs[0].shape[0]     1
 92     outputs[0].shape[1]     1000
 96-151 (remaining shape)       0
```

---

## Inference Flow — End-to-End Walk-Through

This section traces the exact code path for a single inference request —
from the ARM Linux application calling into the host client library, through
the RPMessage IPC boundary, across to the C7x DSP firmware, and back.

### Key Source Files

| File | Side | Role |
|------|------|------|
| `firmware/c7x/arm/include/c7x_compute_client.h` | ARM | Public C API |
| `firmware/c7x/arm/src/c7x_compute_client.cpp` | ARM | Client implementation |
| `firmware/c7x/arm/src/c7x_compute_cli.cpp` | ARM | `c7x_compute` CLI tool |
| `firmware/c7x/common/c7x_compute_protocol.h` | both | Shared message structs |
| `firmware/c7x/dsp/src/compute_service.c` | DSP | Service loop + handlers |
| `firmware/c7x/dsp/src/dyn_loader.c` | DSP | DLOAD ELF loader |
| `firmware/c7x/dsp/src/tvm_model.c` | DSP | Weights/constants manager |

---

### Phase 0: Connection Setup

#### ARM — `c7x_client_open()`
**Source:** `firmware/c7x/arm/src/c7x_compute_client.cpp:148`

1. **Open RPMessage channel** via `rpmsg_open(C7X_DEVICE_ADDR, C7X_SERVICE_ENDPOINT, C7X_SERVICE_NAME)`.
   - `C7X_DEVICE_ADDR = "7e000000.dsp"` — stable device-tree address.
   - `C7X_SERVICE_ENDPOINT = 20` — the well-known endpoint announced by firmware.

2. **Allocate shared DDR buffer** from the DMA heap carveout:

   ```c
   client->dma_heap_fd = open("/dev/dma_heap/carveout_vision_apps_shared-memories", ...);
   ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &heap_data);  // heap_data.len = C7X_SHARED_SIZE
   mapped = mmap(NULL, C7X_SHARED_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, dma_buf_fd, 0);
   client->staging_buf = mapped;                           // input side (offset 0)
   client->result_buf  = mapped + C7X_STAGING_SIZE;       // output side (offset 480 MB)
   ```

3. **Attach buffer to DSP** via remoteproc ioctl — creates the DMA mapping
   that makes the buffer visible to the C7x MMU:

   ```c
   int idx = find_remoteproc_index("7e000000.dsp");
   client->rproc_fd = open("/dev/remoteproc0", O_RDONLY);  // must stay open!
   ioctl(rproc_fd, RPROC_IOC_DMA_BUF_ATTACH, &phys_data);
   client->phys_addr = phys_data.phys;  // = C7X_SHARED_PHYS_BASE = 0x900000000
   ```
   The DSP sees this region at virtual address `0xC0000000` (static MMU mapping).

#### DSP — `compute_service_init()`
**Source:** `firmware/c7x/dsp/src/compute_service.c:1159`

The firmware was already started via `remoteproc`. On boot it:

1. Creates an RPMessage endpoint at `C7X_SERVICE_ENDPOINT = 20`.
2. Announces `"rpmsg_chrdev"` to Linux — creates `/dev/rpmsg*` character device.
3. Calls `dyn_loader_init()` to initialise the DLOAD ELF loader.
4. Calls `tvm_model_init()` to initialise the weights/constants manager.
5. Calls `shm_printf_init()` to redirect DSP `printf` to shared memory.
6. Sets `gServiceRunning = 1` and enters `compute_service_run()`.

See §IPC Details for RPMessage configuration parameters and shutdown sequence.

---

### Phase 1: Load ELF Module

#### ARM — `c7x_client_dyn_load()`
**Source:** `firmware/c7x/arm/src/c7x_compute_client.cpp:435`

```c
// 1. Read ELF file into staging_buf (shared DDR, visible to DSP)
stage_file(client, elf_file, &file_size);
sync_input_to_device(client);   // DMA_BUF_SYNC_END|SYNC_WRITE: flush ARM cache

// 2. Build and send IPC message
struct c7x_msg_dyn_load req = {
    .hdr.type = C7X_MSG_DYN_LOAD,
    .hdr.seq  = ++client->seq,
    .hdr.len  = sizeof(req),
    .elf_size = file_size,
};
send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));

// 3. Preserve in-place rodata: inputs must be staged after the ELF
*handle_out = resp.module_handle;
client->input_data_offset = file_size;
```

#### DSP — `handle_dyn_load()`
**Source:** `firmware/c7x/dsp/src/compute_service.c:143`

```c
// 1. Load ELF from staging buffer (phys 0x900000000 = DSP virt 0xC0000000)
dyn_loader_load(C7X_STAGING_ADDR, req->elf_size, &handle);
//   DLOAD: parse ELF, allocate segments in DDR heap,
//   apply C7x relocations, resolve 61 imported symbols.
//   NOTE: .rodata segments are mapped IN-PLACE from staging_buf.

// 2. Look up the TVM-generated entry point
dyn_loader_query_symbol(handle, "cg_main_dsp", &sym_addr);
g_cg_main_dsp = (cg_main_dsp_fn)(uintptr_t)sym_addr;

// 3. Check for embedded weights
dyn_loader_query_symbol(handle, "_binary_weights_bin_start", &ws_addr);
dyn_loader_query_symbol(handle, "_binary_weights_bin_size",  &wz_addr);
tvm_model_load_weights(ws_addr, *(uint32_t*)wz_addr, &g_embedded_model_id);

// 4. Save pool watermark for workspace reclaim after inference
tvm_dsp_save_infer_watermark();

send_response(C7X_MSG_DYN_LOAD_RESP, ...);
```

---

### Phase 2: Run Inference

#### ARM — `c7x_client_infer()`
**Source:** `firmware/c7x/arm/src/c7x_compute_client.cpp:691`

**Step A — Stage input tensor data:**

```c
// Inputs go AFTER the ELF to avoid corrupting DLOAD'd rodata
data_offset = client->input_data_offset;
for (int i = 0; i < num_inputs; i++) {
    memcpy(staging_buf + data_offset, inputs[i].data, inputs[i].data_size);
    data_offset += inputs[i].data_size;
}
sync_input_to_device(client);   // flush ARM cache → DDR
```

**Step B — Build tensor descriptors with DSP virtual addresses:**

```c
uint64_t cur_addr = C7X_STAGING_ADDR + client->input_data_offset;
for (int i = 0; i < num_inputs; i++) {
    desc_arr[i].data_addr = cur_addr;   // DSP sees this address
    desc_arr[i].data_size = inputs[i].data_size;
    // ... ndim, dtype, shape
    cur_addr += inputs[i].data_size;
}
```

**Step C — Send INFER (≤4 inputs inline) or INFER_LARGE (>4 inputs in DDR):**

```c
req->hdr.type      = C7X_MSG_INFER;
req->module_handle = module_handle;
req->model_id      = model_id;         // 0 → use embedded weights
req->num_inputs    = num_inputs;
for (int i = 0; i < num_inputs; i++)
    req->inputs[i] = desc_arr[i];
send_and_recv(client, req, req_size, resp, sizeof(resp_buf));
```

**Steps D-F — Back from send_and_recv():**

```c
sync_output_from_device(client);   // invalidate ARM cache (DMA_BUF_SYNC_READ)

// Convert DSP virtual addresses → ARM userspace pointers
for (int i = 0; i < resp->num_outputs; i++) {
    uint64_t offset = td_base[i].data_addr - C7X_RESULT_ADDR;
    outputs[i].data = (uint8_t *)client->result_buf + offset;
}

// Read DSP printf output (layer profiles, if -profile-layers was set)
if (resp->printf_size > 0)
    fwrite(printf_buf_ptr, 1, resp->printf_size, stderr);

*cycles = resp->cycles;
```

#### DSP — `handle_infer()`
**Source:** `firmware/c7x/dsp/src/compute_service.c:616`

**Steps A-B — Validate, resolve entry point, resolve constants, cache-invalidate inputs:**

```c
if (g_cg_main_dsp == NULL)
    dyn_loader_query_symbol(req->module_handle, "cg_main_dsp", &sym_addr);

uint32_t eff_model_id = req->model_id;
if (eff_model_id == 0 && g_embedded_model_id != 0)
    eff_model_id = g_embedded_model_id;
tvm_model_get_constants(eff_model_id, &constants, &num_constants);

for (i = 0; i < req->num_inputs; i++) {
    CacheP_inv((void *)(uintptr_t)td->data_addr, td->data_size, CacheP_TYPE_ALL);
    // Build zero-copy NDArray pointing into shared DDR
    ndarrays[i].data = (void *)(uintptr_t)td->data_addr;
    anys[i] = { .type_index = kTVMFFITensor, .v_ptr = &ndarrays[i] };
}
```

**Step C — Call the TVM-generated entry point:**

```c
shm_printf_reset();
start_cycles = __TSC;

ret = g_cg_main_dsp(input_anys,    // TVMFFIAny[] — input NDArrays
                    num_inputs,
                    constants,      // TVMFFIAny[] — weight NDArrays
                    &output_any);   // TVMFFIAny * — output

end_cycles = __TSC;
resp->cycles = end_cycles - start_cycles;
```

`g_cg_main_dsp` is generated by `src/target/c_static/`. Its signature is:
```c
int cg_main_dsp(TVMFFIAny *inputs, int num_inputs,
                TVMFFIAny *constants, TVMFFIAny *output);
```

**Step D — Extract and stage output tensors:**

```c
extract_infer_output(&output_any, resp);
// kTVMFFITensor (single): copy to result_buf + CacheP_wb
// kTVMFFIArray  (multi):  pack consecutively + CacheP_wb

resp->printf_size = shm_printf_finish();
send_response(C7X_MSG_INFER_RESP, ...);
```

---

### Phase 3: Unload Module

#### ARM — `c7x_client_dyn_unload()`
**Source:** `firmware/c7x/arm/src/c7x_compute_client.cpp:477`

```c
req.hdr.type      = C7X_MSG_DYN_UNLOAD;
req.module_handle = handle;
send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
client->input_data_offset = 0;   // next DYN_LOAD can use full staging buffer
```

#### DSP — `handle_dyn_unload()`

The DSP cleanup must follow a strict ordering — see §Dynamic Module Loading
§Module Unload Lifecycle for the detailed rationale. The code sequence is:

```c
TVMDSPRegFileCleanup();               // drops refs to last inference outputs
tvm_model_unload(g_embedded_model_id);// frees model slot before constants
TVMDSPConstantsCleanup();             // frees constants memory pools
dyn_loader_unload(req->module_handle);// frees .text/.data/.bss/.rodata
tvm_dsp_reset_pools();                // reclaim fragmented DDR heap
g_cg_main_dsp = NULL;
```

---

### Phase 4: Disconnect

#### ARM — `c7x_client_close()`

RAII destructors clean up in reverse order:

```c
close(rproc_fd)    // unregisters DMA buf attachment (DSP loses visibility)
close(dma_buf_fd)  // releases dmabuf reference
munmap(shared_buf, C7X_SHARED_SIZE)
close(dma_heap_fd)
close(rpmsg_fd)
```

---

### Complete Sequence Diagram

```
 ARM Linux                 RPMessage IPC            C7x FreeRTOS
 ─────────                 ─────────────            ────────────
 c7x_client_open()
   open RPMsg fd
   alloc DMA heap buf (512 MB)
   mmap shared DDR
   ioctl RPROC_DMA_BUF_ATTACH
                                                    compute_service_init()
                                                      RPMessage_construct(ep=20)
                                                      RPMessage_announce("rpmsg_chrdev")
                                                      dyn_loader_init()
                                                      tvm_model_init()
                                                      shm_printf_init()
                                                      → compute_service_run() loop

 c7x_client_dyn_load("lib0.out")
   fread ELF → staging_buf
   DMA_BUF_SYNC_WRITE (flush)
   ──── C7X_MSG_DYN_LOAD ──────────────────────────►
                                                    handle_dyn_load()
                                                      dyn_loader_load(C7X_STAGING_ADDR)
                                                        parse ELF, alloc DDR segments
                                                        apply C7x relocations
                                                        resolve 61 symbols
                                                      query_symbol("cg_main_dsp")
                                                      tvm_model_load_weights()
                                                      tvm_dsp_save_infer_watermark()
   ◄─── C7X_MSG_DYN_LOAD_RESP ─────────────────────
   handle=1, input_data_offset=elf_size

 c7x_client_infer(handle=1, model_id=0, ...)
   memcpy inputs → staging_buf[elf_size..]
   DMA_BUF_SYNC_WRITE (flush)
   ──── C7X_MSG_INFER ─────────────────────────────►
        inputs[0].data_addr = 0xC0000000 + elf_size
                                                    handle_infer()
                                                      resolve g_cg_main_dsp
                                                      get_constants(model_id)
                                                      CacheP_inv(input regions)
                                                      shm_printf_reset()
                                                      start_cycles = __TSC
                                                      g_cg_main_dsp(inputs, N,
                                                                     constants,
                                                                     &output_any)
                                                        ← TVM kernels execute ──►
                                                      end_cycles = __TSC
                                                      extract_infer_output()
                                                        CacheP_wb(output data)
                                                      shm_printf_finish()
   ◄─── C7X_MSG_INFER_RESP ────────────────────────
        cycles=N, num_outputs=1
        outputs[0].data_addr = 0xDE000000
   DMA_BUF_SYNC_READ (invalidate)
   outputs[0].data = result_buf + offset

 c7x_client_dyn_unload(handle=1)
   ──── C7X_MSG_DYN_UNLOAD ────────────────────────►
                                                    handle_dyn_unload()
                                                      TVMDSPRegFileCleanup()
                                                      tvm_model_unload()
                                                      TVMDSPConstantsCleanup()
                                                      dyn_loader_unload()
                                                      tvm_dsp_reset_pools()
   ◄─── C7X_MSG_DYN_UNLOAD_RESP ───────────────────
   input_data_offset = 0

 c7x_client_close()
   close(rproc_fd)  ← unmaps DMA buf from DSP
   munmap(shared_buf)
```

---

### CLI One-Liner: `c7x_compute run`

The `c7x_compute run` command wraps the full load → infer → unload sequence:

```bash
c7x_compute run lib0.out --input input.bin --output output.bin \
                --shape 1,3,224,224 --dtype float32
```

Internally calls: `c7x_client_open` → `dyn_load` → `infer` → `dyn_unload` →
`close`, then writes all output tensors to the output file.

Output (stdout, parseable by Python):
```json
{"status":"ok","cycles":12345678,"num_outputs":1,"outputs":[
  {"index":0,"ndim":2,"dtype_code":2,"dtype_bits":32,"data_size":4000,"shape":[1,1000]}
]}
```

---

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

This 512 MB region is the `vision_apps_shared-memories` DMA heap carveout,
exclusively for host-DSP communication. The host allocates it via
`/dev/dma_heap/carveout_vision_apps_shared-memories`, and the DSP MMU maps
it at 0xC0000000 with write-back cached attributes (MAIR7, Outer Shareable).

#### Extended DDR (above 4 GB, MMU-translated)

| Region | DSP Virtual | Physical | Size | Cache | Purpose |
|--------|-------------|----------|------|-------|---------|
| Non-cacheable heap | 0x100000000 | 0x880000000 | 32 MB | Non-cached (MAIR4) | DMA-accessible allocations |
| TVM DDR heap | 0x102000000 | 0x882000000 | 128 MB | Cached (MAIR7) | DLOAD segments + TVM workspace |

The TVM DDR heap is where DLOAD allocates code and data segments for loaded
modules, and where the TVM runtime allocates workspace tensors during inference.

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

The `rproc_fd` **must remain open** for the lifetime of the client — closing it
destroys the DMA attachment and makes the shared buffer invisible to the DSP.

### Cache Coherency

All cached shared regions use MAIR7 (Write-Back Read/Write-Allocate) with
Outer Shareable for hardware cache coherency between ARM and DSP. The host
still performs explicit DMA_BUF_SYNC ioctls:
- Before DSP reads: `DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE` (flush ARM cache)
- Before host reads: `DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ` (invalidate ARM cache)

### MMU Configuration

Defined in `dsp/configs/c75ss0.syscfg` (16 regions). Key MAIR attribute indices:

| MAIR | Encoding | Meaning | Used For |
|------|----------|---------|----------|
| MAIR0 | 0x00 | Device-nGnRnE (strongly ordered) | Peripherals, CLEC, DRU, L2SRAM |
| MAIR4 | 0x29 | Normal non-cacheable | IPC VRing, DMA buffers |
| MAIR7 | 0x3D | Write-Back Read/Write-Allocate | Code, data, shared buffer, TVM heap |

All cached regions (MAIR7) use Outer Shareable for hardware cache coherency.

### Memory Budget Example (ResNet-18)

| Resource | Size | Pool | Limit |
|----------|------|------|-------|
| lib0.out ELF (code + 46 MB weights) | ~47 MB | Input buffer | 504 MB |
| DLOAD segments (relocated code+data) | ~47 MB | TVM DDR heap | 128 MB |
| Inference workspace (intermediate tensors) | ~10-20 MB | TVM DDR heap | ~80 MB remaining |
| Input tensor (1,3,224,224 float32) | 0.6 MB | Input buffer | 504 MB |
| Output tensor (1,1000 float32) | 4 KB | Output buffer | 8 MB |

### Memory Address Map Summary

```
Physical (ARM)           DSP Virtual         Content
─────────────────────    ─────────────────   ────────────────────────────────
0x900_0000_0000          0xC000_0000         staging_buf base
  + 0 .. elf_size          ..                ELF bytes (DLOAD in-place rodata)
  + elf_size ..            ..                inference input tensors
0x91F_8000_0000          0xDF80_0000         result_buf base
  + 0 ..                   ..               inference output tensors
  + result_size - 64KB   0xE07F_0000         printf buffer (last 64 KB of result)
0x882_0000_0000          0x1_0200_0000       TVM DDR heap (128 MB)
  + 0 ..                   ..               DLOAD code/data segments
  + segment_end ..         ..               TVM workspace tensors
```

---

## Dynamic Module Loading (DLOAD)

The firmware embeds TI's DLOAD dynamic linker, which loads standard C7x ELF
relocatable objects at runtime. This is the mechanism by which TVM-compiled
models are deployed without reflashing firmware.

### DLOAD Integration

The `dyn_loader.c` module provides:
- **DLIF callbacks**: Firmware-side implementations of the DLOAD loader interface
  (`DLIF_allocate`, `DLIF_copy`, `DLIF_read`, etc.) that allocate from the TVM
  DDR heap via `tvm_dsp_alloc`
- **Symbol export table**: 61 symbols (C library, TVM runtime, VM builtins, math)
  made available to loaded modules at relocation time
- **Load/unload API**: `dyn_loader_load()` takes an ELF from the shared input
  buffer; `dyn_loader_unload()` frees all segments

### Supported Relocations

DLOAD handles 22 C7x-specific relocation types (defined in
`dload/C70_DLOAD_REL/c70_reloc.c`), covering all relocations produced by the
TI CGT C7000 compiler for position-dependent code with external symbol references.

### Symbol Export Table

The export table in `dyn_loader.c` provides these categories to loaded modules:

- **C library**: `printf` (redirected to `shm_printf`), `memcpy`, `memset`,
  `malloc`, `free`, `calloc`, `__c7xabi_cmpd`, etc.
- **TVM runtime**: `TVMBackendAllocWorkspace`, `TVMBackendFreeWorkspace`,
  `TVMFuncCall`, `TVMArgs_Create`, etc.
- **VM builtins (packed)**: `vm_builtin_*` functions using TVMArgs
- **VM builtins (direct C++ API)**: `tvm_dsp_*` direct-call variants for the
  C7x C++ API backend
- **Math**: `expf`, `logf`, `sqrtf`, `powf`, `floorf`, `fmaxf`, etc.

### Module Unload Lifecycle

When the host sends DYN_UNLOAD, `handle_dyn_unload()` must clean up TVM runtime
state **before** calling `dyn_loader_unload()`. This ordering is critical because
the loaded module's `.bss` section contains the static register file used by
`cg_main_dsp`, and `dyn_loader_unload()` frees all ELF segments including `.bss`.

The cleanup sequence (see §Inference Flow §Phase 3 for the annotated code):

1. **`TVMDSPRegFileCleanup()`** — Iterates the static register file (in the
   module's `.bss`), decrements reference counts on heap-allocated NDArray/storage
   objects from the last inference, and calls their deleters to free TVM DDR heap
   memory.

2. **`tvm_model_unload(model_id)`** — Frees the model slot in the `g_models[]`
   table (up to `MAX_MODELS=4`). Must happen before constants cleanup because
   `TVMDSPConstantsCleanup()` resets the constants subsystem state that the model
   slot references.

3. **`TVMDSPConstantsCleanup()`** — Frees all constants memory pools allocated
   by `TVMDSPLoadConstants()` during the INFER setup phase, and resets the
   constants subsystem (`g_initialized = 0`).

4. **`dyn_loader_unload(handle)`** — Calls `tracked_free_all()` to free all ELF
   segments (`.text`, `.data`, `.bss`, `.rodata.weights`) from the TVM DDR heap.

5. **Clear state** — Reset `g_loaded_module_handle`, `g_cg_main_dsp`, and
   `g_embedded_model_id` to zero.

If steps 1-3 are skipped or performed after step 4: the register file memory is
already freed (use-after-free), model slots leak preventing new models from loading
after 4 iterations, and constants memory pools leak reducing available TVM DDR heap.

### Symbol Table Synchronization

When building TVM-compiled modules as DLOAD-loadable ELFs, the module's linker
script must import matching symbol names so that DLOAD can resolve them at load
time. See `src/runtime/ti_dsp/dynmod/` for the linker script templates.

---

## IPC Details

### RPMessage Configuration

| Parameter | Value |
|-----------|-------|
| Service Name | "rpmsg_chrdev" (announced) |
| Endpoint | 20 |
| Max Message Size | 512 bytes |
| VRing Location | 0xAF800000 |

### Initialization Sequence

See §Inference Flow §Phase 0 for the annotated `compute_service_init()`
walk-through. The key steps are:

1. `RPMessage_waitForLinuxReady()` — polls resource table until Linux initializes
   virtio vrings
2. `IpcNotify_registerClient(IPC_NOTIFY_CLIENT_ID_RP_MBOX, ...)` — register
   shutdown callback
3. `RPMessage_construct()` — create endpoint 20
4. `RPMessage_announce("rpmsg_chrdev")` — announce service to Linux
5. `compute_service_run()` — blocking message loop

### Clean Shutdown

1. Linux sends `IPC_NOTIFY_RP_MBOX_SHUTDOWN` via mailbox
2. ISR callback calls `compute_service_stop()` which calls `RPMessage_unblock`
3. Service loop exits, sends `IPC_NOTIFY_RP_MBOX_SHUTDOWN_ACK` from task context
4. Tears down RPMessage endpoint and drivers
5. Disables interrupts and halts with `IDLE` instruction

The ACK must be sent before `RPMessage_destruct()` because that disrupts the
virtio transport needed for the mailbox ACK to reach the kernel.

### Device Discovery

Both the deploy script and host library discover hardware dynamically by matching
the device tree address `7e000000.dsp` in sysfs. The host scans
`/sys/class/rpmsg/rpmsg_ctrlN/device` paths for a symlink resolving to this
address. This approach is robust across reboots where remoteproc/rpmsg indices
may change.

---

## Shared Memory Printf

The DSP's `printf` output is redirected to a 64 KB region at the end of the
output buffer in shared DDR. This replaces the previous approach of writing to
the DebugP trace buffer (`trace0`), which was limited to ~2 KB and could not hold
profile output for models with many layers (e.g. CLISTA-DoA produces ~12 KB of
profile data for 156 layers).

### Architecture

The `shm_printf` module (`dsp/src/shm_printf.c`) uses the TI C7000 compiler's
`add_device()` RTS mechanism (Section 7.2.4 of the C7000 Compiler User's Guide)
to register a custom I/O device named "shmout" that writes to shared memory.
At init time, `freopen()` redirects stdout through this device, so all standard
output functions (`printf`, `fprintf(stdout, ...)`, `fputs`, `puts`) write to the
shared buffer.

Two output paths coexist:

1. **Direct path** (`shm_printf`): The DLOAD symbol alias maps the loaded module's
   `printf` calls directly to `shm_printf()`, which does `vsnprintf` into the
   buffer. Fast path, avoids FILE* overhead.

2. **Device driver path** (`SHM_write`): The `add_device` write callback handles
   any output going through the stdio FILE* machinery (e.g. `fprintf(stdout, ...)`
   from runtime code).

### Buffer Layout

The printf buffer occupies the last 64 KB of the result buffer
(`C7X_PRINTF_BUF_ADDR` = result buffer end - 64 KB):

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
2. **During inference**: DSP `printf` writes directly to SHM buffer via `memcpy`
3. **After inference**: `shm_printf_finish()` calls `CacheP_wb()`, returns byte count
4. **INFER response**: `resp->printf_size` carries the byte count in the single RPMsg
5. **Host reads**: After `sync_output_from_device()`, host reads `printf_size` bytes
   from `result_buf + printf_offset`

Buffer overflow is handled by silent truncation — excess data beyond the 64 KB
text area is dropped without error.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Separate staging/result halves of shared DDR | Avoids cache coherency races: ARM only writes to staging; DSP only writes to result |
| `input_data_offset = elf_size` after DYN_LOAD | DLOAD maps `.rodata` in-place from staging; writing inputs at offset 0 would corrupt embedded weights |
| `rproc_fd` held open for lifetime of `c7x_client` | Closing `rproc_fd` destroys the DMA attachment, making the buffer invisible to the DSP |
| 61 imported symbols resolved at DLOAD time | Eliminates per-call FFI overhead; `cg_main_dsp` calls TVM runtime functions via direct function pointer |
| `__TSC` (64-bit hardware TSC) for cycle counting | Available on C7x without kernel support; avoids FreeRTOS tick resolution limits; won't wrap at ~4.3s like a 32-bit counter |
| Explicit `DMA_BUF_SYNC` ioctls despite MAIR7 Outer Shareable | ARM and DSP have separate cache hierarchies; explicit sync is required for correct coherency — hardware coherency at MAIR7 is between ARM cores, not between ARM and DSP |

---

## Build System Internals

### DSP Firmware

CMake-based cross-compilation using the TI CGT C7000 toolchain. The toolchain
file (`cmake/toolchain-c7000.cmake`) sets paths to the MCU+ SDK, CGT compiler,
and SysConfig tool. The linker script (`configs/linker_c75_freertos.cmd`) defines
all memory sections including the DLOAD code/data placement.

SDK Dependencies:
- TI MCU+ SDK 11_00_00_06
- TI CGT C7000 5.0.1 LTS
- TI SysConfig 1.26.0

Output: `dsp/build/c7x_compute.out` (~6.9 MB)

### Host Application

C++14 CMake build with `aarch64-linux-gnu-g++` cross-compilation. Links against
pthread. Uses RAII wrappers (`UniqueFd`, `MmapRegion`, `UniqueFile`) for automatic
resource cleanup. The build produces a single `c7x_compute` binary that serves as
both CLI tool and library test harness. The public C API header
(`c7x_compute_client.h`) has `extern "C"` guards and is usable from C code.

Dependencies:
- `gcc-aarch64-linux-gnu` and `g++-aarch64-linux-gnu`

Output: `arm/build/c7x_compute`

---

## Relationship to TVM

This firmware is the runtime target for TVM's C static backend when compiling
for C7x DSP. The workflow is:

1. **TVM compilation** (on development host): Compile a neural network model
   using the TVM C static backend, producing `lib0.c` and `weights.bin`
2. **C7x ELF build**: Compile `lib0.c` with the TI CGT C7000 compiler into a
   relocatable ELF (`lib0.out`) with weights embedded in `.rodata.weights`
3. **Deployment**: Copy `lib0.out` to the AM67A target
4. **Execution**: Use the host CLI or library to load the ELF onto the DSP via
   DLOAD, run inference, and retrieve results

The DSP-side tests in `tests/ti-dsp-runtime/dsp-tests/` automate this full
pipeline using pytest, including TVM compilation, C7x ELF building, firmware
deployment, and inference verification.

---

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
+-- arm/
|   +-- build.sh                  # ARM client build and deploy script
|   +-- CMakeLists.txt
|   +-- include/
|   |   +-- c7x_compute_client.h  # Client library API
|   |   +-- c7x_runtime.h         # C++ Module/Function API (DLPack-based)
|   +-- src/
|   |   +-- raii.h                 # RAII wrappers (UniqueFd, MmapRegion, UniqueFile)
|   |   +-- c7x_compute_client.cpp # Client library implementation (C++14)
|   |   +-- c7x_runtime.cc         # c7x::Module C++ wrapper implementation
|   |   +-- rpmsg_wrapper.cpp      # rpmsg_ctrl discovery + endpoint mgmt
|   |   +-- rpmsg_wrapper.h
|   |   +-- c7x_compute_cli.cpp    # CLI tool
|   +-- test/
|       +-- test_c7x_runtime.cpp   # C++ test binary for c7x::Module API
+-- test/
    +-- test_dynmod.sh            # Automated hardware test suite (19 tests)
```
