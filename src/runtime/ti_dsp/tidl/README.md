# TIDL API for TVM DSP Runtime

C7x runtime API for TIDL subgraph execution.  These files are compiled
into the **DLOAD module** (.out), NOT the firmware.  The firmware
provides TIDL algo libraries and shared resources (UDMA, memory pools).

For the full offloading pipeline (Python partitioning, import, codegen,
bridge), see `docs/dsp/tidl-subgraph-offloading.md`.

## Files

| File | Purpose |
|------|---------|
| `tidl_api.c` / `.h` | `init/process/free_tidl_subgraph` — IALG lifecycle |
| `tidl_api_mem.c` / `.h` | `appMemAlloc/appMemFree` via `tvm_dsp_alloc` (128 MB DDR heap) |
| `ti_mem_manager.c` / `.h` | Bump allocator for L1/L2/L3 SRAM pools |
| `tidl_host_stubs.c` | x86 stubs for firmware-provided symbols, `c7x_host` (`HOST_EMULATION`) builds only |

## IALG Lifecycle

```
init_tidl_subgraph(network, network_size, IOParams, udma, is_nchw, rt_info):
    init_mem_regions(L1, L2, L3)            // from firmware symbols
    copy network to writable DDR            // TIDL_COPY_NETWORK_BUF
    TIDL_createParamsInit(cp)
    cp->net = network_copy
    cp->udmaDrvObj = udma                   // from appUdmaGetObj()
    cp->cacheWriteBack = TVM_cacheWbInvRegion
    algNumAlloc() -> numMemRec
    algAlloc(cp, memRec)                    // fills memory requirements
    alloc_mem_records(memRec)               // L1/L2/L3 with DDR fallback
    algInit(handle, memRec, cp)
    init_inbufs/init_outbufs                // IVISION buffer descriptors
    return instance

process_tidl_subgraph(instance, in_tensors[], out_tensors[]):
    algActivate(handle)                     // acquire DMA channels
    connect_input_output_tensors()          // DLTensor* -> IVISION buf
    algProcess(handle, inBufs, outBufs)     // TIDL inference on MMA
    disconnect_input_output_tensors()
    algDeactivate(handle)                   // release DMA channels

free_tidl_subgraph(instance):
    algFree(handle, memRec)
    free_mem_records + all allocations
```

## UDMA Handle Functions

TVM and TIDL share a single UDMA driver instance.  See the firmware
README (`firmware/c7x/README.md`, "EDMA / UDMA Subsystem") for the
full lifecycle.

| Function | Used by | Notes |
|----------|---------|-------|
| `appUdmaGetObj()` | DLOAD module (bridge code) | **Use this one** |
| `getUDMADrvObjPtr()` | TIDL algo libs (firmware internal) | **Never call from modules** |

Both are defined in firmware `tidl_support.c` and return the same
`tvm_dsp_dma_get_udma_handle()` value.  Both must exist in the
firmware and DLOAD export table.

**Why two functions:**
- `appUdmaGetObj` — PSDK OSAL naming convention
- `getUDMADrvObjPtr` — neo-tvm / TIDL internal convention, referenced
  by precompiled `tidl_algo.lib`

**Why modules must not call `getUDMADrvObjPtr`:**
Calling it from a DLOAD module causes firmware hangs.  The exact
mechanism wasn't fully diagnosed, but `appUdmaGetObj` works reliably.

**Why `getUDMADrvObjPtr` must stay in the DLOAD export table:**
Removing it causes firmware hangs when TIDL algo libs run.  The
algo libs reference it internally.

## Bridge Cleanup

The auto-generated `tidl_bridge.c` includes a `tidl_bridge_cleanup()`
function that calls `free_tidl_subgraph` for each lazily-initialized
TIDL instance.  The firmware calls this via DLOAD symbol lookup
before `dyn_loader_unload` to ensure TIDL's IALG handle, DMA
channels, and memory records are released cleanly.

Without this cleanup, module unload causes a crash in
`Udma_chDisable` because TIDL's channels are still registered in
the shared UDMA driver when the module's memory is freed.

## IOBufDesc Struct Layout

The `sTIDL_IOBufDesc_t` struct (params_1.bin artifact) has arrays
sized by a core count constant:

| Header | Constant | Value | sizeof(sTIDL_IOBufDesc_t) |
|--------|----------|-------|---------------------------|
| `itidl_io.h` (c7x-mma-tidl source) | `TIDL_IO_MAX_NUM_CORES` | 4 (hardcoded) | 378,392 bytes |
| `itidl_ti.h` (PSDK) | `TIDL_MAX_NUM_CORES` | 2 (SOC_J722S) | 189,208 bytes |

The import tool writes artifacts using the source header (CORES=4).
The DLOAD module must read them with matching layout.

**Fix:** `tidl_api.c` forces `#define TIDL_MAX_NUM_CORES 4` before
`#include "itidl_ti.h"`.  The DLOAD module build (`CMakeLists.txt`)
includes headers from c7x-mma-tidl source (not PSDK).

## Output Tensor Mapping

`connect_input_output_tensors()` maps TIDL outputs to TVM tensors:
- When `outDataName` is populated (Relax FFI import path): parses
  `tidl_{sg}_o{idx}` to extract tensor index
- When `outDataName` is empty: falls back to sequential mapping (j=i)

## Memory Pools (J722S)

| Pool | Size | Source | DDR fallback |
|------|------|--------|--------------|
| L1 DARAM | firmware | C7x L1 SRAM | Yes |
| L2 SRAM | firmware | C7x L2 SRAM | Yes |
| L3 (aux L2) | 240 KB | `MSMCSIZE_KB=240` | Yes |
| DDR heap | 128 MB | `tvm_dsp_alloc` | N/A |

J722S has no MSMC SRAM.  L3 pool uses auxiliary L2.

## Known Issues

1. **RTS heap too small**: `appMemAlloc` must use `tvm_dsp_alloc`
   (128 MB DDR), not the RTS heap (128 KB).

2. **traceWriteLevel must be 0**: Non-zero requires a non-NULL
   `TIDLWriteBinToFile` callback.

3. **Quant stats required**: `algAlloc` fails with -1125 if
   `isQuantStatsAvailable=0`.  Ensure `inData` and `tidlStatsTool`
   are set in import config.

4. **L3/MSMC size**: `MSMCSIZE_KB` > 240 causes TIDL to request
   memory that doesn't exist on J722S.

## Adaptations from neo-tvm

1. `appMemAlloc/appMemFree` route to `tvm_dsp_alloc` (DDR heap),
   not PSDK OSAL
2. `TIDLRT_LogMetaData()` stubbed out (no-op macro)
3. Memory pool globals renamed to `g_tidl_l1_mem_addr` etc.
4. `TIDL_MAX_NUM_CORES` forced to 4 for IOBufDesc compatibility

## Build Integration

Compiled by `CMakeLists.txt` (`USE_TIDL=ON`) alongside lib0.c and
TIDL artifacts.  Headers from c7x-mma-tidl source tree.

Build all TIDL dependencies:
```bash
cd ~/ml/c7x-mma-tidl
bash build_j722s.sh          # incremental
bash build_j722s.sh clean    # clean + full rebuild
```
