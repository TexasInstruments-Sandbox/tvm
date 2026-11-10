# TIDL API for TVM DSP Runtime

C7x runtime API for TIDL subgraph execution.  These files are compiled
into the **DLOAD module** (.out), NOT the firmware.  The firmware
provides TIDL algo libraries and shared resources (UDMA, memory pools).

## Files

| File | Purpose |
|------|---------|
| `tidl_api.c` / `.h` | `init/process/free_tidl_subgraph` — IALG lifecycle |
| `tidl_api_mem.c` / `.h` | `appMemAlloc/appMemFree` via `tvm_dsp_alloc` (352 MiB DDR heap) |
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
README (`firmware/c7x/README.md`, "UDMA Subsystem") for the
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
`#include "itidl_ti.h"`.  On the DLOAD path these sources are compiled
into the `c7x_compute` firmware (`firmware/c7x/dsp/CMakeLists.txt`, `--tidl
ON`) against the c7x-mma-tidl source headers, and exported to modules;
`c7x_host` compiles them into the host executable.

## Output Tensor Mapping

`connect_input_output_tensors()` maps TIDL outputs to TVM tensors:
- When `outDataName` is populated (Relax FFI import path): parses
  `tidl_{sg}_o{idx}` to extract tensor index
- When `outDataName` is empty: falls back to sequential mapping (j=i)

## Memory Pools (J722S)

Set by `tidl_support_init()` in `firmware/c7x/dsp/src/tidl_support.c`,
following the vision_apps layout (`app_cfg_c7x_1.h`):

| Pool | Region | Size | DDR fallback |
|------|--------|------|--------------|
| L1 DARAM | L2 AUX as L1, `0x7F03C000` | 16 KB | Yes |
| L2 SRAM | L2 AUX, `0x7F000000` | 224 KB (`L2MEMSIZE_KB` in `j722s_config.cfg`) | Yes |
| L3 | L2 MAIN, `0x7E000000` (J722S has no MSMC) | 1.25 MB | Yes |
| DDR heap | `tvm_dsp_alloc` | 352 MiB | N/A |

Both L2 banks are shared with TVM.  The runtime's fast pool lives in L2 AUX, so
the firmware disables it while a TIDL module is loaded (see
`TVMDSPStorageSetFastPoolEnabled` in `vm/storage.h`); otherwise a small TVM
tensor live across a TIDL call is overwritten by TIDL's scratch.

## Known Issues

1. **RTS heap too small**: `appMemAlloc` must use `tvm_dsp_alloc`
   (352 MiB DDR), not the RTS heap (128 KB).

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
