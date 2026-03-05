# TIDL API for TVM DSP Runtime

Source files for the TIDL subgraph API, adapted from neo-tvm
`src/runtime/contrib/tidl/c7x/` for the TVM c_static DLOAD module build.

These files are compiled into the **model .out** (DLOAD module),
NOT the firmware.  They link against TIDL algo libraries
(tidl_algo.lib, tidl_obj_algo.lib, etc.) and MMALIB.

## Files

- `tidl_api.c` / `tidl_api.h` — init/process/free_tidl_subgraph
- `tidl_api_mem.c` / `tidl_api_mem.h` — memory allocation (malloc-based)
- `ti_mem_manager.c` / `ti_mem_manager.h` — simple bump allocator for
  L1/L2/L3 SRAM pools (from c7x-mma-tidl/common/)

## Adaptations from neo-tvm

1. **Memory allocation**: `appMemAlloc/appMemFree` (PSDK OSAL) replaced
   with `malloc/memalign/free` (standard C, provided by firmware via DLOAD).

2. **itidl_rt.h**: Stubbed out.  `TIDLRT_LogMetaData()` replaced with
   no-op macro (debug tracing only).

3. **Memory pool globals**: Renamed from `g_l1_mem_addr` to
   `g_tidl_l1_mem_addr` etc. to match our firmware symbol exports
   (tidl_support.c in firmware).

4. **Include paths**: Need `itidl_ti.h` and `ivision.h` from the PSDK
   at compile time.

## Build integration

These files are compiled by the DSP model toolchain
(`toolchain-j722s-c7x.cmake`) when `USE_TIDL=ON`, alongside
the generated `lib0.c` and TIDL binary artifacts.
