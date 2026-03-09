# TIDL Subgraph Offloading for Relax c_static Backend

Offload supported subgraphs from TVM/Relax models to TI Deep Learning
(TIDL) on the C7x MMA accelerator.  Non-TIDL ops remain in TVM and
execute as generated C code on the C7x scalar pipeline.

## Quick Start

```python
import tvm
from tvm import relax
from tvm.relax.backend.tidl import TIDLOffloadCompiler, LowerTIDLToTIR

compiler = TIDLOffloadCompiler(config={
    "artifacts_dir": "/tmp/tidl_artifacts",
    "tidl_tools_path": "~/ml/c7x-mma-tidl/tidl_tools",
})

# 1. Partition: identify TIDL-supported subgraphs
partitioned = compiler.partition(mod)

# 2. Import: compile subgraphs into TIDL artifacts (net.bin + io.bin)
#    Requires tidl_model_import_relax.so
imported, artifacts = compiler.tidl_import(partitioned)

# 3. Lower: replace TIDL functions with TIR extern stubs
lowered = compiler.lower_tidl(imported, artifacts)

# 4. Compile: generate C code via c_static
target = tvm.target.Target("c_static -mcpu=c7x")
ex = relax.build(lowered, target=target,
                 exec_mode="compiled", system_lib=True)
ex.export_library("model.tar", target=target)
```

Step 2 (import) requires the TIDL import library.  For partition +
lower only (using pre-generated artifacts or stub bridges), use:

```python
from tvm.relax.backend.tidl import partition_for_tidl, LowerTIDLToTIR

partitioned = partition_for_tidl(mod)
lowered = LowerTIDLToTIR()(partitioned)
```

The generated `lib0.c` contains `tidl_subgraph_N_process()` extern
calls for each TIDL subgraph.  These are resolved at link time by a
bridge function (see "Bridge Function" below).

## Supported Operators

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.nn.conv2d` | `conv2d` |
| `tidl.nn.conv2d_bias` | `conv2d` + `add` |
| `tidl.nn.conv2d_relu` | `conv2d` + `relu` |
| `tidl.nn.conv2d_clip` | `conv2d` + `clip` |
| `tidl.nn.conv2d_bias_relu` | `conv2d` + `add` + `relu` |
| `tidl.nn.conv2d_bias_clip` | `conv2d` + `add` + `clip` |
| `tidl.nn.max_pool2d` | `max_pool2d` |
| `tidl.nn.avg_pool2d` | `avg_pool2d` |
| `tidl.nn.batch_norm` | `batch_norm` + `tuple_get_item[0]` |
| `tidl.nn.relu` | `relu` |
| `tidl.add` | `add` (element-wise) |
| `tidl.quantize` | `quantize` (stub) |
| `tidl.dequantize` | `dequantize` (stub) |

Constraint checks run during partitioning:
- Conv2d: kernel <= 7, equal H/W strides
- Pool: kernel <= 3, input rank == 4
- All ops: dtype in {float32, int8, int16, uint8}

## Bridge Function

The lowering pass generates `call_extern("tidl_subgraph_0_process")`
in the TIR, which becomes an unresolved extern in `lib0.c`.  A bridge
function resolves it:

```python
from tvm.relax.backend.tidl import TIDLOffloadCompiler

# Stub bridge (zero-fill output, for testing without TIDL libs)
TIDLOffloadCompiler.generate_bridge(lowered, "tidl_bridge.c", stub=True)

# Real bridge (calls init/process/free_tidl_subgraph)
TIDLOffloadCompiler.generate_bridge(lowered, "tidl_bridge.c", stub=False)
```

This produces `tidl_bridge.c` and `tidl_bridge.h`.  The header provides
`extern "C"` declarations.  The build system uses `-include tidl_bridge.h`
so `lib0.c` can resolve the symbols at compile time.

The real bridge:
1. Lazy-inits the TIDL instance via `init_tidl_subgraph()`
2. Wraps raw `void*` pointers in `DLTensor` structs
3. Flushes input from cache (`TVM_cacheWbInvRegion`)
4. Calls `process_tidl_subgraph(instance, in_tensors, out_tensors)`
5. Invalidates output cache after DMA completes

## Running Tests

```bash
# Partition + codegen tests only (no TI compiler or .so needed):
pytest tvm-relax-tests/tidl-tests/test_tidl_partition.py \
       tvm-relax-tests/tidl-tests/test_tidl_codegen.py -v

# Import tests (needs tidl_model_import_relax.so + c7x-mma-tidl):
pytest tvm-relax-tests/tidl-tests/test_tidl_relax_import.py -v

# All TIDL tests (partition + import + codegen + e2e + hardware):
TI_CGT_C7000_PATH=~/ti/.../ti-cgt-c7000_5.0.1.LTS \
  pytest tvm-relax-tests/tidl-tests/ -v
```

---

## Architecture

### Pipeline

```
Relax IR (conv2d, relu, pool, softmax, ...)
    |
    v
Phase 1: partition()
    |  FuseOpsByPattern: match TIDL patterns, create composites
    |  MergeCompositeFunctions: group into Codegen="tidl" functions
    v
Partitioned IR (TIDL subgraph functions + main)
    |
    v
Phase 3: tidl_import()
    |  Load tidl_model_import_relax.so
    |  For each subgraph: ImportInit -> ImportNode -> Link -> Optimize -> PostProcess
    |  Produces net.bin + io.bin artifacts on disk
    v
Partitioned IR (unchanged) + TIDL artifacts
    |
    v
Phase 4: lower_tidl()
    |  Replace Codegen="tidl" functions with TIR PrimFunc stubs
    |  Each stub: call_extern("tidl_subgraph_N_process", inp, out)
    v
Lowered IR (TIR stubs + remaining Relax ops)
    |
    v
Phase 5: generate_bridge() + relax.build()
    |  c_static codegen emits lib0.c
    v
lib0.c + weights.bin + tidl_bridge.c/h
    |
    v
Cross-compile (TI C7x) -> lib0.out (DLOAD module)
    |  Embedded TIDL artifacts (net.bin + io.bin as .rodata)
    |  Links tidl_api.c (IALG wrapper) + bridge
    v
Deploy to AM67A via c7x_compute
    |  DLOAD loads module, resolves firmware symbols
    v
Inference: TIDL int8 on MMA + TVM float32 on C7x scalar
```

### Firmware integration

TIDL algo libraries (tidl_algo.lib, MMALIB) are linked into the
**firmware**, not the model module.  The firmware exports shared
resources via the DLOAD symbol table:

| Symbol | Purpose |
|--------|---------|
| `TIDL_VISION_FXNS` | IALG function table (from tidl_algo.lib) |
| `appMemAlloc/appMemFree` | DDR heap allocation (via tvm_dsp_alloc) |
| `appUdmaGetObj` | Shared UDMA driver handle |
| `g_l1_mem_addr/size` | L1D SRAM pool for TIDL IALG |
| `g_l2_mem_addr/size` | L2 SRAM pool (shared with TVM) |
| `g_l3_mem_addr/size` | Auxiliary L2 pool (used as L3) |
| `TVM_cacheWbInvRegion` | Cache writeback for DMA coherency |
| `dsp_trace_msg` | Trace output to remoteproc buffer |

### TIDL API files

Located in `src/runtime/ti_dsp/tidl/`, adapted from neo-tvm:

| File | Purpose |
|------|---------|
| `tidl_api.c/h` | `init/process/free_tidl_subgraph` -- IALG lifecycle |
| `tidl_api_mem.c/h` | Memory allocation (appMemAlloc from firmware) |
| `ti_mem_manager.c/h` | Bump allocator for L1/L2/L3 SRAM pools |

### Code generation

No c_static C++ changes were needed.  The existing `CodeGenCStatic`
handles `call_extern` in TIR PrimFuncs natively.

`relax.build()` must use `exec_mode="compiled"` for DSP targets.
The default `"bytecode"` does not generate `__vmtir__main`.

---

## Build Flags

| Flag | Where | Purpose |
|------|-------|---------|
| `USE_TIDL` | `dsp-cpp/CMakeLists.txt` | Compile TIDL API, embed artifacts |
| `TIDL_BRIDGE_SOURCES` | `dsp-cpp/CMakeLists.txt` | Bridge .c file path |
| `TIDL_ARTIFACTS_DIR` | `dsp-cpp/CMakeLists.txt` | Directory with net.bin + io.bin |
| `USE_TIDL_RUNTIME` | firmware `CMakeLists.txt` | Link TIDL algo + MMALIB into firmware |
