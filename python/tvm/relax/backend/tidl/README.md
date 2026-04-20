# TIDL Subgraph Offloading for Relax c_static Backend

Offload supported subgraphs from TVM/Relax models to TI Deep Learning
(TIDL) on the C7x MMA accelerator.  Non-TIDL ops remain in TVM and
execute as generated C code on the C7x scalar pipeline.

## Quick Start

### One-call build (recommended)

```python
from tvm.relax.backend.tidl import TIDLOffloadCompiler

compiler = TIDLOffloadCompiler(config={
    "artifacts_dir": "/tmp/tidl_artifacts",
    "tidl_tools_path": "~/ml/c7x-mma-tidl/tidl_tools",
})

# Single call: partition -> import -> lower -> codegen -> bridge -> build
result = compiler.build(mod, params=param_dict)

result.module_path   # Path to lib0.out (C7x DLOAD module)
result.weights_path  # Path to weights.bin
result.gen_dir       # Path to generated code directory
result.artifacts     # {sg_name: {"net_bin": path, "io_bin": path}}
```

Requires: `tidl_model_import_relax.so`, TI C7x cross-compiler.

### Step-by-step (for debugging or custom pipelines)

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

For partition + lower only (using pre-generated artifacts or stub
bridges, no TIDL library needed):

```python
from tvm.relax.backend.tidl import partition_for_tidl, LowerTIDLToTIR

partitioned = partition_for_tidl(mod)
lowered = LowerTIDLToTIR()(partitioned)
```

The generated `lib0.c` contains `tidl_subgraph_N_process()` extern
calls for each TIDL subgraph.  These are resolved at link time by a
bridge function (see "Bridge Function" below).

## Supported Operators

### Convolution

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.nn.conv2d` | `conv2d` |
| `tidl.nn.conv2d_bias` | `conv2d` + `add` |
| `tidl.nn.conv2d_relu` | `conv2d` + `relu` |
| `tidl.nn.conv2d_clip` | `conv2d` + `clip` |
| `tidl.nn.conv2d_bias_relu` | `conv2d` + `add` + `relu` |
| `tidl.nn.conv2d_bias_clip` | `conv2d` + `add` + `clip` |

### Pooling

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.nn.max_pool2d` | `max_pool2d` |
| `tidl.nn.avg_pool2d` | `avg_pool2d` |

### Reduction

| Pattern | Relax ops matched | Constraint |
|---------|-------------------|------------|
| `tidl.mean` | `mean` | axes [2,3] only (spatial/global avg pool) |
| `tidl.sum` | `sum` | single axis only |
| `tidl.reduce_max` | `max` | axis=2 (HEIGHT in NCHW) only |
| `tidl.reduce_min` | `min` | axis=2 (HEIGHT in NCHW) only |
| `tidl.argmax` | `argmax` | axis=1 (C in NCHW), keepdims=True only |
| `tidl.argmin` | `argmin` | axis=1 (C in NCHW), keepdims=True only |

### Activations

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.nn.relu` | `relu` |
| `tidl.sigmoid` | `sigmoid` |
| `tidl.tanh` | `tanh` |
| `tidl.clip` | `clip` (e.g. relu6 = `clip(x, 0, 6)`) |
| `tidl.nn.leakyrelu` | `leakyrelu` |
| `tidl.nn.prelu` | `prelu` (with learnable per-channel slope) |
| `tidl.elu` | `exp` + `subtract` + `relu` + `multiply` + `add` (ELU composite) |
| `tidl.hard_sigmoid` | `add` + `clip` + `divide` (HardSigmoid: clip(x+3,0,6)/6) |
| `tidl.hard_swish` | `add` + `clip` + `divide` + `multiply` (HardSwish: x * hard_sigmoid(x)) |
| `tidl.mish` | `exp` + `add` + `log` + `tanh` + `multiply` (Mish: x * tanh(softplus(x))) |

### Element-wise

| Pattern | Relax ops matched | Constraint |
|---------|-------------------|------------|
| `tidl.add` | `add` | 4-D inputs only |
| `tidl.multiply` | `multiply` | 4-D inputs only |
| `tidl.subtract` | `subtract` | 4-D inputs only |
| `tidl.divide` | `divide` | 4-D inputs only |
| `tidl.maximum` | `maximum` | 4-D inputs only |
| `tidl.minimum` | `minimum` | 4-D inputs only |

### Linear / Matmul

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.matmul` | `matmul` |
| `tidl.matmul_bias` | `matmul` + `add` (FC with bias) |

### Attention / Transformer

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.nn.softmax` | `softmax` |

### Shape and Layout

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.reshape` | `reshape` |
| `tidl.flatten` | `flatten` |
| `tidl.squeeze` | `squeeze` |
| `tidl.expand_dims` | `expand_dims` |
| `tidl.strided_slice` | `strided_slice` |
| `tidl.permute_dims` | `permute_dims` (transpose) |
| `tidl.concat` | `concat` |

### Normalization

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.nn.layer_norm` | `layer_norm` (gamma/beta from constants) |
| `tidl.nn.instance_norm` | `instance_norm` (gamma/beta from constants) |

### Data Type / Padding

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.cast` | `astype` |
| `tidl.nn.pad` | `nn.pad` (constant zero-padding) |

### Advanced

| Pattern | Relax ops matched | Notes |
|---------|-------------------|-------|
| `tidl.image.resize2d` | `image.resize2d` | 4-D input; bilinear or nearest-neighbor |
| `tidl.take` | `take` | gather along axis |
| `tidl.topk` | `topk` | axis must be HEIGHT or WIDTH; sorted=True required; TIDL always produces values+indices — use `ret_type="both"` and extract `[0]` |
| `tidl.split` | `split` | supports equal sections (`split(x, N, axis)`) and explicit indices (`split(x, [i0, i1, ...], axis)`) |
| `tidl.nn.depth_to_space` | `nn.pixel_shuffle` | DCR mode (depth-column-row); upscale_factor from PixelShuffleAttrs |
| `tidl.expand` | `broadcast_to` | target shape from output struct_info |
| `tidl.scatter_elements` | `scatter_elements` | axis + reduction (update/add/max/min) |
| `tidl.scatter_nd` | `scatter_nd` | N-D indices; reduction (update/add/max/min) |
| `tidl.image.grid_sample` | `image.grid_sample` | bilinear/nearest; zeros padding; 4-D input |
| `tidl.nn.conv2d_transpose` | `nn.conv2d_transpose` | transposed convolution (deconv); optional bias fused |

### Math / Unary

Converted to `TIDL_BatchNormLayer` + hardware LUT by
`tidl_convertRelUToBNLayer` during PostProcessNet.  The LUT is built
automatically from calibration statistics.

Ops with **bounded output** (`abs`, `sqrt`, `erf`, `sin`, `cos`, `atan`,
`negative`, `floor`) produce correct output with random calibration data.
Ops with **unbounded output** (`exp`, `log`, `sinh`, `cosh`, `power`) may
produce all-zero output when random calibration data causes a very large
quantisation scale — use domain-specific bounded inputs for those ops.

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.abs` | `abs` |
| `tidl.sqrt` | `sqrt` |
| `tidl.power` | `power` (x \*\* scalar_exp) |
| `tidl.exp` | `exp` |
| `tidl.log` | `log` |
| `tidl.erf` | `erf` |
| `tidl.floor` | `floor` |
| `tidl.negative` | `negative` |
| `tidl.sin` | `sin` |
| `tidl.cos` | `cos` |
| `tidl.tan` | `tan` |
| `tidl.sinh` | `sinh` |
| `tidl.cosh` | `cosh` |
| `tidl.asin` | `asin` |
| `tidl.acos` | `acos` |
| `tidl.atan` | `atan` |
| `tidl.asinh` | `asinh` |

### Quantization (stubs)

| Pattern | Relax ops matched |
|---------|-------------------|
| `tidl.quantize` | `quantize` |
| `tidl.dequantize` | `dequantize` |

### Constraint checks

Run during partitioning before passing composites to TIDL:

- **All ops:** dtype must be in {float32, int8, int16, uint8}
- **Conv2d:** kernel size 1–7 (6 excluded by TIDL allowlisting),
  H and W strides must be equal, input rank ≤ 4
- **Pool:** kernel ≤ 3, input rank == 4
- **Mean:** input rank == 4, reduction axes must be exactly [2, 3]
  (spatial-only; channel/batch mean rejected)
- **ReduceMax/ReduceMin:** single axis only, must be axis=2 (HEIGHT in 4D
  NCHW); all other axes rejected by TIDL ReduceLayer
- **ArgMax/ArgMin:** axis must be 1 (channel, TIDL_DIM_NUMCH), keepdims=True
  required; other axes or keepdims=False rejected.  Output is int64 — add
  `astype("int32")` after argmax/argmin to keep both ops in one subgraph
- **Sum:** single axis only (multi-axis reduction rejected)
- **Element-wise (add, multiply, subtract, divide, maximum, minimum):**
  input rank must be exactly 4; sub-4D uses (e.g. FC bias add `(1,1000)`)
  are rejected — they cause a crash in TIDL algProcess
- **TopK:** axis must be HEIGHT (2) or WIDTH (3) in 4D NCHW; sorted must
  be True (TIDL constraint).  TIDL always produces both values and
  indices outputs — use `ret_type="both"` and extract `[0]` for values.
  The C++ parser sets `numConstInputs=1` to satisfy the allowlisting
  constraint that expects K as an initializer (in Relax, K is an attr).
- **Split:** any axis; equal sections (`split(x, N)`) and explicit
  cut-point indices (`split(x, [i0, i1, ...])`) both supported.

**Batch normalization:** `prepare()` runs `FoldBatchnormToConv2D` +
`FoldConstant` before partitioning, which algebraically folds
inference-mode batch_norm parameters into the preceding conv2d weights
and bias.  After folding the IR becomes `conv2d + add + relu` (no
batch_norm nodes), so the existing `conv2d_bias_relu` patterns match.

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

The real bridge (supports multiple TIDL subgraphs):
1. Lazy-inits each TIDL instance via `init_tidl_subgraph()`
   using per-subgraph artifact symbols (`_binary_tidl_net_N_*`,
   `_binary_tidl_io_N_*`)
2. Wraps raw `void*` pointers in `DLTensor` structs
3. Flushes input from cache (`TVM_cacheWbInvRegion`)
4. Calls `process_tidl_subgraph(instance, in_tensors, out_tensors)`
5. Invalidates output cache after DMA completes

Shared includes (`tidl_api.h`, `dlpack.h`, externs) are emitted once;
each subgraph gets its own `_process()` function and instance.

## Import Orchestration (`tidl_import()`)

`TIDLOffloadCompiler.tidl_import()` drives the Relax FFI pipeline:

1. Load `tidl_model_import_relax.so`
2. `TIDL_relaxInit()` — initialize with device config and artifacts dir
3. For each composite in each subgraph:
   - Lift constants into VarBindings (`_lift_constants_in_composite`)
   - Construct synthetic `relax.Call` with `UpdateStructInfo`
   - `TIDL_relaxImportNode()` — parse op attrs, register with TIDL
   - `TIDL_relaxImportLinkNode()` — connect data flow edges
4. `TIDL_relaxOptimizeNet()` — run network compiler
5. `TIDL_relaxPostProcessNet()` — write net.bin + params_1.bin

**Config keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `artifacts_dir` | `/tmp/tidl_artifacts` | Output directory for TIDL binaries |
| `tidl_tools_path` | auto-detect from `C7X_MMA_TIDL_PATH` | Path to device_config.cfg |
| `tidl_relax_so_path` | auto-detect | Path to `tidl_model_import_relax.so` |
| `num_calibration_frames` | 1 | Calibration iterations |

**Node naming:** `tidl_{sg_id}_i{idx}` for inputs, `tidl_{sg_id}_o{idx}`
for outputs, sequential integers for internal nodes.  Shapes normalized
to 6D TIDL format `[N,1,1,C,H,W]`.

For C7x runtime details (IALG lifecycle, UDMA, memory pools), see
`src/runtime/ti_dsp/tidl/README.md`.

## Building Dependencies

The import tests and hardware tests require `tidl_model_import_relax.so`
and supporting tools from the c7x-mma-tidl source tree.  Build all
components with:

```bash
cd ~/ml/c7x-mma-tidl
bash build_j722s.sh          # incremental build
bash build_j722s.sh clean    # clean + full rebuild
```

**Important — jump table header dependency:**  The incremental build
tracks `.cpp` file timestamps but not header dependencies.  When
`tidl_parse_relax_jumptable.h` is modified (e.g. to add new ops),
the translation unit that includes it (`tidlParseRelax/tidl_parse_relax.cpp`)
is **not** automatically recompiled.  After editing the jump table,
force a recompile:

```bash
rm ~/ml/c7x-mma-tidl/out/PC/dsp/algo/release/ti_dl/utils/tidlModelImport/tidlParseRelax/tidl_parse_relax.obj
bash build_j722s.sh
```

Or use `bash build_j722s.sh clean` for a full rebuild.

See `tests/ti-dsp-runtime/tidl-tests/README.md` for full prerequisites
and one-time setup steps.

## Running Tests

See `tests/ti-dsp-runtime/tidl-tests/README.md` for the full test list,
run commands, prerequisites, and build instructions.

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
Phase 6: _build_dynmod()
    |  CMakeLists.txt at src/runtime/ti_dsp/dynmod/
    |  TI C7x cross-compile via toolchain-j722s-c7x.cmake
    |  Embeds TIDL artifacts (net.bin + io.bin as .rodata)
    |  Links tidl_api.c (IALG wrapper) + bridge
    v
lib0.out (DLOAD module)
    |
    v
Deploy to AM67A via c7x_compute
    |  DLOAD loads module, resolves firmware symbols
    v
Inference: TIDL int8 on MMA + TVM float32 on C7x scalar

build() runs the entire pipeline in a single call
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

## Visualization

Generate an interactive HTML page showing which ops are offloaded
to TIDL vs executed as TVM C code, with optional per-layer profiling:

```python
from tvm.relax.backend.tidl.visualize import (
    visualize_partitioning,
    parse_layer_profile,
)

# Graph-only view (partition structure)
visualize_partitioning(partitioned_mod, "graph.html")

# With profile data from DSP hardware run
profile = parse_layer_profile(dsp_stdout)
visualize_partitioning(partitioned_mod, "graph.html",
                       profile_data=profile)
```

The HTML has two tabs:
- **Graph** — hierarchical dataflow graph with TIDL (red) and TVM
  (teal) nodes.  Click a TIDL node to expand its 37 internal layers
  with PyTorch source paths.
- **Profile** — per-layer cycle table with horizontal bar chart,
  sorted by cost.  Only shown when ``profile_data`` is provided.

From the command line:
```bash
python test_tidl_resnet_e2e.py --visualize resnet18.html
python test_tidl_resnet_e2e.py --visualize resnet18.html \
    --profile-json profile.json
```

## Build Infrastructure

The C7x DLOAD module build infrastructure lives at
`src/runtime/ti_dsp/dynmod/`:

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Standalone dynmod build (used by `build()` API) |
| `c7x_dynmod/dsp_syms.c` | Pseudo-firmware symbol table for DLOAD |
| `c7x_dynmod/c7x_dynmod.cmd` | Linker script for relocatable ELF |
| `c7x_dynmod/tidl_firmware_syms.c` | TIDL firmware symbol stubs |
| `c7x_dynmod/gen_dsp_syms_sect.py` | Embed dsp_syms.out as .dsp_syms_out section |

The test CMakeLists.txt at `tests/ti-dsp-runtime/dsp-cpp/` references
this location for c7x-dynmod builds, while retaining host/c66x/c7x_host
standalone build targets.

## Build Flags

| Flag | Where | Purpose |
|------|-------|---------|
| `USE_TIDL` | dynmod + test CMakeLists.txt | Compile TIDL API, embed artifacts |
| `TIDL_BRIDGE_SOURCES` | dynmod + test CMakeLists.txt | Bridge .c file path |
| `TIDL_ARTIFACTS_DIR` | dynmod + test CMakeLists.txt | Directory with net.bin + io.bin |
| `USE_TIDL_RUNTIME` | firmware `CMakeLists.txt` | Link TIDL algo + MMALIB into firmware |
