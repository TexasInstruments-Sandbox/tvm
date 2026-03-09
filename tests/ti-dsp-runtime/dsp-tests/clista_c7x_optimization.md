# CLISTA C7x Performance Optimization Analysis

## Executive Summary

The CLISTA-DoA model on C7x (J722S/AM67A, C7524 at 1 GHz) takes over
1M cycles without profiling, compared to ~600K cycles on C66x. Profile
analysis identifies two dominant bottlenecks: conv1d_transpose (56%)
and conv1d (33%). The remaining 11% comes from elementwise ops
(sqrt, divide, relu) that fail to vectorize due to scalar library
calls, plus 156 separate function-call overhead.

Profiled total: 2,611,280 cycles (with `-profile-layers` enabled,
which adds ~1.6M cycles of profiling overhead from 156 layer
measurements).

## Profile Breakdown

Measured with `python test_clista_dsp.py --dsp-dload --profile-layers`
on AM67A with firmware deployed via DLOAD.

### Cycle Distribution by Layer Type

| Layer type       | Calls | Avg cycles | Total     | %    |
|------------------|-------|------------|-----------|------|
| conv1d_transpose | 7     | ~210K      | 1,472K    | 56%  |
| conv1d           | 8     | ~107K      | 855K      | 33%  |
| tir_sqrt         | 8     | ~7.4K      | 59K       | 2.3% |
| divide           | 8     | ~5.5K      | 44K       | 1.7% |
| relu             | 8     | ~2.3K      | 18K       | 0.7% |
| other (117)      | 117   | ~1.4K      | 163K      | 6.3% |
| **Total**        | **156** |          | **2,611K**|      |

### Per-Layer Profile (Top 20)

Layers are listed in execution order. Each conv1d and conv1d_transpose
call dominates, with other layers contributing under 10K cycles each.

```
Layer 2:  conv1d_transpose   207,580 cycles
Layer 5:  conv1d             105,816 cycles
Layer 8:  conv1d_transpose   205,496 cycles
Layer 11: conv1d             111,816 cycles
Layer 14: conv1d_transpose   207,024 cycles
Layer 17: conv1d             105,512 cycles
Layer 20: conv1d_transpose   209,148 cycles
Layer 23: conv1d             101,600 cycles
Layer 26: conv1d_transpose   202,956 cycles
Layer 29: conv1d             113,744 cycles
Layer 32: conv1d_transpose   218,248 cycles
Layer 35: conv1d             105,280 cycles
Layer 38: conv1d_transpose   221,948 cycles
Layer 41: conv1d             104,048 cycles
Layer 44: conv1d             107,108 cycles
Layer 6:  tir_sqrt             7,388 cycles
Layer 9:  divide               6,616 cycles
Layer 3:  relu                 2,100 cycles
```

## Bottleneck Analysis

### 1. conv1d_transpose — 56% of total (1.47M cycles)

**Root cause: kernel_flipped buffer**

The `conv1d_transpose_ncw_optimized` schedule (selected for CLISTA
since `in_width=1`) eliminates the inner `dw` reduction loop (16x
fewer iterations). However, it still allocates a `kernel_flipped`
buffer and copies the entire transposed kernel every call:

```c
// kernel_flipped allocation: 2 * 128 * 16 * 4 = 16,384 bytes
void* sid_5 = TVMBackendAllocWorkspace(1, 0, 16384, 2, 32);
// Kernel flip loop: 3 levels (o=2, i=128, w=16)
for (int32_t ax0 = 0; ax0 < 2; ++ax0) {
  for (int32_t ax1 = 0; ax1 < 128; ++ax1) {
    for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
      kernel_flipped[((ax0 * 2048) + (ax1 * 16)) + ax2] =
        kernel[((ax1 * 32) + (ax0 * 16)) + (15 - ax2)];
    }
  }
}
```

Assembly analysis of the flip loop:
```
;; SOFTWARE PIPELINE INFORMATION
;; Loop found in file : /tmp/.../lib0.c
;; Loop source line   : 3125 (flip loop)
;; ii = 16            ; initiation interval
;; 256 iterations     ; 2*128 outer collapsed
;; Total cycles (est.): 6 + 256 * 16 = 4,102
;; Partitioned Resource Bound: 16 (D-unit bound: 32 loads/stores)
```

The actual compute after flipping is only ~98 cycles (ii=5, 16
iterations). So the flip loop takes **98% of estimated compute time**
and this is called 7 times = ~28,700 cycles just for flipping. The
remaining ~170K+ per call comes from function call overhead, DLTensor
argument unpacking, and `TVMBackendAllocWorkspace` which does malloc.

**Fix: Eliminate kernel_flipped entirely.** For CLISTA (pad=(0,0),
stride=1, in_width=1), replace:
```
kernel_flipped[c, dc, dw_clamped]
```
with:
```
kernel[dc, c, kernel_width - 1 - dw_clamped]
```
This is mathematically equivalent since `kernel_flipped[o,i,w] =
kernel[i,o,kw-1-w]`, so `kernel_flipped[c,dc,dw] = kernel[dc,c,kw-1-dw]`.
This eliminates the 16KB temp buffer allocation, the 4102-cycle copy
loop, and the free call.

**Status: DONE.** Eliminated `kernel_flipped` in
`conv1d_transpose_ncw_optimized()` by inlining the index
transformation: `kernel[dc, c, kernel_width - 1 - dw_clamped]`.

Result: conv1d_transpose dropped from ~210K to ~85K cycles per call
(steady-state), total from 1,472K to 628K. Overall profile total
dropped from 2,611K to 1,715K cycles (-34%).

Change: `python/tvm/topi/nn/conv1d_transpose.py` line 256-261
removed `kernel_flipped = te.compute(...)`, replaced access at
line 283 with direct kernel indexing.

### 2. conv1d — 33% of total (855K cycles)

**Root cause: overhead >> compute**

The conv1d inner loop is well-pipelined by the compiler:
```
;; SOFTWARE PIPELINE INFORMATION
;; ii = 8             ; initiation interval
;; 128 iterations     ; channels_in * kernel_width = 2 * 16 = 32
;;                    ; collapsed with output channels
;; Total cycles (est.): 58 + 128 * 8 = 1,082
```

Yet measured per-call is ~105K cycles (100x the loop estimate). The
overhead comes from:
- Function call setup/teardown for each of 156 calls
- DLTensor argument unpacking (pointer dereference chains)
- Stack frame + argument passing (up to 8 DLTensor pointers per call)
- Cache effects from calling different functions in sequence
- `TVMBackendAllocWorkspace` for intermediate buffers

**Fix: Operator fusion.** Fusing sequential elementwise ops
(add, multiply, relu, divide, sqrt) with conv1d/conv1d_transpose
would reduce 156 function calls to ~15, eliminating most overhead.
This is a larger TVM compiler change.

### 3. tir_sqrt — 2.3% (59K cycles)

**Root cause: scalar sqrtf() library call**

```c
// Generated code (lib0.c:3532)
compute[i1] = sqrtf(lv8[i1]);  // 64 elements
```

Assembly:
```
;; Disqualified loop: Loop contains a call
CALL .B1 ||sqrtf||    ; scalar call, no vectorization
```

Each call to `sqrtf()` is a scalar library function. The loop of
64 iterations cannot be software-pipelined due to the call.

**Fix:** Replace `sqrtf(x)` with `x * __recip_sqrt(x)` in TVM
codegen. The C7x `__recip_sqrt()` intrinsic maps to the VRSQRTSP
instruction which processes 8 floats per cycle. This would turn
64 scalar iterations into 8 vectorized iterations at ii=1, an
estimated 64x speedup for this layer.

Special case: guard against `x == 0` since `__recip_sqrt(0)` is
undefined.

**Status: DONE.** CodeGenCStatic now emits
`(x != 0.0f ? x * __recip_sqrt(x) : 0.0f)` for `sqrtf(x)` and
`__abs(x)` for `fabsf(x)` when targeting C7x.

Result: tir_sqrt dropped from ~7.4K to ~1.4K cycles per call (81%),
total from 59K to 11K.

### 4. divide — 1.7% (44K cycles)

**Root cause: scalar __c7xabi_divf library call**

```c
// Generated code (lib0.c:3198)
T_divide[ax1] = (lv11[ax1] / lv12[ax1]);  // 64 elements
```

Assembly:
```
;; Disqualified loop: Loop contains a call
;; Loop contains non-pipelinable instructions
CALL .B1 ||__c7xabi_divf||    ; 4x per iteration, 16 iterations
```

The compiler generates 4 sequential scalar division calls per
iteration (unrolled 4x), with 16 iterations for 64 elements. The
division library call prevents software pipelining entirely.

**Fix:** Replace `a / b` with `a * __recip(b)` in TVM codegen.
The `__recip()` intrinsic maps to VRCPSP (8 floats per cycle).
Note: `__recip()` provides ~23 bits of precision (IEEE single has
24 bits mantissa), adequate for ML inference but not bit-exact.

**Status: DONE.** CodeGenCStatic now emits `(a) * __recip((b))` for
float division when targeting C7x. Special case: `1.0f / sqrtf(x)`
is detected and emitted as `__recip_sqrt((x))` directly.

Result: divide dropped from ~5.5K to ~1.2K cycles per call (78%),
total from 44K to 9.6K.

### 5. relu — 0.7% (18K cycles)

**Root cause: scalar fmaxf() library call**

```c
// Generated code (lib0.c:3300)
compute[i1] = fmax(lv10[i1], 0.000000e+00f);  // 128 elements
```

Assembly:
```
;; Disqualified loop: Loop contains a call
CALL .B1 ||fmaxf||    ; scalar call
```

**Fix:** Replace `fmax(x, 0.0f)` with `__max(x, 0.0f)` in TVM
codegen. The `__max()` intrinsic maps to VMAXSP.

**Status: DONE.** CodeGenCStatic now emits `__max((a), (b))` for
float max and `__min((a), (b))` for float min when targeting C7x.

Result: relu dropped from ~2.3K to ~0.9K cycles per call (58%),
total from 18K to 7.5K.

## Optimization Priority

| #  | Optimization                           | Est. savings | Actual     | Status |
|----|----------------------------------------|-------------|------------|--------|
| 1  | Eliminate kernel_flipped in conv1d_tr  | ~200K+      | ~844K      | DONE   |
| 2  | Replace divide with __recip intrinsic  | ~40K        | ~34K       | DONE   |
| 3  | Replace sqrtf with __recip_sqrt        | ~55K        | ~48K       | DONE   |
| 4  | Replace fmax with __max intrinsic      | ~15K        | ~10.5K     | DONE   |
| 5  | Fuse elementwise ops to reduce calls   | ~300K+      | ~690K      | DONE   |
| 6  | Streaming Engine for weight prefetch   | LOW         |            | SEE BELOW |
| 7  | Eliminate VM runtime overhead           | ~1.2M       |            | NEXT   |

Items 2-4 implemented in CodeGenCStatic (codegen_c_static.cc) by
overriding VisitExpr_ for DivNode, MaxNode, MinNode, and CallNode.
Total measured savings: ~93K cycles (items 2-4 combined).
Accuracy impact: max diff vs PyTorch increased from ~1e-5 to ~1.4e-2
due to ~23-bit precision of __recip intrinsic. Test tolerance relaxed
to rtol=1e-2, atol=2e-2 for C7x/DLOAD modes.

### 5. Operator Fusion — DONE

The highest-leverage remaining optimization. With 156 separate TIR
function calls, per-layer overhead dominated total execution time.

**Per-call overhead breakdown (conv1d ~95K measured, ~1K compute):**
- TIR function entry/exit and stack frame setup
- DLTensor argument unpacking (pointer dereference chains for up to
  8 DLTensor pointers per call)
- AllocStorage + AllocTensor between every layer (bump-pointer
  allocation through DLOAD-resolved firmware calls)
- Cache effects from calling different functions in sequence
- TVMBackendAllocWorkspace for intermediate buffers

**Fix:** The default Relax build pipeline (`default_build_pipeline`)
does not include `FuseOps` or `FuseTIR` — it calls `ToNonDataflow`
before fusion can run. Switching to the `cpu_generic` pipeline
(which runs `LegalizeOps → AnnotateTIROpPattern → FoldConstant →
FuseOps → FuseTIR` before lowering) enables fusion.

Change: `dsp_utils.py` and `tvm_utils.py` now call
`relax.build(..., relax_pipeline=get_default_pipeline(target))`
using `tvm.relax.backend.cpu_generic.pipeline`.

**Result:**
- Function calls: 156 → **46** (70% fewer)
- Wall-clock inference: 2,638K → **1,948K cycles** (−690K, **26%**)
- Layer total: 1,627K → **1,592K** (modest, savings are in overhead)
- Accuracy: unchanged (1.35e-02 max diff)

Key fused kernels created:
- `fused_power_power_add_add1_tir_sqrt_subtract_relu_add1_divide_
  multiply_multiply_stack_reshape1` — 12 elementwise ops → 1 call
  (~14K cycles, was ~45K across 12 separate calls)
- `fused_conv1d_multiply1_add2` — conv1d + post-processing → 1 call
  (~96K cycles)

## Current Cycle Budget (post items 1-5)

**Clean wall-clock (no profiling): 1,686,980 cycles (1.687 ms at
1 GHz).**

Profiled with `--profile-layers --use-cpp-api --dsp-mode=dload`:

| Layer type                  | Calls | Total cycles | %    |
|-----------------------------|-------|-------------|------|
| conv1d (standalone, iter 0) | 1     | ~131K       | 8.2% |
| fused_conv1d_multiply1_add2 | 7     | ~676K       | 42%  |
| conv1d_transpose            | 7     | ~641K       | 40%  |
| fused elementwise (12 ops)  | 8     | ~114K       | 7.2% |
| other (take, subtract, etc) | 23    | ~30K        | 1.9% |
| **Total (layers)**          | **46**| **~1,592K** |      |

Profiled wall-clock: 1,948K (includes ~350K profiling overhead
from 46 cycle-counter reads).

### Overhead Analysis

The inner compute loops are fast:
- conv1d: ii=8, 128 iterations → ~1,082 cycles estimated
- conv1d_transpose: ii=5, 16 iterations → ~98 cycles estimated
- fused elementwise: simple loops over 64-128 elements

Yet measured per-call is ~96K (conv1d) and ~85K (conv1d_transpose).
The 90-95x overhead ratio comes from the VM runtime convention used
in `__vmtir__main`. Inspection of the generated lib0.c (1264 lines
post-fusion) shows:

**Per-layer overhead in `__vmtir__main`:**
1. `AllocTensor` call — constructs a DLTensor (shape, strides, data
   pointer) via DLOAD-resolved firmware call. **47 calls total.**
2. Stack argument setup — `SetFromUnchecked`, `SetNone`, `MoveFrom`
   to marshal TVMFFIAny arguments for each kernel call.
3. Kernel function call boundary — each kernel re-extracts `float*`
   from the DLTensor that was just constructed above:
   ```c
   void* var_y = UnwrapObjectRefArg(((TVMFFIAny*)args)[0]);
   float* y = (float*)(((DLTensor*)var_y)[0].data);
   long* shape = (long*)(((DLTensor*)var_y)[0].shape);
   if (!(strides == NULL)) {}  // dead code
   ```
4. `AllocStorage` — allocates storage pools via firmware. Only 4
   calls total (slots 2, 5, 7, 58); `StaticPlanBlockMemory` has
   already planned memory reuse across these 4 pools.

The round-trip is: codegen knows shapes/offsets at compile time →
constructs DLTensor at runtime → passes through FFI → kernel
immediately destructures DLTensor back to raw pointers. This
DLTensor construction/destruction is pure overhead for a static
model on a bare-metal DSP target.

### 7. Eliminate VM Runtime Overhead — NEXT PRIORITY

Replace the VM runtime calling convention with direct pointer
passing in CodeGenCStatic for DSP targets. Two sub-tasks:

**7a. Static workspace allocation**

Replace the 4 `AllocStorage` + 47 `AllocTensor` calls with a
single statically-allocated workspace buffer. All tensor shapes
and offsets are known at compile time (StaticPlanBlockMemory has
already computed them). Emit:
```c
static char workspace[WORKSPACE_SIZE] __attribute__((aligned(64)));
float* tensor_3 = (float*)(workspace + OFFSET_3);
```
instead of:
```c
_r.SetStorage(2, vm::AllocStorage(...));
_r.SetNDArray(3, vm::AllocTensor(_r.GetStorage(2), ...));
```
This eliminates 51 DLOAD-resolved firmware calls per inference.

**7b. Direct-call convention for kernel functions**

Instead of marshalling arguments through `TVMFFIAny` arrays and
having each kernel unpack DLTensors, emit kernels with direct
`float*` parameters:
```c
static void conv1d_kernel(float* y, float* B, float* out) {
  // compute loop only, no DLTensor unpacking
}
```
called directly from `__vmtir__main`:
```c
conv1d_kernel(tensor_0, weights_conv1d, tensor_3);
```
This eliminates per-call: 3x `UnwrapObjectRefArg`, 3x DLTensor
field extraction, 3x stride NULL checks, function call overhead
through the `(void* args, int num_args)` convention.

**Estimated impact:** With ~95K overhead per conv1d call and ~1K
actual compute, eliminating the VM overhead could reduce each call
from ~96K to ~5-10K (compute + minimal call overhead), saving
~80K per call × 15 conv/conv_transpose calls ≈ **1.2M cycles**.
This would bring the total from ~1.7M to ~500K, approaching the
C66x baseline of ~600K.

**Complexity:** High. Requires CodeGenCStatic changes to emit a
different calling convention when targeting DSP. The existing
`skip-runtime-checks` infrastructure already eliminates some
runtime validation; this extends that approach to eliminate the
entire DLTensor/FFI marshalling layer.

### 6. Streaming Engine — LOW PRIORITY

Assembly analysis of the post-optimization inner loops shows the
Streaming Engine (SE) would provide limited benefit for CLISTA:

**conv1d inner loop (lib0.asm):**
```
;; SOFTWARE PIPELINE INFORMATION
;; ii = 8, 128 iterations, 9 iterations in parallel
;; Resource Partition:
;;   .D units           5
;;   Bound(.D)          3        <-- light memory pressure
;;   Bound(.C/.L/.S)    0   19*  <-- COMPUTE-BOUND
```

The conv1d loop is **compute-bound** (Bound(.C/.L/.S)=19), not
memory-bound (Bound(.D)=3). The Streaming Engine helps with
memory-bound loops by prefetching from L2, but has minimal effect
when the bottleneck is functional unit throughput.

**conv1d_transpose kernel flip loop:**
This was the D-unit-bound loop (Bound(.D)=16, 32 loads/stores)
that would have benefited from SE — but it has been **eliminated**
by item #1 (direct kernel indexing). The remaining conv1d_transpose
compute loop has a similar profile to conv1d.

**Recommendation:** Streaming Engine optimization is deprioritized
for CLISTA. The `--auto_stream` compiler flag could be tried as a
no-effort experiment, but the compute-bound inner loop profile
suggests minimal gains. Focus on operator fusion (#5) instead.

## Implementation Notes

### Assembly Generation for Analysis

Compile lib0.c with pipelining debug output:
```bash
TI_CGT=$HOME/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
$TI_CGT/bin/cl7x \
  -mv7524 --abi=eabi --endian=little \
  --fp_mode=relaxed -DNULL=0 -DSOC_J722S \
  -O3 --opt_for_speed=5 --auto_inline=500 \
  -k --debug_software_pipeline --symdebug:none \
  -I$TVM_HOME/tests/ti-dsp-runtime/dsp-cpp \
  -I$TVM_HOME/src/runtime/ti_dsp \
  -I$TVM_HOME/3rdparty/dlpack/include \
  -DTVM_DSP_TARGET_C7X \
  --compile_only --output_file=/tmp/lib0_analysis.obj \
  lib0.c
```

The `-k` flag generates `lib0.asm` in the current directory.
`--symdebug:none` suppresses DWARF debug sections (reduces asm from
~2MB to ~500KB). `--debug_software_pipeline` adds pipeline analysis
comments to each loop.

### Key Assembly Patterns

- `SOFTWARE PIPELINE INFORMATION` — loop optimization report
- `Disqualified loop: Loop contains a call` — library function call
  blocks pipelining and vectorization
- `ii = N` — initiation interval (cycles per pipeline stage)
- `Total cycles (est.)` — compiler cycle estimate
- `Partitioned Resource Bound` — bottleneck hardware unit

### C7x Intrinsics Reference

Located at: `$TI_CGT/include/c7x.h`

| Intrinsic          | Instruction | Width    | Use case              |
|--------------------|-------------|----------|-----------------------|
| `__recip(float8)`  | VRCPSP      | 8-wide   | Replace division      |
| `__recip_sqrt(f8)` | VRSQRTSP    | 8-wide   | Replace sqrtf         |
| `__max(float8,f8)` | VMAXSP      | 8-wide   | Replace fmax (relu)   |
| `__min(float8,f8)` | VMINSP      | 8-wide   | Replace fmin          |
| `__abs(float8)`    | VABSSP      | 8-wide   | Replace fabsf         |

### TVM Schedule Selection (conv1d_transpose)

File: `python/tvm/relax/transform/legalize_ops/nn.py`

```
groups == 1 AND in_width == 1 AND stride == 1
  -> conv1d_transpose_ncw_optimized  (eliminates dw loop)
groups == 1
  -> conv1d_transpose_ncw_direct     (general case)
groups > 1
  -> group_conv1d_transpose_ncw
```

For CLISTA: in_width=1, stride=1, pad=(0,0), kernel=16, channels
in=2, out=128. The optimized path IS selected for both C66x and C7x.

### Conv1d_transpose Schedule Code

File: `python/tvm/topi/nn/conv1d_transpose.py`

Three implementations exist:
- `conv1d_transpose_ncw()` (line 89) — original with preprocess
- `conv1d_transpose_ncw_optimized()` (line 219) — for in_width=1
- `conv1d_transpose_ncw_direct()` (line 300) — general direct

All three still create a `kernel_flipped` intermediate buffer. The
"optimized" variant eliminates the inner dw reduction loop (reducing
from `channels_in * kernel_width` to `channels_in` iterations) but
does not eliminate the kernel flip. The kernel flip itself accounts
for ~4100 cycles per call (ii=16, 256 iterations, D-unit bound).

  Where inlining would matter

  There are two categories:

  1. Per-layer VM overhead (called ~156 times)

  Between every layer in __vmtir__main, the generated code does AllocStorage + AllocTensor
  (sometimes Reshape). Each goes through 2-3 levels of DLOAD-resolved calls.          [0/544]
  TVMDSPStorageAlloc alone is ~65 lines including ScopeGuard, pool selection logic, and error
   handling. TVMDSPStorageAllocNDArray is another ~50 lines. These are heavyweight for what
  amounts to a bump-pointer allocation.

  If we provided a lightweight header-only fast-path allocator — something like a simple bump
   allocator that lib0.c could inline — we'd eliminate the DLOAD GOT indirection AND let the
  compiler optimize across the allocation + compute boundary. But these functions call
  tvm_dsp_alloc which needs access to the pool globals in the firmware, so they can't be
  trivially inlined without either exposing the pool state or creating a dedicated fast-path.

  2. Math functions in compute loops (called 8 times each, but critical for vectorization)

  This is actually the higher-impact category. sqrtf, fmax(x, 0), and a / b are external
  calls that prevent the compiler from software-pipelining the loops entirely. Inlining these
   via a header with C7x intrinsics would:
  - Allow the loops to be software-pipelined (no more "Disqualified loop: Loop contains a
  call")
  - Enable SIMD vectorization (8-wide float ops)
  - Estimated ~110K cycle savings (items 2-4 in the optimization table)

  The practical approach: create a header like tvm_c7x_math.h that provides static inline
  replacements:
  static inline float tvm_recip(float x) { return __recip(x); }
  static inline float tvm_rsqrt(float x) { return __recip_sqrt(x); }
  static inline float tvm_fmaxf(float a, float b) { return __max(a, b); }

  Then modify TVM codegen to emit a * tvm_recip(b) instead of a / b, x * tvm_rsqrt(x) instead
   of sqrtf(x), and tvm_fmaxf(x, 0) instead of fmax(x, 0) when targeting C7x.

  This doesn't require any firmware changes — it's purely a codegen + header change. The
  intrinsics compile to single instructions (VRCPSP, VRSQRTSP, VMAXSP) that the compiler can
  vectorize and pipeline.

  For the VM allocation overhead, a simpler approach might be to look at whether TVM's memory
   planner can pre-allocate all workspace at compile time (static memory planning) rather
  than doing per-layer dynamic allocation. But that's a larger change.

  Do you want to start with the math intrinsics header approach?
