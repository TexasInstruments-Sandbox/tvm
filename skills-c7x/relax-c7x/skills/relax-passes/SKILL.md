---
name: relax-passes
description: "Writing TVM Relax and TIR passes for C7x. Use when: implementing a new compiler pass (pattern matching, fusion, legalization, lowering), using DFPattern API (wildcard, is_op, annotations, constraint callbacks), writing PyExprMutator-based lowering, emitting call_extern via te.extern/call_te, constructing TIR PrimFuncs, injecting DMA/buffer operations in TIR, wiring passes into pipeline.py, or debugging pass ordering. NOT for using existing passes (see cstatic) or operator kernel implementation (see dsp-ops)."
---

# Writing Relax and TIR Passes

Patterns and API for implementing compiler passes in this TVM codebase.

The tree is TVM 0.27. If you know TVM from 0.23 or earlier, read "0.27 API names" below first:
`tir` is `tirx`, `struct_info` is `.ty`, `relax.Constant` and `relax.PrimValue` are gone, and
several of these renames fail only when a pass runs, not at import.

## Before Writing a Custom Pass

Prefer leveraging existing TVM/Relax infrastructure and optimization passes to the maximum extent. Custom passes add maintenance burden and can break when TVM upstream changes. Only write a custom pass when:

1. **No existing pass handles the pattern** — check `relax.transform.*` (FuseOps, FuseTIR, LegalizeOps, FoldConstant, DeadCodeElimination, etc.) first
2. **LegalizeOps custom map is insufficient** — if a single op needs a different lowering, a legalization function is cheaper than a full pass
3. **The transformation requires cross-op context** — e.g. QDQ parameter extraction across multiple nodes, or fusing a specific multi-op pattern into one extern call

If the goal is just scheduling or tiling an existing TIR kernel, use the TE schedule API (`cache_read`, `split`, `reorder`, `compute_at`) rather than a TIR pass.

## Pass Types

| Type | Decorator | Signature | Use Case |
|------|-----------|-----------|----------|
| Module pass (Relax) | `@tvm.transform.module_pass(opt_level=0, name="X")` | `transform_module(self, mod, ctx) -> IRModule` | Pattern matching, fusion, lowering |
| PrimFunc pass (TIR) | `@tvm.tirx.transform.prim_func_pass(opt_level=0, name="X")` | `fn(func, mod, ctx) -> PrimFunc` | Buffer manipulation, DMA injection |

## Pattern 1: FuseOpsByPattern + PyExprMutator Lowering

The most common pattern in this codebase. Two phases: match → lower.

### Phase 1: Pattern Matching

```python
from tvm.relax.dpl import wildcard, is_op

def _my_pattern():
    data = wildcard()
    weight = wildcard()
    scale = wildcard()
    dq = is_op("relax.dequantize")(data, scale, wildcard())
    matmul = is_op("relax.matmul")(dq, weight)
    quant = is_op("relax.quantize")(matmul, wildcard(), wildcard())

    annotations = {"data": data, "weight": weight, "scale": scale, "matmul": matmul}
    return quant, annotations, _check_constraints

def _check_constraints(ctx) -> bool:
    """Constraint callback — return False to reject match."""
    w = ctx.annotated_expr["weight"]
    if not isinstance(w.ty, relax.TensorType) or w.ty.shape is None:
        return False
    shape = w.ty.shape  # 0.27: .ty (a Type), not .struct_info
    return int(shape[-1]) % 64 == 0  # dimension alignment
```

Apply patterns:
```python
from tvm import relax

patterns = [
    ("mypkg.op_variant_a", *_variant_a_pattern()),
    ("mypkg.op_variant_b", *_variant_b_pattern()),
]
# Order: longest/most-specific first (greedy matching)
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)
```

### Phase 2: Lowering with PyExprMutator

```python
from tvm.relax.expr_functor import mutator, PyExprMutator

@mutator
class _MyLowerer(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod
        self.count = 0

    def visit_call_(self, call: relax.Call):
        # Only intercept calls to composite functions
        if not isinstance(call.op, relax.GlobalVar):
            return super().visit_call_(call)

        func = self.mod[call.op]
        composite = func.attrs.get("Composite", "") if func.attrs else ""
        if not composite.startswith("mypkg."):
            return super().visit_call_(call)

        # Map params to args
        arg_map = dict(zip(func.params, call.args))

        # Walk bindings to extract named values
        data = self._resolve(func, arg_map, "data")
        weight = self._resolve(func, arg_map, "weight")

        # Compute constants (numpy). A tensor constant is tvm.ir.GenericConst;
        # its tensor is .value (0.23: relax.Constant with .data).
        w_np = weight.value.numpy()
        bias_np = compute_bias(w_np)
        # ti_tensor_const keeps the numpy dtype; relax.const narrows int64/float64
        # to 32 bit when no dtype is given.
        bias_const = ti_tensor_const(np.ascontiguousarray(bias_np))

        # Emit replacement via te.extern
        result = self.builder_.call_te(
            _te_mmalib_call, data, weight, bias_const,
            primfunc_name_hint="mmalib_my_op",
        )
        self.count += 1
        return result
```

### te.extern for call_extern

```python
from tvm import te, tirx

def _te_mmalib_call(data, weight, bias):
    N, K = data.shape[0], weight.shape[1]
    return te.extern(
        [N, K],  # output shape
        [data, weight, bias],  # inputs
        lambda ins, outs: tirx.call_extern(
            "int32",  # return type
            "mmalib_my_op",  # extern function name
            ins[0].data, ins[1].data, ins[2].data,
            outs[0].data,
            N, K,  # dimension args
        ),
        name="mmalib_my_op",
        dtype="int8",  # output dtype
    )
```

### Putting it together (module pass)

```python
@tvm.transform.module_pass(opt_level=0, name="FuseMyOp")
class FuseMyOp:
    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        # Phase 1: Pattern match
        patterns = [("mypkg.variant_a", *_variant_a_pattern())]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: Lower composites
        lowerer = _MyLowerer(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                new_func = lowerer.visit_expr(func)
                lowerer.builder_.update_func(gv, new_func)
        mod = lowerer.builder_.get()

        # Cleanup
        if lowerer.count > 0:
            mod = relax.transform.DeadCodeElimination()(mod)
        return mod
```

## Pattern 1b: FuseOpsByPattern + MergeCompositeFunctions (Subgraph Offload)

`FuseOpsByPattern` is the general mechanism for identifying a sequence of ops to offload — whether to an optimized C7x custom kernel, MMALIB, or TIDL. When multiple adjacent matched composites should be handled as a single unit, `MergeCompositeFunctions` groups them into a subgraph function:

```python
@tvm.transform.module_pass(opt_level=0, name="PartitionForBackend")
class PartitionForBackend:
    def transform_module(self, mod, _ctx):
        # Phase 1: Mark individual ops/sequences as composites
        patterns = [
            ("mybackend.conv2d_bias_relu", *_conv2d_bias_relu_pattern()),
            ("mybackend.relu", *_relu_pattern()),
            ...
        ]
        mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

        # Phase 2: Merge adjacent composites into subgraph functions.
        # MergeCompositeFunctions takes no arguments — it infers the codegen
        # name from each composite's pattern-name prefix (e.g.
        # "mybackend.conv2d_bias_relu" -> "mybackend").
        mod = relax.transform.MergeCompositeFunctions()(mod)
        # Result: functions with attrs["Codegen"] = "mybackend" containing
        # multiple composites that form a connected subgraph
        return mod
```

**When to use Pattern 1 vs 1b:**
- **Pattern 1** (FuseOpsByPattern + Mutator): Each matched pattern is independently lowered to a `call_extern`. Use when TVM controls scheduling and each fused op is a separate extern call (e.g. MMALIB conv2d, custom vectorized kernels).
- **Pattern 1b** (FuseOpsByPattern + MergeCompositeFunctions): Multiple matched patterns are grouped into subgraph functions for external compilation. Use when an external runtime handles the entire subgraph — scheduling, memory, multi-op fusion (e.g. TIDL, or any custom backend that processes a connected graph rather than individual ops).

After `MergeCompositeFunctions`, the subgraph functions are typically processed by a custom compiler (e.g. `TIDLOffloadCompiler.tidl_import()`) that replaces them with `call_extern` stubs pointing to the external runtime.

### Non-composite bridge and cyclic subgraph merging (fixed)

`MergeCompositeFunctions` had a cycle-detection gap: if a non-composite op B sat between
two composite groups (G1 → B → G2, where G2 also had a direct skip-connection arg from G1),
the algorithm could merge G1 and G2 into one subgraph.  The local merge check only looked
at G1's current `group_deps_` set — which was empty when no composite had yet consumed G1's
output — so the skip-connection arg passed unchecked.  The result was a cyclic module that
crashed `relax.build` with SSA ordering violations or a stale DataflowVar reference.

**Fix (implemented):** `UpdateGroupDependencies` in
`src/relax/transform/merge_composite_functions.cc` now injects a self-dep into the upstream
composite group whenever a non-composite group reads from it:

```cpp
// When a non-composite group consumes a composite group's output, mark the
// composite as "closed" so future skip-connection merges are blocked.
if (!GetCodegenName(group_root) && GetCodegenName(arg_group_root)) {
    group_deps_[arg_group_root].insert(arg_group_root);
}
```

This makes the upstream composite appear in `parent_dependencies` for any later composite
that takes it as a direct arg, causing `GetGroupsToMerge` to reject the merge.

Regression tests in `tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py`
(`TestNonCompositeBridgeCycle`).

## Pattern 2: Direct Mutator (No DFPattern)

When the pattern is a simple linear chain, skip `FuseOpsByPattern` and match manually in `visit_call_`:

```python
@mutator
class _DirectMutator(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self._bindings = {}  # var -> value map

    def visit_call_(self, call: relax.Call):
        if call.op.name != "relax.matmul":
            return super().visit_call_(call)

        # Trace backward through bindings
        lhs = self._resolve_var(call.args[0])
        if not (isinstance(lhs, relax.Call) and lhs.op.name == "relax.dequantize"):
            return super().visit_call_(call)

        # ... validate and emit replacement
```

Pre-scan bindings (required for backward tracing):
```python
def _pre_scan_bindings(func):
    """Build var -> value map from function body."""
    var_map = {}
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding, relax.VarBinding):
                var_map[binding.var] = binding.value
    return var_map
```

## Pattern 3: TIR PrimFunc Pass

For post-lowering manipulation (buffer injection, DMA):

```python
import tvm_ffi
from tvm import tirx

def InjectMyTransform(param=128):
    @tvm.tirx.transform.prim_func_pass(opt_level=0, name="InjectMyTransform")
    def _pass(func: tirx.PrimFunc, mod, ctx):
        return _transform(func, param)
    return _pass

def _transform(func, param):
    body = func.body

    # Find target statements (0.27: tvm_ffi.structural_walk replaces post_order_visit)
    found = []
    def _find(stmt):
        if _is_target_call(stmt.value):
            found.append(stmt)
    tvm_ffi.structural_walk(body, (tirx.Evaluate, _find), order="post")
    if not found:
        return func

    # Build new statements. AllocBuffer, Bind and DeclBuffer are flat in 0.27: they have
    # no body, and their scope is the rest of the enclosing SeqStmt.
    l2_buf = tirx.decl_buffer([extent], "int8", name="l2_buf", scope="global.l2sram")
    dma_copy = tirx.Evaluate(tirx.call_extern("int32", "tvm_dsp_dma_copy", ...))
    dma_wait = tirx.Evaluate(tirx.call_extern("int32", "tvm_dsp_dma_wait", ...))

    # Compose: alloc -> dma_copy -> dma_wait -> original body. Carry the function's
    # DeclBuffer statements over if you rebuild the body from scratch (see pitfalls).
    return func.with_body(tirx.SeqStmt([tirx.AllocBuffer(l2_buf), dma_copy, dma_wait, body]))
```

Working example: `ti_mmalib_inject_dma.py` (`_l2_buffer`, `_collect_decl_buffers`, `_with_l2_allocs`).

## Pattern 4: Custom Legalization

Hook into `LegalizeOps` with a custom map:

```python
def _legalize_matmul(bb: relax.BlockBuilder, call: relax.Call):
    """Replace relax.matmul with custom implementation."""
    if not _is_eligible(call):
        return call  # fall through to default
    # ... emit replacement
    return bb.call_te(_te_impl, *call.args)

# In pipeline:
relax.transform.LegalizeOps(customize_legalize_map={
    "relax.matmul": _legalize_matmul,
    "relax.nn.conv2d": _legalize_conv2d,
})
```

## Wiring into pipeline.py

Location: `python/tvm/relax/backend/cpu_generic/pipeline.py`

```python
def legalize_passes(target):
    passes = []
    is_c7x = target.kind.name == "c_static_lib" and str(target.attrs.get("mcpu", "")) == "c7x"
    use_mmalib = target.attrs.get("mmalib", False)

    if is_c7x and use_mmalib:
        passes.append(FuseMyOp())  # insert before or after existing passes

    passes.append(relax.transform.FuseOps())
    passes.append(relax.transform.FuseTIR())
    return passes
```

Rules:
- MMALIB QDQ passes run BEFORE `FuseQDQToInt8Conv2D` (to see intact QDQ graph)
- Pattern passes ordered longest-match-first
- `FoldConstant()` after any pass that creates new constants
- `DeadCodeElimination()` is implicit in most passes (each pass cleans up after itself)

## Key API Reference

| API | Purpose |
|-----|---------|
| `wildcard()` | Match any expression |
| `is_op("relax.nn.conv2d")(arg1, arg2)` | Match specific op with args |
| `ctx.annotated_expr["name"]` | Access matched sub-expression in check callback |
| `relax.transform.FuseOpsByPattern(patterns, bind_constants=False)` | Phase 1: match and group into composites |
| `relax.transform.MergeCompositeFunctions()` | Merge adjacent composites into subgraph functions (codegen name inferred from each composite's pattern-name prefix) |
| `@mutator` + `PyExprMutator` | Phase 2: rewrite matched composites |
| `self.builder_.call_te(fn, *args, primfunc_name_hint="X")` | Emit te function as TIR |
| `self.builder_.emit(relax.op.abs(x))` | Emit Relax op into current block |
| `te.extern(shape, inputs, fcompute, name, dtype)` | Create call_extern tensor expression |
| `tirx.call_extern("int32", "func_name", *args)` | Direct extern call in TIR |
| `ti_tensor_const(np_array)` (`relax.transform.ti_const_utils`) | Wrap numpy as a Relax constant, dtype kept |
| `relax.const(value, dtype)` | Constant with an explicit dtype (narrows 64-bit if dtype is omitted) |
| `func.with_body(new_body)` | Replace PrimFunc body |
| `tvm_ffi.structural_walk(body, (NodeType, fn), order="post")` | Walk TIR tree |
| `tvm_ffi.structural_map(body, (NodeType, fn))` | Rewrite TIR tree |
| `var.same_as(other)` | Compare TVM Var identity (NOT `==` or `id()`) |

## Common Pitfalls

1. **Var comparison**: Use `var.same_as(other)`, never Python `==` or `id()`
2. **Constants**: Always `np.ascontiguousarray()` before wrapping, and use `ti_tensor_const` unless you pass an explicit dtype to `relax.const`
3. **Pattern order**: Longest/most-specific patterns first in the list (greedy matching)
4. **bind_constants=False**: Required when patterns reference constants that should remain as function params (not inlined)
5. **Pre-scan bindings**: `PyExprMutator.visit_call_` sees the call but not necessarily earlier bindings — build a var→value map if you need to trace backward
6. **Cleanup**: Always run `DeadCodeElimination()` after fusion to remove orphaned composites
7. **Composite parameter order**: `FuseOpsByPattern` orders a composite's parameters by the first use of each free variable in the block, and no `FusionPattern` field controls it. A pattern whose first op consumes the second operand (e.g. a `permute_dims` on the grid before `grid_sample`) puts that operand first, which matters to any backend that reads arguments positionally (TIDL does)
8. **Relax `Expr` is not truthy**: `if a and b:` on Relax expressions raises; write `is not None`
9. **Rebuilding a PrimFunc body**: after `FlattenBuffer`, parameters are aliased through `DeclBuffer(x_1, data=x.data)` statements. A pass that builds a new body must carry them over, or MakePackedAPI reports variables that are "used, but are not passed in as API arguments"

## 0.27 API names

The fork moved from TVM 0.23 to 0.27. These renames account for most broken pass code; some only
fail when the pass runs. Full list, including the C++ side: `relax-c7x:tvm-027-api`.

| 0.23 | 0.27 |
|------|------|
| `tvm.tir`, `from tvm.script import tir as T` | `tvm.tirx`, `from tvm.script import tirx as T`; schedule/block world is `tvm.s_tir` (`T.sblock`, `@T.prim_func(s_tir=True)`) |
| `tvm.tir.transform.prim_func_pass` | `tvm.tirx.transform.prim_func_pass`; pass-config keys are `tirx.*` |
| `expr.struct_info`, `TensorStructInfo`, `TupleStructInfo`, `FuncStructInfo` | `expr.ty`, `relax.TensorType`, `relax.TupleType`, `relax.FuncType`; `ret_struct_info` -> `ret_ty`, `sinfo_args` -> `ty_args` |
| `relax._ffi_api.UpdateStructInfo(e, s)` | `relax.expr._update_type(e, ty)` |
| `ty.dtype` (a string) | `ty.dtype` is a `PrimType`: compare `str(ty.dtype)` |
| `relax.Constant`, `c.data` | `tvm.ir.GenericConst`, `c.value` |
| `relax.PrimValue(x)`, `e.value.value` | a plain `IntImm`/`FloatImm` (`relax.prim_value(x)`), `e.value`; test with `tvm.ir.is_prim_expr(e)` |
| `var.name_hint` (Var, DataflowVar) | `var.name`. `GlobalVar.name_hint` and `emit(..., name_hint=)` are unchanged |
| `tvm.nd.array(a)` | `tvm.runtime.tensor(a)` |
| `PrimFunc.buffer_map` | gone: buffer parameters are `Var`s with a `BufferType` |
| `tir.Allocate(..., body)`, `tir.LetStmt(..., body)` | flat `tirx.AllocBuffer(buf)`, `tirx.Bind`; scope is the rest of the `SeqStmt`. `SeqStmt` rejects 0 or 1 elements |
| `tir.stmt_functor.post_order_visit` | `tvm_ffi.structural_walk(root, (NodeType, cb), order="post")` |
| `tvm.tir.Call`, `tir.StringImm` | `tvm.ir.Call(op, args, ret_ty=...)`, `tvm.ir.StringImm` (`tvm.tirx.Call` raises at pass run time) |
| `tvm.contrib.tar`, `tvm.contrib.cc` | `tvm.support.tar`, `tvm.support.cc` |

## Key Files (Examples)

| File | Pass Type | Technique |
|------|-----------|-----------|
| `ti_mmalib_qdq_fusion.py` | FuseOpsByPattern + Mutator | QDQ pattern → call_extern (688 lines) |
| `ti_mmalib_qdq_dwconv.py` | FuseOpsByPattern + Mutator | Depthwise variant |
| `ti_mmalib_qdq_fc.py` | FuseOpsByPattern + Mutator | FC/matmul variant with reshape |
| `ti_mmalib_i16_fc.py` | Direct Mutator | Manual matmul matching + runtime quant (332 lines) |
| `ti_mmalib_inject_dma.py` | TIR PrimFunc pass | Buffer injection, 0.27 flat AllocBuffer/DeclBuffer idioms |
| `ti_residual_add.py` | FuseOpsByPattern + Mutator | Residual pattern, int8 + int16 (557 lines) |
| `schedule_c7x_dma.py` | TIR scheduling pass | DMA tiling via TE schedule API |
| `cpu_generic/pipeline.py` | Pipeline wiring | Pass ordering and target gating (230 lines) |

All in `python/tvm/relax/transform/` (passes) or `python/tvm/relax/backend/cpu_generic/` (pipeline).

## Related Skills

- `relax-c7x:cstatic` — Full pipeline pass order (where your pass fits)
- `relax-c7x:mmalib-offload` — MMALIB-specific pass details and constraints
- `relax-c7x:dsp-ops` — Operator implementations called via call_extern
- `relax-c7x:tvm-027-api` — 0.23 -> 0.27 API changes
