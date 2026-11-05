---
name: tvm-027-api
description: "TVM 0.27 API differences for this fork (rebased from 0.23). Use when: porting or reading pre-0.27 TVM code, examples or docs; a Python error mentions struct_info, TensorStructInfo, relax.Constant, PrimValue, name_hint, tvm.tir, tir.transform, post_order_visit, buffer_map, tvm.nd, tvm.contrib.tar or \"CLI target string form is no longer supported\"; a C++ build error mentions VisitExpr_, VisitStmt_, LetStmtNode, AllocateNode, BufferLoadNode, DataType, ICHECK or IntImm value conversion; writing a CodeGenC subclass (c_static_lib codegen) or a TIR pass; or generated C silently changed (Target::str() is JSON, MakePackedAPI patterns). NOT for C7x-specific pipeline design (see cstatic, relax-passes) or build steps (see build)."
---

# TVM 0.27 API (for 0.23-era code)

This fork was rebased from TVM 0.23 onto 0.27. Most broken code is a rename from the tables below. Some
renames fail only when a pass actually runs (not at import), and a few change behaviour without any
error; those are called out.

Quick check that you are on the right tree: `python -c "import tvm; print(tvm.__version__)"` prints
`0.27.0`. A `0.23.0` means an old checkout or venv is first on the path.

## Python: Relax

| 0.23 | 0.27 |
|------|------|
| `expr.struct_info`, `sinfo` | `expr.ty` (a `Type`) |
| `TensorStructInfo`, `TupleStructInfo`, `FuncStructInfo`, `StructInfo` | `relax.TensorType`, `relax.TupleType` (= `tvm.ir.TupleType`), `relax.FuncType(params, ret, purity=True)`, `Type` |
| `ret_struct_info`, `sinfo_args` | `ret_ty`, `ty_args` |
| `relax._ffi_api.UpdateStructInfo(e, s)`, `relax.expr._update_struct_info` | `relax.expr._update_type(e, ty)` |
| `ty.dtype` was a string | `TensorType.dtype` is a `PrimType`: compare `str(ty.dtype)`; `.shape` is an Expr or `None` |
| `relax.Constant`, `c.data`, `c.data.numpy()` | `tvm.ir.GenericConst`, `c.value`, `c.value.numpy()` (`tvm.ir.Constant` is only the base class) |
| `relax.Constant(tvm.nd.array(a))` | `ti_tensor_const(a)` from `tvm.relax.transform.ti_const_utils`, or `relax.const(a, dtype)` |
| `relax.PrimValue(x)`, `e.value.value` | a plain `IntImm`/`FloatImm`: build with `relax.prim_value(x, dtype=None)`, read `e.value`, test `tvm.ir.is_prim_expr(e)` |
| `var.name_hint` (Var, DataflowVar) | `var.name`. `GlobalVar.name_hint` and `emit(..., name_hint=...)` are unchanged; a scripted rename must only touch Vars |
| `tvm.relax.expr.Call` | `relax.Call` still works; `tvm.relax.expr.Call` does not: `from tvm.ir import Call` |
| `tvm.nd.array(a)` | `tvm.runtime.tensor(a)` |

Behaviour changes with no error:
- **`relax.const` narrows 64-bit.** Without a dtype, an int64/float64 array becomes int32/float32. Use
  `ti_tensor_const` (keeps the numpy dtype) or pass `dtype=` explicitly.
- **Relax `Expr` is not truthy.** `if a and b:` raises "Cannot use and / or / not operator to Expr"; use `is not None`.
- **`FuseOpsByPattern` orders a composite's parameters by the first use of each free variable** in the
  block. No `FusionPattern` field (pattern, annotations, check, attrs_getter) controls it, so a pattern
  whose first op uses the second operand puts that operand first.
- **`FoldConstant` only evaluates `call_tir`.** Constant chains of high-level ops (arange, reshape,
  broadcast_to, ...) stay live; see `tvm.relax.backend.tidl.constant_chains` for a pass that folds them.
- **`relax.image.grid_sample` takes the grid as `[N, 2, H_out, W_out]`** (0.23 and PyTorch use
  `[N, H_out, W_out, 2]`). An old-layout grid gives a wrong output shape, not an error. Both frontends
  insert the permute. The docstring in `python/tvm/relax/op/image/image.py` still shows the old layout;
  the shape rule in `src/relax/op/image/resize.cc` is authoritative.
- **Python keeps the last class definition of a name.** A fork copy of an upstream frontend class
  (e.g. an ONNX converter) defined later in the file silently replaces upstream's.

## Python: TIR

| 0.23 | 0.27 |
|------|------|
| `tvm.tir`, `from tvm.script import tir as T` | `tvm.tirx`, `from tvm.script import tirx as T` |
| schedule / block APIs | `tvm.s_tir` (`s_tir.Schedule`, `sch.get_sblock`, `s_tir.SBlock`, `SBlockRealize`); TVMScript `T.block` -> `T.sblock`, `@T.prim_func(s_tir=True)` for block-based functions, `T.sblock_alloc_buffer` |
| `tvm.tir.transform.prim_func_pass` | `tvm.tirx.transform.prim_func_pass` |
| pass-config keys `tir.*` | `tirx.*` (`tirx.disable_vectorize`, `tirx.lower_dma_to_extern`, ...) |
| `LowerInitBlock`, `InjectSoftwarePipeline`, `LowerOpaqueBlock`, `LoopPartition` | in `s_tir.transform` |
| `PrimFunc.buffer_map` | gone: buffer parameters are `Var`s with a `BufferType` (`tirx.is_buffer_var(p)`, `tirx.buffer_data(p)`) |
| `tir.Allocate(..., body)` | flat `tirx.AllocBuffer(buf)`: no body, scope is the rest of the enclosing `SeqStmt`. The Python wrapper still accepts a legacy `body` and the enclosing constructor (`SeqStmt`, `PrimFunc`, `with_body`, ...) expands it into the flat form; write the flat form in new code |
| `tir.LetStmt(var, value, body)` | flat `tirx.Bind`, same scoping |
| buffer data pointer declared in a `LetStmt` | `tirx.DeclBuffer(buf, data=expr)`: `data` is keyword-only and required ("DeclBuffer requires a physical data binding") |
| `SeqStmt([s])` | `SeqStmt` rejects 0 or 1 elements and flattens nested `SeqStmt`s |
| `tir.stmt_functor.post_order_visit(root, fn)` | `tvm_ffi.structural_walk(root, (NodeType, fn), order="post")`; rewrite with `tvm_ffi.structural_map(root, (NodeType, fn))` |
| `tvm.tir.Call`, `tir.StringImm` | `tvm.ir.Call(op, args, ret_ty=...)`, `tvm.ir.StringImm`. `tvm.tirx.Call` raises AttributeError only when the pass runs |
| `tir.SizeVar` | `tirx.Var` |
| TIR pipeline function returns a pass list | returns `(pipeline, finalize_host_passes, finalize_device_passes)`; so does `get_tir_pipeline(name)`, despite its `-> Pass` annotation |
| `tvm.contrib.tar`, `tvm.contrib.cc`, `tvm.contrib.utils` | `tvm.support.tar`, `tvm.support.cc`, `tvm.support.utils` |
| `tvm.tir.*` in topi | `tvm.tirx.*` |

Behaviour changes with no error:
- **Rebuilding a PrimFunc body drops parameter aliases.** After `FlattenBuffer`, parameters are aliased
  through `DeclBuffer(x_1, data=x.data)`. A pass that builds a new body must carry these over (see
  `_collect_decl_buffers` in `ti_mmalib_inject_dma.py`), or MakePackedAPI fails with variables that are
  "used, but are not passed in as API arguments".
- **`_call_extern_checked`** (fork helper, `ti_mmalib_legalize.py`) returns `SeqStmt([Bind(status, Cast(call)), IfThenElse(...)])`; code that
  read `LetStmt.body` must look at `seq[1]`.

## Target strings

- **CLI strings are rejected** for every kind ("CLI target string form is no longer supported"). The
  native form is a dict: `{"kind": "c_static_lib", "mcpu": "c7x", "use-cpp-api": True}`; booleans are
  JSON booleans.
- The fork converts legacy `c_static_lib -opt=val ...` strings to that dict
  (`_parse_c_static_lib_cli` in `python/tvm/target/target.py`), so c_static_lib strings in tests and
  tools still work. Other kinds (`llvm -mcpu=...`) need the dict form.
- **In C++, `Target::str()` is JSON.** A substring search such as `target_str.find("mcpu=c7")` silently
  never matches. Read typed attributes: `target->GetAttr<ffi::String>("mcpu")`, `GetAttr<bool>`,
  `GetAttr<int64_t>` (not `Integer`/`Bool`).
- **The schema fills defaults before the canonicalizer runs**, so "did the user set this option" checks
  in the canonicalizer never fire. Options with a fork-specific default have no schema default and are
  defaulted at the end of the canonicalizer (`target_kind.cc`).
- **A boolean with a `true` schema default is not stored in the attrs unless set**, so
  `GetAttr<bool>(x).value_or(false)` reads false. The c_static_lib canonicalizer materializes
  `use-cpp-api`, `skip-runtime-checks` and `tidl-kernels`.

## C++: CodeGenC subclasses and TIR visitors

| 0.23 | 0.27 |
|------|------|
| `VisitExpr_(const XNode*, std::ostream&)`, `VisitStmt_(const XNode*)` | `Dispatch_(...)` with the same shapes; recurse with `Dispatch(expr, os)` |
| `AddNode`, `DivNode`, `MinNode`, `MaxNode`, `EQNode`, `CastNode`, `SelectNode`, `BroadcastNode`, `RampNode`, ... | `prim::AddNode` etc. (`tvm/ir/prim/expr.h`) |
| `BufferLoadNode` (`->buffer`, `->dtype`) | `TensorLoadNode` in `tvm::` (not `tvm::tirx::`): `->source.as_or_throw<BufferVar>()`, `->ty.as_or_throw<PrimType>()` |
| `LetStmtNode` (`->body`) | `BindNode`: flat, no body. An override that used to print the body emits its declaration and returns |
| `AllocateNode` | `AllocBufferNode` (flat; `->buffer` is a `BufferVar`) |
| `DataType` (`is_handle`, `is_float`, `lanes`) | `PrimType` (`MatchesCode(DLDataTypeCode::kDLFloat)`, `IsVoid()`, `lanes()`, `WithLanes(1)`); pointer-ness is `ty.as<PointerTypeNode>()` |
| `void PrintType(DataType, std::ostream&) final` | `using CodeGenC::PrintType; void PrintType(const PrimType&, std::ostream&) final;` |
| `op->dtype`, `expr.dtype()` | `op->ty` (a `Type`), `expr.ty()`; `op->ty.as_or_throw<PrimType>()` |
| `var->name_hint` | `var->name` (`GlobalVar` keeps `name_hint`) |
| `ICHECK`, `LOG(FATAL)` | `TVM_FFI_ICHECK`, `TVM_FFI_THROW(InternalError)` |
| `IntImmNode::value` (int64) | `ffi::BigInt`: `static_cast<int64_t>(x->value)` or `x->value.as<int64_t>()`; it does not convert implicitly and cannot be `switch`ed on |
| `Downcast<PrimFunc>(x)` | `x.as_or_throw<PrimFunc>()` |
| `builtin::` | `tirx::builtin::` (`tvm/tirx/builtin.h`); `if_then_else` is `prim::builtin::if_then_else()` |
| `builtin::ret()` | gone: `Return` is a statement (`ReturnNode`) |
| `is_zero`, `is_const_int` | `prim::is_zero`, `prim::is_const_int`; they take `PrimExpr`, and `CallNode::args` are `Expr`, so `.as<PrimExpr>()` first |
| `Array<PrimExpr>` call args | `ffi::Array<Expr>` for `CallNode::args` and `PrintCallExtern` |
| `tvm::StructuralEqual` | `ffi::StructuralEqual` |
| `AssertStmtNode::message` | `error_kind` + `message_parts` (array of `StringImm`) |
| `StmtExprVisitor::VisitExpr_/VisitStmt_` | `ffi::Optional<VisitInterrupt> Visit_(const XNode*)`; add `using StmtExprVisitor::Visit_;`; call `visitor.Visit(stmt)`, not `visitor(stmt)` |
| `StmtExprMutator::VisitStmt_` | `Mutate_(node, InplaceMode) -> UnchangedOr<T>` |
| `std::unordered_map<K, DataType>` + `operator[]` | `PrimType` has no default constructor: use `insert_or_assign` / `emplace` |
| `arith::` (`Analyzer`, `PVar`, `ramp`) | `sym::` (`tvm/sym/analyzer.h`, `sym::PVar`, `sym::ramp`) |
| C++17 | C++20 |

Behaviour that a codegen pattern-matches and that changed:
- **MakePackedAPI's ObjectRef unwrap** is
  `var = if_then_else(type_index == kTVMFFITensor, handle_add_byte_offset(v, 24), v)` (a `Call`, not
  `Select(EQ(...))`), often wrapped in `reinterpret(...)`; `v` is a `tvm_struct_get`, possibly under
  casts. `<param>.type_index` variables are still `Bind(Var(name + ".type_index"), tvm_struct_get(...))`.
  Matchers must look through the `reinterpret`.
- **`restrict`** on data pointers now comes from a `DeclBuffer` override (0.23 did it in the `LetStmt`
  for `buffer->data`).
- **`CodeGenC` owns handle-type registration.** `RegisterHandleType` aborts on a conflicting type; do not
  pre-register buffer types in a subclass. Call `RegisterHandleTypeFromPointer(var, &value)` before
  reading the map.
- **`WithAttr(func, key, IntImm::Int32(1))`** is rejected by `HasNonzeroAttr` ("Cannot convert IntImm to
  int"); store a plain `true` / `int64_t`.
- **Packed lookup in the host C backend** is `TVMFFIEnvModLookupFromImports(module_ctx, name, &out)`;
  `TVMBackendGetFuncFromEnv` and `TVMBackendRunOnce` are gone. c_static_lib emits
  `TVMBackendGetFuncFromGlobalRegistry`, which the DSP runtime implements (firmware service v4.0.0 removed
  the old symbols from the DLOAD export table).

### Porting a CodeGenC subclass: where to look

`src/target/c_static_lib/codegen_c_static_lib.{h,cc}` (`CodeGenCStaticLib : public CodeGenC`) is the
ported example. When porting another override, read the 0.27 implementation that does the same thing
rather than the old file:

| Override | 0.27 reference |
|----------|----------------|
| `Dispatch_` signatures, `PrintType(const PrimType&)`, packed calls, `PrintGetFuncFromBackend` | `src/target/source/codegen_c_host.{h,cc}` |
| Hook signatures (`GetBufferRef`, `PrintVecStore`, `PrintCallExtern`, ...) take `PrimType` / `ffi::Array<Expr>` | `src/target/source/codegen_c.h` |
| `Dispatch_(const BindNode*)` special cases (emit a typed declaration and return) | `src/backend/cuda/codegen/codegen_cuda_host.cc`, then `CodeGenC::Dispatch_(op)` |
| `PrintType` with vector cases, `Dispatch_(FloatImmNode)`, `Dispatch_(AllocBufferNode)`, `address_of` | `src/backend/opencl/codegen/codegen_opencl.cc` |
| Base `TensorLoad`, `BufferStore`, `AllocBuffer`, `DeclBuffer`, `Bind`, `SeqStmt`, `For` | `src/target/source/codegen_c.cc` |
| `print_ssa_form_`, `RegisterHandleTypeFromPointer`, storage scopes | `src/backend/metal/codegen/codegen_metal.cc`, `src/backend/webgpu/codegen/codegen_webgpu.cc` |
| Which statements MakePackedAPI / LowerTVMBuiltin emit | `src/tirx/transform/make_packed_api.cc` (argument unwrapping is in `tvm_ffi_binder.cc`, which it includes), `src/tirx/transform/lower_tvm_builtin.cc` |

Check the IR your matcher expects against those passes, not against the 0.23 output: argument layouts
of `tvm_struct_set`, `anylist_setitem_call_packed` and `tvm_call_packed_lowered` changed (`Expr` args,
`StringImmNode`). A port that compiles can still generate different C; compare the emitted `lib0.c`
for one model against a known-good build (`DSP_KEEP_TEMP=1` keeps it).

## Fork-specific: C7x and TIDL on 0.27

- **Library dispatch passes are off for C7x.** `DispatchSortScan` would rewrite `relax.topk` into the
  `tvm.contrib.sort.topk` packed call, which a standalone c_static_lib executable cannot resolve. The
  symptom (a cl7x error about `TVMBackendGetFuncFromGlobalRegistry` or a `tvm_stack_alloca` declaration
  inside an expression) looks like a codegen bug and is not. See `relax-c7x:cstatic`.
- **The TIDL importer `.so` needs symbols from `libtvm_compiler.so`, which 0.27 loads `RTLD_LOCAL`.**
  Load it only through `tvm.relax.backend.tidl._gate.load_relax_so`, which re-opens the compiler library
  `RTLD_GLOBAL` first. Opened any other way it loads, then the process dies at the first call with
  `symbol lookup error ... Type::IsMissing` (pytest stops mid-file, exit 127, no summary).
- **Generated code and tests:** `TVMBackendGetFuncFromEnv`-era DLOAD modules do not load on v4.x
  firmware; rebuild modules, firmware and the ARM client from the same tree. See `relax-c7x:firmware`.

## Things that look like 0.27 bugs but are not

- An LLVM message about `-mcpu=neoverse-v3ae` on import: the host CPU name is unknown to LLVM 18. Harmless.
- `ninja` after a codegen edit relinks only `libtvm_compiler.so`; `libtvm_runtime.so` is unchanged.
- A test file that selects 0 tests under `-m quick` exits 5, not 0.
- `uv run` (or an unpinned `uv pip install`) can replace the pinned `apache-tvm-ffi` with a PyPI wheel
  of a different ABI; failures then look like FFI bugs. Reinstall: `cd 3rdparty/tvm-ffi && uv pip install .`.

## Related Skills

- `relax-c7x:relax-passes` — writing passes (examples are in the 0.27 API)
- `relax-c7x:cstatic` — c_static_lib backend, target options, pass order
- `relax-c7x:build` — environment, pinned tvm-ffi, version checks
- `relax-c7x:tidl-offload` — TIDL-specific 0.27 behaviour (gate, importer, subgraph I/O)
