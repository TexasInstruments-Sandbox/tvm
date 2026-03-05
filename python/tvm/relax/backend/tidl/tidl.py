# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TIDL offload compiler for TVM/Relax c_static backend.

Orchestrates the multi-phase pipeline:
  1. Partition — identify TIDL-supported subgraphs via pattern matching
  2. Import   — run TIDL compile-time import (produces net.bin / params.bin)
  3. Lower    — replace TIDL functions with extern calls for c_static codegen

Phases 2-3 require the TIDL import library (.so); phase 1 is self-contained.
"""

from typing import Any, Dict, Optional, Tuple

import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.relax import transform
from tvm.relax.expr_functor import PyExprMutator, mutator

from .patterns import get_tidl_patterns


class TIDLOffloadCompiler:
    """Compile Relax modules with TIDL subgraph offloading.

    Parameters
    ----------
    config : dict, optional
        Configuration options:
        - ``artifacts_dir`` (str): Directory for TIDL net.bin / params.bin
        - ``num_subgraphs`` (int): Max subgraphs to offload (0 = unlimited)
        - ``calibration_data`` (dict): Calibration inputs for quantized models
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._artifacts_dir = self.config.get("artifacts_dir", "/tmp/tidl_artifacts")
        self._tidl_lib = None

    # ------------------------------------------------------------------
    # Phase 1: Partitioning (no .so dependency)
    # ------------------------------------------------------------------

    def partition(self, mod: IRModule) -> IRModule:
        """Partition the Relax IR into TIDL and non-TIDL subgraphs.

        Applies ``FuseOpsByPattern`` with TIDL patterns, then merges
        adjacent composite functions belonging to the same ``tidl``
        backend into single subgraph functions annotated with
        ``Codegen="tidl"``.

        Parameters
        ----------
        mod : IRModule
            Input module (should have constants folded / simplified).

        Returns
        -------
        IRModule
            Partitioned module with TIDL subgraph functions.
        """
        patterns = get_tidl_patterns()
        mod = transform.FuseOpsByPattern(
            patterns, bind_constants=True, annotate_codegen=False
        )(mod)
        mod = transform.MergeCompositeFunctions()(mod)
        return mod

    # ------------------------------------------------------------------
    # Phase 1 helpers: preparation passes
    # ------------------------------------------------------------------

    def prepare(self, mod: IRModule, params: Optional[Dict] = None) -> IRModule:
        """Run standard Relax preparation passes before partitioning.

        Parameters
        ----------
        mod : IRModule
            Raw module from frontend import.
        params : dict, optional
            Named parameters to bind as constants.

        Returns
        -------
        IRModule
            Simplified module ready for partitioning.
        """
        if params:
            mod = relax.transform.BindParams("main", params)(mod)
        seq = tvm.transform.Sequential(
            [
                relax.transform.FoldConstant(),
                relax.transform.Normalize(),
            ]
        )
        return seq(mod)

    # ------------------------------------------------------------------
    # Phase 3: TIDL Import (requires .so — stub for now)
    # ------------------------------------------------------------------

    def tidl_import(self, mod: IRModule) -> Tuple[IRModule, Dict]:
        """Import TIDL subgraphs, producing net.bin / params.bin artifacts.

        Requires the TIDL import library to be available at runtime.
        Currently a stub that will be implemented in Phase 3.

        Returns
        -------
        mod : IRModule
            Module (unchanged for now).
        artifacts : dict
            Mapping from subgraph id to artifact paths.
        """
        raise NotImplementedError(
            "TIDL import requires the TIDL import library (.so). "
            "This will be implemented in Phase 3."
        )

    # ------------------------------------------------------------------
    # Phase 4: Lower TIDL to TIR extern calls
    # ------------------------------------------------------------------

    def lower_tidl(self, mod: IRModule, artifacts: Optional[Dict] = None) -> IRModule:
        """Replace TIDL subgraph functions with call_tir to extern stubs.

        Each ``Codegen="tidl"`` function is replaced with a TIR PrimFunc
        that calls ``tidl_subgraph_N_process(input_ptrs..., output_ptr)``
        via ``call_extern``.  The c_static codegen then emits the
        appropriate TIDL init/process/free lifecycle code.

        Parameters
        ----------
        mod : IRModule
            Partitioned module (after ``partition()``).
        artifacts : dict, optional
            TIDL artifact paths (from ``tidl_import``). Not yet used.

        Returns
        -------
        IRModule
            Module with TIDL functions replaced by TIR extern stubs.
        """
        return _lower_tidl_pass(mod)

    # ------------------------------------------------------------------
    # Phase 5: Bridge function generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_bridge(
        mod: IRModule,
        output_path: str,
        stub: bool = True,
        artifacts_dir: Optional[str] = None,
    ) -> str:
        """Generate a C bridge file for TIDL subgraph extern calls.

        Produces a `tidl_bridge.c` that implements
        ``tidl_subgraph_N_process()`` functions referenced by the
        ``call_extern`` in the TIR stubs created by ``lower_tidl()``.

        Parameters
        ----------
        mod : IRModule
            Lowered module (after ``lower_tidl()``).
        output_path : str
            Path to write the generated bridge C file.
        stub : bool
            If True, generate stub implementations that zero-fill
            the output (for pipeline testing without TIDL libs).
            If False, generate real TIDL API calls.
        artifacts_dir : str, optional
            Path to TIDL artifacts (net.bin, io.bin) when stub=False.

        Returns
        -------
        str
            Path to the generated bridge C file.
        """
        subgraphs = _collect_tidl_subgraph_info(mod)
        code = _generate_bridge_code(subgraphs, stub, artifacts_dir)
        with open(output_path, "w") as f:
            f.write(code)

        # Also generate a header with forward declarations so lib0.c
        # can resolve the extern calls at compile time.
        header_path = output_path.replace(".c", ".h")
        header_lines = [
            "/* Auto-generated TIDL bridge declarations. */",
            "#ifndef TIDL_BRIDGE_H_",
            "#define TIDL_BRIDGE_H_",
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
        ]
        for sg in subgraphs:
            if sg["output"] is not None:
                header_lines.append(
                    f"void {sg['name']}_process(void* inp0, void* out0);")
        header_lines += [
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "#endif  /* TIDL_BRIDGE_H_ */",
            "",
        ]
        with open(header_path, "w") as f:
            f.write("\n".join(header_lines))

        return output_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def compile(self, mod: IRModule, params: Optional[Dict] = None) -> IRModule:
        """Full TIDL offload pipeline: prepare -> partition -> import -> lower.

        Parameters
        ----------
        mod : IRModule
            Input module.
        params : dict, optional
            Named parameters to bind.

        Returns
        -------
        IRModule
            Module ready for c_static codegen with TIDL subgraphs lowered
            to extern calls.
        """
        mod = self.prepare(mod, params)
        mod = self.partition(mod)
        mod, artifacts = self.tidl_import(mod)
        mod = self.lower_tidl(mod, artifacts)
        return mod


# ------------------------------------------------------------------
# Internal: Bridge function generation
# ------------------------------------------------------------------


def _collect_tidl_subgraph_info(mod):
    """Collect info about TIDL subgraph TIR stubs in a lowered module.

    Returns list of dicts with keys: name, input_shapes, input_dtypes,
    output_shape, output_dtype.
    """
    subgraphs = []
    for gv, func in mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc) and "tidl_subgraph" in gv.name_hint:
            name = gv.name_hint
            # Extract shapes/dtypes from buffer_map
            inputs = []
            output = None
            for param in func.params:
                buf = func.buffer_map.get(param)
                if buf is not None:
                    shape = [int(d) for d in buf.shape]
                    dtype = str(buf.dtype)
                    if param.name.startswith("output"):
                        output = {"shape": shape, "dtype": dtype}
                    else:
                        inputs.append({"shape": shape, "dtype": dtype})
            subgraphs.append({
                "name": name,
                "inputs": inputs,
                "output": output,
            })
    return subgraphs


def _dtype_sizeof(dtype):
    """Return sizeof() for a TVM dtype string."""
    size_map = {
        "float32": 4, "float16": 2, "float64": 8,
        "int8": 1, "int16": 2, "int32": 4, "int64": 8,
        "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    }
    return size_map.get(dtype, 4)


def _generate_bridge_code(subgraphs, stub, artifacts_dir):
    """Generate C source code for TIDL bridge functions."""
    lines = []
    lines.append("/* Auto-generated TIDL bridge functions. */")
    lines.append("/* Implements tidl_subgraph_N_process() called by lib0.c */")
    lines.append("")
    lines.append("#include <string.h>")
    lines.append("#include <stdint.h>")
    lines.append("")

    # Forward declarations with C linkage (lib0.c is compiled as C++)
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    for sg in subgraphs:
        if sg["output"] is not None:
            lines.append(
                f"void {sg['name']}_process(void* inp0, void* out0);")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")

    for sg in subgraphs:
        name = sg["name"]
        process_fn = f"{name}_process"
        output = sg["output"]

        if output is None:
            continue

        out_shape = output["shape"]
        out_dtype = output["dtype"]
        out_bytes = 1
        for d in out_shape:
            out_bytes *= d
        out_bytes *= _dtype_sizeof(out_dtype)

        if stub:
            # Stub mode: just zero-fill output
            lines.append(f"/* Stub bridge for {name} */")
            lines.append(
                f"/* Output: {out_shape} {out_dtype} = {out_bytes} bytes */")
            lines.append(f"void {process_fn}(void* inp0, void* out0) {{")
            lines.append(f"    memset(out0, 0, {out_bytes});")
            lines.append("}")
            lines.append("")
        else:
            # Real TIDL bridge
            lines.append('#include "tidl_api.h"')
            lines.append('#include "dlpack/dlpack.h"')
            lines.append("")
            lines.append("extern void* appUdmaGetObj(void);")
            lines.append(
                "extern int32_t TVM_cacheWbInvRegion"
                "(void *addr, uint32_t size);")
            lines.append("")

            # Artifact symbols from embedded .rodata sections
            # (produced by bin_to_asm.py with prefix "tidl_net" / "tidl_io")
            lines.append(
                "/* Embedded TIDL artifacts (from bin_to_asm.py) */")
            lines.append(
                "extern unsigned char _binary_tidl_net_start[];")
            lines.append(
                "extern unsigned int _binary_tidl_net_size;")
            lines.append(
                "extern unsigned char _binary_tidl_io_start[];")
            lines.append(f"static void* {name}_instance = NULL;")
            lines.append("")

            in_info = sg["inputs"][0] if sg["inputs"] else output
            in_shape = in_info["shape"]

            lines.append(f"void {process_fn}(void* inp0, void* out0) {{")
            lines.append(f"    if ({name}_instance == NULL) {{")
            lines.append(f"        {name}_instance = init_tidl_subgraph(")
            lines.append(
                "            _binary_tidl_net_start, _binary_tidl_net_size,")
            lines.append(
                "            _binary_tidl_io_start, appUdmaGetObj(),"
                " 1, 0);")
            lines.append("    }")
            lines.append("")

            # Build DLTensor for input
            lines.append("    DLTensor in_tensor;")
            lines.append("    memset(&in_tensor, 0, sizeof(in_tensor));")
            lines.append("    in_tensor.data = inp0;")
            lines.append(f"    in_tensor.ndim = {len(in_shape)};")
            lines.append(
                f"    int64_t in_shape[] = {{{', '.join(str(d) for d in in_shape)}}};")
            lines.append("    in_tensor.shape = in_shape;")
            lines.append("    in_tensor.dtype.code = kDLFloat;")
            lines.append("    in_tensor.dtype.bits = 32;")
            lines.append("    in_tensor.dtype.lanes = 1;")
            lines.append("")

            # Build DLTensor for output
            lines.append("    DLTensor out_tensor;")
            lines.append("    memset(&out_tensor, 0, sizeof(out_tensor));")
            lines.append("    out_tensor.data = out0;")
            lines.append(f"    out_tensor.ndim = {len(out_shape)};")
            lines.append(
                f"    int64_t out_shape[] = {{{', '.join(str(d) for d in out_shape)}}};")
            lines.append("    out_tensor.shape = out_shape;")
            lines.append("    out_tensor.dtype.code = kDLFloat;")
            lines.append("    out_tensor.dtype.bits = 32;")
            lines.append("    out_tensor.dtype.lanes = 1;")
            lines.append("")

            lines.append("    DLTensor* in[] = { &in_tensor };")
            lines.append("    DLTensor* out[] = { &out_tensor };")
            lines.append("")
            # Cache flush input before TIDL reads via DMA
            in_bytes = 1
            for d in in_shape:
                in_bytes *= d
            in_bytes *= _dtype_sizeof(in_info.get("dtype", "float32"))
            lines.append(
                f"    TVM_cacheWbInvRegion(inp0, {in_bytes});")
            lines.append(
                f"    process_tidl_subgraph({name}_instance, in, out);")
            # Cache invalidate output after TIDL writes via DMA
            lines.append(
                f"    TVM_cacheWbInvRegion(out0, {out_bytes});")
            lines.append("}")
            lines.append("")
            break  # only emit real bridge once (shared includes)

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# Internal: TIDL lowering implementation
# ------------------------------------------------------------------


def _make_tidl_tir_stub(name, input_sinfos, output_sinfo):
    """Create a TIR PrimFunc stub that calls an extern TIDL function.

    The stub extracts raw data pointers from DLTensor buffers and
    passes them to ``{name}_process(inp0, inp1, ..., out0)``.
    """
    params = []
    buffer_map = {}

    for i, si in enumerate(input_sinfos):
        shape = [int(d) for d in si.shape]
        dtype = str(si.dtype)
        handle = tvm.tir.Var(f"input{i}", "handle")
        buf = tvm.tir.decl_buffer(shape, dtype, f"inp{i}")
        params.append(handle)
        buffer_map[handle] = buf

    out_shape = [int(d) for d in output_sinfo.shape]
    out_dtype = str(output_sinfo.dtype)
    out_handle = tvm.tir.Var("output0", "handle")
    out_buf = tvm.tir.decl_buffer(out_shape, out_dtype, "out0")
    params.append(out_handle)
    buffer_map[out_handle] = out_buf

    extern_args = [buffer_map[h].data for h in params[:-1]]
    extern_args.append(out_buf.data)

    # Wrap the extern call in a Block → BlockRealize so that Relax
    # analysis passes (e.g., HasReshapePattern) don't crash when
    # inspecting this function.
    extern_call = tvm.tir.Evaluate(
        tvm.tir.call_extern("int32", f"{name}_process", *extern_args)
    )
    block = tvm.tir.Block([], [], [], name + "_block", extern_call)
    body = tvm.tir.BlockRealize([], tvm.tir.const(True, "bool"), block)

    func = tvm.tir.PrimFunc(params, body, buffer_map=buffer_map)
    func = func.with_attr("tir.noalias", True)
    return func


@mutator
class _TIDLCallReplacer(PyExprMutator):
    """Replace calls to Codegen='tidl' functions with call_tir."""

    def __init__(self, mod, tir_stubs, stub_gvars):
        super().__init__(mod)
        self._tir_stubs = tir_stubs
        self._stub_gvars = stub_gvars

    def visit_call_(self, call):
        call = super().visit_call_(call)
        if isinstance(call.op, relax.GlobalVar):
            name = call.op.name_hint
            if name in self._tir_stubs:
                _, _, _, output_sinfo = self._tir_stubs[name]
                tir_gv = self._stub_gvars[name]
                return relax.call_tir(
                    tir_gv,
                    relax.Tuple(list(call.args)),
                    out_sinfo=output_sinfo,
                )
        return call


def _lower_tidl_pass(mod: IRModule) -> IRModule:
    """Implementation of the LowerTIDLToTIR pass."""
    # Find TIDL functions
    tidl_funcs = {}
    for gv, func in mod.functions.items():
        if isinstance(func, relax.Function) and func.attrs:
            if func.attrs.get("Codegen") == "tidl":
                input_sinfos = [p.struct_info for p in func.params]
                output_sinfo = func.ret_struct_info
                tidl_funcs[gv.name_hint] = (gv, func, input_sinfos, output_sinfo)

    if not tidl_funcs:
        return mod

    # Create TIR stubs
    tir_stubs = {}
    for i, (name, (gv, func, input_sinfos, output_sinfo)) in enumerate(
        tidl_funcs.items()
    ):
        sg_name = f"tidl_subgraph_{i}"
        tir_func = _make_tidl_tir_stub(sg_name, input_sinfos, output_sinfo)
        tir_stubs[name] = (tir_func, sg_name, input_sinfos, output_sinfo)

    # Build new module
    builder = relax.BlockBuilder()

    # Add TIR stubs
    stub_gvars = {}
    for name, (tir_func, sg_name, _, _) in tir_stubs.items():
        stub_gvars[name] = builder.add_func(tir_func, sg_name)

    # Replace TIDL calls in main and copy other functions
    replacer = _TIDLCallReplacer(mod, tir_stubs, stub_gvars)
    for gv, func in mod.functions.items():
        name = gv.name_hint
        if name in tidl_funcs:
            continue  # drop TIDL Codegen functions
        if isinstance(func, relax.Function):
            new_func = replacer.visit_expr(func)
            new_func = relax.analysis.remove_all_unused(new_func)
            builder.add_func(new_func, name)
        elif isinstance(func, tvm.tir.PrimFunc):
            builder.add_func(func, name)

    return builder.finalize()


@tvm.ir.transform.module_pass(opt_level=0, name="LowerTIDLToTIR")
class LowerTIDLToTIR:
    """Relax module pass: replace Codegen='tidl' functions with TIR extern stubs.

    After this pass, each TIDL subgraph becomes a TIR PrimFunc that calls
    ``tidl_subgraph_N_process()`` via ``call_extern``.  The main function
    uses ``call_tir`` to invoke these stubs through the normal register file
    / VM builtin pipeline.
    """

    def transform_module(self, mod: IRModule, _ctx) -> IRModule:
        return _lower_tidl_pass(mod)


def partition_for_tidl(mod: IRModule, config: Optional[Dict[str, Any]] = None) -> IRModule:
    """Convenience function: partition a module for TIDL offloading.

    This only runs Phase 1 (pattern matching + merge) and does not
    require the TIDL import library.

    Parameters
    ----------
    mod : IRModule
        Input module.
    config : dict, optional
        TIDL configuration.

    Returns
    -------
    IRModule
        Partitioned module.
    """
    compiler = TIDLOffloadCompiler(config)
    return compiler.partition(mod)
