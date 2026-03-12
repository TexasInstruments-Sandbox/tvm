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

import ctypes
import logging
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.relax import transform
from tvm.relax.expr_functor import PyExprMutator, mutator

from .patterns import get_tidl_patterns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TIDL constants and ctypes structures
# ---------------------------------------------------------------------------

TIDL_DIM_MAX = 6

# eTIDL_ElementType from itidl_io.h
_TIDL_ELEMENT_TYPE = {
    "uint8": 0,    # TIDL_UnsignedChar
    "int8": 1,     # TIDL_SignedChar
    "uint16": 2,   # TIDL_UnsignedShort
    "int16": 3,    # TIDL_SignedShort
    "uint32": 4,   # TIDL_UnsignedWord
    "int32": 5,    # TIDL_SignedWord
    "float32": 6,  # TIDL_SinglePrecFloat
    "uint64": 7,   # TIDL_UnsignedDoubleWord
    "int64": 8,    # TIDL_SignedDoubleWord
}


class TensorDescriptor(ctypes.Structure):
    """Matches TensorDescriptor_t from tidl_import_common.h:240."""

    _fields_ = [
        ("scale", ctypes.c_double),
        ("zp", ctypes.c_int32),
        ("element_type", ctypes.c_int32),
        ("dimValues", ctypes.c_int32 * TIDL_DIM_MAX),
        ("name", ctypes.c_char_p),
    ]


class InOutNodes(ctypes.Structure):
    """Matches InOutNodes_t from tidl_relaxImport.cpp:104."""

    _fields_ = [
        ("this_node", ctypes.c_char_p),
        ("num_in_nodes", ctypes.c_int),
        ("num_out_nodes", ctypes.c_int),
        ("in_nodes", ctypes.c_void_p),
        ("out_nodes", ctypes.c_void_p),
    ]


# ---------------------------------------------------------------------------
# TIDL import helpers
# ---------------------------------------------------------------------------


def _load_tidl_relax_so(path=None):
    """Load tidl_model_import_relax.so and verify FFI functions.

    Resolution order: *path* arg > ``TIDL_RELAX_SO_PATH`` env >
    ``$C7X_MMA_TIDL_PATH/.../tidl_model_import_relax.so``.
    """
    if path is None:
        path = os.environ.get("TIDL_RELAX_SO_PATH")
    if path is None:
        c7x = os.environ.get(
            "C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl")
        )
        path = os.path.join(
            c7x, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"tidl_model_import_relax.so not found: {path}"
        )
    tvm.runtime.load_module(path)
    required = [
        "TIDL_relaxInit",
        "TIDL_relaxImportInit",
        "TIDL_relaxImportNode",
        "TIDL_relaxImportLinkNode",
        "TIDL_relaxOptimizeNet",
        "TIDL_relaxPostProcessNet",
    ]
    for name in required:
        if tvm.get_global_func(name, allow_missing=True) is None:
            raise RuntimeError(
                f"FFI function '{name}' not registered after loading .so"
            )


def _tidl_element_type(dtype_str: str) -> int:
    """Map TVM dtype string to TIDL element type enum."""
    return _TIDL_ELEMENT_TYPE.get(str(dtype_str), 6)  # default float32


def _normalize_shape_6d(shape) -> List[int]:
    """Normalize shape to TIDL 6D: [N, dim1, dim2, C, H, W]."""
    s = [int(d) for d in shape]
    n = len(s)
    if n >= 6:
        return s[:6]
    if n == 4:  # NCHW
        return [s[0], 1, 1, s[1], s[2], s[3]]
    if n == 2:  # NC
        return [s[0], 1, 1, s[1], 1, 1]
    if n == 1:
        return [s[0], 1, 1, 1, 1, 1]
    if n == 3:  # NCH
        return [s[0], 1, 1, s[1], s[2], 1]
    if n == 5:  # NCDHW
        return [s[0], s[1], 1, s[2], s[3], s[4]]
    return s


def _find_tidl_subgraphs(mod: IRModule):
    """Return ``[(gv, func)]`` for ``Codegen="tidl"`` functions."""
    result = []
    for gv, func in mod.functions.items():
        if isinstance(func, relax.Function) and func.attrs:
            if func.attrs.get("Codegen") == "tidl":
                result.append((gv, func))
    return result


def _extract_composite_calls(func):
    """Walk SeqExpr bindings and return composite call triples.

    Handles both first-use calls (immediately after the Composite function
    definition) and reuse calls (later calls to a previously defined
    Composite function).  The Relax partitioner reuses function definitions
    when two operations match the same pattern with the same signature.

    Returns
    -------
    list of (comp_fn, orig_call, binding_var)
        *comp_fn* is the ``Composite``-annotated Function,
        *orig_call* is the Call (whose op is a local Var),
        *binding_var* is the Var the call result is bound to.
    """
    # Map: local function Var -> Composite Function (for reuse detection)
    comp_func_map: Dict[relax.Var, relax.Function] = {}

    results = []
    for block in func.body.blocks:
        pending_fn = None
        for b in block.bindings:
            val = b.value
            if (
                isinstance(val, relax.Function)
                and val.attrs
                and val.attrs.get("Composite")
            ):
                pending_fn = val
                comp_func_map[b.var] = val
            elif isinstance(val, relax.Call):
                if pending_fn is not None:
                    # First call immediately after a composite function def
                    results.append((pending_fn, val, b.var))
                    pending_fn = None
                elif (
                    isinstance(val.op, relax.Var)
                    and val.op in comp_func_map
                ):
                    # Reuse of a previously defined composite function
                    results.append((comp_func_map[val.op], val, b.var))
    return results


def _make_in_out_nodes(this_node, in_names, out_names):
    """Build an :class:`InOutNodes` ctypes struct.

    Returns ``(struct, *refs)`` — the caller must keep all returned
    objects alive until the struct has been consumed by the FFI call.
    """
    this_b = (
        this_node.encode("utf-8")
        if isinstance(this_node, str)
        else this_node
    )

    n_in = len(in_names)
    n_out = len(out_names)

    in_bytes = [
        n.encode("utf-8") if isinstance(n, str) else n for n in in_names
    ]
    out_bytes = [
        n.encode("utf-8") if isinstance(n, str) else n for n in out_names
    ]

    InArray = ctypes.c_char_p * max(n_in, 1)
    OutArray = ctypes.c_char_p * max(n_out, 1)
    in_arr = InArray(*in_bytes) if n_in else InArray()
    out_arr = OutArray(*out_bytes) if n_out else OutArray()

    node = InOutNodes()
    node.this_node = this_b
    node.num_in_nodes = n_in
    node.num_out_nodes = n_out
    node.in_nodes = ctypes.cast(in_arr, ctypes.c_void_p) if n_in else None
    node.out_nodes = ctypes.cast(out_arr, ctypes.c_void_p) if n_out else None

    return node, this_b, in_bytes, out_bytes, in_arr, out_arr


def _lift_constants_in_composite(func):
    """Lift Constants inlined in Call args into separate VarBindings.

    ``TIDL_relaxFindConstants`` only finds Constants that are bound to
    their own VarBinding.  After ``FuseOpsByPattern(bind_constants=True)``
    Constants may appear directly as Call arguments.  This helper rewrites
    the composite function so every such Constant gets its own binding.
    """
    new_bindings = []
    for block in func.body.blocks:
        for b in block.bindings:
            val = b.value
            if isinstance(val, relax.Call):
                new_args = list(val.args)
                for k, arg in enumerate(val.args):
                    if isinstance(arg, relax.Constant):
                        const_var = relax.DataflowVar(
                            f"tidl_const_{len(new_bindings)}",
                            arg.struct_info,
                        )
                        new_bindings.append(
                            relax.VarBinding(const_var, arg)
                        )
                        new_args[k] = const_var

                new_call = relax.Call(
                    val.op, new_args, val.attrs, val.sinfo_args,
                )
                relax._ffi_api.UpdateStructInfo(
                    new_call, val.struct_info
                )
                new_bindings.append(relax.VarBinding(b.var, new_call))
            else:
                new_bindings.append(b)

    new_block = relax.DataflowBlock(new_bindings)
    new_body = relax.SeqExpr([new_block], func.body.body)
    new_func = relax.Function(
        func.params,
        new_body,
        func.ret_struct_info,
        func.is_pure,
        func.attrs,
    )
    return new_func


def _write_calibration_data(
    artifacts_dir, sg_id, input_sinfos, num_frames, user_data=None,
):
    """Write calibration binary for one subgraph.

    If *user_data* is ``None``, random float32 data is generated.
    The file is written to ``{artifacts_dir}/calib_raw_data{sg_id}.bin``.
    """
    calib_path = os.path.join(artifacts_dir, f"calib_raw_data{sg_id}.bin")
    if user_data is not None:
        if isinstance(user_data, np.ndarray):
            user_data.astype("float32").tofile(calib_path)
        else:
            raise TypeError(
                f"calibration_data must be ndarray, got {type(user_data)}"
            )
        return calib_path

    # Generate random float32 data (all inputs concatenated per frame)
    parts = []
    for si in input_sinfos:
        shape = [int(d) for d in si.shape]
        per_frame_shape = shape[1:] if len(shape) > 1 else shape
        data = np.random.rand(num_frames, *per_frame_shape).astype("float32")
        parts.append(data.reshape(num_frames, -1))
    all_data = np.concatenate(parts, axis=1).flatten()
    all_data.tofile(calib_path)
    return calib_path


@dataclass
class TIDLBuildResult:
    """Result of TIDLOffloadCompiler.build().

    Attributes
    ----------
    module_path : Path
        Path to lib0.out (C7x DLOAD relocatable module).
    weights_path : Path
        Path to weights.bin (TVM model constants).
    gen_dir : Path
        Path to generated code directory (lib0.c, weights.bin, etc.).
    artifacts : dict
        TIDL artifacts per subgraph:
        ``{sg_name: {"net_bin": path, "io_bin": path}}``.
    build_dir : Path
        Path to cmake build directory.
    """

    module_path: Path
    weights_path: Path
    gen_dir: Path
    artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    build_dir: Path = field(default_factory=lambda: Path())


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
                relax.transform.FoldBatchnormToConv2D(),
                relax.transform.FoldConstant(),
                relax.transform.Normalize(),
            ]
        )
        return seq(mod)

    # ------------------------------------------------------------------
    # Phase 3: TIDL Import (requires .so)
    # ------------------------------------------------------------------

    def tidl_import(self, mod: IRModule) -> Tuple[IRModule, Dict]:
        """Import TIDL subgraphs, producing net.bin / io.bin artifacts.

        Calls the TIDL import library FFI functions to convert each
        ``Codegen="tidl"`` subgraph into optimized TIDL network binaries.

        Parameters
        ----------
        mod : IRModule
            Partitioned module (after ``partition()``).

        Returns
        -------
        mod : IRModule
            Module (unchanged — artifacts are on disk).
        artifacts : dict
            ``{subgraph_name: {"net_bin": path, "io_bin": path}}``.
        """
        # ---- Load .so -------------------------------------------------
        _load_tidl_relax_so(self.config.get("tidl_relax_so_path"))

        # ---- Resolve paths --------------------------------------------
        artifacts_dir = self._artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        c7x_root = os.environ.get(
            "C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl")
        )
        tidl_tools_path = self.config.get(
            "tidl_tools_path", os.path.join(c7x_root, "tidl_tools")
        )

        # ---- TIDL_relaxInit -------------------------------------------
        init_options = dict(self.config.get("tidl_options", {}))
        init_options.setdefault("tidl_tools_path", tidl_tools_path)
        init_options.setdefault("artifacts_folder", artifacts_dir)
        # FFI expects Map<String, String>
        init_options = {str(k): str(v) for k, v in init_options.items()}

        init_fn = tvm.get_global_func("TIDL_relaxInit")
        ret = init_fn(1, init_options)
        if ret != 0:
            raise RuntimeError(f"TIDL_relaxInit failed (rc={ret})")

        # ---- Per-subgraph import --------------------------------------
        subgraphs = _find_tidl_subgraphs(mod)
        if not subgraphs:
            logger.info("No Codegen='tidl' subgraphs found; nothing to import")
            return mod, {}

        num_calib_frames = self.config.get("num_calibration_frames", 1)
        artifacts: Dict[str, Dict[str, str]] = {}

        for sg_id, (gv, func) in enumerate(subgraphs):
            sg_name = gv.name_hint
            logger.info("Importing TIDL subgraph %d: %s", sg_id, sg_name)

            # -- Tensor descriptors for inputs + outputs ----------------
            input_sinfos = [p.struct_info for p in func.params]
            output_sinfo = func.ret_struct_info
            if isinstance(output_sinfo, relax.TupleStructInfo):
                output_sinfos = list(output_sinfo.fields)
            else:
                output_sinfos = [output_sinfo]

            n_inputs = len(input_sinfos)
            n_outputs = len(output_sinfos)
            n_total = n_inputs + n_outputs

            DescArray = TensorDescriptor * n_total
            descriptors = DescArray()
            _name_refs = []  # prevent GC of byte strings

            for i, si in enumerate(input_sinfos):
                name_b = f"tidl_{sg_id}_i{i}".encode()
                _name_refs.append(name_b)
                shape_6d = _normalize_shape_6d(si.shape)
                descriptors[i].scale = 1.0
                descriptors[i].zp = 0
                descriptors[i].element_type = _tidl_element_type(si.dtype)
                for j in range(TIDL_DIM_MAX):
                    descriptors[i].dimValues[j] = shape_6d[j]
                descriptors[i].name = name_b

            for i, si in enumerate(output_sinfos):
                idx = n_inputs + i
                name_b = f"tidl_{sg_id}_o{i}".encode()
                _name_refs.append(name_b)
                shape_6d = _normalize_shape_6d(si.shape)
                descriptors[idx].scale = 1.0
                descriptors[idx].zp = 0
                descriptors[idx].element_type = _tidl_element_type(si.dtype)
                for j in range(TIDL_DIM_MAX):
                    descriptors[idx].dimValues[j] = shape_6d[j]
                descriptors[idx].name = name_b

            # -- TIDL_relaxImportInit -----------------------------------
            import_init_fn = tvm.get_global_func("TIDL_relaxImportInit")
            ret = import_init_fn(
                sg_id,
                n_inputs,
                n_outputs,
                ctypes.c_void_p(ctypes.addressof(descriptors)),
                1,               # is_nchw
                tidl_tools_path,
                artifacts_dir,
                False,           # isSubgraphOD
            )
            if ret != 0:
                raise RuntimeError(
                    f"TIDL_relaxImportInit failed for '{sg_name}' (rc={ret})"
                )

            # -- Walk composite calls -----------------------------------
            composites = _extract_composite_calls(func)

            # Build var map: Var object -> TIDL node name.
            # Keyed by Var identity (not name_hint) because Relax
            # reuses name hints like "lv" across different bindings
            # within the same function body.
            var_map: Dict[relax.Var, str] = {}
            for i, p in enumerate(func.params):
                var_map[p] = f"tidl_{sg_id}_i{i}"
            for i, (_, _, bvar) in enumerate(composites):
                var_map[bvar] = str(i)

            import_node_fn = tvm.get_global_func("TIDL_relaxImportNode")
            link_node_fn = tvm.get_global_func("TIDL_relaxImportLinkNode")

            for i, (comp_fn, orig_call, bvar) in enumerate(composites):
                comp_name = str(comp_fn.attrs["Composite"])

                # Lift inlined Constants into VarBindings so the
                # C++ parser (TIDL_relaxFindConstants) can find them.
                lifted_fn = _lift_constants_in_composite(comp_fn)

                # Synthetic call: Function as op (C++ expects FunctionNode)
                syn_call = relax.Call(lifted_fn, orig_call.args)
                relax._ffi_api.UpdateStructInfo(
                    syn_call, orig_call.struct_info
                )

                # No per-tensor quantization for float32 models
                zp = np.zeros(1, dtype=np.int32)
                scale = np.ones(1, dtype=np.float32)
                ret = import_node_fn(
                    syn_call,
                    len(zp),
                    ctypes.c_void_p(zp.ctypes.data),
                    len(scale),
                    ctypes.c_void_p(scale.ctypes.data),
                )
                if ret != 0:
                    raise RuntimeError(
                        f"TIDL_relaxImportNode failed for "
                        f"'{comp_name}' (rc={ret})"
                    )

                # -- InOutNodes for linking -----------------------------
                # Input names (use Var identity via var_map)
                in_names = []
                for arg in orig_call.args:
                    if isinstance(arg, relax.Var):
                        in_names.append(
                            var_map.get(arg, arg.name_hint)
                        )

                # Output consumer names (use Var.same_as for identity)
                out_names = []
                for j in range(i + 1, len(composites)):
                    _, future_call, _ = composites[j]
                    for farg in future_call.args:
                        if (
                            isinstance(farg, relax.Var)
                            and farg.same_as(bvar)
                        ):
                            out_names.append(str(j))

                # Terminal node: use subgraph output tensor name
                if not out_names:
                    node_name = f"tidl_{sg_id}_o0"
                else:
                    node_name = str(i)

                refs = _make_in_out_nodes(node_name, in_names, out_names)
                in_out_struct = refs[0]
                ret = link_node_fn(
                    ctypes.c_void_p(ctypes.addressof(in_out_struct))
                )
                if ret != 0:
                    raise RuntimeError(
                        f"TIDL_relaxImportLinkNode failed for "
                        f"node {i} (rc={ret})"
                    )

            # -- TIDL_relaxOptimizeNet ----------------------------------
            optimize_fn = tvm.get_global_func("TIDL_relaxOptimizeNet")
            ret = optimize_fn(sg_id)
            if ret != 0:
                raise RuntimeError(
                    f"TIDL_relaxOptimizeNet failed for "
                    f"'{sg_name}' (rc={ret})"
                )

            # -- Calibration + PostProcess ------------------------------
            _write_calibration_data(
                artifacts_dir,
                sg_id,
                input_sinfos,
                num_calib_frames,
                user_data=self.config.get("calibration_data"),
            )

            postprocess_fn = tvm.get_global_func(
                "TIDL_relaxPostProcessNet"
            )
            ret = postprocess_fn(num_calib_frames)
            if ret != 0:
                raise RuntimeError(
                    f"TIDL_relaxPostProcessNet failed for "
                    f"'{sg_name}' (rc={ret})"
                )

            # Record artifact paths
            artifacts[sg_name] = {
                "net_bin": os.path.join(
                    artifacts_dir, f"subgraph{sg_id}_net.bin"
                ),
                "io_bin": os.path.join(
                    artifacts_dir, f"subgraph{sg_id}_params_1.bin"
                ),
            }
            logger.info(
                "Subgraph %d (%s) imported successfully", sg_id, sg_name
            )

        return mod, artifacts

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

    def compile(self, mod: IRModule, params: Optional[Dict] = None) -> Tuple[IRModule, Dict]:
        """Full TIDL offload pipeline: prepare -> partition -> import -> lower.

        Parameters
        ----------
        mod : IRModule
            Input module.
        params : dict, optional
            Named parameters to bind.

        Returns
        -------
        mod : IRModule
            Module ready for c_static codegen with TIDL subgraphs lowered
            to extern calls.
        artifacts : dict
            TIDL artifacts ``{sg_name: {"net_bin": path, "io_bin": path}}``.
        """
        mod = self.prepare(mod, params)
        mod = self.partition(mod)
        mod, artifacts = self.tidl_import(mod)
        mod = self.lower_tidl(mod, artifacts)
        return mod, artifacts

    # ------------------------------------------------------------------
    # Full build pipeline: compile -> codegen -> bridge -> dynmod
    # ------------------------------------------------------------------

    def build(
        self,
        mod: IRModule,
        params: Optional[Dict] = None,
        target: str = "c_static -mcpu=c7x",
        build_dir: Optional[str] = None,
    ) -> TIDLBuildResult:
        """Full pipeline: compile + codegen + bridge + dynmod build.

        Produces a ready-to-load lib0.out from a Relax module in a
        single call.

        Parameters
        ----------
        mod : IRModule
            Input module (raw, before preparation).
        params : dict, optional
            Named parameters to bind as constants.
        target : str
            TVM target string for c_static codegen.
        build_dir : str, optional
            Directory for cmake build output. If None, a temp dir is used.

        Returns
        -------
        TIDLBuildResult
            Paths to lib0.out, weights.bin, generated code, and artifacts.
        """
        # 1. Full TIDL offload pipeline
        lowered, artifacts = self.compile(mod, params)

        # 2. Compile to C via relax.build
        # Honor profile_layers config: append -profile-layers to target
        if self.config.get("profile_layers", False):
            if "-profile-layers" not in target:
                target += " -profile-layers"
        tvm_target = tvm.target.Target(target)
        with tvm.transform.PassContext(opt_level=0):
            ex = relax.build(
                lowered,
                target=tvm_target,
                exec_mode="compiled",
                system_lib=True,
            )

        # 3. Export and extract generated code
        gen_dir = Path(tempfile.mkdtemp(prefix="tidl_build_gen_"))
        tar_path = gen_dir / "model.tar"
        ex.export_library(str(tar_path), target=tvm_target)
        with tarfile.open(str(tar_path)) as tf:
            tf.extractall(str(gen_dir))
        tar_path.unlink()

        # 4. Generate real TIDL bridge
        bridge_path = gen_dir / "tidl_bridge.c"
        self.generate_bridge(
            lowered,
            str(bridge_path),
            stub=False,
            artifacts_dir=self._artifacts_dir,
        )

        # 5. Build C7x DLOAD module
        if build_dir is None:
            build_path = Path(tempfile.mkdtemp(prefix="tidl_build_cmake_"))
        else:
            build_path = Path(build_dir)
            build_path.mkdir(parents=True, exist_ok=True)

        weights_path = gen_dir / "weights.bin"
        module_path = _build_dynmod(
            generated_dir=gen_dir,
            build_dir=build_path,
            weights_file=weights_path if weights_path.exists() else None,
            tidl_bridge=str(bridge_path),
            use_tidl=bool(artifacts),
            tidl_artifacts_dir=self._artifacts_dir if artifacts else None,
        )

        return TIDLBuildResult(
            module_path=module_path,
            weights_path=weights_path,
            gen_dir=gen_dir,
            artifacts=artifacts,
            build_dir=build_path,
        )


# ------------------------------------------------------------------
# Internal: C7x DLOAD module build
# ------------------------------------------------------------------


def _build_dynmod(
    generated_dir: Path,
    build_dir: Path,
    weights_file: Optional[Path] = None,
    tidl_bridge: Optional[str] = None,
    use_tidl: bool = False,
    tidl_artifacts_dir: Optional[str] = None,
    build_type: str = "Release",
) -> Path:
    """Build a C7x DLOAD relocatable module from generated code.

    Uses the CMakeLists.txt at ``src/runtime/ti_dsp/dynmod/`` with the
    TI C7x cross-compiler toolchain.

    Parameters
    ----------
    generated_dir : Path
        Directory containing lib0.c.
    build_dir : Path
        CMake build output directory.
    weights_file : Path, optional
        Path to weights.bin to embed.
    tidl_bridge : str, optional
        Path to tidl_bridge.c source file.
    use_tidl : bool
        Whether to link TIDL API and artifacts.
    tidl_artifacts_dir : str, optional
        Directory containing TIDL net.bin / io.bin files.
    build_type : str
        CMake build type (Release or Debug).

    Returns
    -------
    Path
        Path to the built lib0.out module.
    """
    tvm_home = Path(__file__).resolve().parents[5]  # python/tvm/relax/backend/tidl -> tvm
    dsp_runtime_dir = tvm_home / "src" / "runtime" / "ti_dsp"
    dynmod_cmake = dsp_runtime_dir / "dynmod"
    toolchain_file = dsp_runtime_dir / "cmake" / "toolchain-j722s-c7x.cmake"

    if not toolchain_file.exists():
        raise FileNotFoundError(f"Toolchain file not found: {toolchain_file}")
    if not dynmod_cmake.exists():
        raise FileNotFoundError(f"Dynmod CMakeLists.txt not found: {dynmod_cmake}")

    generated_dir = Path(generated_dir).resolve()
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure cmake
    cmake_cmd = [
        "cmake",
        f"-DTVM_HOME={tvm_home}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DGENERATED_CODE_DIR={generated_dir}",
    ]
    if weights_file is not None and Path(weights_file).exists():
        cmake_cmd.append(f"-DWEIGHTS_FILE={Path(weights_file).resolve()}")
    if tidl_bridge:
        cmake_cmd.append(f"-DTIDL_BRIDGE_SOURCES={tidl_bridge}")
    if use_tidl:
        cmake_cmd.append("-DUSE_TIDL=ON")
    if tidl_artifacts_dir:
        cmake_cmd.append(f"-DTIDL_ARTIFACTS_DIR={tidl_artifacts_dir}")
    cmake_cmd.append(str(dynmod_cmake))

    log_path = build_dir / "cmake.log"
    logger.info("Building C7x DLOAD module: %s", generated_dir)

    with open(log_path, "w") as f:
        result = subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CMake configuration failed (rc={result.returncode}). "
                f"Check {log_path}"
            )

        result = subprocess.run(
            ["cmake", "--build", "."],
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Build failed (rc={result.returncode}). Check {log_path}"
            )

    module = build_dir / "lib0.out"
    if not module.exists():
        raise FileNotFoundError(f"Module not found after build: {module}")

    logger.info("Built C7x DLOAD module: %s", module)
    return module


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
    """Generate C source code for TIDL bridge functions.

    Supports multiple TIDL subgraphs. Each subgraph gets its own
    embedded artifact symbols (tidl_net_N, tidl_io_N) and its own
    init/process lifecycle.
    """
    lines = []
    lines.append("/* Auto-generated TIDL bridge functions. */")
    lines.append("/* Implements tidl_subgraph_N_process() called by lib0.c */")
    lines.append("")
    lines.append("#include <string.h>")
    lines.append("#include <stdint.h>")
    lines.append("")

    # Filter to subgraphs that have output (i.e., real bridge candidates)
    real_subgraphs = [sg for sg in subgraphs if sg["output"] is not None]

    if not stub and real_subgraphs:
        # Shared includes and externs for real bridge mode (emitted once)
        lines.append('#include "tidl_api.h"')
        lines.append('#include "dlpack/dlpack.h"')
        lines.append("")
        lines.append("extern void* appUdmaGetObj(void);")
        lines.append(
            "extern int32_t TVM_cacheWbInvRegion"
            "(void *addr, uint32_t size);")
        lines.append("")

        # Per-subgraph artifact symbols (from bin_to_asm.py)
        lines.append("/* Embedded TIDL artifacts (from bin_to_asm.py) */")
        for i, sg in enumerate(real_subgraphs):
            lines.append(
                f"extern unsigned char _binary_tidl_net_{i}_start[];")
            lines.append(
                f"extern unsigned int  _binary_tidl_net_{i}_size;")
            lines.append(
                f"extern unsigned char _binary_tidl_io_{i}_start[];")
        lines.append("")

    # Forward declarations with C linkage (lib0.c is compiled as C++)
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    for sg in real_subgraphs:
        lines.append(
            f"void {sg['name']}_process(void* inp0, void* out0);")
    lines.append("void tidl_bridge_cleanup(void);")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")

    for i, sg in enumerate(real_subgraphs):
        name = sg["name"]
        process_fn = f"{name}_process"
        output = sg["output"]

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
            # Real TIDL bridge (per-subgraph)
            lines.append(f"static void* {name}_instance = NULL;")
            lines.append("")

            in_info = sg["inputs"][0] if sg["inputs"] else output
            in_shape = in_info["shape"]

            lines.append(f"void {process_fn}(void* inp0, void* out0) {{")
            lines.append(f"    if ({name}_instance == NULL) {{")
            lines.append(f"        {name}_instance = init_tidl_subgraph(")
            lines.append(
                f"            _binary_tidl_net_{i}_start,"
                f" _binary_tidl_net_{i}_size,")
            lines.append(
                f"            _binary_tidl_io_{i}_start, appUdmaGetObj(),"
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

    # Cleanup function: free all TIDL instances.
    # The firmware calls this before dyn_loader_unload() so TIDL can
    # release DMA channels, IALG memory, and MMA state cleanly.
    if not stub and real_subgraphs:
        lines.append("void tidl_bridge_cleanup(void) {")
        for sg in real_subgraphs:
            name = sg["name"]
            lines.append(f"    if ({name}_instance != NULL) {{")
            lines.append(f"        free_tidl_subgraph({name}_instance);")
            lines.append(f"        {name}_instance = NULL;")
            lines.append("    }")
        lines.append("}")
    else:
        lines.append("void tidl_bridge_cleanup(void) {}")
    lines.append("")

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
