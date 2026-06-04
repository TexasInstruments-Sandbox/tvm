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
    "uint8": 0,  # TIDL_UnsignedChar
    "int8": 1,  # TIDL_SignedChar
    "uint16": 2,  # TIDL_UnsignedShort
    "int16": 3,  # TIDL_SignedShort
    "uint32": 4,  # TIDL_UnsignedWord
    "int32": 5,  # TIDL_SignedWord
    "float32": 6,  # TIDL_SinglePrecFloat
    "uint64": 7,  # TIDL_UnsignedDoubleWord
    "int64": 8,  # TIDL_SignedDoubleWord
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


# Relative path from c7x-mma-tidl root to the Relax import .so.
_RELAX_SO_REL = "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"


def _load_tidl_relax_so(path=None, tidl_tools_path=None):
    """Load tidl_model_import_relax.so and verify FFI functions.

    Resolution order:
      1. *path* arg (explicit override)
      2. ``TIDL_RELAX_SO_PATH`` env var
      3. bundled package (``tvm.data.ti_dsp.paths``)
      4. derived from *tidl_tools_path*: the .so and ``tidl_tools/`` are
         siblings under the same c7x-mma-tidl root, so
         ``os.path.dirname(tidl_tools_path)/<rel>`` gives the .so path
      5. ``$C7X_MMA_TIDL_PATH/<rel>`` env var (no hardcoded default)
    """
    if path is None:
        path = os.environ.get("TIDL_RELAX_SO_PATH")
    if path is None:
        try:
            from tvm.data.ti_dsp.paths import find_tidl_relax_so

            path = find_tidl_relax_so()
        except ImportError:
            pass
    if path is None and tidl_tools_path is not None:
        c7x = os.path.dirname(os.path.abspath(tidl_tools_path))
        path = os.path.join(c7x, _RELAX_SO_REL)
    if path is None:
        c7x = os.environ.get("C7X_MMA_TIDL_PATH")
        if c7x:
            path = os.path.join(c7x, _RELAX_SO_REL)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"tidl_model_import_relax.so not found (tried: {path!r}).  "
            "Pass 'tidl_tools_path' in config or set C7X_MMA_TIDL_PATH."
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
            raise RuntimeError(f"FFI function '{name}' not registered after loading .so")


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


def _flatten_sinfos(sinfo) -> List:
    """Recursively flatten TupleStructInfo into a flat list of TensorStructInfos."""
    if isinstance(sinfo, relax.TupleStructInfo):
        result = []
        for field in sinfo.fields:
            result.extend(_flatten_sinfos(field))
        return result
    return [sinfo]


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
            if isinstance(val, relax.Function) and val.attrs and val.attrs.get("Composite"):
                pending_fn = val
                comp_func_map[b.var] = val
            elif isinstance(val, relax.Call):
                if pending_fn is not None:
                    # First call immediately after a composite function def
                    results.append((pending_fn, val, b.var))
                    pending_fn = None
                elif isinstance(val.op, relax.Var) and val.op in comp_func_map:
                    # Reuse of a previously defined composite function
                    results.append((comp_func_map[val.op], val, b.var))
    return results


def _make_in_out_nodes(this_node, in_names, out_names):
    """Build an :class:`InOutNodes` ctypes struct.

    Returns ``(struct, *refs)`` — the caller must keep all returned
    objects alive until the struct has been consumed by the FFI call.
    """
    this_b = this_node.encode("utf-8") if isinstance(this_node, str) else this_node

    n_in = len(in_names)
    n_out = len(out_names)

    in_bytes = [n.encode("utf-8") if isinstance(n, str) else n for n in in_names]
    out_bytes = [n.encode("utf-8") if isinstance(n, str) else n for n in out_names]

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
                        new_bindings.append(relax.VarBinding(const_var, arg))
                        new_args[k] = const_var

                new_call = relax.Call(
                    val.op,
                    new_args,
                    val.attrs,
                    val.sinfo_args,
                )
                relax._ffi_api.UpdateStructInfo(new_call, val.struct_info)
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
    artifacts_dir,
    sg_id,
    input_sinfos,
    num_frames,
    user_data=None,
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
            raise TypeError(f"calibration_data must be ndarray, got {type(user_data)}")
        return calib_path

    # Generate random float32 data (all inputs concatenated per frame)
    parts = []
    for si in input_sinfos:
        shape = [int(d) for d in si.shape]
        per_frame_shape = shape[1:] if len(shape) > 1 else shape
        data = np.random.randn(num_frames, *per_frame_shape).astype("float32")
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
        Path to lib0.out (c7x_dload) or cg_dsp executable (c7x_host).
    weights_path : Path
        Path to weights.bin (TVM model constants).
    gen_dir : Path
        Path to generated code directory (lib0.c, weights.bin, etc.).
    artifacts : dict
        TIDL artifacts per subgraph:
        ``{sg_name: {"net_bin": path, "io_bin": path}}``.
    build_dir : Path
        Path to cmake build directory.
    exec_mode : str
        Execution mode: "c7x_dload" or "c7x_host".
    """

    module_path: Path
    weights_path: Path
    gen_dir: Path
    artifacts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    build_dir: Path = field(default_factory=lambda: Path())
    exec_mode: str = "c7x_dload"

    def as_vm(self, so_path: str = "libc7x_arm_runtime.so"):  # -> C7xVirtualMachine
        """Return a C7xVirtualMachine wrapping this build result.

        Call this on the AM67A ARM board where ``libc7x_arm_runtime.so`` is
        installed.  The returned VM has the same interface as
        ``relax.VirtualMachine``:

        .. code-block:: python

            vm = result.as_vm()
            out = vm["main"](tvm.nd.array(data))

        Parameters
        ----------
        so_path : str
            Path or name of ``libc7x_arm_runtime.so``.  Defaults to the
            bare library name so ``LD_LIBRARY_PATH`` / ldconfig can find it.
        """
        from tvm.contrib.c7x import C7xVirtualMachine  # noqa: PLC0415

        return C7xVirtualMachine(self.module_path, so_path=so_path)


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
        # Apply all patterns first, then merge once.  Running
        # MergeCompositeFunctions a single time at the end (rather than after
        # each pattern) allows cross-pattern composites to merge into the same
        # Codegen="tidl" subgraph.  Cyclic dependencies in models like YOLOv8
        # are handled upstream by MergeCompositeFunctions itself (self-dep
        # insertion on non-composite bridge ops closes groups at skip
        # connections, preventing incorrect cross-cycle merging).
        for pat in patterns:
            try:
                mod = transform.FuseOpsByPattern(
                    [pat], bind_constants=True, annotate_codegen=False
                )(mod)
            except Exception as e:  # noqa: BLE001
                if "cyclic dependency" in str(e).lower():
                    logger.warning(
                        "Skipping TIDL pattern '%s': cyclic group dependency "
                        "in model graph. Op will run on TVM scalar path.",
                        pat.name,
                    )
                else:
                    raise
        try:
            mod = transform.MergeCompositeFunctions()(mod)
        except Exception as e:  # noqa: BLE001
            if "cyclic dependency" in str(e).lower():
                logger.warning(
                    "MergeCompositeFunctions: cyclic dependency, skipping merge."
                )
            else:
                raise
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
        # ---- Resolve paths --------------------------------------------
        artifacts_dir = self._artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        tidl_tools_path = self.config.get("tidl_tools_path")
        if tidl_tools_path is None:
            c7x_root = os.environ.get("C7X_MMA_TIDL_PATH")
            if c7x_root is None:
                raise ValueError(
                    "Either 'tidl_tools_path' in config or C7X_MMA_TIDL_PATH env must be set."
                )
            tidl_tools_path = os.path.join(c7x_root, "tidl_tools")

        # ---- Load .so -------------------------------------------------
        # Derived from tidl_tools_path if tidl_relax_so_path is not set;
        # both live under the same c7x-mma-tidl root.
        _load_tidl_relax_so(tidl_tools_path=tidl_tools_path)

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
        skip_failing = self.config.get("skip_failing_subgraphs", False)
        max_subgraphs = self.config.get("max_subgraphs", None)
        artifacts: Dict[str, Dict[str, str]] = {}
        failed_subgraphs: set = set()

        # If max_subgraphs is set, rank all candidate subgraphs by estimated
        # FLOPs and pre-mark the lowest compute ones as fallbacks so only the
        # top-N are imported into TIDL.
        if max_subgraphs is not None and len(subgraphs) > max_subgraphs:
            flops_list = [(gv.name_hint, _estimate_subgraph_flops(func)) for gv, func in subgraphs]
            # Sort descending by FLOPs; keep top max_subgraphs
            flops_list.sort(key=lambda x: x[1], reverse=True)
            keep = {name for name, _ in flops_list[:max_subgraphs]}
            for gv, func in subgraphs:
                if gv.name_hint not in keep:
                    failed_subgraphs.add(gv.name_hint)
            logger.info(
                "max_subgraphs=%d: offloading top-%d of %d subgraphs "
                "(by FLOPs estimate); %d fall back to TVM",
                max_subgraphs,
                max_subgraphs,
                len(subgraphs),
                len(failed_subgraphs),
            )
            for i, (name, flops) in enumerate(flops_list):
                status = "TIDL" if name in keep else "TVM "
                logger.debug("  [%s] sg%-3d %.3e FLOPs  %s", status, i, flops, name)

        # ------------------------------------------------------------------ #
        # Require calibration_inputs: random calibration produces            #
        # miscalibrated INT8 scales for intermediate subgraphs and was the  #
        # root cause of the YOLOv8 DFL NaN on c7x_dload.                   #
        # ------------------------------------------------------------------ #
        calib_frames = self.config.get("calibration_inputs")
        if not calib_frames:
            raise ValueError(
                "tidl_import() requires 'calibration_inputs': a list of "
                "per-frame numpy arrays (one (1,C,H,W) array per calibration "
                "image).  Random calibration data produces miscalibrated INT8 "
                "scales for intermediate subgraphs and is not accepted."
            )

        # ------------------------------------------------------------------ #
        # Collect real boundary activations when calibration_inputs is set.  #
        # _build_calibration_module augments main to also emit the tensor     #
        # that enters each TIDL subgraph, then lowers all TIDL ops to TVM   #
        # CPU so we can run the model on actual images.                       #
        # ------------------------------------------------------------------ #
        _sg_calib: Dict[str, np.ndarray] = {}
        if calib_frames:
            logger.info(
                "Building CPU calibration module to collect real boundary activations "
                "for %d subgraph(s) × %d frame(s)",
                len(subgraphs),
                len(calib_frames),
            )
            cpu_mod, output_map = _build_calibration_module(mod, subgraphs)
            _sg_calib = _collect_subgraph_calibration_data(
                cpu_mod, output_map, calib_frames, subgraphs
            )
            logger.info(
                "Calibration data collected for subgraphs: %s",
                list(_sg_calib.keys()),
            )

        for sg_id, (gv, func) in enumerate(subgraphs):
            sg_name = gv.name_hint

            # Pre-skip subgraphs not selected by max_subgraphs FLOPs ranking
            if sg_name in failed_subgraphs:
                logger.info(
                    "Pre-skipping TIDL import for '%s' (sg_id=%d): "
                    "not in top-%s subgraphs by FLOPs",
                    sg_name,
                    sg_id,
                    max_subgraphs if max_subgraphs is not None else "all",
                )
                continue

            logger.info("Importing TIDL subgraph %d: %s", sg_id, sg_name)

            # -- Tensor descriptors for inputs + outputs ----------------
            input_sinfos = [p.struct_info for p in func.params]
            output_sinfo = func.ret_struct_info
            if isinstance(output_sinfo, relax.TupleStructInfo):
                output_sinfos = _flatten_sinfos(output_sinfo)
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
                1,  # is_nchw
                tidl_tools_path,
                artifacts_dir,
                False,  # isSubgraphOD
            )
            if ret != 0:
                raise RuntimeError(f"TIDL_relaxImportInit failed for '{sg_name}' (rc={ret})")

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
                relax._ffi_api.UpdateStructInfo(syn_call, orig_call.struct_info)

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
                    raise RuntimeError(f"TIDL_relaxImportNode failed for '{comp_name}' (rc={ret})")

                # -- InOutNodes for linking -----------------------------
                # Input names (use Var identity via var_map)
                in_names = []
                for arg in orig_call.args:
                    if isinstance(arg, relax.Var):
                        in_names.append(var_map.get(arg, arg.name_hint))

                # Output consumer names (use Var.same_as for identity)
                out_names = []
                for j in range(i + 1, len(composites)):
                    _, future_call, _ = composites[j]
                    for farg in future_call.args:
                        if isinstance(farg, relax.Var) and farg.same_as(bvar):
                            out_names.append(str(j))

                # Terminal node: use subgraph output tensor name
                if not out_names:
                    node_name = f"tidl_{sg_id}_o0"
                else:
                    node_name = str(i)

                refs = _make_in_out_nodes(node_name, in_names, out_names)
                in_out_struct = refs[0]
                ret = link_node_fn(ctypes.c_void_p(ctypes.addressof(in_out_struct)))
                if ret != 0:
                    raise RuntimeError(f"TIDL_relaxImportLinkNode failed for node {i} (rc={ret})")

            # -- TIDL_relaxOptimizeNet ----------------------------------
            optimize_fn = tvm.get_global_func("TIDL_relaxOptimizeNet")
            ret = optimize_fn(sg_id)
            if ret != 0:
                _errmsg = f"TIDL_relaxOptimizeNet failed for '{sg_name}' (rc={ret})"
                if skip_failing:
                    logger.warning(
                        "Skipping TIDL subgraph '%s' (sg_id=%d): %s",
                        sg_name,
                        sg_id,
                        _errmsg,
                    )
                    failed_subgraphs.add(sg_name)
                    _delete_partial_artifacts(artifacts_dir, sg_id)
                    continue
                raise RuntimeError(_errmsg)

            # -- Calibration + PostProcess ------------------------------
            user_data = _sg_calib.get(sg_name)
            if user_data is None:
                raise ValueError(
                    f"No calibration data collected for TIDL subgraph '{sg_name}'. "
                    "The subgraph call was not found when augmenting main(); "
                    "check the partitioned module structure."
                )
            _write_calibration_data(
                artifacts_dir,
                sg_id,
                input_sinfos,
                num_calib_frames,
                user_data=user_data,
            )

            postprocess_fn = tvm.get_global_func("TIDL_relaxPostProcessNet")
            ret = postprocess_fn(num_calib_frames)
            if ret != 0:
                _errmsg = f"TIDL_relaxPostProcessNet failed for '{sg_name}' (rc={ret})"
                if skip_failing:
                    logger.warning(
                        "Skipping TIDL subgraph '%s' (sg_id=%d): %s",
                        sg_name,
                        sg_id,
                        _errmsg,
                    )
                    failed_subgraphs.add(sg_name)
                    _delete_partial_artifacts(artifacts_dir, sg_id)
                    continue
                raise RuntimeError(_errmsg)

            # Record artifact paths; store sg_id directly so lower_tidl
            # can recover it without parsing file names.
            artifacts[sg_name] = {
                "net_bin": os.path.join(artifacts_dir, f"subgraph{sg_id}_net.bin"),
                "io_bin": os.path.join(artifacts_dir, f"subgraph{sg_id}_params_1.bin"),
                "sg_id": sg_id,
            }
            logger.info("Subgraph %d (%s) imported successfully", sg_id, sg_name)

        self._last_failed_subgraphs = failed_subgraphs
        return mod, artifacts

    # ------------------------------------------------------------------
    # Phase 4: Lower TIDL to TIR extern calls
    # ------------------------------------------------------------------

    def lower_tidl(
        self,
        mod: IRModule,
        artifacts: Optional[Dict] = None,
        failed_subgraphs: Optional[set] = None,
    ) -> IRModule:
        """Replace TIDL subgraph functions with call_tir to extern stubs.

        Each ``Codegen="tidl"`` function is replaced with a TIR PrimFunc
        that calls ``tidl_subgraph_N_process(input_ptrs..., output_ptr)``
        via ``call_extern``.  The c_static codegen then emits the
        appropriate TIDL init/process/free lifecycle code.

        Subgraphs in ``failed_subgraphs`` are not lowered to TIDL stubs;
        instead they are kept as regular Relax functions (Codegen attr
        stripped) so the TVM compiler handles them on the scalar path.

        Parameters
        ----------
        mod : IRModule
            Partitioned module (after ``partition()``).
        artifacts : dict, optional
            TIDL artifact paths keyed by subgraph function name.  Each
            entry must contain an ``"sg_id"`` key (the original import
            loop index) so bridge symbols stay consistent when subgraphs
            are skipped.
        failed_subgraphs : set, optional
            Set of subgraph function names that failed TIDL import and
            should fall back to regular TVM compilation.

        Returns
        -------
        IRModule
            Module with TIDL functions replaced by TIR extern stubs.
        """
        return _lower_tidl_pass(
            mod,
            failed_subgraphs=failed_subgraphs or set(),
            artifacts=artifacts,
        )

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
                header_lines.append(f"void {sg['name']}_process({_process_fn_sig(sg)});")
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

    @staticmethod
    def generate_artifacts_c(artifacts_dir: str, output_path: str) -> str:
        """Generate a C source file embedding TIDL artifacts for c7x_host.

        Delegates to the module-level :func:`generate_artifacts_c`.
        See that function for full documentation.
        """
        return generate_artifacts_c(artifacts_dir, output_path)

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
        failed = getattr(self, "_last_failed_subgraphs", set())
        mod = self.lower_tidl(mod, artifacts=artifacts, failed_subgraphs=failed)
        return mod, artifacts

    # ------------------------------------------------------------------
    # Full build pipeline: compile -> codegen -> bridge -> dynmod
    # ------------------------------------------------------------------

    def build(
        self,
        mod: IRModule,
        params: Optional[Dict] = None,
        target: Optional[str] = None,
        build_dir: Optional[str] = None,
        exec_mode: str = "c7x_dload",
    ) -> TIDLBuildResult:
        """Full pipeline: compile + codegen + bridge + build.

        Produces a ready-to-deploy artifact from a Relax module in a
        single call.

        Parameters
        ----------
        mod : IRModule
            Input module (raw, before preparation).
        params : dict, optional
            Named parameters to bind as constants.
        target : str, optional
            TVM target string for c_static codegen.  If None, the default
            is chosen based on *exec_mode*: ``"c_static -mcpu=c7x
            -use-cpp-api=1"`` for c7x_dload, ``"c_static -mcpu=c7x"``
            for c7x_host.
        build_dir : str, optional
            Directory for cmake build output. If None, a temp dir is used.
        exec_mode : str
            Execution mode: ``"c7x_dload"`` (default, builds lib0.out for
            AM67A via TI C7x cross-compiler) or ``"c7x_host"`` (builds
            cg_dsp executable for x86-64 host emulation).

        Returns
        -------
        TIDLBuildResult
            Paths to the built artifact, weights.bin, generated code, and
            TIDL artifacts.  ``result.exec_mode`` reflects the mode used.
        """
        if target is None:
            target = (
                "c_static -mcpu=c7x -use-cpp-api=1"
                if exec_mode == "c7x_dload"
                else "c_static -mcpu=c7x"
            )

        # 1. Full TIDL offload pipeline
        lowered, artifacts = self.compile(mod, params)

        # 2. Compile to C via relax.build
        # Honor profile_layers config: append -profile-layers to target
        if self.config.get("profile_layers", False):
            if "-profile-layers" not in target:
                target += " -profile-layers"
        # When TIDL subgraphs are present, enable the codegen to emit
        # tidl_bridge_init_all() inside cg_main_dsp.
        if artifacts and "-tidl-runtime" not in target:
            target += " -tidl-runtime=1"
        tvm_target = tvm.target.Target(target)
        # Use cpu_generic pipeline (FuseOps+FuseTIR for op fusion) and
        # target-aware TIR pipeline (ScheduleC7xDMATiling for loop
        # reorder, decompose_reduction, DMA tiling).
        from tvm.relax.backend.cpu_generic.pipeline import (
            get_default_pipeline,
        )

        pipeline = get_default_pipeline(tvm_target)
        with tvm_target:
            with tvm.transform.PassContext(opt_level=3):
                ex = relax.build(
                    lowered,
                    target=tvm_target,
                    exec_mode="compiled",
                    system_lib=True,
                    relax_pipeline=pipeline,
                    tir_pipeline=None,  # target-aware TIR pipeline
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

        # 5. Build
        if build_dir is None:
            build_path = Path(tempfile.mkdtemp(prefix="tidl_build_cmake_"))
        else:
            build_path = Path(build_dir)
            build_path.mkdir(parents=True, exist_ok=True)

        weights_path = gen_dir / "weights.bin"

        if exec_mode == "c7x_host":
            tidl_bridge_sources: list[str] = [str(bridge_path)]
            if artifacts:
                artifacts_c = str(gen_dir / "tidl_artifacts.c")
                generate_artifacts_c(self._artifacts_dir, artifacts_c)
                tidl_bridge_sources.append(artifacts_c)
            module_path = _build_c7x_host(
                generated_dir=gen_dir,
                build_dir=build_path,
                tidl_bridge=tidl_bridge_sources,
                use_tidl=bool(artifacts),
            )
        else:
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
            exec_mode=exec_mode,
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
    # Resolve DSP build infrastructure: bundled package first, source tree fallback
    tvm_home = Path(__file__).resolve().parents[5]
    try:
        from tvm.data.ti_dsp.paths import find_dsp_runtime_dir

        dsp_runtime_dir = find_dsp_runtime_dir()
    except ImportError:
        dsp_runtime_dir = None

    if dsp_runtime_dir is None:
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
                f"CMake configuration failed (rc={result.returncode}). Check {log_path}"
            )

        result = subprocess.run(
            ["cmake", "--build", "."],
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed (rc={result.returncode}). Check {log_path}")

    module = build_dir / "lib0.out"
    if not module.exists():
        raise FileNotFoundError(f"Module not found after build: {module}")

    logger.info("Built C7x DLOAD module: %s", module)
    return module


def _build_c7x_host(
    generated_dir: Path,
    build_dir: Path,
    tidl_bridge=None,
    use_tidl: bool = False,
    tidl_artifacts_dir: Optional[str] = None,
    build_type: str = "Release",
) -> Path:
    """Build a C7x host-emulation executable from generated code.

    Uses the CMakeLists.txt at ``tests/ti-dsp-runtime/dsp-cpp/`` with
    g++ and the TI C7000 Host Emulation library.  Mirrors
    ``build_dsp_c7x_host`` in ``dsp_utils.py`` but lives in the Python
    library so ``TIDLOffloadCompiler`` does not depend on test utilities.

    Parameters
    ----------
    generated_dir : Path
        Directory containing lib0.c and weights.bin.
    build_dir : Path
        CMake build output directory.
    tidl_bridge : str | list[str], optional
        Path(s) to TIDL bridge source files.
    use_tidl : bool
        Whether to link PC TIDL algo libs (USE_TIDL cmake flag).
    tidl_artifacts_dir : str, optional
        Directory containing TIDL net.bin / io.bin files.
    build_type : str
        CMake build type (Release or Debug).

    Returns
    -------
    Path
        Path to the built cg_dsp executable.
    """
    tvm_home = Path(__file__).resolve().parents[5]
    dsp_cpp_dir = tvm_home / "tests" / "ti-dsp-runtime" / "dsp-cpp"

    generated_dir = Path(generated_dir).resolve()
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_cmd = [
        "cmake",
        "-DTVM_DSP_TARGET=c7x_host",
        f"-DTVM_HOME={tvm_home}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DGENERATED_CODE_DIR={generated_dir}",
        f"-DWEIGHTS_FILE={generated_dir / 'weights.bin'}",
    ]
    if tidl_bridge is not None:
        if isinstance(tidl_bridge, (list, tuple)):
            cmake_cmd.append("-DTIDL_BRIDGE_SOURCES=" + ";".join(str(p) for p in tidl_bridge))
        else:
            cmake_cmd.append(f"-DTIDL_BRIDGE_SOURCES={tidl_bridge}")
    if use_tidl:
        cmake_cmd.append("-DUSE_TIDL=ON")
    if tidl_artifacts_dir:
        cmake_cmd.append(f"-DTIDL_ARTIFACTS_DIR={tidl_artifacts_dir}")
    cmake_cmd.append(str(dsp_cpp_dir))

    log_path = build_dir / "cmake.log"
    logger.info("Building C7x host emulation executable: %s", generated_dir)

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
                f"CMake configuration failed (rc={result.returncode}). Check {log_path}"
            )

        result = subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed (rc={result.returncode}). Check {log_path}")

    executable = build_dir / "cg_dsp"
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found after build: {executable}")

    logger.info("Built C7x host emulation executable: %s", executable)
    return executable


# ------------------------------------------------------------------
# Internal: Bridge function generation
# ------------------------------------------------------------------


def _collect_tidl_subgraph_info(mod):
    """Collect info about TIDL subgraph TIR stubs in a lowered module.

    Returns list of dicts with keys: name, inputs, output, sg_id.
    sg_id is recovered from the ``tidl_sg_id`` attribute stored on the
    PrimFunc by ``_make_tidl_tir_stub`` so bridge symbol names stay
    consistent with the original import loop index even when subgraphs
    are skipped.
    """
    subgraphs = []
    for gv, func in mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc) and "tidl_subgraph" in gv.name_hint:
            name = gv.name_hint
            # Extract shapes/dtypes from buffer_map
            inputs = []
            outputs = []
            for param in func.params:
                buf = func.buffer_map.get(param)
                if buf is not None:
                    shape = [int(d) for d in buf.shape]
                    dtype = str(buf.dtype)
                    if param.name.startswith("output"):
                        outputs.append({"shape": shape, "dtype": dtype})
                    else:
                        inputs.append({"shape": shape, "dtype": dtype})
            # Recover original sg_id (set by _make_tidl_tir_stub)
            sg_id_attr = func.attrs.get("tidl_sg_id") if func.attrs else None
            sg_id = int(sg_id_attr) if sg_id_attr is not None else len(subgraphs)
            subgraphs.append(
                {
                    "name": name,
                    "inputs": inputs,
                    # Keep "output" (singular) for backward compat with single-output,
                    # and add "outputs" (plural) for multi-output subgraphs.
                    "output": outputs[0] if outputs else None,
                    "outputs": outputs,
                    "sg_id": sg_id,
                }
            )
    return subgraphs


def _dtype_sizeof(dtype):
    """Return sizeof() for a TVM dtype string."""
    size_map = {
        "float32": 4,
        "float16": 2,
        "float64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "uint8": 1,
        "uint16": 2,
        "uint32": 4,
        "uint64": 8,
    }
    return size_map.get(dtype, 4)


def _process_fn_sig(sg):
    """Return the argument list string for a TIDL process function.

    e.g. for 2 inputs 1 output:  "void* inp0, void* inp1, void* out0"
    e.g. for 1 input  2 outputs: "void* inp0, void* out0, void* out1"
    """
    n_in = len(sg.get("inputs", []))
    n_out = len(sg.get("outputs", sg.get("output") and [sg["output"]] or []))
    if n_out == 0:
        n_out = 1
    in_args = ", ".join(f"void* inp{j}" for j in range(n_in))
    out_args = ", ".join(f"void* out{k}" for k in range(n_out))
    return f"{in_args}, {out_args}" if in_args else out_args


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
        lines.append("#include <stdio.h>")
        lines.append('#include "tidl_api.h"')
        lines.append('#include "dlpack/dlpack.h"')
        lines.append("")
        # Use extern "C" so the bridge (compiled as C++) resolves these
        # against the C-linkage definitions in tidl_host_stubs.c /
        # the DLOAD firmware export table.
        lines.append("#ifdef __cplusplus")
        lines.append('extern "C" {')
        lines.append("#endif")
        lines.append("extern int32_t  TVM_cacheWbInvRegion(void *addr, uint32_t size);")
        # appUdmaGetObj is provided by the firmware on real hardware (c7x_dload).
        # Under HOST_EMULATION+REF_ONLY, TIDL skips the NULL UDMA check, so
        # no stub is needed and we pass NULL directly.
        lines.append("#ifndef HOST_EMULATION")
        lines.append("extern void*    appUdmaGetObj(void);")
        lines.append("#endif")
        lines.append("#ifdef __cplusplus")
        lines.append("}")
        lines.append("#endif")
        lines.append("")

        # Per-subgraph artifact symbols (from bin_to_asm.py or generate_artifacts_c).
        # Use sg_id (original import index) so symbols match the names
        # that CMakeLists.txt assigns when embedding subgraph*_net.bin.
        lines.append("/* Embedded TIDL artifacts (_binary_ symbols) */")
        for sg in real_subgraphs:
            sg_id = sg.get("sg_id", real_subgraphs.index(sg))
            lines.append(f"extern unsigned char _binary_tidl_net_{sg_id}_start[];")
            lines.append(f"extern unsigned int  _binary_tidl_net_{sg_id}_size;")
            lines.append(f"extern unsigned char _binary_tidl_io_{sg_id}_start[];")
        lines.append("")

    # Forward declarations with C linkage (lib0.c is compiled as C++)
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    for sg in real_subgraphs:
        lines.append(f"void {sg['name']}_process({_process_fn_sig(sg)});")
    lines.append("void tidl_bridge_cleanup(void);")
    lines.append("int32_t tidl_bridge_init_all(void);")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")

    for sg in real_subgraphs:
        name = sg["name"]
        sg_id = sg.get("sg_id", real_subgraphs.index(sg))
        process_fn = f"{name}_process"
        output = sg["output"]
        inputs = sg["inputs"]
        sig = _process_fn_sig(sg)

        out_shape = output["shape"]
        out_dtype = output["dtype"]
        out_bytes = 1
        for d in out_shape:
            out_bytes *= d
        out_bytes *= _dtype_sizeof(out_dtype)

        all_outputs = sg.get("outputs") or ([output] if output else [])
        n_out = len(all_outputs)

        if stub:
            # Stub mode: zero-fill all outputs
            lines.append(f"/* Stub bridge for {name} */")
            lines.append(f"void {process_fn}({sig}) {{")
            for k, out_info in enumerate(all_outputs):
                o_bytes = 1
                for d in out_info["shape"]:
                    o_bytes *= d
                o_bytes *= _dtype_sizeof(out_info.get("dtype", "float32"))
                lines.append(f"    memset(out{k}, 0, {o_bytes});")
            lines.append("}")
            lines.append("")
        else:
            lines.append(f"static void* {name}_instance = NULL;")
            lines.append("")
            lines.append(f"void {process_fn}({sig}) {{")
            lines.append(f"    if ({name}_instance == NULL) return;")
            lines.append("")

            # Build DLTensor for each input
            for j, in_info in enumerate(inputs):
                in_shape = in_info["shape"]
                lines.append(f"    DLTensor in_tensor{j};")
                lines.append(f"    memset(&in_tensor{j}, 0, sizeof(in_tensor{j}));")
                lines.append(f"    in_tensor{j}.data = inp{j};")
                lines.append(f"    in_tensor{j}.ndim = {len(in_shape)};")
                lines.append(
                    f"    int64_t in_shape{j}[] = {{{', '.join(str(d) for d in in_shape)}}};"
                )
                lines.append(f"    in_tensor{j}.shape = in_shape{j};")
                lines.append(f"    in_tensor{j}.dtype.code = kDLFloat;")
                lines.append(f"    in_tensor{j}.dtype.bits = 32;")
                lines.append(f"    in_tensor{j}.dtype.lanes = 1;")
                lines.append("")

            # Build DLTensor for each output
            for k, out_info in enumerate(all_outputs):
                o_shape = out_info["shape"]
                o_dtype = out_info.get("dtype", "float32")
                o_code = "kDLFloat" if "float" in o_dtype else "kDLInt"
                o_bits = int("".join(c for c in o_dtype if c.isdigit()) or "32")
                lines.append(f"    DLTensor out_tensor{k};")
                lines.append(f"    memset(&out_tensor{k}, 0, sizeof(out_tensor{k}));")
                lines.append(f"    out_tensor{k}.data = out{k};")
                lines.append(f"    out_tensor{k}.ndim = {len(o_shape)};")
                lines.append(
                    f"    int64_t out_shape{k}[] = {{{', '.join(str(d) for d in o_shape)}}};"
                )
                lines.append(f"    out_tensor{k}.shape = out_shape{k};")
                lines.append(f"    out_tensor{k}.dtype.code = {o_code};")
                lines.append(f"    out_tensor{k}.dtype.bits = {o_bits};")
                lines.append(f"    out_tensor{k}.dtype.lanes = 1;")
                lines.append("")

            in_ptrs = ", ".join(f"&in_tensor{j}" for j in range(len(inputs)))
            out_ptrs = ", ".join(f"&out_tensor{k}" for k in range(n_out))
            lines.append(f"    DLTensor* in[] = {{ {in_ptrs} }};")
            lines.append(f"    DLTensor* out[] = {{ {out_ptrs} }};")
            lines.append("")
            for j, in_info in enumerate(inputs):
                in_bytes = 1
                for d in in_info["shape"]:
                    in_bytes *= d
                in_bytes *= _dtype_sizeof(in_info.get("dtype", "float32"))
                lines.append(f"    TVM_cacheWbInvRegion(inp{j}, {in_bytes});")
            lines.append(f"    process_tidl_subgraph({name}_instance, in, out);")
            for k, out_info in enumerate(all_outputs):
                o_bytes = 1
                for d in out_info["shape"]:
                    o_bytes *= d
                o_bytes *= _dtype_sizeof(out_info.get("dtype", "float32"))
                lines.append(f"    TVM_cacheWbInvRegion(out{k}, {o_bytes});")
            lines.append("}")
            lines.append("")

    # Init: initialise all subgraph handles before inference.
    # Called from cg_main_dsp (emitted by codegen when tidl-runtime=1).
    # Returns 0 on success; -1 on the first subgraph that fails to init.
    if not stub and real_subgraphs:
        lines.append("int32_t tidl_bridge_init_all(void) {")
        for sg in real_subgraphs:
            n = sg["name"]
            sid = sg.get("sg_id", real_subgraphs.index(sg))
            lines.append(f"    if ({n}_instance == NULL) {{")
            lines.append(f"        {n}_instance = init_tidl_subgraph(")
            lines.append(f"            _binary_tidl_net_{sid}_start, _binary_tidl_net_{sid}_size,")
            # Under HOST_EMULATION, TIDL skips the NULL UDMA check (REF_ONLY guard),
            # so NULL is safe.  On real hardware call appUdmaGetObj() per-subgraph.
            lines.append("#ifdef HOST_EMULATION")
            lines.append(f"            _binary_tidl_io_{sid}_start, NULL, 1, 0);")
            lines.append("#else")
            lines.append(f"            _binary_tidl_io_{sid}_start, appUdmaGetObj(), 1, 0);")
            lines.append("#endif")
            lines.append(f"        if ({n}_instance == NULL) {{")
            lines.append(f'            printf("[TIDL] init failed for {n}\\n");')
            lines.append("            return -1;")
            lines.append("        }")
            lines.append("    }")
        lines.append("    return 0;")
        lines.append("}")
    else:
        lines.append("int32_t tidl_bridge_init_all(void) { return 0; }")
    lines.append("")

    # Cleanup: free all persistent TIDL handles (called at module teardown).
    if not stub and real_subgraphs:
        lines.append("void tidl_bridge_cleanup(void) {")
        for sg in real_subgraphs:
            n = sg["name"]
            lines.append(f"    if ({n}_instance != NULL) {{")
            lines.append(f"        free_tidl_subgraph({n}_instance);")
            lines.append(f"        {n}_instance = NULL;")
            lines.append("    }")
        lines.append("}")
    else:
        lines.append("void tidl_bridge_cleanup(void) {}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# Internal: TIDL lowering implementation
# ------------------------------------------------------------------


def _make_tidl_tir_stub(name, input_sinfos, output_sinfo, sg_id=None):
    """Create a TIR PrimFunc stub that calls an extern TIDL function.

    The stub extracts raw data pointers from DLTensor buffers and
    passes them to ``{name}_process(inp0, inp1, ..., out0)``.

    Parameters
    ----------
    sg_id : int, optional
        Original subgraph index from the TIDL import loop.  Stored as
        the ``tidl_sg_id`` attribute so ``_collect_tidl_subgraph_info``
        can recover it for correct bridge symbol naming.
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

    # Collect output buffers (one for tensor, N for tuple)
    if isinstance(output_sinfo, relax.TupleStructInfo):
        out_fields = list(output_sinfo.fields)
    else:
        out_fields = [output_sinfo]

    out_bufs = []
    for k, field_si in enumerate(out_fields):
        out_shape = [int(d) for d in field_si.shape]
        out_dtype = str(field_si.dtype)
        out_handle = tvm.tir.Var(f"output{k}", "handle")
        out_buf = tvm.tir.decl_buffer(out_shape, out_dtype, f"out{k}")
        params.append(out_handle)
        buffer_map[out_handle] = out_buf
        out_bufs.append(out_buf)

    extern_args = [buffer_map[h].data for h in params[: len(input_sinfos)]]
    for ob in out_bufs:
        extern_args.append(ob.data)

    # Wrap the extern call in a Block → BlockRealize so that Relax
    # analysis passes (e.g., HasReshapePattern) don't crash when
    # inspecting this function.
    extern_call = tvm.tir.Evaluate(tvm.tir.call_extern("int32", f"{name}_process", *extern_args))
    block = tvm.tir.Block([], [], [], name + "_block", extern_call)
    body = tvm.tir.BlockRealize([], tvm.tir.const(True, "bool"), block)

    func = tvm.tir.PrimFunc(params, body, buffer_map=buffer_map)
    func = func.with_attr("tir.noalias", True)
    if sg_id is not None:
        func = func.with_attr("tidl_sg_id", sg_id)
    return func


def generate_artifacts_c(artifacts_dir: str, output_path: str) -> str:
    """Generate a C source file embedding TIDL net/io artifacts as byte arrays.

    Scans ``artifacts_dir`` for ``subgraph*_net.bin`` / ``subgraph*_params_1.bin``
    files and defines ``_binary_tidl_net_N_start[]`` / ``_binary_tidl_io_N_start[]``
    symbols that match the externs emitted by ``generate_bridge(stub=False)``.

    This is the portable C equivalent of what ``bin_to_asm.py`` + ``cl7x``
    produces for the DLOAD (c7x-dynmod) path.  It lets the c7x_host build
    link real TIDL artifacts without any CMakeLists changes.

    Usage::

        artifacts_c = TIDLOffloadCompiler.generate_artifacts_c(
            artifacts_dir, str(gen_dir / "tidl_artifacts.c")
        )
        exe = build_dsp_c7x_host(
            generated_dir=gen_dir,
            tidl_bridge=[bridge_path, artifacts_c],
            use_tidl=True,
        )
    """
    import glob as _glob
    import re as _re

    lines = [
        "/* Auto-generated: TIDL artifact byte arrays for c7x_host emulation. */",
        "/* Defines _binary_tidl_net_N_start[] / _binary_tidl_io_N_start[]    */",
        "/* matching the externs in tidl_bridge.c.                             */",
        "#include <stdint.h>",
        "",
    ]

    net_bins = sorted(_glob.glob(os.path.join(artifacts_dir, "subgraph*_net.bin")))
    for net_path in net_bins:
        m = _re.search(r"subgraph(\d+)_net\.bin$", net_path)
        if not m:
            continue
        sg_id = int(m.group(1))
        io_path = os.path.join(artifacts_dir, f"subgraph{sg_id}_params_1.bin")
        if not os.path.exists(io_path):
            logger.warning("io.bin missing for sg_id=%d, skipping", sg_id)
            continue
        for sym, path in [
            (f"_binary_tidl_net_{sg_id}", net_path),
            (f"_binary_tidl_io_{sg_id}", io_path),
        ]:
            with open(path, "rb") as f:
                data = f.read()
            hex_bytes = ", ".join(f"0x{b:02x}" for b in data)
            lines += [
                f"unsigned char {sym}_start[] = {{{hex_bytes}}};",
                f"unsigned char {sym}_end[]   = {{}};",
                f"unsigned int  {sym}_size    = sizeof({sym}_start);",
                "",
            ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def _delete_partial_artifacts(artifacts_dir: str, sg_id: int) -> None:
    """Delete net.bin and params_1.bin written by a failed TIDL import.

    TIDL_relaxOptimizeNet and TIDL_relaxPostProcessNet may partially write
    these files before returning a failure code.  If left on disk they are
    picked up by the dynmod CMakeLists.txt glob (subgraph*_net.bin) and
    embedded into lib0.out, where they can corrupt TIDL internal state or
    cause algInit failures for other subgraphs at runtime.
    """
    for suffix in ("_net.bin", "_params_1.bin"):
        path = os.path.join(artifacts_dir, f"subgraph{sg_id}{suffix}")
        if os.path.exists(path):
            os.unlink(path)
            logger.debug("Deleted partial TIDL artifact: %s", path)


def _strip_codegen_attr(func: relax.Function) -> relax.Function:
    """Return a copy of func with ALL attributes removed.

    TIDL subgraph functions only carry a Codegen='tidl' attr (no
    global_symbol or other required attrs).  Removing all attrs ensures
    the function is treated as a plain private Relax function by
    relax.build() — NOT as an external codegen function.

    Setting Codegen="" (empty string) is not sufficient: the c_static
    backend checks for a non-null Codegen attr and treats such functions
    as external, emitting packed function dispatch instead of inlining.
    """
    return relax.Function(func.params, func.body, func.ret_struct_info, func.is_pure)


def _estimate_subgraph_flops(func: relax.Function) -> float:
    """Estimate FLOPs for a Codegen='tidl' subgraph function.

    Walks the composite calls in the function body and sums FLOPs for
    each op using struct_info shapes.  Used to rank subgraphs so that
    the most compute-intensive ones are preferentially offloaded to TIDL
    when ``max_subgraphs`` is set.

    Returns a float FLOPs estimate (higher = more compute-intensive).
    """

    def _prod(shape):
        p = 1
        for d in shape:
            p *= int(d)
        return p

    def _out_elems(expr):
        si = getattr(expr, "struct_info", None)
        if si is not None and hasattr(si, "shape") and si.shape is not None:
            return _prod([int(d) for d in si.shape])
        return 0

    total = [0.0]  # use list for mutation in nested function

    def _walk(expr):
        if isinstance(expr, relax.Call):
            op = expr.op
            if isinstance(op, relax.Function):
                _walk(op.body)
                return
            if isinstance(op, tvm.ir.Op):
                out = _out_elems(expr)
                name = op.name
                if name == "relax.nn.conv2d":
                    if len(expr.args) >= 2:
                        w_si = getattr(expr.args[1], "struct_info", None)
                        if w_si is not None and hasattr(w_si, "shape") and w_si.shape:
                            w = [int(d) for d in w_si.shape]
                            if len(w) == 4:  # OIHW
                                total[0] += 2.0 * out * w[1] * w[2] * w[3]
                                return
                    total[0] += 2.0 * out
                elif name == "relax.matmul":
                    if len(expr.args) >= 1:
                        a_si = getattr(expr.args[0], "struct_info", None)
                        if a_si is not None and hasattr(a_si, "shape") and a_si.shape:
                            a = [int(d) for d in a_si.shape]
                            k = a[-1] if a else 1
                            total[0] += 2.0 * out * k
                            return
                    total[0] += 2.0 * out
                else:
                    total[0] += float(out)
        elif isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    if hasattr(binding, "value"):
                        _walk(binding.value)
            _walk(expr.body)
        elif isinstance(expr, relax.Tuple):
            for field in expr.fields:
                _walk(field)

    _walk(func.body)
    return total[0]


def _expand_inline_composites(func: relax.Function) -> relax.Function:
    """Expand inline composite Function nodes in a fallback subgraph.

    After MergeCompositeFunctions, each merged TIDL subgraph function
    contains per-op composites as inline anonymous Functions (i.e. the
    ``op`` of a ``relax.Call`` is a ``relax.Function`` object, not a
    GlobalVar).  The standard TVM compilation pipeline (VMShapeLower
    etc.) cannot handle these inline lambdas.

    Uses LambdaLift + InlinePrivateFunctions on an isolated single-function
    IRModule so that all edge cases (closures, nested inline functions, etc.)
    are handled by TVM's existing passes.  Running on a mini-module rather
    than the full module avoids polluting the global function namespace or
    interacting with subsequent FuseOpsByPattern passes.
    """
    # Give _f a global_symbol so InlinePrivateFunctions treats it as a
    # public entry point and does not drop it (functions without a
    # global_symbol attr are considered private and may be removed).
    public_func = func.with_attr("global_symbol", "_f")
    tmp_mod = tvm.IRModule({"_f": public_func})
    tmp_mod = relax.transform.LambdaLift()(tmp_mod)
    tmp_mod = relax.transform.ToNonDataflow()(tmp_mod)
    tmp_mod = relax.transform.InlinePrivateFunctions()(tmp_mod)
    tmp_mod = relax.transform.ToNonDataflow()(tmp_mod)
    result = tmp_mod["_f"]
    # Strip the temporary global_symbol before returning.
    result = relax.Function(result.params, result.body, result.ret_struct_info, result.is_pure)
    return relax.analysis.remove_all_unused(result)


@mutator
class _TIDLCallReplacer(PyExprMutator):
    """Replace calls to Codegen='tidl' functions with call_tir.

    Also remaps calls to fallback functions from the old GlobalVar
    (from the original module) to the new GlobalVar registered in the
    builder, avoiding 'There is no definition of GlobalVar' errors during
    subsequent compilation passes.
    """

    def __init__(self, mod, tir_stubs, stub_gvars, fallback_gvars=None):
        super().__init__(mod)
        self._tir_stubs = tir_stubs
        self._stub_gvars = stub_gvars
        self._fallback_gvars = fallback_gvars or {}

    def visit_call_(self, call):
        call = super().visit_call_(call)
        if isinstance(call.op, relax.GlobalVar):
            name = call.op.name_hint
            if name in self._tir_stubs:
                _, _, _, output_sinfo = self._tir_stubs[name]
                tir_gv = self._stub_gvars[name]
                # call_tir expects a list of TensorStructInfo for multi-output;
                # passing TupleStructInfo directly raises a type error.
                if isinstance(output_sinfo, relax.TupleStructInfo):
                    out_sinfo_arg = list(output_sinfo.fields)
                else:
                    out_sinfo_arg = output_sinfo
                return relax.call_tir(
                    tir_gv,
                    relax.Tuple(list(call.args)),
                    out_sinfo=out_sinfo_arg,
                )
            if name in self._fallback_gvars:
                # Remap old GlobalVar → new GlobalVar in the builder.
                new_gv = self._fallback_gvars[name]
                return relax.Call(
                    new_gv,
                    call.args,
                    call.attrs,
                    call.sinfo_args,
                )
        return call


def _lower_tidl_pass(
    mod: IRModule,
    failed_subgraphs=None,
    artifacts=None,
) -> IRModule:
    """Implementation of the LowerTIDLToTIR pass.

    Parameters
    ----------
    mod : IRModule
        Partitioned module.
    failed_subgraphs : set, optional
        Names of Codegen='tidl' functions that failed TIDL import.
        These are kept as regular Relax functions (Codegen stripped)
        rather than lowered to TIR extern stubs.
    artifacts : dict, optional
        Artifact dict from tidl_import, keyed by subgraph function name.
        Each entry must contain an ``"sg_id"`` key so bridge symbol
        names stay consistent when subgraphs are skipped.
    """
    if failed_subgraphs is None:
        failed_subgraphs = set()

    # Partition Codegen='tidl' functions into success vs fallback.
    # Subgraphs with TupleStructInfo output also fall back: the bridge
    # only supports single-output subgraphs.
    success_funcs = {}
    fallback_funcs = {}
    for gv, func in mod.functions.items():
        if isinstance(func, relax.Function) and func.attrs:
            if func.attrs.get("Codegen") == "tidl":
                input_sinfos = [p.struct_info for p in func.params]
                output_sinfo = func.ret_struct_info
                is_failed = gv.name_hint in failed_subgraphs
                if is_failed:
                    fallback_funcs[gv.name_hint] = (gv, func)
                else:
                    success_funcs[gv.name_hint] = (gv, func, input_sinfos, output_sinfo)

    if not success_funcs and not fallback_funcs:
        return mod

    # Create TIR stubs for successfully imported subgraphs.
    # Use sg_id from artifacts so bridge symbols (_binary_tidl_net_N_start)
    # remain aligned with CMake's embedding even when subgraphs are skipped.
    tir_stubs = {}
    for idx, (name, (gv, func, input_sinfos, output_sinfo)) in enumerate(success_funcs.items()):
        sg_id = (artifacts or {}).get(name, {}).get("sg_id", idx)
        sg_name = f"tidl_subgraph_{idx}"
        tir_func = _make_tidl_tir_stub(sg_name, input_sinfos, output_sinfo, sg_id=sg_id)
        tir_stubs[name] = (tir_func, sg_name, input_sinfos, output_sinfo)

    # Build new module
    builder = relax.BlockBuilder()

    # Add TIR stubs for successfully imported subgraphs
    stub_gvars = {}
    for name, (tir_func, sg_name, _, _) in tir_stubs.items():
        stub_gvars[name] = builder.add_func(tir_func, sg_name)

    # Add fallback functions: expand inline composite lambdas so the
    # standard TVM pipeline can compile them, then strip the Codegen attr.
    # We process each function in isolation using _expand_inline_composites
    # rather than running LambdaLift on the whole module (which could lift
    # lambdas from unrelated functions and risk naming conflicts).
    fallback_gvars = {}
    for name, (gv, func) in fallback_funcs.items():
        clean_func = _expand_inline_composites(_strip_codegen_attr(func))
        fallback_gvars[name] = builder.add_func(clean_func, name)

    # Replace TIDL calls in main and copy remaining functions.
    # The replacer handles TIDL→TIR remapping (success_funcs) and
    # old-GVar→new-GVar remapping (fallback_funcs).
    replacer = _TIDLCallReplacer(mod, tir_stubs, stub_gvars, fallback_gvars=fallback_gvars)
    for gv, func in mod.functions.items():
        name = gv.name_hint
        if name in success_funcs:
            continue  # dropped: replaced by TIR stub
        if name in fallback_funcs:
            continue  # already added above
        if isinstance(func, relax.Function):
            new_func = replacer.visit_expr(func)
            new_func = relax.analysis.remove_all_unused(new_func)
            builder.add_func(new_func, name)
        elif isinstance(func, tvm.tir.PrimFunc):
            builder.add_func(func, name)

    result = builder.finalize()

    # Inline private fallback Relax functions into their callers.
    # Fallback functions have no global_symbol (private) and are called by
    # main via regular Relax Call nodes.  Without inlining, relax.build
    # would lower those calls through the packed API
    # (tir.anylist_setitem_call_packed) which codegen_c_static cannot emit.
    # Inlining eliminates the cross-function calls entirely.
    if fallback_funcs:
        result = relax.transform.ToNonDataflow()(result)
        # ToNonDataflow processes bindings in order and cannot remap a
        # DataflowVar that is USED before its DEFINITION in the binding
        # sequence (a forward reference).  This arises when the TIDL
        # partitioner produces a subgraph whose outputs are consumed by
        # ops that appear earlier in main's DataflowBlock binding order.
        # DataflowVar and Var share the same JSON field layout, so a
        # JSON round-trip that replaces the remaining type keys converts
        # any stale forward-reference DataflowVars safely.
        _js = tvm.ir.save_json(result)
        if '"relax.expr.DataflowVar"' in _js:
            result = tvm.ir.load_json(_js.replace('"relax.expr.DataflowVar"', '"relax.expr.Var"'))
        result = relax.transform.InlinePrivateFunctions()(result)

    return result


def _build_calibration_module(
    mod: IRModule,
    tidl_subgraphs: List[Tuple],
):
    """Build a CPU module whose main() also returns TIDL subgraph input activations.

    The returned module is compiled for LLVM and run with real calibration images
    so that ``_collect_subgraph_calibration_data`` can collect the actual
    intermediate feature-map distributions seen at each TIDL subgraph boundary.
    Without this, subgraphs 1..N receive Gaussian random calibration data whose
    range is wrong for deeper activations, causing INT8 miscalibration.

    Mechanism
    ---------
    1. Walk main's DataflowBlock to find every call to a TIDL subgraph GVar.
       Record the arguments (= subgraph input tensors) for each subgraph.
    2. Expose each DataflowVar argument as a regular (block-output) Var binding
       at the end of the DataflowBlock.  Regular Vars can appear in the
       SeqExpr return; DataflowVars cannot (they are block-scoped).
    3. Replace main's return with Tuple([original_ret, calib_var0, calib_var1, ...]).
    4. Lower all TIDL subgraphs to CPU via ``_lower_tidl_pass`` with
       failed_subgraphs = all subgraph names.  This reuses the existing
       fallback path (``_expand_inline_composites`` + ``InlinePrivateFunctions``)
       so all TIDL composite patterns are correctly expanded to TVM ops.
    5. Apply ToNonDataflow to clean up any remaining DataflowVar references.

    Parameters
    ----------
    mod : IRModule
        Partitioned module (after ``partition()``).
    tidl_subgraphs : list of (GlobalVar, relax.Function)
        Output of ``_find_tidl_subgraphs(mod)``.

    Returns
    -------
    cpu_mod : IRModule
        CPU-only module; main returns extended tuple.
    output_map : Dict[str, int]
        Maps ``"{sg_name}_i{j}"`` to a 0-based *extra* index, where
        actual tuple position = ``extra_idx + 1`` (index 0 = original output).
    """
    tidl_gvar_set = {gv for gv, _ in tidl_subgraphs}

    main_func = mod["main"]

    # ------------------------------------------------------------------ #
    # Step 1: Find the arguments passed to each TIDL subgraph call in    #
    # main's DataflowBlock(s).  These args are the subgraph's inputs —   #
    # precisely the tensors whose activation distribution we need for     #
    # accurate INT8 calibration.                                          #
    # ------------------------------------------------------------------ #
    sg_input_args: Dict[str, List] = {}  # sg_name -> [arg0, arg1, ...]
    for block in main_func.body.blocks:
        for b in block.bindings:
            val = b.value
            if isinstance(val, relax.Call) and isinstance(val.op, relax.GlobalVar):
                if val.op in tidl_gvar_set:
                    sg_name = val.op.name_hint
                    if sg_name not in sg_input_args:  # record first call only
                        sg_input_args[sg_name] = list(val.args)

    # ------------------------------------------------------------------ #
    # Step 2: Build the ordered list of extra outputs and the output_map. #
    # Order follows the tidl_subgraphs list so both functions agree on    #
    # tuple index → (sg_name, input_idx) mapping.                         #
    # ------------------------------------------------------------------ #
    extra_args: List[Tuple[str, object]] = []  # [(key, arg_expr), ...]
    output_map: Dict[str, int] = {}
    extra_idx = 0
    for gv, func in tidl_subgraphs:
        sg_name = gv.name_hint
        if sg_name not in sg_input_args:
            continue
        for j, arg in enumerate(sg_input_args[sg_name]):
            key = f"{sg_name}_i{j}"
            output_map[key] = extra_idx
            extra_args.append((key, arg))
            extra_idx += 1

    # ------------------------------------------------------------------ #
    # Step 3: Rebuild main to expose TIDL input args as extra outputs.   #
    #                                                                      #
    # DataflowVars are block-scoped and cannot appear outside the         #
    # DataflowBlock.  Wrap each one in a regular Var binding at the end   #
    # of the block ("output" bindings that escape block scope).  Regular  #
    # Vars (function params, existing block outputs) can be used directly.#
    # ------------------------------------------------------------------ #
    exposure_vars: List[relax.Var] = []
    new_blocks = []
    for block in main_func.body.blocks:
        if isinstance(block, relax.DataflowBlock) and extra_args:
            new_bindings = list(block.bindings)
            for _key, arg in extra_args:
                if isinstance(arg, relax.DataflowVar):
                    # DataflowVar → wrap in a regular Var so it can escape scope.
                    ev = relax.Var(f"_calib_{len(exposure_vars)}", arg.struct_info)
                    new_bindings.append(relax.VarBinding(ev, arg))
                    exposure_vars.append(ev)
                else:
                    # Function param or regular Var — already in scope.
                    exposure_vars.append(arg)
            new_blocks.append(relax.DataflowBlock(new_bindings))
        else:
            new_blocks.append(block)

    # Replace main's return with Tuple([original_ret, calib_var0, ...]).
    original_ret = main_func.body.body
    new_ret = relax.Tuple([original_ret] + exposure_vars)
    new_ret_sinfo = relax.TupleStructInfo(
        [main_func.ret_struct_info] + [v.struct_info for v in exposure_vars]
    )
    new_body = relax.SeqExpr(new_blocks, new_ret)
    new_main = relax.Function(
        main_func.params,
        new_body,
        new_ret_sinfo,
        main_func.is_pure,
        main_func.attrs,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the augmented module, preserving the original         #
    # GlobalVar objects so that TIDL subgraph call sites in main          #
    # (which reference those GVars) remain valid.                         #
    # ------------------------------------------------------------------ #
    main_gv = next(gv for gv in mod.functions if gv.name_hint == "main")
    aug_funcs = dict(mod.functions)
    aug_funcs[main_gv] = new_main
    augmented_mod = tvm.IRModule(aug_funcs)

    # ------------------------------------------------------------------ #
    # Step 5: Lower all TIDL subgraphs to CPU using the existing fallback #
    # path (_expand_inline_composites + InlinePrivateFunctions).          #
    # Passing all sg names as failed_subgraphs ensures no TIR stubs are   #
    # created; every TIDL function is expanded to TVM scalar ops instead. #
    # ------------------------------------------------------------------ #
    all_sg_names = {gv.name_hint for gv, _ in tidl_subgraphs}
    cpu_mod = _lower_tidl_pass(augmented_mod, failed_subgraphs=all_sg_names, artifacts={})

    # _lower_tidl_pass already applies ToNonDataflow when there are fallback
    # functions, but call it again to be safe (no-op if already clean).
    cpu_mod = relax.transform.ToNonDataflow()(cpu_mod)

    return cpu_mod, output_map


def _collect_subgraph_calibration_data(
    cpu_mod: IRModule,
    output_map: Dict[str, int],
    calibration_inputs: List,
    tidl_subgraphs: List[Tuple],
) -> Dict[str, np.ndarray]:
    """Run cpu_mod on calibration frames and collect per-subgraph activations.

    Compiles cpu_mod for LLVM (opt_level=0 for correctness over speed), runs
    it on each calibration image, and assembles a per-subgraph calibration
    binary in the same layout that ``_write_calibration_data`` writes to
    ``calib_raw_data{sg_id}.bin``.

    Layout: for each subgraph, the binary is:
        [inp0_frame0 | inp0_frame1 | ... | inp1_frame0 | ...]  (all float32)
    i.e., ``np.concatenate([np.stack(inp_j_frames) for j], axis=1).flatten()``.

    Parameters
    ----------
    cpu_mod : IRModule
        Output of ``_build_calibration_module``.
    output_map : Dict[str, int]
        Maps ``"{sg_name}_i{j}"`` to extra-tuple index (0-based).
    calibration_inputs : list
        One element per calibration frame.  Each element is either a single
        ``np.ndarray`` (for single-input models) or a list of ``np.ndarray``
        (for multi-input models).
    tidl_subgraphs : list of (GlobalVar, relax.Function)
        Used to determine the number of inputs per subgraph.

    Returns
    -------
    Dict[str, np.ndarray]
        Per-subgraph flat float32 calibration data, keyed by sg_name.
        Subgraphs absent from output_map are omitted (caller falls back
        to ``calibration_data`` or Gaussian random).
    """
    # Compile for LLVM at opt_level=0: correctness matters, not speed.
    with tvm.transform.PassContext(opt_level=0):
        ex = relax.build(
            cpu_mod,
            target=tvm.target.Target("llvm"),
            exec_mode="compiled",
            system_lib=False,
        )
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # per_sg_frames[sg_name][frame_idx][input_idx] = np.ndarray shape (1, ...)
    per_sg_frames: Dict[str, List[List[np.ndarray]]] = {}

    for calib_input in calibration_inputs:
        # Accept both a single array (single-input model) and a list of arrays.
        if isinstance(calib_input, np.ndarray):
            tvm_inputs = [tvm.runtime.tensor(calib_input, device=tvm.cpu())]
        else:
            tvm_inputs = [tvm.runtime.tensor(inp, device=tvm.cpu()) for inp in calib_input]

        # result[0] = original model output; result[extra_idx+1] = extra tensors.
        result = vm["main"](*tvm_inputs)

        frame_data: Dict[str, List[np.ndarray]] = {}
        for gv, func in tidl_subgraphs:
            sg_name = gv.name_hint
            n_inputs = len(func.params)
            inp_arrays: List[np.ndarray] = []
            for j in range(n_inputs):
                key = f"{sg_name}_i{j}"
                if key not in output_map:
                    inp_arrays.append(None)
                    continue
                inp_arrays.append(result[output_map[key] + 1].numpy())
            frame_data[sg_name] = inp_arrays

        for sg_name, inp_arrays in frame_data.items():
            per_sg_frames.setdefault(sg_name, []).append(inp_arrays)

    # ------------------------------------------------------------------ #
    # Assemble calibration binary for each subgraph.                      #
    # Layout per subgraph:                                                 #
    #   parts[j] = (num_frames, inp_j_flat_size)                          #
    #   concat along axis=1 → (num_frames, total_flat) → flatten()        #
    # Matches the layout written by _write_calibration_data (random path).#
    # ------------------------------------------------------------------ #
    sg_calib: Dict[str, np.ndarray] = {}
    for gv, func in tidl_subgraphs:
        sg_name = gv.name_hint
        frames = per_sg_frames.get(sg_name)
        if not frames:
            continue
        parts = []
        for j in range(len(func.params)):
            key = f"{sg_name}_i{j}"
            if key not in output_map:
                continue
            # frames[f][j] has shape (1, ...) — drop batch dim then flatten.
            inp_frames = np.stack(
                [frames[f][j][0].flatten() for f in range(len(frames))], axis=0
            )  # shape: (num_frames, inp_j_flat_size)
            parts.append(inp_frames)
        if parts:
            sg_calib[sg_name] = np.concatenate(parts, axis=1).flatten().astype("float32")

    return sg_calib


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
