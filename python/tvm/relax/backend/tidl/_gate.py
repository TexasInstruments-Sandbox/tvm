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
"""Offload-eligibility gate backed by ``TIDL_relaxAllowNode``.

``patterns.py`` matches structure only; every "can this match be offloaded"
decision is made by the TIDL import library (``tidl_model_import_relax.so``).
A :class:`FusionPattern` check built with :func:`tidl_gate` rebuilds the
composite function that ``FuseOpsByPattern`` would create for the match,
wraps it in a synthetic call (as ``TIDLOffloadCompiler.tidl_import`` does) and
asks the library.

This module sits below both ``patterns.py`` and ``tidl.py`` so that neither
needs a top-level import of the other (``tidl.py`` imports ``patterns.py``).
"""

import atexit
import contextlib
import dataclasses
import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import tvm
from tvm import relax

logger = logging.getLogger(__name__)

#: Function attribute carrying the deny/allow-list name of a composite.
NODE_NAME_ATTR = "tidl_node_name"

# Relative path from the c7x-mma-tidl root to the Relax import .so.
_RELAX_SO_REL = "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"

AllowFn = Callable[[Any], int]

# Allow function installed by partition() for the duration of fusion. Patterns
# are built and registered at import time, long before any config exists, so
# the gate closures read this module-level state instead of taking a parameter.
_active_allow_fn: Optional[AllowFn] = None

# Key of the config TIDL_relaxInit last ran with; init is skipped when equal.
_init_key: Optional[Tuple] = None

# Program index of every binding in the module being fused, installed by
# partition() before each FuseOpsByPattern call. None when a caller drives the
# patterns directly, in which case the gate falls back to a dependency order.
_binding_order: Optional[Dict[Any, int]] = None


# Per-process artifacts directory, created by default_artifacts_dir().
_default_artifacts_dir: Optional[str] = None


def default_artifacts_dir() -> str:
    """Artifacts directory for callers that name none: per process, removed at exit.

    A fixed shared path lets concurrent processes clobber each other's
    ``tidl_import`` output, and one created by another user makes the library's
    ``tempDir`` mkdir, hence ``TIDL_relaxInit``, fail.
    """
    global _default_artifacts_dir  # noqa: PLW0603
    if _default_artifacts_dir is None:
        _default_artifacts_dir = tempfile.mkdtemp(prefix="tidl_artifacts_")
        atexit.register(shutil.rmtree, _default_artifacts_dir, ignore_errors=True)
    return _default_artifacts_dir


def load_relax_so(path: Optional[str] = None, tidl_tools_path: Optional[str] = None) -> None:
    """Load ``tidl_model_import_relax.so`` and verify its FFI functions.

    Resolution order: *path*, ``TIDL_RELAX_SO_PATH``, the bundled package,
    the sibling of *tidl_tools_path*, ``$C7X_MMA_TIDL_PATH``.
    """
    if tvm.get_global_func("TIDL_relaxInit", allow_missing=True) is not None:
        return
    if path is None:
        path = os.environ.get("TIDL_RELAX_SO_PATH")
    if path is None:
        try:
            from tvm.data.ti_dsp.paths import find_tidl_relax_so  # noqa: PLC0415

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
        "TIDL_relaxAllowNode",
    ]
    for name in required:
        if tvm.get_global_func(name, allow_missing=True) is None:
            raise RuntimeError(f"FFI function '{name}' not registered after loading .so")


def resolve_tools_path(config: Dict[str, Any]) -> str:
    """``tidl_tools`` directory from *config* or ``C7X_MMA_TIDL_PATH``."""
    tools = config.get("tidl_tools_path")
    if tools is None:
        root = os.environ.get("C7X_MMA_TIDL_PATH")
        if root is None:
            raise ValueError(
                "Either 'tidl_tools_path' in config or C7X_MMA_TIDL_PATH env must be set."
            )
        tools = os.path.join(root, "tidl_tools")
    return tools


def invalidate_init() -> None:
    """Forget that the library is initialised, so the next caller re-runs init.

    An import mutates the library's global parameters, so anything that reads
    them afterwards -- the gate of a later ``partition()`` in the same process,
    for instance -- must not be served from the cached key.
    """
    global _init_key  # noqa: PLW0603
    _init_key = None


def ensure_initialized(config: Dict[str, Any], artifacts_dir: str, force: bool = False) -> None:
    """Load the ``.so`` and run ``TIDL_relaxInit`` once per distinct config.

    The library must be initialised before the first ``TIDL_relaxAllowNode``
    call (device config, options, deny lists are read at init). An import
    mutates the library's global parameters, so ``tidl_import`` passes
    ``force=True`` to start every import from freshly initialised state, and
    calls :func:`invalidate_init` afterwards so a later ``partition()`` does
    not gate against the state the import left behind.
    """
    global _init_key  # noqa: PLW0603
    tools = resolve_tools_path(config)
    load_relax_so(tidl_tools_path=tools)

    options = dict(config.get("tidl_options", {}))
    options.setdefault("tidl_tools_path", tools)
    options.setdefault("artifacts_folder", artifacts_dir)
    options = {str(k): str(v) for k, v in options.items()}

    key = tuple(sorted(options.items()))
    if key == _init_key and not force:
        return
    os.makedirs(artifacts_dir, exist_ok=True)
    ret = tvm.get_global_func("TIDL_relaxInit")(1, options)
    if ret != 0:
        raise RuntimeError(f"TIDL_relaxInit failed (rc={ret})")
    _init_key = key


_MISSING_LIBRARY_MSG = (
    "TIDL partitioning needs tidl_model_import_relax.so to decide which ops can be "
    "offloaded, and it could not be loaded. Build it (c7x-mma-tidl/build_j722s.sh) and "
    "set C7X_MMA_TIDL_PATH (or config['tidl_tools_path']), or pass config['allow_fn'] "
    "(structure-only tests use lambda call: 1)."
)


def require_library(config: Dict[str, Any], artifacts_dir: str) -> None:
    """Load and initialise the library for a gated partition, or raise clearly.

    Fails closed: an eligibility question with no library to answer it must not
    silently accept (or reject) everything.
    """
    try:
        ensure_initialized(config, artifacts_dir)
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"{_MISSING_LIBRARY_MSG} Underlying error: {e}") from e


def _so_allow_fn(call) -> int:
    """Default allow function: the library's ``TIDL_relaxAllowNode``.

    Reached when a registered pattern is used outside ``TIDLOffloadCompiler.partition``
    (which initialises the library with the compiler's config). Initialises with
    defaults from the environment on first use.
    """
    if _init_key is None:
        require_library({}, default_artifacts_dir())
    return int(tvm.get_global_func("TIDL_relaxAllowNode")(call))


@contextlib.contextmanager
def active_allow_fn(allow_fn: Optional[AllowFn]):
    """Install *allow_fn* as the gate's decision function for a ``with`` block."""
    global _active_allow_fn  # noqa: PLW0603
    prev = _active_allow_fn
    _active_allow_fn = allow_fn
    try:
        yield
    finally:
        _active_allow_fn = prev


def current_allow_fn() -> AllowFn:
    """The installed allow function, or the ``.so`` default when none is."""
    return _active_allow_fn if _active_allow_fn is not None else _so_allow_fn


def _collect_binding_order(mod) -> Dict[Any, int]:
    """``{binding var: program index}`` for every binding in *mod*."""
    order: Dict[Any, int] = {}
    functions = mod.functions  # pyright: ignore[reportAttributeAccessIssue]
    for _, fn in functions.items():
        if not isinstance(fn, relax.Function) or not isinstance(fn.body, relax.SeqExpr):
            continue
        for block in fn.body.blocks:
            for b in block.bindings:
                order[b.var] = len(order)
    return order


@contextlib.contextmanager
def binding_order(mod):
    """Record *mod*'s binding order for the gate, for one fusion pass.

    ``FuseOpsByPattern`` runs the pattern checks against *mod*'s bindings, so
    this must be installed around each individual pass application, with the
    module that pass is given.
    """
    global _binding_order  # noqa: PLW0603
    prev = _binding_order
    _binding_order = _collect_binding_order(mod)
    try:
        yield
    finally:
        _binding_order = prev


# ---------------------------------------------------------------------------
# Composite synthesis
# ---------------------------------------------------------------------------


def _subst(expr, vmap):
    """Replace variables in *expr* per *vmap* (Var/Tuple/TupleGetItem/Call)."""
    if isinstance(expr, relax.Var):
        return vmap.get(expr, expr)
    if isinstance(expr, relax.Tuple):
        return relax.Tuple([_subst(f, vmap) for f in expr.fields])
    if isinstance(expr, relax.TupleGetItem):
        return relax.TupleGetItem(_subst(expr.tuple_value, vmap), expr.index)
    if isinstance(expr, relax.Call):
        return relax.Call(
            expr.op, [_subst(a, vmap) for a in expr.args], expr.attrs, expr.sinfo_args
        )
    return expr


def _uses(expr, var) -> bool:
    """Whether *expr* references *var* directly (Var/Tuple/TupleGetItem/Call)."""
    if isinstance(expr, relax.Var):
        return expr.same_as(var)
    if isinstance(expr, relax.Call):
        return any(_uses(a, var) for a in expr.args)
    if isinstance(expr, relax.Tuple):
        return any(_uses(f, var) for f in expr.fields)
    if isinstance(expr, relax.TupleGetItem):
        return _uses(expr.tuple_value, var)
    return False


def _dependency_order(bindings, root_var) -> List[relax.Var]:
    """Matched bindings in a dependency DFS: operands before users, left first.

    Only an approximation of program order: for a match whose root has two
    computed operands this always emits the left operand's branch first,
    whatever order the enclosing block defines the branches in.
    """
    order: List[relax.Var] = []
    seen = set()

    def visit(e):
        if isinstance(e, relax.Var):
            if e in bindings and e not in seen:
                seen.add(e)
                visit(bindings[e])
                order.append(e)
        elif isinstance(e, relax.Call):
            for a in e.args:
                visit(a)
        elif isinstance(e, relax.Tuple):
            for f in e.fields:
                visit(f)
        elif isinstance(e, relax.TupleGetItem):
            visit(e.tuple_value)

    visit(bindings[root_var])
    return order


def _matched_binding_order(bindings, root_var) -> List[relax.Var]:
    """Matched binding variables in the order fusion emits them.

    ``FuseOpsByPattern`` builds the composite from the enclosing block's
    program order, which ``matched_bindings`` (an unordered map) does not
    carry, so it comes from the index map ``partition()`` installs for the
    module being fused (:func:`binding_order`).

    Getting this wrong is not cosmetic. The binding order fixes the order of
    the free variables, hence the composite's parameters, hence the synthetic
    call's arguments (``varTensorsDims``/``inputDataTypes``), and the order
    ``TIDL_relaxFindConstants`` returns inline constants in (``constants[i]``,
    which parsers index positionally). A multi-branch match ordered
    differently from the block would make the library judge a function that
    is not the one later imported.

    Without that map the dependency order is used only where it is the sole
    valid order: when each binding uses the one before it (a single chain).
    Otherwise this fails closed rather than guess.
    """
    indices = _binding_order
    if indices is not None and all(v in indices for v in bindings):
        order = sorted(bindings, key=lambda v: indices[v])
    else:
        order = _dependency_order(bindings, root_var)
        for prev, nxt in zip(order, order[1:]):
            if not _uses(bindings[nxt], prev):
                raise RuntimeError(
                    f"TIDL gate: the match rooted at {root_var.name_hint} has more than one "
                    "branch, and its binding order is unknown outside "
                    "TIDLOffloadCompiler.partition(). Wrap the FuseOpsByPattern call in "
                    "tvm.relax.backend.tidl._gate.binding_order(mod)."
                )
    # The root is the match's output, so every other matched binding feeds it
    # and already precedes it; make that explicit rather than assumed.
    return [v for v in order if v is not root_var] + [root_var]


def synthesize_composite(
    pattern_name: str, matched_expr, matched_bindings
) -> Tuple[relax.Function, List[relax.Expr], str]:
    """Rebuild the composite ``FuseOpsByPattern`` creates for a match.

    Returns ``(composite_fn, call_args, node_name)``. ``composite_fn`` carries
    ``Composite`` and ``Primitive`` attributes exactly as fusion would, plus
    ``tidl_node_name``. Free variables of the match become parameters;
    Constants stay inline (``bind_constants=True`` semantics).
    """
    bindings = dict(matched_bindings.items())
    root_var = next(v for v, val in bindings.items() if val.same_as(matched_expr))
    order = _matched_binding_order(bindings, root_var)

    # Free variables in order of first use across that binding order; this is
    # the parameter (and call argument) order the parsers see.
    free: List[relax.Var] = []

    def collect_free(e):
        if isinstance(e, relax.Var):
            if e not in bindings and e not in free:
                free.append(e)
        elif isinstance(e, relax.Call):
            for a in e.args:
                collect_free(a)
        elif isinstance(e, relax.Tuple):
            for f in e.fields:
                collect_free(f)
        elif isinstance(e, relax.TupleGetItem):
            collect_free(e.tuple_value)

    for v in order:
        collect_free(bindings[v])
    order = order[:-1]

    params = [relax.Var(v.name_hint, v.struct_info) for v in free]
    vmap = {f: p for f, p in zip(free, params)}
    bb = relax.BlockBuilder()
    with bb.function("tidl_gate_composite", params, private=True):
        with bb.dataflow():
            for v in order:
                vmap[v] = bb.emit(_subst(bindings[v], vmap), v.name_hint)
            out = bb.emit_output(_subst(bindings[root_var], vmap), "gv")
        bb.emit_func_output(out)
    func = bb.get()["tidl_gate_composite"]

    node_name = f"{pattern_name}:{root_var.name_hint}"
    func = func.with_attrs(
        {"Composite": pattern_name, "Primitive": True, NODE_NAME_ATTR: node_name}
    )
    return func, list(free), node_name


def lift_constants_in_composite(func):
    """Lift Constants inlined in Call args into separate VarBindings.

    ``TIDL_relaxFindConstants`` only finds Constants bound to their own
    VarBinding; after ``FuseOpsByPattern(bind_constants=True)`` they may be
    direct Call arguments.
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
                relax._ffi_api.UpdateStructInfo(new_call, val.struct_info)  # pyright: ignore[reportAttributeAccessIssue]
                new_bindings.append(relax.VarBinding(b.var, new_call))
            else:
                new_bindings.append(b)

    new_block = relax.DataflowBlock(new_bindings)
    new_body = relax.SeqExpr([new_block], func.body.body)
    return relax.Function(
        func.params,
        new_body,
        func.ret_struct_info,
        func.is_pure,
        func.attrs,
    )


def make_import_call(func, call_args, struct_info):
    """Synthetic call the library parses: Function as op, constants lifted."""
    call = relax.Call(lift_constants_in_composite(func), call_args)
    relax._ffi_api.UpdateStructInfo(call, struct_info)  # pyright: ignore[reportAttributeAccessIssue]
    return call


@dataclasses.dataclass
class GateResult:
    """What the gate asked the library about one pattern match."""

    pattern: str
    node_name: str
    verdict: bool
    composite: relax.Function
    call: relax.Call
    reason: str = ""


def evaluate_match(pattern_name: str, ctx) -> GateResult:
    """Synthesize the composite for *ctx*'s match and ask the allow function."""
    func, args, node_name = synthesize_composite(
        pattern_name, ctx.matched_expr, ctx.matched_bindings
    )
    call = make_import_call(func, args, ctx.matched_expr.struct_info)
    verdict = bool(current_allow_fn()(call))
    reason = ""
    if not verdict:
        reason_fn = tvm.get_global_func("TIDL_relaxGetLastRejectReason", allow_missing=True)
        reason = str(reason_fn()) if reason_fn is not None else ""
        logger.debug("TIDL declined %s: %s", node_name, reason)
    return GateResult(pattern_name, node_name, verdict, func, call, reason)


def tidl_gate(pattern_name: str) -> Callable[[Any], bool]:
    """``FusionPattern`` check that asks the TIDL library about the match."""

    def check(ctx) -> bool:
        result = evaluate_match(pattern_name, ctx)
        if _records is not None:
            _records.append(result)
        return result.verdict

    return check


# ---------------------------------------------------------------------------
# Recording (tests): capture what the gate asked the library and what it said.
# ---------------------------------------------------------------------------

_records: Optional[List[GateResult]] = None


@contextlib.contextmanager
def recording():
    """Collect a :class:`GateResult` for every pattern-check call in the block."""
    global _records  # noqa: PLW0603
    prev = _records
    _records = []
    try:
        yield _records
    finally:
        _records = prev
