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
"""Emit an ordered manifest of ``main``'s top-level compute-kernel calls.

Runs at the end of ``dataflow_lower_passes`` (after ``CallTIRRewrite`` has
collapsed ``relax.call_tir(gvar, args)`` into a direct ``Call(gvar, args)``,
but before ``finalize_passes`` injects ``relax.vm.alloc_tensor``/
``relax.vm.kill_object``/``relax.null_value`` memory-management plumbing).

That placement was verified empirically against a real per-layer DSP
profile (``-profile-layers`` on beagley-ai hardware): the manifest built at
this point matches the runtime profiler's own layer list, because the C
codegen's profiling instrumentation (``src/target/c_static/
codegen_c_static.cc``) explicitly skips ``vm.builtin`` calls the same way --
"Skip vm.builtin calls as they are just memory management operations". An
earlier candidate (end of ``legalize_passes``) over-predicted: it still
contained custom-PrimFunc reshape calls that ``RewriteDataflowReshape``
(part of ``dataflow_lower_passes``) rewrites into the builtin ``relax.reshape``
op one stage later, which never becomes a profiled call.

Backend classification is read from tags each offload path already
attaches at creation time -- never inferred from a name pattern:

- TIDL: the partitioned subgraph function's existing ``Codegen == "tidl"``
  attribute (set by TIDL's own partitioning, unchanged here).
- MMALIB: the ``c7x_offload_backend`` PrimFunc attribute each
  ``ti_mmalib_qdq_*.py``/``ti_mmalib_legalize.py``/``ti_mmalib_i16_fc.py``
  fusion pass attaches via ``call_te(..., primfunc_attrs={...})`` --  an
  existing, unmodified TVM mechanism (see ``python/tvm/relax/utils.py``'s
  ``gen_call_tir_inputs``), not a new one.
- Anything else: plain TVM-generated C7x code (hand-written vector kernels,
  fused elementwise/reshape/etc.).

Consumed by ``tvm.contrib.c7x.visualize`` (moved there from
``tvm.relax.backend.tidl.visualize`` once it stopped being TIDL-specific --
see the design doc): when the ``c7x_layers`` module attribute this pass sets
is present, per-op backend classification prefers it over the name-prefix
heuristic. Modules that never ran this pass (e.g. a TIDL partition()
snapshot that stops before dataflow_lower_passes) fall back to that
module's own pre-existing heuristics unchanged.
"""

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext


def _classify_backend(func) -> str:
    """Classify a called function's offload backend from tags it already carries."""
    if func is None or not getattr(func, "attrs", None):
        return "tvm"
    if str(func.attrs.get("Codegen", "")) == "tidl":
        return "tidl"
    backend = func.attrs.get("c7x_offload_backend")
    if backend is not None:
        return str(backend)
    return "tvm"


@tvm.transform.module_pass(opt_level=0, name="EmitC7xLayerManifest")
class EmitC7xLayerManifest:
    """Attach an ordered ``{name, backend}`` manifest of main's top-level calls."""

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        main_fn = mod["main"]
        if not isinstance(main_fn, relax.Function):
            return mod

        layers = []
        for block in main_fn.body.blocks:
            for binding in block.bindings:
                val = binding.value
                if not isinstance(val, relax.Call) or not isinstance(val.op, relax.GlobalVar):
                    continue
                func = mod.functions.get(val.op)
                layers.append(
                    {"name": val.op.name_hint, "backend": _classify_backend(func)}
                )

        return mod.with_attr("c7x_layers", layers)


@tvm.instrument.pass_instrument
class LayerManifestCapture:
    """Capture two different module snapshots as a side effect of one real
    pipeline application -- no separate/duplicate pipeline construction.

    `layers` comes from this file's own EmitC7xLayerManifest, which runs
    right after CallTIRRewrite has collapsed ``relax.call_tir(gvar, args)``
    into a direct, walkable ``Call(gvar, args)`` -- verified to match the
    DSP's real ``-profile-layers`` output at that point (see this module's
    docstring).

    `snapshot_mod` is captured *earlier*, at ``ScheduleC7xDMATiling`` (the
    last step of ``legalize_passes``, still before ``CallTIRRewrite``/
    ``dataflow_lower_passes`` run at all). That earlier point is deliberate:
    ``CallTIRRewrite`` and ``finalize_passes`` both rewrite real dataflow in
    ways that break ``tvm.contrib.c7x.visualize``'s ``_extract_graph``
    --

    - ``finalize_passes`` turns it into ``relax.vm.alloc_tensor`` buffer
      passing (a call's own bound var stops carrying the value at all).
    - ``CallTIRRewrite`` itself, even before that, doesn't preserve the
      PyTorch source span ``ti_c7x_span_utils.propagate_span`` attaches
      during ``legalize_passes`` when it rebuilds each ``relax.call_tir``
      into a direct call.

    At ``ScheduleC7xDMATiling``, calls are still the ``relax.call_tir``
    intrinsic form (``Call(op=relax.call_tir, args=[gvar, Tuple(real_args)])``)
    -- unwrapped by ``_extract_graph`` directly -- with both dataflow edges
    and source spans intact.

    Both snapshots come from whichever single pipeline application this
    instrument is attached to: a real ``relax.build()`` call (see
    ``dsp_utils.compile_for_dsp``, which also persists `layers` to
    ``layers.json``), or a bare ``get_default_pipeline(target)(mod)``
    invocation when only the manifest/snapshot are wanted and the expensive
    native compile isn't (see
    ``tvm.contrib.c7x.visualize.visualize_compile``).
    """

    def __init__(self):
        self.layers = None
        self.snapshot_mod = None

    def run_after_pass(self, mod, info):
        if info.name == "ScheduleC7xDMATiling":
            self.snapshot_mod = mod
        elif info.name == "EmitC7xLayerManifest":
            self.layers = mod.attrs.get("c7x_layers")
