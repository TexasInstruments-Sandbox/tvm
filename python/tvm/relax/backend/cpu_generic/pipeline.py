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
"""The Relax CPU backend compilation pipeline and other passes."""

import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext


@tvm.transform.module_pass(opt_level=0, name="ConvertLayoutNHWC")
class _ConvertLayoutNHWC:
    """Convert conv2d NCHW -> NHWC for C7x.

    Only applies when the module actually contains ``relax.nn.conv2d``
    ops.  This avoids triggering a ConvertLayout bug on models that
    only use conv1d with dimension-changing reshapes.
    """

    def transform_module(self, mod: IRModule, _ctx: PassContext) -> IRModule:
        conv2d_op = tvm.ir.Op.get("relax.nn.conv2d")
        found = False

        def _check(expr):
            nonlocal found
            if isinstance(expr, relax.Call) and expr.op == conv2d_op:
                found = True

        for gv in mod.functions:
            func = mod[gv]
            if isinstance(func, relax.Function):
                relax.analysis.post_order_visit(func, _check)
                if found:
                    break

        if not found:
            return mod

        try:
            mod = tvm.relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "HWIO"]})(mod)
            mod = tvm.relax.transform.OptimizeLayoutTransform()(mod)
        except Exception:
            # ConvertLayout can fail on models with mixed-dimension
            # tensors (e.g., YOLO detection heads reshape conv2d
            # outputs into 3D).  Fall back to the original NCHW layout.
            import logging

            logging.getLogger(__name__).warning(
                "ConvertLayout NCHW->NHWC failed, falling back to NCHW"
            )
        return mod


def library_dispatch_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default library dispatch passes for CPU backend."""
    return []


def legalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default legalization passes for CPU backend."""
    is_c7x = target.kind.name == "c_static" and getattr(target, "mcpu", "") == "c7x"
    passes = []

    # MMALIB QDQ fusion runs FIRST — matches the original PT2E QDQ pattern
    # (dequant→conv→quantize) before FuseQDQToInt8Conv2D rewrites it.
    # Int8 residual add fusion runs after MMALIB conv2d fusion to catch
    # the remaining dequant+add+relu+quant between MMALIB layers.
    if is_c7x and target.attrs.get("mmalib", False):
        passes.append(tvm.relax.transform.FuseMMALIBQDQConv2d())
        passes.append(tvm.relax.transform.FuseMMALIBQDQDwConv2d())
        passes.append(tvm.relax.transform.FuseMMALIBQDQFC())
        passes.append(tvm.relax.transform.FuseInt8ResidualAdd())

    # Simplify tuple indices (e.g., TupleGetItem(Tuple(a,b),0) → a)
    # before QDQ elimination — PyTorch export artifacts like pool indices
    # create dead tuple patterns that block pattern matching.
    passes.append(tvm.relax.transform.CanonicalizeBindings())
    passes.append(tvm.relax.transform.DeadCodeElimination())
    # Eliminate redundant QDQ around transparent ops (pool, reshape, etc.)
    passes.append(tvm.relax.transform.EliminateQDQTransparent())

    # QDQ passes handle remaining (non-MMALIB) quantized conv2d ops
    passes += [
        tvm.relax.transform.FuseQDQToInt8Conv2D(),
        tvm.relax.transform.EliminateQDQRoundTrip(),
        # RewriteDequantize: converts manual weight-only INT8 pattern
        # (astype(Constant(int8), "float32") * Constant(scale)) into
        # R.dequantize(). Required before FuseDequantizeMatmul.
        # Safe for all models: only matches int8 Constant + float32 astype
        # + Constant multiply — no-op for pure float or PT2E QDQ models.
        # Previously run manually in smollm_c7x.py; now in the pipeline
        # to support mixed-quantization (W8A8 MLP + weight-only attention).
        tvm.relax.transform.RewriteDequantize(),
        tvm.relax.transform.DeadCodeElimination(),
    ]

    # Int16 MMALIB for MLP layers: converts dequantize+matmul patterns
    # with MLP weight shapes to mmalib_matmul_i16 (non-bias, with shift).
    # Runs BEFORE FuseDequantizeMatmul so MLP layers are captured here;
    # remaining layers (attention) fall through to FuseDequantizeMatmul.
    if is_c7x and target.attrs.get("mmalib", False):
        from tvm.relax.transform.ti_mmalib_i16_fc import LegalizeMLPToMMALIBInt16

        passes.append(LegalizeMLPToMMALIBInt16())

    passes += [
        tvm.relax.transform.FuseDequantizeMatmul(),
    ]

    # SDPA decode fusion: replaces GQA expand+broadcast+transpose+attention_bias
    # with a single tvm_sdpa_decode extern call. Only matches decode models
    # (seq_q=1) with GQA (num_q_heads % num_kv_heads == 0).
    if is_c7x:
        from tvm.relax.transform.ti_fuse_sdpa_decode import FuseSDPADecode

        passes.append(FuseSDPADecode())

    # MMALIB path: skip NHWC, custom int16 legalization.
    # Non-MMALIB path: convert to NHWC, default legalization.
    if is_c7x and target.attrs.get("mmalib", False):
        from tvm.relax.transform.ti_mmalib_legalize import (
            _mmalib_conv2d_legalize,
            _mmalib_matmul_legalize,
        )

        passes.append(
            tvm.relax.transform.LegalizeOps(
                customize_legalize_map={
                    "relax.matmul": _mmalib_matmul_legalize,
                    "relax.nn.conv2d": _mmalib_conv2d_legalize,
                }
            )
        )
    else:
        if is_c7x:
            passes.append(_ConvertLayoutNHWC())
        passes.append(tvm.relax.transform.LegalizeOps())
    passes += [
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FoldConstant(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),
    ]
    # C7x DMA tiling: schedule conv2d PrimFuncs for L2 SRAM after fusion
    if is_c7x:
        l2_budget = int(target.attrs.get("l2-sram-size", 393216))
        passes.append(tvm.relax.transform.ScheduleC7xDMATiling(l2_budget))
    return passes


def dataflow_lower_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default dataflow lowering passes for CPU backend."""
    return [
        relax.transform.RewriteDataflowReshape(),
        relax.transform.ToNonDataflow(),
        relax.transform.RemovePurityChecking(),
        relax.transform.CallTIRRewrite(),
    ]


def finalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """The default finalization passes for CPU backend."""
    return [
        relax.transform.StaticPlanBlockMemory(),
        relax.transform.LowerAllocTensor(),
        relax.transform.KillAfterLastUse(),
        relax.transform.LowerRuntimeBuiltin(),
        relax.transform.ComputePrimValue(),
        relax.transform.VMShapeLower(),
        relax.transform.AttachGlobalSymbol(),
    ]


def get_default_pipeline(target: tvm.target.Target):
    """Return the default compilation pipeline for CPU."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            seq = tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )
            mod = seq(mod)
        return mod

    return _pipeline
