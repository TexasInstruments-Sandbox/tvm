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

# pylint: disable=invalid-name
"""The TIR backend compilation pipeline."""

import tvm
from tvm import tirx
from tvm.backend.cuda import transforms as cuda_transforms


def default_tir_pipeline():
    """The default tirx pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tirx.transform.LowerInitBlock(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tirx.transform.StmtSimplify(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            tirx.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            tirx.transform.UnrollLoop(),
            tirx.transform.StmtSimplify(),
        ]
        if not bool(config.get("tir.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.FP8ComputeLegalize(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
                tirx.transform.SplitHostDevice(),
                cuda_transforms.LowerIket(),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline, finalize_host_passes, finalize_device_passes


def tirx_pipeline():
    """The TIRX pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tirx.transform.LowerTIRx(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tirx.transform.StmtSimplify(),
            tirx.transform.LowerTIRxOpaque(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            tirx.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            tirx.transform.UnrollLoop(),
            tirx.transform.StmtSimplify(),
        ]
        if not bool(config.get("tir.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.FP8ComputeLegalize(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
                tirx.transform.SplitHostDevice(),
                cuda_transforms.LowerIket(),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline, finalize_host_passes, finalize_device_passes


def _c7x_dma_tir_pipeline():
    """TIR pipeline for c_static_lib -mcpu=c7x with async DMA lowering.

    Stripped-down fork of the default s_tir pipeline tailored for a
    single-core CPU DSP target (no GPU, no threads, no shared memory,
    no FP8/BF16).

    Key difference from the default pipeline: ``LowerAsyncDMA`` and
    ``LowerDMAToExtern`` run immediately after ``FlattenBuffer``,
    before ``NarrowDataType(32)``.  ``NarrowDataType`` converts int64
    loop indices to int32, which changes index arithmetic enough that
    ``IdentifyMemCpy`` (used by ``LowerAsyncDMA``) can no longer
    prove contiguity of the copy regions.
    """
    # Imported here so that importing tvm.tirx does not require s_tir to be loaded.
    from tvm import s_tir  # pylint: disable=import-outside-toplevel

    # Register L2 SRAM memory info so StorageRewrite can enforce
    # capacity limits and use proper alignment for global.l2sram buffers.
    # AM67A (J722S) C7x has 1.25 MB of L2 SRAM, 512-bit SIMD.
    _L2_SRAM_BYTES = 1280 * 1024  # 1.25 MB
    _C7X_SIMD_BITS = 512

    @tvm.register_global_func("tvm.info.mem.global.l2sram", override=True)
    def _mem_info_l2sram():
        return tvm.ir.make_node(
            "target.MemoryInfo",
            unit_bits=8,
            max_simd_bits=_C7X_SIMD_BITS,
            max_num_bits=_L2_SRAM_BYTES * 8,
            head_address=None,
        )

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tirx.transform.AddNoAlias(),
            s_tir.transform.LowerInitBlock(),
            s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
            s_tir.transform.ConvertBlocksToOpaque(),
            s_tir.transform.CompactBufferAllocation(),
            s_tir.transform.LowerAutoCopy(),
            s_tir.transform.LowerMatchBuffer(),
            tirx.transform.StmtSimplify(),
            tirx.transform.SimplifyPowerTwo(),
            s_tir.transform.InjectSoftwarePipeline(),
            s_tir.transform.LowerOpaqueBlock(),
            tirx.transform.HoistReductionInit(),
            tirx.transform.PromoteReductionAccumulator(),
            tirx.transform.FlattenBuffer(),
            # DMA lowering must precede NarrowDataType (see docstring).
            s_tir.transform.LowerAsyncDMA(),
            tirx.transform.LowerDMAToExtern(),
            tirx.transform.NarrowDataType(32),
            s_tir.transform.LoopPartition(),
            tirx.transform.VectorizeLoop(not bool(config.get("tirx.disable_vectorize", False))),
            s_tir.transform.InjectDoubleBuffer(),
        ]
        if not bool(config.get("tirx.disable_storage_rewrite", False)):
            passes.append(tirx.transform.StorageRewrite())
        # Inject blocking DMA for MMALIB extern PrimFuncs: prefetch input
        # (and weights when they fit) into L2 SRAM before the MMALIB call.
        # Runs after StorageRewrite (flat TIR, easy manipulation) and before
        # LowerL2SramAlloc (which converts AllocBuffer(global.l2sram) -> tvm_l2_alloc).
        from tvm.relax.transform.ti_mmalib_inject_dma import (  # pylint: disable=import-outside-toplevel
            InjectMMALIBDMA,
        )

        passes.append(InjectMMALIBDMA())
        # Lower global.l2sram allocations to inline bump allocator calls
        # before LowerTVMBuiltin converts them to TVMBackendAllocWorkspace.
        passes.append(tirx.transform.LowerL2SramAlloc())
        passes.extend(
            [
                s_tir.transform.HoistIfThenElse(),
                tirx.transform.UnrollLoop(),
                s_tir.transform.RenormalizeSplitPattern(),
                tirx.transform.StmtSimplify(),
                tirx.transform.RemoveNoOp(),
                s_tir.transform.RewriteUnsafeSelect(),
            ]
        )
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
                tirx.transform.MakePackedAPI(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline, finalize_host_passes, finalize_device_passes


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tirx.transform.LowerWarpMemory(),
        tirx.transform.StmtSimplify(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


def finalize_device_passes_tirx():  # pylint: disable=unused-argument
    """The TIRx finalization passes for TIR backend."""
    device_pass_list = [tirx.transform.LowerIntrin()]
    return tvm.ir.transform.Sequential(device_pass_list)


# global map of pre-built pipelines
PIPELINE_MAP = {"default": default_tir_pipeline, "tirx": tirx_pipeline}


def register_tir_pipeline(name: str, pipeline_factory) -> None:
    """Register a named TIR pipeline factory."""

    PIPELINE_MAP[name] = pipeline_factory


def get_tir_pipeline(name: str | None = None, **kwargs) -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline
    """
    if name == "default":
        # for now, default to s_tir pipeline
        name = "s_tir"
    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name},candidates are {list(PIPELINE_MAP.keys())}"
        )
    return PIPELINE_MAP[name](**kwargs)


def get_default_tir_pipeline(
    target: tvm.target.Target,  # pylint: disable=unused-argument
) -> tvm.transform.Pass:
    """Get the default TIR pipeline for the given target."""
    if target.kind.name == "c_static_lib" and getattr(target, "mcpu", "") == "c7x":
        return _c7x_dma_tir_pipeline()
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return get_tir_pipeline("adreno")
    else:
        return get_tir_pipeline("s_tir")
