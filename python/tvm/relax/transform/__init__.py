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
"""Relax transformations."""

# isort: skip_file
# .transform must be imported FIRST — it loads the C extension and defines
# core symbols (function_pass, FuseOpsByPattern, etc.) that all downstream
# TI modules depend on at import time.

from .transform import (
    AdjustMatmulOrder,
    AllocateWorkspace,
    AlterOpImpl,
    AnnotateTIROpPattern,
    AttachAttrLayoutFreeBuffers,
    AttachGlobalSymbol,
    BindParams,
    BindSymbolicVars,
    BundleModelParams,
    CallTIRRewrite,
    CanonicalizeBindings,
    CombineParallelMatmul,
    ComputePrimValue,
    ConvertLayout,
    ConvertToDataflow,
    DataflowBlockPass,
    DataflowUseInplaceCalls,
    DeadCodeElimination,
    DecomposeOpsForInference,
    DecomposeOpsForTraining,
    EliminateCommonSubexpr,
    ExpandMatmulOfSum,
    ExpandTupleArguments,
    FewShotTuning,
    FoldConstant,
    FunctionPass,
    FuseOps,
    FuseOpsByPattern,
    FuseTIR,
    FusionPattern,
    Gradient,
    InlinePrivateFunctions,
    KillAfterLastUse,
    LambdaLift,
    LazyGetInput,
    LazySetOutput,
    LegalizeOps,
    LiftTransformParams,
    LowerAllocTensor,
    LowerRuntimeBuiltin,
    MergeCompositeFunctions,
    MetaScheduleApplyDatabase,
    MetaScheduleTuneIRMod,
    MetaScheduleTuneTIR,
    Normalize,
    NormalizeGlobalVar,
    PatternCheckContext,
    RealizeVDevice,
    RemovePurityChecking,
    RemoveUnusedOutputs,
    RemoveUnusedParameters,
    ReorderPermuteDimsAfterConcat,
    ReorderTakeAfterMatmul,
    RewriteCUDAGraph,
    RewriteDataflowReshape,
    RunCodegen,
    SpecializePrimFuncBasedOnCallSite,
    SplitCallTIRByPattern,
    SplitLayoutRewritePreproc,
    StaticPlanBlockMemory,
    ToMixedPrecision,
    ToNonDataflow,
    TopologicalSort,
    UpdateParamStructInfo,
    UpdateVDevice,
    VMBuiltinLower,
    VMShapeLower,
    dataflowblock_pass,
    function_pass,
)

# Import to register the legalization functions.
from . import legalize_ops
from .attach_external_modules import AttachExternModules
from .eliminate_qdq_roundtrip import EliminateQDQRoundTrip
from .fast_math import FastMathTransform
from .fold_batch_norm_to_conv2d_for_inference import FoldBatchnormToConv2D
from .fuse_dequantize_matmul import FuseDequantizeMatmul
from .fuse_qdq_to_int8 import FuseQDQToInt8Conv2D
from .fuse_transpose_matmul import FuseTransposeMatmul
from .ipc_allreduce_rewrite import IPCAllReduceRewrite
from .lazy_transform_params import LazyTransformParams
from .lower_gpu_ipc_alloc_storage import LowerGPUIPCAllocStorage
from .optimize_layout_transform import OptimizeLayoutTransform
from .remove_redundant_reshape import RemoveRedundantReshape
from .rewrite_dequantize import RewriteDequantize
from .schedule_c7x_dma import ScheduleC7xDMATiling
from .ti_eliminate_qdq_transparent import EliminateQDQTransparent
from .ti_fuse_qdq_tidl_activation import FuseQDQToTIDLActivation
from .ti_fuse_qdq_tidl_avgpool import FuseQDQToTIDLAvgPool
from .ti_fuse_qdq_tidl_layernorm import FuseQDQToTIDLLayerNorm
from .ti_residual_add import FuseInt8ResidualAdd, FuseInt16ResidualAdd
from .ti_mmalib_i16_fc import LegalizeMLPToMMALIBInt16
from .ti_mmalib_inject_dma import InjectMMALIBDMA
from .ti_mmalib_legalize import MMALIBLegalize
from .ti_mmalib_passes import get_mmalib_i16_fc_pass, get_mmalib_legalize_map, get_mmalib_qdq_passes
from .ti_mmalib_qdq_dwconv import FuseMMALIBQDQDwConv2d
from .ti_mmalib_qdq_fc import FuseMMALIBQDQFC, FuseMMALIBQDQFCI16
from .ti_mmalib_qdq_fusion import FuseMMALIBQDQConv2d
from .ti_mmalib_qdq_i16_conv import FuseMMALIBQDQConv2dI16
from .ti_mmalib_qdq_i16_dwconv import FuseMMALIBQDQDwConv2dI16
