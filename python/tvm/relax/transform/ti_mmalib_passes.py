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
"""Central registry for all MMALIB-specific pipeline passes.

The c_static pipeline (pipeline.py) has three distinct MMALIB insertion
points, each constrained to a specific position in the overall pass order:

  1. QDQ fusion — BEFORE FuseQDQToInt8Conv2D
       FuseMMALIBQDQConv2d, FuseMMALIBQDQDwConv2d, FuseMMALIBQDQFC,
       FuseInt8ResidualAdd, and their int16 counterparts (Phase 2b/2c).
       These must see the intact PT2E QDQ graph before the standard
       QDQ elimination passes remove the quantize/dequantize nodes.

  2. Int16 FC legalization — AFTER RewriteDequantize, BEFORE FuseDequantizeMatmul
       LegalizeMLPToMMALIBInt16 converts weight-only int8 matmul patterns to
       MMALIB int16 calls (LLM inference path, no calibration needed).

  3. Custom LegalizeOps map — replaces the default LegalizeOps call
       Provides MMALIB-specific legalization for float32 conv2d and matmul,
       bypassing the default loop-based legalization.

Adding a new MMALIB pass: update the relevant factory function below.
pipeline.py needs no changes — it calls these three functions unchanged.

All imports are deferred (inside functions) so this module can be imported
even when the MMALIB headers or compiled TVM extensions are unavailable.
"""


def get_mmalib_qdq_passes() -> list:
    """Return the ordered list of QDQ fusion passes for MMALIB.

    Must be inserted BEFORE FuseQDQToInt8Conv2D in pipeline.py so that
    these passes see the original PT2E QDQ structure.

    Phase 2b (active): FuseMMALIBQDQConv2dI16, FuseMMALIBQDQFCI16, FuseInt16ResidualAdd.
    Phase 2c (pending): FuseMMALIBQDQDwConv2dI16 — uncomment when implemented.
    """
    from tvm.relax.transform.ti_mmalib_qdq_dwconv import FuseMMALIBQDQDwConv2d
    from tvm.relax.transform.ti_mmalib_qdq_fc import FuseMMALIBQDQFC
    from tvm.relax.transform.ti_mmalib_qdq_fusion import FuseMMALIBQDQConv2d
    from tvm.relax.transform.ti_residual_add import FuseInt8ResidualAdd, FuseInt16ResidualAdd

    # --- int8 QDQ passes (Phase 1, active) ---
    passes = [
        FuseMMALIBQDQConv2d(),
        FuseMMALIBQDQDwConv2d(),
        FuseMMALIBQDQFC(),
        FuseInt8ResidualAdd(),
    ]

    # --- int16 QDQ passes (Phase 2b active, Phase 2c dwconv pending) ---
    from tvm.relax.transform.ti_mmalib_qdq_fc import FuseMMALIBQDQFCI16
    from tvm.relax.transform.ti_mmalib_qdq_i16_conv import FuseMMALIBQDQConv2dI16

    passes += [
        FuseMMALIBQDQConv2dI16(),
        FuseMMALIBQDQFCI16(),
        FuseInt16ResidualAdd(),  # int16 skip-connection fusion (Phase 2c)
    ]

    # --- int16 depthwise QDQ pass (Phase 2c, active) ---
    from tvm.relax.transform.ti_mmalib_qdq_i16_dwconv import FuseMMALIBQDQDwConv2dI16

    passes += [FuseMMALIBQDQDwConv2dI16()]

    return passes


def get_mmalib_i16_fc_pass():
    """Return the int16 FC legalization pass for LLM inference.

    Must be inserted AFTER RewriteDequantize and BEFORE FuseDequantizeMatmul
    in pipeline.py so that it sees the dequantize(w_int8) + matmul pattern
    produced by RewriteDequantize, and MLP layers are captured before the
    generic FuseDequantizeMatmul pass consumes them.
    """
    from tvm.relax.transform.ti_mmalib_i16_fc import LegalizeMLPToMMALIBInt16

    return LegalizeMLPToMMALIBInt16()


def get_mmalib_legalize_map() -> dict:
    """Return the custom LegalizeOps map for MMALIB int16 kernels.

    Used by pipeline.py to replace the default LegalizeOps() call with
    an MMALIB-specific version that routes float32 conv2d and matmul to
    mmalib_conv2d_i16 and mmalib_matmul_i16 respectively.

    Returns a dict suitable for:
        LegalizeOps(customize_legalize_map=get_mmalib_legalize_map())
    """
    from tvm.relax.transform.ti_mmalib_legalize import (
        _mmalib_conv2d_legalize,
        _mmalib_matmul_legalize,
    )

    return {
        "relax.matmul": _mmalib_matmul_legalize,
        "relax.nn.conv2d": _mmalib_conv2d_legalize,
    }
