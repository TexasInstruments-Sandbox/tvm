"""Unit tests for FuseQDQToTIDLMaxPool's TIDL-vs-scalar kernel selection.

Verifies the `use_tidl_maxpool` toggle picks the right call_extern target:
the TIDL kernel by default, and the scalar c7x_int8_max_pool fallback when
False (no-TIDL firmware).  No hardware needed — tests the pass at the Relax
IR level.
"""

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax import TensorStructInfo

pytestmark = pytest.mark.quick


def _build_qdq_maxpool_model():
    """Build a minimal model: dequantize(x) -> max_pool2d -> quantize(out).

    Uses matching scale/zp on the DQ and Q so _check_maxpool's transparent
    condition holds and the pattern fuses.
    """
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, 64, 112, 112), "int8"))
    s = relax.Constant(np.array(0.00935, dtype=np.float32))
    z = relax.Constant(np.array(0, dtype=np.int8))

    with bb.function("main", [x]):
        with bb.dataflow():
            dq = bb.emit(relax.op.dequantize(x, s, z))
            pool = bb.emit(
                relax.op.nn.max_pool2d(dq, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))
            )
            result = bb.emit(relax.op.quantize(pool, s, z, out_dtype="int8"))
            bb.emit_output(result)
        bb.emit_func_output(result)

    return bb.finalize()


def test_default_emits_tidl_kernel():
    """Default (use_tidl_maxpool=True) lowers to c7x_int8_max_pool_tidl."""
    mod = tvm.relax.transform.FuseQDQToTIDLMaxPool()(_build_qdq_maxpool_model())
    assert "c7x_int8_max_pool_tidl" in mod.script()


def test_fallback_emits_scalar_kernel():
    """use_tidl_maxpool=False lowers to the scalar c7x_int8_max_pool."""
    mod = tvm.relax.transform.FuseQDQToTIDLMaxPool(False)(_build_qdq_maxpool_model())
    text = mod.script()
    # The TIDL variant must not appear (it is absent from no-TIDL firmware);
    # the scalar kernel must.
    assert "c7x_int8_max_pool_tidl" not in text
    assert "c7x_int8_max_pool" in text
