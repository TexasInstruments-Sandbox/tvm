"""Unit tests for FuseQDQToTIDLMaxPool's TIDL-vs-scalar kernel selection.

Verifies the `use_tidl_maxpool` toggle picks the right call_extern target:
the TIDL kernel by default, and the scalar c7x_int8_max_pool fallback when
False (no-TIDL firmware).  No hardware needed — tests the pass at the Relax
IR level.
"""

import numpy as np
import pytest

from tvm.relax.transform.ti_const_utils import ti_tensor_const
import tvm
from tvm import relax
from tvm.relax import TensorType as TensorStructInfo

pytestmark = pytest.mark.quick


def _build_qdq_maxpool_model(out_scale=0.00935, per_channel=False):
    """Build a minimal model: dequantize(x) -> max_pool2d -> quantize(out).

    By default the DQ and Q share scale/zp, so _check_maxpool's transparent
    condition holds and the pattern fuses.
    """
    bb = relax.BlockBuilder()
    x = relax.Var("x", TensorStructInfo((1, 64, 112, 112), "int8"))
    shape = (64,) if per_channel else ()
    s = ti_tensor_const(np.full(shape, 0.00935, dtype=np.float32))
    s_out = ti_tensor_const(np.full(shape, out_scale, dtype=np.float32))
    z = ti_tensor_const(np.zeros(shape, dtype=np.int8))
    axis = 1 if per_channel else -1

    with bb.function("main", [x]):
        with bb.dataflow():
            dq = bb.emit(relax.op.dequantize(x, s, z, axis=axis))
            pool = bb.emit(
                relax.op.nn.max_pool2d(dq, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))
            )
            result = bb.emit(relax.op.quantize(pool, s_out, z, axis=axis, out_dtype="int8"))
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


def test_scale_change_is_not_fused():
    """The kernel copies int8 codes, so a pool that requantizes stays in float."""
    mod = tvm.relax.transform.FuseQDQToTIDLMaxPool()(_build_qdq_maxpool_model(0.0131))
    text = mod.script()
    assert "c7x_int8_max_pool" not in text
    assert "R.nn.max_pool2d" in text


def test_per_channel_qparams_are_not_fused():
    """Per-channel qparams leave the pool in float rather than failing the check."""
    mod = tvm.relax.transform.FuseQDQToTIDLMaxPool()(_build_qdq_maxpool_model(per_channel=True))
    text = mod.script()
    assert "c7x_int8_max_pool" not in text
    assert "R.nn.max_pool2d" in text
