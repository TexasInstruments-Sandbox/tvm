"""Regression tests for MMALIB's shared validate-then-fold bias helper.

The int16 conv2d/dwconv2d lowerers used to fold the float bias into the
int64 accumulator scale (bias / (d_scale * w_scale)) *before* checking
that the weight scale's size matched the channel count, unlike the int8
lowerers (which validate first). When the weight-scale array has a size
that is neither 1 (broadcasts fine) nor the channel count -- e.g. a
partial per-group scale from a mismatched pattern match -- slicing both
arrays to [:n_channels] can leave them with different lengths, and the
division raised an unhandled numpy ValueError instead of cleanly
declining the fusion.

_validate_and_fold_bias (ti_mmalib_legalize.py) now backs every MMALIB
QDQ lowerer (int8 and int16, conv2d/dwconv2d/FC) and always validates
before folding, closing this off structurally rather than per-caller.
This is tested directly at the helper level rather than by constructing
a Relax graph with a mismatched scale: relax.op.dequantize's own shape
inference already rejects a scale whose size doesn't match the axis it
dequantizes, so the malformed shape this guards against cannot actually
reach the lowerer through normal graph construction today -- but the
helper must still not crash if some future pattern variant ever
produces one.

Pure Python unit tests (no Relax IR, no hardware / DSP build).
"""

import numpy as np
import pytest

from tvm.relax.transform.ti_mmalib_legalize import _validate_and_fold_bias

pytestmark = pytest.mark.quick


def test_mismatched_weight_scale_declines_without_crashing():
    """size strictly between 1 and n_channels must decline, not raise.

    Reproduces the exact shapes that crashed the old (pre-refactor)
    int16 conv2d lowerer: n_channels=4, a 2-element weight scale, and a
    4-element bias -- dividing bias[:4] by (d_scale * w_scale[:4]) mixes
    shapes (4,) and (2,), which numpy cannot broadcast.
    """
    n_channels = 4
    w_scale_np = np.full(2, 0.003, dtype=np.float32)
    bias_np = np.linspace(-0.5, 0.5, n_channels, dtype=np.float32)

    result = _validate_and_fold_bias(
        w_scale_np,
        n_channels,
        bias_np,
        d_scale_val=0.002,
        o_scale_val=0.006,
        bias_dtype=np.int64,
        op_name="test op",
    )

    assert result is None


def test_validates_before_folding_int8_path_too():
    """Same mismatched-size guard applies to the int8 (int32 bias) path."""
    n_channels = 8
    w_scale_np = np.full(3, 0.01, dtype=np.float32)
    bias_np = np.linspace(-1.0, 1.0, n_channels, dtype=np.float32)

    result = _validate_and_fold_bias(
        w_scale_np,
        n_channels,
        bias_np,
        d_scale_val=0.01,
        o_scale_val=0.02,
        op_name="test op",
    )

    assert result is None


def test_matching_weight_scale_size_succeeds():
    """Sanity check: a correctly-sized weight scale still folds normally."""
    n_channels = 4
    w_scale_np = np.full(n_channels, 0.003, dtype=np.float32)
    bias_np = np.linspace(-0.5, 0.5, n_channels, dtype=np.float32)

    result = _validate_and_fold_bias(
        w_scale_np,
        n_channels,
        bias_np,
        d_scale_val=0.002,
        o_scale_val=0.006,
        bias_dtype=np.int64,
        op_name="test op",
    )

    assert result is not None
    scale_u8, shift_u8, bias_folded = result
    assert scale_u8 is not None and shift_u8 is not None and bias_folded is not None
    assert scale_u8.shape == (n_channels,)
    assert shift_u8.shape == (n_channels,)
    assert bias_folded.shape == (n_channels,)
    assert bias_folded.dtype == np.int64
