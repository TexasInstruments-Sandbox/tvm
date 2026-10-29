#!/usr/bin/env python
"""Numeric spot checks for configurations the eligibility rules *accept*.

``test_tidl_declines.py`` shows that unsupported configurations are declined.
This file covers the other half: for each op whose rules were moved into the
TIDL library, an accepted configuration at (or next to) the boundary is built
through the real TIDL pipeline, run on the AM67A, and compared with a TVM CPU
reference. It is what catches a silent mis-parse: a wrong axis, stride or
padding is *accepted* and only shows up as a wrong value.

Output is INT8-quantized by TIDL, so the comparison is against the reference's
dynamic range, not bit-exact. Run serially, one board:

    pytest --rootdir=. tidl-tests/test_tidl_declines_hw.py --dsp-mode=c7x_dload \
        --board j722s-evm -v
"""

import os

import numpy as np
import pytest
from dsp_utils import get_current_board

import tvm
from conftest import tidl_build_and_run
from tvm import relax
from tvm.relax.backend.tidl import TIDLOffloadCompiler

C7X_MMA_TIDL_PATH = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
)

pytestmark = pytest.mark.skipif(not os.path.isfile(RELAX_SO_PATH), reason="relax .so not built")

op = relax.op
nn = relax.op.nn
F32 = "float32"
SHAPE = (1, 8, 16, 16)


def _const(shape, seed=0, scale=0.5):
    rng = np.random.RandomState(seed)
    return relax.const((rng.randn(*shape) * scale).astype(F32))


def _module(fn, shape=SHAPE):
    x = relax.Var("x", relax.TensorType(shape, F32))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(fn(bb, x))
        bb.emit_func_output(out)
    mod = bb.get()
    # The frontends mark how many of main's params are runtime inputs; codegen
    # reads it (and emits "-1 inputs" if it is missing).
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


def _split_first_slice(bb, x):
    """relu -> split into 4 channel slices -> first slice (one subgraph, one output)."""
    parts = bb.emit(op.split(nn.relu(x), 4, axis=1))
    return bb.emit(relax.TupleGetItem(parts, 0))


def _take(indices, axis):
    return _module(lambda bb, x: op.take(x, relax.const(np.array(indices, "int32")), axis=axis))


# id -> module. One hardware run per operator (plan decision D3).
CASES = {
    "softmax-width": _module(lambda bb, x: nn.softmax(x, axis=3)),
    "softmax-height": _module(lambda bb, x: nn.softmax(x, axis=2)),
    "softmax-last-neg": _module(lambda bb, x: nn.softmax(x, axis=-1)),
    "strided-slice-height": _module(lambda bb, x: op.strided_slice(x, [2], [4], [12])),
    "pad-hw": _module(lambda bb, x: nn.pad(x, [0, 0, 0, 0, 1, 2, 3, 4])),
    "concat-channel": _module(lambda bb, x: op.concat([x, nn.relu(x)], axis=1)),
    "split-channel-first-slice": _module(_split_first_slice),
    "reduce-max-height": _module(lambda bb, x: op.max(x, axis=[2], keepdims=True)),
    "mean-spatial": _module(lambda bb, x: op.mean(x, axis=[2, 3], keepdims=True)),
    "sum-channel": _module(lambda bb, x: op.sum(x, axis=[1], keepdims=True)),
    "max-pool-3x3-s2": _module(lambda bb, x: nn.max_pool2d(x, (3, 3), (2, 2), padding=(1, 1))),
    "avg-pool-2x2-s2": _module(lambda bb, x: nn.avg_pool2d(x, (2, 2), (2, 2))),
    "conv-7x7-s2": _module(
        lambda bb, x: nn.conv2d(x, _const((8, 8, 7, 7)), strides=(2, 2), padding=(3, 3))
    ),
    "conv-3x3": _module(lambda bb, x: nn.conv2d(x, _const((8, 8, 3, 3)), padding=(1, 1))),
    "conv-transpose-3x3-s2-outpad1": _module(
        lambda bb, x: nn.conv2d_transpose(
            x, _const((8, 4, 3, 3)), strides=(2, 2), padding=(1, 1), output_padding=(1, 1)
        )
    ),
    "conv-transpose-4x4-s2": _module(
        lambda bb, x: nn.conv2d_transpose(x, _const((8, 4, 4, 4)), strides=(2, 2), padding=(1, 1))
    ),
    "resize-nearest-2x": _module(
        lambda bb, x: op.image.resize2d(x, (32, 32), method="nearest_neighbor")
    ),
    "prelu-channel-slope": _module(
        lambda bb, x: nn.prelu(x, relax.const(np.linspace(0.05, 0.4, 8).astype(F32)), axis=1)
    ),
    "permute-nhwc": _module(lambda bb, x: op.permute_dims(x, [0, 2, 3, 1])),
    "reshape": _module(lambda bb, x: op.reshape(x, relax.ShapeExpr([1, 16, 8, 16]))),
    "eltwise-add": _module(lambda bb, x: op.add(x, nn.relu(x))),
    "eltwise-multiply": _module(lambda bb, x: op.multiply(x, nn.relu(x))),
    "eltwise-subtract": _module(lambda bb, x: op.subtract(x, nn.relu(x))),
    "eltwise-maximum": _module(lambda bb, x: op.maximum(x, nn.relu(x))),
    # Constant int32 indices used to be copied as float values and read back as
    # int32 bit patterns: only index 0 came out right.
    "take-const-indices-width-gather": _take([0, 2, 5, 9], 3),
    "take-const-indices-channel-gather": _take([0, 3, 4, 7], 1),
    "take-const-indices-height-slice": _take([2, 3, 4, 5], 2),
    # Imported as a global average pool, whose NC11 output stands in for Relax's NC.
    "mean-spatial-no-keepdims": _module(lambda bb, x: op.mean(x, axis=[2, 3], keepdims=False)),
    "instance-norm-spatial": _module(
        lambda bb, x: nn.instance_norm(
            x,
            relax.const(np.linspace(0.5, 1.5, 8).astype(F32)),
            relax.const(np.linspace(-0.5, 0.5, 8).astype(F32)),
            1,
            [2, 3],
        )
    ),
    "layer-norm-width": _module(
        lambda bb, x: nn.layer_norm(
            x,
            relax.const(np.linspace(0.5, 1.5, 16).astype(F32)),
            relax.const(np.linspace(-0.5, 0.5, 16).astype(F32)),
            axes=[-1],
        )
    ),
    "power-scalar-exponent": _module(
        lambda bb, x: op.power(nn.relu(x), relax.const(np.float32(2)))
    ),
    "resize-nearest-asymmetric-floor-4x": _module(
        lambda bb, x: op.image.resize2d(
            x,
            (64, 64),
            method="nearest_neighbor",
            coordinate_transformation_mode="asymmetric",
            rounding_method="floor",
        )
    ),
}


def _reference(mod, data):
    ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"](tvm.runtime.tensor(data)).numpy()


@pytest.mark.parametrize("case", sorted(CASES))
def test_accepted_config_matches_reference(case, dsp_mode, tmp_path):
    mod = CASES[case]
    rng = np.random.RandomState(123)
    data = rng.randn(*SHAPE).astype(F32)
    calib = [rng.randn(*SHAPE).astype(F32) for _ in range(4)]

    compiler = TIDLOffloadCompiler(
        {
            "board": get_current_board(),
            "artifacts_dir": str(tmp_path / "tidl_artifacts"),
            "tidl_tools_path": TIDL_TOOLS_PATH,
            "num_calibration_frames": len(calib),
            "calibration_inputs": calib,
        }
    )
    out, n_subgraphs = tidl_build_and_run(compiler, mod, {}, data, tmp_path, dsp_mode)
    assert n_subgraphs >= 1, "the op was not offloaded to TIDL"

    ref = _reference(mod, data)
    out = np.asarray(out, dtype=np.float64).reshape(ref.shape)
    span = float(np.max(np.abs(ref))) or 1.0
    err = float(np.max(np.abs(out - ref))) / span
    cos = float(
        np.dot(out.ravel(), ref.ravel()) / ((np.linalg.norm(out) * np.linalg.norm(ref)) or 1.0)
    )
    print(f"  {case}: max|err|/max|ref| = {err:.4f}, cosine = {cos:.5f}")
    # INT8 quantization noise, not a structural error: a wrong axis/stride/pad
    # gives order-1 relative error and a low cosine.
    assert err < 0.15, f"max relative error {err:.3f}"
    assert cos > 0.99, f"cosine similarity {cos:.4f}"
