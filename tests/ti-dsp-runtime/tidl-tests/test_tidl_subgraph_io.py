#!/usr/bin/env python
"""Regression tests for how TIDL subgraph inputs and outputs are wired.

Three bugs, each found on YOLO detection heads and each silent: the model
compiled and ran, and only the numbers were wrong.

* Multi-output subgraphs.  Only the composite with no consumer inside the
  subgraph was named after the subgraph output, so a subgraph with several
  outputs, or whose output is also read inside it, imported one output buffer.
  The others were never written on the device.  ``params_1.bin`` must declare
  one output buffer per subgraph output.
* Rank-3 layout.  A rank-3 subgraph input was described to TIDL as
  ``[A,1,1,B,C,1]`` while TIDL places a rank-3 tensor produced by a reshape
  inside the network as ``[1,1,1,A,B,C]``.  An axis-1 concat of reshape outputs
  then landed on the channel axis and its last operand came back as zeros.
* Small tensors live across a TIDL call.  The runtime put storage of 32 KB or
  less in the same L2 bank TIDL uses as scratch, so a sibling branch produced
  before a TIDL subgraph and read after it was overwritten.  This one only
  shows on the device.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tvm import relax

_TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TESTS_DIR / "dsp-cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsp_utils import get_current_board, tidl_tools_config  # noqa: E402

from tvm.relax.backend.tidl import TIDLOffloadCompiler  # noqa: E402

pytestmark = [pytest.mark.needs_tidl_so]



def _compiler(artifacts_dir, calib):
    return TIDLOffloadCompiler(
        config={
            "board": get_current_board(),
            "artifacts_dir": str(artifacts_dir),
            **tidl_tools_config(),
            "calibration_inputs": calib,
            "skip_failing_subgraphs": False,
        }
    )


def _function(x_ty, body):
    """main(x) = body(bb, x); ``num_input`` is what the frontends set."""
    x = relax.Var("x", x_ty)
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            gv = bb.emit_output(body(bb, x))
        bb.emit_func_output(gv)
    mod = bb.get()
    mod["main"] = mod["main"].with_attr("num_input", 1)
    return mod


def _cpu(bb, x):
    """An op that stays on TVM, so ``x`` reaches TIDL as a CPU-produced tensor."""
    return bb.emit(relax.op.multiply(x, relax.const(1.0, "float32")))


def _cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _run(compiler, lowered, artifacts, x, tmp_path, dsp_mode, multi_output=False):
    from dsp_utils import (  # noqa: PLC0415
        INPUT_BIN_FILE,
        run_dsp_dload,
        run_dsp_host,
        write_tensors_to_file,
    )

    result = compiler.codegen_and_build(
        lowered, artifacts, build_dir=str(tmp_path / "build"), exec_mode=dsp_mode
    )
    if dsp_mode == "c7x_host":
        write_tensors_to_file([x], str(result.build_dir / INPUT_BIN_FILE))
        out = run_dsp_host(result.module_path, working_dir=result.build_dir)
    else:
        out, _, _ = run_dsp_dload(
            result.module_path,
            result.weights_path,
            [x],
            embedded_weights=True,
            multi_output=multi_output,
        )
    return out if multi_output else np.asarray(out)


def _require_exec(dsp_mode):
    from conftest import has_c7x_host_env, has_tidl_pc_libs  # noqa: PLC0415

    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip(f"requires --dsp-mode=c7x_host or c7x_dload, got {dsp_mode!r}")
    if dsp_mode == "c7x_host" and not (has_c7x_host_env() and has_tidl_pc_libs()):
        pytest.skip("c7x_host needs TI_CGT_C7000_PATH and the PC TIDL algo libs")


# --- multi-output subgraphs ------------------------------------------------------------


def _three_outputs(bb, x):
    """conv -> relu, then two more convs on it: all three are subgraph outputs, and
    the first is also read inside the subgraph."""
    rng = np.random.default_rng(0)

    def conv(t, k):
        w = relax.const(rng.standard_normal((8, 8, 3, 3)).astype("float32") * 0.1)
        return bb.emit(relax.op.nn.relu(relax.op.nn.conv2d(t, w, padding=(1, 1, 1, 1))))

    c = conv(_cpu(bb, x), 0)
    return bb.emit(relax.Tuple([c, conv(c, 1), conv(c, 2)]))


@pytest.mark.quick
def test_every_subgraph_output_is_a_tidl_output_buffer():
    shape = (1, 8, 16, 16)
    rng = np.random.default_rng(1)
    mod = _function(relax.TensorType(shape, "float32"), _three_outputs)
    with tempfile.TemporaryDirectory() as art:
        compiler = _compiler(art, [rng.random(shape, dtype=np.float32) for _ in range(2)])
        _, artifacts = compiler.compile(mod, {})
        assert artifacts, "nothing was imported"
        assert len(artifacts) == 1, f"expected one subgraph, got {sorted(artifacts)}"
        (art,) = artifacts.values()
        names = sorted(o.name for o in art["outputs"])
        sg_id = art["sg_id"]
        assert names == [f"tidl_{sg_id}_o{k}" for k in range(3)], names


# --- rank-3 layout ---------------------------------------------------------------------

_HEADS = (8, 4, 2)  # spatial size of each detection head, like YOLO's 80/40/20 at 1/10 scale
_WIDTHS = (2, 2, 6)  # box xy, box wh, class scores


def _detect_concat(bb, x):
    """Per head: concat three rank-5 slices, reshape to (1, 3*h*w, C) (rank 3, produced
    inside the subgraph), then concat the heads on axis 1.  The rank-3 operands of the
    last concat come from reshapes, which is the layout TIDL infers on its own."""
    rows = [3 * h * h for h in _HEADS]
    total = sum(_WIDTHS)
    heads = bb.emit(relax.op.split(_cpu(bb, x), list(np.cumsum(rows)[:-1].tolist()), axis=1))
    outs = []
    for k, h in enumerate(_HEADS):
        t = bb.emit(relax.op.reshape(relax.TupleGetItem(heads, k), (1, 3, h, h, total)))
        parts = bb.emit(relax.op.split(t, list(np.cumsum(_WIDTHS)[:-1].tolist()), axis=4))
        ops = [_cpu(bb, relax.TupleGetItem(parts, j)) for j in range(len(_WIDTHS))]
        c = bb.emit(relax.op.concat(ops, axis=4))
        outs.append(bb.emit(relax.op.reshape(c, (1, rows[k], total))))
    return bb.emit(relax.op.concat(outs, axis=1))


@pytest.mark.quick
def test_rank3_concat_of_reshapes(dsp_mode, tmp_path):
    """The model is a permutation-free copy, so TIDL must return its input."""
    _require_exec(dsp_mode)
    shape = (1, sum(3 * h * h for h in _HEADS), sum(_WIDTHS))
    rng = np.random.default_rng(2)
    mod = _function(relax.TensorType(shape, "float32"), _detect_concat)
    compiler = _compiler(tmp_path / "art", [rng.random(shape, dtype=np.float32) for _ in range(2)])
    lowered, artifacts = compiler.compile(mod, {})
    assert artifacts, "nothing was imported"
    x = rng.random(shape, dtype=np.float32)
    out = _run(compiler, lowered, artifacts, x, tmp_path, dsp_mode)
    rows = np.cumsum([0] + [3 * h * h for h in _HEADS])
    for k, (a, b) in enumerate(zip(rows[:-1], rows[1:])):
        err = float(np.abs(out.reshape(shape)[:, a:b] - x[:, a:b]).max())
        assert err < 0.05, f"head {k} (rows {a}..{b}) max error {err}"


# --- small tensor live across a TIDL call -----------------------------------------------


def _sibling_across_tidl(bb, x):
    """Two branches from one CPU tensor: the TIDL branch is a conv+relu, the sibling
    stays on TVM, is small enough (4 KB) for the L2 fast pool, is produced before the
    TIDL call and read after it, by a TVM op."""
    w = relax.const(np.random.default_rng(3).standard_normal((8, 8, 3, 3)).astype("float32") * 0.1)
    y = _cpu(bb, x)
    sibling = bb.emit(relax.op.subtract(y, relax.const(0.5, "float32")))
    tidl = bb.emit(relax.op.nn.relu(relax.op.nn.conv2d(y, w, padding=(1, 1, 1, 1))))
    # Multiplying by a constant is not offloaded, so the sibling is read on TVM.
    return bb.emit(
        relax.Tuple([tidl, bb.emit(relax.op.multiply(sibling, relax.const(1.0, "float32")))])
    )


@pytest.mark.core
def test_small_tensor_live_across_tidl_call(dsp_mode, tmp_path):
    """Host emulation does not share L2 with TIDL, so this is only meaningful on the
    board; it runs on c7x_host too so a wiring mistake shows up without one."""
    _require_exec(dsp_mode)
    shape = (1, 8, 8, 16)  # 4 KB, under the 32 KB fast-pool threshold
    rng = np.random.default_rng(4)
    mod = _function(relax.TensorType(shape, "float32"), _sibling_across_tidl)
    compiler = _compiler(tmp_path / "art", [rng.random(shape, dtype=np.float32) for _ in range(2)])
    lowered, artifacts = compiler.compile(mod, {})
    assert len(artifacts) == 1, f"expected one subgraph, got {sorted(artifacts)}"
    x = rng.random(shape, dtype=np.float32)
    out = _run(compiler, lowered, artifacts, x, tmp_path, dsp_mode, multi_output=True)
    sibling = np.asarray(out[1]).reshape(shape)
    err = float(np.abs(sibling - (x - 0.5)).max())
    assert err < 1e-5, f"TVM-side tensor read after the TIDL call is wrong: max error {err}"
