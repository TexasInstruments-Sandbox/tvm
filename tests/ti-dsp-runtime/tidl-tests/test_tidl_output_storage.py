#!/usr/bin/env python
"""Regression tests for TIDL output storage padding.

TIDL writes past the logical output tensor: it appends ``outPadCh`` padding
channels plus a spare, each ``outChannelPitch`` elements.  The Relax memory
planner sizes the output from shape x dtype alone, so ``algProcess`` overran the
allocation and corrupted the tail of its own output -- correct on the first
inference after a module load, wrong on every one after that.

``PadTIDLOutputStorage`` grows the allocation (not the tensor) so the padding
lands in owned memory.  Two layers of coverage here:

  * host-only unit tests for the pass and the descriptor reader, which need no
    board and no TIDL install;
  * a hardware test that runs one TIDL module under both c7x_compute verbs.
    ``run`` infers once, ``profile`` infers twice and reads back the second
    result, so a mismatch between them is exactly this bug.  The model's output
    must be big enough to reach the overrun -- an 8 KiB output does, the 32-byte
    output in dsp-tests/test_c7x_vm_dsp.py's MLP never could, which is why that
    suite's repeated-inference test did not catch this.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm import relax

_TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TESTS_DIR / "dsp-cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tvm.relax.backend.tidl.output_storage import (  # noqa: E402
    MODULE_ATTR,
    PadTIDLOutputStorage,
)

_HW = 16
_CHANNELS = 8
_LOGICAL_BYTES = _CHANNELS * _HW * _HW * 4  # 8192

C7X_MMA_TIDL_PATH = os.environ.get(
    "C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl")
)
TIDL_TOOLS_PATH = os.path.join(C7X_MMA_TIDL_PATH, "tidl_tools")
RELAX_SO_PATH = os.path.join(
    C7X_MMA_TIDL_PATH, "ti_dl/utils/tidlModelImport/out/tidl_model_import_relax.so"
)


def _module_with_tidl_call(padded: int):
    """Build main() calling a TIDL subgraph stub, shaped like the real pipeline.

    Mirrors what the pipeline produces just after StaticPlanBlockMemory: the
    returned tensor is still a builtin.alloc_tensor (the planner will not place
    a function result), and the subgraph is a direct GlobalVar call with the
    output appended to the arguments by CallTIRRewrite.
    """
    from tvm.script import tir as T

    @T.prim_func
    def tidl_subgraph_0(a: T.handle, b: T.handle):
        T.evaluate(0)

    shape = (1, _CHANNELS, _HW, _HW)
    bb = relax.BlockBuilder()
    gv_sub = bb.add_func(tidl_subgraph_0, "tidl_subgraph_0")
    x = relax.Var("x", relax.TensorStructInfo(shape, "float32"))
    with bb.function("main", [x]):
        alloc = bb.emit(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr(list(shape)), relax.DataTypeImm("float32"), 0, "global"
            ),
            name_hint="alloc",
        )
        bb.emit(
            relax.Call(gv_sub, [x, alloc], sinfo_args=[relax.ObjectStructInfo()]),
            name_hint="_tidl_call",
        )
        bb.emit_func_output(alloc)
    mod = bb.get()
    if padded:
        mod = mod.with_attr(MODULE_ATTR, {"tidl_subgraph_0": [padded]})
    return mod


def _alloc_storage_sizes(mod) -> list:
    """Byte sizes of every memory.alloc_storage in main, in order."""
    op = tvm.ir.Op.get("relax.memory.alloc_storage")
    sizes = []
    for block in mod["main"].body.blocks:
        for binding in block.bindings:
            value = getattr(binding, "value", None)
            if isinstance(value, relax.Call) and value.op == op:
                sizes.append(int(value.args[0].values[0]))
    return sizes


def _has_builtin_alloc(mod) -> bool:
    op = tvm.ir.Op.get("relax.builtin.alloc_tensor")
    for block in mod["main"].body.blocks:
        for binding in block.bindings:
            value = getattr(binding, "value", None)
            if isinstance(value, relax.Call) and value.op == op:
                return True
    return False


@pytest.mark.quick
@pytest.mark.core
class TestPadTIDLOutputStoragePass:
    """Pass behaviour, no board and no TIDL install required."""

    def test_splits_builtin_alloc_into_padded_storage(self):
        """builtin.alloc_tensor becomes oversized storage + same-shape view."""
        padded = _LOGICAL_BYTES + 4096
        mod = PadTIDLOutputStorage()(_module_with_tidl_call(padded))

        assert _alloc_storage_sizes(mod) == [padded], (
            "the TIDL output must be backed by a storage of exactly the "
            "requested size"
        )
        assert not _has_builtin_alloc(mod), (
            "builtin.alloc_tensor has no size argument, so it must be replaced "
            "rather than left in place"
        )
        # The tensor itself must be untouched: same shape and dtype, offset 0.
        # That is what makes this fix copy-free.
        out_sinfo = mod["main"].ret_struct_info
        assert list(out_sinfo.shape) == [1, _CHANNELS, _HW, _HW]
        assert out_sinfo.dtype == "float32"

        alloc_tensor = tvm.ir.Op.get("relax.memory.alloc_tensor")
        offsets = [
            int(b.value.args[1].value)
            for block in mod["main"].body.blocks
            for b in block.bindings
            if isinstance(getattr(b, "value", None), relax.Call)
            and b.value.op == alloc_tensor
        ]
        assert offsets == [0], "the tensor must sit at offset 0 of its storage"

    def test_noop_without_attribute(self):
        """Non-TIDL builds share this pipeline, so no attribute means no change."""
        mod = _module_with_tidl_call(padded=0)
        out = PadTIDLOutputStorage()(mod)
        assert _has_builtin_alloc(out), "allocation must be left alone"
        assert _alloc_storage_sizes(out) == []

    def test_noop_when_allocation_already_large_enough(self):
        """A requirement at or below the logical size must not shrink anything."""
        mod = PadTIDLOutputStorage()(_module_with_tidl_call(padded=1024))
        # 1024 < 8192 logical: splitting would under-allocate, so the pass may
        # either leave builtin.alloc_tensor alone or emit storage >= logical.
        sizes = _alloc_storage_sizes(mod)
        assert all(s >= _LOGICAL_BYTES for s in sizes), (
            f"storage must never be smaller than the tensor: {sizes}"
        )

    def test_survives_the_rest_of_lowering(self):
        """The rewritten IR must still lower; it runs mid-pipeline."""
        mod = PadTIDLOutputStorage()(_module_with_tidl_call(_LOGICAL_BYTES + 4096))
        mod = relax.transform.LowerAllocTensor()(mod)
        mod = relax.transform.KillAfterLastUse()(mod)
        mod = relax.transform.LowerRuntimeBuiltin()(mod)
        assert mod["main"] is not None


@pytest.mark.quick
@pytest.mark.core
class TestIOBufDescReader:
    """The descriptor reader must validate its own layout assumptions."""

    def test_rejects_layout_mismatch(self, tmp_path):
        """A short blob must raise, not silently read fields at wrong offsets.

        tidl_api.c documents how a TIDL_MAX_NUM_CORES mismatch shifts every
        field in this struct; reading garbage offsets would produce a plausible
        but wrong allocation size.
        """
        from tvm.relax.backend.tidl.io_buf_desc import parse_io_buf_desc

        header = os.path.join(C7X_MMA_TIDL_PATH, "ti_dl", "inc", "itidl_io.h")
        if not os.path.isfile(header):
            pytest.skip("itidl_io.h not available")
        bogus = tmp_path / "subgraph0_params_1.bin"
        bogus.write_bytes(b"\0" * 64)
        with pytest.raises(ValueError, match="layout mismatch"):
            parse_io_buf_desc(str(bogus), header_path=header)


class TestTIDLRepeatedInference:
    """Hardware: a second inference must match the first, byte for byte."""

    @pytest.fixture(autouse=True)
    def _check_deps(self, dsp_mode):
        if dsp_mode != "c7x_dload":
            pytest.skip("requires --dsp-mode=c7x_dload (needs a board)")
        if os.environ.get("TI_CGT_C7000_PATH") is None:
            pytest.skip("TI_CGT_C7000_PATH not set")
        if not os.path.isfile(RELAX_SO_PATH):
            pytest.skip(f"tidl_model_import_relax.so not found at {RELAX_SO_PATH}")

    def test_repeat_matches_single_inference(self, tmp_path):
        """Run one TIDL module under both c7x_compute verbs and compare.

        ``profile`` sets repeat=2 and returns the second inference's output, so
        any difference from ``run`` means state or memory from the first
        inference leaked into the second.  The output is 8 KiB, chosen to exceed
        TIDL's padding overrun -- smaller outputs cannot expose it.
        """
        from tvm.relax.backend.tidl import TIDLOffloadCompiler
        from tvm.relax.frontend import nn

        from dsp_utils import run_dsp_dload  # noqa: PLC0415

        class ConvReluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2D(3, _CHANNELS, 3, 1, 1, bias=False)

            def main(self, x):
                return nn.relu(self.conv(x))

        input_spec = {"x": nn.spec.Tensor((1, 3, _HW, _HW), "float32")}
        mod, param_spec = ConvReluModel().export_tvm(spec={"main": input_spec})
        np.random.seed(42)
        params = [
            tvm.runtime.tensor(
                np.random.rand(*p.shape).astype("float32"), device=tvm.cpu()
            )
            for _, p in param_spec
        ]
        param_dict = dict(zip(mod["main"].params[1:], params))
        input_data = np.random.randn(1, 3, _HW, _HW).astype("float32")

        compiler = TIDLOffloadCompiler(
            config={
                "artifacts_dir": str(tmp_path / "tidl_artifacts"),
                "tidl_tools_path": TIDL_TOOLS_PATH,
                "num_calibration_frames": 2,
                "calibration_inputs": [
                    np.random.randn(1, 3, _HW, _HW).astype("float32") for _ in range(2)
                ],
            }
        )
        result = compiler.build(
            mod,
            params=param_dict,
            exec_mode="c7x_dload",
            build_dir=str(tmp_path / "build"),
        )
        assert len(result.artifacts) > 0, "model did not offload to TIDL"

        single, _, _ = run_dsp_dload(
            result.module_path,
            result.weights_path,
            [input_data],
            embedded_weights=True,
            profile=False,
        )
        repeated, _, _ = run_dsp_dload(
            result.module_path,
            result.weights_path,
            [input_data],
            embedded_weights=True,
            profile=True,
        )
        single = np.asarray(single).reshape(-1)
        repeated = np.asarray(repeated).reshape(-1)
        assert single.size * 4 == _LOGICAL_BYTES, (
            f"test assumes an {_LOGICAL_BYTES}-byte output, got {single.size * 4}"
        )

        ndiff = int((single != repeated).sum())
        if ndiff:
            # The failure signature is a verbatim copy of the buffer's head
            # appearing in its tail; report the offset to make triage quick.
            n = len(single)
            shift = next(
                (s for s in range(1, n) if np.array_equal(repeated[s:], single[: n - s])),
                None,
            )
            detail = (
                f" second inference's tail is a copy of the head from element "
                f"{shift} ({shift * 4} bytes)"
                if shift is not None
                else ""
            )
            pytest.fail(
                f"repeat=2 output differs from a single inference in {ndiff} of {n} "
                f"elements.{detail} TIDL output storage is likely not padded for "
                "outPadCh/outChannelPitch -- see PadTIDLOutputStorage."
            )
