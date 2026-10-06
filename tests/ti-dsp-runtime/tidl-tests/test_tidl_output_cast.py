#!/usr/bin/env python
"""Regression tests for the TIDL output-dtype-mismatch cast insertion pass.

TIDL hardware may produce a different output dtype than Relax declares.
Example: TIDL argmax produces one byte per element, but Relax declares int32.
Or a quantized subgraph outputs int8 but the rest of the graph expects
float32.

``TIDLOffloadCompiler._fix_output_dtype_mismatches`` reads each subgraph's
``params_1.bin`` (via ``io_buf_desc``), compares TIDL's actual per-output
element type/scale/zero-point against the dtype the lowered TIR stub
declares, and -- on mismatch -- inserts dequant/cast math in ``@main`` at the
call site via ``_insert_output_casts_in_main``.

These tests pin down three bugs found by review of the first implementation:

1. ``OutputCastInserter.visit_call_`` matched subgraph calls with
   ``isinstance(call.op, relax.GlobalVar)``. By the time this pass runs,
   ``_TIDLCallReplacer`` (part of ``lower_tidl``) has already rewritten every
   subgraph call into ``relax.call_tir(tir_gv, args, out_sinfo=...)``, whose
   ``.op`` is the ``relax.call_tir`` builtin Op -- never a ``GlobalVar``. The
   check never matched, so no cast was ever inserted.
2. The dequant formula was ``data * (1/scale_inv) + zp``. TIDL's own
   convention (``tidl_import_quantize.cpp:1490``, "Xfloat = (Xfixed -
   zp)/scale") and the runtime kernel ``copy_zfscale_linear``
   (``arm-tidl/rt/src/tidl_rt_ovx_datamove.cpp``) both dequantize as
   ``(data - zp) / scale``, with ``outTensorScale`` read as-is (no
   inversion).
3. Per-output TIDL fields were compared against Relax output dtypes by raw
   descriptor position, ignoring that TIDL can reorder output slots
   (``outDataName``, see ``io_buf_desc.TIDLIOBufDesc.output_slot_map``).
"""

import struct
import tempfile

import pytest
import tvm
from tvm import relax
from tvm.relax.analysis import post_order_visit
from tvm.relax.backend.tidl import TIDLOffloadCompiler
from tvm.relax.backend.tidl.io_buf_desc import TIDLIOBufDesc
from tvm.relax.backend.tidl.tidl import _collect_tidl_subgraph_info

pytestmark = [pytest.mark.quick, pytest.mark.core]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tidl_subgraph_stub(name, out_dtypes, sg_id):
    """Build a minimal TIR PrimFunc shaped like a lowered TIDL subgraph stub.

    Params are named ``input0`` and ``output0..output{N-1}``, matching the
    naming ``_fix_output_dtype_mismatches`` keys off of. The body is a no-op:
    this pass never executes the stub, only reads its signature.
    """
    a = tvm.tir.Var("input0", "handle")
    in_buf = tvm.tir.decl_buffer((4,), "int8", name="input0_buf")
    params = [a]
    buffer_map = {a: in_buf}
    for i, dt in enumerate(out_dtypes):
        b = tvm.tir.Var(f"output{i}", "handle")
        out_buf = tvm.tir.decl_buffer((4,), dt, name=f"output{i}_buf")
        params.append(b)
        buffer_map[b] = out_buf
    func = tvm.tir.PrimFunc(params, tvm.tir.Evaluate(0), buffer_map=buffer_map)
    func = func.with_attr("global_symbol", name)
    if sg_id is not None:
        func = func.with_attr("tidl_sg_id", sg_id)
    return func


def _build_module(sg_name, out_dtypes, sg_id: int | None = 0):
    """Build a module with one TIDL subgraph TIR stub and a @main that calls
    it via call_tir, in a single BlockBuilder session so the callee GlobalVar
    carries the stub's struct info (required for call_tir normalization).
    """
    stub = _make_tidl_subgraph_stub(sg_name, out_dtypes, sg_id)
    if len(out_dtypes) == 1:
        out_sinfo = relax.TensorStructInfo((4,), out_dtypes[0])
    else:
        out_sinfo = [relax.TensorStructInfo((4,), dt) for dt in out_dtypes]

    bb = relax.BlockBuilder()
    stub_gv = bb.add_func(stub, sg_name)
    x = relax.Var("x", relax.TensorStructInfo((4,), "int8"))
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(relax.call_tir(stub_gv, (x,), out_sinfo=out_sinfo))
        bb.emit_func_output(out)
    return bb.get()


def _collect_calls(expr):
    calls = []

    def visit(e):
        if isinstance(e, relax.Call):
            calls.append(e)

    post_order_visit(expr, visit)
    return calls


def _binding_map(func):
    """Map every bound Var -> its value expression, across all blocks.

    The mutator emits each sub-expression as its own binding (ANF form), so a
    Call's args are often Vars referencing an earlier binding rather than the
    nested Call directly; resolve through this map to inspect the real
    expression tree.
    """
    bindings = {}
    for block in func.body.blocks:
        for binding in block.bindings:
            bindings[binding.var] = binding.value
    return bindings


def _resolve(expr, bindings):
    while isinstance(expr, relax.Var) and expr in bindings:
        expr = bindings[expr]
    return expr


def _const_value(expr):
    assert isinstance(expr, relax.Constant), f"expected a Constant, got {type(expr)}"
    return float(expr.data.numpy())


# ---------------------------------------------------------------------------
# Bug 1 + 2: the mutator must fire, and must implement (data - zp) / scale
# ---------------------------------------------------------------------------


class TestOutputCastInsertion:
    def test_cast_inserted_in_main(self):
        """The dead-mutator regression: the call site must actually change."""
        mod = _build_module("tidl_subgraph_0", ["int8"])
        mismatches = {"tidl_subgraph_0": [(0, "float32", "int8", 2.0, 5)]}

        compiler = TIDLOffloadCompiler()
        new_main = compiler._insert_output_casts_in_main(mod, mismatches)["main"]

        op_names = {
            call.op.name for call in _collect_calls(new_main.body) if isinstance(call.op, tvm.ir.Op)
        }
        assert "relax.subtract" in op_names, f"no subtract inserted, ops={op_names}"
        assert "relax.divide" in op_names, f"no divide inserted, ops={op_names}"

    def test_dequant_math_correct(self):
        """The wrong-formula regression: must be (data - zp) / scale, in that
        order, with scale read as-is (not inverted)."""
        mod = _build_module("tidl_subgraph_0", ["int8"])
        scale, zp = 2.0, 5
        mismatches = {"tidl_subgraph_0": [(0, "float32", "int8", scale, zp)]}

        compiler = TIDLOffloadCompiler()
        new_main = compiler._insert_output_casts_in_main(mod, mismatches)["main"]

        bindings = _binding_map(new_main)
        divides = [
            c for c in _collect_calls(new_main.body) if c.op == tvm.ir.Op.get("relax.divide")
        ]
        assert len(divides) == 1
        divide_call = divides[0]
        dividend, divisor = (_resolve(a, bindings) for a in divide_call.args)
        assert isinstance(dividend, relax.Call) and dividend.op == tvm.ir.Op.get(
            "relax.subtract"
        ), "zero point must be subtracted before dividing by scale"
        shift = _resolve(dividend.args[1], bindings)
        assert _const_value(divisor) == pytest.approx(scale)
        assert _const_value(shift) == pytest.approx(float(zp))


# ---------------------------------------------------------------------------
# Bug 3: multi-output slot reordering
# ---------------------------------------------------------------------------


class TestOutputSlotRemap:
    def test_output_slot_map_direct(self):
        """TIDLIOBufDesc.output_slot_map must honor outDataName, not raw order."""
        name0 = b"tidl_sg0_o1".ljust(16, b"\0")
        name1 = b"tidl_sg0_o0".ljust(16, b"\0")
        blob = struct.pack("<i", 2) + name0 + name1
        fields = {
            "numOutputBuf": (0, 1, 4, "int32_t", None),
            "outDataName": (4, 2 * 16, 1, "int8_t", 16),
        }
        desc = TIDLIOBufDesc(fields, blob)
        assert desc.output_slot_map(2) == [1, 0]

    def test_multi_output_mismatch_uses_remapped_slot(self, monkeypatch):
        """End-to-end: descriptor outputs are in reversed slot order; the
        mismatch detected (and the scale/zp used for its cast) must follow
        TVM output slot, not raw descriptor position.

        Descriptor index 0 (float32, scale=4.0, zp=7) is slot 1.
        Descriptor index 1 (int8, scale=3.0, zp=10) is slot 0.
        TVM declares slot 0 as float32 and slot 1 as int8 -- the mirror
        image of the descriptor -- so both slots are real mismatches only
        once correctly remapped. A raw positional (unfixed) comparison would
        compare desc[0]=float32 against tvm slot0=float32 (no mismatch) and
        desc[1]=int8 against tvm slot1=int8 (no mismatch) -- i.e. it would
        silently miss both.
        """

        class _FakeDesc:
            def scalar(self, name):
                assert name == "numOutputBuf"
                return 2

            def ints(self, name, count=None):
                return {"outElementType": [6, 1], "outZeroPoint": [7, 10]}[name]

            def floats(self, name, count=None):
                assert name == "outTensorScale"
                return [4.0, 3.0]

            def output_slot_map(self, num_out=None):
                return [1, 0]

        monkeypatch.setattr(
            "tvm.relax.backend.tidl.io_buf_desc.parse_io_buf_desc",
            lambda *a, **k: _FakeDesc(),
        )

        mod = _build_module("tidl_subgraph_0", ["float32", "int8"])

        with tempfile.NamedTemporaryFile() as f:
            artifacts = {"tidl_subgraph_0": {"sg_id": 0, "io_bin": f.name}}
            compiler = TIDLOffloadCompiler()
            new_mod = compiler._fix_output_dtype_mismatches(mod, artifacts)

        new_main = new_mod["main"]
        bindings = _binding_map(new_main)
        divides = [
            c for c in _collect_calls(new_main.body) if c.op == tvm.ir.Op.get("relax.divide")
        ]
        assert len(divides) == 2, "expected one cast per output (both are real mismatches)"

        pairs = set()
        for divide_call in divides:
            dividend, divisor = (_resolve(a, bindings) for a in divide_call.args)
            shift = _resolve(dividend.args[1], bindings)
            pairs.add((_const_value(shift), _const_value(divisor)))

        # slot 0 -> descriptor index 1 (zp=10, scale=3.0)
        # slot 1 -> descriptor index 0 (zp=7, scale=4.0)
        assert pairs == {(10.0, 3.0), (7.0, 4.0)}, (
            f"got {pairs}, expected the remapped-by-slot (zp, scale) pairs -- "
            "a positional (unremapped) comparison would detect no mismatches at all"
        )


# ---------------------------------------------------------------------------
# Narrow TIDL output (argmax/argmin): storage must match what TIDL writes
# ---------------------------------------------------------------------------


class TestNarrowIntegerOutput:
    """TIDL writes one uint8 per argmax index while Relax declares int32.

    Casting a tensor still declared (and sized) as int32 cannot unpack bytes
    TIDL packed four to the word, so the stub buffer and the call_tir result
    must be re-declared uint8 before the widening cast.
    """

    @staticmethod
    def _fix(monkeypatch):
        class _FakeDesc:
            def scalar(self, name):
                return 1

            def ints(self, name, count=None):
                return {"outElementType": [0], "outZeroPoint": [3]}[name]

            def floats(self, name, count=None):
                return [2.0]

            def output_slot_map(self, num_out=None):
                return [0]

        monkeypatch.setattr(
            "tvm.relax.backend.tidl.io_buf_desc.parse_io_buf_desc",
            lambda *a, **k: _FakeDesc(),
        )
        mod = _build_module("tidl_subgraph_0", ["int32"])
        with tempfile.NamedTemporaryFile() as f:
            artifacts = {"tidl_subgraph_0": {"sg_id": 0, "io_bin": f.name}}
            return TIDLOffloadCompiler()._fix_output_dtype_mismatches(mod, artifacts)

    def test_stub_output_buffer_is_retyped(self, monkeypatch):
        new_mod = self._fix(monkeypatch)
        stub = new_mod["tidl_subgraph_0"]
        out_param = [p for p in stub.params if p.name.startswith("output")][0]
        assert str(stub.buffer_map[out_param].dtype) == "uint8"
        assert int(stub.attrs["tidl_sg_id"]) == 0

    def test_call_tir_declares_tidl_dtype_then_widens(self, monkeypatch):
        new_mod = self._fix(monkeypatch)
        new_main = new_mod["main"]
        calls = _collect_calls(new_main.body)
        tir_calls = [c for c in calls if c.op == tvm.ir.Op.get("relax.call_tir")]
        assert len(tir_calls) == 1
        assert tir_calls[0].struct_info.dtype == "uint8"
        assert new_main.ret_struct_info.dtype == "int32"
        astypes = [c for c in calls if c.op == tvm.ir.Op.get("relax.astype")]
        assert len(astypes) == 1

    def test_survives_call_tir_rewrite(self, monkeypatch):
        """The callee GlobalVar's cached signature must follow the retyped stub.

        Otherwise CallTIRRewrite (run by relax.build) type-checks the uint8
        output allocation against the stale int32 signature and aborts.
        """
        new_mod = self._fix(monkeypatch)
        gv = new_mod.get_global_var("tidl_subgraph_0")
        assert "uint8" in str(gv.struct_info)
        relax.transform.CallTIRRewrite()(new_mod)

    def test_integer_widening_skips_dequant(self, monkeypatch):
        """scale=2.0, zp=3 must not turn index values into (idx - 3) / 2."""
        new_mod = self._fix(monkeypatch)
        ops = {c.op for c in _collect_calls(new_mod["main"].body)}
        assert tvm.ir.Op.get("relax.divide") not in ops
        assert tvm.ir.Op.get("relax.subtract") not in ops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMissingSgId:
    """A TIDL stub without ``tidl_sg_id`` must raise, not fall back to a
    positional index: sg_id selects the artifact (io_bin, dtypes) and the
    bridge symbol names, so a guessed value binds the wrong subgraph."""

    def test_fix_output_dtype_mismatches_raises(self):
        mod = _build_module("tidl_subgraph_0", ["int8"], sg_id=None)
        with pytest.raises(ValueError, match="tidl_subgraph_0.*tidl_sg_id"):
            TIDLOffloadCompiler()._fix_output_dtype_mismatches(mod, {})

    def test_collect_subgraph_info_raises(self):
        mod = _build_module("tidl_subgraph_0", ["int8"], sg_id=None)
        with pytest.raises(ValueError, match="tidl_subgraph_0.*tidl_sg_id"):
            _collect_tidl_subgraph_info(mod)

    def test_present_sg_id_is_used(self):
        """Accept-side sibling: a non-zero sg_id is read back, not replaced by an index."""
        mod = _build_module("tidl_subgraph_3", ["int8"], sg_id=3)
        assert [sg["sg_id"] for sg in _collect_tidl_subgraph_info(mod)] == [3]
