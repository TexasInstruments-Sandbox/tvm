"""DropSafeSoftmaxGuard removes the safe-softmax guard only where it can't fire.

torch.export decomposes scaled_dot_product_attention through
aten._safe_softmax, which wraps the softmax in
where(all(scores == -inf), 0, softmax(scores)). Without a mask the scores are
the Q K^T matmul and the guard is dead; with an attention mask or is_causal
the scores include the mask and the guard must stay.

Relax IR-level tests plus one C7x codegen check; no DSP needed.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import tvm
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import compile_for_dsp  # noqa: E402

pytestmark = pytest.mark.quick

_GUARD_OPS = ("relax.equal", "relax.logical_not", "relax.max", "relax.where")


class _SDPA(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, q, k, v, *mask):
        if self.mode == "mask":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[0])
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.mode == "causal")


def _import_sdpa(mode):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    args = (q, k, v) + ((torch.zeros(1, 2, 5, 5),) if mode == "mask" else ())
    return from_exported_program(torch.export.export(_SDPA(mode), args))


def _op_names(mod):
    names = []

    def visit(expr):
        if isinstance(expr, tvm.relax.Call) and isinstance(expr.op, tvm.ir.Op):
            names.append(expr.op.name)

    tvm.relax.analysis.post_order_visit(mod["main"].body, visit)
    return names


def test_import_has_guard():
    """The decomposition this pass targets: if torch stops emitting the guard,
    the pass has nothing to match and this test says so."""
    names = _op_names(_import_sdpa("none"))
    for op in _GUARD_OPS:
        assert op in names, op


def test_unmasked_guard_dropped():
    mod = tvm.relax.transform.DropSafeSoftmaxGuard()(_import_sdpa("none"))
    names = _op_names(mod)
    for op in _GUARD_OPS:
        assert op not in names, op
    assert names.count("relax.nn.softmax") == 1
    assert names.count("relax.matmul") == 2


@pytest.mark.parametrize("mode", ["mask", "causal"])
def test_masked_guard_kept(mode):
    before = _import_sdpa(mode)
    after = tvm.relax.transform.DropSafeSoftmaxGuard()(before)
    tvm.ir.assert_structural_equal(after, before)


def test_c7x_pipeline_drops_guard():
    """The c7x pipeline runs the pass: no equal/not/cast/max function remains."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = compile_for_dsp(
            _import_sdpa("none"),
            "c_static_lib -mcpu=c7x -use-cpp-api=1",
            output_dir=Path(tmp),
        )
        src = "\n".join(p.read_text() for p in sorted(gen.glob("lib*.c")))
    assert "softmax" in src
    assert "logical_not" not in src
    assert "_equal" not in src
