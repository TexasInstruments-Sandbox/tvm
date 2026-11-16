"""FuseInt8ResidualAdd fuses only adds whose operands both have the output's shape.

c7x_int8_residual_add_relu reads both operands over the output's element count, so
an operand that broadcasts would be read past its end. Swin adds its relative
position bias ([1, heads, N, N]) and its shift mask ([1, windows, 1, N, N]) to the
attention scores that way; those adds must stay on the generic path.

Relax IR-level tests plus one end-to-end check on c7x_host or c7x_dload.
"""

import numpy as np
import pytest
import torch

import tvm
from tvm import relax
from tvm.relax.transform.ti_residual_add import FuseInt8ResidualAdd

pytestmark = pytest.mark.quick


class _Add(torch.nn.Module):
    def forward(self, x, b):
        return x + b


class _AddParam(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(shape))

    def forward(self, x):
        return x + self.bias


def _quantize(model, args):
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    from tvm.relax.frontend.torch import C7xMMAQuantizer

    prepared = prepare_pt2e(torch.export.export(model.eval(), args).module(), C7xMMAQuantizer())
    prepared(*args)
    return convert_pt2e(prepared)


def _import(q, args):
    from tvm.relax.frontend.torch import from_exported_program

    mod = from_exported_program(torch.export.export(q, args), keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)
    weights = params.get("main", [])
    if weights:
        bind = dict(zip(mod["main"].params[len(args) :], weights))
        mod = relax.transform.BindParams("main", bind)(mod)
    return mod


def _fused_adds(mod):
    """Calls to the int8 residual-add PrimFunc after FuseInt8ResidualAdd."""
    mod = FuseInt8ResidualAdd()(mod)
    count = 0

    def visit(expr):
        nonlocal count
        if (
            isinstance(expr, relax.Call)
            and isinstance(expr.op, tvm.ir.Op)
            and expr.op.name == "relax.call_tir"
            and expr.args[0].name_hint.startswith("int8_residual_add")
        ):
            count += 1

    relax.analysis.post_order_visit(mod["main"].body, visit)
    return count


def _inputs(x_shape, b_shape):
    torch.manual_seed(0)
    return torch.randn(x_shape), torch.randn(b_shape)


def test_same_shape_add_fused():
    args = _inputs((8, 3, 4, 4), (8, 3, 4, 4))
    assert _fused_adds(_import(_quantize(_Add(), args), args)) == 1


@pytest.mark.parametrize(
    "x_shape, b_shape",
    [
        ((8, 3, 4, 4), (1, 3, 4, 4)),  # Swin's relative position bias
        ((1, 8, 3, 4, 4), (1, 8, 1, 4, 4)),  # Swin's shift mask
        ((1, 3, 4, 4), (8, 3, 4, 4)),  # the first operand broadcasts
    ],
)
def test_broadcast_add_not_fused(x_shape, b_shape):
    args = _inputs(x_shape, b_shape)
    assert _fused_adds(_import(_quantize(_Add(), args), args)) == 0


@pytest.mark.parametrize("b_shape, fused", [((8, 3, 4, 4), 1), ((1, 3, 4, 4), 0)])
def test_constant_operand(b_shape, fused):
    torch.manual_seed(0)
    model, args = _AddParam(b_shape), (torch.randn(8, 3, 4, 4),)
    assert _fused_adds(_import(_quantize(model, args), args)) == fused


def test_broadcast_add_on_dsp_matches_pytorch(dsp_mode):
    if dsp_mode not in ("c7x_host", "c7x_dload"):
        pytest.skip("requires c7x_host or c7x_dload")
    from dsp_utils import compile_and_run_dsp, get_target_string

    # The bias add of a Swin attention block, at Swin-T's first-stage sizes.
    args = _inputs((64, 3, 49, 49), (1, 3, 49, 49))
    q = _quantize(_Add(), args)
    with torch.no_grad():
        ref = q(*args).numpy()
    results = compile_and_run_dsp(
        mod=_import(q, args),
        input_data=[a.numpy() for a in args],
        target_string=get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1",
        execution_mode=dsp_mode,
    )
    dsp = results[f"{dsp_mode}_result"].reshape(ref.shape)
    (out_q,) = [
        n
        for n in q.graph.nodes
        if "quantize_per_tensor" in str(n.target) and "dequantize" not in str(n.target)
    ][-1:]
    # The generic path computes dq -> add -> q in float, as the reference does, so
    # they differ by at most a code where float rounding splits a tie.
    codes = np.round(np.abs(dsp - ref) / float(out_q.args[1]))
    assert codes.max() <= 1
