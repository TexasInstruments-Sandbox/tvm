"""Unit tests for C7xMMAQuantizer.

Pure-Python tests — no DSP hardware or TI toolchain required.
Run with: pytest tests/ti-dsp-runtime/pt2e-tests/ -m quick -v
"""

import warnings

import pytest
import torch
import torch.nn as nn
from pt2e_utils import quantize_pt2e  # noqa: E402

from tvm.relax.frontend.torch import C7xMMAQuantizer

# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1, groups=8)

    def forward(self, x):
        return self.conv(x)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x)


class MmModel(nn.Module):
    """Exposes aten.mm directly."""

    def forward(self, a, b):
        return torch.mm(a, b)


class AddmmModel(nn.Module):
    """Exposes aten.addmm (bias + matmul)."""

    def forward(self, bias, a, b):
        return torch.addmm(bias, a, b)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _call_function_targets(gm) -> list[str]:
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


def _has_quantize(targets: list[str]) -> bool:
    return any("quantize_per_tensor" in t for t in targets)


def _has_dequantize(targets: list[str]) -> bool:
    return any("dequantize_per_tensor" in t or "dequantize_per_channel" in t for t in targets)


# ---------------------------------------------------------------------------
# Q/DQ presence
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_conv_int8_sym_produces_qdq():
    q = quantize_pt2e(ConvModel(), (torch.randn(1, 3, 8, 8),), C7xMMAQuantizer("int8", True))
    targets = _call_function_targets(q)
    assert _has_quantize(targets), f"missing quantize_per_tensor in {targets}"
    assert _has_dequantize(targets), f"missing dequantize in {targets}"


@pytest.mark.quick
def test_conv_int8_affine_produces_qdq():
    q = quantize_pt2e(ConvModel(), (torch.randn(1, 3, 8, 8),), C7xMMAQuantizer("int8", False))
    targets = _call_function_targets(q)
    assert _has_quantize(targets)
    assert _has_dequantize(targets)


@pytest.mark.quick
def test_linear_int8_produces_qdq():
    q = quantize_pt2e(LinearModel(), (torch.randn(1, 16),), C7xMMAQuantizer("int8"))
    targets = _call_function_targets(q)
    assert _has_quantize(targets)
    assert _has_dequantize(targets)


@pytest.mark.quick
def test_depthwise_conv_int8_produces_qdq():
    # depthwise conv2d uses the same aten.conv2d.default op; groups handled by TVM passes
    q = quantize_pt2e(
        DepthwiseConvModel(), (torch.randn(1, 8, 8, 8),), C7xMMAQuantizer("int8")
    )
    targets = _call_function_targets(q)
    assert _has_quantize(targets)
    assert _has_dequantize(targets)


# ---------------------------------------------------------------------------
# Weight zero-point must be zero (MMALIB requirement)
# ---------------------------------------------------------------------------


def _get_per_channel_zp(gm, node) -> torch.Tensor:
    """Resolve the zero_point get_attr node to its tensor value."""
    zp_node = node.args[2]
    assert zp_node.op == "get_attr", f"unexpected zp node op: {zp_node.op}"
    return getattr(gm, zp_node.target)


@pytest.mark.quick
def test_conv_weight_zero_point_is_zero():
    q = quantize_pt2e(ConvModel(), (torch.randn(1, 3, 8, 8),), C7xMMAQuantizer("int8"))
    found = False
    for node in q.graph.nodes:
        if "dequantize_per_channel" in str(node.target):
            zp = _get_per_channel_zp(q, node)
            assert torch.all(zp == 0), f"weight zero_point not zero: {zp}"
            found = True
    assert found, "no dequantize_per_channel node found"


@pytest.mark.quick
def test_linear_weight_zero_point_is_zero():
    q = quantize_pt2e(LinearModel(), (torch.randn(1, 16),), C7xMMAQuantizer("int8"))
    found = False
    for node in q.graph.nodes:
        if "dequantize_per_channel" in str(node.target):
            zp = _get_per_channel_zp(q, node)
            assert torch.all(zp == 0), f"weight zero_point not zero: {zp}"
            found = True
    assert found, "no dequantize_per_channel node found"


# ---------------------------------------------------------------------------
# int16
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_int16_force_symmetric_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        quantizer = C7xMMAQuantizer(dtype="int16", symmetric_activations=False)
    assert quantizer.symmetric_activations is True
    assert len(w) == 1
    assert "symmetric" in str(w[0].message).lower()


@pytest.mark.quick
def test_int16_produces_int16_quantize_nodes():
    q = quantize_pt2e(ConvModel(), (torch.randn(1, 3, 8, 8),), C7xMMAQuantizer("int16"))
    int16_found = False
    for node in q.graph.nodes:
        if "quantize_per_tensor" in str(node.target):
            if node.args[-1] == torch.int16:
                int16_found = True
                break
    assert int16_found, "no int16 quantize_per_tensor node found"


# ---------------------------------------------------------------------------
# No re-annotation (composed quantizer pattern)
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_no_double_annotation():
    """A node annotated by a first pass must not be overwritten by a second pass."""
    model = ConvModel().eval()
    example = (torch.randn(1, 3, 8, 8),)
    exported = torch.export.export(model, example).module()

    q1 = C7xMMAQuantizer("int8")
    q2 = C7xMMAQuantizer("int8")

    q1.annotate(exported)
    first_annotations = {
        n.name: n.meta.get("quantization_annotation")
        for n in exported.graph.nodes
        if n.meta.get("quantization_annotation") is not None
    }

    # Second pass must be a no-op: existing annotation objects must be preserved by identity.
    q2.annotate(exported)
    second_annotations = {
        n.name: n.meta.get("quantization_annotation")
        for n in exported.graph.nodes
        if n.meta.get("quantization_annotation") is not None
    }

    assert first_annotations.keys() == second_annotations.keys()
    for name in first_annotations:
        assert first_annotations[name] is second_annotations[name], (
            f"annotation for {name} was replaced on second pass"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_invalid_dtype_raises():
    with pytest.raises(ValueError, match="dtype must be"):
        C7xMMAQuantizer(dtype="int4")


# ---------------------------------------------------------------------------
# mm / addmm: both inputs annotated with act_spec (no per-channel weight)
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_mm_both_inputs_use_act_spec():
    example = (torch.randn(4, 8), torch.randn(8, 4))
    model = MmModel().eval()
    exported = torch.export.export(model, example).module()
    quantizer = C7xMMAQuantizer("int8")
    quantizer.annotate(exported)

    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.mm.default:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None
            specs = list(ann.input_qspec_map.values())
            assert len(specs) == 2
            for spec in specs:
                assert spec.qscheme == torch.per_tensor_symmetric


@pytest.mark.quick
def test_addmm_bias_not_annotated():
    example = (torch.randn(4), torch.randn(4, 8), torch.randn(8, 4))
    model = AddmmModel().eval()
    exported = torch.export.export(model, example).module()
    quantizer = C7xMMAQuantizer("int8")
    quantizer.annotate(exported)

    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.addmm.default:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None
            # args[1] and args[2] annotated, args[0] (bias) is not
            assert node.args[0] not in ann.input_qspec_map, "bias should not be annotated"
            assert node.args[1] in ann.input_qspec_map
            assert node.args[2] in ann.input_qspec_map


# ---------------------------------------------------------------------------
# add.Tensor: residual/skip connections
# ---------------------------------------------------------------------------


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x) + x


@pytest.mark.quick
def test_add_tensor_both_inputs_annotated():
    """aten.add.Tensor (residual add): both inputs annotated with act_spec."""
    model = ResidualModel().eval()
    example = (torch.randn(1, 8, 8, 8),)
    exported = torch.export.export(model, example).module()
    quantizer = C7xMMAQuantizer("int8")
    quantizer.annotate(exported)

    add_nodes = [
        n
        for n in exported.graph.nodes
        if n.op == "call_function" and n.target is torch.ops.aten.add.Tensor
    ]
    assert len(add_nodes) == 1, f"expected 1 add node, found {len(add_nodes)}"
    ann = add_nodes[0].meta.get("quantization_annotation")
    assert ann is not None, "add.Tensor not annotated"
    assert add_nodes[0].args[0] in ann.input_qspec_map
    assert add_nodes[0].args[1] in ann.input_qspec_map
    assert ann.output_qspec is not None


@pytest.mark.quick
def test_add_tensor_produces_qdq():
    """Residual model: Q/DQ ops appear on both add inputs after convert_pt2e."""
    q = quantize_pt2e(ResidualModel(), (torch.randn(1, 8, 8, 8),), C7xMMAQuantizer("int8"))
    targets = _call_function_targets(q)
    assert _has_quantize(targets)
    assert _has_dequantize(targets)
