"""Unit tests for C7xMMAQuantizer.

Pure-Python tests — no DSP hardware or TI toolchain required.
Run with: pytest tests/ti-dsp-runtime/pt2e-tests/ -m quick -v
"""

import warnings
from typing import Any

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
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_conv_histogram_observer_produces_qdq(dtype):
    q = quantize_pt2e(
        ConvModel(),
        (torch.randn(1, 3, 8, 8),),
        C7xMMAQuantizer(dtype, activation_observer="histogram"),
    )
    targets = _call_function_targets(q)
    assert _has_quantize(targets)
    assert _has_dequantize(targets)


class ConvReLU6Model(nn.Module):
    """conv -> ReLU6 -> conv, optionally with the first conv's output also added back."""

    def __init__(self, inplace=True, branch=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.next = nn.Conv2d(8, 8, 1)
        self.branch = branch

    def forward(self, x):
        y = self.conv(x)
        if self.branch:
            return self.next(self.relu6(y.clone())) + y
        return self.next(self.relu6(y))


def _output_scale(gm, target_prefix: str) -> float:
    """The scale of the quantize node after the first op whose target starts with prefix."""
    node = next(
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and str(n.target).startswith(target_prefix)
    )
    (q,) = [u for u in node.users if "quantize_per_tensor" in str(u.target)]
    return q.args[1]


# Large enough that the first conv's output reaches well beyond 6.
_RELU6_INPUT = (torch.randn(1, 3, 8, 8) * 50,)


@pytest.mark.quick
@pytest.mark.parametrize(("dtype", "half_range"), [("int8", 127.5), ("int16", 32767.5)])
@pytest.mark.parametrize("inplace", [True, False])
def test_conv_feeding_relu6_gets_relu6_range(dtype, half_range, inplace):
    q = quantize_pt2e(ConvReLU6Model(inplace), _RELU6_INPUT, C7xMMAQuantizer(dtype))
    conv_scale = _output_scale(q, "aten.conv2d")
    assert conv_scale == pytest.approx(6.0 / half_range)
    # ReLU6 saturates here, so its output's min/max observer picks the same
    # scale and the clamp needs no requantize.
    assert _output_scale(q, "aten.hardtanh") == pytest.approx(conv_scale)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("model", "symmetric"),
    [(ConvReLU6Model(branch=True), True), (ConvReLU6Model(), False)],
    ids=["conv-output-has-other-users", "asymmetric"],
)
def test_conv_relu6_range_kept_calibrated(model, symmetric):
    q = quantize_pt2e(model, _RELU6_INPUT, C7xMMAQuantizer("int8", symmetric))
    # A calibrated range covers the conv's whole output, far beyond 6.
    assert _output_scale(q, "aten.conv2d") > 2 * 6.0 / 127.5


class ConvSiLUModel(nn.Module):
    """conv -> SiLU -> conv, optionally with the first conv's output also added back."""

    def __init__(self, branch=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.act = nn.SiLU()
        self.next = nn.Conv2d(8, 8, 1)
        self.branch = branch

    def forward(self, x):
        y = self.conv(x)
        if self.branch:
            return self.next(self.act(y)) + y
        return self.next(self.act(y))


class ConvReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.next = nn.Conv2d(8, 8, 1)

    def forward(self, x):
        return self.next(torch.relu(self.conv(x)))


def _quantize_scales(model, quantizer) -> tuple[list[float], list[str]]:
    """Every activation quantize node's scale in graph order, and the graph's ops."""
    torch.manual_seed(0)
    q = quantize_pt2e(model, (torch.randn(1, 3, 64, 64),), quantizer)
    targets = _call_function_targets(q)
    scales = [
        n.args[1]
        for n in q.graph.nodes
        if n.op == "call_function"
        and "quantize_per_tensor" in str(n.target)
        and "dequantize" not in str(n.target)
    ]
    return scales, targets


@pytest.mark.quick
def test_tensors_around_silu_get_percentile_range():
    model = ConvSiLUModel()
    (clip, clip_ops) = _quantize_scales(model, C7xMMAQuantizer("int8"))
    (plain, plain_ops) = _quantize_scales(
        model, C7xMMAQuantizer("int8", smooth_activation_percentile=None)
    )
    # input, conv output (the SiLU's input), SiLU output, next conv output
    assert len(clip) == len(plain) == 4
    assert clip_ops == plain_ops, "the graph's structure must not change"
    assert clip[0] == plain[0]
    assert clip[1] < plain[1] and clip[2] < plain[2], "percentile range is inside min/max"
    assert clip[3] == plain[3]


@pytest.mark.quick
@pytest.mark.parametrize(
    ("model", "dtype"),
    [(ConvReLUModel(), "int8"), (ConvSiLUModel(), "int16")],
    ids=["relu-network", "int16"],
)
def test_percentile_range_leaves_other_tensors_alone(model, dtype):
    clip = _quantize_scales(model, C7xMMAQuantizer(dtype))
    plain = _quantize_scales(model, C7xMMAQuantizer(dtype, smooth_activation_percentile=None))
    assert clip == plain


@pytest.mark.quick
def test_conv_output_with_other_users_keeps_its_range():
    clip, _ = _quantize_scales(ConvSiLUModel(branch=True), C7xMMAQuantizer("int8"))
    plain, _ = _quantize_scales(
        ConvSiLUModel(branch=True), C7xMMAQuantizer("int8", smooth_activation_percentile=None)
    )
    # The conv output also feeds the add, so it isn't only a SiLU input.
    assert clip[1] == plain[1]
    assert clip[2] < plain[2], "the SiLU output still gets a percentile range"


@pytest.mark.quick
@pytest.mark.parametrize("percentile", [0, 50, -1])
def test_invalid_smooth_activation_percentile_is_rejected(percentile):
    with pytest.raises(ValueError, match="smooth_activation_percentile"):
        C7xMMAQuantizer("int8", smooth_activation_percentile=percentile)


@pytest.mark.quick
def test_invalid_depthwise_block_percentile_is_rejected():
    with pytest.raises(ValueError, match="depthwise_block_percentile"):
        C7xMMAQuantizer("int8", depthwise_block_percentile=50)


@pytest.mark.quick
def test_invalid_depthwise_input_headroom_is_rejected():
    with pytest.raises(ValueError, match="depthwise_input_headroom"):
        C7xMMAQuantizer("int8", depthwise_input_headroom=0.5)


class StemIntoDepthwise(nn.Module):
    """conv -> activation -> depthwise -> project, the stem's 16 channels 100x apart.

    second_reader: the conv output also feeds an add, so the chain isn't the conv's
    only use.
    """

    def __init__(self, act, second_reader=False):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        with torch.no_grad():
            self.stem.weight[:8] *= 0.01  # half the channels 100x smaller
            self.stem.bias.zero_()
        self.act = act
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.project = nn.Conv2d(16, 8, 1)
        self.second_reader = second_reader

    def forward(self, x):
        y = self.stem(x)
        out = self.project(self.dw(self.act(y)))
        return out + y[:, :8] if self.second_reader else out


def _activation_quantizes(model, quantizer):
    """(per-channel quantize nodes' scale tensors, per-tensor scales) after convert."""
    torch.manual_seed(0)
    q = quantize_pt2e(model, (torch.randn(1, 3, 16, 16),), quantizer)
    per_channel, per_tensor = [], []
    for n in q.graph.nodes:
        target = str(n.target)
        if n.op != "call_function" or "dequantize" in target:
            continue
        if "quantize_per_channel" in target:
            per_channel.append(getattr(q, n.args[1].target))
        elif "quantize_per_tensor" in target:
            per_tensor.append(float(n.args[1]))
    return per_channel, per_tensor


@pytest.mark.quick
@pytest.mark.parametrize(
    "act",
    [nn.Hardswish(inplace=True), nn.Hardswish(), nn.SiLU(inplace=True), nn.ReLU()],
    ids=["hardswish_", "hardswish", "silu_", "relu"],
)
def test_depthwise_input_chain_gets_per_channel_scales(act):
    per_channel, _ = _activation_quantizes(StemIntoDepthwise(act), C7xMMAQuantizer("int8"))
    # The conv output and the depthwise input (for ReLU, the same tensor on both sides).
    assert len(per_channel) == 2
    _, plain = _activation_quantizes(
        StemIntoDepthwise(act), C7xMMAQuantizer("int8", depthwise_input_headroom=None)
    )
    cap = max(plain)
    for scales in per_channel:
        assert scales.numel() == 16
        assert float(scales.max()) <= cap * (1 + 1e-6), "never coarser than per tensor"
        assert float(scales.max() / scales.min()) > 10, "the small channels get a finer grid"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("model", "quantizer"),
    [
        (ConvReLUModel(), C7xMMAQuantizer("int8")),
        (StemIntoDepthwise(nn.ReLU6()), C7xMMAQuantizer("int8")),
        (StemIntoDepthwise(nn.Hardswish()), C7xMMAQuantizer("int16")),
        (StemIntoDepthwise(nn.Hardswish()), C7xMMAQuantizer("int8", symmetric_activations=False)),
        (StemIntoDepthwise(nn.Hardswish()), C7xMMAQuantizer("int8", depthwise_input_headroom=None)),
    ],
    ids=["no-depthwise", "relu6", "int16", "asymmetric", "off"],
)
def test_depthwise_input_per_channel_left_out(model, quantizer):
    per_channel, _ = _activation_quantizes(model, quantizer)
    assert per_channel == []


class ResidualStem(nn.Module):
    """MobileNetV3-Large's stem: conv -> hardswish, read by a depthwise block and by the
    block's residual add."""

    def __init__(self, inplace):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        with torch.no_grad():
            self.stem.weight[:8] *= 0.01
        self.act = nn.Hardswish(inplace=inplace)
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.project = nn.Conv2d(16, 16, 1)

    def forward(self, x):
        y = self.act(self.stem(x))
        return self.project(self.dw(y)) + y


@pytest.mark.quick
@pytest.mark.parametrize("inplace", [True, False], ids=["hardswish_", "hardswish"])
def test_other_readers_of_a_chain_keep_per_tensor_scales(inplace):
    """The depthwise conv reads the activation per channel; the add, per tensor."""
    torch.manual_seed(0)
    q = quantize_pt2e(ResidualStem(inplace), (torch.randn(1, 3, 16, 16),), C7xMMAQuantizer("int8"))
    nodes = list(q.graph.nodes)
    dw = next(
        n for n in nodes
        if n.target == torch.ops.aten.conv2d.default and len(n.args) > 6 and n.args[6] == 16
    )
    add = next(n for n in nodes if "add" in str(n.target))
    assert "dequantize_per_channel" in str(dw.args[0].target)
    assert all("dequantize_per_tensor" in str(a.target) for a in add.args[:2])

class DepthwiseReLUBlock(nn.Module):
    """MobileNetV3's ReLU block: expand -> ReLU -> depthwise -> ReLU -> project."""

    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(3, 16, 1)
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.project = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        return self.project(torch.relu(self.dw(torch.relu(self.expand(x)))))


class SqueezeExciteBlock(nn.Module):
    """conv -> act -> x * gate(pool(x)) -> project; EfficientNet's with SiLU/sigmoid."""

    def __init__(self, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.act = act()
        self.fc1 = nn.Conv2d(16, 4, 1)
        self.fc2 = nn.Conv2d(4, 16, 1)
        self.project = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        y = self.act(self.conv(x))
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(y.mean((2, 3), keepdim=True)))))
        return self.project(y * gate)


def _scales_with_outlier(model, quantizer) -> list[float]:
    """Activation scales in graph order, calibrated on an input with one outlier."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    x[0, :, 32, 32] = 40.0  # a few large values a 0.01% percentile clips
    q = quantize_pt2e(model, (x,), quantizer)
    return [
        float(n.args[1])
        for n in q.graph.nodes
        if n.op == "call_function"
        and "quantize_per_tensor" in str(n.target)
        and "dequantize" not in str(n.target)
    ]


def _changed(model, dtype="int8", option="depthwise_block_percentile"):
    """Positions whose scale the rule `option` turns off changes, and whether all shrank."""
    off: dict[str, Any] = {option: None}
    clip = _scales_with_outlier(model, C7xMMAQuantizer(dtype))
    plain = _scales_with_outlier(model, C7xMMAQuantizer(dtype, **off))
    assert len(clip) == len(plain)
    changed = [i for i, (a, b) in enumerate(zip(clip, plain)) if a != b]
    return changed, all(clip[i] < plain[i] for i in changed), len(clip)


@pytest.mark.quick
def test_relu_tensors_next_to_a_depthwise_conv_get_percentile_range():
    changed, shrank, n = _changed(DepthwiseReLUBlock())
    # Both convs read by a ReLU next to the depthwise conv (and the ReLU outputs that
    # share their scales) clip; the model input and the projection's output don't.
    assert changed and shrank
    assert 0 not in changed and n - 1 not in changed


@pytest.mark.quick
def test_squeeze_excite_output_after_silu_gets_percentile_range():
    changed, shrank, _ = _changed(SqueezeExciteBlock())
    assert len(changed) == 1 and shrank, "only the gated SiLU output, read by the projection"


@pytest.mark.quick
@pytest.mark.parametrize(
    ("model", "dtype"),
    [
        (ConvReLUModel(), "int8"),
        (SqueezeExciteBlock(act=nn.ReLU), "int8"),
        (DepthwiseReLUBlock(), "int16"),
    ],
    ids=["relu-without-depthwise", "relu-squeeze-excite", "int16"],
)
def test_depthwise_block_rule_leaves_other_tensors_alone(model, dtype):
    changed, _, _ = _changed(model, dtype)
    assert changed == []


class DenseLayers(nn.Module):
    """DenseNet's stem and two dense layers, each a BatchNorm over a concat.

    concat_conv: the concat is read by a conv (with a BatchNorm after it, which
    folds), as in Inception, instead.
    """

    def __init__(self, concat_conv=False):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.norm1, self.conv1 = nn.BatchNorm2d(8), nn.Conv2d(8, 4, 3, padding=1)
        self.norm2, self.conv2 = nn.BatchNorm2d(12), nn.Conv2d(12, 4, 3, padding=1)
        self.conv_norm = nn.BatchNorm2d(4)
        self.concat_conv = concat_conv

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.stem(x)), 2)
        if self.concat_conv:
            return self.conv_norm(self.conv2(torch.cat([x, self.conv1(x)], 1)))
        new = self.conv1(torch.relu(self.norm1(torch.cat([x], 1))))
        return self.conv2(torch.relu(self.norm2(torch.cat([x, new], 1))))


@pytest.mark.quick
def test_concat_inputs_before_batch_norm_get_percentile_range():
    changed, shrank, _ = _changed(DenseLayers(), option="concat_batch_norm_percentile")
    # Quantize nodes in graph order: input; stem conv, its ReLU and max pool (one
    # shared scale); BatchNorm, ReLU; the first layer's new features; BatchNorm, ReLU;
    # output. The stem and the new features clip, nothing else.
    assert changed == [1, 2, 3, 6] and shrank


@pytest.mark.quick
@pytest.mark.parametrize(
    ("model", "dtype"),
    [(DenseLayers(concat_conv=True), "int8"), (DenseLayers(), "int16")],
    ids=["concat-into-conv", "int16"],
)
def test_concat_batch_norm_rule_leaves_other_tensors_alone(model, dtype):
    changed, _, _ = _changed(model, dtype, option="concat_batch_norm_percentile")
    assert changed == []


@pytest.mark.quick
def test_invalid_concat_batch_norm_percentile_is_rejected():
    with pytest.raises(ValueError, match="concat_batch_norm_percentile"):
        C7xMMAQuantizer("int8", concat_batch_norm_percentile=0)


class TinyViT(nn.Module):
    """A class token and position embedding, two pre-norm blocks, a LayerNorm head.

    attention_bias: the first block's branch adds a learned bias inside itself, as
    Swin's attention does.
    """

    def __init__(self, attention_bias=False):
        super().__init__()
        self.embed = nn.Linear(8, 16)
        self.cls = nn.Parameter(torch.randn(1, 1, 16) * 0.01)
        self.pos = nn.Parameter(torch.randn(1, 5, 16) * 0.02)
        self.ln1, self.fc1 = nn.LayerNorm(16), nn.Linear(16, 16)
        self.bias = nn.Parameter(torch.randn(5, 16)) if attention_bias else None
        self.out1 = nn.Linear(16, 16)
        self.ln2, self.fc2 = nn.LayerNorm(16), nn.Linear(16, 16)
        self.norm, self.head = nn.LayerNorm(16), nn.Linear(16, 4)

    def forward(self, x):
        t = torch.cat([self.cls.expand(x.shape[0], -1, -1), self.embed(x)], dim=1) + self.pos
        y = self.fc1(self.ln1(t))
        if self.bias is not None:
            y = y + self.bias
        t = t + self.out1(y)
        t = t + self.fc2(self.ln2(t))
        return self.head(self.norm(t)[:, 0])


class TinyConvNeXt(nn.Module):
    """Two ConvNeXt blocks: depthwise conv -> LayerNorm -> linear, plus the input."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.dw = nn.ModuleList(nn.Conv2d(8, 8, 3, padding=1, groups=8) for _ in range(2))
        self.ln = nn.ModuleList(nn.LayerNorm(8) for _ in range(2))
        self.fc = nn.ModuleList(nn.Linear(8, 8) for _ in range(2))

    def forward(self, x):
        x = self.stem(x)
        for dw, ln, fc in zip(self.dw, self.ln, self.fc):
            x = x + fc(ln(dw(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x


def _quantized(model, example, quantizer):
    torch.manual_seed(0)
    return quantize_pt2e(model, example, quantizer)


def _nodes(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _is_dq(node) -> bool:
    return "dequantize" in str(getattr(node, "target", ""))


def _requantized(node) -> bool:
    return any("quantize_per_tensor" in str(u.target) and not _is_dq(u) for u in node.users)


_ADD = torch.ops.aten.add.Tensor
_LAYER_NORM = torch.ops.aten.layer_norm.default
_VIT_INPUT = (torch.randn(1, 4, 8),)


@pytest.mark.quick
def test_layernorm_residual_stream_stays_float():
    q = _quantized(TinyViT(), _VIT_INPUT, C7xMMAQuantizer("int8"))
    adds = _nodes(q, _ADD)
    # The position embedding add and the two residual adds.
    assert len(adds) == 3
    assert not any(_requantized(a) for a in adds)
    assert not any(_is_dq(ln.args[0]) for ln in _nodes(q, _LAYER_NORM))
    # The adds read the stream as float, and the linears still read int8.
    assert not _is_dq(adds[1].args[0]) and not _is_dq(adds[2].args[0])
    assert all(_is_dq(fc.args[0]) for fc in _nodes(q, torch.ops.aten.linear.default))


@pytest.mark.quick
@pytest.mark.parametrize(
    "quantizer",
    [C7xMMAQuantizer("int8", float_layernorm_residuals=False), C7xMMAQuantizer("int16")],
    ids=["off", "int16"],
)
def test_layernorm_residual_stream_quantized(quantizer):
    q = _quantized(TinyViT(), _VIT_INPUT, quantizer)
    assert all(_requantized(a) for a in _nodes(q, _ADD))
    assert all(_is_dq(ln.args[0]) for ln in _nodes(q, _LAYER_NORM))


@pytest.mark.quick
def test_add_inside_a_layernorm_branch_stays_quantized():
    q = _quantized(TinyViT(attention_bias=True), _VIT_INPUT, C7xMMAQuantizer("int8"))
    embed, bias, first, second = _nodes(q, _ADD)
    assert _requantized(bias), "the bias add isn't a residual add"
    assert not any(_requantized(a) for a in (embed, first, second))


@pytest.mark.quick
def test_conv_reading_a_layernorm_residual_stream_reads_int8():
    q = _quantized(TinyConvNeXt(), (torch.randn(1, 3, 8, 8),), C7xMMAQuantizer("int8"))
    first, second = _nodes(q, _ADD)
    convs = _nodes(q, torch.ops.aten.conv2d.default)
    dws = [n for n in convs if len(n.args) > 6 and n.args[6] == 8]
    assert len(dws) == 2
    assert _is_dq(dws[1].args[0]), "the second block's depthwise conv reads an int8 copy"
    assert first in second.args, "the second residual add reads the stream as float"
    assert not _requantized(second)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("make_model", "example"),
    [
        (lambda: ResidualModel(), (torch.randn(1, 8, 8, 8),)),
        (lambda: SqueezeExciteBlock(), (torch.randn(1, 3, 16, 16),)),
    ],
    ids=["resnet-add", "no-layernorm"],
)
def test_layernorm_residual_rule_leaves_other_adds_alone(make_model, example):
    on = _quantized(make_model(), example, C7xMMAQuantizer("int8"))
    off = _quantized(
        make_model(), example, C7xMMAQuantizer("int8", float_layernorm_residuals=False)
    )
    assert _call_function_targets(on) == _call_function_targets(off)


@pytest.mark.quick
def test_unknown_activation_observer_is_rejected():
    with pytest.raises(ValueError, match="activation_observer"):
        C7xMMAQuantizer("int8", activation_observer="no-such-observer")


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


# ---------------------------------------------------------------------------
# Transparent ops (Category 1 annotation gaps)
# ---------------------------------------------------------------------------


class TransparentOpsModel(nn.Module):
    """Model that exercises max_pool2d, view, permute, flatten."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)                          # annotated weight op
        x = torch.nn.functional.max_pool2d(x, 2)  # transparent
        b, c = x.shape[:2]
        x = x.view(b, c, -1)                      # transparent (view)
        x = x.permute(0, 2, 1)                    # transparent (permute)
        x = torch.flatten(x, 1)                   # transparent (flatten)
        return x


@pytest.mark.quick
def test_transparent_ops_get_annotated():
    """max_pool2d, view, permute, flatten are annotated with input act_spec."""
    model = TransparentOpsModel().eval()
    example = (torch.randn(1, 4, 8, 8),)
    exported = torch.export.export(model, example).module()
    quantizer = C7xMMAQuantizer("int8")
    quantizer.annotate(exported)

    transparent_targets = {
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.view.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.flatten.using_ints,
    }
    annotated = set()
    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target in transparent_targets:
            ann = node.meta.get("quantization_annotation")
            assert ann is not None, f"{node.target} not annotated"
            assert node.args[0] in ann.input_qspec_map, f"{node.target} args[0] missing"
            annotated.add(node.target)

    assert annotated == transparent_targets, f"missing annotations for: {transparent_targets - annotated}"


@pytest.mark.quick
def test_transparent_ops_output_uses_shared_spec():
    """Transparent ops use SharedQuantizationSpec for output so scales match."""
    from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec

    model = TransparentOpsModel().eval()
    example = (torch.randn(1, 4, 8, 8),)
    exported = torch.export.export(model, example).module()
    quantizer = C7xMMAQuantizer("int8")
    quantizer.annotate(exported)

    transparent_targets = {
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.view.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.flatten.using_ints,
    }
    for node in exported.graph.nodes:
        if node.op == "call_function" and node.target in transparent_targets:
            ann = node.meta.get("quantization_annotation")
            assert isinstance(ann.output_qspec, SharedQuantizationSpec), (
                f"{node.target} output_qspec should be SharedQuantizationSpec, "
                f"got {type(ann.output_qspec)}"
            )


class _InplaceSiluPool(nn.Module):
    """YOLO's SPPF shape: a max pool after an in-place SiLU, which isn't annotated."""

    def forward(self, x):
        y = torch.nn.functional.silu(x * 1.0, inplace=True)
        return torch.nn.functional.max_pool2d(y, 5, stride=1, padding=2)


@pytest.mark.quick
def test_max_pool_after_inplace_silu_keeps_the_clipped_scale():
    """The rule clips the pool's input edge; its output must follow, or it requantizes."""
    x = torch.randn(1, 1, 64, 64)
    x[0, 0, 32, 32] = 50.0  # one outlier: the percentile clips it, a min/max output would not
    q = quantize_pt2e(_InplaceSiluPool(), (x,), C7xMMAQuantizer("int8"))
    (pool,) = [n for n in q.graph.nodes if n.target == torch.ops.aten.max_pool2d.default]
    dq_in = pool.args[0]
    (q_out,) = [u for u in pool.users if "quantize_per_tensor" in str(u.target)]
    assert "dequantize" in str(dq_in.target)
    assert float(q_out.args[1]) == float(dq_in.args[1])
    assert int(q_out.args[2]) == int(dq_in.args[2])


@pytest.mark.quick
def test_transparent_ops_produce_qdq():
    """After convert_pt2e, transparent ops get QDQ wrappers (for EliminateQDQTransparent)."""
    q = quantize_pt2e(
        TransparentOpsModel(), (torch.randn(1, 4, 8, 8),), C7xMMAQuantizer("int8")
    )
    targets = _call_function_targets(q)
    assert _has_quantize(targets), "no quantize nodes after convert_pt2e"
    assert _has_dequantize(targets), "no dequantize nodes after convert_pt2e"
