# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Shared helper for propagating PyTorch source spans through fusion and
legalization passes that replace one or more matched ops with a single new
call.

The PT2E/torch.export importer attaches a span (``call.span.source_name``,
a PyTorch module path like ``"model.model.0.conv
[torch.nn.modules.conv.Conv2d]"``) to most ops on the raw imported graph.
``tvm.contrib.c7x.visualize``'s ``_extract_graph`` already reads this
into each node's ``source`` field -- that plumbing needs no changes. What's
missing is that every c7x fusion/legalization pass that matches one or more
of these ops and replaces them with a single new call (``call_te``/
``call_tir``, or a plain legalized op) constructs that replacement without
ever copying a span onto it, so the association with the user's original
model is silently lost from that point on. Confirmed empirically: 0/172
calls in a real compiled yolov8n carry a span by the end of
``legalize_passes``, across every pass family (MMALIB conv/dwconv/fc,
activation fusion, residual-add fusion, movement/concat/avgpool/layernorm
fusion, TIDL maxpool, plain LegalizeOps legalization).

``propagate_span`` is the one place that gets this right, so every pass
calls it once instead of re-implementing "rebuild this Call with the first
available span" ~18 slightly different ways.
"""

from tvm import relax


def find_composite_span(func: relax.Function):
    """Return the first span found among a Composite function's own bindings,
    skipping ``dequantize``/``quantize`` bindings.

    ``FuseOpsByPattern`` groups matched ops into a ``Composite``-tagged
    function; the *outer* call site to that function is never given a span
    (same reason ``tvm.contrib.c7x.visualize``'s own
    ``_extract_composite_calls`` has to reach into the composite body for
    TIDL), but the ops *inside* it retain whatever span they had before
    matching.

    The dequantize/quantize bindings PT2E inserts around the real op are
    *not* a reliable source, despite being structurally first in every QDQ
    composite: confirmed empirically that every single one of them carries
    the same non-user-facing span --
    ``torch.export._trace._non_strict_export.<locals>...Wrapper`` -- an
    artifact of the quantizer's own insertion machinery, not the user's
    model. The actual matched compute op (conv2d, sigmoid, relu, ...) right
    next to it carries the correct PyTorch module path. So this skips
    dequantize/quantize specifically rather than genuinely taking "the
    first span found".
    """
    for block in func.body.blocks:
        for binding in block.bindings:
            val = binding.value
            op_name = str(val.op.name) if hasattr(val, "op") and hasattr(val.op, "name") else ""
            if op_name in ("relax.quantize", "relax.dequantize"):
                continue
            span = getattr(val, "span", None)
            if span is not None:
                return span
    return None


def propagate_span(new_call: relax.Call, source_expr) -> relax.Call:
    """Return new_call rebuilt with source_expr's span, if it has one.

    Parameters
    ----------
    new_call : relax.Call
        The freshly constructed replacement call (typically the not-yet-
        emitted return value of ``BlockBuilder.call_te`` or a legalized op),
        which has no span of its own yet.
    source_expr : Expr or Span or None
        The matched op to take a span from -- normally the single most-
        representative op a QDQ pattern matched around (e.g. the central
        ``conv``/``pool``/``relu`` node). May be an Expr (its ``.span`` is
        used) or a ``Span`` directly (e.g. from ``find_composite_span``,
        used as-is). May be ``None`` or spanless, in which case new_call is
        returned unchanged.

    Returns
    -------
    relax.Call
        new_call unchanged if source_expr has no span, otherwise a new Call
        with every field copied from new_call except ``span``.
    """
    span = source_expr.span if hasattr(source_expr, "span") else source_expr
    if span is None:
        return new_call
    return relax.Call(
        new_call.op,
        new_call.args,
        new_call.attrs,
        new_call.sinfo_args,
        span=span,
    )
