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
"""Unit tests for tvm.contrib.c7x.io_meta byte arithmetic.

Pure arithmetic over a hand-built IRModule -- no compilation, no DSP, no
board. The module mirrors the shape of SmolLM-135M prefill, whose declared
capacity is what the per-buffer dmabuf protocol allocates at DYN_LOAD:

  62 inputs   (input_ids, cache_position, 60 KV cache tensors)
  61 outputs  returned as R.Tuple(R.Tuple(logits), kv_0, ..., kv_59)

The nested one-element inner tuple is not contrived -- it is what
torch.export's single-value return leaves behind once
smollm_c7x._add_kv_scatter_outputs() appends the KV scatter results to a
return that is a Var carrying TupleStructInfo rather than a literal
relax.Tuple. TVM's lowering flattens it before codegen (the one-element
make_tuple is elided outright), so the byte totals here must match that
flattening rather than reject it.

The expected constants are not recomputed from the formula under test; they
are the values read back from a real compiled prefill module's
tvm_dsp_io_meta.bin on beagley-ai, so this pins the arithmetic to observed
hardware behaviour rather than to itself.
"""

import struct
from typing import Optional

import pytest

import tvm
from tvm import relax
from tvm.contrib.c7x.io_meta import (
    IO_META_FLAG_SIZES_EXACT,
    IO_META_MAGIC,
    IO_META_VERSION,
    compute_io_meta,
    compute_io_meta_bytes,
    kv_resident_output_bytes,
)

# SmolLM-135M-Instruct prefill geometry: --prefill-len 64 --max-cache-len 256.
_PREFILL_LEN = 64
_VOCAB_SIZE = 49152
_NUM_KV_TENSORS = 60  # 2 * 30 layers
_KV_SHAPE = [1, 3, 256, 64]  # [batch, num_kv_heads, max_cache_len, head_dim]

# Observed in /tmp/smollm_beagley/prefill/tvm_dsp_io_meta.bin.
_EXPECTED_NUM_INPUTS = 62
_EXPECTED_NUM_OUTPUTS = 61
_EXPECTED_INPUT_BYTES = 11_802_496
_EXPECTED_OUTPUT_BYTES = 24_384_320
# Same module under C7X_INFER_FLAG_KV_RESIDENT: logits only, 1 descriptor.
_EXPECTED_KV_RESIDENT_OUTPUT_BYTES = 12_583_040


def _build_prefill_shaped_module(nest_logits: bool = True) -> tvm.IRModule:
    """A module with SmolLM prefill's entry signature.

    nest_logits=False returns the flat R.Tuple(logits, kv_0, ...) form, to
    show the nesting makes no difference to the totals.
    """
    bb = relax.BlockBuilder()
    params = [
        relax.Var("input_ids", relax.TensorStructInfo([1, _PREFILL_LEN], "int64")),
        relax.Var("cache_position", relax.TensorStructInfo([_PREFILL_LEN], "int64")),
    ] + [
        relax.Var(f"kv_{i}", relax.TensorStructInfo(_KV_SHAPE, "float32"))
        for i in range(_NUM_KV_TENSORS)
    ]

    with bb.function("main", params):
        logits = bb.emit(relax.op.zeros(relax.ShapeExpr([1, _PREFILL_LEN, _VOCAB_SIZE]), "float32"))
        # Stand-ins for the scatter_elements results that write the updated
        # K/V back; only their struct_info matters here.
        kv_outs = [bb.emit(relax.op.add(p, p)) for p in params[2:]]
        first = bb.emit(relax.Tuple([logits])) if nest_logits else logits
        bb.emit_func_output(bb.emit(relax.Tuple([first, *kv_outs])))

    return bb.get()


def _unpack(blob: Optional[bytes]) -> dict:
    assert blob is not None, "expected exact sizes for a fully static signature"
    magic, version, num_in, num_out, in_bytes, out_bytes, flags, reserved = struct.unpack(
        "<IIIIQQII", blob
    )
    return {
        "magic": magic,
        "version": version,
        "num_inputs": num_in,
        "num_outputs": num_out,
        "input_bytes": in_bytes,
        "output_bytes": out_bytes,
        "flags": flags,
        "reserved": reserved,
    }


@pytest.mark.quick
def test_declared_sizes_match_compiled_prefill():
    """compute_io_meta_bytes() must reproduce the blob a real prefill build
    embedded, including flattening the nested return to 61 outputs."""
    meta = _unpack(compute_io_meta_bytes(_build_prefill_shaped_module()))

    assert meta["magic"] == IO_META_MAGIC
    assert meta["version"] == IO_META_VERSION
    assert meta["flags"] == IO_META_FLAG_SIZES_EXACT
    assert meta["reserved"] == 0
    assert meta["num_inputs"] == _EXPECTED_NUM_INPUTS
    assert meta["num_outputs"] == _EXPECTED_NUM_OUTPUTS
    assert meta["input_bytes"] == _EXPECTED_INPUT_BYTES
    assert meta["output_bytes"] == _EXPECTED_OUTPUT_BYTES


@pytest.mark.quick
def test_nesting_does_not_change_declared_sizes():
    """A one-element tuple wrapping the first output must be transparent --
    the compiled module's call ABI flattens it, so the declaration must too.
    Before _flatten_output_tensors recursed, the nested form returned None
    and no metadata was written at all."""
    nested = compute_io_meta_bytes(_build_prefill_shaped_module(nest_logits=True))
    flat = compute_io_meta_bytes(_build_prefill_shaped_module(nest_logits=False))
    assert nested == flat


@pytest.mark.quick
def test_kv_resident_output_bytes_counts_logits_only():
    """Under C7X_INFER_FLAG_KV_RESIDENT the firmware diverts every output
    past the first to C7X_KV_ADDR, so output_buf only needs logits plus one
    descriptor -- roughly half the full declaration."""
    mod = _build_prefill_shaped_module()
    kv_resident = kv_resident_output_bytes(mod)
    assert kv_resident == _EXPECTED_KV_RESIDENT_OUTPUT_BYTES

    declared = compute_io_meta(mod)
    assert declared is not None
    assert kv_resident < declared.output_bytes

    # Keeping every output must reproduce the declaration exactly: both go
    # through _buf_bytes(), so the two figures cannot drift apart.
    assert (
        kv_resident_output_bytes(mod, num_returned=_EXPECTED_NUM_OUTPUTS)
        == declared.output_bytes
    )


@pytest.mark.quick
def test_compute_io_meta_agrees_with_the_packed_blob():
    """The struct layout is the wire format the firmware reads; the fields are
    what callers sizing a reservation want.  They must not disagree."""
    mod = _build_prefill_shaped_module()
    fields = compute_io_meta(mod)
    packed = _unpack(compute_io_meta_bytes(mod))

    assert fields is not None
    assert fields.num_inputs == packed["num_inputs"]
    assert fields.num_outputs == packed["num_outputs"]
    assert fields.input_bytes == packed["input_bytes"]
    assert fields.output_bytes == packed["output_bytes"]


@pytest.mark.quick
def test_kv_resident_output_bytes_is_none_for_symbolic_shape():
    """Same defensive fallback as compute_io_meta_bytes(): a dim TVM never
    resolved to a constant means no exact byte count, so no declaration."""
    n = tvm.tir.SizeVar("n", "int64")
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([1, 4], "float32"))
    with bb.function("main", [x]):
        out = bb.emit(relax.op.zeros(relax.ShapeExpr([1, n]), "float32"))
        bb.emit_func_output(bb.emit(relax.Tuple([out, out])))
    mod = bb.get()

    assert kv_resident_output_bytes(mod) is None
    assert compute_io_meta_bytes(mod) is None
