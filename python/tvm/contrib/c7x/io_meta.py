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
"""Compile-time IO metadata for the c7x per-buffer dmabuf protocol.

Computes declared input/output byte capacities for a compiled module's
entry function, matching ``struct tvm_dsp_io_meta`` in
``src/runtime/ti_dsp/firmware/c7x/common/c7x_compute_protocol.h`` and the
symbol lookup in ``handle_dyn_load()``
(``src/runtime/ti_dsp/firmware/c7x/dsp/src/compute_service.c``).
"""

import struct
from pathlib import Path
from typing import List, NamedTuple, Optional, Union

import tvm
from tvm import relax

# Little-endian bytes "TIOM", read back as a uint32 on both the ARM host and
# the (little-endian) C7x DSP.
IO_META_MAGIC = 0x4D4F4954
IO_META_VERSION = 1
IO_META_FLAG_SIZES_EXACT = 1 << 0

# Must match sizeof(struct c7x_tensor_desc) in c7x_compute_protocol.h.
TENSOR_DESC_SIZE = 80
_ALIGN = 64


def _round_up(nbytes: int, align: int = _ALIGN) -> int:
    return (nbytes + align - 1) & ~(align - 1)


def _tensor_nbytes(sinfo: relax.StructInfo) -> Optional[int]:
    """Byte size of one tensor, or None if it isn't a TensorStructInfo with an
    entirely compile-time-constant shape."""
    if not isinstance(sinfo, relax.TensorStructInfo):
        return None
    shape = sinfo.shape
    if not isinstance(shape, relax.ShapeExpr):
        return None
    nbytes = tvm.DataType(sinfo.dtype).itemsize
    for dim in shape.values:
        if not isinstance(dim, tvm.tir.IntImm):
            return None
        nbytes *= int(dim.value)
    return nbytes


def _sizes_or_none(sinfos: List[relax.StructInfo]) -> Optional[List[int]]:
    """[_tensor_nbytes(s) for s in sinfos], or None if any tensor's size is unknown."""
    sizes = []
    for sinfo in sinfos:
        nbytes = _tensor_nbytes(sinfo)
        if nbytes is None:
            return None
        sizes.append(nbytes)
    return sizes


def _flatten_output_tensors(sinfo: relax.StructInfo) -> Optional[List[relax.StructInfo]]:
    """Flatten a return StructInfo into a list of TensorStructInfo, recursing
    through arbitrarily nested tuples.

    A nested tuple field (e.g. a single-tensor return wrapped as
    ``R.Tuple(R.Tuple(R.Tensor(...)), ...)`` -- seen from SmolLM's
    torch.export-derived return combined with its KV scatter outputs) has no
    effect on the compiled module's actual call ABI: TVM's lowering
    (ExpandTupleArguments/FuseTIR) already flattens nested Relax tuples into
    a flat output list before codegen, so the byte-size total computed here
    must match that flattening rather than rejecting it. Returns None if
    sinfo contains anything that isn't a Tensor or a Tuple of such (e.g. an
    Object/PrimValue field)."""
    if isinstance(sinfo, relax.TensorStructInfo):
        return [sinfo]
    if isinstance(sinfo, relax.TupleStructInfo):
        flat = []
        for field in sinfo.fields:
            sub = _flatten_output_tensors(field)
            if sub is None:
                return None
            flat.extend(sub)
        return flat
    return None


def _buf_bytes(sizes: List[int]) -> int:
    """Capacity one dmabuf needs to hold ``sizes`` tensors and their descriptors.

    The descriptor region (D9) precedes tensor data at the front of
    ``input_buf``.  On the output side the firmware appends the descriptor
    array *after* the tensor data, and only when it doesn't fit inline in the
    IPC response (``extract_infer_output()``); room is reserved for it
    unconditionally rather than replicating that threshold here, because the
    descriptors cost a few hundred bytes next to the tensor data whereas a
    capacity short by exactly that much makes every inference of a
    many-output model fail with C7X_STATUS_ERR_SIZE.
    """
    return _round_up(len(sizes) * TENSOR_DESC_SIZE) + sum(_round_up(s) for s in sizes)


class IoMeta(NamedTuple):
    """Declared entry-point IO capacity for one compiled module."""

    num_inputs: int
    num_outputs: int
    input_bytes: int
    output_bytes: int


def compute_io_meta(mod: tvm.IRModule, entry_name: str = "main") -> Optional[IoMeta]:
    """Declared input_buf/output_buf capacity for ``mod[entry_name]``.

    Returns None if any input/output tensor has a genuinely non-static
    shape (a symbolic dim TVM never resolved to a compile-time constant) --
    entry shapes are static in practice, so this is a defensive fallback,
    not a routine path. Nested output tuples are not such a case: they're
    flattened by ``_flatten_output_tensors`` to match how the runtime ABI
    already flattens them. When this does return None, the caller should
    skip writing the file; a missing symbol is a supported "no metadata"
    state on the firmware side, resolved at runtime via
    ``c7x_client_reserve_io()`` instead.
    """
    func = mod[entry_name]

    input_sinfos = [p.struct_info for p in func.params]
    if not all(isinstance(s, relax.TensorStructInfo) for s in input_sinfos):
        return None
    output_sinfos = _flatten_output_tensors(func.ret_struct_info)
    if output_sinfos is None:
        return None

    input_sizes = _sizes_or_none(input_sinfos)
    if input_sizes is None:
        return None
    output_sizes = _sizes_or_none(output_sinfos)
    if output_sizes is None:
        return None

    return IoMeta(
        num_inputs=len(input_sizes),
        num_outputs=len(output_sizes),
        input_bytes=_buf_bytes(input_sizes),
        output_bytes=_buf_bytes(output_sizes),
    )


def compute_io_meta_bytes(mod: tvm.IRModule, entry_name: str = "main") -> Optional[bytes]:
    """Pack a ``tvm_dsp_io_meta`` blob for ``mod[entry_name]``.

    Returns None under the same condition as ``compute_io_meta()``, whose
    docstring explains it.  Callers that want the numbers rather than the
    wire format should use that instead of unpacking this.
    """
    meta = compute_io_meta(mod, entry_name)
    if meta is None:
        return None

    return struct.pack(
        "<IIIIQQII",
        IO_META_MAGIC,
        IO_META_VERSION,
        meta.num_inputs,
        meta.num_outputs,
        meta.input_bytes,
        meta.output_bytes,
        IO_META_FLAG_SIZES_EXACT,
        0,  # reserved
    )


def kv_resident_output_bytes(
    mod: tvm.IRModule, entry_name: str = "main", num_returned: int = 1
) -> Optional[int]:
    """``output_buf`` bytes needed when only the first ``num_returned`` output
    tensors are written there.

    ``C7X_INFER_FLAG_KV_RESIDENT`` makes the DSP copy every output past the
    first ``num_returned`` to the persistent ``C7X_KV_ADDR`` region rather
    than into ``output_buf`` (``copy_kv_to_fixed_region()`` in
    ``compute_service.c``), so a caller that always sets the flag needs far
    less than the entry signature implies -- for SmolLM prefill, 12,583,040
    bytes against a declared 24,384,320.

    This does not make ``compute_io_meta_bytes()`` wrong: that stays the
    correct declaration for the *same* module invoked without the flag,
    where all outputs do land in ``output_buf``. One compiled module has two
    footprints depending on the calling convention, and only the caller
    knows which one it will use, so this is the per-call figure to pass to
    ``c7x_client_reserve_io()``.

    Returns None under the same condition as ``compute_io_meta()`` (a
    non-static shape), and goes through the same ``_buf_bytes()`` as the
    declaration so the two figures cannot drift apart.
    """
    output_sinfos = _flatten_output_tensors(mod[entry_name].ret_struct_info)
    if output_sinfos is None:
        return None
    sizes = _sizes_or_none(output_sinfos[:num_returned])
    if sizes is None:
        return None
    return _buf_bytes(sizes)


def write_io_meta(
    mod: tvm.IRModule, output_path: Union[str, Path], entry_name: str = "main"
) -> bool:
    """Write a ``tvm_dsp_io_meta.bin`` for ``mod[entry_name]`` to ``output_path``.

    Returns False (no file written) when ``compute_io_meta_bytes`` can't
    derive exact sizes -- see its docstring.
    """
    data = compute_io_meta_bytes(mod, entry_name)
    if data is None:
        return False
    Path(output_path).write_bytes(data)
    return True
