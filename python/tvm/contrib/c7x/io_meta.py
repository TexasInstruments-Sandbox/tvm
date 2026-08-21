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
from typing import List, Optional, Union

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
    dtype = tvm.DataType(sinfo.dtype)
    nbytes = (dtype.bits * dtype.lanes + 7) // 8
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
    """Flatten a return StructInfo into a list of TensorStructInfo, recursing one
    level into tuples (the shape the c7x protocol's output descriptors take).
    Returns None if sinfo contains anything else (e.g. a nested tuple)."""
    if isinstance(sinfo, relax.TensorStructInfo):
        return [sinfo]
    if isinstance(sinfo, relax.TupleStructInfo):
        if not all(isinstance(f, relax.TensorStructInfo) for f in sinfo.fields):
            return None
        return list(sinfo.fields)
    return None


def compute_io_meta_bytes(mod: tvm.IRModule, entry_name: str = "main") -> Optional[bytes]:
    """Pack a ``tvm_dsp_io_meta`` blob for ``mod[entry_name]``.

    Returns None if any input/output tensor has a non-static shape (no model
    in the current regression set does -- entry shapes are static in
    practice -- so this is a defensive fallback, not a routine path). The
    caller should then skip writing the file; a missing symbol is a
    supported "no metadata" state on the firmware side, resolved at runtime
    via ``c7x_client_reserve_io()`` instead.
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

    num_inputs = len(input_sizes)
    num_outputs = len(output_sizes)

    # Descriptor region (D9) precedes tensor data at the front of input_buf.
    descs_bytes = _round_up(num_inputs * TENSOR_DESC_SIZE)
    input_bytes = descs_bytes + sum(_round_up(s) for s in input_sizes)
    output_bytes = sum(_round_up(s) for s in output_sizes)

    return struct.pack(
        "<IIIIQQII",
        IO_META_MAGIC,
        IO_META_VERSION,
        num_inputs,
        num_outputs,
        input_bytes,
        output_bytes,
        IO_META_FLAG_SIZES_EXACT,
        0,  # reserved
    )


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
