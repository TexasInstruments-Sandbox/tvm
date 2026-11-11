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
"""What TIDL reads and writes for each output of an imported subgraph.

``TIDL_relaxGetOutputInfo`` in the TIDL import library reports, per output,
the fields of the ``sTIDL_IOBufDesc_t`` it wrote to ``subgraph<N>_params_1.bin``
(the same bytes that are embedded in the compiled module). The library reads
them with the header it was built against, so TVM never parses the struct.

TVM uses them for two things:

- **Output storage size.** TIDL writes padding channels (``padCh``) plus one
  spare past the logical tensor, each ``channelPitch`` elements. Handing TIDL a
  buffer sized for the logical tensor lets ``algProcess`` write past the end
  (see ``output_storage.py``).
- **Output dtype.** TIDL can produce an element type other than the one TVM
  declared, and can reorder outputs; ``_fix_output_dtype_mismatches`` casts
  back using ``elementType``, ``tensorScale``, ``zeroPoint`` and the slot.
"""

import json
from dataclasses import dataclass
from typing import Dict, List

import tvm

FFI_NAME = "TIDL_relaxGetOutputInfo"

# eTIDL_ElementType -> size in bytes. Mirrors tidl_element_size() in
# src/runtime/ti_dsp/tidl/tidl_api.c so host and device agree.
_ELEMENT_SIZE = {
    0: 1,  # TIDL_UnsignedChar
    1: 1,  # TIDL_SignedChar
    2: 2,  # TIDL_UnsignedShort
    3: 2,  # TIDL_SignedShort
    4: 4,  # TIDL_UnsignedWord
    5: 4,  # TIDL_SignedWord
    6: 4,  # TIDL_SinglePrecFloat
    7: 8,  # TIDL_UnsignedDoubleWord
    8: 8,  # TIDL_SignedDoubleWord
    9: 1,  # TIDL_Bool
    10: 2,  # TIDL_BFloat16
}


@dataclass(frozen=True)
class TIDLOutput:
    """One TIDL output buffer, as reported by the import library."""

    name: str
    element_type: int
    num_channels: int
    pad_ch: int
    channel_pitch: int
    num_batches: int
    dim1: int
    dim2: int
    buf_size: int
    tensor_scale: float
    zero_point: int

    @property
    def slot(self) -> int:
        """TVM output slot, from ``name`` (``"tidl_<sg>_o<slot>"``), or -1.

        The same parsing ``connect_input_output_tensors()`` in tidl_api.c uses
        to bind outputs, so host and device agree. TIDL may reorder outputs, so
        the slot is not the descriptor index in general.
        """
        pos = self.name.rfind("o")
        digits = self.name[pos + 1 :] if pos >= 0 else ""
        return int(digits) if digits.isdigit() else -1

    def storage_bytes(self) -> int:
        """Bytes that must be allocated for this output.

        Follows TI's own reference harness, ``ti_dl/test/src/tidl_rt.c:567``::

            outputMemRequired = batches * DIM1 * DIM2
                              * (outNumChannels + outPadCh + 1)
                              * outChannelPitch * elementSizeBytes
            buf = malloc(outputMemRequired * elementSizeBytes)

        including the second ``* elementSizeBytes``. That looks redundant, and
        it is what makes TI's harness immune to this bug, so it is reproduced
        deliberately rather than "cleaned up": smaller candidates were measured
        insufficient on a conv with an 8192-byte output, where ``algProcess``
        still corrupted the tail at both ``outBufSize * elementSize`` (9216) and
        the single-``elementSize`` harness formula (10240), and only stopped
        above 10624. For int8 outputs -- the common TIDL case -- elementSize is
        1 and the doubling costs nothing.
        """
        esize = _ELEMENT_SIZE.get(self.element_type, 1)
        elems = (
            max(1, self.num_batches)
            * max(1, self.dim1)
            * max(1, self.dim2)
            * (self.num_channels + self.pad_ch + 1)
            * self.channel_pitch
        )
        # Never go below what the descriptor itself reports.
        return max(elems * esize * esize, self.buf_size * esize)


def slot_of(outputs: List[TIDLOutput], index: int) -> int:
    """TVM output slot of descriptor entry *index* (falls back to *index*)."""
    slot = outputs[index].slot
    return slot if slot >= 0 else index


def output_byte_sizes(outputs: List[TIDLOutput]) -> Dict[int, int]:
    """Map TVM output slot -> bytes that must be allocated for that output."""
    return {slot_of(outputs, i): o.storage_bytes() for i, o in enumerate(outputs)}


def query(artifacts_dir: str, sg_id: int) -> List[TIDLOutput]:
    """Ask the loaded import library for subgraph *sg_id*'s outputs.

    Call after ``TIDL_relaxPostProcessNet`` for that subgraph. Raises
    RuntimeError if the library predates the query, if it reports an error
    (missing or wrong-size descriptor), or if the network compiler did not run,
    in which case the padding TIDL writes is not known.
    """
    fn = tvm.get_global_func(FFI_NAME, allow_missing=True)
    if fn is None:
        raise RuntimeError(
            f"tidl_model_import_relax.so does not provide {FFI_NAME}: it predates this "
            "TVM. Rebuild c7x-mma-tidl and re-stage the TIDL tools directory "
            "(src/runtime/ti_dsp/build_runtime.sh tidl-tools)."
        )
    info = json.loads(str(fn(artifacts_dir, sg_id)))
    if "error" in info:
        raise RuntimeError(f"{FFI_NAME} failed for subgraph {sg_id}: {info['error']}")
    if not info["compiled"]:
        raise RuntimeError(
            f"TIDL subgraph {sg_id} was imported without the network compiler, so the "
            "padding TIDL writes past each output is unknown. Check that ti_cnnperfsim.out "
            "is in the TIDL tools directory and that the import is not float or 32-bit."
        )
    return [
        TIDLOutput(
            name=o["name"],
            element_type=o["elementType"],
            num_channels=o["numChannels"],
            pad_ch=o["padCh"],
            channel_pitch=o["channelPitch"],
            num_batches=o["numBatches"],
            dim1=o["dim1"],
            dim2=o["dim2"],
            buf_size=o["bufSize"],
            tensor_scale=float(o["tensorScale"]),
            zero_point=o["zeroPoint"],
        )
        for o in info["outputs"]
    ]
