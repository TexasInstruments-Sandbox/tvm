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
"""Reader for TIDL's ``sTIDL_IOBufDesc_t`` (``subgraph<N>_params_1.bin``).

The import tool writes one of these blobs per subgraph.  The field we need is
``outBufSize``: the number of elements TIDL will actually write for each output,
which is **larger** than the logical tensor because TIDL appends padding
channels (``outPadCh``) plus one spare, each of ``outChannelPitch`` elements.
Handing TIDL a buffer sized only for the logical tensor lets ``algProcess``
write past the end.

Layout is derived by parsing the struct out of ``itidl_io.h`` rather than
hard-coding offsets, and the computed total is checked against the blob size.
The wheel cannot ship ``itidl_io.h``, so ``packaging/ti_dsp/build_wheel.sh``
runs this module once (``python -m tvm.relax.backend.tidl.io_buf_desc``) to
write the parsed layout as JSON, from the same c7x-mma-tidl tree the bundled
import library comes from; :func:`load_layout` prefers that table and falls back
to the header in a source checkout.
That check matters: ``src/runtime/ti_dsp/tidl/tidl_api.c`` already documents
how a ``TIDL_MAX_NUM_CORES`` mismatch silently shifts every field in this
struct, so a layout change must fail loudly here instead of returning
plausible-looking garbage.
"""

import argparse
import json
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

# Element type -> size in bytes, from the eTIDL_ElementType defines in
# itidl_io.h.  Mirrors tidl_element_size() in
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

# Scalar sizes for the types that appear in sTIDL_IOBufDesc_t.  Every field is
# either a 4-byte word or an int8_t name array whose length is a multiple of 4,
# so fields pack with no padding and offsets are a running sum.
_TYPE_SIZE = {
    "int8_t": 1,
    "uint8_t": 1,
    "int32_t": 4,
    "uint32_t": 4,
    "float32_tidl": 4,
    "float": 4,
}

_FIELD_RE = re.compile(
    r"^\s*(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<dims>(?:\[[^\]]+\]\s*)*)\s*;"
)
_DIM_RE = re.compile(r"\[([^\]]+)\]")


STRUCT_NAME = "sTIDL_IOBufDesc_t"
LAYOUT_FILE = f"{STRUCT_NAME}.layout.json"


def _default_header_path() -> str:
    root = os.environ.get("C7X_MMA_TIDL_PATH", os.path.expanduser("~/ml/c7x-mma-tidl"))
    return os.path.join(root, "ti_dl", "inc", "itidl_io.h")


def _strip_casts(expr: str) -> str:
    """Drop ``(int32_t)`` / ``(uint32_t)`` casts.

    Must happen before digits are extracted: the ``32`` in ``int32_t`` would
    otherwise be read as the value of ``#define X ((int32_t) 128)``.
    """
    return re.sub(r"\(\s*u?int\d+_t\s*\)", "", expr)


def _parse_defines(text: str) -> Dict[str, int]:
    """Collect ``#define NAME ((int32_t) 123)`` / ``#define NAME 123U`` values."""
    out: Dict[str, int] = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+(.+)$", text, re.MULTILINE):
        name, raw = m.group(1), _strip_casts(m.group(2))
        num = re.search(r"(\d+)", raw)
        if num:
            out[name] = int(num.group(1))
    return out


def _eval_dim(expr: str, defines: Dict[str, int]) -> int:
    """Evaluate a dimension like ``A * (uint32_t)B`` using header defines."""
    cleaned = _strip_casts(expr)
    for name, value in defines.items():
        cleaned = re.sub(rf"\b{re.escape(name)}\b", str(value), cleaned)
    if not re.fullmatch(r"[\d\s*+()]+", cleaned):
        raise ValueError(f"Cannot evaluate IOBufDesc dimension {expr!r} (got {cleaned!r})")
    return int(eval(cleaned))  # noqa: S307 - restricted to digits and * + ( ) above


class TIDLIOBufDesc:
    """Field accessors for one parsed ``subgraph<N>_params_1.bin``."""

    def __init__(self, fields: Dict[str, tuple], blob: bytes):
        self._fields = fields  # name -> (offset, count, elem_size, ctype, row_stride)
        self._blob = blob

    def ints(self, name: str, count: Optional[int] = None) -> List[int]:
        """Read an int32 field as a list (optionally only the first ``count``)."""
        offset, total, elem_size, ctype, _ = self._fields[name]
        if elem_size != 4 or ctype not in ("int32_t", "uint32_t"):
            raise TypeError(f"{name} is {ctype}, not an int32 array")
        n = total if count is None else min(count, total)
        return list(struct.unpack_from(f"<{n}i", self._blob, offset))

    def floats(self, name: str, count: Optional[int] = None) -> List[float]:
        """Read a float32 field as a list (optionally only the first ``count``)."""
        offset, total, elem_size, ctype, _ = self._fields[name]
        if elem_size != 4 or ctype not in ("float32_tidl", "float"):
            raise TypeError(f"{name} is {ctype}, not a float32 array")
        n = total if count is None else min(count, total)
        return list(struct.unpack_from(f"<{n}f", self._blob, offset))

    def scalar(self, name: str) -> int:
        return self.ints(name, 1)[0]

    def name_at(self, name: str, index: int) -> str:
        """Read one entry of an int8_t[N][STRING_SIZE] name array."""
        offset, total, esize, _, stride = self._fields[name]
        if stride is None:
            raise TypeError(f"{name} is not a 2-D name array")
        if (index + 1) * stride > total * esize:
            raise IndexError(f"{name}[{index}] out of range")
        raw = self._blob[offset + index * stride : offset + (index + 1) * stride]
        return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")

    def output_slot_map(self, num_out: Optional[int] = None) -> List[int]:
        """Map descriptor index -> TVM output slot, via ``outDataName``.

        The slot comes from ``outDataName``, encoded ``"tidl_<sg>_o<slot>"`` --
        the same parsing ``connect_input_output_tensors()`` in tidl_api.c uses to
        bind outputs, so the mapping agrees with the device side.  TIDL may
        reorder outputs during compilation, so slot != index in general.

        Returns a list ``slots`` where ``slots[i]`` is the TVM output slot for
        descriptor index ``i``. Falls back to ``slot = i`` when ``outDataName``
        is absent or doesn't carry a trailing ``o<digits>`` suffix.
        """
        n = self.scalar("numOutputBuf") if num_out is None else num_out
        slots = []
        for i in range(n):
            slot = i
            try:
                out_name = self.name_at("outDataName", i)
            except (KeyError, TypeError, IndexError):
                out_name = ""
            pos = out_name.rfind("o")
            if pos >= 0:
                digits = out_name[pos + 1 :]
                if digits.isdigit():
                    slot = int(digits)
            slots.append(slot)
        return slots

    def output_byte_sizes(self) -> Dict[int, int]:
        """Map TVM output slot -> bytes that must be allocated for that output.

        See :meth:`output_slot_map` for how the slot is derived from
        ``outDataName``.

        The size follows TI's own reference harness,
        ``ti_dl/test/src/tidl_rt.c:567``::

            outputMemRequired = batches * DIM1 * DIM2
                              * (outNumChannels + outPadCh + 1)
                              * outChannelPitch * elementSizeBytes
            buf = malloc(outputMemRequired * elementSizeBytes)

        including the second ``* elementSizeBytes``.  That looks redundant, and
        it is what makes TI's harness immune to this bug, so it is reproduced
        deliberately rather than "cleaned up": smaller candidates were measured
        insufficient on a conv with a 8192-byte output, where ``algProcess``
        still corrupted the tail at both ``outBufSize * elementSize`` (9216) and
        the single-``elementSize`` harness formula (10240), and only stopped
        above 10624.  For int8 outputs -- the common TIDL case -- elementSize is
        1 and the doubling costs nothing.
        """
        num_out = self.scalar("numOutputBuf")
        elem_types = self.ints("outElementType", num_out)
        num_ch = self.ints("outNumChannels", num_out)
        pad_ch = self.ints("outPadCh", num_out)
        ch_pitch = self.ints("outChannelPitch", num_out)
        batches = self.ints("outNumBatches", num_out)
        dim1 = self.ints("outDIM1", num_out)
        dim2 = self.ints("outDIM2", num_out)
        buf_sizes = self.ints("outBufSize", num_out)
        slot_map = self.output_slot_map(num_out)

        result: Dict[int, int] = {}
        for i in range(num_out):
            slot = slot_map[i]
            esize = _ELEMENT_SIZE.get(elem_types[i], 1)
            elems = (
                max(1, batches[i])
                * max(1, dim1[i])
                * max(1, dim2[i])
                * (num_ch[i] + pad_ch[i] + 1)
                * ch_pitch[i]
            )
            # Never go below what the descriptor itself reports.
            result[slot] = max(elems * esize * esize, buf_sizes[i] * esize)
        return result


def parse_layout_from_header(header_path: str) -> Dict[str, Any]:
    """Parse ``sTIDL_IOBufDesc_t`` out of ``itidl_io.h`` into a layout table.

    Returns ``{"struct": ..., "size": total_bytes, "fields": {name: [offset,
    count, elem_size, ctype, row_stride]}}``; the JSON form of this is what the
    wheel bundles.
    """
    with open(header_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    body = re.search(rf"typedef struct\s*\{{(.*?)\}}\s*{STRUCT_NAME}\s*;", text, re.DOTALL)
    if body is None:
        raise ValueError(f"Could not find {STRUCT_NAME} in {header_path}")
    defines = _parse_defines(text)

    fields: Dict[str, list] = {}
    offset = 0
    for line in body.group(1).splitlines():
        if "//" in line:
            line = line.split("//", 1)[0]
        m = _FIELD_RE.match(line)
        if m is None:
            continue
        ctype = m.group("type")
        if ctype not in _TYPE_SIZE:
            raise ValueError(f"Unhandled type {ctype!r} in {STRUCT_NAME}")
        esize = _TYPE_SIZE[ctype]
        dims = [_eval_dim(d, defines) for d in _DIM_RE.findall(m.group("dims"))]
        count = 1
        for d in dims:
            count *= d
        # For 2-D name arrays keep the inner stride so name_at() can index rows.
        stride = dims[-1] * esize if len(dims) == 2 else None
        fields[m.group("name")] = [offset, count, esize, ctype, stride]
        offset += count * esize
    return {"struct": STRUCT_NAME, "size": offset, "fields": fields}


def _load_layout_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        layout = json.load(fh)
    if (
        not isinstance(layout, dict)
        or layout.get("struct") != STRUCT_NAME
        or not isinstance(layout.get("size"), int)
        or not isinstance(layout.get("fields"), dict)
    ):
        raise ValueError(f"{path} is not a valid {STRUCT_NAME} layout table")
    return layout


def load_layout(header_path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Find the ``sTIDL_IOBufDesc_t`` layout; returns ``(layout, source)``.

    Lookup order: *header_path*, the table bundled in the wheel, then
    ``itidl_io.h`` under ``$C7X_MMA_TIDL_PATH`` (source checkouts).  Raises
    FileNotFoundError naming every place it looked.
    """
    if header_path is not None:
        if not os.path.isfile(header_path):
            raise FileNotFoundError(f"itidl_io.h not found at {header_path!r}")
        return parse_layout_from_header(header_path), header_path

    tried = []
    try:
        from tvm.data.ti_dsp.paths import find_tidl_iobufdesc_layout  # noqa: PLC0415

        bundled = find_tidl_iobufdesc_layout()
    except ImportError:
        bundled = None
    if bundled is not None:
        return _load_layout_json(bundled), bundled
    tried.append(f"bundled tvm/data/ti_dsp/tidl/{LAYOUT_FILE}")

    header = _default_header_path()
    if os.path.isfile(header):
        return parse_layout_from_header(header), header
    tried.append(header)
    raise FileNotFoundError(
        f"No {STRUCT_NAME} layout found (tried: {', '.join(tried)}). Install a "
        "wheel built with --tidl ON, or set C7X_MMA_TIDL_PATH to a c7x-mma-tidl tree."
    )


def read_io_buf_desc(
    blob_path: str, layout: Dict[str, Any], source: str = "layout"
) -> TIDLIOBufDesc:
    """Read a ``subgraph<N>_params_1.bin`` blob with an already loaded layout.

    Raises ValueError if the layout does not account for exactly the blob's
    bytes -- see the module docstring.
    """
    with open(blob_path, "rb") as fh:
        blob = fh.read()
    if layout["size"] != len(blob):
        raise ValueError(
            f"{STRUCT_NAME} layout mismatch: {source} implies {layout['size']} bytes "
            f"but {blob_path} is {len(blob)} bytes. The header used to write the blob "
            "differs from the one the layout came from (see TIDL_MAX_NUM_CORES in "
            "tidl_api.c); refusing to read fields at wrong offsets."
        )
    fields = {name: tuple(f) for name, f in layout["fields"].items()}
    return TIDLIOBufDesc(fields, blob)


def parse_io_buf_desc(blob_path: str, header_path: Optional[str] = None) -> TIDLIOBufDesc:
    """Parse a ``subgraph<N>_params_1.bin`` written by the TIDL import tool.

    The layout comes from :func:`load_layout`.  Raises FileNotFoundError if no
    layout is available and ValueError if it does not match the blob size.
    """
    layout, source = load_layout(header_path)
    return read_io_buf_desc(blob_path, layout, source)


def tidl_output_storage_bytes(artifacts_dir: str, sg_id: int) -> Dict[int, int]:
    """Per-output-slot byte requirement for one subgraph, or {} if unavailable."""
    blob = os.path.join(artifacts_dir, f"subgraph{sg_id}_params_1.bin")
    if not os.path.isfile(blob):
        return {}
    return parse_io_buf_desc(blob).output_byte_sizes()


def main(argv: Optional[List[str]] = None) -> None:
    """Write the layout table parsed from ``itidl_io.h`` (used by build_wheel.sh)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--header", required=True, help="path to itidl_io.h")
    parser.add_argument("--out", required=True, help=f"output path ({LAYOUT_FILE})")
    args = parser.parse_args(argv)
    layout = parse_layout_from_header(args.header)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=1, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
