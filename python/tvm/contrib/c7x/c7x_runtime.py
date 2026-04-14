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
"""C7x Arm runtime wrapper — Python inference API for TVM c_static modules.

Provides a VirtualMachine-compatible interface that routes inference to the
C7x DSP via the c7x_compute IPC service using ctypes bindings to
libc7x_arm_runtime.so.

Usage (identical to relax.VirtualMachine on CPU):

    from tvm.contrib.c7x import C7xVirtualMachine
    import tvm, numpy as np

    # Standard (copy-based inputs, zero-copy outputs):
    vm = C7xVirtualMachine("/models/resnet18.out")
    out = vm["main"](tvm.nd.array(data))   # same syntax as CPU

    # Zero-copy inputs — pre-stage in shared DDR:
    inp = vm.create_input((1, 3, 224, 224), "float32")
    inp.copyfrom(data)           # write directly to staging buffer
    out = vm["main"](inp)        # no staging memcpy

    # Zero-copy outputs — skip the output copy to tvm.nd:
    out_np = vm.run_nocopy(data) # returns numpy view of result DDR

    # Context manager:
    with C7xVirtualMachine("/models/resnet18.out") as vm:
        for batch in data:
            result = vm["main"](tvm.nd.array(batch))

Zero-copy analysis
------------------
- Outputs: c7x_client_infer() already returns pointers into the mmap'd result
  DDR buffer.  vm["main"]() copies them to tvm.nd.array for safety (valid
  across multiple inferences).  run_nocopy() returns numpy views directly
  (zero-copy; valid until the next run_nocopy() call).
- Inputs: standard path copies user buffer to staging DDR (one memcpy).
  create_input() pre-allocates a tensor IN the staging buffer so the memcpy
  is skipped when that tensor is passed to run() / vm["main"]().
"""

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# tvm imported locally in methods that use it (avoids import-order issues)

# ---------------------------------------------------------------------------
# DLDataType codes matching DLPack
# ---------------------------------------------------------------------------
_DL_INT   = 0
_DL_UINT  = 1
_DL_FLOAT = 2

_NUMPY_DTYPE_CODE = {
    "f": _DL_FLOAT,
    "i": _DL_INT,
    "u": _DL_UINT,
}

_DLTYPE_TO_NUMPY: dict = {
    (_DL_FLOAT, 32): np.float32,
    (_DL_FLOAT, 16): np.float16,
    (_DL_FLOAT, 64): np.float64,
    (_DL_INT,   8):  np.int8,
    (_DL_INT,   16): np.int16,
    (_DL_INT,   32): np.int32,
    (_DL_INT,   64): np.int64,
    (_DL_UINT,  8):  np.uint8,
    (_DL_UINT,  16): np.uint16,
    (_DL_UINT,  32): np.uint32,
    (_DL_UINT,  64): np.uint64,
}

# Maximum pre-allocated output descriptors per inference
_MAX_OUTPUTS = 64

# ---------------------------------------------------------------------------
# ctypes structs matching c7x_compute_client.h
# ---------------------------------------------------------------------------


class _C7xTensorDesc(ctypes.Structure):
    """Matches c7x_tensor_desc_t in c7x_compute_client.h.

    Layout (80 bytes with natural alignment):
      void*     data        8
      size_t    data_size   8
      int32_t   ndim        4
      int32_t   dtype_code  4
      int32_t   dtype_bits  4
      int32_t   _pad        4   (compiler alignment to 8-byte boundary)
      int64_t   shape[6]   48
    """

    _fields_ = [
        ("data",       ctypes.c_void_p),
        ("data_size",  ctypes.c_size_t),
        ("ndim",       ctypes.c_int32),
        ("dtype_code", ctypes.c_int32),
        ("dtype_bits", ctypes.c_int32),
        ("_pad",       ctypes.c_int32),
        ("shape",      ctypes.c_int64 * 6),
    ]


assert ctypes.sizeof(_C7xTensorDesc) == 80, (
    f"_C7xTensorDesc size mismatch: {ctypes.sizeof(_C7xTensorDesc)} != 80"
)


# ---------------------------------------------------------------------------
# Load libc7x_arm_runtime.so and set up function signatures
# ---------------------------------------------------------------------------

# Cache loaded CDLL objects by resolved path.  Multiple C7xVirtualMachine
# instances sharing the same .so avoid redundant dlopen() calls and
# ctypes.util.find_library() subprocess spawns.
_lib_cache: dict = {}


def _load_runtime_lib(so_path: str) -> ctypes.CDLL:
    """Load libc7x_arm_runtime.so and configure argtypes/restypes."""
    # Search order: explicit path → LD_LIBRARY_PATH → system lib dirs
    if not os.path.isabs(so_path) and not so_path.startswith("./"):
        found = ctypes.util.find_library("c7x_arm_runtime")
        if found:
            so_path = found

    if so_path in _lib_cache:
        return _lib_cache[so_path]

    try:
        lib = ctypes.CDLL(so_path)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load {so_path}: {exc}\n"
            "Install libc7x_arm_runtime.so to /usr/local/lib/ and run ldconfig."
        ) from exc

    # c7x_client_t *c7x_client_open(void)
    lib.c7x_client_open.restype = ctypes.c_void_p
    lib.c7x_client_open.argtypes = []

    # void c7x_client_close(c7x_client_t *)
    lib.c7x_client_close.restype = None
    lib.c7x_client_close.argtypes = [ctypes.c_void_p]

    # int c7x_client_dyn_load(c7x_client_t *, const char *, uint32_t *)
    lib.c7x_client_dyn_load.restype = ctypes.c_int
    lib.c7x_client_dyn_load.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]

    # int c7x_client_dyn_unload(c7x_client_t *, uint32_t)
    lib.c7x_client_dyn_unload.restype = ctypes.c_int
    lib.c7x_client_dyn_unload.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    # int c7x_client_infer(c7x_client_t *, uint32_t, uint32_t,
    #                      const c7x_tensor_desc_t *, int,
    #                      c7x_tensor_desc_t *, int *, uint64_t *)
    lib.c7x_client_infer.restype = ctypes.c_int
    lib.c7x_client_infer.argtypes = [
        ctypes.c_void_p,                            # client
        ctypes.c_uint32,                             # module_handle
        ctypes.c_uint32,                             # model_id
        ctypes.POINTER(_C7xTensorDesc),              # inputs
        ctypes.c_int,                                # num_inputs
        ctypes.POINTER(_C7xTensorDesc),              # outputs (out)
        ctypes.POINTER(ctypes.c_int),                # num_outputs (out)
        ctypes.POINTER(ctypes.c_uint64),             # cycles (out, optional)
    ]

    # void *c7x_client_get_input_buffer(c7x_client_t *, size_t *)
    lib.c7x_client_get_input_buffer.restype = ctypes.c_void_p
    lib.c7x_client_get_input_buffer.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
    ]

    # void *c7x_client_get_output_buffer(c7x_client_t *, size_t *)
    lib.c7x_client_get_output_buffer.restype = ctypes.c_void_p
    lib.c7x_client_get_output_buffer.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
    ]

    # size_t c7x_client_get_input_data_offset(c7x_client_t *)
    lib.c7x_client_get_input_data_offset.restype = ctypes.c_size_t
    lib.c7x_client_get_input_data_offset.argtypes = [ctypes.c_void_p]

    # const char *c7x_strerror(int)
    lib.c7x_strerror.restype = ctypes.c_char_p
    lib.c7x_strerror.argtypes = [ctypes.c_int]

    _lib_cache[so_path] = lib
    return lib


# ---------------------------------------------------------------------------
# Input tensor helpers
# ---------------------------------------------------------------------------


def _build_input_desc(arr: np.ndarray) -> Tuple[_C7xTensorDesc, int]:
    """Build a _C7xTensorDesc from a contiguous numpy array.

    Returns (desc, data_ptr_int) where data_ptr_int is the raw integer
    address of the numpy array's data buffer (for pre-staged detection).
    """
    desc = _C7xTensorDesc()
    data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
    desc.data = data_ptr
    desc.data_size = arr.nbytes
    desc.ndim = arr.ndim
    desc.dtype_code = _NUMPY_DTYPE_CODE.get(arr.dtype.kind, _DL_FLOAT)
    desc.dtype_bits = arr.dtype.itemsize * 8
    for i, s in enumerate(arr.shape):
        desc.shape[i] = s
    return desc, data_ptr.value or 0


# ---------------------------------------------------------------------------
# C7xVirtualMachine
# ---------------------------------------------------------------------------


class C7xVirtualMachine:
    """Relax VirtualMachine-compatible interface for C7x DSP execution.

    Provides an API identical to ``relax.VirtualMachine`` for CPU:

    .. code-block:: python

        # CPU:
        vm = relax.VirtualMachine(ex, tvm.cpu())
        out = vm["main"](tvm.nd.array(data))

        # C7x (this class):
        vm = C7xVirtualMachine(result.module_path)
        out = vm["main"](tvm.nd.array(data))   # identical

    Zero-copy modes:

    - **Outputs** (``run_nocopy``): returns numpy views backed by result DDR;
      valid until the next ``run_nocopy`` call.
    - **Inputs** (``create_input``): pre-allocates a tvm.nd.NDArray in
      staging DDR; passing it to ``vm["main"]()`` skips the staging memcpy.
    """

    def __init__(
        self,
        module_path: Union[str, Path],
        so_path: str = "libc7x_arm_runtime.so",
    ) -> None:
        self._module_path = Path(module_path)
        self._lib = _load_runtime_lib(so_path)
        self._client: Optional[int] = None   # c7x_client_t* (as int)
        self._handle: Optional[int] = None   # module handle
        self._model_id: int = 0              # 0 = embedded weights

        # Pre-allocated output descriptor array reused across calls
        self._out_descs = (_C7xTensorDesc * _MAX_OUTPUTS)()
        self._num_outputs = ctypes.c_int(0)
        self._cycles = ctypes.c_uint64(0)

        # Staging buffer info (set lazily after load)
        self._staging_ptr: int = 0
        self._staging_size: int = 0

        # Byte offset past the loaded ELF; create_input() allocates from here.
        # Initialized to 0; set from c7x_client_get_input_data_offset() after load.
        self._staging_alloc_offset: int = 0

        # Pre-allocated input tensors (zero-copy path).
        # Dict maps ctypes data pointer (int) → numpy array backed by staging DDR.
        # O(1) lookup during _build_input_descs().
        self._input_slots: dict = {}

        # Zero-copy output views from the last run_nocopy() call.
        self._last_nocopy_outputs: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # VirtualMachine interface
    # ------------------------------------------------------------------

    def __getitem__(self, _name: str):
        """Return a callable for the named function.

        All function names route to the single ``cg_main_dsp`` entry point.
        The returned callable accepts tvm.nd.array or numpy inputs and returns
        a single tvm.nd.array or a list of them (matching VirtualMachine).
        """
        def call(*inputs):
            self._ensure_loaded()
            return self._run(inputs)
        return call

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _infer_raw(self, inputs) -> int:
        """Stage inputs, call c7x_client_infer, return num_outputs.

        Fills self._out_descs[0..num_outputs-1].  Callers wrap the outputs
        in the format they need (tvm.nd.array or numpy view).
        """
        in_descs, _np_refs = self._build_input_descs(inputs)
        num_inputs = len(in_descs)
        in_arr = (_C7xTensorDesc * num_inputs)(*in_descs)
        handle = int(self._handle)  # type: ignore[arg-type]  # non-None after _ensure_loaded

        self._num_outputs.value = 0
        rc = self._lib.c7x_client_infer(
            self._client,
            ctypes.c_uint32(handle),
            ctypes.c_uint32(self._model_id),
            in_arr,
            ctypes.c_int(num_inputs),
            self._out_descs,
            ctypes.byref(self._num_outputs),
            ctypes.byref(self._cycles),
        )
        if rc != 0:
            msg = self._lib.c7x_strerror(rc)
            raise RuntimeError(
                f"C7xVirtualMachine: inference failed: "
                f"{msg.decode() if msg else 'unknown error'} (rc={rc})"
            )
        return int(self._num_outputs.value)

    def _run(self, inputs):  # -> tvm.nd.NDArray or list
        """Run inference, return tvm.nd.array outputs (copy from result DDR)."""
        num_outputs = self._infer_raw(inputs)

        import tvm as _tvm  # local import to avoid top-level coupling
        _nd_array = _tvm.nd.array  # type: ignore[attr-defined]
        results = [_nd_array(self._desc_to_ndarray(self._out_descs[i]))
                   for i in range(num_outputs)]
        return results[0] if len(results) == 1 else results

    def run_nocopy(
        self, *inputs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Run inference and return zero-copy numpy views of result DDR.

        Outputs are numpy arrays whose data backs directly into the mmap'd
        result buffer — **no copy**.  They are valid until the next
        ``run_nocopy()`` call.  For safety across multiple calls, use
        ``vm["main"](...)`` instead (copies outputs to new memory).

        Input handling is identical to ``vm["main"]``: standard inputs are
        copied to staging DDR; pre-staged inputs (from ``create_input()``)
        skip the copy.
        """
        self._ensure_loaded()
        num_outputs = self._infer_raw(inputs)
        results = [self._desc_to_numpy_view(self._out_descs[i])
                   for i in range(num_outputs)]
        self._last_nocopy_outputs = results
        return results[0] if len(results) == 1 else results

    # ------------------------------------------------------------------
    # Zero-copy input: pre-allocate tensor in staging DDR
    # ------------------------------------------------------------------

    def create_input(self, shape: tuple, dtype: str):
        """Pre-allocate an input tensor in the DSP staging buffer.

        Returns a ``tvm.nd.NDArray`` backed by shared DDR.  Filling it
        (e.g., via ``inp.copyfrom(data)`` or ``inp.numpy()[:] = data``)
        writes directly to staging DDR.  Passing this tensor to
        ``vm["main"](inp)`` skips the staging memcpy in the C layer.

        The tensor is valid until ``close()`` is called.
        """
        import tvm as _tvm  # local import

        self._ensure_loaded()

        np_dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * np_dtype.itemsize

        if self._staging_ptr == 0:
            raise RuntimeError(
                "C7xVirtualMachine.create_input: staging buffer not available"
            )
        if self._staging_alloc_offset + nbytes > self._staging_size:
            raise RuntimeError(
                f"C7xVirtualMachine.create_input: staging buffer full "
                f"(need {nbytes} bytes, offset={self._staging_alloc_offset}, "
                f"size={self._staging_size})"
            )

        # Build a numpy array backed by shared memory at the staging offset
        data_ptr = self._staging_ptr + self._staging_alloc_offset
        buf = (ctypes.c_byte * nbytes).from_address(data_ptr)
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
        # Register in dict keyed by ctypes data pointer for O(1) lookup.
        # Holding both the numpy array and buf keeps the ctypes buffer alive.
        self._input_slots[arr.ctypes.data] = arr

        # Advance staging offset (aligned to 64 bytes)
        self._staging_alloc_offset += nbytes
        self._staging_alloc_offset = (self._staging_alloc_offset + 63) & ~63

        _nd_from_dlpack = _tvm.nd.from_dlpack  # type: ignore[attr-defined]
        return _nd_from_dlpack(arr.__dlpack__())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-init: open IPC connection and DYN_LOAD the module."""
        if self._client is not None:
            return
        if not self._module_path.exists():
            raise FileNotFoundError(
                f"C7xVirtualMachine: module not found: {self._module_path}"
            )

        client = self._lib.c7x_client_open()
        if not client:
            raise RuntimeError(
                "C7xVirtualMachine: c7x_client_open() failed — "
                "is the c7x_compute firmware running?"
            )

        handle = ctypes.c_uint32(0)
        rc = self._lib.c7x_client_dyn_load(
            client,
            str(self._module_path).encode(),
            ctypes.byref(handle),
        )
        if rc != 0:
            msg = self._lib.c7x_strerror(rc)
            self._lib.c7x_client_close(client)
            raise RuntimeError(
                f"C7xVirtualMachine: c7x_client_dyn_load({self._module_path}) "
                f"failed: {msg.decode() if msg else 'unknown'} (rc={rc})"
            )

        self._client = client
        self._handle = handle.value

        # Cache staging buffer base, size, and real ELF-end offset.
        sz = ctypes.c_size_t(0)
        self._staging_ptr = self._lib.c7x_client_get_input_buffer(
            self._client, ctypes.byref(sz)
        ) or 0
        self._staging_size = sz.value
        # Use the actual ELF end offset (set by dyn_load) instead of a
        # conservative guess.  This prevents CreateInput() from accidentally
        # landing inside the loaded ELF's in-place rodata segments.
        self._staging_alloc_offset = int(
            self._lib.c7x_client_get_input_data_offset(self._client)
        )

    def close(self) -> None:
        """Unload module and close IPC connection.  Idempotent."""
        if self._handle is not None and self._client is not None:
            self._lib.c7x_client_dyn_unload(
                self._client, ctypes.c_uint32(self._handle)
            )
            self._handle = None
        if self._client is not None:
            self._lib.c7x_client_close(self._client)
            self._client = None
        self._input_slots.clear()
        self._staging_alloc_offset = 0
        self._last_nocopy_outputs = []

    def __enter__(self) -> "C7xVirtualMachine":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_cycles(self) -> int:
        """DSP TSC cycle count from the most recent inference (0 if unavailable)."""
        return int(self._cycles.value)

    @property
    def is_loaded(self) -> bool:
        """True if the module is currently loaded on the DSP."""
        return self._client is not None and self._handle is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input_descs(
        self, inputs
    ) -> Tuple[List[_C7xTensorDesc], List[np.ndarray]]:
        """Convert inputs (tvm.nd or numpy) to _C7xTensorDesc list.

        Pre-staged inputs (data pointer within staging_buf range) are detected
        and passed through without copying in the C layer.
        """
        descs = []
        np_refs: List[np.ndarray] = []  # keep arrays alive during infer call
        for inp in inputs:
            if hasattr(inp, "numpy"):
                # tvm.nd.NDArray — O(1) check against the staging-backed slot
                # dict.  If the NDArray was created via create_input(), its
                # data pointer is a key in self._input_slots; pass that numpy
                # array directly so the C layer skips the staging memcpy.
                np_arr: Optional[np.ndarray] = None
                try:
                    inp_ptr = inp.handle.contents.data  # type: ignore[union-attr]
                    np_arr = self._input_slots.get(inp_ptr)
                except Exception:  # noqa: BLE001
                    pass
                if np_arr is None:
                    try:
                        np_arr = inp.numpy()
                    except Exception:  # noqa: BLE001
                        np_arr = np.asarray(inp)
                    np_arr = np.ascontiguousarray(np_arr)
            else:
                np_arr = np.ascontiguousarray(np.asarray(inp))

            desc, _ = _build_input_desc(np_arr)
            descs.append(desc)
            np_refs.append(np_arr)
        return descs, np_refs

    def _desc_to_ndarray(self, desc: _C7xTensorDesc) -> np.ndarray:
        """Convert output _C7xTensorDesc to a numpy array (copy from result DDR)."""
        dtype = _DLTYPE_TO_NUMPY.get(
            (int(desc.dtype_code), int(desc.dtype_bits)), np.float32
        )
        shape = tuple(int(desc.shape[i]) for i in range(desc.ndim))
        size = int(desc.data_size)
        if desc.data and size > 0:
            buf = (ctypes.c_byte * size).from_address(desc.data)
            # np.frombuffer gives a view of the ctypes buffer; .copy() makes
            # a single allocation+copy (avoids the double-copy of bytes(buf)).
            return np.frombuffer(buf, dtype=dtype).copy().reshape(shape)
        return np.zeros(shape, dtype=dtype)

    def _desc_to_numpy_view(self, desc: _C7xTensorDesc) -> np.ndarray:
        """Wrap output _C7xTensorDesc as zero-copy numpy view of result DDR.

        The returned array is valid until the next run_nocopy() call.
        """
        dtype = _DLTYPE_TO_NUMPY.get(
            (int(desc.dtype_code), int(desc.dtype_bits)), np.float32
        )
        shape = tuple(int(desc.shape[i]) for i in range(desc.ndim))
        size = int(desc.data_size)
        if desc.data and size > 0:
            buf = (ctypes.c_byte * size).from_address(desc.data)
            # NO .copy() — arr.base is the ctypes buffer backed by result DDR
            return np.frombuffer(buf, dtype=dtype).reshape(shape)
        return np.zeros(shape, dtype=dtype)
