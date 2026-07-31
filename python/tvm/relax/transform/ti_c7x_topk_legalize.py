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
"""Legalize relax.topk for the C7x/c_static backend.

relax.topk has no default TIR legalization anywhere in this TVM fork --
its only CPU-side lowering (``topi.topk``, python/tvm/topi/sort.py) is a
runtime packed-function call (``tvm.contrib.sort.topk``), inserted by the
``DispatchSortScan`` pass as part of relax.build()'s *default* pipeline.
c_static's own pipeline (cpu_generic/pipeline.py) never runs
DispatchSortScan, and even if it did, c_static's standalone-C executables
have no TVM runtime to satisfy a packed call anyway. This module legalizes
``relax.topk`` directly to a ``call_extern`` into a hand-written C7x kernel
(``src/runtime/ti_dsp/kernels/c7x_topk.cpp``), the same way MMALIB
conv2d/matmul are wired in -- see ``ti_mmalib_legalize.py``.

Scope: only what real graphs actually use today (e.g. yolo26's one2one
detection head, via C7xMMAQuantizer's own topk-reachability exclusion --
see c7x_mma_quantizer.py -- which keeps this op's whole input region in
float): axis=-1 (innermost), ret_type="both", largest=True, float32 input.
Other combinations raise a clear error rather than silently producing wrong
results.
"""

from tvm import te, tir
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Call, Expr


def _te_c7x_topk(data: te.Tensor, k: int, dtype: str) -> list:
    """TE-level topk over the innermost axis, calling the c7x_topk kernel.

    Mirrors ``topi.topk``'s te.extern structure (python/tvm/topi/sort.py),
    swapping the ``tvm.contrib.sort.topk`` packed call for a call_extern
    into our own C7x kernel.
    """
    in_shape = [int(s) for s in data.shape]
    n = in_shape[-1]
    batch = 1
    for s in in_shape[:-1]:
        batch *= s
    out_shape = in_shape[:-1] + [k]

    data_buf = tir.decl_buffer(data.shape, data.dtype, "topk_data_buf", data_alignment=8)
    val_buf = tir.decl_buffer(out_shape, data.dtype, "topk_val_buf", data_alignment=8)
    idx_buf = tir.decl_buffer(out_shape, dtype, "topk_idx_buf", data_alignment=8)

    def fcompute(ins, outs):
        return tir.call_extern(
            "int32", "c7x_topk", ins[0].data, outs[0].data, outs[1].data, batch, n, k
        )

    return te.extern(
        [out_shape, out_shape],
        [data],
        fcompute,
        in_buffers=[data_buf],
        out_buffers=[val_buf, idx_buf],
        name="c7x_topk",
        tag="c7x_topk",
    )


def c7x_topk_legalize(bb: BlockBuilder, call: Call) -> Expr:
    """``customize_legalize_map`` entry for ``relax.topk`` on C7x/c_static."""
    attrs = call.attrs
    data = call.args[0]
    sinfo = data.struct_info
    ndim = sinfo.ndim
    axis = attrs.axis if attrs.axis >= 0 else attrs.axis + ndim

    if axis != ndim - 1:
        raise NotImplementedError(
            "c7x_topk_legalize only supports axis=-1 (innermost); got "
            f"axis={attrs.axis} on a {ndim}-d tensor."
        )
    if not attrs.largest:
        raise NotImplementedError("c7x_topk_legalize only supports largest=True.")
    if str(attrs.ret_type) != "both":
        raise NotImplementedError(
            f"c7x_topk_legalize only supports ret_type='both'; got {attrs.ret_type!r}."
        )
    if sinfo.dtype != "float32":
        raise NotImplementedError(
            f"c7x_topk_legalize only supports float32 input; got dtype={sinfo.dtype!r}."
        )

    return bb.call_te(_te_c7x_topk, data, int(attrs.k), str(attrs.dtype))
