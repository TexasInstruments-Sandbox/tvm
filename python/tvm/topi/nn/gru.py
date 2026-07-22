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
# pylint: disable=invalid-name
"""General GRU implementation using te.extern + a hand-written TIR loop.

Mirrors lstm.py's structure (precompute the input-hidden matmul outside the
recurrence, do the genuinely-sequential hidden-hidden part inside the
recurrence), but cannot use te.scan the way lstm.py does: te.create_prim_func
(the TE->TIR path used by relax.BlockBuilder.call_te) explicitly rejects
te.ScanOp ("Only te.placeholder and te.compute are allowed for now"), so
te.scan-based ops are unreachable from Relax. te.extern with an ir_builder-
authored body is the alternative recurrence mechanism TVM itself uses for the
same reason (see topi/scan.py's cumsum/cumprod) -- it lowers to a genuine TIR
`for` loop that survives Relax's c_static build pipeline unchanged, instead of
unrolling into one op per timestep.
"""
from tvm import te, tir
from tvm.topi import tag


def gru(
    Xs,
    Wi,
    Wh,
    Bi=None,
    Bh=None,
    h_init=None,
    f_act=tir.sigmoid,
    h_act=tir.tanh,
    reverse=False,
):
    """Single-direction GRU over a full sequence, PyTorch nn.GRU semantics.

    Parameters
    ----------
    Xs : te.Tensor
        Input sequence with shape `(seq_len, batch_size, in_dim)`.
    Wi : te.Tensor
        Input weight matrix with shape `(3 * hidden_dim, in_dim)`, packed by
        gate in PyTorch's fixed r (reset), z (update), n (new) order.
    Wh : te.Tensor
        Hidden weight matrix with shape `(3 * hidden_dim, hidden_dim)`. Packed
        as `Wi`.
    Bi : te.Tensor, optional
        Input bias with shape `(3 * hidden_dim,)`, by default None. Packed as
        `Wi`.
    Bh : te.Tensor, optional
        Hidden bias with shape as `Bi`, by default None. Packed as `Wi`.
    h_init : te.Tensor, optional
        Initial hidden state with shape `(batch_size, hidden_dim)`, zero if
        None.
    f_act, h_act : F, optional
        Gate activation functions: `f_act` for the reset/update gates,
        `h_act` for the new-gate candidate.
    reverse : bool, optional
        Whether to process `Xs` in reverse, by default False.

    Returns
    -------
    result : te.Tensor
        Hidden states with shape `(seq_len, batch_size, hidden_dim)`.
    """
    seq_len, batch_size, in_dim = Xs.shape
    assert (
        Wi.shape[0] % 3 == 0
    ), f"dim 0 of input weight should be 3 * hidden_dim, but {Wi.shape[0]} is not divisible by 3"
    hidden_size = Wi.shape[0] // 3

    # Precompute the input-to-hidden matmul for all 3 gates and all timesteps
    # outside the recurrence -- this is the bulk of the FLOPs and is an
    # ordinary te.compute, so it gets TVM's normal scheduling/vectorization.
    # Only the genuinely sequential hidden-hidden part below needs the
    # hand-written loop.
    ki = te.reduce_axis((0, in_dim), name="ki2h")
    Xi2h = te.compute(
        (seq_len, batch_size, 3, hidden_size),
        lambda t, b, i, j: te.sum(Xs[t, b, ki] * Wi[i * hidden_size + j, ki], axis=ki),
        name="Xi2h",
    )
    if Bi is not None:
        Xi2h = te.compute(
            Xi2h.shape,
            lambda t, b, i, j: Xi2h[t, b, i, j] + Bi[i * hidden_size + j],
            name="Xi2h_bias",
            tag=tag.INJECTIVE,
        )

    if h_init is None:
        h_init = te.compute(
            (batch_size, hidden_size), lambda b, j: tir.const(0.0, Xs.dtype), name="h_init"
        )

    def pos_of(t):
        # Which original-sequence position iteration `t` writes to/reads from.
        return (seq_len - 1 - t) if reverse else t

    def gen_ir(ins, outs):
        if Bh is not None:
            xi2h_buf, wh_buf, hinit_buf, bh_buf = ins
        else:
            xi2h_buf, wh_buf, hinit_buf = ins
            bh_buf = None
        out_buf = outs[0]

        ib = tir.ir_builder.create()
        xi2h_p = ib.buffer_ptr(xi2h_buf)
        wh_p = ib.buffer_ptr(wh_buf)
        hinit_p = ib.buffer_ptr(hinit_buf)
        bh_p = ib.buffer_ptr(bh_buf) if bh_buf is not None else None
        out_p = ib.buffer_ptr(out_buf)

        acc_r = ib.allocate("float32", (1,), name="acc_r", scope="local")
        acc_z = ib.allocate("float32", (1,), name="acc_z", scope="local")
        acc_n = ib.allocate("float32", (1,), name="acc_n", scope="local")
        h_prev = ib.allocate("float32", (1,), name="h_prev", scope="local")

        with ib.for_range(0, seq_len, name="t", kind="serial") as t:
            pos = pos_of(t)
            pos_prev = pos_of(t - 1)
            with ib.for_range(0, batch_size, name="b", kind="serial") as b:
                with ib.for_range(0, hidden_size, name="j", kind="serial") as j:
                    acc_r[0] = 0.0
                    acc_z[0] = 0.0
                    acc_n[0] = 0.0
                    with ib.for_range(0, hidden_size, name="k", kind="serial") as k:
                        with ib.if_scope(t == 0):
                            h_prev[0] = hinit_p[b, k]
                        with ib.else_scope():
                            h_prev[0] = out_p[pos_prev, b, k]
                        acc_r[0] += h_prev[0] * wh_p[0 * hidden_size + j, k]
                        acc_z[0] += h_prev[0] * wh_p[1 * hidden_size + j, k]
                        acc_n[0] += h_prev[0] * wh_p[2 * hidden_size + j, k]

                    r_bias = bh_p[0 * hidden_size + j] if bh_p is not None else 0.0
                    z_bias = bh_p[1 * hidden_size + j] if bh_p is not None else 0.0
                    n_bias = bh_p[2 * hidden_size + j] if bh_p is not None else 0.0

                    r_gate = f_act(xi2h_p[pos, b, 0, j] + acc_r[0] + r_bias)
                    z_gate = f_act(xi2h_p[pos, b, 1, j] + acc_z[0] + z_bias)
                    n_gate = h_act(xi2h_p[pos, b, 2, j] + r_gate * (acc_n[0] + n_bias))

                    with ib.if_scope(t == 0):
                        h_prev[0] = hinit_p[b, j]
                    with ib.else_scope():
                        h_prev[0] = out_p[pos_prev, b, j]
                    out_p[pos, b, j] = (1.0 - z_gate) * n_gate + z_gate * h_prev[0]

        return ib.get()

    input_tensors = [Xi2h, Wh, h_init]
    if Bh is not None:
        input_tensors.append(Bh)

    return te.extern(
        (seq_len, batch_size, hidden_size),
        input_tensors,
        gen_ir,
        name="gru_scan",
        dtype=Xs.dtype,
    )
