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
"""Commons for Relax frontend."""
from typing import Dict, List, Tuple
import numpy as _np

import tvm
from tvm import relax
from tvm import topi


def detach_params(mod: tvm.IRModule) -> Tuple[tvm.IRModule, Dict[str, List[tvm.runtime.Tensor]]]:
    """Detach the attribute "params" in the functions of the input IRModule as
    separate dictionary of params.

    Parameters
    ----------
    mod : tvm.IRModule
        The IRModule whose functions' "param" attribute is going to be detached.

    Returns
    -------
    detached_mod : tvm.IRModule
        The IRModule after the detachment.

    params_dict : Dict[str, List[tvm.runtime.Tensor]]
        The detached params. The dict keys corresponds to the names of the
        functions in the input IRModule that have attribute "params".
    """
    detached_mod = tvm.IRModule()
    params_dict = dict()
    for gv, func in mod.functions_items():
        if "params" in func.attrs:
            params = list(func.attrs["params"])
            if not all([isinstance(param, tvm.runtime.Tensor) for param in params]):
                raise ValueError('The value "params" attribute is expected to be a list of Tensor.')
            params_dict[gv.name_hint] = params
            detached_mod[gv] = func.without_attr("params")
        else:
            detached_mod[gv] = func
    return detached_mod, params_dict


def autopad(
    bb,
    data,
    strides,
    kernel_shape,
    dilations=(1, 1),
    pad_type="constant",
    deconv=False,
    mode="SAME_UPPER",
    pad_value=0.0,
):
    """
    Perform autopadding with dynamic input shapes
    """
    # get attributes as constants
    strides = _np.array(strides)
    dilated_kernel_shape = _np.array(
        [(kernel - 1) * dilation + 1 for kernel, dilation in zip(kernel_shape, dilations)]
    )
    # get input shape
    ndim = data.struct_info.ndim
    data_shape = list(data.struct_info.shape)
    shape = data_shape[2:ndim]

    # set up integer constants
    zero = 0
    one = 1
    two = 2

    # Calculate total padding
    mod = shape % strides

    left = _np.maximum(dilated_kernel_shape - strides, zero)
    right = _np.maximum(dilated_kernel_shape - mod, zero)

    total_pad = _np.where(_np.equal(mod, zero), left, right)
    if deconv:
        total_pad = _np.array(kernel_shape) - one - total_pad

    # split total padding into before and after
    pad_before = _np.floor_divide(total_pad, two)
    pad_after = total_pad - pad_before

    # combine
    if "LOWER" in mode:
        pad = _np.concatenate(
            [_np.reshape(pad_after, [-1, 1]), _np.reshape(pad_before, [-1, 1])], axis=1
        )
    else:
        pad = _np.concatenate(
            [_np.reshape(pad_before, [-1, 1]), _np.reshape(pad_after, [-1, 1])], axis=1
        )

    # pad N and C with zeros
    pad = _np.concatenate([_np.zeros([2, 2], dtype="int64"), pad], axis=0)

    if pad_type not in ["constant", "edge", "reflect"]:
        raise tvm.error.OpAttributeInvalid(
            "Value " + pad_type + ' in attribute "mode" is invalid for operator Pad.'
        )

    if pad_type == "constant":
        return bb.emit_te(topi.nn.pad, data, pad[:, 0].tolist(), pad[:, 1].tolist(), pad_value)
    elif pad_type == "reflect":
        return bb.emit_te(
            topi.nn.mirror_pad, data, pad[:, 0].tolist(), pad[:, 1].tolist(), "REFLECT"
        )
    else:
        # edge mode - replicate border values
        return bb.emit_te(topi.nn.replicate_pad, data, pad[:, 0].tolist(), pad[:, 1].tolist())

#Begin TI
def unbind(data, axis=0):
    """Unbind operation removes a tensor dimension and returns a list of all slices along a given dimension, with specified axis removed"""
    shape = data.struct_info.shape
    if axis >= len(shape):
        raise AttributeError("Please check input dim, it shouldn't be greater than or equal to rank.")

    selections = int(shape[axis])
    if selections == 1:
        return [relax.op.squeeze(data, axis=[axis])]

    res_split = relax.op.split(data, selections, axis)
    ret = []
    for i in range(selections):
        ret.append(relax.op.squeeze(res_split[i], axis=[axis]))
    return ret


def rnn_cell(input_seqs, hidden_state, w_inp, w_hid, b_inp=None, b_hid=None, backwards=False, act=None, sequence_lens=None, input_dtype=None, hidden_shape=None, clip=None):
    """RNN cell implementation for Relax."""
    if act is None:
        act = relax.op.tanh

    outputs_list = []
    seq_len = len(input_seqs)

    mask_seqs = None
    if sequence_lens is not None:
        seq_len_dtype = sequence_lens.struct_info.dtype

        arange = relax.op.arange(0, seq_len, dtype=seq_len_dtype)
        arange = relax.op.expand_dims(arange, 1)

        seq_len_shape = sequence_lens.struct_info.shape
        sequence_lens_broadcast = relax.op.broadcast_to(sequence_lens, [seq_len, seq_len_shape[0]])

        mask = relax.op.less(arange, sequence_lens_broadcast)

        dtype = input_dtype if input_dtype is not None else "float32"
        mask_float = relax.op.astype(mask, dtype=dtype)

        mask_tensor = relax.op.expand_dims(mask_float, 2)
        mask_seqs = []
        for i in range(seq_len):
            mask_seqs.append(relax.op.take(mask_tensor, relax.const(i), axis=0))

    seq_order = reversed(range(seq_len)) if backwards else range(seq_len)

    for idx in seq_order:
        x_t = input_seqs[idx]
        xwt = relax.op.matmul(x_t, relax.op.permute_dims(w_inp, axes=(1, 0)))
        hwt = relax.op.matmul(hidden_state, relax.op.permute_dims(w_hid, axes=(1, 0)))
        if b_inp is not None:
            xwt = xwt + b_inp
        if b_hid is not None:
            hwt = hwt + b_hid
        new_hidden = act(xwt + hwt)

        if clip is not None:
            new_hidden = relax.op.clip(new_hidden, -clip, clip)

        if mask_seqs is not None:
            mask_idx = mask_seqs[idx]
            one = relax.const(1.0)
            hidden_state = mask_idx * new_hidden + (one - mask_idx) * hidden_state
            outputs_list.append(mask_idx * hidden_state)
        else:
            hidden_state = new_hidden
            outputs_list.append(hidden_state)

    if backwards:
        outputs_list = list(reversed(outputs_list))

    return outputs_list, hidden_state
#End TI

#Begin TI
def lstm_cell(input_seqs, hidden_state, cell_state, w_inp, w_hid, b_inp=None, b_hid=None,
              backwards=False, f_act=None, g_act=None, h_act=None, p_i=None, p_f=None, p_o=None, sequence_lens=None, input_dtype=None, hidden_shape=None, clip=None, input_forget=0):
    """LSTM cell implementation for Relax."""
    if f_act is None:
        f_act = relax.op.sigmoid
    if g_act is None:
        g_act = relax.op.tanh
    if h_act is None:
        h_act = relax.op.tanh

    outputs_list = []
    seq_len = len(input_seqs)

    mask_seqs = None
    if sequence_lens is not None:
        seq_len_dtype = sequence_lens.struct_info.dtype

        arange = relax.op.arange(0, seq_len, dtype=seq_len_dtype)
        arange = relax.op.expand_dims(arange, 1)

        seq_len_shape = sequence_lens.struct_info.shape
        sequence_lens_broadcast = relax.op.broadcast_to(sequence_lens, [seq_len, seq_len_shape[0]])

        mask = relax.op.less(arange, sequence_lens_broadcast)

        dtype = input_dtype if input_dtype is not None else "float32"
        mask_float = relax.op.astype(mask, dtype=dtype)

        mask_tensor = relax.op.expand_dims(mask_float, 2)
        mask_seqs = []
        for i in range(seq_len):
            mask_seqs.append(relax.op.take(mask_tensor, relax.const(i), axis=0))

    seq_order = reversed(range(seq_len)) if backwards else range(seq_len)

    for idx in seq_order:
        x_t = input_seqs[idx]
        # Compute gates
        xwt = relax.op.matmul(x_t, relax.op.permute_dims(w_inp, axes=(1, 0)))
        hwt = relax.op.matmul(hidden_state, relax.op.permute_dims(w_hid, axes=(1, 0)))

        # Add biases
        if b_inp is not None:
            xwt = xwt + b_inp
        if b_hid is not None:
            hwt = hwt + b_hid

        gates = xwt + hwt

        # Split into 4 gates (input, forget, cell, output)
        gate_splits = relax.op.split(gates, 4, axis=-1)
        inp_gate = gate_splits[0]
        fgt_gate = gate_splits[1]
        cell_gate = gate_splits[2]
        otp_gate = gate_splits[3]

        # Apply peephole connections (before activations for input/forget gates)
        if p_i is not None and p_f is not None:
            inp_gate = f_act(inp_gate + p_i * cell_state)
            if input_forget:
                # When input_forget=1, forget gate is complement of input gate
                fgt_gate = relax.const(1.0) - inp_gate
            else:
                fgt_gate = f_act(fgt_gate + p_f * cell_state)
        else:
            inp_gate = f_act(inp_gate)
            if input_forget:
                # When input_forget=1, forget gate is complement of input gate
                fgt_gate = relax.const(1.0) - inp_gate
            else:
                fgt_gate = f_act(fgt_gate)

        cell_gate = g_act(cell_gate)

        # Update cell state: c_t = forget_gate * c_t-1 + input_gate * cell_gate
        new_cell_state = fgt_gate * cell_state + inp_gate * cell_gate

        # Apply output gate activation with peephole
        if p_o is not None:
            otp_gate = f_act(otp_gate + p_o * new_cell_state)
        else:
            otp_gate = f_act(otp_gate)

        new_hidden_state = otp_gate * h_act(new_cell_state)

        if clip is not None:
            new_hidden_state = relax.op.clip(new_hidden_state, -clip, clip)

        if mask_seqs is not None:
            mask_idx = mask_seqs[idx]
            one = relax.const(1.0)
            hidden_state = mask_idx * new_hidden_state + (one - mask_idx) * hidden_state
            cell_state = mask_idx * new_cell_state + (one - mask_idx) * cell_state
            outputs_list.append(mask_idx * hidden_state)
        else:
            hidden_state = new_hidden_state
            cell_state = new_cell_state
            outputs_list.append(hidden_state)

    if backwards:
        outputs_list = list(reversed(outputs_list))

    return outputs_list, hidden_state, cell_state
#End TI

#Begin TI
def gru_cell(
    input_seqs,
    hidden_state,
    w_inp,
    w_hid,
    b_inp=None,
    b_hid=None,
    rz_act=None,
    n_act=None,
    backwards=False,
    linear_before_reset=False,
    sequence_lens=None,
    input_dtype=None,
    hidden_shape=None,
    clip=None,
):
    """
    Common implementation of GRU cell for all frontends of TVM """
    if rz_act is None:
        rz_act = relax.op.sigmoid
    if n_act is None:
        n_act = relax.op.tanh

    outputs_list = []

    seq_len = len(input_seqs)
    if input_dtype is None:
        input_dtype = w_inp.struct_info.dtype
    if hidden_shape is None:
        hidden_shape = list(hidden_state.struct_info.shape)

    mask_seqs = None
    if sequence_lens is not None:
        seq_len_dtype = sequence_lens.struct_info.dtype

        arange = relax.op.arange(0, seq_len, dtype=seq_len_dtype)
        arange = relax.op.expand_dims(arange, 1)

        seq_len_shape = sequence_lens.struct_info.shape
        sequence_lens_broadcast = relax.op.broadcast_to(sequence_lens, [seq_len, seq_len_shape[0]])

        mask = relax.op.less(arange, sequence_lens_broadcast)

        dtype = input_dtype if input_dtype is not None else "float32"
        mask_float = relax.op.astype(mask, dtype=dtype)

        mask_tensor = relax.op.expand_dims(mask_float, 2)
        mask_seqs = []
        for i in range(seq_len):
            mask_seqs.append(relax.op.take(mask_tensor, relax.const(i), axis=0))

    seq_order = reversed(range(seq_len)) if backwards else range(seq_len)

    for idx in seq_order:
        x_t = input_seqs[idx]
        xwt = relax.op.matmul(x_t, relax.op.permute_dims(w_inp, axes=(1, 0)))
        if linear_before_reset:
            hwt = relax.op.matmul(hidden_state, relax.op.permute_dims(w_hid, axes=(1, 0)))
            xwt_splits = relax.op.split(xwt, 3, axis=-1)
            hwt_splits = relax.op.split(hwt, 3, axis=-1)
            # Gate order is [z, r, h] for update, reset, and candidate gates
            i_z, i_r, i_n = xwt_splits[0], xwt_splits[1], xwt_splits[2]
            h_z, h_r, h_n = hwt_splits[0], hwt_splits[1], hwt_splits[2]
            if b_inp is not None and b_hid is not None:
                b_inp_splits = relax.op.split(b_inp, 3, axis=-1)
                b_hid_splits = relax.op.split(b_hid, 3, axis=-1)
                b_iz, b_ir, b_in = b_inp_splits[0], b_inp_splits[1], b_inp_splits[2]
                b_hz, b_hr, b_hn = b_hid_splits[0], b_hid_splits[1], b_hid_splits[2]
                r_gate = rz_act(i_r + b_ir + h_r + b_hr)
                z_gate = rz_act(i_z + b_iz + h_z + b_hz)
                n_gate = n_act(i_n + b_in + r_gate * h_n + b_hn)
            else:
                r_gate = rz_act(i_r + h_r)
                z_gate = rz_act(i_z + h_z)
                n_gate = n_act(i_n + r_gate * h_n)
        else:
            xwt_splits = relax.op.split(xwt, 3, axis=-1)
            w_hid_splits = relax.op.split(w_hid, 3, axis=0)
            # Gate order is [z, r, h] for update, reset, and candidate gates
            i_z, i_r, i_n = xwt_splits[0], xwt_splits[1], xwt_splits[2]
            w_hz, w_hr, w_hn = w_hid_splits[0], w_hid_splits[1], w_hid_splits[2]
            r_gate = i_r + relax.op.matmul(hidden_state, relax.op.permute_dims(w_hr, axes=(1, 0)))
            z_gate = i_z + relax.op.matmul(hidden_state, relax.op.permute_dims(w_hz, axes=(1, 0)))
            if b_inp is not None and b_hid is not None:
                b_inp_splits = relax.op.split(b_inp, 3, axis=-1)
                b_hid_splits = relax.op.split(b_hid, 3, axis=-1)
                # Bias order is [z, r, h] for update, reset, and candidate gates
                b_iz, b_ir, b_in = b_inp_splits[0], b_inp_splits[1], b_inp_splits[2]
                b_hz, b_hr, b_hn = b_hid_splits[0], b_hid_splits[1], b_hid_splits[2]
                r_gate += b_ir + b_hr
                r_gate = rz_act(r_gate)
                z_gate += b_iz + b_hz
                z_gate = rz_act(z_gate)
                i_n += b_in
                h_n = relax.op.matmul(r_gate * hidden_state, relax.op.permute_dims(w_hn, axes=(1, 0))) + b_hn
                n_gate = n_act(i_n + h_n)
            else:
                r_gate = rz_act(r_gate)
                z_gate = rz_act(z_gate)
                h_n = relax.op.matmul(r_gate * hidden_state, relax.op.permute_dims(w_hn, axes=(1, 0)))
                n_gate = n_act(i_n + h_n)

        new_hidden = (hidden_state - n_gate) * z_gate + n_gate

        if clip is not None:
            new_hidden = relax.op.clip(new_hidden, -clip, clip)

        if mask_seqs is not None:
            mask_idx = mask_seqs[idx]
            one = relax.const(1.0)
            hidden_state = mask_idx * new_hidden + (one - mask_idx) * hidden_state
            outputs_list.append(mask_idx * hidden_state)
        else:
            hidden_state = new_hidden
            outputs_list.append(hidden_state)

    if backwards:
        outputs_list = list(reversed(outputs_list))

    return outputs_list, hidden_state
#End TI
