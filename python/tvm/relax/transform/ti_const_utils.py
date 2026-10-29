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
"""Tensor-constant helper for the C7x passes."""

import numpy as np

import tvm
from tvm.relax import expr as _relax_expr


def ti_tensor_const(value) -> "tvm.ir.GenericConst":
    """Wrap an array as a Relax tensor constant, preserving its dtype.

    ``relax.const`` narrows int64/float64 to int32/float32 when no dtype is
    given, which would silently change MMALIB bias/scale constants.
    """
    if isinstance(value, tvm.runtime.Tensor):
        return _relax_expr._ffi_api.MakeTensorConst(value, None, None)
    return _relax_expr._ffi_api.MakeTensorConst(tvm.runtime.tensor(np.asarray(value)), None, None)
