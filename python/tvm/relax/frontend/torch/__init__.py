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
"""
PyTorch Frontends for constructing Relax programs, with the model importers
"""
from .exported_program_translator import from_exported_program
from .fx_translator import from_fx
from .dynamo import relax_dynamo, dynamo_capture_subgraphs


def __getattr__(name):
    # C7xMMAQuantizer pulls in torchao at module import time, and torchao is an
    # optional dependency of the torch frontend.  Importing
    # tvm.relax.frontend.torch must not require torchao (upstream only requires
    # torch); only using the quantizer should.  Lazily resolve the symbol so a
    # missing torchao surfaces at the point of use instead of at import.
    if name == "C7xMMAQuantizer":
        from .c7x_mma_quantizer import C7xMMAQuantizer

        return C7xMMAQuantizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
