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
"""Relax backends"""

from . import contrib, cpu_generic, cuda, gpu_generic, metal, rocm, adreno
from .dispatch_sampling import DispatchSampling
from .dispatch_sort_scan import DispatchSortScan
from .pattern_registry import get_pattern, get_patterns_with_prefix


def __getattr__(name):
    # The TIDL backend pulls in the C7x/TIDL offload stack; import it lazily
    # so `import tvm.relax.backend` doesn't pay that cost unless tidl is
    # actually used.
    if name == "tidl":
        import importlib

        return importlib.import_module(f"{__name__}.tidl")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
