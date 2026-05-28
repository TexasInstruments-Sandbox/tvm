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
"""TIDL backend for Relax BYOC partitioning and compilation.

This package provides TIDL (TI Deep Learning) subgraph offloading for the
TVM/Relax c_static backend targeting C7x DSP with MMA accelerator.
"""

# Re-export from canonical location (TIDL-independent, lives in tvm.contrib.c7x)
from tvm.contrib.c7x import C7xVirtualMachine as C7xVirtualMachine
from .patterns import get_tidl_patterns as get_tidl_patterns
from .tidl import LowerTIDLToTIR as LowerTIDLToTIR
from .tidl import TIDLBuildResult as TIDLBuildResult
from .tidl import TIDLOffloadCompiler as TIDLOffloadCompiler
from .tidl import generate_artifacts_c as generate_artifacts_c
from .tidl import partition_for_tidl as partition_for_tidl
