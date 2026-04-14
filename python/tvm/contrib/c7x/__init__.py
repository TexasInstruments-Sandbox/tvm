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
"""TI AM67A C7x DSP runtime integration.

Provides an ARM-side Python API that routes TVM inference to the
C7x DSP via the c7x_compute IPC service.

Usage (identical to relax.VirtualMachine on CPU):

    from tvm.contrib.c7x import C7xVirtualMachine
    import tvm, numpy as np

    vm = C7xVirtualMachine("/path/to/lib0.out")
    out = vm["main"](tvm.nd.array(np.random.randn(1, 64).astype("float32")))

Requires ``libc7x_arm_runtime.so`` installed on the AM67A ARM board.
Build and deploy with:

    cd src/runtime/ti_dsp/firmware/c7x/arm
    ./build.sh && ./build.sh deploy
"""

from .c7x_runtime import C7xVirtualMachine as C7xVirtualMachine
