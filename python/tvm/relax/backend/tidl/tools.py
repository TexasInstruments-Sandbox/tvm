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
"""Locate the TIDL tools directory that TIDL offload reads from.

Every TIDL file TVM uses while compiling a TIDL-offloaded model lives in one
directory, laid out by ``packaging/ti_dsp/stage_tidl_tools.sh``::

    tidl_model_import_relax.so, tidl_model_custom_import.so   import library
    tidl_tools/                     tools the import library runs
    sTIDL_IOBufDesc_t.layout.json   params_1.bin layout
    host/                           c7x_host link libraries

The directory is ``config["tidl_tools_dir"]`` if given, otherwise the copy in
the package (``tvm/data/ti_dsp/tidl``). There is deliberately no environment
variable: a stale one silently mixes files from different TIDL builds, which
must all match. A configured directory that is missing or incomplete is an
error, never a fallback to the packaged copy.
"""

import os
from typing import Any, Dict, Optional

CONFIG_KEY = "tidl_tools_dir"

RELAX_SO = "tidl_model_import_relax.so"
TOOLS_SUBDIR = "tidl_tools"
LAYOUT_FILE = "sTIDL_IOBufDesc_t.layout.json"
HOST_SUBDIR = "host"

# What each part of the flow needs, for checking a directory up front.
_IMPORT_FILES = (
    RELAX_SO,
    "tidl_model_custom_import.so",
    os.path.join(TOOLS_SUBDIR, "PC_dsp_test_dl_algo.out"),
    os.path.join(TOOLS_SUBDIR, "ti_cnnperfsim.out"),
    os.path.join(TOOLS_SUBDIR, "device_config.cfg"),
)
_LAYOUT_FILES = (LAYOUT_FILE,)
_HOST_FILES = (os.path.join(HOST_SUBDIR, "libtidl_api_host.a"),)

_STAGE_HINT = (
    "Stage it with packaging/ti_dsp/stage_tidl_tools.sh (source tree: "
    "src/runtime/ti_dsp/build_runtime.sh tidl-tools), or install a "
    "tvm-ti-c7x-compile wheel built with TIDL support."
)


def _check_removed_keys(config: Dict[str, Any]) -> None:
    if "tidl_tools_path" in config:
        raise ValueError(
            "config['tidl_tools_path'] is no longer supported. Drop it to use the TIDL "
            f"tools in the package, or set config['{CONFIG_KEY}'] to a staged TIDL tools "
            "directory (the directory that contains tidl_tools/, not tidl_tools/ itself)."
        )


def _require(root: str, files, source: str) -> str:
    missing = [f for f in files if not os.path.exists(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(
            f"TIDL tools directory {root!r} ({source}) is missing: "
            f"{', '.join(missing)}. {_STAGE_HINT}"
        )
    return root


def tidl_tools_dir(config: Optional[Dict[str, Any]] = None, *, need: str = "import") -> str:
    """Return the TIDL tools directory, checked for what *need* uses.

    *need* is ``"import"`` (partition and import: the import library and its
    tools), ``"layout"`` (output sizing) or ``"host"`` (c7x_host link).
    Raises FileNotFoundError naming the directory and the missing files.
    """
    config = config or {}
    _check_removed_keys(config)
    files = {"import": _IMPORT_FILES, "layout": _LAYOUT_FILES, "host": _HOST_FILES}[need]
    configured = config.get(CONFIG_KEY)
    if configured is not None:
        return _require(
            os.path.abspath(os.path.expanduser(str(configured))),
            files,
            f"config['{CONFIG_KEY}']",
        )
    from tvm.data.ti_dsp.paths import bundled_tidl_tools_dir  # noqa: PLC0415

    return _require(str(bundled_tidl_tools_dir()), files, "packaged")


def relax_so_path(config: Optional[Dict[str, Any]] = None) -> str:
    return os.path.join(tidl_tools_dir(config), RELAX_SO)


def import_tools_path(config: Optional[Dict[str, Any]] = None) -> str:
    """The ``tidl_tools`` subdirectory the import library is told to run from."""
    return os.path.join(tidl_tools_dir(config), TOOLS_SUBDIR)


def layout_path(config: Optional[Dict[str, Any]] = None) -> str:
    return os.path.join(tidl_tools_dir(config, need="layout"), LAYOUT_FILE)


def host_lib_dir(config: Optional[Dict[str, Any]] = None) -> str:
    return os.path.join(tidl_tools_dir(config, need="host"), HOST_SUBDIR)
