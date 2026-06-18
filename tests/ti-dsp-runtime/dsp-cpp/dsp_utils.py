"""
DSP compilation and execution utilities for TVM C static backend.

This module provides utilities for compiling TVM IRModules to C code for DSP,
building executables for host emulation and TI DSP hardware, and running inference
with output comparison.

The module supports:
- Host emulation using the DSP runtime (for development and debugging)
- C66x cross-compilation and execution on AWRL6844 hardware
- C7x host emulation using TI Host Emulation library
- C7x DLOAD dynamic module loading via c7x_compute firmware
- Output comparison against reference (PyTorch, LLVM, etc.)
"""

import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

import tvm
from tvm import relax
from tvm.contrib import tar

# Configure logger for this module
logger = logging.getLogger(__name__)


def _run_timed(cmd, cwd, log_f, label):
    """Run a subprocess, log elapsed wall-clock time to log_f and stdout."""
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT, check=False
    )
    elapsed = time.perf_counter() - t0
    msg = f"[timing] {label}: {elapsed:.1f}s\n"
    log_f.write(msg)
    log_f.flush()
    print(f"    {msg}", end="")
    return result


# Tensor file format constants
TVM_TENSOR_FILE_MAGIC = 0x54564D54  # "TVMT" in ASCII
TVM_TENSOR_FILE_VERSION = 1

# Module paths
_MODULE_DIR = Path(__file__).parent.resolve()
_TVM_HOME = _MODULE_DIR.parent.parent.parent  # tests/ti-dsp-runtime/dsp-cpp -> tvm
_DSP_RUNTIME_DIR = _TVM_HOME / "src" / "runtime" / "ti_dsp"

# Default output file names (used by both Python and C++)
OUTPUT_BIN_FILE = "output.bin"
OUTPUT_NPY_FILE = "output.npy"
INPUT_BIN_FILE = "input.bin"

# DSP modes that use C7x code generation
_C7X_MODES = ("c7x_host", "c7x_dload")

# Module-level test name for workspace naming (set by pytest fixture)
_current_test_name: Optional[str] = None


def set_current_test_name(name: Optional[str]) -> None:
    """Set the current test name for workspace directory naming."""
    global _current_test_name
    _current_test_name = name


def get_target_string(
    dsp_mode: str,
    profile_layers: bool = False,
    use_cpp_api: bool = False,
) -> str:
    """
    Get the c_static target string for the given DSP execution mode.

    Centralizes the mode-to-target mapping so that test files do not
    need to enumerate DSP modes individually.

    Args:
        dsp_mode: Execution mode ("c66x_host", "c66x", "c7x_host", or "c7x_dload")
        profile_layers: Append -profile-layers flag
        use_cpp_api: Append -use-cpp-api=1 flag

    Returns:
        Target string, e.g. "c_static -mcpu=c7x -use-cpp-api=1"
    """
    if dsp_mode in _C7X_MODES:
        target = "c_static -mcpu=c7x"
    else:
        target = "c_static -mcpu=c66x"
    if profile_layers:
        target += " -profile-layers"
    if use_cpp_api:
        target += " -use-cpp-api=1"
    return target


def assert_dsp_comparison(
    dsp_results: dict,
    comparison: dict,
) -> None:
    """
    Assert DSP results: check for errors and verify comparison passes.

    This replaces the per-mode boilerplate that was duplicated across every
    test file. It dynamically discovers result keys so that adding a new
    execution mode (e.g. c7x_host) does not require updating each test.

    Raises:
        AssertionError: If any execution error occurred, any mode's
            comparison failed, or no results were produced.
    """
    # Check for execution errors
    error_keys = [k for k in dsp_results if k.endswith("_error")]
    for key in error_keys:
        mode = key.removesuffix("_error")
        raise AssertionError(f"{mode} execution error: {dsp_results[key]}")

    # Print profile data if available (cycles, layer traces)
    for mode_prefix in ("c7x_dload", "c66x", "c7x_host", "c66x_host"):
        cycles_key = f"{mode_prefix}_cycles"
        stdout_key = f"{mode_prefix}_stdout"
        if cycles_key in dsp_results:
            cycles = dsp_results[cycles_key]
            print(f"\n{mode_prefix} cycles: {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)")
        if stdout_key in dsp_results:
            stdout = dsp_results[stdout_key]
            # Print layer profile and TIDL trace sections
            for line in stdout.split("\n"):
                if any(
                    k in line
                    for k in (
                        "Layer Profile",
                        "Total:",
                        "cycles",
                        "TIDL Per-Layer",
                        "End TIDL",
                        "Iteration",
                        "input_offset",
                    )
                ):
                    print(line)

    # Check each mode's comparison results
    diff_keys = [k for k in comparison if k.endswith("_vs_ref_max_diff")]
    for key in diff_keys:
        passed_key = key.replace("_max_diff", "_passed")
        if not comparison.get(passed_key, False):
            mode = key.removesuffix("_vs_ref_max_diff")
            raise AssertionError(f"{mode} failed: max diff = {comparison[key]:.2e}")

    # At least one mode should have produced results
    assert diff_keys, "No DSP results available. Check hardware connection or execution mode."


@contextmanager
def temporary_dsp_workspace(name: Optional[str] = None, cleanup: bool = True):
    """
    Create a temporary workspace for DSP compilation and execution.

    This context manager creates a unique temporary directory for each test execution,
    enabling parallel test execution without race conditions from shared build directories.
    The workspace is automatically cleaned up after use (unless disabled for debugging).

    Args:
        name: Optional name for the workspace directory.  If None, uses the
            module-level _current_test_name (set by pytest fixture).  When a
            name is available, the directory is deterministically named
            (e.g. /tmp/dsp_test_conv2d_dsp_c7x_host__20260505_143027/).
            Falls back to a random suffix when no name is set.
        cleanup: Whether to cleanup temp directory after use (default: True)
                Can be overridden by setting DSP_KEEP_TEMP=1 environment variable

    Yields:
        Path object pointing to the temporary workspace directory

    Environment Variables:
        DSP_KEEP_TEMP: Set to "1" to disable cleanup for debugging purposes
    """
    resolved_name = name if name is not None else _current_test_name
    if resolved_name:
        safe_name = re.sub(r"[^\w\-.]", "_", resolved_name)[:80]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"dsp_{safe_name}_{timestamp}"
        temp_dir = str(Path(tempfile.gettempdir()) / dir_name)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="dsp_build_")
    temp_path = Path(temp_dir)

    try:
        yield temp_path

    finally:
        should_cleanup = cleanup and os.getenv("DSP_KEEP_TEMP") != "1"
        if should_cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            logger.info(f"Keeping temporary workspace for debugging: {temp_dir}")


def write_tensors_to_file(tensors: List[np.ndarray], filename: str) -> None:
    """
    Write list of numpy arrays to binary tensor file.

    File format (from tensor_file_format.md):
    - File header: magic (uint32), version (uint32), num_tensors (uint32)
    - Per tensor: ndim, shape[], dtype_code, dtype_bits, data_size, data

    Args:
        tensors: List of numpy arrays to write
        filename: Output file path
    """
    # Map numpy dtype to DLPack type code
    dtype_map = {
        np.int8: (0, 8),
        np.int16: (0, 16),
        np.int32: (0, 32),
        np.int64: (0, 64),
        np.uint8: (1, 8),
        np.uint16: (1, 16),
        np.uint32: (1, 32),
        np.uint64: (1, 64),
        np.float16: (2, 16),
        np.float32: (2, 32),
        np.float64: (2, 64),
    }

    with open(filename, "wb") as f:
        # Write file header
        f.write(struct.pack("<I", TVM_TENSOR_FILE_MAGIC))
        f.write(struct.pack("<I", TVM_TENSOR_FILE_VERSION))
        f.write(struct.pack("<I", len(tensors)))

        for tensor in tensors:
            # Ensure C-contiguous layout
            arr = np.ascontiguousarray(tensor)

            # Get dtype info
            dtype_key = arr.dtype.type
            if dtype_key not in dtype_map:
                raise ValueError(f"Unsupported dtype: {arr.dtype}")
            dtype_code, dtype_bits = dtype_map[dtype_key]

            # Write tensor header
            f.write(struct.pack("<i", arr.ndim))
            for dim in arr.shape:
                f.write(struct.pack("<q", dim))
            f.write(struct.pack("<i", dtype_code))
            f.write(struct.pack("<i", dtype_bits))

            # Write data size and data
            data_bytes = arr.tobytes()
            f.write(struct.pack("<q", len(data_bytes)))
            f.write(data_bytes)

    logger.debug(f"Wrote {len(tensors)} tensors to {filename}")


def read_tensors_from_file(filename: str) -> List[np.ndarray]:
    """
    Read list of numpy arrays from binary tensor file.

    Args:
        filename: Input file path

    Returns:
        List of numpy arrays
    """
    # Map DLPack type code to numpy dtype
    dtype_map = {
        (0, 8): np.int8,
        (0, 16): np.int16,
        (0, 32): np.int32,
        (0, 64): np.int64,
        (1, 8): np.uint8,
        (1, 16): np.uint16,
        (1, 32): np.uint32,
        (1, 64): np.uint64,
        (2, 16): np.float16,
        (2, 32): np.float32,
        (2, 64): np.float64,
    }

    with open(filename, "rb") as f:
        # Read and validate file header
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != TVM_TENSOR_FILE_MAGIC:
            raise ValueError(
                f"Invalid magic number: 0x{magic:08X} (expected 0x{TVM_TENSOR_FILE_MAGIC:08X})"
            )

        version = struct.unpack("<I", f.read(4))[0]
        if version != TVM_TENSOR_FILE_VERSION:
            raise ValueError(f"Unsupported version: {version} (expected {TVM_TENSOR_FILE_VERSION})")

        num_tensors = struct.unpack("<I", f.read(4))[0]
        tensors = []

        for _ in range(num_tensors):
            # Read tensor header
            ndim = struct.unpack("<i", f.read(4))[0]
            shape = tuple(struct.unpack("<q", f.read(8))[0] for _ in range(ndim))
            dtype_code = struct.unpack("<i", f.read(4))[0]
            dtype_bits = struct.unpack("<i", f.read(4))[0]
            data_size = struct.unpack("<q", f.read(8))[0]

            # Get numpy dtype
            dtype = dtype_map.get((dtype_code, dtype_bits))
            if dtype is None:
                raise ValueError(f"Unsupported dtype: code={dtype_code}, bits={dtype_bits}")

            # Read data
            data = f.read(data_size)
            if len(data) != data_size:
                raise ValueError(f"Unexpected EOF: read {len(data)} bytes, expected {data_size}")

            arr = np.frombuffer(data, dtype=dtype).reshape(shape)
            tensors.append(arr.copy())  # Copy to make writeable

    logger.debug(f"Read {len(tensors)} tensors from {filename}")
    return tensors


def compile_for_dsp(
    mod: tvm.IRModule,
    target_string: str = "c_static -mcpu=c66x",
    output_dir: Optional[Path] = None,
    relax_pipeline=None,
) -> Path:
    """
    Compile TVM IRModule to C code for DSP.

    This function compiles a TVM IRModule using the c_static backend and exports
    the generated files (lib0.c, devc.c, weights.bin) to the output directory.

    Args:
        mod: TVM IRModule to compile (should have parameters bound)
        target_string: Target specification (default: "c_static -mcpu=c66x")
        output_dir: Directory to store generated files.
                   If None, creates a temporary directory.
        relax_pipeline: Custom Relax compilation pipeline.
                   If None, uses the default cpu_generic pipeline.

    Returns:
        Path to directory containing lib0.c, devc.c, weights.bin
    """
    logger.info(f"Compiling for DSP with target: {target_string}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="tvm_dsp_"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = tvm.target.Target(target_string)

    # Compile the Relax module
    # Use cpu_generic pipeline which includes FuseOps+FuseTIR for operator
    # fusion, reducing per-layer function call overhead on DSP targets.
    if relax_pipeline is None:
        from tvm.relax.backend.cpu_generic.pipeline import get_default_pipeline

        relax_pipeline = get_default_pipeline(target)

    from tvm.ir.instrument import PassTimingInstrument

    timing_inst = PassTimingInstrument()
    t_build = time.perf_counter()
    logger.debug("Building Relax module...")
    with target:
        with tvm.transform.PassContext(opt_level=3, instruments=[timing_inst]):
            executable = relax.build(
                mod,
                target,
                exec_mode="compiled",
                system_lib=True,
                relax_pipeline=relax_pipeline,
                tir_pipeline=None,  # use target-aware TIR pipeline
            )
            # render() must be called before PassContext exits — exit_pass_ctx
            # clears the TLS profile store on context __exit__.
            pass_timing = timing_inst.render()
    t_build = time.perf_counter() - t_build
    logger.info(f"[timing] relax.build: {t_build:.1f}s")
    print(f"    [timing] relax.build: {t_build:.1f}s")
    if pass_timing:
        print("    [passes]\n" + pass_timing)

    t_export = time.perf_counter()
    tar_path = output_dir / "model_library.tar"
    logger.debug(f"Exporting library to: {tar_path}")
    executable.export_library(str(tar_path), target=target)
    t_export = time.perf_counter() - t_export
    logger.info(f"[timing] export_library: {t_export:.1f}s")
    print(f"    [timing] export_library: {t_export:.1f}s")

    # Extract generated files
    logger.debug(f"Extracting generated files to: {output_dir}")
    tar.untar(str(tar_path), str(output_dir))

    # Verify generated files exist
    lib_files = sorted(output_dir.glob("lib*.c"))
    weights_path = output_dir / "weights.bin"
    if not lib_files:
        raise FileNotFoundError(f"No generated code (lib*.c) found in {output_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    logger.info(f"Generated files in: {output_dir}")
    for lib_path in lib_files:
        logger.info(f"  {lib_path.name}: {lib_path.stat().st_size} bytes")
    logger.info(f"  weights.bin: {weights_path.stat().st_size} bytes")

    return output_dir


def build_dsp_host(
    generated_dir: Path,
    dsp_cpp_dir: Optional[Path] = None,
    build_type: str = "Release",
    build_dir: Optional[Path] = None,
) -> Path:
    """
    Build DSP executable for host emulation using dsp-cpp/CMakeLists.txt.

    Args:
        generated_dir: Directory containing lib0.c and weights.bin.
        dsp_cpp_dir: Directory containing CMakeLists.txt and main_dsp.cpp.
                    If None, uses this module's directory.
        build_type: Build type - "Release" (default) or "Debug".
        build_dir: Optional build directory. If None, creates one in dsp_cpp_dir.
                  When provided, enables isolated builds for parallel test execution.

    Returns:
        Path to the cg_dsp executable
    """
    if dsp_cpp_dir is None:
        dsp_cpp_dir = _MODULE_DIR

    generated_dir = Path(generated_dir).resolve()
    dsp_cpp_dir = Path(dsp_cpp_dir).resolve()

    logger.info(f"Building DSP host executable ({build_type})")
    logger.info(f"  Generated code: {generated_dir}")
    logger.info(f"  Build config: {dsp_cpp_dir}")

    # Create build directory - use provided one or create in source tree
    if build_dir is None:
        build_suffix = "" if build_type == "Release" else f"-{build_type.lower()}"
        build_dir = dsp_cpp_dir / f"build{build_suffix}"
    else:
        build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Build dir: {build_dir}")

    # Configure cmake - use absolute path to source directory
    cmake_cmd = [
        "cmake",
        "-DTVM_DSP_TARGET=host",
        f"-DTVM_HOME={_TVM_HOME}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DGENERATED_CODE_DIR={generated_dir}",
        f"-DWEIGHTS_FILE={generated_dir / 'weights.bin'}",
        str(dsp_cpp_dir),
    ]

    log_path = build_dir / "cmake.log"
    logger.debug(f"Running CMake: {' '.join(cmake_cmd)}")

    with open(log_path, "w") as f:
        result = subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed. Check {log_path} for details.")

        # Build
        logger.debug("Building cg_dsp target...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed. Check {log_path} for details.")

    executable = build_dir / "cg_dsp"
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")

    logger.info(f"Built host executable: {executable}")
    return executable


def build_dsp_c66x(
    generated_dir: Path,
    dsp_cpp_dir: Optional[Path] = None,
    build_type: str = "Release",
    build_dir: Optional[Path] = None,
) -> Path:
    """
    Cross-compile DSP executable for C66x using TI compiler.

    Args:
        generated_dir: Directory containing lib0.c and weights.bin.
        dsp_cpp_dir: Directory containing CMakeLists.txt and main_dsp.cpp.
                    If None, uses this module's directory.
        build_type: Build type - "Release" (default) or "Debug".
        build_dir: Optional build directory. If None, creates one in dsp_cpp_dir.
                  When provided, enables isolated builds for parallel test execution.

    Returns:
        Path to the cg_dsp_c66x.out executable
    """
    if dsp_cpp_dir is None:
        dsp_cpp_dir = _MODULE_DIR

    generated_dir = Path(generated_dir).resolve()
    dsp_cpp_dir = Path(dsp_cpp_dir).resolve()

    # Toolchain file
    toolchain_file = _DSP_RUNTIME_DIR / "cmake" / "toolchain-awrl6844.cmake"
    if not toolchain_file.exists():
        raise FileNotFoundError(f"Toolchain file not found: {toolchain_file}")

    logger.info(f"Building C66x executable ({build_type})")
    logger.info(f"  Generated code: {generated_dir}")
    logger.info(f"  Toolchain: {toolchain_file}")

    # Create build directory - use provided one or create in source tree
    if build_dir is None:
        build_suffix = "" if build_type == "Release" else f"-{build_type.lower()}"
        build_dir = dsp_cpp_dir / f"build-awrl6844{build_suffix}"
    else:
        build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Build dir: {build_dir}")

    # Configure cmake with C66x toolchain - use absolute path to source directory
    cmake_cmd = [
        "cmake",
        "-DTVM_DSP_TARGET=c66x",
        "-DTVM_DSP_DEVICE=awrl6844",
        f"-DTVM_HOME={_TVM_HOME}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DGENERATED_CODE_DIR={generated_dir}",
        f"-DWEIGHTS_FILE={generated_dir / 'weights.bin'}",
        str(dsp_cpp_dir),
    ]

    log_path = build_dir / "cmake.log"
    logger.debug(f"Running CMake: {' '.join(cmake_cmd)}")

    with open(log_path, "w") as f:
        result = subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed. Check {log_path} for details.")

        # Build
        logger.debug("Building cg_dsp_c66x target...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed. Check {log_path} for details.")

    executable = build_dir / "cg_dsp_c66x.out"
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")

    logger.info(f"Built C66x executable: {executable}")
    return executable


def build_dsp_c7x_host(
    generated_dir: Path,
    dsp_cpp_dir: Optional[Path] = None,
    build_type: str = "Release",
    build_dir: Optional[Path] = None,
    tidl_bridge: Optional[str] = None,
    use_tidl: bool = False,
    tidl_artifacts_dir: Optional[str] = None,
) -> Path:
    """
    Build DSP executable for C7x host emulation using TI Host Emulation library.

    This compiles TVM-generated C code (from c_static -mcpu=c7x) with system g++
    and the TI C7000 Host Emulation library, producing an x86-64 executable that
    emulates C7x vector types and intrinsics.

    Requires TI_CGT_C7000_PATH environment variable to be set.

    Args:
        generated_dir: Directory containing lib0.c and weights.bin.
        dsp_cpp_dir: Directory containing CMakeLists.txt and main_dsp.cpp.
                    If None, uses this module's directory.
        build_type: Build type - "Release" (default) or "Debug".
        build_dir: Optional build directory. If None, creates one in dsp_cpp_dir.
                  When provided, enables isolated builds for parallel test execution.
        tidl_bridge: Optional path to tidl_bridge.c source file.
        use_tidl: If True, pass -DUSE_TIDL=ON to cmake to link PC TIDL algo libs.
        tidl_artifacts_dir: Directory with TIDL artifacts (subgraph*_net.bin etc.).
                            Required when use_tidl=True.

    Returns:
        Path to the cg_dsp executable
    """
    if dsp_cpp_dir is None:
        dsp_cpp_dir = _MODULE_DIR

    generated_dir = Path(generated_dir).resolve()
    dsp_cpp_dir = Path(dsp_cpp_dir).resolve()

    # Check for TI_CGT_C7000_PATH
    ti_cgt_c7000 = os.environ.get("TI_CGT_C7000_PATH")
    if not ti_cgt_c7000:
        raise EnvironmentError(
            "TI_CGT_C7000_PATH environment variable not set. "
            "Set it to the TI C7000 CGT installation path."
        )

    logger.info(f"Building C7x host emulation executable ({build_type})")
    logger.info(f"  Generated code: {generated_dir}")
    logger.info(f"  TI_CGT_C7000_PATH: {ti_cgt_c7000}")

    # Create build directory
    if build_dir is None:
        build_suffix = "" if build_type == "Release" else f"-{build_type.lower()}"
        build_dir = dsp_cpp_dir / f"build-c7x-host{build_suffix}"
    else:
        build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Build dir: {build_dir}")

    # Configure cmake
    cmake_cmd = [
        "cmake",
        "-DTVM_DSP_TARGET=c7x_host",
        f"-DTVM_HOME={_TVM_HOME}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DGENERATED_CODE_DIR={generated_dir}",
        f"-DWEIGHTS_FILE={generated_dir / 'weights.bin'}",
    ]
    if tidl_bridge:
        # Accept a single path or a list; cmake list items are separated by ";"
        if isinstance(tidl_bridge, (list, tuple)):
            cmake_cmd.append("-DTIDL_BRIDGE_SOURCES=" + ";".join(str(p) for p in tidl_bridge))
        else:
            cmake_cmd.append(f"-DTIDL_BRIDGE_SOURCES={tidl_bridge}")
    if use_tidl:
        cmake_cmd.append("-DUSE_TIDL=ON")
    if tidl_artifacts_dir:
        cmake_cmd.append(f"-DTIDL_ARTIFACTS_DIR={tidl_artifacts_dir}")
    cmake_cmd.append(str(dsp_cpp_dir))

    log_path = build_dir / "cmake.log"
    logger.debug(f"Running CMake: {' '.join(cmake_cmd)}")

    with open(log_path, "w") as f:
        result = subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed. Check {log_path} for details.")

        # Build
        logger.debug("Building cg_dsp target (c7x_host)...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            cwd=str(build_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed. Check {log_path} for details.")

    executable = build_dir / "cg_dsp"
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")

    logger.info(f"Built C7x host emulation executable: {executable}")
    return executable


def build_dsp_dynmod(
    generated_dir: Union[str, Path],
    dsp_cpp_dir: Optional[Path] = None,
    build_type: str = "Release",
    build_dir: Optional[Path] = None,
    weights_file: Optional[Union[str, Path]] = None,
    tidl_bridge: Optional[str] = None,
    use_tidl: bool = False,
    tidl_artifacts_dir: Optional[str] = None,
    fp_reassoc_off: bool = False,
    lib0_cflags: str = "",
) -> Path:
    """
    Build TVM-generated lib0.c as a DLOAD-compatible C7x relocatable module.

    This uses a two-stage link process:
      Stage 1: dsp_syms.c → dsp_syms.out (pseudo-firmware symbol table)
      Stage 2: lib0.c + dsp_syms.out → lib0.out (relocatable module)

    The resulting lib0.out can be loaded at runtime by the c7x_compute firmware's
    DLOAD dynamic loader via: c7x_compute load lib0.out

    Args:
        generated_dir: Directory containing lib0.c (from TVM compilation).
        dsp_cpp_dir: Directory containing CMakeLists.txt for dynmod build.
            If None, uses ``src/runtime/ti_dsp/dynmod/`` (canonical location).
            Falls back to this module's directory for backward compatibility.
        build_type: Build type - "Release" (default) or "Debug".
        build_dir: Optional build directory. If None, creates one in dsp_cpp_dir.
        fp_reassoc_off: If True, compile lib0.c with ``--fp_reassoc=off`` to
            disable the cl7x floating-point reassociation optimization.  This
            prevents the compiler from reordering matmul accumulations, which
            causes large numerical divergence in models with ill-conditioned
            weights (e.g. LLMs).  Adds ~27% cycle overhead.
        lib0_cflags: Extra flags appended to lib0.c compilation only, passed
            via ``-DLIB0_USER_FLAGS``.  Use for testing alternative optimization
            levels or enabling ``-k`` (keep assembly output alongside .obj).

    Returns:
        Path to the lib0.out relocatable module
    """
    if dsp_cpp_dir is None:
        # Prefer canonical dynmod location; fall back to test dir
        dynmod_dir = _DSP_RUNTIME_DIR / "dynmod"
        if dynmod_dir.exists():
            dsp_cpp_dir = dynmod_dir
        else:
            dsp_cpp_dir = _MODULE_DIR

    generated_dir = Path(generated_dir).resolve()
    dsp_cpp_dir = Path(dsp_cpp_dir).resolve()

    # Use the same C7x toolchain
    toolchain_file = _DSP_RUNTIME_DIR / "cmake" / "toolchain-j722s-c7x.cmake"
    if not toolchain_file.exists():
        raise FileNotFoundError(f"Toolchain file not found: {toolchain_file}")

    logger.info(f"Building C7x dynamic module ({build_type})")
    logger.info(f"  Generated code: {generated_dir}")

    # Create build directory
    if build_dir is None:
        build_suffix = "" if build_type == "Release" else f"-{build_type.lower()}"
        build_dir = dsp_cpp_dir / f"build-dynmod{build_suffix}"
    else:
        build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Build dir: {build_dir}")

    # Auto-detect weights.bin from generated_dir if not specified
    if weights_file is None:
        auto_weights = generated_dir / "weights.bin"
        if auto_weights.exists():
            weights_file = auto_weights
    if weights_file is not None and Path(weights_file).exists():
        logger.info(f"  Weights: {weights_file} (will be embedded)")

    # Configure cmake with C7x toolchain + C7X_DYNMOD flag
    cmake_cmd = [
        "cmake",
        "-DC7X_DYNMOD=ON",
        f"-DTVM_HOME={_TVM_HOME}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DGENERATED_CODE_DIR={generated_dir}",
        str(dsp_cpp_dir),
    ]
    if weights_file is not None and Path(weights_file).exists():
        cmake_cmd.insert(-1, f"-DWEIGHTS_FILE={Path(weights_file).resolve()}")
    if tidl_bridge:
        cmake_cmd.insert(-1, f"-DTIDL_BRIDGE_SOURCES={tidl_bridge}")
    if use_tidl:
        cmake_cmd.insert(-1, "-DUSE_TIDL=ON")
    if tidl_artifacts_dir:
        cmake_cmd.insert(-1, f"-DTIDL_ARTIFACTS_DIR={tidl_artifacts_dir}")
    if fp_reassoc_off:
        cmake_cmd.insert(-1, "-DFP_REASSOC_OFF=ON")
    if lib0_cflags:
        cmake_cmd.insert(-1, f"-DLIB0_USER_FLAGS={lib0_cflags}")

    log_path = build_dir / "cmake.log"
    logger.debug(f"Running CMake: {' '.join(cmake_cmd)}")

    with open(log_path, "w") as f:
        result = _run_timed(cmake_cmd, build_dir, f, "cmake configure")
        if result.returncode != 0:
            raise RuntimeError(f"CMake configuration failed. Check {log_path} for details.")

        logger.debug("Building c7x_dynmod target...")
        result = _run_timed(
            ["cmake", "--build", ".", "--parallel"], build_dir, f, "cmake build (all stages)"
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed. Check {log_path} for details.")

    module = build_dir / "lib0.out"
    if not module.exists():
        raise FileNotFoundError(f"Module not found: {module}")

    logger.info(f"Built C7x dynamic module: {module}")
    return module


def run_dsp_remote(
    c7x_compute_cli: str,
    dynmod_path: Union[str, Path],
    weights_path: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    host: str = "am67a",
    user: str = "root",
    remote_dir: str = "/tmp/c7x_compute",
) -> dict:
    """
    Run inference on remote AM67A via c7x_compute CLI over SSH.

    Steps:
      1. scp lib0.out, weights.bin, input.bin to target
      2. c7x_compute model-load weights.bin → model_id
      3. c7x_compute load lib0.out → module_handle
      4. c7x_compute infer <handle> <model_id> --input input.bin --output output.bin
      5. scp output.bin back

    Args:
        c7x_compute_cli: Name or path of c7x_compute CLI on target.
        dynmod_path: Path to lib0.out (local).
        weights_path: Path to weights.bin (local).
        input_path: Path to input tensor binary (local).
        output_path: Path to write output tensor binary (local).
                    If None, writes to input_path's directory.
        host: Remote hostname or IP.
        user: SSH user.
        remote_dir: Working directory on remote target.

    Returns:
        Dict with keys: output_path, module_handle, model_id, cycles
    """
    dynmod_path = Path(dynmod_path).resolve()
    weights_path = Path(weights_path).resolve()
    input_path = Path(input_path).resolve()
    if output_path is None:
        output_path = input_path.parent / "output_remote.bin"
    output_path = Path(output_path).resolve()

    remote = f"{user}@{host}"
    remote_dynmod = f"{remote_dir}/lib0.out"
    remote_weights = f"{remote_dir}/weights.bin"
    remote_input = f"{remote_dir}/input.bin"
    remote_output = f"{remote_dir}/output.bin"

    def ssh_cmd(cmd: str) -> str:
        result = subprocess.run(
            ["ssh", remote, cmd],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed: {cmd}\n{result.stderr}")
        return result.stdout.strip()

    def scp_to(local: Path, remote_path: str):
        subprocess.run(
            ["scp", str(local), f"{remote}:{remote_path}"],
            check=True,
            capture_output=True,
        )

    def scp_from(remote_path: str, local: Path):
        subprocess.run(
            ["scp", f"{remote}:{remote_path}", str(local)],
            check=True,
            capture_output=True,
        )

    logger.info(f"Running remote inference on {host}")

    # Create remote directory
    ssh_cmd(f"mkdir -p {remote_dir}")

    # Transfer files
    logger.info("  Transferring files to target...")
    scp_to(dynmod_path, remote_dynmod)
    scp_to(weights_path, remote_weights)
    scp_to(input_path, remote_input)

    # Load weights
    logger.info("  Loading weights...")
    out = ssh_cmd(f"{c7x_compute_cli} model-load {remote_weights}")
    logger.debug(f"  model-load: {out}")

    # Load module
    logger.info("  Loading dynamic module...")
    out = ssh_cmd(f"{c7x_compute_cli} load {remote_dynmod}")
    logger.debug(f"  load: {out}")

    # Run inference
    logger.info("  Running inference...")
    out = ssh_cmd(f"{c7x_compute_cli} infer {remote_input} {remote_output}")
    logger.debug(f"  infer: {out}")

    # Retrieve output
    logger.info("  Retrieving output...")
    scp_from(remote_output, output_path)

    logger.info(f"  Output written to: {output_path}")
    return {"output_path": output_path, "remote_log": out}


def run_dsp_dload(
    module_path: Union[str, Path],
    weights_path: Union[str, Path],
    input_tensors: List[np.ndarray],
    target_host: str = "am67a",
    target_user: str = "root",
    remote_dir: str = "/tmp/c7x_compute",
    c7x_compute_cli: str = "/usr/local/bin/c7x_compute",
    embedded_weights: bool = False,
    profile_layers: bool = False,
    profile: bool = False,
    multi_output: bool = False,
) -> tuple:
    """
    Run inference via c7x_compute DLOAD pipeline over SSH.

    Drives the full load → model-load → infer → unload sequence on a remote
    AM67A target using the c7x_compute CLI. The input tensor is sent as raw
    binary; the output is reconstructed from raw binary using shape/dtype
    metadata parsed from the CLI's stdout.

    Currently supports single-input models only (c7x_compute CLI limitation).

    Args:
        module_path: Path to lib0.out (DLOAD-compatible relocatable module).
        weights_path: Path to weights.bin (TVM model constants).
        input_tensors: List of numpy arrays (currently only first is used).
        target_host: Remote hostname or IP.
        target_user: SSH user.
        remote_dir: Working directory on remote target.
        c7x_compute_cli: Name or path of c7x_compute CLI on target.
        embedded_weights: If True, weights are embedded in lib0.out. Skips
            SCP of weights.bin, model-load step, and model-unload cleanup.
            The firmware auto-detects embedded weights at dyn-load time.
        profile: If True, use 'c7x_compute profile' instead of 'run'.
            This sets repeat=2 in the INFER request so the firmware runs
            inference twice: iteration 1 includes init, iteration 2 is
            steady-state.  Per-layer breakdowns are printed by the DSP.
        multi_output: If True, return all output tensors as a list.
            If False (default), return the first output tensor only
            for backward compatibility with single-output models.

    Returns:
        Tuple of (output, stdout_string, cycles).
        When multi_output=False: output is a single np.ndarray (first output).
        When multi_output=True: output is a list[np.ndarray] of all outputs.
        cycles is the inference cycle count from the DSP's TSC counter
        (0 if unavailable).

    Raises:
        RuntimeError: If any SSH/SCP command or CLI step fails.
        ValueError: If output metadata cannot be parsed from CLI stdout.
    """
    module_path = Path(module_path).resolve()
    weights_path = Path(weights_path).resolve()

    remote = f"{target_user}@{target_host}"
    remote_module = f"{remote_dir}/lib0.out"
    remote_weights = f"{remote_dir}/weights.bin"
    remote_input = f"{remote_dir}/input.bin"
    remote_output = f"{remote_dir}/output.bin"

    # Numpy dtype → CLI dtype string
    dtype_to_str = {
        np.dtype("float32"): "float32",
        np.dtype("float16"): "float16",
        np.dtype("int64"): "int64",
        np.dtype("int32"): "int32",
        np.dtype("int16"): "int16",
        np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8",
    }

    # CLI dtype code/bits → numpy dtype
    code_bits_to_dtype = {
        (2, 32): np.float32,
        (2, 16): np.float16,
        (0, 64): np.int64,
        (0, 32): np.int32,
        (0, 16): np.int16,
        (0, 8): np.int8,
        (1, 16): np.uint16,
        (1, 8): np.uint8,
    }

    def ssh_cmd(cmd: str) -> str:
        """Run command on remote target via SSH."""
        result = subprocess.run(
            ["ssh", remote, cmd],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (rc={result.returncode}): {cmd}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result.stdout.strip()

    def scp_to(local: Path, remote_path: str):
        subprocess.run(
            ["scp", "-q", str(local), f"{remote}:{remote_path}"],
            check=True,
            capture_output=True,
        )

    def scp_from(remote_path: str, local: Path):
        subprocess.run(
            ["scp", "-q", f"{remote}:{remote_path}", str(local)],
            check=True,
            capture_output=True,
        )

    logger.info(f"Running DLOAD inference on {target_host}")

    # Build semicolon-separated shape/dtype for multi-input support
    shape_parts = []
    dtype_parts = []
    for arr in input_tensors:
        arr = np.ascontiguousarray(arr)
        shape_parts.append(",".join(str(d) for d in arr.shape))
        dt = dtype_to_str.get(arr.dtype)
        if dt is None:
            raise ValueError(f"Unsupported input dtype: {arr.dtype}")
        dtype_parts.append(dt)
    shape_str = ";".join(shape_parts)
    dtype_str = ";".join(dtype_parts)

    # Create remote directory
    ssh_cmd(f"mkdir -p {remote_dir}")

    # Concatenate all input tensors into one binary file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        for arr in input_tensors:
            tmp.write(np.ascontiguousarray(arr).tobytes())
        local_input = Path(tmp.name)

    try:
        # Transfer files to target
        logger.info("  Transferring files to target...")
        scp_to(module_path, remote_module)
        if not embedded_weights:
            scp_to(weights_path, remote_weights)
        scp_to(local_input, remote_input)
    finally:
        local_input.unlink(missing_ok=True)

    # Run composite load+infer+unload via single SSH call.
    # Profile output (layer traces) goes to stderr; JSON goes to stdout.
    cli_verb = "profile" if profile else "run"
    logger.info(f"  Running composite load+infer+unload ({cli_verb})...")
    run_cmd = (
        f"{c7x_compute_cli} {cli_verb}"
        f" --module {remote_module}"
        f" --input {remote_input}"
        f" --output {remote_output}"
        f" --shape '{shape_str}'"
        f" --dtype '{dtype_str}'"
    )
    run_result = subprocess.run(
        ["ssh", remote, run_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if run_result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (rc={run_result.returncode}): {run_cmd}\n"
            f"stdout: {run_result.stdout}\nstderr: {run_result.stderr}"
        )
    out = run_result.stdout.strip()
    dsp_stderr = run_result.stderr.strip()
    logger.debug(f"  run stdout: {out}")
    if dsp_stderr:
        logger.debug(f"  run stderr: {dsp_stderr}")

    # Parse JSON from stdout. DSP printf text may precede the JSON
    # (from c7x_client_open/close info messages); split on the first '{'.
    json_start = out.find("{")
    if json_start < 0:
        raise RuntimeError(f"No JSON found in c7x_compute run output: {out}")
    dsp_text = out[:json_start].strip()
    # Append DSP profile output from stderr (layer traces, iteration headers)
    if dsp_stderr:
        dsp_text = (dsp_text + "\n" + dsp_stderr).strip() if dsp_text else dsp_stderr
    if dsp_text:
        logger.debug(f"  DSP printf: {dsp_text}")

    result = json.loads(out[json_start:])

    if result.get("status") != "ok":
        stage = result.get("stage", "unknown")
        error = result.get("error", "unknown error")
        raise RuntimeError(f"c7x_compute run failed at {stage}: {error}")

    cycles = result.get("cycles", 0)
    logger.info(f"  Inference cycles: {cycles:,}")

    # Retrieve output binary
    logger.info("  Retrieving output...")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        local_output = Path(tmp.name)

    try:
        scp_from(remote_output, local_output)
        raw_data = local_output.read_bytes()
    finally:
        local_output.unlink(missing_ok=True)

    # Parse all output tensors from JSON metadata + raw binary
    output_arrays = _parse_dsp_outputs(result, raw_data, code_bits_to_dtype)

    logger.info(f"  {len(output_arrays)} output tensor(s)")
    stdout_str = f"[run] {out}"
    if dsp_text:
        stdout_str = dsp_text + "\n" + stdout_str
    if multi_output:
        return output_arrays, stdout_str, cycles
    return output_arrays[0], stdout_str, cycles


def run_dsp_host(
    executable: Path,
    working_dir: Optional[Path] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Run DSP executable on host emulation and return output as numpy array(s).

    The executable writes output to a binary file (output.bin) with shape header.
    This function runs the executable and parses the output file.

    Args:
        executable: Path to the cg_dsp executable
        working_dir: Directory to run from (for output file).
                    If None, uses executable's directory.

    Returns:
        Single numpy array if one output, or list of numpy arrays if multiple outputs
    """
    executable = Path(executable).resolve()
    if working_dir is None:
        working_dir = executable.parent
    working_dir = Path(working_dir).resolve()

    logger.info(f"Running host emulation: {executable}")
    logger.debug(f"Working directory: {working_dir}")

    # Run executable
    result = subprocess.run(
        [str(executable)],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    # Log output for debugging
    if result.stdout:
        logger.debug(f"stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Host execution failed (code {result.returncode}): {result.stderr}")

    # Parse output file (new tensor file format)
    output_path = working_dir / OUTPUT_BIN_FILE
    if not output_path.exists():
        raise FileNotFoundError(
            f"Output file not found: {output_path}. "
            "The executable may not have written output correctly."
        )

    outputs = read_tensors_from_file(str(output_path))
    if not outputs:
        raise ValueError("No tensors found in output file")

    # Return single array for single output, list for multiple outputs
    if len(outputs) == 1:
        logger.info(f"Output shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
        return outputs[0]
    else:
        logger.info(f"Multiple outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            logger.info(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
        return outputs


def run_dsp_c66x(
    executable: Path,
    working_dir: Optional[Path] = None,
    timeout_ms: int = 60000,
) -> tuple:
    """
    Deploy and run DSP executable on C66x hardware, return output as numpy array(s).

    Uses run_on_c66x.sh script to load and execute the program via CCS Debug
    Server Scripting. CCS handles file I/O by intercepting standard C I/O calls.

    Args:
        executable: Path to the cg_dsp_c66x.out executable
        working_dir: Directory for output file.
                    If None, uses executable's directory.
        timeout_ms: Execution timeout in milliseconds (default: 60000)

    Returns:
        Tuple of (output, stdout string) where output is a single numpy array
        for single output models, or list of numpy arrays for multi-output models
    """
    executable = Path(executable).resolve()
    if working_dir is None:
        working_dir = executable.parent
    working_dir = Path(working_dir).resolve()

    # Find run script
    run_script = _DSP_RUNTIME_DIR / "scripts" / "run_on_c66x.sh"
    if not run_script.exists():
        raise FileNotFoundError(f"Run script not found: {run_script}")

    logger.info(f"Running on C66x hardware: {executable}")
    logger.debug(f"Working directory: {working_dir}")
    logger.debug(f"Timeout: {timeout_ms}ms")

    # Run on C66x
    cmd = [str(run_script), str(executable), "--timeout", str(timeout_ms)]

    result = subprocess.run(
        cmd,
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    # Log output for debugging
    if result.stdout:
        logger.debug(f"stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"C66x execution failed (code {result.returncode}): {result.stderr}")

    # Parse output file (new tensor file format; CCS handles file I/O)
    output_path = working_dir / OUTPUT_BIN_FILE
    if not output_path.exists():
        raise FileNotFoundError(
            f"Output file not found: {output_path}. "
            "CCS may not have intercepted file I/O correctly."
        )

    outputs = read_tensors_from_file(str(output_path))
    if not outputs:
        raise ValueError("No tensors found in output file")

    # Return single array for single output, list for multiple outputs
    if len(outputs) == 1:
        logger.info(f"Output shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
        return outputs[0], result.stdout
    else:
        logger.info(f"Multiple outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            logger.info(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
        return outputs, result.stdout


def compile_and_run_dsp(
    mod: tvm.IRModule,
    input_data: Union[np.ndarray, tuple],
    target_string: str = "c_static -mcpu=c66x",
    execution_mode: str = "c66x_host",
    build_type: str = "Release",
    timeout_ms: int = 60000,
    profile_layers: bool = False,
    profile: bool = False,
    relax_pipeline=None,
    fp_reassoc_off: bool = False,
    multi_output: bool = False,
) -> dict:
    """
    End-to-end: compile, build, run, and return results.

    This is the main entry point for DSP compilation and execution. It handles
    the complete workflow:
    1. Compile TVM IRModule to C code
    2. Build executable for host emulation and/or DSP hardware (in isolated temp dir)
    3. Write input data to input.bin
    4. Run inference and read output from output.bin
    5. Return results as numpy arrays

    Each execution uses a unique temporary build directory to avoid stale file
    issues when running multiple tests or models. This enables parallel test
    execution and prevents build artifacts from one model contaminating another.

    Args:
        mod: TVM IRModule to compile (should have parameters bound)
        input_data: Input data as numpy array or tuple of arrays.
        target_string: Target specification (default: "c_static -mcpu=c66x")
        execution_mode: "c66x_host", "c66x", "c7x_host", or "c7x_dload"
        build_type: Build type - "Release" (default) or "Debug".
        timeout_ms: Execution timeout for DSP hardware in milliseconds
        profile_layers: Enable trace buffer monitoring during DLOAD inference
        profile: Use c7x_compute profile (repeat=2) for init/steady-state
            separation.  Only applies to c7x_dload mode.
        fp_reassoc_off: Compile lib0.c with ``--fp_reassoc=off`` (c7x_dload
            only).  Prevents the cl7x -O2 optimizer from reordering float
            accumulations, fixing large logit divergence in LLMs.  ~27%
            cycle overhead.  No effect on other execution modes.

    Returns:
        Dictionary with results:
        {
            "c66x_host_result": np.ndarray,  # if mode is c66x_host
            "c66x_result": np.ndarray,       # if mode is c66x
            "c7x_dload_result": np.ndarray,  # if mode is c7x_dload
            "generated_dir": Path,           # path to generated code directory
        }

    Environment Variables:
        DSP_KEEP_TEMP: Set to "1" to preserve temp build directories for debugging
    """
    logger.info(f"compile_and_run_dsp: mode={execution_mode}, build_type={build_type}")

    results = {}

    # c7x_host mode always uses C7x code generation
    if execution_mode == "c7x_host" and "mcpu=c7x" not in target_string:
        target_string = "c_static -mcpu=c7x"

    # Convert input_data to list of arrays if needed
    input_tensors: List[np.ndarray]
    if isinstance(input_data, np.ndarray):
        input_tensors = [input_data]
    else:
        input_tensors = list(input_data)

    # Single workspace for both TVM compilation and native build
    with temporary_dsp_workspace() as workspace:
        # Step 1: Compile TVM IRModule into workspace root
        generated_dir = compile_for_dsp(
            mod, target_string, output_dir=workspace, relax_pipeline=relax_pipeline
        )
        results["generated_dir"] = generated_dir

        # Step 2: Build and run based on execution mode
        build_dir = workspace / f"build-{execution_mode}"

        if execution_mode == "c66x_host":
            logger.info("Building and running C66x host emulation...")
            host_exe = build_dsp_host(generated_dir, build_type=build_type, build_dir=build_dir)
            input_file = build_dir / INPUT_BIN_FILE
            write_tensors_to_file(input_tensors, str(input_file))
            results["c66x_host_result"] = run_dsp_host(host_exe)

        elif execution_mode == "c66x":
            logger.info("Building and running on C66x hardware...")
            try:
                c66x_exe = build_dsp_c66x(generated_dir, build_type=build_type, build_dir=build_dir)
                input_file = build_dir / INPUT_BIN_FILE
                write_tensors_to_file(input_tensors, str(input_file))
                c66x_output, c66x_stdout = run_dsp_c66x(c66x_exe, timeout_ms=timeout_ms)
                results["c66x_result"] = c66x_output
                results["c66x_stdout"] = c66x_stdout
            except Exception as e:
                error_msg = str(e)
                if "XDS110" in error_msg or "emulator" in error_msg.lower():
                    logger.warning(f"C66x hardware not available: {error_msg[:100]}...")
                    results["c66x_error"] = "Hardware not connected (XDS110 debug probe)"
                else:
                    raise

        elif execution_mode == "c7x_host":
            logger.info("Building and running C7x host emulation...")
            c7x_host_exe = build_dsp_c7x_host(
                generated_dir, build_type=build_type, build_dir=build_dir
            )
            input_file = build_dir / INPUT_BIN_FILE
            write_tensors_to_file(input_tensors, str(input_file))
            results["c7x_host_result"] = run_dsp_host(c7x_host_exe)

        elif execution_mode == "c7x_dload":
            logger.info("Building DLOAD module and running via c7x_compute...")
            weights_path = workspace / "weights.bin"
            module_path = build_dsp_dynmod(
                generated_dir,
                build_dir=build_dir,
                weights_file=weights_path,
                fp_reassoc_off=fp_reassoc_off,
            )
            c7x_dload_output, c7x_dload_stdout, c7x_dload_cycles = run_dsp_dload(
                module_path,
                weights_path,
                input_tensors,
                embedded_weights=True,
                profile_layers=profile_layers,
                profile=profile,
                multi_output=multi_output,
            )
            results["c7x_dload_result"] = c7x_dload_output
            results["c7x_dload_stdout"] = c7x_dload_stdout
            results["c7x_dload_cycles"] = c7x_dload_cycles

    return results


def _parse_dsp_outputs(
    result: dict,
    raw_data: bytes,
    code_bits_to_dtype: dict,
) -> List[np.ndarray]:
    """Extract output tensors from c7x_compute JSON result and raw binary.

    The output binary file contains all output tensors packed back-to-back
    in index order.  The JSON result contains per-tensor metadata (shape,
    dtype, data_size) that tells us how to slice the binary.

    Args:
        result: Parsed JSON dict from c7x_compute stdout.
        raw_data: Raw bytes read from the output .bin file.
        code_bits_to_dtype: Mapping from (dtype_code, dtype_bits) to
            numpy dtype used by the calling function.

    Returns:
        List of numpy arrays, one per output tensor.
    """
    outputs_meta = result.get("outputs", [])
    if not outputs_meta:
        raise ValueError("No outputs in c7x_compute run result")

    arrays = []
    offset = 0
    for meta in sorted(outputs_meta, key=lambda m: m["index"]):
        dtype_code = meta["dtype_code"]
        dtype_bits = meta["dtype_bits"]
        data_size = meta["data_size"]
        shape = tuple(meta["shape"])

        np_dtype = code_bits_to_dtype.get((dtype_code, dtype_bits))
        if np_dtype is None:
            raise ValueError(f"Unsupported output dtype: code={dtype_code}, bits={dtype_bits}")

        chunk = raw_data[offset : offset + data_size]
        if len(chunk) != data_size:
            raise ValueError(
                f"Output {meta['index']} size mismatch: "
                f"got {len(chunk)} bytes, expected {data_size}"
            )
        arr = np.frombuffer(chunk, dtype=np_dtype).reshape(shape).copy()
        arrays.append(arr)
        offset += data_size

    return arrays


def run_dsp_local(
    module_path: Union[str, Path],
    weights_path: Union[str, Path],
    input_tensors: List[np.ndarray],
    c7x_compute: str = "/usr/local/bin/c7x_compute",
    work_dir: str = "/tmp/c7x_local",
    embedded_weights: bool = True,
    multi_output: bool = False,
    timeout_s: int = 300,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Run c7x_compute as a local subprocess (for use on the AM67A ARM side).

    This is the board-side equivalent of run_dsp_dload: it calls
    c7x_compute directly via subprocess instead of over SSH/SCP.
    The KV cache and other large tensors are written to and read from
    a tmpfs directory (/tmp/c7x_local) at RAM speed (~5 ms for 11 MB),
    eliminating the SSH transfer bottleneck for interactive chat.

    Args:
        module_path: Path to lib0.out on the local filesystem.
        weights_path: Path to weights.bin on the local filesystem.
        input_tensors: List of numpy arrays to pass as model inputs.
        c7x_compute: Path to the c7x_compute CLI binary.
        work_dir: Directory for temporary input/output files (tmpfs).
        embedded_weights: If True, weights are embedded in lib0.out.
            Skips the model-load step (firmware auto-detects).
        multi_output: If True, return all output tensors as a list.
            If False (default), return the first tensor only.
        timeout_s: Subprocess timeout in seconds.

    Returns:
        When multi_output=False: single np.ndarray (first output).
        When multi_output=True: list[np.ndarray] of all outputs.

    Raises:
        RuntimeError: If c7x_compute fails or times out.
        FileNotFoundError: If module_path does not exist.
    """
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"DLOAD module not found: {module_path}")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    input_bin = work / "input.bin"
    output_bin = work / "output.bin"

    # Dtype → CLI string (same mapping as run_dsp_dload)
    dtype_to_str = {
        np.dtype("float32"): "float32",
        np.dtype("float16"): "float16",
        np.dtype("int64"): "int64",
        np.dtype("int32"): "int32",
        np.dtype("int16"): "int16",
        np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8",
    }
    code_bits_to_dtype = {
        (2, 32): np.float32,
        (2, 16): np.float16,
        (0, 64): np.int64,
        (0, 32): np.int32,
        (0, 8): np.int8,
        (1, 8): np.uint8,
    }

    # Build shape/dtype strings for multi-input
    shape_parts, dtype_parts = [], []
    for arr in input_tensors:
        arr = np.ascontiguousarray(arr)
        shape_parts.append(",".join(str(d) for d in arr.shape))
        dt = dtype_to_str.get(arr.dtype)
        if dt is None:
            raise ValueError(f"Unsupported input dtype: {arr.dtype}")
        dtype_parts.append(dt)

    # Write all input tensors to a single binary file
    with open(input_bin, "wb") as f:
        for arr in input_tensors:
            f.write(np.ascontiguousarray(arr).tobytes())

    cmd = [
        c7x_compute,
        "run",
        "--module",
        str(module_path),
        "--input",
        str(input_bin),
        "--output",
        str(output_bin),
        "--shape",
        ";".join(shape_parts),
        "--dtype",
        ";".join(dtype_parts),
    ]

    logger.info(
        f"run_dsp_local: {module_path.name}, {len(input_tensors)} input(s), work_dir={work_dir}"
    )

    result_proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    if result_proc.returncode != 0:
        raise RuntimeError(
            f"c7x_compute failed (rc={result_proc.returncode}):\n"
            f"stdout: {result_proc.stdout}\nstderr: {result_proc.stderr}"
        )

    out = result_proc.stdout.strip()
    json_start = out.find("{")
    if json_start < 0:
        raise RuntimeError(f"No JSON in c7x_compute output: {out}")

    result_json = json.loads(out[json_start:])
    if result_json.get("status") != "ok":
        raise RuntimeError(f"c7x_compute run error: {result_json.get('error', 'unknown')}")

    cycles = result_json.get("cycles", 0)
    logger.info(f"  Cycles: {cycles:,}")

    raw_data = output_bin.read_bytes()
    output_arrays = _parse_dsp_outputs(result_json, raw_data, code_bits_to_dtype)

    logger.info(f"  {len(output_arrays)} output tensor(s)")
    if multi_output:
        return output_arrays, result_proc.stdout, cycles
    return output_arrays[0], result_proc.stdout, cycles


def compare_results(
    results: dict,
    reference: np.ndarray,
    reference_name: str = "reference",
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """
    Compare DSP results against a reference.

    Args:
        results: Dictionary from compile_and_run_dsp()
        reference: Reference numpy array (e.g., from PyTorch)
        reference_name: Name of reference for logging (e.g., "PyTorch")
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        Dictionary with comparison metrics, e.g.:
        {
            "c66x_host_vs_ref_max_diff": float,
            "c66x_host_vs_ref_passed": bool,
        }
    """
    comparison = {}

    if "c66x_host_result" in results:
        c66x_host = results["c66x_host_result"]
        diff = np.max(np.abs(c66x_host - reference))
        passed = np.allclose(c66x_host, reference, atol=atol, rtol=rtol)
        comparison["c66x_host_vs_ref_max_diff"] = float(diff)
        comparison["c66x_host_vs_ref_passed"] = passed
        logger.info(f"C66x_host vs {reference_name}: max diff = {diff:.2e}, passed = {passed}")

    if "c66x_result" in results:
        c66x = results["c66x_result"]
        diff = np.max(np.abs(c66x - reference))
        passed = np.allclose(c66x, reference, atol=atol, rtol=rtol)
        comparison["c66x_vs_ref_max_diff"] = float(diff)
        comparison["c66x_vs_ref_passed"] = passed
        logger.info(f"C66x vs {reference_name}: max diff = {diff:.2e}, passed = {passed}")

    if "c7x_host_result" in results:
        c7x_host = results["c7x_host_result"]
        diff = np.max(np.abs(c7x_host - reference))
        passed = np.allclose(c7x_host, reference, atol=atol, rtol=rtol)
        comparison["c7x_host_vs_ref_max_diff"] = float(diff)
        comparison["c7x_host_vs_ref_passed"] = passed
        logger.info(f"C7x_host vs {reference_name}: max diff = {diff:.2e}, passed = {passed}")

    if "c7x_dload_result" in results:
        c7x_dload = results["c7x_dload_result"]
        diff = np.max(np.abs(c7x_dload - reference))
        passed = np.allclose(c7x_dload, reference, atol=atol, rtol=rtol)
        comparison["c7x_dload_vs_ref_max_diff"] = float(diff)
        comparison["c7x_dload_vs_ref_passed"] = passed
        logger.info(f"C7x_dload vs {reference_name}: max diff = {diff:.2e}, passed = {passed}")

    return comparison
