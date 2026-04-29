"""
Utility functions for TVM CStatic testing framework.

This module provides utilities for processing Relax IR modules and compiling/running
them on different targets, with special support for CStatic target compilation and
execution through generated C++ code.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import tvm
from tvm import relax
from tvm.contrib import tar
from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default cpp directory path (relative to this file's location)
# This allows tvm_utils to work from any working directory
_MODULE_DIR = Path(__file__).parent.resolve()
_DEFAULT_CPP_DIR = _MODULE_DIR / "cpp"


@contextmanager
def temporary_cpp_workspace(base_cpp_dir: str | Path | None = None, cleanup: bool = True):
    """
    Create a temporary workspace for CStatic compilation and execution.

    This context manager creates a unique temporary directory for each test execution,
    enabling parallel test execution without race conditions. The workspace is populated
    with template files from the base cpp directory and is automatically cleaned up
    after use (unless disabled for debugging).

    Args:
        base_cpp_dir: Path to template cpp directory containing CMakeLists.txt and main.cpp.
                     If None (default), uses the cpp directory relative to this module.
        cleanup: Whether to cleanup temp directory after use (default: True)
                Can be overridden by setting CSTATIC_KEEP_TEMP=1 environment variable

    Yields:
        Path object pointing to the temporary workspace directory

    Example:
        >>> with temporary_cpp_workspace() as workspace:
        ...     # workspace is a unique temp directory with template files
        ...     build_dir = workspace / "build"
        ...     build_dir.mkdir()
        ...     # ... perform build and execution ...
        ...     # workspace is automatically cleaned up after the block

    Environment Variables:
        CSTATIC_KEEP_TEMP: Set to "1" to disable cleanup for debugging purposes
    """
    # Create unique temp directory with descriptive prefix
    temp_dir = tempfile.mkdtemp(prefix="cpp_cstatic_", dir=None)
    temp_path = Path(temp_dir)

    try:
        # Copy template files from base cpp directory
        # Use module-relative default if not specified
        if base_cpp_dir is None:
            base_path = _DEFAULT_CPP_DIR
        else:
            base_path = Path(base_cpp_dir)

        # Verify template files exist
        cmake_file = base_path / "CMakeLists.txt"
        main_file = base_path / "main.cpp"

        if not cmake_file.exists():
            raise FileNotFoundError(f"Template file not found: {cmake_file}")
        if not main_file.exists():
            raise FileNotFoundError(f"Template file not found: {main_file}")

        # Copy template files to temp workspace
        shutil.copy2(cmake_file, temp_path / "CMakeLists.txt")
        shutil.copy2(main_file, temp_path / "main.cpp")

        yield temp_path

    finally:
        # Cleanup unless disabled via parameter or environment variable
        should_cleanup = cleanup and os.getenv("CSTATIC_KEEP_TEMP") != "1"
        if should_cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            # If cleanup is disabled, print the path for debugging
            print(f"Keeping temporary workspace for debugging: {temp_dir}")


def model_returns_tuple(mod: tvm.IRModule, func_name: str = "main") -> bool:
    """
    Determine if the Relax model returns a tuple by inspecting the return type.

    Args:
        mod: TVM Relax IRModule to inspect
        func_name: Name of the function to inspect (default: "main")

    Returns:
        True if the specified function returns a tuple, False if it returns a single tensor
    """
    # Check if the function exists in the module
    if func_name not in mod:
        raise ValueError(
            f"Function '{func_name}' not found in IRModule. Available functions: {list(mod.functions.keys())}"
        )

    target_func = mod[func_name]

    # Handle different function types (Relax vs TIR)
    if hasattr(target_func, "ret_struct_info"):
        # Relax function - use ret_struct_info
        ret_info = target_func.ret_struct_info
        # Check if return struct info indicates tuple/array
        return (
            hasattr(ret_info, "fields")
            or (hasattr(ret_info, "__class__") and "Tuple" in str(ret_info.__class__))
            or (hasattr(ret_info, "name") and "Array" in str(ret_info))
        )
    elif hasattr(target_func, "ret_type"):
        # TIR function - use ret_type
        ret_type = target_func.ret_type
        return isinstance(ret_type, (tvm.ir.TupleType, tvm.ir.type.TupleType)) or (
            hasattr(ret_type, "name") and "Array" in str(ret_type)
        )
    else:
        # Fallback: assume single return for unknown function types
        return False


def process_relax(mod: tvm.IRModule) -> tvm.IRModule:
    """
    Process a Relax IR module by detaching parameters and binding them to the main function.

    This function separates model parameters from the main function inputs and binds them
    as constants, which is useful for models where weights are known at compile time.

    Args:
        mod: The TVM IRModule containing Relax functions

    Returns:
        Modified IRModule with parameters bound to the main function
    """
    mod, params = relax.frontend.detach_params(mod)

    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)  # pyright: ignore[reportArgumentType]
    # mod.show(show_meta=False)

    return mod


def compile_and_run_on_target(
    target_string: str,
    mod: tvm.IRModule,
    input: Union[np.ndarray, Tuple[np.ndarray, ...]],
    verbose_output: bool = False,
    entry_func_name: str = "main",
) -> Union[np.ndarray, list[np.ndarray]]:
    """
    Compile a Relax module for a target and run it with given input.

    This function handles two execution paths:
    - c_static target: Exports to C++, compiles with CMake/make, runs binary,
      and loads outputs from NPZ file
    - Other targets: Uses TVM VirtualMachine for execution

    The function automatically detects whether the model returns a tuple by inspecting
    the Relax IR module's return type.

    Args:
        target_string: Target specification (e.g., "c_static", "llvm")
        mod: TVM IRModule containing the Relax function to execute
        input: Input data as numpy array or tuple of numpy arrays for multiple inputs
        verbose_output: Enable verbose compilation output with instrumentation
        entry_func_name: Name of the function to execute (default: "main")

    Returns:
        - Single np.ndarray if model has one output
        - List[np.ndarray] if model has multiple outputs

        This behavior is consistent across both c_static and VM execution paths.

    Note:
        For CStatic targets, this function:
        1. Creates a unique temporary workspace (enables parallel test execution)
        2. Copies template files (CMakeLists.txt, main.cpp) to workspace
        3. Exports the compiled module to a tar file
        4. Extracts generated files (lib0.c, devc.c) to workspace
        5. Builds the C++ executable with CMake/make
        6. Executes the binary which writes outputs to outputs.npz
        7. Loads all outputs from the NPZ file
        8. Automatically cleans up the workspace (unless CSTATIC_KEEP_TEMP=1)

        For other targets, it uses TVM's VirtualMachine directly.

        The temporary workspace approach allows multiple tests to run in parallel
        without interfering with each other.
    """

    logger.debug(f"Compiling for target: {target_string}")

    # For c_static host builds, disable use-cpp-api if not explicitly set.
    # The use-cpp-api=1 default generates code using tvm::dsp::vm::AnyArray
    # which is only compatible with the DSP runtime, not the standard TVM runtime.
    if target_string.startswith("c_static") and "-use-cpp-api" not in target_string:
        target_string = target_string + " -use-cpp-api=0"
        logger.debug(f"Disabled use-cpp-api for host build: {target_string}")

    target = tvm.target.Target(target_string)

    # Automatically detect if model returns tuple
    returns_tuple = model_returns_tuple(mod, entry_func_name)
    logger.debug(f"Model returns tuple: {returns_tuple}")

    # Compile the Relax module with optional instrumentation for debugging
    # Use cpu_generic pipeline which includes FuseOps+FuseTIR for operator
    # fusion, reducing per-layer function call overhead.
    from tvm.relax.backend.cpu_generic.pipeline import get_default_pipeline

    logger.debug("Building Relax module...")
    pipeline = get_default_pipeline(target)
    with target:
        instruments = [PrintBeforeAll(), PrintAfterAll()] if verbose_output else []
        with tvm.transform.PassContext(opt_level=3, instruments=instruments):
            executable = relax.build(
                mod, target, exec_mode="compiled", system_lib=True, relax_pipeline=pipeline
            )
    logger.debug("Relax module build complete")

    # for mod in executable.mod.imported_modules:
    #   print(mod.type_key)
    #   print(mod.get_source())

    if target_string.startswith("c_static"):
        # CStatic target: Export to C++, compile and run as native binary
        logger.debug("Using C Static backend - creating temporary workspace")
        # Use temporary workspace for parallel test execution
        with temporary_cpp_workspace() as cpp_dir:
            logger.debug(f"Workspace created at: {cpp_dir}")

            # Export compiled module to tar file in temp workspace
            tar_path = cpp_dir / "model_library.tar"
            logger.debug(f"Exporting library to: {tar_path}")
            executable.export_library(str(tar_path), target=target)

            # Extract generated files (lib0.c, devc.c) to temp workspace
            logger.debug("Extracting generated C files...")
            tar.untar(str(tar_path), str(cpp_dir))

            # Create build directory
            build_dir = cpp_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Build directory created: {build_dir}")

            # Build the C++ executable with conditional compilation flags using CMake
            log_path = cpp_dir / "cmake.log"
            logger.debug("Configuring and building C++ binary with CMake...")
            with open(log_path, "w+") as f:
                # Configure CMake with conditional flags
                cmake_cmd = ["cmake", ".."]
                # Pass TVM_HOME explicitly so it works in isolated temp dirs
                tvm_home = os.environ.get("TVM_HOME", "")
                if tvm_home:
                    cmake_cmd.append(f"-DTVM_HOME={tvm_home}")
                if returns_tuple:
                    cmake_cmd.append("-DMODEL_RETURNS_TUPLE=ON")
                cmake_cmd.append(f"-DMODEL_ENTRY_FUNCTION={entry_func_name}")
                num_inputs = len(input) if isinstance(input, tuple) else 1
                cmake_cmd.append(f"-DMODEL_NUM_INPUTS={num_inputs}")

                logger.debug(f"Running CMake: {' '.join(cmake_cmd)}")
                result = subprocess.run(cmake_cmd, cwd=str(build_dir), stdout=f, stderr=f, check=False)
                if result.returncode != 0:
                    f.seek(0)
                    cmake_output = f.read()
                    raise RuntimeError(
                        f"CMake configuration failed (exit code {result.returncode}).\n"
                        f"Command: {' '.join(cmake_cmd)}\n"
                        f"Log:\n{cmake_output}"
                    )

                # Build with make
                logger.debug("Building cg_static target...")
                result = subprocess.run(
                    ["make", "cg_static"], cwd=str(build_dir), stdout=f, stderr=f, check=False
                )
                if result.returncode != 0:
                    f.seek(0)
                    build_output = f.read()
                    raise RuntimeError(
                        f"Build failed (exit code {result.returncode}).\n"
                        f"Log:\n{build_output}"
                    )
                logger.debug("Copying binary to workspace...")
                subprocess.run(
                    ["make", "copy_binary"], cwd=str(build_dir), stdout=f, stderr=f, check=True
                )

            logger.debug("Build complete")

            # Write the input(s) to file(s) so they can be read from the C++ application
            if isinstance(input, tuple):
                # Multiple inputs: save each array with numbered suffix
                logger.debug(f"Writing {len(input)} input arrays to workspace")
                for i, inp in enumerate(input):
                    np.save(str(cpp_dir / f"input_{i}.npy"), inp)
            else:
                # Single input: keep original behavior
                logger.debug("Writing input array to workspace")
                np.save(str(cpp_dir / "input_0.npy"), input)

            # Check if the binary exists before running it
            binary_path = cpp_dir / "cg_static"
            if not binary_path.exists():
                raise FileNotFoundError(
                    f"CStatic binary '{binary_path}' not found. "
                    f"Build process may have failed. Check {log_path} for details."
                )

            # Run the binary (outputs written to NPZ file)
            logger.debug("Executing C Static binary...")
            cmd_result = subprocess.run(
                ["./cg_static"],
                cwd=str(cpp_dir),
                capture_output=True,
                text=True,
                check=False,  # Don't raise immediately, we want to capture stderr
            )
            logger.debug(f"Binary execution completed with exit code {cmd_result.returncode}")

            # Check if the binary failed and include stderr in the exception
            if cmd_result.returncode != 0:
                error_msg = f"C Static binary execution failed (exit code {cmd_result.returncode})"
                if cmd_result.stderr:
                    # Strip "Error: " prefix if present for cleaner error messages
                    stderr_msg = cmd_result.stderr.strip()
                    if stderr_msg.startswith("Error: "):
                        stderr_msg = stderr_msg[7:]  # Remove "Error: " prefix
                    error_msg = stderr_msg  # Use the detailed error message from stderr
                raise RuntimeError(error_msg)

            # Load NPZ file containing all outputs
            outputs_npz_path = cpp_dir / "outputs.npz"
            if not outputs_npz_path.exists():
                raise FileNotFoundError(
                    f"Output file {outputs_npz_path} not found. Binary may have failed silently."
                )

            # Load all outputs from NPZ
            logger.debug(f"Loading outputs from: {outputs_npz_path}")
            outputs_dict = np.load(str(outputs_npz_path))

            # Extract outputs in order: output_0, output_1, ...
            output_arrays = []
            i = 0
            while f"output_{i}" in outputs_dict:
                output_arrays.append(outputs_dict[f"output_{i}"])
                i += 1

            logger.debug(f"Loaded {len(output_arrays)} output arrays")

            # Validate that we got at least one output
            if len(output_arrays) == 0:
                raise RuntimeError("No outputs found in outputs.npz")

            # Return single array or list to match VM behavior
            if len(output_arrays) == 1:
                cpp_result = output_arrays[0]
                logger.debug(f"Returning single output with shape: {cpp_result.shape}")
            else:
                cpp_result = output_arrays
                shapes = [arr.shape for arr in cpp_result]
                logger.debug(f"Returning {len(cpp_result)} outputs with shapes: {shapes}")

            # Result is captured before workspace cleanup
            return cpp_result

    else:
        # Other targets: Use TVM VirtualMachine for execution
        logger.debug(f"Using TVM VirtualMachine for target: {target_string}")
        dev = tvm.cpu()
        vm = relax.VirtualMachine(executable, dev)

        # Create tvm_data for single input or tuple of inputs
        if isinstance(input, tuple):
            logger.debug(f"Creating {len(input)} input arrays for VM")
            tvm_data = tuple(tvm.runtime.tensor(inp, device=dev) for inp in input)
        else:
            logger.debug("Creating single input array for VM")
            tvm_data = tvm.runtime.tensor(input, device=dev)

        # Execute the function - may return NDArray or Array<NDArray>
        logger.debug(f"Executing VM function: {entry_func_name}")
        # Flatten tuple arguments when calling the function
        if isinstance(tvm_data, tuple):
            tvm_result: tvm.runtime.ndarray.NDArray | tvm.ir.container.Array = vm[entry_func_name](
                *tvm_data
            )
        else:
            tvm_result: tvm.runtime.ndarray.NDArray | tvm.ir.container.Array = vm[entry_func_name](
                tvm_data
            )
        logger.debug("VM execution complete")

        # Handle both single output (NDArray) and multiple outputs (Array<NDArray>)
        if isinstance(tvm_result, tvm.ir.container.Array):
            # Multiple outputs - return all as list
            result = [arr.numpy() for arr in tvm_result]  # pyright: ignore
            logger.debug(f"Returning {len(result)} outputs from VM")
            return result
        else:
            # Single output
            result = tvm_result.numpy()
            logger.debug(f"Returning single output from VM with shape: {result.shape}")
            return result
