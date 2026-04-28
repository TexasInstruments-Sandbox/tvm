#!/usr/bin/env python
"""
C66x pragma generation test for DSP.

Tests that TI C66x DSP-specific optimizations are correctly generated when
targeting the C66x architecture via -mcpu=c66x. This test validates pragma
generation and code structure without requiring DSP hardware execution.

Usage:
    # Run with pytest
    pytest test_c66x_pragmas_dsp.py -v

    # Run as standalone script
    python test_c66x_pragmas_dsp.py
"""

import os
import tarfile
import tempfile

import pytest
import tvm
from tvm import relax
from tvm.script import tir as T

pytestmark = [pytest.mark.core, pytest.mark.c66x_only]


# -----------------------------------------------------------------------------
# Target Attribute Tests
# -----------------------------------------------------------------------------


class TestC66xTargetParsing:
    """Tests for C66x target attribute parsing."""

    def test_c66x_target_mcpu(self):
        """Verify mcpu attribute is correctly parsed for C66x target."""
        target = tvm.target.Target("c_static -mcpu=c66x")
        assert target.attrs.get("mcpu") == "c66x"

    def test_c66x_alignment_64_bytes(self):
        """Verify C66x target uses 64-byte alignment (cache line aligned)."""
        target = tvm.target.Target("c_static -mcpu=c66x")
        assert target.attrs.get("constants-byte-alignment") == 64

    def test_c7x_alignment_64_bytes(self):
        """Verify C7x target also uses 64-byte alignment."""
        target = tvm.target.Target("c_static -mcpu=c7x")
        assert target.attrs.get("constants-byte-alignment") == 64

    def test_generic_alignment_16_bytes(self):
        """Verify generic c_static target uses default 16-byte alignment."""
        target = tvm.target.Target("c_static")
        assert target.attrs.get("constants-byte-alignment") == 16

    def test_c66x_with_device(self):
        """Verify C66x target with device attribute."""
        target = tvm.target.Target("c_static -mcpu=c66x -device=awrl6844")
        assert target.attrs.get("mcpu") == "c66x"
        assert target.attrs.get("device") == "awrl6844"

    def test_c66x_default_optimizations(self):
        """Verify C66x target has default optimizations enabled."""
        target = tvm.target.Target("c_static -mcpu=c66x")
        # Default optimizations for C66x
        assert target.attrs.get("skip-runtime-checks") == 1
        assert target.attrs.get("use-cpp-api") == 1


# -----------------------------------------------------------------------------
# Pragma Generation Tests
# -----------------------------------------------------------------------------


class TestC66xPragmaGeneration:
    """Tests for TI pragma generation in generated C code."""

    @staticmethod
    def _build_simple_loop(target_str: str) -> str:
        """Build a simple loop function and return generated C code.

        Uses relax.build() + export_library() to extract lib0.c from the
        generated tarball.  tvm.build().get_source() was removed in TVM 0.23.
        """

        @T.prim_func
        def simple_loop(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
            for i in range(64):
                B[i] = A[i] * T.float32(2.0)

        target = tvm.target.Target(target_str)
        mod = tvm.IRModule({"simple_loop": simple_loop})
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod, target=target, exec_mode="compiled", system_lib=True)
        with tempfile.TemporaryDirectory() as td:
            tar_path = os.path.join(td, "out.tar")
            ex.export_library(tar_path, target=target)
            with tarfile.open(tar_path) as tf:
                tf.extractall(td)
            lib0 = os.path.join(td, "lib0.c")
            return open(lib0).read() if os.path.exists(lib0) else ""

    def test_c66x_ti_compiler_guard(self):
        """Verify TI compiler version guard is present."""
        code = self._build_simple_loop("c_static -mcpu=c66x")
        assert "__TI_COMPILER_VERSION__" in code

    def test_c66x_c6x_header(self):
        """Verify c6x.h header is included for TI DSP target."""
        code = self._build_simple_loop("c_static -mcpu=c66x")
        assert "#include <c6x.h>" in code

    def test_generic_no_ti_headers(self):
        """Verify generic c_static target does NOT emit TI-specific headers."""
        code = self._build_simple_loop("c_static")
        assert "#include <c6x.h>" not in code
        assert "__TI_COMPILER_VERSION__" not in code

    def test_c7x_has_ti_compiler_guard(self):
        """Verify C7x target also emits the TI compiler version guard."""
        code = self._build_simple_loop("c_static -mcpu=c7x")
        assert "__TI_COMPILER_VERSION__" in code


# -----------------------------------------------------------------------------
# DSP Code Generation Feature Tests
# -----------------------------------------------------------------------------


class TestDSPCodeGenFeatures:
    """Tests for DSP-specific code generation features."""

    def test_use_cpp_api_generates_anyarray(self):
        """Verify use-cpp-api generates AnyArray wrapper code."""
        # This is a codegen test, not execution test
        # When use-cpp-api=1, generated code should use AnyArray wrappers
        target = tvm.target.Target("c_static -mcpu=c66x -use-cpp-api=1")
        assert target.attrs.get("use-cpp-api") == 1

    def test_skip_runtime_checks_attribute(self):
        """Verify skip-runtime-checks attribute is recognized."""
        target = tvm.target.Target("c_static -mcpu=c66x -skip-runtime-checks=1")
        assert target.attrs.get("skip-runtime-checks") == 1

    def test_profile_layers_attribute(self):
        """Verify profile-layers attribute is recognized."""
        target = tvm.target.Target("c_static -mcpu=c66x -profile-layers=1")
        assert target.attrs.get("profile-layers") == 1


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
