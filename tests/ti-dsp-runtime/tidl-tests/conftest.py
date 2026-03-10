"""Pytest configuration for TIDL tests.

This file is intentionally minimal.  TIDL tests do not use the
``--dsp-mode`` fixture from ``dsp-tests/conftest.py`` because they
manage their own build and execution flows (stub bridge via c7x_host,
or full hardware via run_dsp_dload).

The file must exist so pytest treats ``tidl-tests/`` as a separate
test root and does not inherit the ``dsp-tests/`` conftest, which
would require ``--dsp-mode`` for every invocation.
"""
