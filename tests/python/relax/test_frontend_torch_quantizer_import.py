#!/usr/bin/env python
"""The torch frontend must not import torchao at package import time.

C7xMMAQuantizer depends on torchao, an optional dependency.  Importing
``tvm.relax.frontend.torch`` (and thus ``from_exported_program`` etc.)
must succeed without torchao installed, matching upstream TVM which only
requires ``torch``.  The quantizer class is resolved lazily on attribute
access instead.
"""
import sys

import pytest
import tvm

pytest.importorskip("torch")


def test_torch_frontend_does_not_import_torchao_eagerly():
    import tvm.relax.frontend.torch as torch_frontend

    # Importing the frontend must not have pulled in the quantizer module
    # (which imports torchao at its top level).
    assert "tvm.relax.frontend.torch.c7x_mma_quantizer" not in sys.modules

    # Accessing the symbol lazily imports it and returns the class.
    quantizer_cls = torch_frontend.C7xMMAQuantizer
    assert quantizer_cls is not None
    assert "tvm.relax.frontend.torch.c7x_mma_quantizer" in sys.modules


def test_torch_frontend_lazy_import_unknown_name():
    import tvm.relax.frontend.torch as torch_frontend

    with pytest.raises(AttributeError):
        torch_frontend.__getattr__("does_not_exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
