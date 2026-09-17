#!/usr/bin/env python
"""tvm.relax.backend must not import the TIDL backend eagerly.

The TIDL backend drags in the C7x/TIDL offload stack; importing
``tvm.relax.backend`` should not pay that cost unless ``tidl`` is actually
used.
"""
import sys

import pytest
import tvm


def test_backend_does_not_import_tidl_eagerly():
    import tvm.relax.backend as relax_backend

    assert "tvm.relax.backend.tidl" not in sys.modules

    # Attribute access lazily imports and returns the submodule.
    tidl = relax_backend.tidl
    assert tidl is not None
    assert "tvm.relax.backend.tidl" in sys.modules


def test_backend_lazy_import_unknown_name():
    import tvm.relax.backend as relax_backend

    with pytest.raises(AttributeError):
        relax_backend.__getattr__("does_not_exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
