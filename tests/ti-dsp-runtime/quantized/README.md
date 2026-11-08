# Quantized Model Tests

End-to-end tests for INT8-quantized TorchVision and YOLO models on the
TVM `c_static_lib` backend, with and without MMALIB offload.

See [Quantized Model Sweep](../../../docs-c7x/contributor-guide/testing/quantized-sweep.md)
in the docs site for the full sweep status and known issues.

`calib_images/` holds the real images the per-model classifier tests
calibrate on; see its `ATTRIBUTION.md` for sources and licenses.
