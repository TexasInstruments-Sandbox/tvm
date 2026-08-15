# Examples: YOLO26 & ResNet-18

Standalone scripts for running models on BeagleY-AI. Located at
`tests/ti-dsp-runtime/examples/`.

## YOLO26 Object Detection

### Description

`run_yolo26_detection.py` + `yolo26_board_runner.py` run YOLO26n object
detection on the real JPEGs in `tests/cstatic/test_images/`, on a BeagleY-AI
(or AM67A) board, using the Python offload API
(`tvm.contrib.c7x.C7xVirtualMachine`).

Two files because the work is inherently split across two machines:

- `run_yolo26_detection.py` runs on the **dev host** — it needs the full
  PyTorch + TVM + ultralytics stack to quantize and compile the model, and
  it's the "main" script you actually invoke.
- `yolo26_board_runner.py` runs **on the board** — `run_yolo26_detection.py`
  deploys it there over SSH and invokes it remotely. It needs only `numpy`
  and the deployed `libc7x_arm_runtime.so`, since `C7xVirtualMachine` talks
  to the DSP over the board's local rpmsg IPC channel and cannot be used
  remotely.

### Prerequisites

- Firmware and `libc7x_arm_runtime.so` already deployed on the target board
  — see [Deploying Firmware](deploying-firmware.md) (`./build.sh deploy`).
  Same prerequisite `quantized/test_quantized_yolo.py`'s
  `--dsp-mode=c7x_dload` runs already have.
- `TI_CGT_C7000_PATH` set (compiling for C7x).
- Passwordless SSH as `root` to the board.
- `numpy` in the board's system Python 3 (`yolo26_board_runner.py` needs it;
  it is not preinstalled on a stock BeagleY-AI image). Install with
  `apt-get install python3-numpy` on the board -- if `apt` can't resolve its
  mirrors, set `http_proxy`/`https_proxy` first.

### Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=<path to>/ti-cgt-c7000_5.0.1.LTS
python examples/run_yolo26_detection.py
```

Run with `--help` for the full option list (image selection, confidence
threshold, board target, `--compile-only`/`--inference-only`, build/output
directories). `yolo26_board_runner.py --help` documents the board-side
script's own flags, for running it by hand.

Each run prints per-image DSP cycles and wall-clock latency alongside the
surviving detections, and saves annotated JPEGs to
`examples/yolo26_detections/`.

### Visualizing the MMALIB offload

```bash
python examples/run_yolo26_detection.py --visualize yolo26n_mmalib.html
```

Generates an interactive HTML graph of which ops get offloaded to MMALIB vs.
run as TVM-generated C (`tvm.contrib.c7x.visualize.visualize_compile`), by
recompiling the model up through the Relax passes independently of the rest
of the pipeline -- no board required. `--visualize` works with
`--compile-only`, and combines with any other flag, since it doesn't depend
on deployment or inference. Without `--profile-layers` (see below) it has no
per-layer cycle counts to overlay, so the graph is structural only.

### Per-layer cycle profiling

```bash
python examples/run_yolo26_detection.py --profile-layers
python examples/run_yolo26_detection.py --profile-layers --visualize yolo26n_mmalib.html
```

`--profile-layers` compiles the model with per-layer DSP cycle counting and
prints a `===== TVM Layer Profile =====` block after each image runs on the
board. Combine with `--visualize` to overlay those cycle counts on the
offload graph instead of a structural-only one. Requires a real board run,
so it does not combine with `--compile-only`.

## ResNet-18 Classification (C++ API)

### Description

`run_resnet18_classification.py` + `resnet18_board_runner.cpp` classify a
real JPEG from `tests/cstatic/test_images/` on a BeagleY-AI (or AM67A)
board, using the C++ offload API (`c7x::Module`) -- the C++ counterpart to
the YOLO26 example above.

Same two-machine split as YOLO26, but with a different division of labor:

- `run_resnet18_classification.py` runs on the **dev host** -- quantizes
  and compiles ResNet-18 (calibrated on the real input image, not random
  noise, so the result is a plausible label rather than just internally
  consistent), preprocesses the image, and cross-compiles
  `resnet18_board_runner.cpp` for aarch64 with a single `g++` invocation
  (no CMake target).
- `resnet18_board_runner.cpp` runs **on the board** -- unlike the Python
  board runner, it also does the postprocessing itself (argmax + ImageNet
  label lookup) and prints the human-readable top-5 result directly; no
  output tensor is shipped back to the host.

`resnet18_board_runner.cpp` is built on `common/c7x_infer.h`, a small
header-only library shared by every C++ example: it hides the
`c7x::Module`/`DLTensor` boilerplate and provides raw-tensor-file input
and CLI dtype parsing. A future C++ example (e.g. object detection) would
`#include` the same header and do its own task-specific decoding, rather
than duplicating it -- see the header's own comments for what is and
isn't in scope for it.

### Prerequisites

Same as YOLO26 above (firmware + `libc7x_arm_runtime.so` deployed,
`TI_CGT_C7000_PATH`, passwordless SSH), plus an `aarch64-linux-gnu-g++`
cross-compiler on the dev host -- the same one
`src/runtime/ti_dsp/firmware/c7x/arm/build.sh` itself uses to build
`libc7x_arm_runtime.so`. No `numpy` or Python is needed on the board for
this example, since `resnet18_board_runner` is a native binary.

### Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=<path to>/ti-cgt-c7000_5.0.1.LTS
python examples/run_resnet18_classification.py
```

Run with `--help` for the full option list (image selection, board target,
`--compile-only`/`--inference-only`, build directory). Prints the top-5
ImageNet predictions for the chosen image.
