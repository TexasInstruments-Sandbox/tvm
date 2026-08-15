# Examples: YOLO26 & ResNet-18

Standalone scripts for running models on BeagleY-AI. Located at
`tests/ti-dsp-runtime/examples/`.

The two examples illustrate the two offload APIs: YOLO26 detection uses
the Python API (`tvm.contrib.c7x.C7xVirtualMachine`), and ResNet-18
classification uses the C++ API (`c7x::Module`). See [Python / C++ API
Reference](python-api.md) for the APIs themselves. Both quantize their
model (PT2E int8) before compiling for MMALIB offload -- see
[Quantization](quantization.md) for that step and
[Compilation](compilation.md) for how the compiled model gets built into
what each script actually deploys.

!!! note "Already run by `validate_all.sh`"
    Both examples below run automatically as the last step of
    [Getting Started](getting-started.md)'s three-command flow --
    `build_all.sh --wheels` then `validate_all.sh` builds the wheels,
    deploys firmware, and runs both scripts end to end against real
    hardware. This page is for running either script by hand.

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

- Firmware, `libc7x_arm_runtime.so`, and the `tvm-ti-c7x-compile` wheel this
  script imports `tvm` from -- see [Getting Started](getting-started.md) for
  the `build_all.sh --wheels` + `validate_all.sh` flow that builds and
  deploys both. Same prerequisite `quantized/test_quantized_yolo.py`'s
  `--dsp-mode=c7x_dload` runs already have.
- `TI_CGT_C7000_PATH` set (compiling for C7x) -- already baked into
  `docker/Dockerfile.ci_c7x`, so this only needs setting explicitly when
  running outside that image.
- Passwordless SSH as `root` to the board.
- `numpy` in the board's system Python 3 (`yolo26_board_runner.py` needs it;
  it is not preinstalled on a stock BeagleY-AI image). Install with
  `apt-get install python3-numpy` on the board -- if `apt` can't resolve its
  mirrors, set `http_proxy`/`https_proxy` first.

### Running

Recommended, inside the same `ci_c7x` container [Getting
Started](getting-started.md) builds and deploys with -- this assumes
`build_all.sh --wheels` and at least one `validate_all.sh --board
beagley-ai` already ran against this checkout, which is what creates
`.venv-ci-c7x` with the `tvm-ti-c7x-compile` wheel installed:

```bash
docker/bash.sh --net=host -v ~/.ssh:$(pwd)/.ssh:ro tvm.ci_c7x -- bash -c '
    source .venv-ci-c7x/bin/activate
    unset PYTHONPATH
    cd tests/ti-dsp-runtime
    python examples/run_yolo26_detection.py --board beagley-ai
'
```

Outside Docker (e.g. a host without Docker at all), set
`TI_CGT_C7000_PATH` explicitly and run against the source tree:

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=<path to>/ti-cgt-c7000_5.0.1.LTS
python examples/run_yolo26_detection.py --board beagley-ai
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

Same as YOLO26 above (firmware, `libc7x_arm_runtime.so`, and the
`tvm-ti-c7x-compile` wheel deployed via `build_all.sh --wheels` +
`validate_all.sh`; `TI_CGT_C7000_PATH`; passwordless SSH), plus an
`aarch64-linux-gnu-g++` cross-compiler on the dev host -- the same one
`src/runtime/ti_dsp/firmware/c7x/arm/build.sh` itself uses to build
`libc7x_arm_runtime.so`, and also already baked into
`docker/Dockerfile.ci_c7x`. No `numpy` or Python is needed on the board for
this example, since `resnet18_board_runner` is a native binary.

### Running

Recommended, inside the same `ci_c7x` container [Getting
Started](getting-started.md) builds and deploys with (see the YOLO26
Running section above for what this assumes about `.venv-ci-c7x`):

```bash
docker/bash.sh --net=host -v ~/.ssh:$(pwd)/.ssh:ro tvm.ci_c7x -- bash -c '
    source .venv-ci-c7x/bin/activate
    unset PYTHONPATH
    cd tests/ti-dsp-runtime
    python examples/run_resnet18_classification.py --board beagley-ai
'
```

Outside Docker, set `TI_CGT_C7000_PATH` explicitly and run against the
source tree:

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=<path to>/ti-cgt-c7000_5.0.1.LTS
python examples/run_resnet18_classification.py --board beagley-ai
```

Run with `--help` for the full option list (image selection, board target,
`--compile-only`/`--inference-only`, build directory). Prints the top-5
ImageNet predictions for the chosen image.
