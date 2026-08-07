# Examples

Standalone scripts for running models on BeagleY-AI

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
  — see `src/runtime/ti_dsp/firmware/c7x/arm/README.md`
  (`./build.sh deploy`). Same prerequisite
  `quantized/test_quantized_yolo.py`'s `--dsp-mode=c7x_dload` runs already
  have.
- `TI_CGT_C7000_PATH` set (compiling for C7x).
- Passwordless SSH as `root` to the board.
- `numpy` in the board's system Python 3 (`yolo26_board_runner.py` needs it;
  it is not preinstalled on a stock BeagleY-AI image). Install with
  `apt-get install python3-numpy` on the board -- if `apt` can't resolve its
  mirrors, set `http_proxy`/`https_proxy` (e.g. TI's `http://wwwgate.ti.com:80`)
  first.

### Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
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
