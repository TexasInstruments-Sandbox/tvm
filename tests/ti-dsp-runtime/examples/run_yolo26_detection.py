#!/usr/bin/env python3
"""
YOLO26 object detection on real images, on the BeagleY-AI C7x DSP.

This is the host-side driver: it quantizes and compiles YOLO26n, sends it to
the board, and runs it on the real JPEGs in ``tests/cstatic/test_images/``
using the Python offload API documented in ``MODEL_API.md`` /
``tvm.contrib.c7x`` -- ``C7xVirtualMachine`` -- rather than the SSH/CLI test
harness that ``quantized/test_quantized_yolo.py`` uses for automated
comparisons. That inference call itself only happens on the *board*
(``C7xVirtualMachine`` talks to the DSP over rpmsg, so it only works
on-device) -- see ``yolo26_board_runner.py``, which this script deploys and
runs there over SSH.

Pipeline (all numbered steps below happen on the dev host except step 6):

  1. Quantize YOLO26n (PT2E int8, reusing quantized/model_utils.py -- this
     already has the fixes YOLO26's NMS-free "one2one" head needs).
  2. Compile it for the C7x DSP with MMALIB offload (c_static backend).
  3. Build a DLOAD module (lib0.out) with weights embedded in it.
  4. Load + preprocess the real test images (resize to the network's input
     size, matching how the model was calibrated -- no letterbox).
  5. Deploy lib0.out + preprocessed inputs + the board runner to the board.
  6. Run inference on the board (yolo26_board_runner.py, using
     C7xVirtualMachine -- the actual point of this example).
  7. Retrieve the raw detections and turn them into pixel coordinates on
     the original image.
  8. Draw the surviving (above --conf-threshold) boxes and save + print them.

Usage (run from tests/ti-dsp-runtime/ -- yolo26n.pt is cached there):

    cd tests/ti-dsp-runtime
    export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
    python examples/run_yolo26_detection.py
    python examples/run_yolo26_detection.py --image dog.jpg --conf-threshold 0.1
    python examples/run_yolo26_detection.py --board j722s-evm
    python examples/run_yolo26_detection.py --compile-only  # just produce lib0.out
    python examples/run_yolo26_detection.py --inference-only  # reuse it, skip recompiling

Prerequisites: firmware + libc7x_arm_runtime.so already deployed on the
board (see src/runtime/ti_dsp/firmware/c7x/arm/README.md, "./build.sh
deploy") -- the same prerequisite quantized/test_quantized_yolo.py's
``c7x_dload`` runs already have.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw

_THIS_DIR = Path(__file__).parent
_TVM_HOME = _THIS_DIR.parent.parent.parent  # examples -> ti-dsp-runtime -> tests -> tvm
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
_QUANTIZED_DIR = _THIS_DIR.parent / "quantized"
_CSTATIC_DIR = _TVM_HOME / "tests" / "cstatic"
_TEST_IMAGES_DIR = _CSTATIC_DIR / "test_images"
_C7X_RUNTIME_PY = _TVM_HOME / "python" / "tvm" / "contrib" / "c7x" / "c7x_runtime.py"
_BOARD_RUNNER_PY = _THIS_DIR / "yolo26_board_runner.py"

sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_QUANTIZED_DIR))
sys.path.insert(0, str(_CSTATIC_DIR))

# ruff: noqa: E402  (sys.path must be set up before these imports resolve)
from dsp_utils import (  # pyright: ignore[reportMissingImports]
    add_board_arg,
    build_dsp_dynmod,
    compile_for_dsp,
    get_board_hostname,
    get_target_string,
    set_current_board,
)
from model_utils import (  # pyright: ignore[reportMissingImports]
    YOLO_INPUT_SHAPE,
    create_quantized_yolo_model,
)
from od_yolo import Detection, load_image  # pyright: ignore[reportMissingImports]

MODEL_NAME = "yolo26n"

# All three sample images ship in tests/cstatic/test_images/; --image narrows
# this down to just one.
ALL_IMAGES = ["dog.jpg", "bird_0.jpg", "YellowLabradorLooking_new.jpg"]

# A few visually distinct colors, cycled by class id, purely for readability
# when drawing boxes -- not meaningful beyond that.
_BOX_COLORS = [
    (220, 40, 40), (40, 160, 60), (40, 100, 220), (220, 160, 20), (160, 40, 200),
]


def compile_yolo26_for_board(board: str, build_dir: Path) -> Path:
    """Quantize + compile YOLO26n, build a DLOAD module, return its path.

    Steps 1-3 of the module docstring's pipeline. Reuses
    quantized/model_utils.py's create_quantized_yolo_model (already handles
    YOLO26's one2one-head export fixes) and dsp-cpp/dsp_utils.py's DSP
    compile/build helpers -- the same functions
    quantized/test_quantized_yolo.py uses, just driven manually instead of
    through compile_and_run_dsp()'s SSH/CLI test harness.

    build_dir is a plain, reusable directory (not an auto-cleaned tempdir):
    every file this produces has a fixed name (lib0.c, weights.bin, lib0.out,
    ...), so re-running just overwrites it in place -- handy for --compile-only,
    where inspecting/reusing a known, stable path is the point.
    """
    set_current_board(board)
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Quantizing {MODEL_NAME} (PT2E int8, calibrated on the real test images)...")
    mod, _quantized_gm, _random_input = create_quantized_yolo_model(MODEL_NAME, version="v26")

    print("[2/3] Compiling for C7x + MMALIB...")
    # use_cpp_api=True is not a style choice -- DLOAD requires C++ API codegen
    # (see test_resnet_dsp.py's docstring: needed for multi-element make_tuple
    # support), and this script only ever targets c7x_dload, so it's always
    # on, not a CLI flag. -mmalib=1 offloads conv2d/matmul to the C7x MMA
    # accelerator (required for YOLO -- see quantized/README.md).
    target_string = get_target_string("c7x_dload", use_cpp_api=True) + " -mmalib=1"
    generated_dir = compile_for_dsp(mod, target_string=target_string, output_dir=build_dir)

    print("[3/3] Building DLOAD module (weights embedded)...")
    weights_path = generated_dir / "weights.bin"
    module_path = build_dsp_dynmod(
        generated_dir=generated_dir,
        build_dir=build_dir / "build",
        weights_file=weights_path if weights_path.exists() else None,
    )
    return module_path


def preprocess_image(image_path: Path) -> tuple:
    """Load a real image and preprocess it the way the model was calibrated.

    Returns (input_array, pil_image) where input_array is float32 NCHW in
    [0, 1] at the network's input resolution, and pil_image is the
    original-resolution RGB image (kept around to draw on later).

    Matches quantized/model_utils.py's _load_yolo_calibration_frames: a
    plain resize to (size, size), no letterbox padding -- the model's own
    calibration data was built the same way, so this keeps preprocessing
    consistent between calibration and this demo's real inference.
    """
    size = YOLO_INPUT_SHAPE[-1]
    image = load_image(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load test image: {image_path}")
    resized = image.resize((size, size))
    chw = (np.array(resized).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return chw[np.newaxis, ...], image


def deploy_and_run_on_board(
    board: str,
    module_path: Path,
    input_paths: list,
    remote_dir: str,
) -> Path:
    """scp everything the board needs, run inference there, scp results back.

    This is the host<->board handoff: C7xVirtualMachine (used inside
    yolo26_board_runner.py) only works when run on the board itself, since it
    talks to the DSP over the on-device rpmsg IPC channel. Mirrors
    dsp-tests/test_c7x_vm_dsp.py's _run_c7x_vm_on_board() helper, generalized
    for real image inputs/outputs instead of one fixed-shape test tensor.

    Returns the local directory containing the retrieved output_<name>.npy
    files.
    """
    host = get_board_hostname(board)
    remote = f"root@{host}"

    def ssh(cmd: str) -> None:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", remote, cmd],
            capture_output=True, text=True, timeout=300, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ssh {remote} {cmd!r} failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        if result.stdout:
            print(result.stdout, end="")

    def scp_to(*local_paths) -> None:
        subprocess.run(
            ["scp", "-q", *[str(p) for p in local_paths], f"{remote}:{remote_dir}/"],
            check=True, timeout=180,
        )

    def scp_from(remote_glob: str, local_dir: Path) -> None:
        subprocess.run(
            ["scp", "-q", f"{remote}:{remote_dir}/{remote_glob}", str(local_dir)],
            check=True, timeout=180,
        )

    print(f"[4/5] Deploying to {remote}:{remote_dir} ...")
    ssh(f"mkdir -p {remote_dir}")
    scp_to(module_path, _BOARD_RUNNER_PY, _C7X_RUNTIME_PY, *input_paths)

    print("[5/5] Running inference on the board (C7xVirtualMachine.run_nocopy)...")
    ssh(
        f"cd {remote_dir} && python3 yolo26_board_runner.py "
        f"--module {module_path.name} --input-dir ."
    )

    local_outputs = module_path.parent / "outputs"
    local_outputs.mkdir(exist_ok=True)
    scp_from("output_*.npy", local_outputs)
    return local_outputs


def postprocess_and_draw(
    detections_raw: np.ndarray,
    original_image,
    class_names: dict,
    conf_threshold: float,
) -> list:
    """Filter/scale raw [300, 6] detections, draw them on original_image in
    place, and return the surviving Detection list (caller does the printing).

    detections_raw rows are [x1, y1, x2, y2, confidence, class_idx] in pixel
    coordinates of the network's (square) input resolution -- yolo26's
    NMS-free one2one head has already done its own top-k selection, so no
    further NMS is needed here, just a confidence cutoff (see
    quantized/README.md's discussion of YOLO26's one2one head).
    """
    size = YOLO_INPUT_SHAPE[-1]
    orig_w, orig_h = original_image.size
    scale_x, scale_y = orig_w / size, orig_h / size

    detections = []
    for x1, y1, x2, y2, score, class_idx in detections_raw:
        if score < conf_threshold:
            continue
        box = (
            max(0, min(x1 * scale_x, orig_w - 1)),
            max(0, min(y1 * scale_y, orig_h - 1)),
            max(0, min(x2 * scale_x, orig_w - 1)),
            max(0, min(y2 * scale_y, orig_h - 1)),
        )
        label = int(round(class_idx))
        label_name = str(class_names.get(label, label))
        det = Detection(box=box, label=label, label_name=label_name, score=float(score))
        detections.append(det)
    detections.sort(key=lambda d: -d.score)

    draw = ImageDraw.Draw(original_image)
    for det in detections:
        color = _BOX_COLORS[det.label % len(_BOX_COLORS)]
        box_int = tuple(int(v) for v in det.box)
        draw.rectangle(box_int, outline=color, width=3)
        caption = f"{det.label_name} {det.score * 100:.0f}%"
        # 7px/char and 12px tall are just PIL's default bitmap font metrics --
        # sized to fit the caption behind it, nothing more precise than that.
        label_bg = (box_int[0], box_int[1] - 12, box_int[0] + 7 * len(caption), box_int[1])
        draw.rectangle(label_bg, fill=color)
        draw.text((box_int[0] + 2, box_int[1] - 12), caption, fill=(255, 255, 255))

    return detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_board_arg(parser)
    parser.add_argument(
        "--image", choices=ALL_IMAGES, default=None,
        help="Run on just this image instead of all three test images",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25,
        help="Minimum confidence to keep a detection (default: 0.25)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_THIS_DIR / "yolo26_detections",
        help="Directory to save annotated images to (default: examples/yolo26_detections/)",
    )
    parser.add_argument(
        "--remote-dir", default="/tmp/yolo26_example",
        help="Working directory on the board (default: /tmp/yolo26_example)",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=_THIS_DIR / "build",
        help="Directory for compiled artifacts, including lib0.out (default: "
        "examples/build/) -- reused as-is on repeat runs, not a tempdir.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--compile-only", action="store_true",
        help="Stop after building the DLOAD module (lib0.out) -- skip deploying to "
        "and running on the board. Useful for iterating on compilation.",
    )
    mode_group.add_argument(
        "--inference-only", action="store_true",
        help="Skip quantization/compilation -- reuse the lib0.out already built at "
        "--build-dir (e.g. from a prior --compile-only run) and go straight to "
        "deploying/running/postprocessing. Useful for iterating on the board-side "
        "or postprocessing code without repaying the compile cost each time.",
    )
    args = parser.parse_args()
    if args.board is None:
        args.board = "beagley-ai"  # this example's primary target board
    return args


def main() -> int:
    args = parse_args()
    image_names = [args.image] if args.image else ALL_IMAGES
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.inference_only:
        module_path = args.build_dir / "build" / "lib0.out"
        if not module_path.exists():
            print(
                f"--inference-only: no compiled module at {module_path}.\n"
                "Run once without --inference-only (or with --compile-only) first.",
                file=sys.stderr,
            )
            return 1
        print(f"--inference-only: reusing compiled module: {module_path}")
    else:
        module_path = compile_yolo26_for_board(args.board, args.build_dir)
        if args.compile_only:
            print(f"\n--compile-only: stopping after build.\nDLOAD module: {module_path}")
            return 0

    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    # Absolute path, unlike create_quantized_yolo_model's internal loader --
    # avoids depending on the process's cwd containing yolo26n.pt.
    class_names = YOLO(str(_THIS_DIR.parent / f"{MODEL_NAME}.pt")).names

    print(f"\nPreprocessing {len(image_names)} image(s)...")
    input_paths, original_images = [], {}
    for name in image_names:
        input_array, original_image = preprocess_image(_TEST_IMAGES_DIR / name)
        tag = Path(name).stem
        input_path = args.build_dir / f"input_{tag}.npy"
        np.save(input_path, input_array)
        input_paths.append(input_path)
        original_images[tag] = original_image

    outputs_dir = deploy_and_run_on_board(args.board, module_path, input_paths, args.remote_dir)

    print("\n" + "=" * 60)
    for name in image_names:
        tag = Path(name).stem
        output_path = outputs_dir / f"output_{tag}.npy"
        detections_raw = np.load(output_path)[0]  # drop the batch dim -> [300, 6]

        detections = postprocess_and_draw(
            detections_raw, original_images[tag], class_names, args.conf_threshold
        )

        print(f"\n{name}: {len(detections)} detection(s) above conf={args.conf_threshold}")
        for det in detections:
            x1, y1, x2, y2 = det.box
            print(
                f"  {det.label_name:<15s} {det.score * 100:5.1f}%  "
                f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )

        out_path = args.output_dir / f"{tag}_detected.jpg"
        original_images[tag].save(out_path)
        print(f"  saved -> {out_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
