#!/usr/bin/env python3
"""
ResNet-18 image classification on a real image, using the C++ offload API
(c7x::Module), running on the C7x/MMA (BeagleY-AI).

This is the host-side driver -- the C++ counterpart to
run_yolo26_detection.py: it quantizes and compiles ResNet-18, sends it to
the board, and classifies a real JPEG from tests/cstatic/test_images/.
Also see resnet18_board_runner.cpp, which this script cross-compiles,
deploys, and runs on the board over SSH.

Unlike the Python example, the board-side program (resnet18_board_runner)
also does the postprocessing (argmax + label lookup) itself, on the board
-- its stdout is the final human-readable result, so no output tensor is
shipped back to the host. See tests/ti-dsp-runtime/examples/common/c7x_infer.h
for the small shared library the board-side program is built on ("common
inference code"); resnet18_board_runner.cpp is the task-specific "main
application" on top of it.

Pipeline (all steps happen on the dev host except step 5):

  1. Load real image, preprocess it with ResNet18_Weights' own transform
     (resize/crop/normalize -- matches how the pretrained weights were
     trained).
  2. Quantize ResNet-18 (PT2E int8), calibrated on that same real image --
     not random noise, unlike quantized/model_utils.py's
     create_quantized_resnet_model(), which only needs DSP-vs-CPU
     consistency for its own tests, not real classification accuracy.
  3. Compile it for the C7x DSP with MMALIB offload (c_static backend) and
     build a DLOAD module (lib0.out).
  4. Cross-compile resnet18_board_runner.cpp against the already
     cross-compiled libc7x_arm_runtime.so.
  5. Deploy lib0.out + the preprocessed input + ImageNet labels + the
     board-runner binary to the board, run it over SSH, and print its
     stdout (the top-5 predictions).

Usage (run from tests/ti-dsp-runtime/):

    cd tests/ti-dsp-runtime
    export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
    python examples/run_resnet18_classification.py
    python examples/run_resnet18_classification.py --image dog.jpg
    python examples/run_resnet18_classification.py --board j722s-evm
    python examples/run_resnet18_classification.py --compile-only
    python examples/run_resnet18_classification.py --inference-only

Prerequisites: firmware + libc7x_arm_runtime.so already deployed on the
board (see src/runtime/ti_dsp/firmware/c7x/arm/README.md, "./build.sh
deploy") -- the same prerequisite run_yolo26_detection.py already has, plus
an aarch64-linux-gnu-g++ cross-compiler on the dev host (same one
src/runtime/ti_dsp/firmware/c7x/arm/build.sh itself uses).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).parent
_TVM_HOME = _THIS_DIR.parent.parent.parent  # examples -> ti-dsp-runtime -> tests -> tvm
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
_QUANTIZED_DIR = _THIS_DIR.parent / "quantized"
_CSTATIC_DIR = _TVM_HOME / "tests" / "cstatic"
_TEST_IMAGES_DIR = _CSTATIC_DIR / "test_images"
_ARM_DIR = _TVM_HOME / "src" / "runtime" / "ti_dsp" / "firmware" / "c7x" / "arm"
_ARM_INCLUDE_DIR = _ARM_DIR / "include"
_DLPACK_INCLUDE_DIR = _TVM_HOME / "3rdparty" / "tvm-ffi" / "3rdparty" / "dlpack" / "include"
_COMMON_DIR = _THIS_DIR / "common"
_BOARD_RUNNER_CPP = _THIS_DIR / "resnet18_board_runner.cpp"

sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_QUANTIZED_DIR))

# ruff: noqa: E402  (sys.path must be set up before these imports resolve)
from dsp_utils import (  # pyright: ignore[reportMissingImports]
    add_board_arg,
    build_dsp_dynmod,
    compile_for_dsp,
    get_board_hostname,
    get_target_string,
    set_current_board,
)
from model_utils import _pt2e_quantize  # pyright: ignore[reportMissingImports]

INPUT_SHAPE = (1, 3, 224, 224)

ALL_IMAGES = ["dog.jpg", "bird_0.jpg", "YellowLabradorLooking_new.jpg"]
DEFAULT_IMAGE = "YellowLabradorLooking_new.jpg"


def preprocess_image(image_path: Path):
    """Load a real image and preprocess it exactly as the pretrained weights expect.

    Uses ResNet18_Weights.DEFAULT's own transform (resize, center-crop to
    224, ToTensor, ImageNet mean/std normalization) -- getting this right
    matters for both quantization calibration (step below) and the
    inference input itself; create_quantized_resnet_model()'s own random
    calibration data skips this entirely because it only needs internal
    DSP-vs-CPU consistency, not a real classification result.

    Returns a torch.Tensor of shape (1, 3, 224, 224).
    """
    from PIL import Image  # noqa: PLC0415
    from torchvision.models import ResNet18_Weights  # noqa: PLC0415

    transform = ResNet18_Weights.DEFAULT.transforms()
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def compile_resnet18_for_board(board: str, input_tensor: torch.Tensor, build_dir: Path) -> Path:
    """Quantize + compile ResNet-18, build a DLOAD module, return its path.

    Calibrates on the same real, correctly-preprocessed image used for
    inference (a single real sample already fixes the activation-range
    mismatch that random-noise calibration would otherwise cause) rather
    than reusing quantized/model_utils.py's create_quantized_resnet_model(),
    which calibrates on random noise -- fine for that module's own
    DSP-vs-CPU consistency tests, not for a demo meant to produce a
    plausible real-world label.
    """
    from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415

    set_current_board(board)
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Quantizing ResNet-18 (PT2E int8, calibrated on the real input image)...")
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    example_args = (input_tensor,)
    mod, _quantized_gm = _pt2e_quantize(torch_model, example_args, calibration_data=[input_tensor])

    print("[2/3] Compiling for C7x + MMALIB...")
    # use_cpp_api=True: DLOAD requires C++ API codegen (see
    # test_quantized_resnet.py's own default); -mmalib=1 offloads conv2d/FC
    # to the C7x MMA accelerator.
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


def arm_build_dir(board: str) -> Path:
    """Where arm/build.sh puts libc7x_arm_runtime.so for this board.

    Mirrors board_build_dir.sh's resolve_board_build_dir() for the two
    boards arm/build.sh actually supports (no --ddr override here, same as
    run_yolo26_detection.py).
    """
    ddr = "4gb" if board == "beagley-ai" else "8gb"
    suffix = "" if (board == "j722s-evm" and ddr == "8gb") else f"-{board}-{ddr}"
    return _ARM_DIR / f"build{suffix}"


def cross_compile_board_runner(board: str, build_dir: Path) -> Path:
    """Cross-compile resnet18_board_runner.cpp for aarch64.

    A single g++ invocation against the already cross-compiled
    libc7x_arm_runtime.so (built by arm/build.sh) -- no CMake target, no
    build-system integration, demonstrating that c7x_runtime.h really only
    needs a header, DLPack, and the .so to link against.
    """
    so_dir = arm_build_dir(board)
    so_path = so_dir / "libc7x_arm_runtime.so"
    if not so_path.exists():
        raise FileNotFoundError(
            f"{so_path} not found. Build it first:\n"
            f"  cd src/runtime/ti_dsp/firmware/c7x/arm && ./build.sh --board {board}"
        )

    out_path = build_dir / "resnet18_board_runner"
    cmd = [
        "aarch64-linux-gnu-g++", "-std=c++14", "-O2",
        "-I", str(_ARM_INCLUDE_DIR),
        "-I", str(_COMMON_DIR),
        "-I", str(_DLPACK_INCLUDE_DIR),
        str(_BOARD_RUNNER_CPP),
        "-L", str(so_dir), "-lc7x_arm_runtime",
        "-o", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def deploy_and_run_on_board(
    board: str,
    module_path: Path,
    runner_path: Path,
    input_path: Path,
    labels_path: Path,
    remote_dir: str,
) -> str:
    """scp everything the board needs, run resnet18_board_runner, return its stdout.

    Mirrors run_yolo26_detection.py's deploy_and_run_on_board(), but there
    is no results retrieval step: resnet18_board_runner already prints the
    human-readable top-5 result itself, so its stdout (forwarded over SSH)
    is the final answer.
    """
    host = get_board_hostname(board)
    remote = f"root@{host}"

    def ssh(cmd: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", remote, cmd],
            capture_output=True, text=True, timeout=300, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ssh {remote} {cmd!r} failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    def scp_to(*local_paths) -> None:
        subprocess.run(
            ["scp", "-q", *[str(p) for p in local_paths], f"{remote}:{remote_dir}/"],
            check=True, timeout=180,
        )

    print(f"[4/5] Deploying to {remote}:{remote_dir} ...")
    ssh(f"mkdir -p {remote_dir}")
    scp_to(module_path, runner_path, input_path, labels_path)
    ssh(f"chmod +x {remote_dir}/{runner_path.name}")

    print("[5/5] Running inference on the board (c7x::Module)...")
    result = ssh(
        f"cd {remote_dir} && ./{runner_path.name} "
        f"{module_path.name} {input_path.name} {labels_path.name}"
    )
    return result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_board_arg(parser)
    parser.add_argument(
        "--image", choices=ALL_IMAGES, default=DEFAULT_IMAGE,
        help=f"Test image to classify (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--remote-dir", default="/tmp/resnet18_example",
        help="Working directory on the board (default: /tmp/resnet18_example)",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=_THIS_DIR / "build-resnet18",
        help="Directory for compiled artifacts, including lib0.out (default: "
        "examples/build-resnet18/) -- reused as-is on repeat runs, not a "
        "tempdir. Deliberately distinct from run_yolo26_detection.py's own "
        "default (examples/build/): build_dsp_dynmod() compiles every "
        "lib*.c file it finds in this directory, so two examples sharing "
        "one build dir would link each other's leftover generated code "
        "together and fail with duplicate MMALIB wrapper symbols. Also "
        "matches this project's build-<name>/ convention (see .gitignore's "
        "build-*/ rule) rather than build_<name>/, so it's gitignored too.",
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
        "cross-compiling/deploying/running.",
    )
    args = parser.parse_args()
    if args.board is None:
        args.board = "beagley-ai"  # this example's primary target board
    return args


def main() -> int:
    args = parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = preprocess_image(_TEST_IMAGES_DIR / args.image)
    input_path = args.build_dir / "input.bin"
    np.ascontiguousarray(input_tensor.numpy(), dtype=np.float32).tofile(input_path)

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
        module_path = compile_resnet18_for_board(args.board, input_tensor, args.build_dir)
        if args.compile_only:
            print(f"\n--compile-only: stopping after build.\nDLOAD module: {module_path}")
            return 0

    from torchvision.models import ResNet18_Weights  # noqa: PLC0415

    labels_path = args.build_dir / "labels.txt"
    labels_path.write_text("\n".join(ResNet18_Weights.DEFAULT.meta["categories"]) + "\n")

    runner_path = cross_compile_board_runner(args.board, args.build_dir)

    stdout = deploy_and_run_on_board(
        args.board, module_path, runner_path, input_path, labels_path, args.remote_dir
    )
    print(f"\n{args.image}:")
    print(stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
