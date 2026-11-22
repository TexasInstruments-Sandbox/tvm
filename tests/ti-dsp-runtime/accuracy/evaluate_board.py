#!/usr/bin/env python
"""Measure a quantized classifier's accuracy on the C7x board.

Compiles the model as the quantized sweep does (c_static_lib with MMALIB), runs it
on the board on the first --n images of evaluate.py's split, and scores the
board's logits against the float model's and the Q/DQ graph's on the same images.
The quantization recipe and images match evaluate.py's, so the board's result can
be read next to an evaluate.py report on more images.

The board runs every image in one C7xVirtualMachine session, so the module loads
once. It needs the tvm-ti-c7x-inference package on the board and firmware built
for --board.

    C7X_DATASETS_DIR=~/datasets python accuracy/evaluate_board.py resnet18 \\
        --board beagley-ai --n 200 --work /tmp/acc --out board.json

--stage prepare compiles and computes the host logits into --work; --stage run
then uses the board and writes the report. Separate stages let one model compile
while the board runs another. Never run two --stage run at once on one board.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SUITE_DIR = _THIS_DIR.parent
for _path in (
    _SUITE_DIR.parent.parent / "python",
    _SUITE_DIR.parent / "cstatic",
    _SUITE_DIR / "dsp-cpp",
    _SUITE_DIR / "pt2e-tests",
    _THIS_DIR,
):
    sys.path.insert(0, str(_path))

# ruff: noqa: E402  (sys.path must be set up before these imports resolve)
from cl_torchvision import load_model_with_preprocessing
from dsp_utils import (
    add_board_arg,
    build_dsp_dynmod,
    compile_for_dsp,
    get_board_hostname,
    get_target_string,
)
from eval_datasets import datasets_dir, imagenetv2_split
from evaluate import _EXAMPLE_IMAGE, _load_chunks, _quantize, image_loader, summarize
from pt2e_utils import import_quantized
from quant_recipe import BASELINE_CALIB_IMAGES

# Runs on the board: one session, one image at a time, as the module's batch is 1.
_BOARD_RUNNER = """
import sys
import numpy as np
from tvm.contrib.c7x import C7xVirtualMachine

module, inputs, out, so_path = sys.argv[1:5]
x = np.load(inputs, mmap_mode="r")
logits, cycles = [], []
with C7xVirtualMachine(module, so_path=so_path) as vm:
    for i in range(len(x)):
        try:
            y = vm.run_nocopy(np.ascontiguousarray(x[i]))
        except RuntimeError:
            print(f"inference failed on image {i} of {len(x)}", file=sys.stderr)
            raise
        logits.append(np.array(y, dtype=np.float32).reshape(-1))
        cycles.append(vm.last_cycles)
np.savez(out, logits=np.stack(logits), cycles=np.array(cycles, dtype=np.int64))
"""


def prepare(model_name: str, n_eval: int, seed: int, work: Path) -> None:
    """Compile the model and save its inputs and host logits under work/model_name."""
    out = work / model_name
    out.mkdir(parents=True, exist_ok=True)
    split = imagenetv2_split(datasets_dir(), n_eval, 0, seed)
    model, preprocess, weights = load_model_with_preprocessing(model_name)
    model = model.eval()
    load = image_loader(preprocess)
    recipe = BASELINE_CALIB_IMAGES
    quantized = _quantize(model, load, recipe, [])

    paths = [path for path, _ in split.eval_items]
    inputs, float_logits, quant_logits = [], [], []
    with torch.no_grad():
        for x in _load_chunks(paths, load):
            inputs.append(x.numpy())
            float_logits.append(model(x).numpy())
            quant_logits.append(quantized(x).numpy())
    np.save(out / "inputs.npy", np.stack(inputs).astype(np.float32))
    np.savez(
        out / "host.npz",
        float_logits=np.concatenate(float_logits),
        quant_logits=np.concatenate(quant_logits),
        labels=np.array([label for _, label in split.eval_items]),
    )

    mod = import_quantized(quantized, (load(_EXAMPLE_IMAGE),))
    target = get_target_string("c7x_dload", use_cpp_api=True) + " -mmalib=1"
    generated = compile_for_dsp(mod, target, output_dir=out / "build")
    module = build_dsp_dynmod(
        generated, build_dir=out / "build" / "dynmod", weights_file=out / "build" / "weights.bin"
    )
    (out / "lib0.out").write_bytes(Path(module).read_bytes())
    meta = {"model": model_name, "weights": str(weights), "recipe": recipe.name, "seed": seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _ssh(host: str, command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", host, command], check=True, **kwargs)


def run_on_board(model_name: str, work: Path, so_path: str, remote_dir: str) -> dict:
    """Run a prepared model on the board and return its report."""
    out = work / model_name
    host = f"root@{get_board_hostname()}"
    remote = f"{remote_dir}/{model_name}"
    _ssh(host, f"rm -rf {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}")
    try:
        subprocess.run(
            ["scp", "-q", str(out / "lib0.out"), str(out / "inputs.npy"), f"{host}:{remote}/"],
            check=True,
        )
        _ssh(
            host,
            f"cd {shlex.quote(remote)} && python3 - lib0.out inputs.npy board.npz "
            f"{shlex.quote(so_path)}",
            input=_BOARD_RUNNER.encode(),
        )
        subprocess.run(
            ["scp", "-q", f"{host}:{remote}/board.npz", str(out / "board.npz")], check=True
        )
    finally:
        _ssh(host, f"rm -rf {shlex.quote(remote)}")
    return report(model_name, work)


def report(model_name: str, work: Path) -> dict:
    """Score the board's logits against the float model's and the Q/DQ graph's."""
    out = work / model_name
    host = np.load(out / "host.npz")
    board = np.load(out / "board.npz")
    float_logits, quant_logits, labels = host["float_logits"], host["quant_logits"], host["labels"]
    board_logits = board["logits"].reshape(quant_logits.shape)

    dot = (board_logits * quant_logits).sum(axis=1)
    norms = np.linalg.norm(board_logits, axis=1) * np.linalg.norm(quant_logits, axis=1)
    cosine = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
    # Ties break by lowest class index, as in summarize().
    board_top1 = np.argsort(-board_logits, axis=1, kind="stable")[:, 0]
    quant_top1 = np.argsort(-quant_logits, axis=1, kind="stable")[:, 0]
    return {
        **json.loads((out / "meta.json").read_text()),
        "dataset": "imagenetv2-matched-frequency",
        "n_eval": len(labels),
        "quant": summarize(float_logits, quant_logits, labels),
        "board": summarize(float_logits, board_logits, labels),
        "board_vs_quant": {
            "top1_agreement": float((board_top1 == quant_top1).mean()),
            "cosine_mean": float(cosine.mean()),
            "cosine_min": float(cosine.min()),
            "max_abs_diff": float(np.abs(board_logits - quant_logits).max()),
        },
        "cycles_median": int(np.median(board["cycles"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy of a quantized classifier on the C7x")
    parser.add_argument("models", nargs="+")
    add_board_arg(parser)
    parser.add_argument("--n", type=int, default=200, help="evaluation images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["all", "prepare", "run"], default="all")
    parser.add_argument("--work", type=Path, required=True, help="directory for builds and logits")
    parser.add_argument("--out", type=Path, help="JSON report path (stages all and run)")
    parser.add_argument(
        "--so-path",
        default="/usr/local/lib/libc7x_arm_runtime.so",
        help="libc7x_arm_runtime.so on the board, matching its firmware",
    )
    parser.add_argument("--remote-dir", default="/root/c7x_accuracy")
    args = parser.parse_args()
    if args.board is None:
        parser.error("--board is required")
    if args.stage != "prepare" and args.out is None:
        parser.error("--out is required unless --stage prepare")

    reports = []
    for name in args.models:
        if args.stage in ("all", "prepare"):
            prepare(name, args.n, args.seed, args.work)
        if args.stage in ("all", "run"):
            reports.append(run_on_board(name, args.work, args.so_path, args.remote_dir))
            # Rewritten after each model, so a later failure keeps these results.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
