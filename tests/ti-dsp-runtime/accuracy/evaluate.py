#!/usr/bin/env python
"""Measure how much accuracy PT2E quantization costs a TorchVision classifier.

Runs the float model and the Q/DQ graph from convert_pt2e on the same images, in
PyTorch only: no compile, DSP or board. See eval_datasets.py for the data this
needs.

    C7X_DATASETS_DIR=~/datasets python accuracy/evaluate.py resnet18 --out report.json

--recipe picks a named QuantRecipe. --calibrate-on-dataset N keeps that recipe's
quantizer settings but calibrates on N held-out dataset images instead. That is
an analysis, not a recipe, and deliberately has no name in quant_recipe.py.
"""

import argparse
import itertools
import json
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SUITE_DIR = _THIS_DIR.parent
_TESTS_DIR = _SUITE_DIR.parent
for _path in (
    _TESTS_DIR.parent / "python",
    _TESTS_DIR / "cstatic",
    _SUITE_DIR / "dsp-cpp",
    _THIS_DIR,
):
    sys.path.insert(0, str(_path))

# ruff: noqa: E402  (sys.path must be set up before these imports resolve)
from cl_torchvision import load_image, load_model_with_preprocessing
from eval_datasets import MAX_EVAL_IMAGES, datasets_dir, imagenetv2_split
from quant_recipe import (
    BASELINE_CALIB_IMAGES,
    BASELINE_TEST_IMAGE,
    QuantRecipe,
    calibrate_and_convert,
    export_module,
    quantize_with_recipe,
)
from torch.fx import GraphModule

# A spread of residual and depthwise-heavy classifiers. YOLO needs a detection dataset.
DEFAULT_MODELS = (
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "shufflenet_v2_x0_5",
    "efficientnet_b0",
    "regnet_x_400mf",
    "squeezenet1_1",
)
RECIPES = {r.name: r for r in (BASELINE_CALIB_IMAGES, BASELINE_TEST_IMAGE)}
# The tests' input, used for export and by BASELINE_TEST_IMAGE, as in the sweep.
_EXAMPLE_IMAGE = _TESTS_DIR / "cstatic" / "test_images" / "dog.jpg"
_LOAD_CHUNK = 32
_LOAD_THREADS = 4


def _quantize(
    model: torch.nn.Module,
    load: Callable[[Path], torch.Tensor],
    recipe: QuantRecipe,
    calibration_paths: list[Path],
) -> GraphModule:
    """Quantize model with recipe, or on calibration_paths if there are any."""
    example = (load(_EXAMPLE_IMAGE),)
    if not calibration_paths:
        return quantize_with_recipe(model, example, recipe, load)
    samples = [(load(path),) for path in calibration_paths]
    return calibrate_and_convert(export_module(model, example), recipe.quantizer(), samples)


def _load_chunks(paths: list[Path], load: Callable[[Path], torch.Tensor]) -> Iterable[torch.Tensor]:
    """Decode images ahead of the model, holding at most a chunk in memory."""
    with ThreadPoolExecutor(_LOAD_THREADS) as pool:
        iterator = iter(paths)
        while chunk := list(itertools.islice(iterator, _LOAD_CHUNK)):
            yield from pool.map(load, chunk)


def summarize(
    float_logits: np.ndarray, quant_logits: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Compute accuracy and fidelity metrics.

    Args:
        float_logits, quant_logits: [n, classes] outputs on the same images.
        labels: [n] class indices.

    Raises:
        ValueError: quant_logits has a NaN or infinity, which no metric can
            score. A broken model should fail loudly, not report a NaN mean.
    """
    bad = ~np.isfinite(quant_logits).all(axis=1)
    if bad.any():
        raise ValueError(f"non-finite quantized output on {int(bad.sum())} of {len(bad)} images")

    # Quantized logits sit on a grid, so the largest can tie. A stable sort
    # breaks a tie by lowest class index, as np.argmax does (which the DSP tests
    # use for top-1); the default sort's order varies with numpy build and CPU.
    float_top5 = np.argsort(-float_logits, axis=1, kind="stable")[:, :5]
    quant_top5 = np.argsort(-quant_logits, axis=1, kind="stable")[:, :5]
    top1_tied = (quant_logits == quant_logits.max(axis=1, keepdims=True)).sum(axis=1) > 1
    float_ok = float_top5[:, 0] == labels
    quant_ok = quant_top5[:, 0] == labels
    delta = quant_ok.astype(np.float64) - float_ok.astype(np.float64)

    dot = (float_logits * quant_logits).sum(axis=1)
    norms = np.linalg.norm(float_logits, axis=1) * np.linalg.norm(quant_logits, axis=1)
    # An all-zero output has no direction: score it 0 rather than dividing by zero.
    cosine = np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)
    log_p = torch.log_softmax(torch.from_numpy(float_logits), dim=1).numpy()
    log_q = torch.log_softmax(torch.from_numpy(quant_logits), dim=1).numpy()
    kl = (np.exp(log_p) * (log_p - log_q)).sum(axis=1)

    return {
        "top1_float": float(float_ok.mean()),
        "top1_quant": float(quant_ok.mean()),
        "top1_delta": float(delta.mean()),
        # Paired over the same images, so this is what says whether a delta is real.
        "top1_delta_stderr": float(delta.std(ddof=1) / np.sqrt(len(delta)))
        if len(delta) > 1
        else 0.0,
        "top5_float": float((float_top5 == labels[:, None]).any(axis=1).mean()),
        "top5_quant": float((quant_top5 == labels[:, None]).any(axis=1).mean()),
        "top1_agreement": float((float_top5[:, 0] == quant_top5[:, 0]).mean()),
        "top1_tied_quant": float(top1_tied.mean()),
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "kl_mean": float(kl.mean()),
    }


def evaluate(
    model_name: str,
    recipe: QuantRecipe,
    n_eval: int,
    calibrate_on_dataset: int = 0,
    seed: int = 0,
) -> dict:
    """Return a JSON-serializable report for one model and calibration."""
    split = imagenetv2_split(datasets_dir(), n_eval, calibrate_on_dataset, seed)
    model, preprocess, weights = load_model_with_preprocessing(model_name)
    model = model.eval()

    def load(path: Path) -> torch.Tensor:
        image = load_image(str(path))
        if image is None:
            raise FileNotFoundError(path)
        tensor = preprocess(image)
        assert isinstance(tensor, torch.Tensor)
        return tensor.unsqueeze(0)

    quantized = _quantize(model, load, recipe, split.calibration_paths)

    float_logits, quant_logits = [], []
    paths = [path for path, _ in split.eval_items]
    with torch.no_grad():
        for x in _load_chunks(paths, load):
            float_logits.append(model(x).numpy())
            quant_logits.append(quantized(x).numpy())

    metrics = summarize(
        np.concatenate(float_logits),
        np.concatenate(quant_logits),
        np.array([label for _, label in split.eval_items]),
    )
    return {
        "model": model_name,
        "weights": str(weights),
        "recipe": recipe.name,
        "calibration": f"dataset-{calibrate_on_dataset}" if calibrate_on_dataset else recipe.name,
        "dataset": "imagenetv2-matched-frequency",
        "n_eval": len(paths),
        "seed": seed,
        **metrics,
    }


def _print_table(reports: list[dict]) -> None:
    print(
        f"{'model':<22}{'calibration':<24}{'float':>7}{'quant':>7}{'delta':>7}{'+/-':>6}"
        f"{'agree':>7}{'tied':>7}{'cos':>7}{'cosmin':>8}{'KL':>7}"
    )
    for r in reports:
        print(
            f"{r['model']:<22}{r['calibration']:<24}"
            f"{r['top1_float']:>7.2%}{r['top1_quant']:>7.2%}{r['top1_delta']:>7.2%}"
            f"{r['top1_delta_stderr']:>6.2%}{r['top1_agreement']:>7.2%}{r['top1_tied_quant']:>7.2%}"
            f"{r['cosine_mean']:>7.3f}{r['cosine_min']:>8.3f}{r['kl_mean']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy of PT2E-quantized classifiers vs float")
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--recipe", choices=sorted(RECIPES), default=BASELINE_CALIB_IMAGES.name)
    parser.add_argument("--calibrate-on-dataset", type=int, default=0, metavar="N")
    parser.add_argument("--n", type=int, default=MAX_EVAL_IMAGES, help="evaluation images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=f"torch threads (0: torch default); decoding images adds {_LOAD_THREADS} more",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.models:
        reports.append(
            evaluate(name, RECIPES[args.recipe], args.n, args.calibrate_on_dataset, args.seed)
        )
        # Rewritten after each model, so a later failure keeps these results.
        args.out.write_text(json.dumps(reports, indent=2) + "\n")
    _print_table(reports)


if __name__ == "__main__":
    main()
