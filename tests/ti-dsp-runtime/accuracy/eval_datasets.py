"""Datasets for measuring quantized model accuracy.

Nothing here downloads data, and no data or file list belongs in this
repository. Point C7X_DATASETS_DIR at a directory holding:

    imagenetv2/imagenetv2-matched-frequency-format-val/<class index>/<hash>.jpeg

That is ImageNetV2 MatchedFrequency: 10,000 images, 10 for each of ImageNet's
1000 classes. Its images have the same terms of use as ImageNet's. Accuracy on
it isn't comparable with published ImageNet numbers, so read results as a
difference from the float model on the same images.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path

_IMAGENETV2_SUBDIR = Path("imagenetv2") / "imagenetv2-matched-frequency-format-val"
_NUM_CLASSES = 1000
_IMAGES_PER_CLASS = 10
# Held out of evaluation, for an analysis that calibrates on dataset images.
_CALIBRATION_POOL_PER_CLASS = 2
MAX_EVAL_IMAGES = _NUM_CLASSES * (_IMAGES_PER_CLASS - _CALIBRATION_POOL_PER_CLASS)


@dataclass(frozen=True)
class Split:
    """Disjoint evaluation and calibration images."""

    eval_items: list[tuple[Path, int]]  # (image path, class index)
    calibration_paths: list[Path]


def datasets_dir() -> Path:
    root = os.environ.get("C7X_DATASETS_DIR")
    if not root:
        raise RuntimeError("set C7X_DATASETS_DIR to the directory holding the datasets")
    return Path(root)


def imagenetv2_split(
    root: Path, n_eval: int = MAX_EVAL_IMAGES, n_calibration: int = 512, seed: int = 0
) -> Split:
    """Return a fixed, class-balanced split of ImageNetV2.

    Any n_eval is balanced to within one image per class. The same arguments
    always give the same split.
    """
    if not 0 < n_eval <= MAX_EVAL_IMAGES:
        raise ValueError(f"n_eval must be in 1..{MAX_EVAL_IMAGES}, got {n_eval}")
    pool_size = _NUM_CLASSES * _CALIBRATION_POOL_PER_CLASS
    if not 0 <= n_calibration <= pool_size:
        raise ValueError(f"n_calibration must be in 0..{pool_size}, got {n_calibration}")

    base = root / _IMAGENETV2_SUBDIR
    rng = random.Random(seed)
    eval_by_class: dict[int, list[Path]] = {}
    pool: list[Path] = []
    # Walk class indices rather than sorted folder names: the folders are named
    # by number, and sorting them as strings, as torchvision's ImageFolder does,
    # puts "10" before "2" and silently scrambles the labels.
    for label in range(_NUM_CLASSES):
        paths = sorted((base / str(label)).glob("*.jpeg"))
        if len(paths) != _IMAGES_PER_CLASS:
            raise FileNotFoundError(
                f"expected {_IMAGES_PER_CLASS} images in {base / str(label)}, found {len(paths)}"
            )
        rng.shuffle(paths)
        pool.extend(paths[:_CALIBRATION_POOL_PER_CLASS])
        eval_by_class[label] = paths[_CALIBRATION_POOL_PER_CLASS:]

    labels = list(range(_NUM_CLASSES))
    rng.shuffle(labels)
    per_class = _IMAGES_PER_CLASS - _CALIBRATION_POOL_PER_CLASS
    eval_items = [(eval_by_class[label][i], label) for i in range(per_class) for label in labels]
    return Split(eval_items[:n_eval], rng.sample(pool, n_calibration))
