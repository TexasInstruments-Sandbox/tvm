"""Unit tests for the accuracy harness. No dataset, weights, DSP or toolchain needed."""

from collections import Counter

import numpy as np
import pytest
from eval_datasets import MAX_EVAL_IMAGES, imagenetv2_split
from evaluate import summarize


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    """An ImageNetV2 directory tree whose images are empty files."""
    root = tmp_path_factory.mktemp("datasets")
    base = root / "imagenetv2" / "imagenetv2-matched-frequency-format-val"
    for label in range(1000):
        class_dir = base / str(label)
        class_dir.mkdir(parents=True)
        for i in range(10):
            (class_dir / f"{label}_{i}.jpeg").touch()
    return root


@pytest.mark.quick
def test_split_labels_come_from_the_class_folder(fake_root):
    # Would fail on a string sort of the folder names, which puts "10" before "2".
    for path, label in imagenetv2_split(fake_root, n_eval=MAX_EVAL_IMAGES).eval_items:
        assert path.parent.name == str(label)


@pytest.mark.quick
def test_split_eval_and_calibration_are_disjoint(fake_root):
    split = imagenetv2_split(fake_root, n_eval=MAX_EVAL_IMAGES, n_calibration=2000)
    eval_paths = {path for path, _ in split.eval_items}
    assert len(eval_paths) == MAX_EVAL_IMAGES
    assert len(set(split.calibration_paths)) == 2000
    assert not eval_paths & set(split.calibration_paths)


@pytest.mark.quick
@pytest.mark.parametrize("n_eval", [100, 1000, 1500, MAX_EVAL_IMAGES])
def test_split_is_class_balanced_to_within_one_image(fake_root, n_eval):
    split = imagenetv2_split(fake_root, n_eval=n_eval)
    assert len(split.eval_items) == n_eval
    per_class = Counter(label for _, label in split.eval_items)
    counts = [per_class[label] for label in range(1000)]
    assert max(counts) - min(counts) <= 1


@pytest.mark.quick
def test_split_depends_only_on_its_arguments(fake_root):
    first = imagenetv2_split(fake_root, n_eval=500, n_calibration=50, seed=1)
    assert first == imagenetv2_split(fake_root, n_eval=500, n_calibration=50, seed=1)
    assert first != imagenetv2_split(fake_root, n_eval=500, n_calibration=50, seed=2)


@pytest.mark.quick
@pytest.mark.parametrize("kwargs", [{"n_eval": 0}, {"n_eval": 8001}, {"n_calibration": 2001}])
def test_split_rejects_sizes_out_of_range(fake_root, kwargs):
    with pytest.raises(ValueError, match="must be in"):
        imagenetv2_split(fake_root, **kwargs)


@pytest.mark.quick
def test_split_rejects_an_incomplete_class(tmp_path):
    with pytest.raises(FileNotFoundError, match="expected 10 images"):
        imagenetv2_split(tmp_path)


@pytest.mark.quick
def test_summarize_identical_outputs_lose_nothing():
    logits = np.array([[3.0, 1, 0, 0, 0, 0], [0, 3.0, 1, 0, 0, 0]])
    metrics = summarize(logits, logits.copy(), np.array([0, 2]))
    assert metrics["top1_float"] == metrics["top1_quant"] == 0.5
    assert metrics["top5_float"] == 1.0
    assert metrics["top1_delta"] == 0.0
    assert metrics["top1_delta_stderr"] == 0.0
    assert metrics["top1_agreement"] == 1.0
    assert metrics["cosine_mean"] == pytest.approx(1.0)
    assert metrics["kl_mean"] == pytest.approx(0.0)


@pytest.mark.quick
def test_summarize_breaks_top1_ties_like_argmax():
    # Tied maxima at classes 3 and 900, in the shape quantized logits take. An
    # unstable sort would resolve some of these to 900.
    quant_logits = np.zeros((200, 1000), dtype=np.float32)
    quant_logits[:, [3, 900]] = 5.0
    labels = np.full(200, 3)
    assert (np.argmax(quant_logits, axis=1) == labels).all()

    metrics = summarize(quant_logits.copy(), quant_logits, labels)
    assert metrics["top1_quant"] == 1.0
    assert metrics["top1_tied_quant"] == 1.0


@pytest.mark.quick
def test_summarize_scores_an_all_zero_output_without_nan():
    float_logits = np.array([[5.0, 0.0], [5.0, 0.0]])
    quant_logits = np.array([[5.0, 0.0], [0.0, 0.0]])
    metrics = summarize(float_logits, quant_logits, np.zeros(2, dtype=int))
    assert metrics["cosine_min"] == 0.0
    assert metrics["cosine_mean"] == pytest.approx(0.5)


@pytest.mark.quick
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_summarize_rejects_non_finite_quantized_output(bad_value):
    float_logits = np.array([[5.0, 0.0], [5.0, 0.0]])
    quant_logits = np.array([[5.0, 0.0], [bad_value, 0.0]])
    with pytest.raises(ValueError, match="1 of 2 images"):
        summarize(float_logits, quant_logits, np.zeros(2, dtype=int))


@pytest.mark.quick
def test_summarize_delta_is_quant_minus_float_and_paired():
    # Image 0: float right, quantized wrong. Images 1-3: both right.
    float_logits = np.array([[5.0, 0], [5.0, 0], [5.0, 0], [5.0, 0]])
    quant_logits = np.array([[0, 5.0], [5.0, 0], [5.0, 0], [5.0, 0]])
    metrics = summarize(float_logits, quant_logits, np.zeros(4, dtype=int))
    assert metrics["top1_delta"] == pytest.approx(-0.25)
    assert metrics["top1_delta_stderr"] == pytest.approx(0.25)
    assert metrics["top1_agreement"] == pytest.approx(0.75)
    assert metrics["kl_mean"] > 0
