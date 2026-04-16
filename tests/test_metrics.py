"""Test metric computation helpers."""

import numpy as np
import pytest

from laurium.components.evaluate_metrics import compute_metrics


@pytest.fixture(name="imperfect_multiclass_eval_pred")
def imperfect_multiclass_eval_pred_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Create multiclass logits and labels with known scores."""
    logits = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.4, 0.1, 0.5],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    labels = np.array(
        [
            0,  # predicted 0, true 0
            0,  # predicted 0, true 0
            0,  # predicted 0, true 0
            0,  # predicted 2, true 0
            1,  # predicted 1, true 1
            1,  # predicted 1, true 1
            1,  # predicted 2, true 1
            0,  # predicted 0, true 0
            2,  # predicted 1, true 2
            2,  # predicted 2, true 2
        ]
    )

    # Confusion matrix:
    # [[4, 0, 1],
    #  [0, 2, 1],
    #  [0, 1, 1]]

    return logits, labels


@pytest.fixture(name="perfect_multiclass_eval_pred")
def perfect_multiclass_eval_pred_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Create multiclass logits and labels with perfect scores."""
    logits = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
    )
    labels = np.array([0, 1, 2])

    return logits, labels


def test_compute_metrics_accuracy_perfect_predictions(
    perfect_multiclass_eval_pred: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test accuracy score on perfect predictions."""
    result = compute_metrics(perfect_multiclass_eval_pred, ["accuracy"])

    assert result["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_accuracy_imperfect_predictions(
    imperfect_multiclass_eval_pred: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test accuracy score on imperfect predictions."""
    result = compute_metrics(imperfect_multiclass_eval_pred, ["accuracy"])

    assert result["accuracy"] == pytest.approx(7 / 10)


def test_compute_metrics_non_accuracy_uses_defaults(
    imperfect_multiclass_eval_pred: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test non-accuracy metrics using default config and averaging values."""
    result = compute_metrics(imperfect_multiclass_eval_pred, ["f1"])

    assert result["f1"] == pytest.approx(88 / 135)


def test_compute_metrics_mixed_metric_list_merges_results(
    imperfect_multiclass_eval_pred: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test result dictionary contains values from all requested metrics."""
    result = compute_metrics(
        imperfect_multiclass_eval_pred, ["accuracy", "f1"]
    )

    assert set(result) == {"accuracy", "f1"}
    assert result["accuracy"] == pytest.approx(7 / 10)
    assert result["f1"] == pytest.approx(88 / 135)


@pytest.mark.parametrize(
    ["average", "expected_f1"],
    [
        ("macro", 88 / 135),
        ("micro", 7 / 10),
        ("weighted", 163 / 225),
    ],
)
def test_compute_metrics_f1_average_variants(
    imperfect_multiclass_eval_pred: tuple[np.ndarray, np.ndarray],
    average: str,
    expected_f1: float,
) -> None:
    """Test that averaging strategies return expected scores."""
    result = compute_metrics(
        imperfect_multiclass_eval_pred,
        ["f1"],
        config_name="multiclass",
        average=average,
    )

    assert result["f1"] == pytest.approx(expected_f1)
