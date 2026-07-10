import numpy as np
import pytest

from matchlens.evaluation import (
    compute_per_class_metrics,
    compute_prediction_distribution,
    compute_probability_diagnostics,
)

TARGET_NAMES = ["Away Win", "Draw", "Home Win"]


def test_per_class_metrics_one_row_per_class_and_zero_division():
    # Never predict Draw (class 1); should not crash and Draw precision/recall/f1 = 0
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 0, 2, 0, 2, 2])

    rows = compute_per_class_metrics(y_true, y_pred, TARGET_NAMES)

    assert len(rows) == 3
    assert [r["class_name"] for r in rows] == TARGET_NAMES
    for row in rows:
        assert {"precision", "recall", "f1", "support"} <= set(row.keys())

    draw = rows[1]
    assert draw["class_name"] == "Draw"
    assert draw["precision"] == 0.0
    assert draw["recall"] == 0.0
    assert draw["f1"] == 0.0
    assert draw["support"] == 2


def test_prediction_distribution_counts_and_proportions():
    y_pred = np.array([0, 0, 2, 2, 2])
    rows = compute_prediction_distribution(y_pred, TARGET_NAMES)

    assert len(rows) == 3
    by_class = {r["predicted_class"]: r for r in rows}
    assert by_class["Away Win"]["count"] == 2
    assert by_class["Draw"]["count"] == 0
    assert by_class["Home Win"]["count"] == 3
    assert by_class["Away Win"]["proportion"] == pytest.approx(0.4)
    assert by_class["Draw"]["proportion"] == pytest.approx(0.0)
    assert by_class["Home Win"]["proportion"] == pytest.approx(0.6)
    assert sum(r["proportion"] for r in rows) == pytest.approx(1.0)


def test_probability_diagnostics_finite_and_in_range():
    y_true = np.array([0, 1, 2, 0])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
        ]
    )
    labels = [0, 1, 2]

    diag = compute_probability_diagnostics(y_true, y_proba, labels=labels)

    assert np.isfinite(diag["log_loss"])
    assert np.isfinite(diag["multiclass_brier_score"])
    assert diag["multiclass_brier_score"] >= 0.0
    assert 0.0 <= diag["mean_top_probability"] <= 1.0
    assert 0.0 <= diag["mean_probability_gap"] <= 1.0
    assert diag["mean_entropy"] >= 0.0
