"""Tests for exploratory draw-focused experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matchlens.artifacts import save_phase1_draw_experiment_results
from matchlens.draw_experiments import (
    DRAW_EXPERIMENT_COLUMNS,
    apply_draw_margin_rule,
    apply_draw_probability_threshold,
    build_draw_class_weight,
    run_draw_focused_experiments,
)
from matchlens.evaluation import compute_probability_diagnostics


def test_build_draw_class_weight_resolves_draw_from_class_names():
    # Non-standard order so Draw is not index 1 — guards against hardcoded indices
    class_names = ["Home Win", "Away Win", "Draw"]
    weights = build_draw_class_weight(class_names, 1.5)
    assert weights[2] == 1.5
    assert weights[0] == 1.0
    assert weights[1] == 1.0
    assert set(weights.keys()) == {0, 1, 2}

    class_names_alt = ["Draw", "Home Win", "Away Win"]
    weights_alt = build_draw_class_weight(class_names_alt, 2.0)
    assert weights_alt[0] == 2.0
    assert weights_alt[1] == 1.0
    assert weights_alt[2] == 1.0


def test_apply_draw_probability_threshold():
    class_names = ["Away Win", "Draw", "Home Win"]
    # Columns: Away Win, Draw, Home Win
    y_proba = np.array(
        [
            [0.20, 0.35, 0.45],  # P(Draw)=0.35 >= 0.30 → Draw
            [0.50, 0.20, 0.30],  # P(Draw)=0.20 < 0.30 → argmax Away Win
            [0.10, 0.30, 0.60],  # P(Draw)=0.30 >= 0.30 → Draw
            [0.10, 0.29, 0.61],  # P(Draw)=0.29 < 0.30 → argmax Home Win
        ]
    )
    pred = apply_draw_probability_threshold(y_proba, class_names, threshold=0.30)
    assert pred.tolist() == [1, 0, 1, 2]


def test_apply_draw_margin_rule():
    class_names = ["Away Win", "Draw", "Home Win"]
    # Row 0: win=0.40, draw=0.36, gap=0.04 <= 0.05 → Draw
    # Row 1: win=0.50, draw=0.30, gap=0.20 > 0.05 → argmax Home Win
    # Row 2: win=0.42 (Away), draw=0.40, gap=0.02 <= 0.05 → Draw
    # Row 3: top-two gap is small (Home vs Away) but Draw is far behind — must NOT
    # select Draw via a generic top-two rule; win-Draw gap = 0.45-0.10 = 0.35 > 0.05
    y_proba = np.array(
        [
            [0.24, 0.36, 0.40],
            [0.20, 0.30, 0.50],
            [0.42, 0.40, 0.18],
            [0.45, 0.10, 0.45],
        ]
    )
    pred = apply_draw_margin_rule(y_proba, class_names, margin=0.05)
    assert pred.tolist() == [1, 2, 1, 0]  # last row: argmax Away Win (tie → first)


def test_experiment_result_schema_and_ids():
    rng = np.random.default_rng(0)
    n_train, n_test, n_features = 60, 20, 4
    X_train = pd.DataFrame(rng.normal(size=(n_train, n_features)))
    X_test = pd.DataFrame(rng.normal(size=(n_test, n_features)))
    y_train = pd.Series(rng.integers(0, 3, size=n_train))
    y_test = pd.Series(rng.integers(0, 3, size=n_test))
    class_names = ["Away Win", "Draw", "Home Win"]

    results = run_draw_focused_experiments(
        X_train, y_train, X_test, y_test, class_names
    )
    assert list(results.columns) == DRAW_EXPERIMENT_COLUMNS
    assert list(results["experiment_id"]) == [
        "drawexp_01",
        "drawexp_02",
        "drawexp_03",
        "drawexp_04",
        "drawexp_05",
        "drawexp_06",
    ]
    assert results["experiment_id"].nunique() == 6
    assert len(results) == 6
    assert results["feature_set_name"].eq("overall_form").all()


def test_probability_invariance_for_postprocessing_experiments():
    rng = np.random.default_rng(1)
    n_train, n_test, n_features = 60, 20, 4
    X_train = pd.DataFrame(rng.normal(size=(n_train, n_features)))
    X_test = pd.DataFrame(rng.normal(size=(n_test, n_features)))
    y_train = pd.Series(rng.integers(0, 3, size=n_train))
    y_test = pd.Series(rng.integers(0, 3, size=n_test))
    class_names = ["Away Win", "Draw", "Home Win"]

    results = run_draw_focused_experiments(
        X_train, y_train, X_test, y_test, class_names
    )
    by_id = results.set_index("experiment_id")
    baseline = by_id.loc["drawexp_01"]
    for exp_id in ("drawexp_05", "drawexp_06"):
        row = by_id.loc[exp_id]
        assert bool(row["probabilities_changed"]) is False
        assert row["log_loss"] == pytest.approx(baseline["log_loss"])
        assert row["multiclass_brier_score"] == pytest.approx(
            baseline["multiclass_brier_score"]
        )
        assert row["mean_top_probability"] == pytest.approx(
            baseline["mean_top_probability"]
        )
        assert row["mean_probability_gap"] == pytest.approx(
            baseline["mean_probability_gap"]
        )
        assert row["mean_entropy"] == pytest.approx(baseline["mean_entropy"])

    # Same probability vectors → identical diagnostics regardless of hard labels
    y_proba = np.array(
        [
            [0.20, 0.40, 0.40],
            [0.50, 0.20, 0.30],
        ]
    )
    labels = [0, 1, 2]
    y_true = np.array([1, 0])
    diag_baseline = compute_probability_diagnostics(y_true, y_proba, labels=labels)
    _ = apply_draw_probability_threshold(y_proba, class_names)
    _ = apply_draw_margin_rule(y_proba, class_names)
    assert compute_probability_diagnostics(y_true, y_proba, labels=labels) == diag_baseline


def test_save_phase1_draw_experiment_results(tmp_path):
    df = pd.DataFrame(
        [
            {
                col: (
                    "drawexp_01"
                    if col == "experiment_id"
                    else ("baseline" if col in ("intervention_type", "intervention_value") else 0)
                )
                for col in DRAW_EXPERIMENT_COLUMNS
            }
        ]
    )
    # Fill required-ish string columns sanely
    df["model"] = "Logistic regression (overall form)"
    df["decision_rule"] = "argmax"
    df["feature_set_name"] = "overall_form"
    df["probabilities_changed"] = False
    df["notes"] = "test"

    path = save_phase1_draw_experiment_results(df, outputs_dir=tmp_path)
    assert path.name == "phase1_draw_experiment_results.csv"
    assert path.exists()
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == DRAW_EXPERIMENT_COLUMNS
    assert len(loaded) == 1
