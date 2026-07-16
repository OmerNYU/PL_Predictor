"""Exploratory draw-focused Phase 1 experiments (class weights and decision rules).

This track does not replace the official Phase 1 baseline or saved best model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matchlens.evaluation import draw_metrics_from_eval, evaluate_predictions

DRAW_EXPERIMENT_COLUMNS = [
    "experiment_id",
    "model",
    "intervention_type",
    "intervention_value",
    "decision_rule",
    "feature_set_name",
    "accuracy",
    "macro_f1",
    "draw_precision",
    "draw_recall",
    "draw_f1",
    "predicted_draw_count",
    "predicted_draw_proportion",
    "log_loss",
    "multiclass_brier_score",
    "mean_top_probability",
    "mean_probability_gap",
    "mean_entropy",
    "probabilities_changed",
    "notes",
]

DRAW_SUMMARY_DISPLAY_COLUMNS = [
    "experiment_id",
    "accuracy",
    "macro_f1",
    "draw_precision",
    "draw_recall",
    "draw_f1",
    "predicted_draw_proportion",
]

FEATURE_SET_NAME = "overall_form"


def build_draw_class_weight(
    class_names: Sequence[str],
    draw_weight: float,
) -> dict[int, float]:
    """Build sklearn class_weight dict with Draw at draw_weight and others at 1.0."""
    draw_index = list(class_names).index("Draw")
    return {
        i: (float(draw_weight) if i == draw_index else 1.0)
        for i in range(len(class_names))
    }


def apply_draw_probability_threshold(
    y_proba,
    class_names: Sequence[str],
    threshold: float = 0.30,
) -> np.ndarray:
    """Predict Draw when P(Draw) >= threshold; otherwise keep argmax."""
    y_proba = np.asarray(y_proba, dtype=float)
    draw_idx = list(class_names).index("Draw")
    pred = np.argmax(y_proba, axis=1).astype(int)
    pred[y_proba[:, draw_idx] >= threshold] = draw_idx
    return pred


def apply_draw_margin_rule(
    y_proba,
    class_names: Sequence[str],
    margin: float = 0.05,
) -> np.ndarray:
    """Predict Draw when strongest win prob minus P(Draw) <= margin; else argmax."""
    y_proba = np.asarray(y_proba, dtype=float)
    names = list(class_names)
    draw_idx = names.index("Draw")
    home_idx = names.index("Home Win")
    away_idx = names.index("Away Win")
    pred = np.argmax(y_proba, axis=1).astype(int)
    win_probability = np.maximum(y_proba[:, home_idx], y_proba[:, away_idx])
    pred[win_probability - y_proba[:, draw_idx] <= margin] = draw_idx
    return pred


def _row_from_eval(
    *,
    experiment_id: str,
    model: str,
    intervention_type: str,
    intervention_value: str,
    decision_rule: str,
    eval_dict: dict,
    probabilities_changed: bool,
    notes: str,
) -> dict:
    draw = draw_metrics_from_eval(eval_dict)
    prob = eval_dict.get("probability_diagnostics") or {}
    return {
        "experiment_id": experiment_id,
        "model": model,
        "intervention_type": intervention_type,
        "intervention_value": intervention_value,
        "decision_rule": decision_rule,
        "feature_set_name": FEATURE_SET_NAME,
        "accuracy": eval_dict["accuracy"],
        "macro_f1": eval_dict["macro_f1"],
        "draw_precision": draw["draw_precision"],
        "draw_recall": draw["draw_recall"],
        "draw_f1": draw["draw_f1"],
        "predicted_draw_count": draw["predicted_draw_count"],
        "predicted_draw_proportion": draw["predicted_draw_proportion"],
        "log_loss": prob.get("log_loss"),
        "multiclass_brier_score": prob.get("multiclass_brier_score"),
        "mean_top_probability": prob.get("mean_top_probability"),
        "mean_probability_gap": prob.get("mean_probability_gap"),
        "mean_entropy": prob.get("mean_entropy"),
        "probabilities_changed": probabilities_changed,
        "notes": notes,
    }


def run_draw_focused_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Run the six fixed draw-focused experiments on overall-form features."""
    class_names = list(class_names)
    rows: list[dict] = []

    # Experiment 1 — unweighted baseline (mirrors phase1_06)
    model_baseline = LogisticRegression(max_iter=1000)
    model_baseline.fit(X_train, y_train)
    proba_baseline = model_baseline.predict_proba(X_test)
    pred_baseline = model_baseline.predict(X_test)
    eval_01 = evaluate_predictions(
        "Logistic regression (overall form)",
        y_test,
        pred_baseline,
        target_names=class_names,
        print_report=False,
        y_proba=proba_baseline,
    )
    rows.append(
        _row_from_eval(
            experiment_id="drawexp_01",
            model="Logistic regression (overall form)",
            intervention_type="baseline",
            intervention_value="baseline",
            decision_rule="argmax",
            eval_dict=eval_01,
            probabilities_changed=False,
            notes="Unweighted overall-form LR; mirrors phase1_06 hard predictions and metrics.",
        )
    )

    # Experiment 2 — balanced class weights
    model_balanced = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_balanced.fit(X_train, y_train)
    proba_02 = model_balanced.predict_proba(X_test)
    pred_02 = model_balanced.predict(X_test)
    eval_02 = evaluate_predictions(
        "Logistic regression (overall form, balanced)",
        y_test,
        pred_02,
        target_names=class_names,
        print_report=False,
        y_proba=proba_02,
    )
    rows.append(
        _row_from_eval(
            experiment_id="drawexp_02",
            model="Logistic regression (overall form, balanced)",
            intervention_type="class_weight",
            intervention_value="balanced",
            decision_rule="argmax",
            eval_dict=eval_02,
            probabilities_changed=True,
            notes="sklearn class_weight='balanced' on overall-form LR.",
        )
    )

    # Experiment 3 — draw weight 1.5
    cw_15 = build_draw_class_weight(class_names, 1.5)
    model_15 = LogisticRegression(max_iter=1000, class_weight=cw_15)
    model_15.fit(X_train, y_train)
    proba_03 = model_15.predict_proba(X_test)
    pred_03 = model_15.predict(X_test)
    eval_03 = evaluate_predictions(
        "Logistic regression (overall form, draw weight 1.5)",
        y_test,
        pred_03,
        target_names=class_names,
        print_report=False,
        y_proba=proba_03,
    )
    rows.append(
        _row_from_eval(
            experiment_id="drawexp_03",
            model="Logistic regression (overall form, draw weight 1.5)",
            intervention_type="class_weight",
            intervention_value="1.5",
            decision_rule="argmax",
            eval_dict=eval_03,
            probabilities_changed=True,
            notes="Custom class weights: Draw=1.5, Home Win=1.0, Away Win=1.0.",
        )
    )

    # Experiment 4 — draw weight 2.0
    cw_20 = build_draw_class_weight(class_names, 2.0)
    model_20 = LogisticRegression(max_iter=1000, class_weight=cw_20)
    model_20.fit(X_train, y_train)
    proba_04 = model_20.predict_proba(X_test)
    pred_04 = model_20.predict(X_test)
    eval_04 = evaluate_predictions(
        "Logistic regression (overall form, draw weight 2.0)",
        y_test,
        pred_04,
        target_names=class_names,
        print_report=False,
        y_proba=proba_04,
    )
    rows.append(
        _row_from_eval(
            experiment_id="drawexp_04",
            model="Logistic regression (overall form, draw weight 2.0)",
            intervention_type="class_weight",
            intervention_value="2.0",
            decision_rule="argmax",
            eval_dict=eval_04,
            probabilities_changed=True,
            notes="Custom class weights: Draw=2.0, Home Win=1.0, Away Win=1.0.",
        )
    )

    # Experiment 5 — draw probability threshold (baseline proba, hard labels only)
    pred_05 = apply_draw_probability_threshold(
        proba_baseline, class_names, threshold=0.30
    )
    eval_05 = evaluate_predictions(
        "Logistic regression (overall form, draw threshold 0.30)",
        y_test,
        pred_05,
        target_names=class_names,
        print_report=False,
        y_proba=proba_baseline,
    )
    rows.append(
        _row_from_eval(
            experiment_id="drawexp_05",
            model="Logistic regression (overall form, draw threshold 0.30)",
            intervention_type="decision_rule",
            intervention_value="0.30",
            decision_rule="draw_probability_threshold",
            eval_dict=eval_05,
            probabilities_changed=False,
            notes="Post-process baseline proba: predict Draw if P(Draw) >= 0.30.",
        )
    )

    # Experiment 6 — draw vs win margin (baseline proba, hard labels only)
    pred_06 = apply_draw_margin_rule(proba_baseline, class_names, margin=0.05)
    eval_06 = evaluate_predictions(
        "Logistic regression (overall form, draw margin 0.05)",
        y_test,
        pred_06,
        target_names=class_names,
        print_report=False,
        y_proba=proba_baseline,
    )
    rows.append(
        _row_from_eval(
            experiment_id="drawexp_06",
            model="Logistic regression (overall form, draw margin 0.05)",
            intervention_type="decision_rule",
            intervention_value="0.05",
            decision_rule="draw_vs_win_margin",
            eval_dict=eval_06,
            probabilities_changed=False,
            notes=(
                "Post-process baseline proba: predict Draw if "
                "max(P(Home Win), P(Away Win)) - P(Draw) <= 0.05."
            ),
        )
    )

    return pd.DataFrame(rows, columns=DRAW_EXPERIMENT_COLUMNS)


def print_draw_experiment_summary(draw_results: pd.DataFrame) -> None:
    """Print a compact draw-focused experiment summary (no probability columns)."""
    print(f"\n{'=' * 72}")
    print("Phase 1 — draw-focused experiment summary (exploratory)")
    print(f"{'=' * 72}")
    display_cols = [
        c for c in DRAW_SUMMARY_DISPLAY_COLUMNS if c in draw_results.columns
    ]
    display_df = draw_results[display_cols]
    with pd.option_context("display.max_colwidth", None):
        print(
            display_df.to_string(
                index=False,
                formatters={
                    "accuracy": lambda x: f"{x:.4f}",
                    "macro_f1": lambda x: f"{x:.4f}",
                    "draw_precision": lambda x: f"{x:.4f}",
                    "draw_recall": lambda x: f"{x:.4f}",
                    "draw_f1": lambda x: f"{x:.4f}",
                    "predicted_draw_proportion": lambda x: f"{x:.4f}",
                },
            )
        )
