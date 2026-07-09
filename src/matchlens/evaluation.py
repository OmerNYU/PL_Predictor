from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

PHASE1_SPLIT_METHOD = "chronological_80_20_by_match_date"


def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    *,
    target_names: Optional[Sequence[str]] = None,
    print_report: bool = True,
) -> dict:
    """
    Shared metrics for any classifier: accuracy, macro F1, confusion matrix.

    y_true and y_pred should be aligned (same length) and use the same label encoding.
    Pass target_names (e.g. le.classes_) for readable confusion-matrix headers in the printed summary.
    Set print_report=False to only compute metrics (e.g. for side-by-side comparison tables).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if target_names is not None:
        labels = np.arange(len(target_names), dtype=int)
    else:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    name_for = (
        {i: target_names[i] for i in range(len(target_names))}
        if target_names is not None
        else {lab: str(lab) for lab in labels}
    )

    if print_report:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print("Confusion matrix (rows = actual, columns = predicted):")
        col_hdr = "".join(f"{name_for[int(l)]:>14}" for l in labels)
        print(f"{'':>12}{col_hdr}")
        for i, row_label in enumerate(labels):
            row_parts = "".join(f"{cm[i, j]:>14}" for j in range(len(labels)))
            print(f"{name_for[int(row_label)]:>12}{row_parts}")

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "labels": labels,
    }


def print_confusion_matrix_compact(
    model_name: str,
    cm: np.ndarray,
    *,
    target_names: Sequence[str],
) -> None:
    """Text confusion matrix with a short header (no metric banners)."""
    labels = np.arange(len(target_names), dtype=int)
    name_for = {i: target_names[i] for i in range(len(target_names))}
    print(f"\n{model_name}")
    print("  Confusion matrix (rows = actual, columns = predicted):")
    col_hdr = "".join(f"{name_for[int(l)]:>14}" for l in labels)
    print(f"  {'':>10}{col_hdr}")
    for i, row_label in enumerate(labels):
        row_parts = "".join(f"{cm[i, j]:>14}" for j in range(len(labels)))
        print(f"  {name_for[int(row_label)]:>10}{row_parts}")


def baseline_always_home_win(n: int, le: LabelEncoder) -> np.ndarray:
    """Always predict home win."""
    home_win_encoded = int(le.transform(["Home Win"])[0])
    return np.full(shape=n, fill_value=home_win_encoded, dtype=int)


def baseline_most_frequent_class(y_train: pd.Series, n: int) -> np.ndarray:
    """Majority class from the training labels."""
    most_frequent = int(y_train.value_counts().idxmax())
    return np.full(shape=n, fill_value=most_frequent, dtype=int)


def baseline_random_by_train_freq(
    y_train: pd.Series, n: int, *, random_state: int = 42
) -> np.ndarray:
    """Random labels with training class frequencies."""
    freqs = y_train.value_counts(normalize=True).sort_index()
    classes = freqs.index.to_numpy(dtype=int)
    probs = freqs.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    return rng.choice(classes, size=n, replace=True, p=probs)


def build_phase1_experiment_specs(
    *,
    features_core: list[str],
    features_venue_form: list[str],
    features_overall: list[str],
    features_overall_diff: list[str],
) -> list[dict]:
    feature_set_core = ", ".join(features_core)
    feature_set_full = ", ".join(features_venue_form)
    feature_set_overall = ", ".join(features_overall)
    feature_set_overall_diff = ", ".join(features_overall_diff)

    return [
        {
            "experiment_id": "phase1_01",
            "model": "Always Home Win",
            "eval_model_name": "Baseline: always home win",
            "features": feature_set_core,
            "notes": "Predict Home Win for every test match.",
        },
        {
            "experiment_id": "phase1_02",
            "model": "Most Frequent Class",
            "eval_model_name": "Baseline: majority class",
            "features": feature_set_core,
            "notes": "Predict the majority class from the training set.",
        },
        {
            "experiment_id": "phase1_03",
            "model": "Class-Frequency Random",
            "eval_model_name": "Baseline: random (train class frequencies)",
            "features": feature_set_core,
            "notes": "Random labels sampled from training class frequencies (random_state=42).",
        },
        {
            "experiment_id": "phase1_04",
            "model": "Logistic Regression",
            "eval_model_name": "Logistic regression",
            "features": feature_set_core,
            "notes": "sklearn LogisticRegression; max_iter=1000; default hyperparameters.",
        },
        {
            "experiment_id": "phase1_05",
            "model": "Logistic Regression (rolling form)",
            "eval_model_name": "Logistic regression (rolling form)",
            "features": feature_set_full,
            "notes": (
                "Adds home_points_avg, away_points_avg, home_goal_diff_avg, away_goal_diff_avg "
                "(prior up to 5 same-role matches, shift(1)); same LR defaults as phase1_04."
            ),
        },
        {
            "experiment_id": "phase1_06",
            "model": "Logistic Regression (overall form)",
            "eval_model_name": "Logistic regression (overall form)",
            "features": feature_set_overall,
            "notes": (
                "Adds home_team_points_avg_overall, away_team_points_avg_overall, "
                "home_team_goal_diff_avg_overall, away_team_goal_diff_avg_overall "
                "(prior up to 5 matches in all venues per team, shift(1)); same LR defaults as phase1_04."
            ),
        },
        {
            "experiment_id": "phase1_07",
            "model": "Logistic Regression (overall form + matchup diff)",
            "eval_model_name": "Logistic regression (overall form + matchup diff)",
            "features": feature_set_overall_diff,
            "notes": (
                "Adds points_avg_overall_diff and goal_diff_avg_overall_diff (home minus away) "
                "on top of the phase1_06 overall-form features; same LR defaults as phase1_04."
            ),
        },
    ]


def build_experiment_results(
    phase1_evals: list[dict],
    experiment_specs: list[dict],
    *,
    split_method: str = PHASE1_SPLIT_METHOD,
) -> pd.DataFrame:
    eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
    return pd.DataFrame(
        [
            {
                "experiment_id": spec["experiment_id"],
                "model": spec["model"],
                "features": spec["features"],
                "split_method": split_method,
                "accuracy": eval_by_model_name[spec["eval_model_name"]]["accuracy"],
                "macro_f1": eval_by_model_name[spec["eval_model_name"]]["macro_f1"],
                "notes": spec["notes"],
            }
            for spec in experiment_specs
        ]
    )


def print_experiment_summary(experiment_results: pd.DataFrame) -> None:
    print(f"\n{'=' * 72}")
    print("Phase 1 — evaluation summary (held-out test set)")
    print(f"{'=' * 72}")
    with pd.option_context("display.max_colwidth", None):
        print(
            experiment_results.to_string(
                index=False,
                formatters={
                    "accuracy": lambda x: f"{x:.4f}",
                    "macro_f1": lambda x: f"{x:.4f}",
                },
            )
        )


def print_all_confusion_matrices(
    phase1_evals: list[dict],
    *,
    target_names: Sequence[str],
) -> None:
    print(f"\n{'—' * 72}")
    print("Phase 1 — confusion matrices (test set; rows = actual, cols = predicted)")
    print(f"{'—' * 72}")
    for e in phase1_evals:
        print_confusion_matrix_compact(
            e["model_name"], e["confusion_matrix"], target_names=target_names
        )


def select_best_logistic_eval(phase1_evals: list[dict]) -> dict:
    logistic_evals = [
        e for e in phase1_evals if e["model_name"].startswith("Logistic regression")
    ]
    return max(logistic_evals, key=lambda e: (e["accuracy"], e["macro_f1"]))
