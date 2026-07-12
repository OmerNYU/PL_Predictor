from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

PHASE1_SPLIT_METHOD = "chronological_80_20_by_match_date"
_ENTROPY_EPS = 1e-12

CONFIDENCE_TIERS = ("High", "Medium", "Low")
CONFIDENCE_HIGH_TOP = 0.55
CONFIDENCE_HIGH_GAP = 0.15
CONFIDENCE_MEDIUM_TOP = 0.45
CONFIDENCE_MEDIUM_GAP = 0.08

EXPERIMENT_SUMMARY_DISPLAY_COLUMNS = [
    "experiment_id",
    "model",
    "features",
    "split_method",
    "accuracy",
    "macro_f1",
    "notes",
]


def compute_per_class_metrics(
    y_true,
    y_pred,
    target_names: Sequence[str],
) -> list[dict]:
    """Per-class precision, recall, F1, and support (zero_division=0)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.arange(len(target_names), dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return [
        {
            "class_name": target_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(target_names))
    ]


def compute_prediction_distribution(
    y_pred,
    target_names: Sequence[str],
) -> list[dict]:
    """Predicted class counts and proportions for every class in target_names."""
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_pred)
    labels = np.arange(len(target_names), dtype=int)
    counts = np.bincount(y_pred.astype(int), minlength=len(target_names))
    rows = []
    for i, lab in enumerate(labels):
        count = int(counts[lab]) if lab < len(counts) else 0
        proportion = float(count / n) if n > 0 else 0.0
        rows.append(
            {
                "predicted_class": target_names[i],
                "count": count,
                "proportion": proportion,
            }
        )
    return rows


def compute_per_sample_probability_features(y_proba) -> dict:
    """Per-row top/second probability, gap, and entropy from a probability matrix."""
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D array of shape (n_samples, n_classes)")
    n_classes = y_proba.shape[1]
    sorted_probs = np.sort(y_proba, axis=1)
    top = sorted_probs[:, -1]
    second = (
        sorted_probs[:, -2] if n_classes >= 2 else np.zeros(len(y_proba), dtype=float)
    )
    gap = top - second
    entropy = -np.sum(y_proba * np.log(y_proba + _ENTROPY_EPS), axis=1)
    return {
        "top_probability": top,
        "second_probability": second,
        "probability_gap": gap,
        "entropy": entropy,
    }


def compute_probability_diagnostics(
    y_true,
    y_proba,
    *,
    labels: Sequence[int],
) -> dict:
    """Probability-quality diagnostics for models that expose predict_proba."""
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)

    ll = float(log_loss(y_true, y_proba, labels=labels_arr))

    one_hot = np.zeros_like(y_proba)
    # Map true labels to column indices matching labels_arr order
    label_to_col = {int(lab): i for i, lab in enumerate(labels_arr)}
    for i, yt in enumerate(y_true):
        col = label_to_col[int(yt)]
        one_hot[i, col] = 1.0
    brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))

    feats = compute_per_sample_probability_features(y_proba)

    return {
        "log_loss": ll,
        "multiclass_brier_score": brier,
        "mean_top_probability": float(np.mean(feats["top_probability"])),
        "mean_probability_gap": float(np.mean(feats["probability_gap"])),
        "mean_entropy": float(np.mean(feats["entropy"])),
    }


def assign_confidence_tiers(
    y_proba,
    *,
    high_top: float = CONFIDENCE_HIGH_TOP,
    high_gap: float = CONFIDENCE_HIGH_GAP,
    medium_top: float = CONFIDENCE_MEDIUM_TOP,
    medium_gap: float = CONFIDENCE_MEDIUM_GAP,
) -> np.ndarray:
    """Assign High / Medium / Low confidence tiers from predicted probabilities."""
    feats = compute_per_sample_probability_features(y_proba)
    top = feats["top_probability"]
    gap = feats["probability_gap"]
    tiers = np.full(len(top), "Low", dtype=object)
    medium = (top >= medium_top) & (gap >= medium_gap)
    high = (top >= high_top) & (gap >= high_gap)
    tiers[medium] = "Medium"
    tiers[high] = "High"
    return tiers


def compute_confidence_diagnostics(
    y_true,
    y_pred,
    y_proba,
    target_names: Sequence[str],
) -> list[dict]:
    """Aggregate accuracy and draw stats within each confidence tier."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba, dtype=float)
    feats = compute_per_sample_probability_features(y_proba)
    tiers = assign_confidence_tiers(y_proba)
    n = len(y_true)
    draw_idx = list(target_names).index("Draw")

    rows: list[dict] = []
    for tier in CONFIDENCE_TIERS:
        mask = tiers == tier
        count = int(mask.sum())
        proportion = float(count / n) if n > 0 else 0.0
        if count == 0:
            rows.append(
                {
                    "confidence_tier": tier,
                    "count": 0,
                    "proportion": 0.0,
                    "accuracy": float("nan"),
                    "mean_top_probability": float("nan"),
                    "mean_probability_gap": float("nan"),
                    "mean_entropy": float("nan"),
                    "predicted_draw_count": 0,
                    "predicted_draw_proportion": 0.0,
                    "actual_draw_count": 0,
                    "actual_draw_proportion": 0.0,
                }
            )
            continue

        pred_draw = int(np.sum(y_pred[mask] == draw_idx))
        actual_draw = int(np.sum(y_true[mask] == draw_idx))
        rows.append(
            {
                "confidence_tier": tier,
                "count": count,
                "proportion": proportion,
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
                "mean_top_probability": float(np.mean(feats["top_probability"][mask])),
                "mean_probability_gap": float(np.mean(feats["probability_gap"][mask])),
                "mean_entropy": float(np.mean(feats["entropy"][mask])),
                "predicted_draw_count": pred_draw,
                "predicted_draw_proportion": float(pred_draw / count),
                "actual_draw_count": actual_draw,
                "actual_draw_proportion": float(actual_draw / count),
            }
        )
    return rows


def attach_prediction_diagnostics(
    eval_dict: dict,
    y_true,
    y_pred,
    *,
    target_names: Sequence[str],
    y_proba=None,
) -> dict:
    """Attach per-class, distribution, and optional probability diagnostics to an eval dict."""
    eval_dict = dict(eval_dict)
    eval_dict["class_metrics"] = compute_per_class_metrics(
        y_true, y_pred, target_names
    )
    eval_dict["prediction_distribution"] = compute_prediction_distribution(
        y_pred, target_names
    )
    if y_proba is not None:
        labels = np.arange(len(target_names), dtype=int)
        eval_dict["probability_diagnostics"] = compute_probability_diagnostics(
            y_true, y_proba, labels=labels
        )
        eval_dict["confidence_diagnostics"] = compute_confidence_diagnostics(
            y_true, y_pred, y_proba, target_names
        )
    return eval_dict


def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    *,
    target_names: Optional[Sequence[str]] = None,
    print_report: bool = True,
    y_proba=None,
) -> dict:
    """
    Shared metrics for any classifier: accuracy, macro F1, confusion matrix.

    y_true and y_pred should be aligned (same length) and use the same label encoding.
    Pass target_names (e.g. le.classes_) for readable confusion-matrix headers in the printed summary.
    Set print_report=False to only compute metrics (e.g. for side-by-side comparison tables).
    When y_proba is provided (logistic models), also attach probability diagnostics.
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

    result = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "labels": labels,
    }
    if target_names is not None:
        result = attach_prediction_diagnostics(
            result,
            y_true,
            y_pred,
            target_names=target_names,
            y_proba=y_proba,
        )
    return result


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


def draw_metrics_from_eval(eval_dict: dict) -> dict:
    """Extract draw-specific fields from an eval dict's attached diagnostics."""
    class_metrics = {row["class_name"]: row for row in eval_dict.get("class_metrics", [])}
    draw = class_metrics.get(
        "Draw",
        {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    )
    dist = {
        row["predicted_class"]: row
        for row in eval_dict.get("prediction_distribution", [])
    }
    draw_dist = dist.get("Draw", {"count": 0, "proportion": 0.0})
    return {
        "draw_precision": float(draw["precision"]),
        "draw_recall": float(draw["recall"]),
        "draw_f1": float(draw["f1"]),
        "predicted_draw_count": int(draw_dist["count"]),
        "predicted_draw_proportion": float(draw_dist["proportion"]),
    }


def build_experiment_results(
    phase1_evals: list[dict],
    experiment_specs: list[dict],
    *,
    split_method: str = PHASE1_SPLIT_METHOD,
) -> pd.DataFrame:
    eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
    rows = []
    for spec in experiment_specs:
        ev = eval_by_model_name[spec["eval_model_name"]]
        draw = draw_metrics_from_eval(ev)
        rows.append(
            {
                "experiment_id": spec["experiment_id"],
                "model": spec["model"],
                "features": spec["features"],
                "split_method": split_method,
                "accuracy": ev["accuracy"],
                "macro_f1": ev["macro_f1"],
                "draw_precision": draw["draw_precision"],
                "draw_recall": draw["draw_recall"],
                "draw_f1": draw["draw_f1"],
                "predicted_draw_count": draw["predicted_draw_count"],
                "predicted_draw_proportion": draw["predicted_draw_proportion"],
                "notes": spec["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_class_metrics_table(
    phase1_evals: list[dict],
    experiment_specs: list[dict],
) -> pd.DataFrame:
    eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
    rows = []
    for spec in experiment_specs:
        ev = eval_by_model_name[spec["eval_model_name"]]
        for cm in ev.get("class_metrics", []):
            rows.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "model": spec["model"],
                    "class_name": cm["class_name"],
                    "precision": cm["precision"],
                    "recall": cm["recall"],
                    "f1": cm["f1"],
                    "support": cm["support"],
                }
            )
    return pd.DataFrame(rows)


def build_prediction_distribution_table(
    phase1_evals: list[dict],
    experiment_specs: list[dict],
) -> pd.DataFrame:
    eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
    rows = []
    for spec in experiment_specs:
        ev = eval_by_model_name[spec["eval_model_name"]]
        for dist in ev.get("prediction_distribution", []):
            rows.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "model": spec["model"],
                    "predicted_class": dist["predicted_class"],
                    "count": dist["count"],
                    "proportion": dist["proportion"],
                }
            )
    return pd.DataFrame(rows)


def build_probability_diagnostics_table(
    phase1_evals: list[dict],
    experiment_specs: list[dict],
) -> pd.DataFrame:
    eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
    rows = []
    for spec in experiment_specs:
        ev = eval_by_model_name[spec["eval_model_name"]]
        prob = ev.get("probability_diagnostics")
        if prob is None:
            continue
        rows.append(
            {
                "experiment_id": spec["experiment_id"],
                "model": spec["model"],
                "log_loss": prob["log_loss"],
                "multiclass_brier_score": prob["multiclass_brier_score"],
                "mean_top_probability": prob["mean_top_probability"],
                "mean_probability_gap": prob["mean_probability_gap"],
                "mean_entropy": prob["mean_entropy"],
            }
        )
    return pd.DataFrame(rows)


def build_confidence_diagnostics_table(
    phase1_evals: list[dict],
    experiment_specs: list[dict],
) -> pd.DataFrame:
    eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
    rows = []
    for spec in experiment_specs:
        ev = eval_by_model_name[spec["eval_model_name"]]
        conf = ev.get("confidence_diagnostics")
        if conf is None:
            continue
        for row in conf:
            rows.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "model": spec["model"],
                    "confidence_tier": row["confidence_tier"],
                    "count": row["count"],
                    "proportion": row["proportion"],
                    "accuracy": row["accuracy"],
                    "mean_top_probability": row["mean_top_probability"],
                    "mean_probability_gap": row["mean_probability_gap"],
                    "mean_entropy": row["mean_entropy"],
                    "predicted_draw_count": row["predicted_draw_count"],
                    "predicted_draw_proportion": row["predicted_draw_proportion"],
                    "actual_draw_count": row["actual_draw_count"],
                    "actual_draw_proportion": row["actual_draw_proportion"],
                }
            )
    return pd.DataFrame(rows)


def print_experiment_summary(experiment_results: pd.DataFrame) -> None:
    print(f"\n{'=' * 72}")
    print("Phase 1 — evaluation summary (held-out test set)")
    print(f"{'=' * 72}")
    display_cols = [
        c for c in EXPERIMENT_SUMMARY_DISPLAY_COLUMNS if c in experiment_results.columns
    ]
    display_df = experiment_results[display_cols]
    with pd.option_context("display.max_colwidth", None):
        print(
            display_df.to_string(
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
