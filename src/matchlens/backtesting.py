from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression

from matchlens.evaluation import draw_metrics_from_eval, evaluate_predictions

MIN_BACKTEST_TRAIN_ROWS = 1000
MIN_BACKTEST_TEST_ROWS = 100
BACKTEST_FEATURE_SET_NAME = "overall_form"
BACKTEST_MODEL_NAME = "Logistic regression (overall form)"

BACKTEST_RESULT_COLUMNS = [
    "test_season",
    "train_start_date",
    "train_end_date",
    "test_start_date",
    "test_end_date",
    "train_rows",
    "test_rows",
    "model",
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
]


def add_season_start_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add season_start_year from Date (Aug–Jul Premier League seasons)."""
    out = df.copy()
    year = out["Date"].dt.year
    month = out["Date"].dt.month
    out["season_start_year"] = year.where(month >= 8, year - 1).astype(int)
    return out


def iter_season_backtest_folds(
    df: pd.DataFrame,
    *,
    min_train_rows: int = MIN_BACKTEST_TRAIN_ROWS,
    min_test_rows: int = MIN_BACKTEST_TEST_ROWS,
) -> list[dict]:
    """
    Build train-on-past / test-on-season folds.

    Requires a season_start_year column. Skips seasons that fail the min-size guards.
    """
    if "season_start_year" not in df.columns:
        raise ValueError("df must include season_start_year; call add_season_start_year first")

    folds: list[dict] = []
    seasons = sorted(df["season_start_year"].unique())
    for test_season in seasons:
        train_mask = df["season_start_year"] < test_season
        test_mask = df["season_start_year"] == test_season
        train_rows = int(train_mask.sum())
        test_rows = int(test_mask.sum())
        if train_rows < min_train_rows or test_rows < min_test_rows:
            continue

        train_dates = df.loc[train_mask, "Date"]
        test_dates = df.loc[test_mask, "Date"]
        folds.append(
            {
                "test_season": int(test_season),
                "train_index": df.index[train_mask],
                "test_index": df.index[test_mask],
                "train_start_date": train_dates.min().date().isoformat(),
                "train_end_date": train_dates.max().date().isoformat(),
                "test_start_date": test_dates.min().date().isoformat(),
                "test_end_date": test_dates.max().date().isoformat(),
                "train_rows": train_rows,
                "test_rows": test_rows,
            }
        )
    return folds


def build_backtest_summary(backtest_results: pd.DataFrame) -> dict:
    """Aggregate mean metrics across backtest seasons."""
    if backtest_results.empty:
        return {
            "number_of_backtest_seasons": 0,
            "first_test_season": None,
            "last_test_season": None,
            "mean_accuracy": None,
            "mean_macro_f1": None,
            "mean_draw_recall": None,
            "mean_log_loss": None,
            "mean_multiclass_brier_score": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    seasons = backtest_results["test_season"].astype(int)
    return {
        "number_of_backtest_seasons": int(len(backtest_results)),
        "first_test_season": int(seasons.min()),
        "last_test_season": int(seasons.max()),
        "mean_accuracy": float(backtest_results["accuracy"].mean()),
        "mean_macro_f1": float(backtest_results["macro_f1"].mean()),
        "mean_draw_recall": float(backtest_results["draw_recall"].mean()),
        "mean_log_loss": float(backtest_results["log_loss"].mean()),
        "mean_multiclass_brier_score": float(
            backtest_results["multiclass_brier_score"].mean()
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_season_backtest(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    class_names: Sequence[str],
    target_col: str = "result_encoded",
    min_train_rows: int = MIN_BACKTEST_TRAIN_ROWS,
    min_test_rows: int = MIN_BACKTEST_TEST_ROWS,
    model_name: str = BACKTEST_MODEL_NAME,
    feature_set_name: str = BACKTEST_FEATURE_SET_NAME,
) -> tuple[pd.DataFrame, dict]:
    """
    Season walk-forward backtest for one logistic feature track.

    Trains LogisticRegression(max_iter=1000) on all prior seasons and evaluates
    on each eligible test season.
    """
    working = df if "season_start_year" in df.columns else add_season_start_year(df)
    folds = iter_season_backtest_folds(
        working,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )

    rows: list[dict] = []
    for fold in folds:
        train_idx = fold["train_index"]
        test_idx = fold["test_index"]
        X_train = working.loc[train_idx, list(features)]
        X_test = working.loc[test_idx, list(features)]
        y_train = working.loc[train_idx, target_col]
        y_test = working.loc[test_idx, target_col]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        ev = evaluate_predictions(
            model_name,
            y_test,
            y_pred,
            target_names=class_names,
            print_report=False,
            y_proba=y_proba,
        )
        draw = draw_metrics_from_eval(ev)
        prob = ev["probability_diagnostics"]

        rows.append(
            {
                "test_season": fold["test_season"],
                "train_start_date": fold["train_start_date"],
                "train_end_date": fold["train_end_date"],
                "test_start_date": fold["test_start_date"],
                "test_end_date": fold["test_end_date"],
                "train_rows": fold["train_rows"],
                "test_rows": fold["test_rows"],
                "model": model_name,
                "feature_set_name": feature_set_name,
                "accuracy": ev["accuracy"],
                "macro_f1": ev["macro_f1"],
                "draw_precision": draw["draw_precision"],
                "draw_recall": draw["draw_recall"],
                "draw_f1": draw["draw_f1"],
                "predicted_draw_count": draw["predicted_draw_count"],
                "predicted_draw_proportion": draw["predicted_draw_proportion"],
                "log_loss": prob["log_loss"],
                "multiclass_brier_score": prob["multiclass_brier_score"],
                "mean_top_probability": prob["mean_top_probability"],
                "mean_probability_gap": prob["mean_probability_gap"],
                "mean_entropy": prob["mean_entropy"],
            }
        )

    backtest_results = pd.DataFrame(rows, columns=BACKTEST_RESULT_COLUMNS)
    summary = build_backtest_summary(backtest_results)
    return backtest_results, summary


def print_backtest_summary(summary: dict) -> None:
    """Print a short backtesting summary section."""
    print("\nPhase 1 — backtesting summary")
    n = summary.get("number_of_backtest_seasons", 0)
    if not n:
        print("  No eligible backtest seasons (min train/test size not met).")
        return

    first = summary["first_test_season"]
    last = summary["last_test_season"]
    print(f"  Test seasons: {first}–{last}")
    print(f"  Mean accuracy: {summary['mean_accuracy']:.4f}")
    print(f"  Mean macro F1: {summary['mean_macro_f1']:.4f}")
    print(f"  Mean draw recall: {summary['mean_draw_recall']:.4f}")
