from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from matchlens.evaluation import PHASE1_SPLIT_METHOD


def save_best_logistic_confusion_matrix(
    cm,
    *,
    model_name: str,
    class_names: list[str],
    outputs_dir: Path | str = "outputs",
) -> Path:
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    output_path = outputs_path / "best_logistic_confusion_matrix.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    return output_path


def save_phase1_outputs(
    *,
    experiment_results: pd.DataFrame,
    n_rows_loaded: int,
    df,
    n_train: int,
    n_test: int,
    train_dates: pd.Series,
    test_dates: pd.Series,
    best_logistic_eval: dict,
    class_metrics: pd.DataFrame | None = None,
    prediction_distribution: pd.DataFrame | None = None,
    probability_diagnostics: pd.DataFrame | None = None,
    confidence_diagnostics: pd.DataFrame | None = None,
    outputs_dir: Path | str = "outputs",
) -> None:
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    experiment_results.to_csv(outputs_path / "phase1_experiment_results.csv", index=False)

    if class_metrics is not None:
        class_metrics.to_csv(outputs_path / "phase1_class_metrics.csv", index=False)
    if prediction_distribution is not None:
        prediction_distribution.to_csv(
            outputs_path / "phase1_prediction_distribution.csv", index=False
        )
    if probability_diagnostics is not None:
        probability_diagnostics.to_csv(
            outputs_path / "phase1_probability_diagnostics.csv", index=False
        )
    if confidence_diagnostics is not None:
        confidence_diagnostics.to_csv(
            outputs_path / "phase1_confidence_diagnostics.csv", index=False
        )

    phase1_metadata = {
        "rows_loaded": n_rows_loaded,
        "modeling_cohort_size": len(df),
        "cohort_start_date": df["Date"].min().date().isoformat(),
        "cohort_end_date": df["Date"].max().date().isoformat(),
        "train_rows": n_train,
        "test_rows": n_test,
        "train_start_date": train_dates.min().date().isoformat(),
        "train_end_date": train_dates.max().date().isoformat(),
        "test_start_date": test_dates.min().date().isoformat(),
        "test_end_date": test_dates.max().date().isoformat(),
        "split_method": PHASE1_SPLIT_METHOD,
        "best_logistic_model_name": best_logistic_eval["model_name"],
        "best_logistic_accuracy": best_logistic_eval["accuracy"],
        "best_logistic_macro_f1": best_logistic_eval["macro_f1"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(outputs_path / "phase1_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(phase1_metadata, f, indent=2)
        f.write("\n")

    print("\nPhase 1 artifacts saved to outputs/")
    if (
        class_metrics is not None
        or prediction_distribution is not None
        or probability_diagnostics is not None
    ):
        print("Phase 1 evaluation diagnostics saved to outputs/")
    if confidence_diagnostics is not None:
        print("Phase 1 confidence diagnostics saved to outputs/")


def save_phase1_models(
    *,
    best_model,
    best_model_spec: dict,
    best_logistic_eval: dict,
    le_home: LabelEncoder,
    le_away: LabelEncoder,
    le_result: LabelEncoder,
    class_names: list[str],
    train_dates: pd.Series,
    test_dates: pd.Series,
    models_dir: Path | str = "models",
) -> None:
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_path / "best_logistic_model.joblib")
    joblib.dump(
        {"home": le_home, "away": le_away, "result": le_result},
        models_path / "label_encoders.joblib",
    )

    best_model_metadata = {
        "model_name": best_logistic_eval["model_name"],
        "model_type": "LogisticRegression",
        "feature_set_name": best_model_spec["feature_set_name"],
        "features": list(best_model_spec["features"]),
        "class_names": class_names,
        "accuracy": best_logistic_eval["accuracy"],
        "macro_f1": best_logistic_eval["macro_f1"],
        "split_method": PHASE1_SPLIT_METHOD,
        "train_start_date": train_dates.min().date().isoformat(),
        "train_end_date": train_dates.max().date().isoformat(),
        "test_start_date": test_dates.min().date().isoformat(),
        "test_end_date": test_dates.max().date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(models_path / "best_logistic_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(best_model_metadata, f, indent=2)
        f.write("\n")

    print("Phase 1 model artifacts saved to models/")


def save_phase1_backtest_outputs(
    *,
    backtest_results: pd.DataFrame,
    backtest_summary: dict,
    outputs_dir: Path | str = "outputs",
) -> None:
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    backtest_results.to_csv(outputs_path / "phase1_backtest_results.csv", index=False)
    with open(outputs_path / "phase1_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(backtest_summary, f, indent=2)
        f.write("\n")

    print("Phase 1 backtesting artifacts saved to outputs/")


def save_phase1_draw_experiment_results(
    draw_results: pd.DataFrame,
    *,
    outputs_dir: Path | str = "outputs",
) -> Path:
    """Save exploratory draw-focused experiment results (does not affect best model)."""
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    output_path = outputs_path / "phase1_draw_experiment_results.csv"
    draw_results.to_csv(output_path, index=False)
    print("Phase 1 draw-focused experiment results saved to outputs/")
    return output_path
