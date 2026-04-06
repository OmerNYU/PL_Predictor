import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from typing import Optional, Sequence
import seaborn as sns
import numpy as np

df = pd.read_csv("premier-league-matches.csv")
_n_rows_loaded = len(df)
_raw_nulls = df.isnull().sum()

df["result"] = df["FTR"].map({"H": "Home Win", "A": "Away Win", "D": "Draw"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Never use these (or direct derivatives) as model inputs — same-fixture outcome.
_OUTCOME_LEAKAGE_COLS = frozenset(
    {
        "HomeGoals",
        "AwayGoals",
        "FTR",
        "result",
        "result_encoded",
        "home_points",
        "away_points",
        "home_goal_diff",
        "away_goal_diff",
    }
)

# Per-row helpers (outcome-based); only shifted rolling means below enter the model.
df["home_points"] = df["FTR"].map({"H": 3, "D": 1, "A": 0})
df["away_points"] = df["FTR"].map({"A": 3, "D": 1, "H": 0})
df["home_goal_diff"] = df["HomeGoals"] - df["AwayGoals"]
df["away_goal_diff"] = df["AwayGoals"] - df["HomeGoals"]

# Rolling prematch stats: shift(1) within team×role stream excludes the current fixture.
df["home_goals_avg"] = df.groupby("Home")["HomeGoals"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["away_goals_avg"] = df.groupby("Away")["AwayGoals"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["home_conceded_avg"] = df.groupby("Home")["AwayGoals"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["away_conceded_avg"] = df.groupby("Away")["HomeGoals"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["home_points_avg"] = df.groupby("Home")["home_points"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["away_points_avg"] = df.groupby("Away")["away_points"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["home_goal_diff_avg"] = df.groupby("Home")["home_goal_diff"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df["away_goal_diff_avg"] = df.groupby("Away")["away_goal_diff"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df = df.dropna()

print("\nPhase 1 — dataset")
print(f"  Rows loaded from CSV:     {_n_rows_loaded}")
print(f"  Rows in modeling cohort:  {len(df)}  (after rolling features and complete cases)")
print(
    f"  Cohort date span:         {df['Date'].min().date()} — {df['Date'].max().date()}"
)
if _raw_nulls.sum() > 0:
    print("  Missing values in raw CSV (by column):")
    print(_raw_nulls[_raw_nulls > 0].to_string(header=False))

_class_vc = df["result"].value_counts()
_class_prop = df["result"].value_counts(normalize=True)
_class_summary = pd.DataFrame(
    {"count": _class_vc, "proportion": _class_prop.round(4)}
).sort_values("count", ascending=False)
_class_summary.index.name = None
print("\nPhase 1 — class distribution (modeling cohort)")
print(_class_summary.to_string())

le_home = LabelEncoder()
le_away = LabelEncoder()
le_result = LabelEncoder()

df["home_encoded"] = le_home.fit_transform(df["Home"])
df["away_encoded"] = le_away.fit_transform(df["Away"])
df["result_encoded"] = le_result.fit_transform(df["result"])

features_core = [
    "home_encoded",
    "away_encoded",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
]
features = features_core + [
    "home_points_avg",
    "away_points_avg",
    "home_goal_diff_avg",
    "away_goal_diff_avg",
]
_overlap = _OUTCOME_LEAKAGE_COLS.intersection(features)
if _overlap:
    raise ValueError(
        "Target leakage: these columns must not be model inputs: "
        + ", ".join(sorted(_overlap))
    )

X = df[features]
X_core = df[features_core]
y = df["result_encoded"]

split_idx = int(len(df) * 0.8)
n_train, n_test = split_idx, len(df) - split_idx
_train_dates = df["Date"].iloc[:split_idx]
_test_dates = df["Date"].iloc[split_idx:]

print("\nPhase 1 — chronological split (rows ordered by Date; no shuffle)")
print("  Policy: first 80% of rows → train, remainder → test.")
print(f"  Train rows: {n_train}    date range: {_train_dates.min().date()} — {_train_dates.max().date()}")
print(f"  Test rows:  {n_test}    date range: {_test_dates.min().date()} — {_test_dates.max().date()}")
print("  All reported metrics below use the test slice only.")

print("\nPhase 1 — model features (prematch only; no same-match scores or result)")
for _i, _name in enumerate(features, start=1):
    print(f"  {_i}. {_name}")
print(
    "  Rolling *_avg: prior up-to-5 same-role matches per team (shift(1).rolling(5))."
)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
X_train_core, X_test_core = X_core.iloc[:split_idx], X_core.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


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


# Same test window as the classifier.
baseline_pred_always_home_win = baseline_always_home_win(len(y_test), le_result)
baseline_pred_most_frequent = baseline_most_frequent_class(y_train, len(y_test))
baseline_pred_random_weighted = baseline_random_by_train_freq(y_train, len(y_test))


model_lr_core = LogisticRegression(max_iter=1000)
model_lr_core.fit(X_train_core, y_train)
predictions_core = model_lr_core.predict(X_test_core)

model_lr_form = LogisticRegression(max_iter=1000)
model_lr_form.fit(X_train, y_train)
predictions_form = model_lr_form.predict(X_test)

class_names = list(le_result.classes_)

phase1_evals = [
    evaluate_predictions(
        "Logistic regression",
        y_test,
        predictions_core,
        target_names=class_names,
        print_report=False,
    ),
    evaluate_predictions(
        "Baseline: always home win",
        y_test,
        baseline_pred_always_home_win,
        target_names=class_names,
        print_report=False,
    ),
    evaluate_predictions(
        "Baseline: majority class",
        y_test,
        baseline_pred_most_frequent,
        target_names=class_names,
        print_report=False,
    ),
    evaluate_predictions(
        "Baseline: random (train class frequencies)",
        y_test,
        baseline_pred_random_weighted,
        target_names=class_names,
        print_report=False,
    ),
    evaluate_predictions(
        "Logistic regression (rolling form)",
        y_test,
        predictions_form,
        target_names=class_names,
        print_report=False,
    ),
]

# Phase 1 experiment manifest (add rows for new runs).
eval_by_model_name = {e["model_name"]: e for e in phase1_evals}
_phase1_feature_set_core = ", ".join(features_core)
_phase1_feature_set_full = ", ".join(features)
_phase1_split = "chronological_80_20_by_match_date"
_phase1_experiment_rows = [
    {
        "experiment_id": "phase1_01",
        "model": "Always Home Win",
        "eval_model_name": "Baseline: always home win",
        "features": _phase1_feature_set_core,
        "notes": "Predict Home Win for every test match.",
    },
    {
        "experiment_id": "phase1_02",
        "model": "Most Frequent Class",
        "eval_model_name": "Baseline: majority class",
        "features": _phase1_feature_set_core,
        "notes": "Predict the majority class from the training set.",
    },
    {
        "experiment_id": "phase1_03",
        "model": "Class-Frequency Random",
        "eval_model_name": "Baseline: random (train class frequencies)",
        "features": _phase1_feature_set_core,
        "notes": "Random labels sampled from training class frequencies (random_state=42).",
    },
    {
        "experiment_id": "phase1_04",
        "model": "Logistic Regression",
        "eval_model_name": "Logistic regression",
        "features": _phase1_feature_set_core,
        "notes": "sklearn LogisticRegression; max_iter=1000; default hyperparameters.",
    },
    {
        "experiment_id": "phase1_05",
        "model": "Logistic Regression (rolling form)",
        "eval_model_name": "Logistic regression (rolling form)",
        "features": _phase1_feature_set_full,
        "notes": (
            "Adds home_points_avg, away_points_avg, home_goal_diff_avg, away_goal_diff_avg "
            "(prior up to 5 same-role matches, shift(1)); same LR defaults as phase1_04."
        ),
    },
]
experiment_results = pd.DataFrame(
    [
        {
            "experiment_id": spec["experiment_id"],
            "model": spec["model"],
            "features": spec["features"],
            "split_method": _phase1_split,
            "accuracy": eval_by_model_name[spec["eval_model_name"]]["accuracy"],
            "macro_f1": eval_by_model_name[spec["eval_model_name"]]["macro_f1"],
            "notes": spec["notes"],
        }
        for spec in _phase1_experiment_rows
    ]
)
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

print(f"\n{'—' * 72}")
print("Phase 1 — confusion matrices (test set; rows = actual, cols = predicted)")
print(f"{'—' * 72}")
for e in phase1_evals:
    print_confusion_matrix_compact(
        e["model_name"], e["confusion_matrix"], target_names=class_names
    )


lr_form_eval = phase1_evals[-1]
cm = lr_form_eval["confusion_matrix"]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix — Logistic regression (rolling form)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
