import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from typing import Optional, Sequence
import seaborn as sns
import numpy as np

df = pd.read_csv('premier-league-matches.csv')

print(df.shape)    #how many rows and columns are there
print(df.head())   #first 5 rows of the dataframe
print(df.dtypes)   #data types of the columns
print(df.isnull().sum())  #check for null values


# decode the result column first
df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})
# Ensure match dates are datetime so ordering is truly chronological.
df['Date'] = pd.to_datetime(df['Date'])
# Sort by date before any split to avoid leaking future matches into training.
df = df.sort_values('Date').reset_index(drop=True)

# build pre-match rolling averages (what we'd actually know before kickoff)
df['home_goals_avg'] = df.groupby('Home')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_goals_avg'] = df.groupby('Away')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df = df.dropna()

le_home = LabelEncoder()
le_away = LabelEncoder()
le_result = LabelEncoder()

df['home_encoded'] = le_home.fit_transform(df['Home'])
df['away_encoded'] = le_away.fit_transform(df['Away'])
df['result_encoded'] = le_result.fit_transform(df['result'])

print(df['result'].value_counts())
print(df['result'].value_counts(normalize=True).round(3))

# quick correlation check
print(df.corr(numeric_only=True)['result_encoded'].sort_values())


#visualize the correlation

df['HomeGoals'].hist(bins=10)
plt.title('Home Goals Distribution')
plt.show()

features = [
    'home_encoded', 'away_encoded',
    'home_goals_avg', 'away_goals_avg',
    'home_conceded_avg', 'away_conceded_avg'
]

X = df[features]
y = df['result_encoded']

# Time-dependent prediction should respect time order:
# train on earlier matches (first 80%), test on later matches (last 20%).
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
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
    """Baseline 1: always predict 'Home Win' (simple sanity check)."""
    home_win_encoded = int(le.transform(['Home Win'])[0])
    return np.full(shape=n, fill_value=home_win_encoded, dtype=int)


def baseline_most_frequent_class(y_train: pd.Series, n: int) -> np.ndarray:
    """Baseline 2: predict the most frequent training class (majority-class baseline)."""
    most_frequent = int(y_train.value_counts().idxmax())
    return np.full(shape=n, fill_value=most_frequent, dtype=int)


def baseline_random_by_train_freq(
    y_train: pd.Series, n: int, *, random_state: int = 42
) -> np.ndarray:
    """Baseline 3: random predictions weighted by training class frequencies."""
    freqs = y_train.value_counts(normalize=True).sort_index()
    classes = freqs.index.to_numpy(dtype=int)
    probs = freqs.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    return rng.choice(classes, size=n, replace=True, p=probs)


# Naive baselines help contextualize model performance on the same (chronological) test set.
baseline_pred_always_home_win = baseline_always_home_win(len(y_test), le_result)
baseline_pred_most_frequent = baseline_most_frequent_class(y_train, len(y_test))
baseline_pred_random_weighted = baseline_random_by_train_freq(y_train, len(y_test))


#Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training complete!")


# Predict outcomes
predictions = model.predict(X_test)

pred_labels = le_result.inverse_transform(predictions[:10])
print("Sample predicted labels:", pred_labels)

class_names = list(le_result.classes_)

phase1_evals = [
    evaluate_predictions(
        "Logistic regression",
        y_test,
        predictions,
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
]

comparison = pd.DataFrame(
    [
        {
            "Model": e["model_name"],
            "Accuracy": e["accuracy"],
            "Macro F1": e["macro_f1"],
        }
        for e in phase1_evals
    ]
)
print(f"\n{'=' * 72}")
print("Phase 1 evaluation — chronological test set (same setup for all methods)")
print(f"{'=' * 72}")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\n{'—' * 72}")
print("Confusion matrices (same label order as above)")
print(f"{'—' * 72}")
for e in phase1_evals:
    print_confusion_matrix_compact(
        e["model_name"], e["confusion_matrix"], target_names=class_names
    )

lr_eval = phase1_evals[0]
cm = lr_eval["confusion_matrix"]
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix — Logistic regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
