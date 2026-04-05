# Premier League Match Predictor

Baseline machine learning project for predicting Premier League match outcomes (`Home Win`, `Draw`, `Away Win`) from historical match data.

## Overview

This project implements a baseline workflow in `main.py`:

- Loads `premier-league-matches.csv` and prints basic data quality (shape, missing values).
- Maps full-time result `FTR` to labels, parses dates, and sorts rows chronologically.
- Builds **pre-match** rolling features (5-match window with `shift(1)` so only past matches contribute).
- Drops incomplete rows after feature construction and prints cohort stats (date range, outcome counts and proportions).
- Encodes home/away teams and outcomes with `LabelEncoder`.
- Splits data **in time**: first 80% of rows (by date order) for training, last 20% for testing—no random shuffle.
- Trains **logistic regression** and compares it on the same test window to three **naive baselines** (always home win, majority class, random labels weighted by training class frequencies).
- Reports **accuracy**, **macro F1**, and **confusion matrices** for every model; prints a Phase 1 summary table and a **seaborn** heatmap for logistic regression only.

The focus is correctness, leakage avoidance, and interpretable evaluation before more advanced modeling.

## Project Structure

```text
PL_Predictor/
├── main.py
├── premier-league-matches.csv
└── README.md
```

## Features Used

Model input features:

- `home_encoded`, `away_encoded` — team identity (`LabelEncoder` on `Home` / `Away`)
- `home_goals_avg`, `away_goals_avg` — rolling mean goals scored (5 prior matches per team)
- `home_conceded_avg`, `away_conceded_avg` — rolling mean goals conceded (5 prior matches per team)

Rolling statistics use `groupby(team).transform` with `shift(1)` before `rolling(5, min_periods=1).mean()`, so each row reflects only information available **before** that fixture.

## Model, Baselines, and Evaluation

- **Classifier:** `sklearn.linear_model.LogisticRegression` with `max_iter=1000` (other hyperparameters at sklearn defaults).
- **Baselines (same held-out test slice):**
  - Always predict **Home Win**
  - Always predict the **majority class** from the training labels
  - **Random** labels drawn with probabilities matching training class frequencies (`random_state=42`)
- **Split:** Chronological **80% / 20%** by row order after sorting by `Date` (`split_idx = int(len(df) * 0.8)`).
- **Metrics:** Accuracy, macro-averaged F1 (`zero_division=0`), confusion matrix (rows = actual, columns = predicted).
- **Console output:** Data quality and cohort summaries, split description, compact confusion matrices for all four approaches, a **Phase 1** table (`experiment_results`) with experiment id, model name, feature list, split method, accuracy, macro F1, and notes.
- **Plot:** Confusion matrix heatmap for logistic regression only (`matplotlib` + `seaborn`).

Helper functions in `main.py` include `evaluate_predictions` (shared metrics), `print_confusion_matrix_compact`, and the baseline generators above.

## Installation

From the project directory:

```bash
python3 -m pip install pandas numpy matplotlib scikit-learn seaborn
```

If your environment uses `python` instead of `python3`, replace accordingly.

## Run

```bash
cd /path/to/PL_Predictor
python3 main.py
```

Ensure `premier-league-matches.csv` sits next to `main.py` (path is relative to the working directory).

## Current Limitations

- Team identity is a simple `LabelEncoder` (no learned embeddings or hierarchical structure).
- Feature set is small (no odds, league table, shots, Elo, etc.).
- Single chronological split (no walk-forward or cross-validation).
- Everything lives in one script (not yet refactored into importable modules or a training/inference package).

## Next Improvements

- Richer pre-match features and optional external signals (e.g. bookmaker odds where available).
- Additional models (e.g. gradient boosting) with the same chronological evaluation harness.
- Walk-forward or rolling validation for more stable estimates of generalization.
- Save fitted model, encoders, and feature definitions for reproducible inference.
- Modular layout (`src/` or package) with CLI or config-driven experiments.
