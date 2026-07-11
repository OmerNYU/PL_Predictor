# Premier League Match Predictor

Baseline machine learning project for predicting Premier League match outcomes (`Home Win`, `Draw`, `Away Win`) from historical match data.

## Overview

This project implements a Phase 1 experiment harness in `src/matchlens/` (entry point: `main.py`):

- Loads `premier-league-matches.csv`, maps full-time result `FTR` to labels, parses dates, and sorts rows chronologically.
- Builds **pre-match** rolling features (5-match window with `shift(1)` so only past matches contribute).
- Drops incomplete rows after feature construction and prints cohort stats (date range, outcome counts and proportions).
- Encodes home/away teams and outcomes with `LabelEncoder`.
- Runs a **leakage guard** to ensure same-fixture outcome columns never enter model feature sets.
- Splits data **in time**: first 80% of rows (by date order) for training, last 20% for testing—no random shuffle.
- Trains **four logistic regression variants** and compares them on the same test window to **three naive baselines** (always home win, majority class, random labels weighted by training class frequencies).
- Reports **accuracy**, **macro F1**, and **confusion matrices** for all seven experiments; prints a Phase 1 `experiment_results` summary table and a **seaborn** heatmap for the best-performing logistic regression variant.

The focus is correctness, leakage avoidance, and interpretable evaluation before more advanced modeling.

## Project Structure

```text
PL_Predictor/
├── main.py
├── premier-league-matches.csv
├── requirements.txt
├── src/
│   └── matchlens/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       ├── artifacts.py
│       └── pipeline.py
├── outputs/          # generated artifacts (gitignored contents)
├── models/           # serialized models (gitignored contents)
└── README.md
```

## Phase 1 Experiments

Seven experiments (`phase1_01`–`phase1_07`) are evaluated on the same held-out test slice:

| ID | Model | Feature set |
|----|-------|-------------|
| `phase1_01` | Always Home Win | core (6) |
| `phase1_02` | Most Frequent Class | core (6) |
| `phase1_03` | Class-Frequency Random | core (6) |
| `phase1_04` | Logistic Regression | core (6) |
| `phase1_05` | Logistic Regression (rolling form) | venue-form (10) |
| `phase1_06` | Logistic Regression (overall form) | overall-form (10) |
| `phase1_07` | Logistic Regression (overall + matchup diff) | overall + diff (12) |

## Feature Sets

Four distinct feature sets are used across experiments. All rolling statistics use `shift(1)` before `rolling(5, min_periods=1).mean()`, so each row reflects only information available **before** that fixture.

### Core (6 features)

Used by all three baselines and the core logistic regression (`phase1_04`):

- `home_encoded`, `away_encoded` — team identity (`LabelEncoder` on `Home` / `Away`)
- `home_goals_avg`, `away_goals_avg` — rolling mean goals scored (5 prior same-role matches per team)
- `home_conceded_avg`, `away_conceded_avg` — rolling mean goals conceded (5 prior same-role matches per team)

### Venue-form (10 features)

Core set plus venue-specific form (`phase1_05`):

- `home_points_avg`, `away_points_avg` — rolling mean points from prior same-role (home/away) matches
- `home_goal_diff_avg`, `away_goal_diff_avg` — rolling mean goal difference from prior same-role matches

### Overall-form (10 features)

Core set plus venue-agnostic recent form (`phase1_06`):

- `home_team_points_avg_overall`, `away_team_points_avg_overall` — rolling mean points across all venues
- `home_team_goal_diff_avg_overall`, `away_team_goal_diff_avg_overall` — rolling mean goal difference across all venues

Overall rollings are computed on each team's chronological appearance stream (home and away matches combined).

### Overall + matchup diff (12 features)

Overall-form set plus head-to-head strength differences (`phase1_07`):

- `points_avg_overall_diff` — home minus away overall points rolling average
- `goal_diff_avg_overall_diff` — home minus away overall goal-difference rolling average

## Leakage Guard

`src/matchlens/features.py` defines `OUTCOME_LEAKAGE_COLS`, a frozen set of columns that must never be used as model inputs because they encode the same-fixture outcome (e.g. `FTR`, `HomeGoals`, `AwayGoals`, `home_points`, `away_points`, `home_goal_diff`, `away_goal_diff`).

At startup, the venue-form, overall-form, and overall+diff feature sets are checked against this set. If any leakage column is present, a `ValueError` is raised before training begins.

## Model, Baselines, and Evaluation

- **Classifier:** `sklearn.linear_model.LogisticRegression` with `max_iter=1000` (other hyperparameters at sklearn defaults).
- **Baselines (same held-out test slice):**
  - Always predict **Home Win**
  - Always predict the **majority class** from the training labels
  - **Random** labels drawn with probabilities matching training class frequencies (`random_state=42`)
- **Split:** Chronological **80% / 20%** by row order after sorting by `Date` (`split_idx = int(len(df) * 0.8)`). All reported metrics use the test slice only.
- **Metrics:** Accuracy, macro-averaged F1 (`zero_division=0`), confusion matrix (rows = actual, columns = predicted). Phase 1 also saves per-class metrics, prediction distributions, and logistic probability diagnostics under `outputs/` for inspection (including the known zero-draw logistic behavior). After the single-split run, a season walk-forward backtest diagnostic is written for the overall-form logistic setup (`phase1_backtest_results.csv`, `phase1_backtest_summary.json`).
- **Console output:** Data quality and cohort summaries, split description, compact confusion matrices for all seven experiments, and a **Phase 1** `experiment_results` table with `experiment_id`, `model`, `features`, `split_method`, `accuracy`, `macro_f1`, and `notes`.
- **Plot:** Confusion matrix heatmap for the **best logistic regression** variant—selected by highest test accuracy, with macro F1 as tie-breaker (`matplotlib` + `seaborn`).

Helper functions in `src/matchlens/evaluation.py` include `evaluate_predictions` (shared metrics), `print_confusion_matrix_compact`, and the baseline generators above.

## Installation

From the project directory:

```bash
python3 -m pip install -r requirements.txt
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
- Primary reported metrics still use a single chronological 80/20 split; Phase 1 also runs a season walk-forward backtest diagnostic on the overall-form logistic setup (separate from model selection).
- Phase 1 writes `phase1_experiment_results.csv`, `best_logistic_confusion_matrix.png`, and `phase1_run_metadata.json` under `outputs/` (gitignored), plus evaluation diagnostics: `phase1_class_metrics.csv`, `phase1_prediction_distribution.csv`, and `phase1_probability_diagnostics.csv`. Season backtesting adds `phase1_backtest_results.csv` and `phase1_backtest_summary.json`. Best model, encoders, and metadata are saved under `models/` (gitignored).

## Next Improvements

- Richer pre-match features and optional external signals (e.g. bookmaker odds where available).
- Additional models (e.g. gradient boosting) with the same chronological evaluation harness.
- Use season backtesting more heavily for model comparison and hyperparameter choices.
- CLI or config-driven experiments.
