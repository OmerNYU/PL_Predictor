# Premier League Match Predictor

Baseline machine learning project for predicting Premier League match outcomes (`Home Win`, `Draw`, `Away Win`) from historical match data.

## Overview

This project currently implements a full baseline workflow in `main.py`:

- loads and inspects the dataset
- builds pre-match rolling features
- encodes categorical variables
- trains a logistic regression classifier
- evaluates predictions with a classification report and confusion matrix

The focus right now is correctness and clarity before moving to more advanced modeling.

## Project Structure

```text
PL_Predictor/
├── main.py
├── premier-league-matches.csv
└── README.md
```

## Features Used

Model input features:

- `home_encoded`
- `away_encoded`
- `home_goals_avg`
- `away_goals_avg`
- `home_conceded_avg`
- `away_conceded_avg`

Rolling features use a 5-match window and apply `shift(1)` before rolling, so each row uses only information available before kickoff.

## Model and Evaluation

- **Model:** Logistic Regression (`max_iter=1000`)
- **Split:** `train_test_split(test_size=0.2, random_state=42)`
- **Outputs:**
  - class distribution prints
  - numeric correlation print
  - sample decoded predictions
  - `classification_report`
  - confusion matrix heatmap (`seaborn`)

## Installation

From the project directory:

```bash
python3 -m pip install pandas matplotlib scikit-learn seaborn
```

If your environment uses `python` instead of `python3`, replace accordingly.

## Run

```bash
cd /home/omerh/projects/PL_Predictor
python3 main.py
```

## Current Limitations

- random train/test split (not time-aware)
- team IDs use `LabelEncoder` (simple baseline representation)
- limited feature set (no odds, form points, shots, Elo, etc.)
- script-based pipeline (not yet modularized into `src/`)

## Next Improvements

- switch to chronological validation
- add stronger pre-match features
- compare additional models (Random Forest, XGBoost, etc.)
- persist trained model and preprocessing artifacts
- refactor into a reusable training/inference pipeline
