from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class ChronologicalSplit:
    split_idx: int
    n_train: int
    n_test: int
    train_dates: pd.Series
    test_dates: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_matches(csv_path: str = "premier-league-matches.csv") -> tuple[pd.DataFrame, int, pd.Series]:
    """Load CSV, map FTR to result labels, parse dates, and sort chronologically."""
    df = pd.read_csv(csv_path)
    n_rows_loaded = len(df)
    raw_nulls = df.isnull().sum()

    df["result"] = df["FTR"].map({"H": "Home Win", "A": "Away Win", "D": "Draw"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df, n_rows_loaded, raw_nulls


def chronological_train_test_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "result_encoded",
    train_frac: float = 0.8,
) -> ChronologicalSplit:
    """Split rows chronologically: first train_frac for train, remainder for test."""
    split_idx = int(len(df) * train_frac)
    n_train, n_test = split_idx, len(df) - split_idx
    train_dates = df["Date"].iloc[:split_idx]
    test_dates = df["Date"].iloc[split_idx:]

    X = df[list(feature_cols)]
    y = df[target_col]

    return ChronologicalSplit(
        split_idx=split_idx,
        n_train=n_train,
        n_test=n_test,
        train_dates=train_dates,
        test_dates=test_dates,
        X_train=X.iloc[:split_idx],
        X_test=X.iloc[split_idx:],
        y_train=y.iloc[:split_idx],
        y_test=y.iloc[split_idx:],
    )
