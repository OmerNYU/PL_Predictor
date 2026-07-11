import pandas as pd
import pytest

from matchlens.backtesting import (
    BACKTEST_RESULT_COLUMNS,
    add_season_start_year,
    build_backtest_summary,
    iter_season_backtest_folds,
    run_season_backtest,
)
from matchlens.features import FEATURES_OVERALL


def test_season_start_year_derivation():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2017-08-12", "2017-09-01", "2018-01-15", "2018-05-13"]
            )
        }
    )
    out = add_season_start_year(df)
    assert list(out["season_start_year"]) == [2017, 2017, 2017, 2017]

    df2 = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2018-08-11", "2019-03-02", "2019-08-10", "2020-01-01"]
            )
        }
    )
    out2 = add_season_start_year(df2)
    assert list(out2["season_start_year"]) == [2018, 2018, 2019, 2019]


def test_season_backtest_folds_train_past_test_season_no_overlap():
    # Three seasons with enough rows for a tiny min threshold
    rows = []
    for season, n in [(2015, 5), (2016, 5), (2017, 5)]:
        for i in range(n):
            # Aug for season start year == calendar year
            rows.append(
                {
                    "Date": pd.Timestamp(year=season, month=8, day=1) + pd.Timedelta(days=i),
                    "season_start_year": season,
                    "result_encoded": i % 3,
                }
            )
    df = pd.DataFrame(rows)

    folds = iter_season_backtest_folds(df, min_train_rows=5, min_test_rows=5)
    # 2015 skipped (no prior train); 2016 and 2017 eligible
    assert [f["test_season"] for f in folds] == [2016, 2017]

    for fold in folds:
        train_idx = set(fold["train_index"])
        test_idx = set(fold["test_index"])
        assert train_idx.isdisjoint(test_idx)

        train_seasons = set(df.loc[fold["train_index"], "season_start_year"])
        test_seasons = set(df.loc[fold["test_index"], "season_start_year"])
        assert test_seasons == {fold["test_season"]}
        assert all(s < fold["test_season"] for s in train_seasons)

    # Below-threshold seasons skipped
    tiny = iter_season_backtest_folds(df, min_train_rows=100, min_test_rows=5)
    assert tiny == []


def test_backtest_result_schema_and_summary():
    # Tiny synthetic frame with FEATURES_OVERALL columns + labels
    n_per_season = 40
    seasons = [2015, 2016, 2017]
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            row = {
                "Date": pd.Timestamp(year=season, month=9, day=1)
                + pd.Timedelta(days=i),
                "result_encoded": i % 3,
            }
            for j, feat in enumerate(FEATURES_OVERALL):
                row[feat] = float((i + j + season) % 7)
            rows.append(row)
    df = pd.DataFrame(rows)
    class_names = ["Away Win", "Draw", "Home Win"]

    results, summary = run_season_backtest(
        df,
        features=FEATURES_OVERALL,
        class_names=class_names,
        min_train_rows=40,
        min_test_rows=40,
    )

    assert list(results.columns) == BACKTEST_RESULT_COLUMNS
    assert len(results) == 2  # 2016 and 2017
    assert summary["number_of_backtest_seasons"] == 2
    assert summary["first_test_season"] == 2016
    assert summary["last_test_season"] == 2017
    assert "mean_accuracy" in summary
    assert "generated_at" in summary

    # Chronology: train ends before or on day before test starts within season logic
    for _, row in results.iterrows():
        assert pd.Timestamp(row["train_end_date"]) < pd.Timestamp(row["test_start_date"])


def test_build_backtest_summary_empty():
    empty = pd.DataFrame(columns=BACKTEST_RESULT_COLUMNS)
    summary = build_backtest_summary(empty)
    assert summary["number_of_backtest_seasons"] == 0
    assert summary["first_test_season"] is None
