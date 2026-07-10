import pandas as pd

from matchlens.data import chronological_train_test_split


def _synthetic_modeling_df(n_rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
            "f1": list(range(n_rows)),
            "f2": [x * 0.5 for x in range(n_rows)],
            "result_encoded": ([0, 1, 2] * (n_rows // 3 + 1))[:n_rows],
        }
    )


def test_chronological_train_test_split_row_counts():
    df = _synthetic_modeling_df(10)
    split = chronological_train_test_split(df, ["f1", "f2"])

    assert split.split_idx == 8
    assert split.n_train == 8
    assert split.n_test == 2
    assert split.n_train + split.n_test == len(df)


def test_chronological_train_test_split_no_overlap_and_order():
    df = _synthetic_modeling_df(10)
    split = chronological_train_test_split(df, ["f1", "f2"])

    train_indices = set(split.X_train.index)
    test_indices = set(split.X_test.index)
    assert train_indices.isdisjoint(test_indices)
    assert list(split.X_train.index) == list(range(split.n_train))
    assert list(split.X_test.index) == list(range(split.n_train, len(df)))


def test_chronological_train_test_split_preserves_chronology():
    df = _synthetic_modeling_df(10)
    split = chronological_train_test_split(df, ["f1", "f2"])

    assert split.train_dates.max() <= split.test_dates.min()
    assert len(split.y_train) == split.n_train
    assert len(split.y_test) == split.n_test


def test_chronological_train_test_split_uses_first_rows_for_train():
    df = _synthetic_modeling_df(10)
    split = chronological_train_test_split(df, ["f1", "f2"])

    assert split.X_train["f1"].tolist() == df["f1"].iloc[:8].tolist()
    assert split.X_test["f1"].tolist() == df["f1"].iloc[8:].tolist()
