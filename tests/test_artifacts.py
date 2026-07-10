import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from matchlens.artifacts import save_phase1_models, save_phase1_outputs
from matchlens.evaluation import PHASE1_SPLIT_METHOD
from matchlens.features import FEATURES_OVERALL

REQUIRED_MODEL_METADATA_KEYS = {
    "model_name",
    "model_type",
    "feature_set_name",
    "features",
    "class_names",
    "accuracy",
    "macro_f1",
    "split_method",
    "train_start_date",
    "train_end_date",
    "test_start_date",
    "test_end_date",
    "generated_at",
}


@pytest.fixture
def label_encoders():
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_result = LabelEncoder()
    le_home.fit(["Alpha", "Beta"])
    le_away.fit(["Gamma", "Delta"])
    le_result.fit(["Home Win", "Draw", "Away Win"])
    return le_home, le_away, le_result


@pytest.fixture
def date_slices():
    train_dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    test_dates = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-04"]))
    return train_dates, test_dates


@pytest.fixture
def best_logistic_eval():
    return {
        "model_name": "Logistic regression (overall form)",
        "accuracy": 0.5121,
        "macro_f1": 0.3640,
    }


@pytest.fixture
def fitted_model():
    model = LogisticRegression(max_iter=10)
    model.fit([[0, 1], [1, 0]], [0, 1])
    return model


def test_save_phase1_models_metadata_schema(
    tmp_path: Path,
    label_encoders,
    date_slices,
    best_logistic_eval,
    fitted_model,
):
    le_home, le_away, le_result = label_encoders
    train_dates, test_dates = date_slices
    best_model_spec = {
        "feature_set_name": "overall_form",
        "features": FEATURES_OVERALL,
    }
    class_names = list(le_result.classes_)

    save_phase1_models(
        best_model=fitted_model,
        best_model_spec=best_model_spec,
        best_logistic_eval=best_logistic_eval,
        le_home=le_home,
        le_away=le_away,
        le_result=le_result,
        class_names=class_names,
        train_dates=train_dates,
        test_dates=test_dates,
        models_dir=tmp_path,
    )

    metadata_path = tmp_path / "best_logistic_model_metadata.json"
    assert metadata_path.exists()
    assert (tmp_path / "best_logistic_model.joblib").exists()
    assert (tmp_path / "label_encoders.joblib").exists()

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert REQUIRED_MODEL_METADATA_KEYS.issubset(metadata.keys())
    assert metadata["split_method"] == PHASE1_SPLIT_METHOD
    assert metadata["features"] == FEATURES_OVERALL
    assert metadata["model_type"] == "LogisticRegression"


def test_save_phase1_outputs_metadata_schema(
    tmp_path: Path,
    date_slices,
    best_logistic_eval,
):
    train_dates, test_dates = date_slices
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    experiment_results = pd.DataFrame(
        {
            "experiment_id": ["phase1_01"],
            "model": ["Always Home Win"],
            "features": ["home_encoded"],
            "split_method": [PHASE1_SPLIT_METHOD],
            "accuracy": [0.4],
            "macro_f1": [0.2],
            "notes": ["test"],
        }
    )

    save_phase1_outputs(
        experiment_results=experiment_results,
        n_rows_loaded=3,
        df=df,
        n_train=2,
        n_test=1,
        train_dates=train_dates,
        test_dates=test_dates,
        best_logistic_eval=best_logistic_eval,
        outputs_dir=tmp_path,
    )

    metadata_path = tmp_path / "phase1_run_metadata.json"
    assert metadata_path.exists()
    assert (tmp_path / "phase1_experiment_results.csv").exists()

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["rows_loaded"] == 3
    assert metadata["best_logistic_model_name"] == best_logistic_eval["model_name"]
    assert "generated_at" in metadata
