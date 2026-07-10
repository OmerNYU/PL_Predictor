import pandas as pd
import pytest

from matchlens.features import (
    FEATURES_CORE,
    FEATURES_OVERALL,
    FEATURES_OVERALL_DIFF,
    FEATURES_VENUE_FORM,
    OUTCOME_LEAKAGE_COLS,
    assert_no_leakage,
    build_features,
)

EXPECTED_FEATURES_CORE = [
    "home_encoded",
    "away_encoded",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
]

EXPECTED_VENUE_FORM_SUFFIX = [
    "home_points_avg",
    "away_points_avg",
    "home_goal_diff_avg",
    "away_goal_diff_avg",
]

EXPECTED_OVERALL_SUFFIX = [
    "home_team_points_avg_overall",
    "away_team_points_avg_overall",
    "home_team_goal_diff_avg_overall",
    "away_team_goal_diff_avg_overall",
]

EXPECTED_OVERALL_DIFF_SUFFIX = [
    "points_avg_overall_diff",
    "goal_diff_avg_overall_diff",
]


def test_feature_set_core_schema_and_order():
    assert FEATURES_CORE == EXPECTED_FEATURES_CORE
    assert len(FEATURES_CORE) == 6


def test_feature_set_venue_form_schema_and_order():
    assert FEATURES_VENUE_FORM[:6] == FEATURES_CORE
    assert FEATURES_VENUE_FORM[6:] == EXPECTED_VENUE_FORM_SUFFIX
    assert len(FEATURES_VENUE_FORM) == 10


def test_feature_set_overall_schema_and_order():
    assert FEATURES_OVERALL[:6] == FEATURES_CORE
    assert FEATURES_OVERALL[6:] == EXPECTED_OVERALL_SUFFIX
    assert len(FEATURES_OVERALL) == 10


def test_feature_set_overall_diff_schema_and_order():
    assert FEATURES_OVERALL_DIFF[:10] == FEATURES_OVERALL
    assert FEATURES_OVERALL_DIFF[10:] == EXPECTED_OVERALL_DIFF_SUFFIX
    assert len(FEATURES_OVERALL_DIFF) == 12


def test_leakage_guard_passes_for_valid_feature_sets():
    assert_no_leakage(
        (
            ("venue form (10)", FEATURES_VENUE_FORM),
            ("overall form (10)", FEATURES_OVERALL),
            ("overall form + matchup diff (12)", FEATURES_OVERALL_DIFF),
        )
    )


@pytest.mark.parametrize("forbidden_col", sorted(OUTCOME_LEAKAGE_COLS))
def test_leakage_guard_raises_for_forbidden_column(forbidden_col: str):
    invalid_features = FEATURES_CORE + [forbidden_col]
    with pytest.raises(ValueError, match="Target leakage"):
        assert_no_leakage((("test set", invalid_features),))


def _rolling_fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Home": ["Beta", "Gamma", "Delta", "Alpha", "Delta", "Alpha"],
            "Away": ["Delta", "Delta", "Beta", "Beta", "Gamma", "Gamma"],
            "HomeGoals": [1, 2, 0, 3, 0, 0],
            "AwayGoals": [0, 0, 1, 0, 1, 2],
            "FTR": ["H", "H", "A", "H", "A", "A"],
            "Date": pd.to_datetime(
                [
                    "2019-12-01",
                    "2019-12-08",
                    "2019-12-15",
                    "2020-01-01",
                    "2020-01-05",
                    "2020-01-08",
                ]
            ),
        }
    )


def test_rolling_features_use_prior_matches_only():
    raw = _rolling_fixture_df()
    featured = build_features(raw)

    alpha_home = featured[featured["Home"] == "Alpha"].sort_values("Date")
    assert len(alpha_home) == 1

    second_alpha_home = alpha_home.iloc[0]
    assert second_alpha_home["HomeGoals"] == 0
    assert second_alpha_home["home_goals_avg"] == pytest.approx(3.0)

    first_alpha_home_date = pd.Timestamp("2020-01-01")
    assert first_alpha_home_date not in set(alpha_home["Date"])
