from __future__ import annotations

import pandas as pd

# Never use these (or direct derivatives) as model inputs — same-fixture outcome.
OUTCOME_LEAKAGE_COLS = frozenset(
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

FEATURES_CORE = [
    "home_encoded",
    "away_encoded",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
]
FEATURES_VENUE_FORM = FEATURES_CORE + [
    "home_points_avg",
    "away_points_avg",
    "home_goal_diff_avg",
    "away_goal_diff_avg",
]
FEATURES_OVERALL = FEATURES_CORE + [
    "home_team_points_avg_overall",
    "away_team_points_avg_overall",
    "home_team_goal_diff_avg_overall",
    "away_team_goal_diff_avg_overall",
]
FEATURES_OVERALL_DIFF = FEATURES_OVERALL + [
    "points_avg_overall_diff",
    "goal_diff_avg_overall_diff",
]


def assert_no_leakage(
    feature_sets: tuple[tuple[str, list[str]], ...],
) -> None:
  for feat_set_name, feat_set in feature_sets:
    overlap = OUTCOME_LEAKAGE_COLS.intersection(feat_set)
    if overlap:
      raise ValueError(
          f"Target leakage ({feat_set_name}): these columns must not be model inputs: "
          + ", ".join(sorted(overlap))
      )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
  """Add outcome helpers, rolling prematch features, and drop incomplete rows."""
  # Per-row helpers (outcome-based); only shifted rolling means below enter the model.
  df = df.copy()
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

  # Overall (venue-agnostic) prematch rollings: each team's matches in chronological order.
  home_app = pd.DataFrame(
      {
          "orig_idx": df.index,
          "team": df["Home"],
          "date": df["Date"],
          "points": df["home_points"],
          "goal_diff": df["home_goal_diff"],
          "role": "home",
      }
  )
  away_app = pd.DataFrame(
      {
          "orig_idx": df.index,
          "team": df["Away"],
          "date": df["Date"],
          "points": df["away_points"],
          "goal_diff": df["away_goal_diff"],
          "role": "away",
      }
  )
  appearances = pd.concat([home_app, away_app], ignore_index=True)
  appearances = appearances.sort_values(["team", "date", "orig_idx"])
  roll = lambda s: s.shift(1).rolling(5, min_periods=1).mean()
  appearances["points_avg_overall"] = appearances.groupby("team", sort=False)[
      "points"
  ].transform(roll)
  appearances["goal_diff_avg_overall"] = appearances.groupby("team", sort=False)[
      "goal_diff"
  ].transform(roll)
  overall_wide = appearances.pivot(
      index="orig_idx", columns="role", values=["points_avg_overall", "goal_diff_avg_overall"]
  )
  df["home_team_points_avg_overall"] = overall_wide["points_avg_overall"]["home"]
  df["away_team_points_avg_overall"] = overall_wide["points_avg_overall"]["away"]
  df["home_team_goal_diff_avg_overall"] = overall_wide["goal_diff_avg_overall"]["home"]
  df["away_team_goal_diff_avg_overall"] = overall_wide["goal_diff_avg_overall"]["away"]
  df["points_avg_overall_diff"] = (
      df["home_team_points_avg_overall"] - df["away_team_points_avg_overall"]
  )
  df["goal_diff_avg_overall_diff"] = (
      df["home_team_goal_diff_avg_overall"] - df["away_team_goal_diff_avg_overall"]
  )

  return df.dropna()
