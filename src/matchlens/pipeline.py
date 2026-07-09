from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from matchlens.artifacts import (
    save_best_logistic_confusion_matrix,
    save_phase1_models,
    save_phase1_outputs,
)
from matchlens.data import load_matches
from matchlens.evaluation import (
    PHASE1_SPLIT_METHOD,
    baseline_always_home_win,
    baseline_most_frequent_class,
    baseline_random_by_train_freq,
    build_experiment_results,
    build_phase1_experiment_specs,
    evaluate_predictions,
    print_all_confusion_matrices,
    print_experiment_summary,
    select_best_logistic_eval,
)
from matchlens.features import (
    FEATURES_CORE,
    FEATURES_OVERALL,
    FEATURES_OVERALL_DIFF,
    FEATURES_VENUE_FORM,
    assert_no_leakage,
    build_features,
)


def run_phase1_pipeline() -> None:
    df, n_rows_loaded, raw_nulls = load_matches()
    df = build_features(df)

    print("\nPhase 1 — dataset")
    print(f"  Rows loaded from CSV:     {n_rows_loaded}")
    print(f"  Rows in modeling cohort:  {len(df)}  (after rolling features and complete cases)")
    print(
        f"  Cohort date span:         {df['Date'].min().date()} — {df['Date'].max().date()}"
    )
    if raw_nulls.sum() > 0:
        print("  Missing values in raw CSV (by column):")
        print(raw_nulls[raw_nulls > 0].to_string(header=False))

    class_vc = df["result"].value_counts()
    class_prop = df["result"].value_counts(normalize=True)
    class_summary = pd.DataFrame(
        {"count": class_vc, "proportion": class_prop.round(4)}
    ).sort_values("count", ascending=False)
    class_summary.index.name = None
    print("\nPhase 1 — class distribution (modeling cohort)")
    print(class_summary.to_string())

    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_result = LabelEncoder()

    df["home_encoded"] = le_home.fit_transform(df["Home"])
    df["away_encoded"] = le_away.fit_transform(df["Away"])
    df["result_encoded"] = le_result.fit_transform(df["result"])

    features_core = FEATURES_CORE
    features = FEATURES_VENUE_FORM
    features_overall = FEATURES_OVERALL
    features_overall_diff = FEATURES_OVERALL_DIFF

    assert_no_leakage(
        (
            ("venue form (10)", features),
            ("overall form (10)", features_overall),
            ("overall form + matchup diff (12)", features_overall_diff),
        )
    )

    X = df[features]
    X_core = df[features_core]
    X_overall = df[features_overall]
    X_overall_diff = df[features_overall_diff]
    y = df["result_encoded"]

    split_idx = int(len(df) * 0.8)
    n_train, n_test = split_idx, len(df) - split_idx
    train_dates = df["Date"].iloc[:split_idx]
    test_dates = df["Date"].iloc[split_idx:]

    print("\nPhase 1 — chronological split (rows ordered by Date; no shuffle)")
    print("  Policy: first 80% of rows → train, remainder → test.")
    print(f"  Train rows: {n_train}    date range: {train_dates.min().date()} — {train_dates.max().date()}")
    print(f"  Test rows:  {n_test}    date range: {test_dates.min().date()} — {test_dates.max().date()}")
    print("  All reported metrics below use the test slice only.")

    print("\nPhase 1 — model features (prematch only; no same-match scores or result)")
    for i, name in enumerate(features, start=1):
        print(f"  {i}. {name}")
    print(
        "  Rolling *_avg: prior up-to-5 same-role matches per team (shift(1).rolling(5))."
    )

    print("\nPhase 1 — model features (logistic regression, overall recent form)")
    print("  Prematch only; no same-match scores or result.")
    for i, name in enumerate(features_overall, start=1):
        print(f"  {i}. {name}")
    print(
        "  Rolling *_overall: prior up-to-5 matches per team in all venues "
        "(shift(1).rolling(5) on team chronological appearance stream)."
    )

    print("\nPhase 1 — model features (logistic regression, overall + matchup differences)")
    print("  Prematch only; includes venue-agnostic home-away difference features.")
    for i, name in enumerate(features_overall_diff, start=1):
        print(f"  {i}. {name}")
    print(
        "  *_diff features are home minus away values derived from overall prematch rollings."
    )

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    X_train_core, X_test_core = X_core.iloc[:split_idx], X_core.iloc[split_idx:]
    X_train_overall, X_test_overall = X_overall.iloc[:split_idx], X_overall.iloc[split_idx:]
    X_train_overall_diff, X_test_overall_diff = (
        X_overall_diff.iloc[:split_idx],
        X_overall_diff.iloc[split_idx:],
    )
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    baseline_pred_always_home_win = baseline_always_home_win(len(y_test), le_result)
    baseline_pred_most_frequent = baseline_most_frequent_class(y_train, len(y_test))
    baseline_pred_random_weighted = baseline_random_by_train_freq(y_train, len(y_test))

    model_lr_core = LogisticRegression(max_iter=1000)
    model_lr_core.fit(X_train_core, y_train)
    predictions_core = model_lr_core.predict(X_test_core)

    model_lr_form = LogisticRegression(max_iter=1000)
    model_lr_form.fit(X_train, y_train)
    predictions_form = model_lr_form.predict(X_test)

    model_lr_overall = LogisticRegression(max_iter=1000)
    model_lr_overall.fit(X_train_overall, y_train)
    predictions_overall = model_lr_overall.predict(X_test_overall)

    model_lr_overall_diff = LogisticRegression(max_iter=1000)
    model_lr_overall_diff.fit(X_train_overall_diff, y_train)
    predictions_overall_diff = model_lr_overall_diff.predict(X_test_overall_diff)

    logistic_model_registry = {
        "Logistic regression": {
            "model": model_lr_core,
            "feature_set_name": "core",
            "features": features_core,
        },
        "Logistic regression (rolling form)": {
            "model": model_lr_form,
            "feature_set_name": "venue_form",
            "features": features,
        },
        "Logistic regression (overall form)": {
            "model": model_lr_overall,
            "feature_set_name": "overall_form",
            "features": features_overall,
        },
        "Logistic regression (overall form + matchup diff)": {
            "model": model_lr_overall_diff,
            "feature_set_name": "overall_form_plus_matchup_diff",
            "features": features_overall_diff,
        },
    }

    class_names = list(le_result.classes_)

    phase1_evals = [
        evaluate_predictions(
            "Logistic regression",
            y_test,
            predictions_core,
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
        evaluate_predictions(
            "Logistic regression (rolling form)",
            y_test,
            predictions_form,
            target_names=class_names,
            print_report=False,
        ),
        evaluate_predictions(
            "Logistic regression (overall form)",
            y_test,
            predictions_overall,
            target_names=class_names,
            print_report=False,
        ),
        evaluate_predictions(
            "Logistic regression (overall form + matchup diff)",
            y_test,
            predictions_overall_diff,
            target_names=class_names,
            print_report=False,
        ),
    ]

    experiment_specs = build_phase1_experiment_specs(
        features_core=features_core,
        features_venue_form=features,
        features_overall=features_overall,
        features_overall_diff=features_overall_diff,
    )
    experiment_results = build_experiment_results(
        phase1_evals,
        experiment_specs,
        split_method=PHASE1_SPLIT_METHOD,
    )
    print_experiment_summary(experiment_results)
    print_all_confusion_matrices(phase1_evals, target_names=class_names)

    best_logistic_eval = select_best_logistic_eval(phase1_evals)
    best_model_spec = logistic_model_registry[best_logistic_eval["model_name"]]

    save_best_logistic_confusion_matrix(
        best_logistic_eval["confusion_matrix"],
        model_name=best_logistic_eval["model_name"],
        class_names=class_names,
    )
    save_phase1_outputs(
        experiment_results=experiment_results,
        n_rows_loaded=n_rows_loaded,
        df=df,
        n_train=n_train,
        n_test=n_test,
        train_dates=train_dates,
        test_dates=test_dates,
        best_logistic_eval=best_logistic_eval,
    )
    save_phase1_models(
        best_model=best_model_spec["model"],
        best_model_spec=best_model_spec,
        best_logistic_eval=best_logistic_eval,
        le_home=le_home,
        le_away=le_away,
        le_result=le_result,
        class_names=class_names,
        train_dates=train_dates,
        test_dates=test_dates,
    )
