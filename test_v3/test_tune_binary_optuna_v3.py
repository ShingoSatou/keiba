from __future__ import annotations

import argparse
from pathlib import Path

from scripts_v3.tune_binary_optuna_v3 import (
    TrialResult,
    _build_best_output,
    _build_best_params_output,
    _trial_results_to_frame,
    select_best_trial_result,
)


def _trial_result(
    *,
    trial_number: int,
    value_mean_logloss: float,
    feature_set: str = "base",
    input_path: str = "data/features_v3.parquet",
    state: str = "COMPLETE",
    benter_mean: float | None = None,
    benter_median: float | None = None,
    benter_min: float | None = None,
) -> TrialResult:
    return TrialResult(
        trial_number=trial_number,
        state=state,
        value_mean_logloss=value_mean_logloss,
        feature_set=feature_set,
        input_path=input_path,
        params={
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
        },
        fold_logloss={1: value_mean_logloss + 0.01, 2: value_mean_logloss - 0.01},
        fold_best_iteration={1: 120, 2: 140, 3: 130},
        fold_benter_r2_valid=None
        if benter_mean is None
        else {1: benter_mean, 2: benter_mean + 0.001},
        benter_r2_valid_mean=benter_mean,
        benter_r2_valid_median=benter_median,
        benter_r2_valid_min=benter_min,
    )


def test_select_best_trial_result_prefers_win_constraint_satisfying_trial() -> None:
    baseline_summary = {
        "benter_r2_valid_median": 0.120,
        "benter_r2_valid_min": 0.080,
    }
    constrained = _trial_result(
        trial_number=1,
        value_mean_logloss=0.119,
        benter_mean=0.125,
        benter_median=0.119,
        benter_min=0.071,
    )
    unconstrained_better_logloss = _trial_result(
        trial_number=2,
        value_mean_logloss=0.110,
        benter_mean=0.113,
        benter_median=0.115,
        benter_min=0.069,
    )

    selected, constraint_passed, selection_mode = select_best_trial_result(
        results=[constrained, unconstrained_better_logloss],
        task="win",
        baseline_summary=baseline_summary,
        study_best_trial_number=2,
    )

    assert selected.trial_number == 1
    assert constraint_passed is True
    assert selection_mode == "constrained_logloss"


def test_select_best_trial_result_falls_back_to_study_best_when_no_win_trial_passes() -> None:
    baseline_summary = {
        "benter_r2_valid_median": 0.120,
        "benter_r2_valid_min": 0.080,
    }
    study_best = _trial_result(
        trial_number=3,
        value_mean_logloss=0.109,
        benter_mean=0.110,
        benter_median=0.110,
        benter_min=0.060,
    )
    other_complete = _trial_result(
        trial_number=4,
        value_mean_logloss=0.112,
        benter_mean=0.111,
        benter_median=0.116,
        benter_min=0.069,
    )

    selected, constraint_passed, selection_mode = select_best_trial_result(
        results=[study_best, other_complete],
        task="win",
        baseline_summary=baseline_summary,
        study_best_trial_number=3,
    )

    assert selected.trial_number == 3
    assert constraint_passed is False
    assert selection_mode == "study_best_fallback"


def test_select_best_trial_result_uses_lowest_complete_logloss_for_place() -> None:
    pruned = _trial_result(
        trial_number=1,
        value_mean_logloss=0.090,
        state="PRUNED",
        benter_mean=None,
        benter_median=None,
        benter_min=None,
    )
    best_complete = _trial_result(
        trial_number=2,
        value_mean_logloss=0.101,
        feature_set="te",
        input_path="data/features_v3_te.parquet",
        benter_mean=None,
        benter_median=None,
        benter_min=None,
    )
    slower_complete = _trial_result(
        trial_number=3,
        value_mean_logloss=0.108,
        benter_mean=None,
        benter_median=None,
        benter_min=None,
    )

    selected, constraint_passed, selection_mode = select_best_trial_result(
        results=[pruned, best_complete, slower_complete],
        task="place",
        baseline_summary={},
        study_best_trial_number=2,
    )

    assert selected.trial_number == 2
    assert selected.feature_set == "te"
    assert constraint_passed is False
    assert selection_mode == "min_complete"


def test_tuner_serialization_helpers_include_feature_set_and_benter_fields() -> None:
    args = argparse.Namespace(
        task="win",
        model="lgbm",
        holdout_year=2025,
        num_boost_round=2000,
    )
    selected_trial = _trial_result(
        trial_number=7,
        value_mean_logloss=0.118,
        feature_set="te",
        input_path="data/features_v3_te.parquet",
        benter_mean=0.124,
        benter_median=0.123,
        benter_min=0.075,
    )
    baseline_summary = {
        "input": "data/features_v3.parquet",
        "feature_set": "base",
        "value_mean_logloss": 0.121,
        "fold_logloss": {1: 0.122, 2: 0.120},
        "fold_best_iteration": {1: 115, 2: 118},
        "benter_r2_valid_mean": 0.126,
        "benter_r2_valid_median": 0.125,
        "benter_r2_valid_min": 0.080,
    }
    fixed_config = {
        "cv_window_policy": "fixed_sliding",
        "train_window_years": 4,
        "operational_mode": "t10_only",
        "include_entity_id_features": False,
    }
    outputs = {
        "storage": Path("data/optuna/binary_v3_win_lgbm.sqlite3"),
    }

    frame = _trial_results_to_frame([selected_trial])
    assert frame.loc[0, "feature_set"] == "te"
    assert frame.loc[0, "input"] == "data/features_v3_te.parquet"
    assert frame.loc[0, "benter_r2_valid_mean"] == 0.124
    assert frame.loc[0, "fold/1/logloss"] == selected_trial.fold_logloss[1]

    best_output = _build_best_output(
        args=args,
        outputs=outputs,
        study_name="binary_v3_win_lgbm",
        total_trials=12,
        selected_trial=selected_trial,
        baseline_summary=baseline_summary,
        fixed_config=fixed_config,
        constraint_passed=True,
        selection_mode="constrained_logloss",
    )
    assert best_output["best_feature_set"] == "te"
    assert best_output["best_input"] == "data/features_v3_te.parquet"
    assert best_output["best_benter_summary"]["benter_r2_valid_median"] == 0.123
    assert best_output["constraint_passed"] is True

    best_params = _build_best_params_output(
        args=args,
        selected_trial=selected_trial,
    )
    assert best_params["input"] == "data/features_v3_te.parquet"
    assert best_params["feature_set"] == "te"
    assert best_params["train_window_years"] == 4
    assert best_params["include_entity_id_features"] is False
    assert best_params["lgbm_params"]["learning_rate"] == 0.03
    assert best_params["final_num_boost_round"] == 130
