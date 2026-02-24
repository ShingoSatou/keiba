from __future__ import annotations

import pytest

from scripts_v2.tune_ranker_xgb_optuna_v2 import (  # noqa: E402
    COLSAMPLE_BYTREE_RANGE,
    FEATURE_SET_CHOICES,
    GAMMA_RANGE,
    LR_RANGE,
    MAX_DEPTH_RANGE,
    MIN_CHILD_WEIGHT_RANGE,
    OBJECTIVE_CHOICES,
    REG_ALPHA_RANGE,
    REG_LAMBDA_RANGE,
    SUBSAMPLE_RANGE,
    drop_entity_id_features,
    suggest_feature_set,
    suggest_xgb_params,
)

optuna = pytest.importorskip("optuna")


def test_drop_entity_id_features_always_removes_raw_ids():
    cols = ["a", "jockey_key", "b", "trainer_key", "c"]
    assert drop_entity_id_features(cols) == ["a", "b", "c"]


def test_optuna_trial_suggestion_is_within_declared_ranges():
    def objective(trial: optuna.Trial) -> float:
        feature_set = suggest_feature_set(trial)
        assert feature_set in FEATURE_SET_CHOICES

        params = suggest_xgb_params(trial)
        assert params["objective"] in OBJECTIVE_CHOICES
        assert LR_RANGE[0] <= params["learning_rate"] <= LR_RANGE[1]
        assert MAX_DEPTH_RANGE[0] <= params["max_depth"] <= MAX_DEPTH_RANGE[1]
        assert MIN_CHILD_WEIGHT_RANGE[0] <= params["min_child_weight"] <= MIN_CHILD_WEIGHT_RANGE[1]
        assert GAMMA_RANGE[0] <= params["gamma"] <= GAMMA_RANGE[1]
        assert SUBSAMPLE_RANGE[0] <= params["subsample"] <= SUBSAMPLE_RANGE[1]
        assert COLSAMPLE_BYTREE_RANGE[0] <= params["colsample_bytree"] <= COLSAMPLE_BYTREE_RANGE[1]
        assert REG_LAMBDA_RANGE[0] <= params["reg_lambda"] <= REG_LAMBDA_RANGE[1]
        assert REG_ALPHA_RANGE[0] <= params["reg_alpha"] <= REG_ALPHA_RANGE[1]
        return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    study.optimize(objective, n_trials=50)
