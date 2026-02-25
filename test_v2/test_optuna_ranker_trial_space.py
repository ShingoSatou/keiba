from __future__ import annotations

import pytest

from scripts_v2.tune_ranker_optuna_v2 import (
    BAGGING_FRACTION_RANGE,
    BAGGING_FREQ_RANGE,
    FEATURE_FRACTION_RANGE,
    FEATURE_SET_CHOICES,
    LR_RANGE,
    MIN_CHILD_SAMPLES_RANGE,
    NUM_LEAVES_RANGE,
    REG_ALPHA_RANGE,
    REG_LAMBDA_RANGE,
    drop_entity_id_features,
    suggest_feature_set,
    suggest_lgbm_params,
)

optuna = pytest.importorskip("optuna")


def test_drop_entity_id_features_always_removes_raw_ids():
    cols = ["a", "jockey_key", "b", "trainer_key", "c"]
    assert drop_entity_id_features(cols) == ["a", "b", "c"]


def test_optuna_trial_suggestion_is_within_declared_ranges():
    def objective(trial: optuna.Trial) -> float:
        feature_set = suggest_feature_set(trial)
        assert feature_set in FEATURE_SET_CHOICES

        params = suggest_lgbm_params(trial)
        assert LR_RANGE[0] <= params["learning_rate"] <= LR_RANGE[1]
        assert NUM_LEAVES_RANGE[0] <= params["num_leaves"] <= NUM_LEAVES_RANGE[1]
        assert (
            MIN_CHILD_SAMPLES_RANGE[0] <= params["min_child_samples"] <= MIN_CHILD_SAMPLES_RANGE[1]
        )
        assert REG_LAMBDA_RANGE[0] <= params["reg_lambda"] <= REG_LAMBDA_RANGE[1]
        assert REG_ALPHA_RANGE[0] <= params["reg_alpha"] <= REG_ALPHA_RANGE[1]
        assert FEATURE_FRACTION_RANGE[0] <= params["feature_fraction"] <= FEATURE_FRACTION_RANGE[1]
        assert BAGGING_FRACTION_RANGE[0] <= params["bagging_fraction"] <= BAGGING_FRACTION_RANGE[1]
        assert BAGGING_FREQ_RANGE[0] <= params["bagging_freq"] <= BAGGING_FREQ_RANGE[1]
        return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    study.optimize(objective, n_trials=50)
