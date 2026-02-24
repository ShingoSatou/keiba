from __future__ import annotations

import pytest

from scripts_v2.tune_ranker_cat_optuna_v2 import (  # noqa: E402
    BAGGING_TEMPERATURE_RANGE,
    DEPTH_RANGE,
    FEATURE_SET_CHOICES,
    L2_LEAF_REG_RANGE,
    LEAF_ESTIMATION_ITER_RANGE,
    LOSS_FUNCTION_CHOICES,
    LR_RANGE,
    MIN_DATA_IN_LEAF_RANGE,
    RANDOM_STRENGTH_RANGE,
    RSM_RANGE,
    drop_entity_id_features,
    suggest_cat_params,
    suggest_feature_set,
)

optuna = pytest.importorskip("optuna")


def test_drop_entity_id_features_always_removes_raw_ids():
    cols = ["a", "jockey_key", "b", "trainer_key", "c"]
    assert drop_entity_id_features(cols) == ["a", "b", "c"]


def test_optuna_trial_suggestion_is_within_declared_ranges():
    def objective(trial: optuna.Trial) -> float:
        feature_set = suggest_feature_set(trial)
        assert feature_set in FEATURE_SET_CHOICES

        params = suggest_cat_params(trial)
        assert params["loss_function"] in LOSS_FUNCTION_CHOICES
        assert LR_RANGE[0] <= params["learning_rate"] <= LR_RANGE[1]
        assert DEPTH_RANGE[0] <= params["depth"] <= DEPTH_RANGE[1]
        assert L2_LEAF_REG_RANGE[0] <= params["l2_leaf_reg"] <= L2_LEAF_REG_RANGE[1]
        assert RANDOM_STRENGTH_RANGE[0] <= params["random_strength"] <= RANDOM_STRENGTH_RANGE[1]
        assert (
            BAGGING_TEMPERATURE_RANGE[0]
            <= params["bagging_temperature"]
            <= BAGGING_TEMPERATURE_RANGE[1]
        )
        assert RSM_RANGE[0] <= params["rsm"] <= RSM_RANGE[1]
        assert MIN_DATA_IN_LEAF_RANGE[0] <= params["min_data_in_leaf"] <= MIN_DATA_IN_LEAF_RANGE[1]
        assert (
            LEAF_ESTIMATION_ITER_RANGE[0]
            <= params["leaf_estimation_iterations"]
            <= LEAF_ESTIMATION_ITER_RANGE[1]
        )
        return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    study.optimize(objective, n_trials=50)
