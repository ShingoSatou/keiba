from __future__ import annotations

import pandas as pd
import pytest

from scripts_v3.feature_registry_v3 import (
    BINARY_ENTITY_ID_FEATURES,
    FORBIDDEN_FINAL_ODDS_FEATURES,
    PL_META_DEFAULT_ODDS_FEATURES,
    PL_REQUIRED_PRED_FEATURES_META,
    PL_REQUIRED_PRED_FEATURES_STACK,
    PL_STACK_CORE_FEATURES,
    PL_STACK_INTERACTION_FEATURES,
    STACKER_PLACE_ODDS_FEATURES,
    STACKER_REQUIRED_PRED_FEATURES_PLACE,
    STACKER_REQUIRED_PRED_FEATURES_WIN,
    STACKER_WIN_ODDS_FEATURES,
    get_binary_feature_columns,
    get_pl_feature_columns,
    get_pl_required_pred_columns,
    get_stacker_feature_columns,
    validate_feature_contract,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_id": 202601010101,
                "track_code": 1,
                "surface": 1,
                "distance_m": 1600,
                "going": 2,
                "weather": 1,
                "field_size": 16,
                "grade_code": 3,
                "race_type_code": 1,
                "weight_type_code": 1,
                "condition_code_min_age": 3,
                "age": 4,
                "sex": 1,
                "carried_weight": 56.0,
                "body_weight": 470.0,
                "body_weight_diff": 2.0,
                "is_jockey_change": 0,
                "days_since_lag1": 28,
                "lag1_distance_diff": 200,
                "lag1_course_type_match": 1,
                "lag1_finish_pos": 2,
                "lag2_finish_pos": 4,
                "lag3_finish_pos": 6,
                "lag1_speed_index": 72.0,
                "lag2_speed_index": 69.0,
                "lag3_speed_index": 66.0,
                "lag1_up3_index": 70.0,
                "lag2_up3_index": 67.0,
                "lag3_up3_index": 65.0,
                "apt_same_distance_top3_rate_2y": 0.4,
                "apt_same_going_top3_rate_2y": 0.3,
                "meta_dm_time_x10": 123,
                "meta_dm_rank": 2,
                "meta_tm_score": 71.0,
                "meta_tm_rank": 3,
                "jockey_top3_rate_6m": 0.35,
                "trainer_top3_rate_6m": 0.31,
                "rel_lag1_speed_index_z": 0.8,
                "rel_lag1_speed_index_rank": 2,
                "rel_lag1_speed_index_pct": 0.9,
                "rel_carried_weight_z": -0.3,
                "rel_jockey_top3_rate_z": 0.4,
                "rel_meta_tm_score_z": 0.7,
                "odds_win_t20": 4.8,
                "odds_win_t15": 4.6,
                "odds_win_t10": 4.5,
                "odds_t10_data_kbn": 3,
                "p_win_odds_t10_raw": 0.22,
                "p_win_odds_t10_norm": 0.18,
                "odds_place_t20_lower": 1.7,
                "odds_place_t20_upper": 2.2,
                "place_width_log_ratio_t20": 0.26,
                "odds_place_t15_lower": 1.6,
                "odds_place_t15_upper": 2.1,
                "place_width_log_ratio_t15": 0.27,
                "odds_place_t10_lower": 1.5,
                "odds_place_t10_upper": 2.0,
                "place_width_log_ratio_t10": 0.29,
                "place_width_log_ratio": 0.29,
                "odds_win_final": 3.8,
                "odds_final_data_kbn": 4,
                "odds_final_announce_dt": "2026-03-01T14:59:00Z",
                "p_win_odds_final_raw": 0.26,
                "p_win_odds_final_norm": 0.2,
                "p_win_odds_final_norm_cal_isotonic": 0.21,
                "p_win_odds_final_norm_cal_logreg": 0.2,
                "jockey_key": 1001,
                "trainer_key": 2001,
                "finish_pos": 1,
                "y_win": 1,
                "y_place": 1,
                "y_top3": 1,
                "pl_score": 0.1,
                "p_top3": 0.4,
                "fold_id": 0,
                "valid_year": 2025,
                "p_win_lgbm": 0.12,
                "p_win_xgb": 0.13,
                "p_win_cat": 0.14,
                "p_place_lgbm": 0.25,
                "p_place_xgb": 0.26,
                "p_place_cat": 0.27,
                "p_win_meta": 0.131,
                "p_place_meta": 0.261,
                "p_win_stack": 0.141,
                "p_place_stack": 0.271,
                "z_win_stack": -1.806,
                "z_place_stack": -0.989,
                "z_win_stack_x_z_place_stack": 1.786,
                "z_win_stack_x_place_width_log_ratio": -0.524,
                "z_place_stack_x_place_width_log_ratio": -0.287,
                "z_win_stack_x_field_size": -28.896,
                "z_place_stack_x_field_size": -15.824,
                "z_win_stack_x_distance_m": -2889.6,
                "z_place_stack_x_distance_m": -1582.4,
                "z_win_stack_race_centered": 0.0,
                "z_place_stack_race_centered": 0.0,
                "place_width_log_ratio_race_centered": 0.0,
                "z_win_stack_rank_pct": 1.0,
                "z_place_stack_rank_pct": 1.0,
                "place_width_log_ratio_rank_pct": 1.0,
                "extra_numeric_probe": 999.0,
            }
        ]
    )


def test_registry_functions_return_unique_columns() -> None:
    frame = _sample_frame()

    binary_cols = get_binary_feature_columns(
        frame,
        include_entity_ids=False,
        operational_mode="t10_only",
    )
    required_pred_cols = [*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES]
    pl_cols = get_pl_feature_columns(
        frame,
        feature_profile="meta_default",
        required_pred_cols=required_pred_cols,
        include_context=True,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert len(binary_cols) == len(set(binary_cols))
    assert len(pl_cols) == len(set(pl_cols))
    assert all(col not in binary_cols for col in BINARY_ENTITY_ID_FEATURES)
    assert "extra_numeric_probe" not in pl_cols


def test_meta_default_required_pred_columns_are_compact() -> None:
    required = get_pl_required_pred_columns("meta_default")
    assert required == [*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES]


def test_stack_default_required_pred_columns_are_compact() -> None:
    required = get_pl_required_pred_columns("stack_default")
    assert required == PL_REQUIRED_PRED_FEATURES_STACK


def test_stacker_feature_columns_are_task_specific() -> None:
    frame = _sample_frame()
    win_cols = get_stacker_feature_columns(frame, task="win")
    place_cols = get_stacker_feature_columns(frame, task="place")
    expected_win = [*STACKER_REQUIRED_PRED_FEATURES_WIN, *STACKER_WIN_ODDS_FEATURES]
    expected_place = [*STACKER_REQUIRED_PRED_FEATURES_PLACE, *STACKER_PLACE_ODDS_FEATURES]

    assert all(col in win_cols for col in expected_win)
    assert all(col in place_cols for col in expected_place)
    assert all(col not in win_cols for col in STACKER_PLACE_ODDS_FEATURES)
    assert all(col not in place_cols for col in STACKER_WIN_ODDS_FEATURES)


def test_stack_default_feature_columns_use_logit_and_interaction_contract() -> None:
    frame = _sample_frame()
    feat_cols = get_pl_feature_columns(
        frame,
        feature_profile="stack_default",
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_STACK,
        include_context=True,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert feat_cols[: len(PL_STACK_CORE_FEATURES)] == PL_STACK_CORE_FEATURES
    assert all(col in feat_cols for col in PL_STACK_INTERACTION_FEATURES)
    assert "p_win_stack" not in feat_cols
    assert "p_place_stack" not in feat_cols


def test_validate_feature_contract_raises_on_forbidden_columns() -> None:
    with pytest.raises(ValueError):
        validate_feature_contract(
            ["odds_win_t10", FORBIDDEN_FINAL_ODDS_FEATURES[0]],
            operational_mode="t10_only",
            stage="binary",
        )

    with pytest.raises(ValueError):
        validate_feature_contract(
            ["field_size", "finish_pos"],
            operational_mode="includes_final",
            stage="pl",
        )
