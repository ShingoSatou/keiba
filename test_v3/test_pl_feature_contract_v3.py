from __future__ import annotations

import pandas as pd

from scripts_v3.feature_registry_v3 import (
    FINAL_ODDS_BASE_FEATURES,
    PL_CONTEXT_FEATURES_SMALL,
    PL_REQUIRED_PRED_FEATURES,
    PL_T10_ODDS_FEATURES,
)
from scripts_v3.train_pl_v3 import _collect_pl_feature_columns


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_id": 202601010101,
                "horse_id": "H0001",
                "horse_no": 1,
                "race_date": "2026-03-01",
                "field_size": 16,
                "surface": 1,
                "distance_m": 1600,
                "going": 2,
                "apt_same_distance_top3_rate_2y": 0.4,
                "apt_same_going_top3_rate_2y": 0.3,
                "rel_lag1_speed_index_z": 0.8,
                "rel_meta_tm_score_z": 0.7,
                "odds_win_t10": 4.5,
                "odds_t10_data_kbn": 3,
                "p_win_odds_t10_raw": 0.22,
                "p_win_odds_t10_norm": 0.18,
                "odds_win_final": 3.8,
                "odds_final_data_kbn": 4,
                "p_win_odds_final_raw": 0.26,
                "p_win_odds_final_norm": 0.2,
                "p_win_lgbm": 0.12,
                "p_win_xgb": 0.13,
                "p_win_cat": 0.14,
                "p_place_lgbm": 0.25,
                "p_place_xgb": 0.26,
                "p_place_cat": 0.27,
                "p_win_odds_t10_norm_cal_isotonic": 0.19,
                "jockey_key": 1001,
                "trainer_key": 2001,
                "finish_pos": 1,
                "extra_numeric_probe": 999.0,
            }
        ]
    )


def test_pl_default_contract_is_required_preds_plus_t10_and_small_context() -> None:
    required_pred_cols = [*PL_REQUIRED_PRED_FEATURES, "p_win_odds_t10_norm_cal_isotonic"]
    feat_cols = _collect_pl_feature_columns(
        _sample_frame(),
        required_pred_cols=required_pred_cols,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert feat_cols[: len(required_pred_cols)] == required_pred_cols
    assert all(col in feat_cols for col in PL_T10_ODDS_FEATURES)
    assert all(col in feat_cols for col in PL_CONTEXT_FEATURES_SMALL)
    assert all(col not in feat_cols for col in FINAL_ODDS_BASE_FEATURES)
    assert "extra_numeric_probe" not in feat_cols
    assert "jockey_key" not in feat_cols
    assert "trainer_key" not in feat_cols
    assert "finish_pos" not in feat_cols


def test_pl_can_include_final_odds_only_when_requested() -> None:
    feat_cols = _collect_pl_feature_columns(
        _sample_frame(),
        required_pred_cols=PL_REQUIRED_PRED_FEATURES,
        include_final_odds=True,
        operational_mode="includes_final",
    )

    assert all(col in feat_cols for col in FINAL_ODDS_BASE_FEATURES)


def test_pl_extra_numeric_columns_do_not_expand_feature_list() -> None:
    frame = _sample_frame()
    with_extra = frame.assign(extra_numeric_probe_2=123.0, extra_numeric_probe_3=-5.0)

    base_cols = _collect_pl_feature_columns(
        frame,
        required_pred_cols=PL_REQUIRED_PRED_FEATURES,
        include_final_odds=False,
        operational_mode="t10_only",
    )
    extra_cols = _collect_pl_feature_columns(
        with_extra,
        required_pred_cols=PL_REQUIRED_PRED_FEATURES,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert extra_cols == base_cols
