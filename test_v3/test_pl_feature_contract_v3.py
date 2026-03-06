from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts_v3.cv_policy_v3 import FoldSpec
from scripts_v3.feature_registry_v3 import (
    FINAL_ODDS_BASE_FEATURES,
    PL_CONTEXT_FEATURES_SMALL,
    PL_META_DEFAULT_ODDS_FEATURES,
    PL_REQUIRED_PRED_FEATURES_META,
    PL_REQUIRED_PRED_FEATURES_RAW_LEGACY,
    PL_T10_ODDS_FEATURES,
)
from scripts_v3.train_pl_v3 import (
    _artifact_from_fit,
    _build_pl_meta_payload,
    _collect_pl_feature_columns,
    _resolve_output_paths,
    parse_args,
)


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
                "p_win_meta": 0.131,
                "p_place_meta": 0.261,
                "p_win_odds_t10_norm_cal_isotonic": 0.19,
                "jockey_key": 1001,
                "trainer_key": 2001,
                "finish_pos": 1,
                "extra_numeric_probe": 999.0,
            }
        ]
    )


def test_pl_meta_default_contract_is_compact() -> None:
    required_pred_cols = [*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES]
    feat_cols = _collect_pl_feature_columns(
        _sample_frame(),
        feature_profile="meta_default",
        required_pred_cols=required_pred_cols,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert feat_cols[: len(required_pred_cols)] == required_pred_cols
    assert all(col not in feat_cols for col in PL_T10_ODDS_FEATURES if col != "p_win_odds_t10_norm")
    assert all(col in feat_cols for col in PL_CONTEXT_FEATURES_SMALL)
    assert all(col not in feat_cols for col in FINAL_ODDS_BASE_FEATURES)
    assert "extra_numeric_probe" not in feat_cols
    assert "jockey_key" not in feat_cols
    assert "trainer_key" not in feat_cols
    assert "finish_pos" not in feat_cols


def test_pl_raw_legacy_can_include_final_odds_only_when_requested() -> None:
    feat_cols = _collect_pl_feature_columns(
        _sample_frame(),
        feature_profile="raw_legacy",
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_RAW_LEGACY,
        include_final_odds=True,
        operational_mode="includes_final",
    )

    assert all(col in feat_cols for col in FINAL_ODDS_BASE_FEATURES)


def test_pl_extra_numeric_columns_do_not_expand_feature_list() -> None:
    frame = _sample_frame()
    with_extra = frame.assign(extra_numeric_probe_2=123.0, extra_numeric_probe_3=-5.0)

    base_cols = _collect_pl_feature_columns(
        frame,
        feature_profile="raw_legacy",
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_RAW_LEGACY,
        include_final_odds=False,
        operational_mode="t10_only",
    )
    extra_cols = _collect_pl_feature_columns(
        with_extra,
        feature_profile="raw_legacy",
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_RAW_LEGACY,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert extra_cols == base_cols


def test_pl_artifact_and_meta_include_cv_policy() -> None:
    args = parse_args([])
    folds = [FoldSpec(fold_id=1, train_years=(2020, 2021, 2022, 2023), valid_year=2024)]
    feature_cols = _collect_pl_feature_columns(
        _sample_frame(),
        feature_profile="meta_default",
        required_pred_cols=[*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES],
        include_final_odds=False,
        operational_mode="t10_only",
    )
    artifact = _artifact_from_fit(
        feature_cols=feature_cols,
        weights=np.ones(len(feature_cols)),
        stats={
            "median": {col: 0.0 for col in feature_cols},
            "mean": {col: 0.0 for col in feature_cols},
            "std": {col: 1.0 for col in feature_cols},
        },
        args=args,
        train_years=[2021, 2022, 2023, 2024],
        train_rows=10,
        train_races=2,
        required_pred_cols=[*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES],
    )
    meta = _build_pl_meta_payload(
        args=args,
        folds=folds,
        features_path=Path("data/features_v3.parquet"),
        oof_output=Path("data/oof/pl_v3_oof.parquet"),
        wide_oof_output=Path("data/oof/pl_v3_wide_oof.parquet"),
        metrics_output=Path("data/oof/pl_v3_cv_metrics.json"),
        model_output=Path("models/pl_v3_recent_window.joblib"),
        all_years_model_output=Path("models/pl_v3_all_years.joblib"),
        holdout_output=Path("data/oof/pl_v3_holdout_2025_pred.parquet"),
        required_pred_cols=[*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES],
        pl_feature_cols=feature_cols,
        cv_summary={},
        recent_df=_sample_frame().assign(year=2024),
        all_df=_sample_frame().assign(year=2024),
        years=[2024],
        input_paths={"win_meta_oof": "data/oof/win_meta_oof.parquet"},
        holdout_summary=None,
    )

    assert artifact["cv_policy"]["cv_window_policy"] == "fixed_sliding"
    assert artifact["cv_policy"]["train_window_years"] == 4
    assert artifact["cv_policy"]["holdout_year"] == 2025
    assert artifact["pl_feature_profile"] == "meta_default"
    assert artifact["meta_input_mode"] == "grouped_reference_oof"
    assert artifact["forbidden_feature_check_passed"] is True
    assert meta["cv_policy"]["valid_years"] == [2024]
    assert meta["pl_feature_profile"] == "meta_default"
    assert meta["meta_input_mode"] == "grouped_reference_oof"
    assert meta["forbidden_feature_check_passed"] is True
    assert (
        meta["cv_policy"]["window_definition"]
        == "train = previous 4 years only, valid = current year"
    )


def test_raw_legacy_suffix_paths_are_applied() -> None:
    args = parse_args(["--pl-feature-profile", "raw_legacy"])
    outputs = _resolve_output_paths(args)
    assert args.pl_feature_profile == "raw_legacy"
    assert str(outputs["oof"]).endswith("pl_v3_oof_raw_legacy.parquet")
    assert str(outputs["metrics"]).endswith("pl_v3_cv_metrics_raw_legacy.json")
    assert str(outputs["holdout"]).endswith("pl_v3_holdout_2025_pred_raw_legacy.parquet")
