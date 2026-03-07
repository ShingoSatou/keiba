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
    PL_REQUIRED_PRED_FEATURES_STACK,
    PL_STACK_CORE_FEATURES,
    PL_STACK_INTERACTION_FEATURES,
)
from scripts_v3.pl_v3_common import materialize_stack_default_pl_features
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
                "track_code": 1,
                "surface": 1,
                "distance_m": 1600,
                "going": 2,
                "weather": 1,
                "grade_code": 3,
                "race_type_code": 1,
                "weight_type_code": 1,
                "condition_code_min_age": 3,
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
                "p_win_stack": 0.141,
                "p_place_stack": 0.271,
                "place_width_log_ratio": 0.29,
                "p_win_odds_t10_norm_cal_isotonic": 0.19,
                "jockey_key": 1001,
                "trainer_key": 2001,
                "finish_pos": 1,
                "extra_numeric_probe": 999.0,
            }
        ]
    )


def test_pl_stack_default_contract_uses_stack_logits_and_interactions() -> None:
    frame = materialize_stack_default_pl_features(_sample_frame())
    feat_cols = _collect_pl_feature_columns(
        frame,
        feature_profile="stack_default",
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_STACK,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert feat_cols[: len(PL_STACK_CORE_FEATURES)] == PL_STACK_CORE_FEATURES
    assert all(col in feat_cols for col in PL_STACK_INTERACTION_FEATURES)
    assert "p_win_stack" not in feat_cols
    assert "p_place_stack" not in feat_cols
    assert "extra_numeric_probe" not in feat_cols
    assert "track_code" not in feat_cols
    assert "surface" not in feat_cols
    assert "field_size" not in feat_cols
    assert "distance_m" not in feat_cols


def test_pl_stack_default_materialization_only_requires_minimal_raw_inputs() -> None:
    frame = _sample_frame().drop(
        columns=[
            "track_code",
            "surface",
            "going",
            "weather",
            "grade_code",
            "race_type_code",
            "weight_type_code",
            "condition_code_min_age",
        ]
    )

    out = materialize_stack_default_pl_features(frame)

    assert "z_win_stack" in out.columns
    assert "z_place_stack" in out.columns
    assert "z_win_stack_x_field_size" in out.columns
    assert "z_win_stack_x_distance_m" in out.columns


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
    assert "odds_win_t10" not in feat_cols
    assert "odds_t10_data_kbn" not in feat_cols
    assert "p_win_odds_t10_raw" not in feat_cols
    assert all(col in feat_cols for col in PL_CONTEXT_FEATURES_SMALL)
    assert all(col not in feat_cols for col in FINAL_ODDS_BASE_FEATURES)


def test_pl_extra_numeric_columns_do_not_expand_feature_list() -> None:
    frame = _sample_frame()
    with_extra = frame.assign(extra_numeric_probe_2=123.0, extra_numeric_probe_3=-5.0)

    required_pred_cols = [*PL_REQUIRED_PRED_FEATURES_META, *PL_META_DEFAULT_ODDS_FEATURES]
    base_cols = _collect_pl_feature_columns(
        frame,
        feature_profile="meta_default",
        required_pred_cols=required_pred_cols,
        include_final_odds=False,
        operational_mode="t10_only",
    )
    extra_cols = _collect_pl_feature_columns(
        with_extra,
        feature_profile="meta_default",
        required_pred_cols=required_pred_cols,
        include_final_odds=False,
        operational_mode="t10_only",
    )

    assert extra_cols == base_cols


def test_pl_artifact_and_meta_include_strict_temporal_stacker_metadata() -> None:
    args = parse_args([])
    stack_frame = materialize_stack_default_pl_features(_sample_frame())
    folds: list[FoldSpec] = []
    feature_cols = _collect_pl_feature_columns(
        stack_frame,
        feature_profile="stack_default",
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_STACK,
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
        train_years=[2022, 2023, 2024],
        train_rows=10,
        train_races=2,
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_STACK,
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
        required_pred_cols=PL_REQUIRED_PRED_FEATURES_STACK,
        pl_feature_cols=feature_cols,
        cv_summary={},
        recent_df=stack_frame.assign(year=2024),
        all_df=stack_frame.assign(year=2024),
        years=[2022, 2023, 2024],
        input_paths={"win_stack_oof": "data/oof/win_stack_oof.parquet"},
        holdout_summary=None,
        year_coverage={
            "base_oof_years": [2020, 2021, 2022, 2023, 2024],
            "stacker_oof_years": [2022, 2023, 2024],
            "pl_oof_valid_years": [],
            "pl_holdout_train_years": [2022, 2023, 2024],
        },
        year_coverage_output=Path("data/oof/v3_pipeline_year_coverage.json"),
    )

    assert args.pl_feature_profile == "stack_default"
    assert args.train_window_years == 3
    assert artifact["cv_policy"]["cv_window_policy"] == "fixed_sliding"
    assert artifact["cv_policy"]["train_window_years"] == 3
    assert artifact["cv_policy"]["holdout_year"] == 2025
    assert artifact["pl_feature_profile"] == "stack_default"
    assert artifact["meta_input_mode"] == "strict_temporal_stacker_oof"
    assert artifact["forbidden_feature_check_passed"] is True
    assert meta["cv_policy"]["valid_years"] == []
    assert meta["pl_feature_profile"] == "stack_default"
    assert meta["meta_input_mode"] == "strict_temporal_stacker_oof"
    assert meta["output_paths"]["year_coverage"].endswith("v3_pipeline_year_coverage.json")


def test_legacy_profiles_use_suffix_paths() -> None:
    meta_args = parse_args(["--pl-feature-profile", "meta_default"])

    meta_outputs = _resolve_output_paths(meta_args)

    assert str(meta_outputs["oof"]).endswith("pl_v3_oof_meta_default.parquet")
    assert str(meta_outputs["metrics"]).endswith("pl_v3_cv_metrics_meta_default.json")
