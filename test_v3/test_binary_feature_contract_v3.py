from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts_v3.cv_policy_v3 import FoldSpec
from scripts_v3.feature_registry_v3 import (
    BINARY_ENTITY_ID_FEATURES,
    BINARY_T10_ODDS_FEATURES,
    FEATURE_MANIFEST_VERSION,
    FORBIDDEN_FINAL_ODDS_FEATURES,
    POST_RACE_FORBIDDEN_FEATURES,
    get_binary_feature_columns,
)
from scripts_v3.train_binary_model_v3 import (
    _apply_params_json,
    _build_feature_manifest_payload,
    _build_meta_payload,
    _resolve_binary_feature_columns,
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
                "year": 2026,
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
                "jockey_target_label_mean_6m": 1.8,
                "trainer_target_label_mean_6m": 1.6,
                "rel_jockey_target_label_mean_z": 0.25,
                "te_window_time": 123,
                "odds_win_t10": 4.5,
                "odds_t10_data_kbn": 3,
                "p_win_odds_t10_raw": 0.22,
                "p_win_odds_t10_norm": 0.18,
                "odds_win_final": 3.8,
                "odds_final_data_kbn": 4,
                "odds_final_announce_dt": "2026-03-01T14:59:00Z",
                "p_win_odds_final_raw": 0.26,
                "p_win_odds_final_norm": 0.2,
                "jockey_key": 1001,
                "trainer_key": 2001,
                "finish_pos": 1,
                "target_label": 3,
                "y_win": 1,
                "y_place": 1,
                "y_top3": 1,
                "pl_score": 0.1,
                "p_top3": 0.4,
                "p_win_stack": 0.33,
                "fold_id": 0,
                "valid_year": 2025,
            }
        ]
    )


def test_binary_default_contract_excludes_final_odds_and_entity_ids() -> None:
    feat_cols = get_binary_feature_columns(
        _sample_frame(),
        include_entity_ids=False,
        operational_mode="t10_only",
    )

    assert all(col not in feat_cols for col in FORBIDDEN_FINAL_ODDS_FEATURES)
    assert all(col not in feat_cols for col in BINARY_ENTITY_ID_FEATURES)
    assert all(col not in feat_cols for col in BINARY_T10_ODDS_FEATURES)
    assert all(col not in feat_cols for col in POST_RACE_FORBIDDEN_FEATURES)


def test_binary_can_opt_in_entity_id_features() -> None:
    feat_cols = get_binary_feature_columns(
        _sample_frame(),
        include_entity_ids=True,
        operational_mode="t10_only",
    )

    assert "jockey_key" in feat_cols
    assert "trainer_key" in feat_cols


def test_binary_te_input_includes_safe_te_columns_only() -> None:
    frame = _sample_frame()

    base_feat_cols, _, _ = _resolve_binary_feature_columns(
        frame=frame,
        input_path=Path("data/features_v3.parquet"),
        include_entity_id_features=False,
        operational_mode="t10_only",
    )
    te_feat_cols, _, _ = _resolve_binary_feature_columns(
        frame=frame,
        input_path=Path("data/features_v3_te.parquet"),
        include_entity_id_features=False,
        operational_mode="t10_only",
    )

    te_only_cols = {
        "jockey_target_label_mean_6m",
        "trainer_target_label_mean_6m",
        "rel_jockey_target_label_mean_z",
    }
    assert all(col not in base_feat_cols for col in te_only_cols)
    assert all(col in te_feat_cols for col in te_only_cols)
    assert "te_window_time" not in te_feat_cols
    assert "p_win_stack" not in te_feat_cols
    assert all(col not in te_feat_cols for col in FORBIDDEN_FINAL_ODDS_FEATURES)
    assert all(col not in te_feat_cols for col in BINARY_ENTITY_ID_FEATURES)


def test_binary_manifest_and_meta_include_feature_contract_fields() -> None:
    args = parse_args([])
    folds = [FoldSpec(fold_id=1, train_years=(2020, 2021, 2022, 2023), valid_year=2024)]
    feat_cols = get_binary_feature_columns(
        _sample_frame(),
        include_entity_ids=False,
        operational_mode="t10_only",
    )
    categorical_cols: list[str] = []

    manifest = _build_feature_manifest_payload(
        args=args,
        feat_cols=feat_cols,
        categorical_cols=categorical_cols,
        include_entity_id_features=False,
        forbidden_feature_check_passed=True,
        valid_years=[2024],
    )
    meta = _build_meta_payload(
        args=args,
        folds=folds,
        label_col="y_win",
        pred_col="p_win_lgbm",
        input_path=Path("data/features_v3.parquet"),
        feat_cols=feat_cols,
        categorical_cols=categorical_cols,
        outputs={
            "oof": Path("data/oof/win_lgbm_oof.parquet"),
            "metrics": Path("data/oof/win_lgbm_cv_metrics.json"),
            "model": Path("models/win_lgbm_v3.txt"),
            "all_years_model": Path("models/win_lgbm_all_years_v3.txt"),
            "feature_manifest": Path("models/win_lgbm_feature_manifest_v3.json"),
            "holdout": Path("data/holdout/win_lgbm_holdout_pred_v3.parquet"),
        },
        metrics_summary={"n_folds": 0},
        recent_df=_sample_frame(),
        all_df=_sample_frame(),
        years=[2024],
        include_entity_id_features=False,
        forbidden_feature_check_passed=True,
    )

    for payload in (manifest, meta):
        assert payload["cv_policy"]["cv_window_policy"] == "fixed_sliding"
        assert payload["cv_policy"]["train_window_years"] == 4
        assert payload["cv_policy"]["valid_years"] == [2024]
        assert payload["cv_policy"]["holdout_year"] == 2025
        assert payload["operational_mode"] == "t10_only"
        assert payload["include_entity_id_features"] is False
        assert payload["feature_columns"] == feat_cols
        assert payload["forbidden_feature_check_passed"] is True
        assert payload["feature_manifest_version"] == FEATURE_MANIFEST_VERSION

    assert args.train_window_years == 4


def test_apply_params_json_uses_tuned_input_and_holdout_defaults(tmp_path: Path) -> None:
    train_input = tmp_path / "features_v3_te.parquet"
    holdout_input = tmp_path / "features_v3_te_2025.parquet"
    params_path = tmp_path / "binary_v3_win_lgbm_best_params.json"
    train_input.write_bytes(b"parquet")
    holdout_input.write_bytes(b"parquet")
    params_path.write_text("{}", encoding="utf-8")

    args = parse_args(["--task", "win", "--model", "lgbm"])
    params = {
        "task": "win",
        "model": "lgbm",
        "input": str(train_input),
        "feature_set": "te",
        "train_window_years": 4,
        "operational_mode": "t10_only",
        "include_entity_id_features": False,
        "lgbm_params": {
            "learning_rate": 0.031,
            "num_leaves": 111,
            "min_data_in_leaf": 44,
            "lambda_l1": 0.12,
            "lambda_l2": 1.5,
            "feature_fraction": 0.88,
            "bagging_fraction": 0.77,
            "bagging_freq": 5,
        },
        "final_num_boost_round": 137,
    }

    _apply_params_json(args, params, argv=[], params_path=params_path)

    assert args.input == str(train_input)
    assert args.holdout_input == str(holdout_input)
    assert args.learning_rate == 0.031
    assert args.num_leaves == 111
    assert args.min_data_in_leaf == 44
    assert args._tuned_final_iterations == 137
    assert args._applied_params_json == str(params_path)
    assert args._applied_params_feature_set == "te"


def test_apply_params_json_preserves_explicit_cli_overrides(tmp_path: Path) -> None:
    train_input = tmp_path / "features_v3_te.parquet"
    holdout_input = tmp_path / "features_v3_te_2025.parquet"
    params_path = tmp_path / "binary_v3_win_lgbm_best_params.json"
    train_input.write_bytes(b"parquet")
    holdout_input.write_bytes(b"parquet")
    params_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    argv = [
        "--task",
        "win",
        "--model",
        "lgbm",
        "--input",
        "data/features_v3.parquet",
        "--learning-rate",
        "0.2",
        "--num-boost-round",
        "123",
    ]
    args = parse_args(argv)
    params = {
        "task": "win",
        "model": "lgbm",
        "input": str(train_input),
        "feature_set": "te",
        "lgbm_params": {
            "learning_rate": 0.031,
            "num_leaves": 111,
            "min_data_in_leaf": 44,
            "lambda_l1": 0.12,
            "lambda_l2": 1.5,
            "feature_fraction": 0.88,
            "bagging_fraction": 0.77,
            "bagging_freq": 5,
        },
        "final_num_boost_round": 137,
    }

    _apply_params_json(args, params, argv=argv, params_path=params_path)

    assert args.input == "data/features_v3.parquet"
    assert args.holdout_input == ""
    assert args.learning_rate == 0.2
    assert args.num_leaves == 111
    assert args._tuned_final_iterations is None


def test_apply_params_json_rejects_task_model_mismatch(tmp_path: Path) -> None:
    params_path = tmp_path / "binary_v3_win_lgbm_best_params.json"
    params_path.write_text("{}", encoding="utf-8")
    args = parse_args(["--task", "win", "--model", "lgbm"])

    try:
        _apply_params_json(
            args,
            {"task": "place", "model": "lgbm"},
            argv=[],
            params_path=params_path,
        )
    except SystemExit as exc:
        assert "task mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("SystemExit not raised for mismatched params-json task")
