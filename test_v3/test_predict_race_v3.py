from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scripts_v3.feature_registry_v3 import PL_STACK_CORE_FEATURES, PL_STACK_INTERACTION_FEATURES
from scripts_v3.predict_race_v3 import _score_with_pl, main


def _meta_model_artifact(pred_col: str, feature_columns: list[str]) -> dict[str, object]:
    x = np.array(
        [
            [0.80, 0.75, 0.70],
            [0.30, 0.35, 0.40],
            [0.78, 0.72, 0.69],
            [0.28, 0.33, 0.38],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 1, 0], dtype=int)
    model = LogisticRegression(random_state=42, max_iter=3000, solver="lbfgs")
    model.fit(x, y)
    return {
        "pred_col": pred_col,
        "feature_columns": feature_columns,
        "model": model,
        "preprocess": {"type": "identity"},
    }


def _pl_artifact(feature_columns: list[str], *, profile: str) -> dict[str, object]:
    return {
        "feature_columns": feature_columns,
        "weights": [0.2] * len(feature_columns),
        "preprocess": {
            "median": {col: 0.0 for col in feature_columns},
            "mean": {col: 0.0 for col in feature_columns},
            "std": {col: 1.0 for col in feature_columns},
        },
        "cv_policy": {
            "cv_window_policy": "fixed_sliding",
            "train_window_years": 4,
            "holdout_year": 2025,
            "window_definition": "train = previous 4 years only, valid = current year",
        },
        "pl_feature_profile": profile,
    }


def _input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": [202503010101, 202503010101],
            "horse_id": ["H001", "H002"],
            "horse_no": [1, 2],
            "race_date": ["2025-03-01", "2025-03-01"],
            "track_code": [5, 5],
            "field_size": [2, 2],
            "surface": [1, 1],
            "distance_m": [1600, 1600],
            "going": [2, 2],
            "weather": [1, 1],
            "grade_code": [3, 3],
            "race_type_code": [1, 1],
            "weight_type_code": [1, 1],
            "condition_code_min_age": [3, 3],
            "apt_same_distance_top3_rate_2y": [0.42, 0.25],
            "apt_same_going_top3_rate_2y": [0.38, 0.20],
            "rel_lag1_speed_index_z": [0.8, -0.3],
            "rel_meta_tm_score_z": [0.7, -0.2],
            "odds_win_t20": [4.2, 12.0],
            "odds_win_t15": [4.0, 11.5],
            "odds_win_t10": [3.8, 11.2],
            "p_win_odds_t20_norm": [0.74, 0.26],
            "p_win_odds_t15_norm": [0.742, 0.258],
            "p_win_odds_t10_raw": [0.26, 0.09],
            "p_win_odds_t10_norm": [0.23, 0.08],
            "d_logit_win_15_20": [0.01, -0.01],
            "d_logit_win_10_15": [0.02, -0.02],
            "d_logit_win_10_20": [0.03, -0.03],
            "odds_t10_data_kbn": [3, 3],
            "odds_win_t20_asof_dt": ["2025-03-01 14:40:00", "2025-03-01 14:40:00"],
            "odds_win_t20_announce_dt": ["2025-03-01 14:39:00", "2025-03-01 14:39:00"],
            "odds_win_t15_asof_dt": ["2025-03-01 14:45:00", "2025-03-01 14:45:00"],
            "odds_win_t15_announce_dt": ["2025-03-01 14:44:00", "2025-03-01 14:44:00"],
            "odds_t10_asof_dt": ["2025-03-01 14:50:00", "2025-03-01 14:50:00"],
            "odds_t10_announce_dt": ["2025-03-01 14:49:00", "2025-03-01 14:49:00"],
            "odds_place_t20_lower": [1.5, 2.7],
            "odds_place_t20_upper": [1.9, 3.5],
            "place_mid_prob_t20": [0.592, 0.325],
            "place_width_log_ratio_t20": [0.236, 0.260],
            "odds_place_t15_lower": [1.5, 2.6],
            "odds_place_t15_upper": [1.9, 3.4],
            "place_mid_prob_t15": [0.592, 0.336],
            "place_width_log_ratio_t15": [0.236, 0.268],
            "odds_place_t10_lower": [1.4, 2.5],
            "odds_place_t10_upper": [1.8, 3.3],
            "place_mid_prob_t10": [0.630, 0.348],
            "place_width_log_ratio_t10": [0.251, 0.278],
            "d_place_mid_10_20": [0.038, 0.023],
            "d_place_width_10_20": [0.015, 0.018],
            "place_width_log_ratio": [0.251, 0.278],
            "odds_place_t20_asof_dt": ["2025-03-01 14:40:00", "2025-03-01 14:40:00"],
            "odds_place_t20_announce_dt": ["2025-03-01 14:39:00", "2025-03-01 14:39:00"],
            "odds_place_t15_asof_dt": ["2025-03-01 14:45:00", "2025-03-01 14:45:00"],
            "odds_place_t15_announce_dt": ["2025-03-01 14:44:00", "2025-03-01 14:44:00"],
            "odds_place_t10_asof_dt": ["2025-03-01 14:50:00", "2025-03-01 14:50:00"],
            "odds_place_t10_announce_dt": ["2025-03-01 14:49:00", "2025-03-01 14:49:00"],
            "p_win_lgbm": [0.80, 0.30],
            "p_win_xgb": [0.75, 0.35],
            "p_win_cat": [0.70, 0.40],
            "p_place_lgbm": [0.82, 0.48],
            "p_place_xgb": [0.77, 0.45],
            "p_place_cat": [0.72, 0.43],
        }
    )


def _stack_meta_json(
    pred_col: str,
    feature_columns: list[str],
    model_path: Path,
) -> dict[str, object]:
    return {
        "task": "win" if pred_col == "p_win_stack" else "place",
        "model": "lgbm",
        "pred_col": pred_col,
        "feature_columns": feature_columns,
        "output_paths": {"main_model": str(model_path)},
    }


def test_score_with_pl_rejects_final_features_in_t10_path() -> None:
    frame = pd.DataFrame({"p_win_odds_final_norm": [0.2]})
    artifact = _pl_artifact(["p_win_odds_final_norm"], profile="meta_default")
    with pytest.raises(SystemExit):
        _score_with_pl(frame, artifact)


def test_predict_race_meta_default_path_builds_meta_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "race.parquet"
    output_path = tmp_path / "pred.parquet"
    frame = _input_frame()
    frame.to_parquet(input_path, index=False)

    win_meta_model = tmp_path / "win_meta_v3.pkl"
    place_meta_model = tmp_path / "place_meta_v3.pkl"
    joblib.dump(
        _meta_model_artifact("p_win_meta", ["p_win_lgbm", "p_win_xgb", "p_win_cat"]),
        win_meta_model,
    )
    joblib.dump(
        _meta_model_artifact(
            "p_place_meta",
            ["p_place_lgbm", "p_place_xgb", "p_place_cat"],
        ),
        place_meta_model,
    )

    pl_model = tmp_path / "pl.joblib"
    joblib.dump(
        _pl_artifact(
            [
                "p_win_meta",
                "p_place_meta",
                "p_win_odds_t10_norm",
                "field_size",
                "surface",
                "distance_m",
                "going",
                "apt_same_distance_top3_rate_2y",
                "apt_same_going_top3_rate_2y",
                "rel_lag1_speed_index_z",
                "rel_meta_tm_score_z",
            ],
            profile="meta_default",
        ),
        pl_model,
    )

    rc = main(
        [
            "--input",
            str(input_path),
            "--pl-model",
            str(pl_model),
            "--win-meta-model",
            str(win_meta_model),
            "--place-meta-model",
            str(place_meta_model),
            "--skip-base-model-inference",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    out = pd.read_parquet(output_path)
    assert {"p_win_meta", "p_place_meta", "pl_score", "p_top3"}.issubset(out.columns)
    assert out["p_win_meta"].between(0.0, 1.0).all()
    assert out["p_place_meta"].between(0.0, 1.0).all()


def test_predict_race_stack_default_path_materializes_stack_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "race_stack.parquet"
    output_path = tmp_path / "pred_stack.parquet"
    frame = _input_frame().assign(
        p_win_stack=[0.78, 0.22],
        p_place_stack=[0.80, 0.40],
    )
    frame.to_parquet(input_path, index=False)

    pl_model = tmp_path / "pl_stack.joblib"
    joblib.dump(
        _pl_artifact(
            [*PL_STACK_CORE_FEATURES, *PL_STACK_INTERACTION_FEATURES],
            profile="stack_default",
        ),
        pl_model,
    )

    rc = main(
        [
            "--input",
            str(input_path),
            "--pl-model",
            str(pl_model),
            "--skip-base-model-inference",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    out = pd.read_parquet(output_path)
    expected_cols = {
        "p_win_stack",
        "p_place_stack",
        "z_win_stack",
        "z_place_stack",
        "place_width_log_ratio",
    }
    assert expected_cols.issubset(out.columns)
    assert out["pl_score"].notna().all()
    assert out["p_top3"].notna().all()


def test_predict_race_stacker_meta_fails_fast_on_missing_market_features(tmp_path: Path) -> None:
    input_path = tmp_path / "race_missing_stack_feature.parquet"
    pl_model = tmp_path / "pl_stack.joblib"
    stack_model_path = tmp_path / "stack_model.txt"
    win_stack_meta = tmp_path / "win_stack_meta.json"
    place_stack_meta = tmp_path / "place_stack_meta.json"

    frame = _input_frame().drop(columns=["d_logit_win_10_20"])
    frame.to_parquet(input_path, index=False)

    joblib.dump(
        _pl_artifact(
            [*PL_STACK_CORE_FEATURES, *PL_STACK_INTERACTION_FEATURES],
            profile="stack_default",
        ),
        pl_model,
    )
    stack_model_path.write_text("placeholder", encoding="utf-8")
    win_stack_meta.write_text(
        json.dumps(
            _stack_meta_json(
                "p_win_stack",
                ["p_win_odds_t20_norm", "d_logit_win_10_20"],
                stack_model_path,
            )
        ),
        encoding="utf-8",
    )
    place_stack_meta.write_text(
        json.dumps(
            _stack_meta_json(
                "p_place_stack",
                ["place_mid_prob_t10", "d_place_width_10_20"],
                stack_model_path,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Input missing model features"):
        main(
            [
                "--input",
                str(input_path),
                "--pl-model",
                str(pl_model),
                "--skip-base-model-inference",
                "--win-stack-meta",
                str(win_stack_meta),
                "--place-stack-meta",
                str(place_stack_meta),
            ]
        )
