from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts_v3.build_features_v3_te import (
    build_features_v3_te,
    build_features_v3_te_meta_payload,
    main,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_id": 202501010101,
                "horse_id": "H001",
                "horse_no": 1,
                "race_date": "2025-01-01",
                "y_win": 1,
                "y_place": 1,
                "finish_pos": 1.0,
                "odds_win_final": 3.1,
                "odds_final_data_kbn": 4,
                "p_win_odds_final_raw": 1 / 3.1,
                "p_win_odds_final_norm": 0.32,
                "odds_win_t20": 3.4,
                "odds_win_t15": 3.2,
                "odds_win_t10": 3.0,
                "odds_t10_data_kbn": 3,
                "p_win_odds_t10_raw": 1 / 3.0,
                "p_win_odds_t20_norm": 0.30,
                "p_win_odds_t15_norm": 0.31,
                "p_win_odds_t10_norm": 0.33,
                "d_logit_win_15_20": 0.05,
                "d_logit_win_10_15": 0.07,
                "d_logit_win_10_20": 0.12,
                "odds_place_t20_lower": 1.4,
                "odds_place_t20_upper": 1.7,
                "odds_place_t15_lower": 1.3,
                "odds_place_t15_upper": 1.6,
                "odds_place_t10_lower": 1.2,
                "odds_place_t10_upper": 1.5,
                "place_mid_prob_t20": 0.65,
                "place_mid_prob_t15": 0.69,
                "place_mid_prob_t10": 0.75,
                "d_place_mid_10_20": 0.10,
                "d_place_width_10_20": -0.03,
                "place_width_log_ratio": 0.22,
            },
            {
                "race_id": 202501010101,
                "horse_id": "H002",
                "horse_no": 2,
                "race_date": "2025-01-01",
                "y_win": 0,
                "y_place": 1,
                "finish_pos": 2.0,
                "odds_win_final": 4.3,
                "odds_final_data_kbn": 4,
                "p_win_odds_final_raw": 1 / 4.3,
                "p_win_odds_final_norm": 0.23,
                "odds_win_t20": 4.6,
                "odds_win_t15": 4.4,
                "odds_win_t10": 4.2,
                "odds_t10_data_kbn": 3,
                "p_win_odds_t10_raw": 1 / 4.2,
                "p_win_odds_t20_norm": 0.22,
                "p_win_odds_t15_norm": 0.23,
                "p_win_odds_t10_norm": 0.24,
                "d_logit_win_15_20": 0.03,
                "d_logit_win_10_15": 0.02,
                "d_logit_win_10_20": 0.05,
                "odds_place_t20_lower": 1.7,
                "odds_place_t20_upper": 2.0,
                "odds_place_t15_lower": 1.6,
                "odds_place_t15_upper": 1.9,
                "odds_place_t10_lower": 1.5,
                "odds_place_t10_upper": 1.8,
                "place_mid_prob_t20": 0.54,
                "place_mid_prob_t15": 0.57,
                "place_mid_prob_t10": 0.61,
                "d_place_mid_10_20": 0.07,
                "d_place_width_10_20": -0.02,
                "place_width_log_ratio": 0.18,
            },
        ]
    )


def _te_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_id": 202501010101,
                "horse_id": "H001",
                "horse_no": 1,
                "jockey_target_label_mean_6m": 1.8,
                "trainer_target_label_mean_6m": 1.6,
                "rel_jockey_target_label_mean_z": 0.25,
                "te_window_time": 123,
                "p_win_stack": 0.4,
                "odds_win_final": 2.9,
            },
            {
                "race_id": 202501010101,
                "horse_id": "H002",
                "horse_no": 2,
                "jockey_target_label_mean_6m": 1.4,
                "trainer_target_label_mean_6m": 1.3,
                "rel_jockey_target_label_mean_z": -0.15,
                "te_window_time": 123,
                "p_win_stack": 0.2,
                "odds_win_final": 4.0,
            },
        ]
    )


def test_build_features_v3_te_merges_safe_columns_only() -> None:
    merged, te_cols = build_features_v3_te(_base_frame(), _te_source_frame())

    assert te_cols == [
        "jockey_target_label_mean_6m",
        "trainer_target_label_mean_6m",
        "rel_jockey_target_label_mean_z",
    ]
    assert all(col in merged.columns for col in te_cols)
    assert "te_window_time" not in merged.columns
    assert "p_win_stack" not in merged.columns
    assert list(merged["horse_no"]) == [1, 2]


def test_build_features_v3_te_rejects_unmatched_rows() -> None:
    te_source = _te_source_frame().iloc[[0]].copy()

    try:
        build_features_v3_te(_base_frame(), te_source)
    except ValueError as exc:
        assert "missing 1 rows" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unmatched TE rows")


def test_main_writes_te_output_and_meta(tmp_path: Path) -> None:
    base_path = tmp_path / "features_v3.parquet"
    te_source_path = tmp_path / "features_v2_te.parquet"
    output_path = tmp_path / "features_v3_te.parquet"
    meta_path = tmp_path / "features_v3_te_meta.json"

    _base_frame().to_parquet(base_path, index=False)
    _te_source_frame().to_parquet(te_source_path, index=False)

    exit_code = main(
        [
            "--base-input",
            str(base_path),
            "--te-source-input",
            str(te_source_path),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_path),
        ]
    )

    assert exit_code == 0
    output = pd.read_parquet(output_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert len(output.columns) == len(_base_frame().columns) + 3
    assert meta["input_path"] == str(base_path)
    assert meta["te_source_input_path"] == str(te_source_path)
    assert meta["te_feature_columns"] == [
        "jockey_target_label_mean_6m",
        "trainer_target_label_mean_6m",
        "rel_jockey_target_label_mean_z",
    ]
    assert meta["contains_stacker_timeseries_columns"] is True
    assert meta["coverage"]["jockey_target_label_mean_6m_notna_rate"] == 1.0


def test_build_features_v3_te_meta_payload_records_join_keys() -> None:
    merged, te_cols = build_features_v3_te(_base_frame(), _te_source_frame())
    payload = build_features_v3_te_meta_payload(
        merged,
        base_input_path=Path("data/features_v3.parquet"),
        te_source_input_path=Path("data/features_v2_te.parquet"),
        output_path=Path("data/features_v3_te.parquet"),
        te_feature_columns=te_cols,
    )

    assert payload["te_join_keys"] == ["race_id", "horse_id", "horse_no"]
    assert payload["te_feature_columns"] == te_cols
