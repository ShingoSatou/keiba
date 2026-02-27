from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts_v2.calibration_v2_common import (
    assert_fold_integrity,
    build_calibration_features,
    build_calibration_walkforward_splits,
    fit_logistic_calibrator,
    predict_top3_proba,
)


def _mock_oof_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for year in (2022, 2023, 2024):
        for race_idx in (1, 2):
            race_id = year * 100 + race_idx
            for horse_no, target_label in ((1, 3), (2, 2), (3, 1), (4, 0)):
                score = float(5 - horse_no) + (year - 2022) * 0.1
                rows.append(
                    {
                        "race_id": race_id,
                        "horse_id": f"H{race_id}_{horse_no}",
                        "horse_no": horse_no,
                        "target_label": target_label,
                        "valid_year": year,
                        "field_size": 4.0,
                        "stack_score": score,
                    }
                )
    return pd.DataFrame(rows)


def test_walkforward_split_definition():
    splits = build_calibration_walkforward_splits([2022, 2023, 2024])
    assert splits == [([2022], 2023), ([2022, 2023], 2024)]


def test_calibration_feature_columns_are_present_and_finite():
    frame, feature_cols = build_calibration_features(_mock_oof_frame(), score_col="stack_score")
    assert feature_cols == [
        "percentile_rank",
        "z_score",
        "field_size",
        "score_diff_from_top",
        "gap_1st_to_3rd",
    ]
    assert np.isfinite(frame[feature_cols].to_numpy(dtype=float)).all()
    assert set(frame["is_top3"].unique().tolist()) == {0, 1}


def test_fold_integrity_detects_race_overlap():
    train = pd.DataFrame({"race_id": [1001], "valid_year": [2022]})
    valid = pd.DataFrame({"race_id": [1001], "valid_year": [2023]})
    with pytest.raises(ValueError, match="Race leakage"):
        assert_fold_integrity(train, valid, valid_year=2023)


def test_logreg_predict_probability_in_range():
    frame, feature_cols = build_calibration_features(_mock_oof_frame(), score_col="stack_score")
    train_df = frame[frame["valid_year"].isin([2022, 2023])].copy()
    valid_df = frame[frame["valid_year"] == 2024].copy()
    model = fit_logistic_calibrator(
        train_df,
        feature_cols=feature_cols,
        c_value=1.0,
        class_weight="balanced",
        max_iter=3000,
        seed=42,
    )
    pred = predict_top3_proba(model, valid_df, feature_cols=feature_cols, method="logreg")
    assert pred.shape == (len(valid_df),)
    assert np.isfinite(pred).all()
    assert np.all((pred >= 0.0) & (pred <= 1.0))
