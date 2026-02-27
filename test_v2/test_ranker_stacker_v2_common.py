from __future__ import annotations

import numpy as np
import pandas as pd

from scripts_v2.ranker_stacking_v2_common import (
    add_meta_features,
    build_meta_walkforward_splits,
    fit_logreg_multiclass,
    merge_ranker_oofs,
    predict_convex,
    predict_logreg_expected,
    softmax_weights,
)


def _make_oof(model_shift: float) -> pd.DataFrame:
    rows: list[dict] = []
    for year in (2021, 2022, 2023):
        for race_idx in (1, 2):
            race_id = year * 100 + race_idx
            for horse_no, target_label in ((1, 3), (2, 2), (3, 1)):
                score = float(target_label) + model_shift + (0.01 * horse_no)
                rows.append(
                    {
                        "race_id": race_id,
                        "horse_id": f"H{race_id}_{horse_no}",
                        "horse_no": horse_no,
                        "t_race": pd.Timestamp(f"{year}-01-01"),
                        "race_date": pd.Timestamp(f"{year}-01-01"),
                        "target_label": target_label,
                        "field_size": 3.0,
                        "ranker_score": score,
                        "ranker_rank": horse_no,
                        "ranker_percentile": 1.0 - ((horse_no - 1) / 2.0),
                        "fold_id": year - 2020,
                        "valid_year": year,
                    }
                )
    return pd.DataFrame(rows)


def test_merge_ranker_oofs_aligns_rows(tmp_path):
    lgbm_path = tmp_path / "lgbm.parquet"
    xgb_path = tmp_path / "xgb.parquet"
    cat_path = tmp_path / "cat.parquet"

    _make_oof(0.0).to_parquet(lgbm_path, index=False)
    _make_oof(0.1).to_parquet(xgb_path, index=False)
    _make_oof(-0.2).to_parquet(cat_path, index=False)

    merged = merge_ranker_oofs(lgbm_path, xgb_path, cat_path)
    assert len(merged) == 18
    assert {"lgbm_score", "xgb_score", "cat_score"}.issubset(set(merged.columns))
    assert merged["valid_year"].unique().tolist() == [2021, 2022, 2023]


def test_meta_features_and_convex_prediction():
    base = pd.DataFrame(
        {
            "race_id": [1, 1],
            "horse_id": ["A", "B"],
            "horse_no": [1, 2],
            "target_label": [3, 1],
            "valid_year": [2022, 2022],
            "lgbm_percentile": [0.9, 0.1],
            "xgb_percentile": [0.8, 0.2],
            "cat_percentile": [0.7, 0.3],
        }
    )
    feat, feature_cols = add_meta_features(base)
    assert "pct_mean" in feature_cols
    assert feat.loc[0, "pct_mean"] == np.mean([0.9, 0.8, 0.7])

    weights = softmax_weights([1.0, 0.0, -1.0])
    preds = predict_convex(feat, weights)
    assert preds.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert preds[0] > preds[1]


def test_walkforward_split_definition():
    splits = build_meta_walkforward_splits([2021, 2022, 2023, 2024])
    assert splits == [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]


def test_logreg_multiclass_expected_score_is_finite():
    train = pd.DataFrame(
        {
            "lgbm_percentile": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            "xgb_percentile": [0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
            "cat_percentile": [0.7, 0.3, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6],
            "pct_mean": [0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
            "pct_std": [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
            "pct_lgbm_minus_xgb": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
            "pct_lgbm_minus_cat": [0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2],
            "pct_xgb_minus_cat": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
            "target_label": [3, 0, 2, 1, 3, 0, 2, 1],
        }
    )
    feature_cols = [
        "lgbm_percentile",
        "xgb_percentile",
        "cat_percentile",
        "pct_mean",
        "pct_std",
        "pct_lgbm_minus_xgb",
        "pct_lgbm_minus_cat",
        "pct_xgb_minus_cat",
    ]
    model = fit_logreg_multiclass(
        train,
        feature_cols,
        c_value=1.0,
        class_weight=None,
        max_iter=2000,
    )
    pred = predict_logreg_expected(model, train, feature_cols)
    assert pred.shape == (len(train),)
    assert np.isfinite(pred).all()
