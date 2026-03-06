from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts_v3.train_meta_v3_common import (
    _load_prediction_frame,
    _merge_prediction_features,
    _resolve_cv_splits,
    main,
)


def _sample_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for race_idx, year in enumerate(range(2020, 2026), start=1):
        for horse_no in (1, 2):
            rows.append(
                {
                    "race_id": 1000 + race_idx,
                    "horse_id": f"H{race_idx:02d}{horse_no}",
                    "horse_no": horse_no,
                    "race_date": f"{year}-01-0{horse_no}",
                    "field_size": 2,
                    "y_win": 1 if horse_no == 1 else 0,
                    "y_place": 1,
                }
            )
    return pd.DataFrame(rows)


def _prediction_frame(features: pd.DataFrame, pred_col: str, offset: float) -> pd.DataFrame:
    out = features[["race_id", "horse_no"]].copy()
    out[pred_col] = [
        0.55 + offset if horse_no == 1 else 0.25 + offset for horse_no in out["horse_no"]
    ]
    out["valid_year"] = pd.to_datetime(features["race_date"]).dt.year.astype(int)
    return out


def test_meta_join_uses_race_id_and_horse_no(tmp_path: Path) -> None:
    features = _sample_features().iloc[:4].copy()
    pred_paths: dict[str, Path] = {}
    for pred_col, offset in (
        ("p_win_lgbm", 0.00),
        ("p_win_xgb", 0.01),
        ("p_win_cat", 0.02),
    ):
        path = tmp_path / f"{pred_col}.parquet"
        _prediction_frame(features, pred_col, offset).to_parquet(path, index=False)
        pred_paths[pred_col] = path

    merged = _merge_prediction_features(features, pred_paths=pred_paths)
    assert list(merged.columns).count("p_win_lgbm") == 1
    assert merged["p_win_lgbm"].notna().all()
    assert merged["p_win_xgb"].notna().all()
    assert merged["p_win_cat"].notna().all()


def test_meta_join_duplicate_keys_raise(tmp_path: Path) -> None:
    dup = pd.DataFrame(
        {
            "race_id": [1, 1],
            "horse_no": [1, 1],
            "p_win_lgbm": [0.1, 0.2],
            "valid_year": [2024, 2024],
        }
    )
    path = tmp_path / "dup.parquet"
    dup.to_parquet(path, index=False)
    with pytest.raises(SystemExit):
        _load_prediction_frame(path, "p_win_lgbm")


def test_meta_grouped_cv_keeps_same_race_in_single_fold() -> None:
    frame = _sample_features().iloc[:10].copy()
    splits, payload = _resolve_cv_splits(
        frame,
        label_col="y_win",
        n_splits=5,
        cv_strategy="auto",
        seed=42,
    )
    assert payload["n_splits"] == 5
    for train_idx, valid_idx in splits:
        train_races = set(frame.iloc[train_idx]["race_id"].unique())
        valid_races = set(frame.iloc[valid_idx]["race_id"].unique())
        assert train_races.isdisjoint(valid_races)


def test_meta_grouped_cv_falls_back_to_group_kfold(monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn import model_selection

    class BrokenStratifiedGroupKFold:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def split(self, *args, **kwargs):
            raise ValueError("force fallback")

    monkeypatch.setattr(model_selection, "StratifiedGroupKFold", BrokenStratifiedGroupKFold)
    frame = _sample_features().iloc[:10].copy()
    _, payload = _resolve_cv_splits(
        frame,
        label_col="y_win",
        n_splits=5,
        cv_strategy="auto",
        seed=42,
    )
    assert payload["used"] == "group_kfold"


def test_meta_main_writes_reference_flags(tmp_path: Path) -> None:
    train_features = _sample_features().iloc[:10].copy()
    holdout_features = _sample_features().iloc[10:].copy()
    features_path = tmp_path / "features.parquet"
    holdout_path = tmp_path / "holdout.parquet"
    train_features.to_parquet(features_path, index=False)
    holdout_features.to_parquet(holdout_path, index=False)

    oof_paths: list[Path] = []
    holdout_pred_paths: list[Path] = []
    for pred_col, offset in (
        ("p_win_lgbm", 0.00),
        ("p_win_xgb", 0.01),
        ("p_win_cat", 0.02),
    ):
        oof_path = tmp_path / f"{pred_col}_oof.parquet"
        holdout_pred_path = tmp_path / f"{pred_col}_holdout.parquet"
        _prediction_frame(train_features, pred_col, offset).to_parquet(oof_path, index=False)
        _prediction_frame(holdout_features, pred_col, offset).drop(
            columns=["valid_year"]
        ).to_parquet(
            holdout_pred_path,
            index=False,
        )
        oof_paths.append(oof_path)
        holdout_pred_paths.append(holdout_pred_path)

    oof_output = tmp_path / "win_meta_oof.parquet"
    holdout_output = tmp_path / "win_meta_holdout.parquet"
    metrics_output = tmp_path / "win_meta_metrics.json"
    model_output = tmp_path / "win_meta_v3.pkl"
    meta_output = tmp_path / "win_meta_bundle_meta_v3.json"

    rc = main(
        [
            "--task",
            "win",
            "--features-input",
            str(features_path),
            "--holdout-input",
            str(holdout_path),
            "--lgbm-oof",
            str(oof_paths[0]),
            "--xgb-oof",
            str(oof_paths[1]),
            "--cat-oof",
            str(oof_paths[2]),
            "--lgbm-holdout",
            str(holdout_pred_paths[0]),
            "--xgb-holdout",
            str(holdout_pred_paths[1]),
            "--cat-holdout",
            str(holdout_pred_paths[2]),
            "--oof-output",
            str(oof_output),
            "--holdout-output",
            str(holdout_output),
            "--metrics-output",
            str(metrics_output),
            "--model-output",
            str(model_output),
            "--meta-output",
            str(meta_output),
        ]
    )

    assert rc == 0
    assert oof_output.exists()
    assert holdout_output.exists()
    metrics = json.loads(metrics_output.read_text(encoding="utf-8"))
    meta = json.loads(meta_output.read_text(encoding="utf-8"))
    assert metrics["cv_is_temporal"] is False
    assert metrics["group_key"] == "race_id"
    assert metrics["meta_oof_is_strict_temporal"] is False
    assert metrics["meta_metrics_are_reference_only"] is True
    assert meta["cv_is_temporal"] is False
    assert meta["group_key"] == "race_id"
    assert meta["meta_oof_is_strict_temporal"] is False
    assert meta["meta_metrics_are_reference_only"] is True
