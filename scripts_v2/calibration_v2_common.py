from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_FEATURE_COLUMNS = [
    "percentile_rank",
    "z_score",
    "field_size",
    "score_diff_from_top",
    "gap_1st_to_3rd",
]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_calibration_walkforward_splits(years: list[int]) -> list[tuple[list[int], int]]:
    uniq = sorted({int(year) for year in years})
    if len(uniq) < 2:
        raise ValueError("Need at least two years to build calibration walk-forward splits.")
    splits: list[tuple[list[int], int]] = []
    for idx in range(1, len(uniq)):
        splits.append((uniq[:idx], uniq[idx]))
    return splits


def assert_fold_integrity(train_df: pd.DataFrame, valid_df: pd.DataFrame, valid_year: int) -> None:
    if train_df.empty or valid_df.empty:
        raise ValueError("Train/valid frame is empty.")
    train_max_year = int(train_df["valid_year"].max())
    if train_max_year >= int(valid_year):
        raise ValueError(
            "Temporal leakage detected: "
            f"train max year={train_max_year}, valid year={int(valid_year)}"
        )
    train_races = set(train_df["race_id"].unique().tolist())
    valid_races = set(valid_df["race_id"].unique().tolist())
    overlap = train_races & valid_races
    if overlap:
        raise ValueError(f"Race leakage detected across train/valid: {len(overlap)} races")


def build_calibration_features(
    frame: pd.DataFrame,
    *,
    score_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    required = {"race_id", "horse_id", "horse_no", "target_label", "valid_year"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if score_col not in frame.columns:
        raise ValueError(f"Score column not found: {score_col}")

    out = frame.copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out["target_label"] = pd.to_numeric(out["target_label"], errors="coerce").astype("Int64")
    out["valid_year"] = pd.to_numeric(out["valid_year"], errors="coerce").astype("Int64")
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna(subset=["race_id", "horse_no", "target_label", "valid_year", score_col]).copy()
    if out.empty:
        raise ValueError("No valid rows after coercion.")

    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out["target_label"] = out["target_label"].astype(int)
    out["valid_year"] = out["valid_year"].astype(int)
    out["horse_id"] = out["horse_id"].astype(str)

    race_size = out.groupby("race_id", sort=False)["race_id"].transform("size").astype(float)
    if "field_size" in out.columns:
        out["field_size"] = pd.to_numeric(out["field_size"], errors="coerce")
        out["field_size"] = out["field_size"].fillna(race_size)
    else:
        out["field_size"] = race_size

    out = out.sort_values(
        ["race_id", score_col, "horse_no"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    out["rank_within_race"] = out.groupby("race_id", sort=False).cumcount() + 1
    race_size = out.groupby("race_id", sort=False)["race_id"].transform("size").astype(float)
    out["percentile_rank"] = np.where(
        race_size <= 1.0,
        1.0,
        1.0 - (out["rank_within_race"].astype(float) - 1.0) / (race_size - 1.0),
    )

    race_mean = out.groupby("race_id", sort=False)[score_col].transform("mean")
    race_std = out.groupby("race_id", sort=False)[score_col].transform("std").replace(0.0, np.nan)
    out["z_score"] = ((out[score_col] - race_mean) / race_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    top_score = out.groupby("race_id", sort=False)[score_col].transform("max")
    out["score_diff_from_top"] = top_score - out[score_col]

    gap_map = (
        out.groupby("race_id", sort=False)[score_col]
        .apply(_gap_1st_to_3rd)
        .to_dict()
    )
    out["gap_1st_to_3rd"] = out["race_id"].map(gap_map).astype(float)
    out["is_top3"] = (out["target_label"] > 0).astype(int)

    for column in CALIBRATION_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    out = out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    return out, list(CALIBRATION_FEATURE_COLUMNS)


def _gap_1st_to_3rd(scores: pd.Series) -> float:
    values = np.sort(scores.to_numpy(dtype=float))[::-1]
    if values.size < 3:
        return 0.0
    return float(values[0] - values[2])


def fit_logistic_calibrator(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    c_value: float,
    class_weight: str | None,
    max_iter: int,
    seed: int,
) -> Pipeline | dict[str, float]:
    y_train = train_df["is_top3"].astype(int)
    if int(y_train.nunique()) <= 1:
        return {"constant_prob": float(y_train.iloc[0])}
    x_train = train_df[feature_cols].copy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight=class_weight,
            max_iter=int(max_iter),
            solver="lbfgs",
            random_state=int(seed),
        ),
    )
    model.fit(x_train, y_train)
    return model


def fit_isotonic_calibrator(train_df: pd.DataFrame) -> IsotonicRegression | dict[str, float]:
    y_train = train_df["is_top3"].astype(int)
    if int(y_train.nunique()) <= 1:
        return {"constant_prob": float(y_train.iloc[0])}
    x_train = pd.to_numeric(train_df["percentile_rank"], errors="coerce").to_numpy(dtype=float)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(x_train, y_train.to_numpy(dtype=float))
    return model


def predict_top3_proba(
    model: Pipeline | IsotonicRegression | dict[str, float],
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    method: str,
) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(frame), float(model["constant_prob"]), dtype=float)

    if method == "logreg":
        x = frame[feature_cols].copy()
        return model.predict_proba(x)[:, 1]
    if method == "isotonic":
        x = pd.to_numeric(frame["percentile_rank"], errors="coerce").to_numpy(dtype=float)
        return model.predict(x)
    raise ValueError(f"Unknown method: {method}")


def compute_binary_metrics(
    y_true: np.ndarray | pd.Series,
    p_pred: np.ndarray | pd.Series,
    *,
    n_bins: int,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and p_pred length mismatch.")
    if y.shape[0] == 0:
        raise ValueError("Cannot compute metrics for empty input.")

    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    try:
        logloss_value: float | None = float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        logloss_value = None

    brier_value = float(brier_score_loss(y, p))
    reliability, ece_value = _reliability_bins(y, p, n_bins=n_bins)
    return {
        "n_rows": int(y.shape[0]),
        "base_rate": float(np.mean(y)),
        "logloss": logloss_value,
        "brier": brier_value,
        "ece": ece_value,
        "reliability": reliability,
    }


def _reliability_bins(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_bins: int,
) -> tuple[list[dict[str, Any]], float]:
    if n_bins <= 1:
        raise ValueError("n_bins must be > 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_pred, edges[1:-1], right=False)
    n_total = float(len(p_pred))
    bins: list[dict[str, Any]] = []
    ece_value = 0.0

    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_id": idx,
                    "left": float(edges[idx]),
                    "right": float(edges[idx + 1]),
                    "count": 0,
                    "mean_pred": None,
                    "frac_pos": None,
                }
            )
            continue
        mean_pred = float(np.mean(p_pred[mask]))
        frac_pos = float(np.mean(y_true[mask]))
        ece_value += abs(mean_pred - frac_pos) * (count / n_total)
        bins.append(
            {
                "bin_id": idx,
                "left": float(edges[idx]),
                "right": float(edges[idx + 1]),
                "count": count,
                "mean_pred": mean_pred,
                "frac_pos": frac_pos,
            }
        )
    return bins, float(ece_value)
