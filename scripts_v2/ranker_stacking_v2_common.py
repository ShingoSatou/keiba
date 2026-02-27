from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import ndcg_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v2.train_ranker_v2 import (  # noqa: E402
    _coerce_feature_matrix,
    _group_sizes,
    _rank_within_race,
)

KEY_COLUMNS = ["race_id", "horse_id", "horse_no"]
META_BASE_COLUMNS = [
    "race_id",
    "horse_id",
    "horse_no",
    "t_race",
    "race_date",
    "target_label",
    "field_size",
    "fold_id",
    "valid_year",
]
PERCENTILE_COLUMNS = ["lgbm_percentile", "xgb_percentile", "cat_percentile"]
META_METHOD_CHOICES = ("convex", "ridge", "logreg_multiclass", "lgbm_ranker")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at: {path}")
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_column(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame))


def load_ranker_oof(path: Path, prefix: str) -> pd.DataFrame:
    required = {
        "race_id",
        "horse_id",
        "horse_no",
        "target_label",
        "ranker_score",
        "ranker_rank",
        "ranker_percentile",
        "valid_year",
    }
    frame = pd.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required OOF columns at {path}: {missing}")

    out = pd.DataFrame(
        {
            "race_id": pd.to_numeric(frame["race_id"], errors="coerce").astype("Int64"),
            "horse_id": frame["horse_id"].astype(str),
            "horse_no": pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64"),
            "t_race": pd.to_datetime(_safe_column(frame, "t_race", pd.NaT), errors="coerce"),
            "race_date": pd.to_datetime(_safe_column(frame, "race_date", pd.NaT), errors="coerce"),
            "target_label": pd.to_numeric(frame["target_label"], errors="coerce").astype("Int64"),
            "field_size": pd.to_numeric(_safe_column(frame, "field_size", np.nan), errors="coerce"),
            "fold_id": pd.to_numeric(_safe_column(frame, "fold_id", np.nan), errors="coerce"),
            "valid_year": pd.to_numeric(frame["valid_year"], errors="coerce").astype("Int64"),
            f"{prefix}_score": pd.to_numeric(frame["ranker_score"], errors="coerce"),
            f"{prefix}_rank": pd.to_numeric(frame["ranker_rank"], errors="coerce"),
            f"{prefix}_percentile": pd.to_numeric(frame["ranker_percentile"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["race_id", "horse_no", "target_label", "valid_year"]).copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out["target_label"] = out["target_label"].astype(int)
    out["valid_year"] = out["valid_year"].astype(int)
    if out.empty:
        raise ValueError(f"OOF has no rows after coercion: {path}")
    return out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def merge_ranker_oofs(lgbm_path: Path, xgb_path: Path, cat_path: Path) -> pd.DataFrame:
    lgbm = load_ranker_oof(lgbm_path, "lgbm")
    xgb = load_ranker_oof(xgb_path, "xgb")
    cat = load_ranker_oof(cat_path, "cat")

    merged = lgbm.merge(
        xgb[
            KEY_COLUMNS
            + [
                "target_label",
                "valid_year",
                "fold_id",
                "xgb_score",
                "xgb_rank",
                "xgb_percentile",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("", "_xgb"),
    )
    merged = merged.merge(
        cat[
            KEY_COLUMNS
            + [
                "target_label",
                "valid_year",
                "fold_id",
                "cat_score",
                "cat_rank",
                "cat_percentile",
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("", "_cat"),
    )

    expected_rows = len(lgbm)
    if len(merged) != expected_rows:
        raise ValueError(
            f"OOF alignment mismatch across models: lgbm={len(lgbm)} merged={len(merged)}"
        )

    for column in ("target_label", "valid_year"):
        for suffix in ("_xgb", "_cat"):
            rhs = f"{column}{suffix}"
            if not np.array_equal(merged[column].to_numpy(), merged[rhs].to_numpy()):
                raise ValueError(f"Mismatch on {column} between base models.")
            merged = merged.drop(columns=[rhs])

    if "fold_id_xgb" in merged.columns:
        fold_ref = pd.to_numeric(merged["fold_id"], errors="coerce")
        for suffix in ("_xgb", "_cat"):
            rhs = pd.to_numeric(merged[f"fold_id{suffix}"], errors="coerce")
            if not np.all((fold_ref.isna() & rhs.isna()) | (fold_ref == rhs)):
                raise ValueError("Mismatch on fold_id between base models.")
            merged = merged.drop(columns=[f"fold_id{suffix}"])

    merged = merged.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    return merged


def add_meta_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    for col in PERCENTILE_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Missing percentile column: {col}")

    pct = out[PERCENTILE_COLUMNS].to_numpy(dtype=float)
    out["pct_mean"] = np.mean(pct, axis=1)
    out["pct_std"] = np.std(pct, axis=1)
    out["pct_lgbm_minus_xgb"] = out["lgbm_percentile"] - out["xgb_percentile"]
    out["pct_lgbm_minus_cat"] = out["lgbm_percentile"] - out["cat_percentile"]
    out["pct_xgb_minus_cat"] = out["xgb_percentile"] - out["cat_percentile"]

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
    return out, feature_cols


def build_meta_walkforward_splits(years: list[int]) -> list[tuple[list[int], int]]:
    uniq = sorted({int(y) for y in years})
    if len(uniq) < 2:
        raise ValueError("Need at least two years to build meta walk-forward splits.")
    splits: list[tuple[list[int], int]] = []
    for idx in range(1, len(uniq)):
        splits.append((uniq[:idx], uniq[idx]))
    return splits


def ndcg_at_3(frame: pd.DataFrame, score_col: str) -> float:
    values: list[float] = []
    for _, sub in frame.groupby("race_id", sort=False):
        y_true = sub["target_label"].to_numpy(dtype=float)
        y_score = sub[score_col].to_numpy(dtype=float)
        score = float(ndcg_score([y_true], [y_score], k=3))
        if np.isfinite(score):
            values.append(score)
    if not values:
        return float("nan")
    return float(np.mean(values))


def ndcg_by_year(frame: pd.DataFrame, score_col: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for year, sub in frame.groupby("valid_year", sort=True):
        out[int(year)] = ndcg_at_3(sub, score_col=score_col)
    return out


def add_rank_columns(frame: pd.DataFrame, score_col: str, prefix: str) -> pd.DataFrame:
    work = frame.copy()
    work["ranker_score"] = pd.to_numeric(work[score_col], errors="coerce")
    ranked = _rank_within_race(work)
    ranked[f"{prefix}_rank"] = ranked["ranker_rank"].astype(int)
    ranked[f"{prefix}_percentile"] = ranked["ranker_percentile"].astype(float)
    ranked = ranked.drop(columns=["ranker_score", "ranker_rank", "ranker_percentile"])
    return ranked


def softmax_weights(logits: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    denom = float(np.sum(exp_arr))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid logits for softmax weights.")
    return exp_arr / denom


def predict_convex(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    if len(weights) != 3:
        raise ValueError("convex weights must have length=3")
    pct = frame[PERCENTILE_COLUMNS].to_numpy(dtype=float)
    return np.dot(pct, weights)


def fit_ridge(train_df: pd.DataFrame, feature_cols: list[str], alpha: float) -> Ridge:
    X_train = _coerce_feature_matrix(train_df, feature_cols)
    y_train = train_df["target_label"].astype(float)
    model = Ridge(alpha=float(alpha))
    model.fit(X_train, y_train)
    return model


def fit_logreg_multiclass(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    c_value: float,
    class_weight: str | None,
    max_iter: int,
) -> LogisticRegression | dict[str, float]:
    X_train = _coerce_feature_matrix(train_df, feature_cols)
    y_train = train_df["target_label"].astype(int)
    if int(y_train.nunique()) <= 1:
        return {"constant_expected": float(y_train.iloc[0])}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            max_iter=int(max_iter),
            class_weight=class_weight,
            solver="lbfgs",
        ),
    )
    model.fit(X_train, y_train)
    return model


def predict_logreg_expected(
    model: LogisticRegression | dict[str, float],
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(frame), float(model["constant_expected"]), dtype=float)
    X = _coerce_feature_matrix(frame, feature_cols)
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=float)
    return np.dot(proba, classes)


def fit_lgbm_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    feature_cols: list[str],
    *,
    params: dict[str, Any],
    seed: int,
    early_stopping_rounds: int,
) -> tuple[LGBMRanker, int]:
    train = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    X_train = _coerce_feature_matrix(train, feature_cols)
    y_train = train["target_label"].astype(int)
    g_train = _group_sizes(train)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        colsample_bytree=float(params["feature_fraction"]),
        subsample=float(params["bagging_fraction"]),
        subsample_freq=int(params["bagging_freq"]),
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )
    callbacks: list[Any] = []
    fit_kwargs: dict[str, Any] = {
        "group": g_train,
        "eval_metric": "ndcg",
        "eval_at": [3],
    }
    if valid_df is not None:
        valid = valid_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
            drop=True
        )
        X_valid = _coerce_feature_matrix(valid, feature_cols)
        y_valid = valid["target_label"].astype(int)
        g_valid = _group_sizes(valid)
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
        fit_kwargs.update(
            {
                "eval_set": [(X_valid, y_valid)],
                "eval_group": [g_valid],
                "callbacks": callbacks,
            }
        )
    model.fit(X_train, y_train, **fit_kwargs)
    best_iteration = int(getattr(model, "best_iteration_", 0) or int(params["n_estimators"]))
    if best_iteration <= 0:
        best_iteration = int(params["n_estimators"])
    return model, best_iteration


def predict_lgbm_ranker(
    model: LGBMRanker,
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    best_iteration: int,
) -> np.ndarray:
    X = _coerce_feature_matrix(frame, feature_cols)
    return model.predict(X, num_iteration=int(best_iteration))
