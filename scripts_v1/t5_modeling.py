from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scripts.predict import FEATURE_COLS as FUNDAMENTAL_FEATURES

MARKET_SERIES_MINUTES = tuple(range(60, 0, -5))
MARKET_POINT_COLUMNS = [f"M_odds_tminus_{minute}" for minute in MARKET_SERIES_MINUTES]
MARKET_DERIVED_FEATURES = [
    "M_odds_win_t5",
    "M_imp_prob_t5",
    "M_log_odds_t5",
    "M_odds_rank_t5",
    "M_odds_missing_flag",
    "M_odds_series_missing_points",
    "M_odds_slope_log",
    "M_odds_volatility_log",
    "M_odds_min",
    "M_odds_max",
    "M_odds_last_change",
    "M_odds_jump_flag",
    "M_win_pool_total_t5",
    "M_win_pool_total_tminus_60",
    "M_win_pool_growth_60to5",
    "M_win_pool_growth_ratio_60to5",
    "M_market_entropy_t5",
]
MARKET_FEATURES = [*MARKET_POINT_COLUMNS, *MARKET_DERIVED_FEATURES]
MODEL_M_FEATURES = [*FUNDAMENTAL_FEATURES, *MARKET_FEATURES]
MODEL_O_FEATURES = [
    "M_odds_win_t5",
    "M_log_odds_t5",
    "M_imp_prob_t5",
    "M_odds_rank_t5",
    "M_odds_slope_log",
    "M_odds_volatility_log",
    "M_odds_last_change",
    "M_win_pool_total_t5",
    "M_win_pool_growth_60to5",
    "M_win_pool_growth_ratio_60to5",
    "M_market_entropy_t5",
    *MARKET_POINT_COLUMNS,
]
META_FEATURES = [
    "pF_raw",
    "pM_raw",
    "o_close_hat",
    "log_o_close_hat",
    "M_odds_win_t5",
    "M_log_odds_t5",
]
BUNDLE_VERSION = "t5_bundle_v1"


@dataclass
class BettingStats:
    bets: int
    races: int
    bet_ratio: float
    hit_rate: float
    roi: float
    total_profit: float
    max_drawdown: float


def coerce_model_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    matrix = df.reindex(columns=feature_names).copy()
    for column in matrix.columns:
        if pd.api.types.is_numeric_dtype(matrix[column]):
            continue
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def predict_proba(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    return np.asarray(model.predict(matrix), dtype=float)


def clamp_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)


def binary_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"logloss": None, "auc": None, "brier": None}
    clipped = clamp_probs(prob)
    metrics: dict[str, float | None] = {
        "logloss": float(log_loss(y_true, clipped)),
        "brier": float(np.mean((clipped - y_true.to_numpy(dtype=float)) ** 2)),
    }
    if y_true.nunique() >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, clipped))
    else:
        metrics["auc"] = None
    return metrics


def select_one_per_race(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    keys = ["race_id"]
    if "asof_ts" in df.columns:
        keys.append("asof_ts")
    ranked = df.sort_values(
        keys + [score_col, "horse_no"],
        ascending=[True] * len(keys) + [False, True],
    )
    return ranked.groupby(keys, as_index=False, sort=False).head(1).reset_index(drop=True)


def evaluate_betting(
    df: pd.DataFrame,
    threshold: float,
    score_col: str = "EV_hat",
    is_win_col: str = "is_win",
    odds_col: str = "odds_win_final",
) -> BettingStats:
    top = select_one_per_race(df, score_col=score_col)
    if top.empty:
        return BettingStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    races = int(top["race_id"].nunique())
    selected = top[top[score_col] >= threshold].copy()
    bets = int(len(selected))
    if bets == 0:
        return BettingStats(0, races, 0.0, 0.0, 0.0, 0.0, 0.0)

    selected["profit"] = np.where(
        selected[is_win_col].astype(int) == 1,
        pd.to_numeric(selected[odds_col], errors="coerce") - 1.0,
        -1.0,
    )
    selected["profit"] = pd.to_numeric(selected["profit"], errors="coerce").fillna(-1.0)
    total_profit = float(selected["profit"].sum())
    hit_rate = float((selected[is_win_col].astype(int) == 1).mean())
    roi = float(total_profit / bets)
    cumulative = selected["profit"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    return BettingStats(
        bets=bets,
        races=races,
        bet_ratio=float(bets / races) if races > 0 else 0.0,
        hit_rate=hit_rate,
        roi=roi,
        total_profit=total_profit,
        max_drawdown=max_drawdown,
    )


def choose_tau(
    calib_df: pd.DataFrame,
    min_bet_ratio: float = 0.10,
    score_col: str = "EV_hat",
) -> dict[str, Any]:
    if calib_df.empty:
        return {
            "tau": 0.0,
            "selected_by": "fallback_empty",
            "min_bet_ratio": min_bet_ratio,
            "candidates_checked": 0,
            "stats": BettingStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0).__dict__,
        }

    top = select_one_per_race(calib_df, score_col=score_col)
    scores = pd.to_numeric(top[score_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(scores) == 0:
        empty_stats = BettingStats(
            0,
            int(top["race_id"].nunique()),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        return {
            "tau": 0.0,
            "selected_by": "fallback_no_scores",
            "min_bet_ratio": min_bet_ratio,
            "candidates_checked": 0,
            "stats": empty_stats.__dict__,
        }

    quantiles = np.linspace(0.05, 0.95, 91)
    taus = sorted({float(np.quantile(scores, q)) for q in quantiles})

    eligible: list[tuple[float, BettingStats]] = []
    all_stats: list[tuple[float, BettingStats]] = []
    for tau in taus:
        stats = evaluate_betting(top, threshold=tau, score_col=score_col)
        all_stats.append((tau, stats))
        if stats.bet_ratio >= min_bet_ratio:
            eligible.append((tau, stats))

    if eligible:
        best_tau, best_stats = max(
            eligible,
            key=lambda item: (item[1].roi, item[1].total_profit, -abs(item[0])),
        )
        return {
            "tau": float(best_tau),
            "selected_by": "roi_with_min_ratio",
            "min_bet_ratio": float(min_bet_ratio),
            "candidates_checked": len(taus),
            "stats": best_stats.__dict__,
        }

    best_tau, best_stats = max(
        all_stats,
        key=lambda item: (item[1].bet_ratio, item[1].roi, item[1].total_profit),
    )
    return {
        "tau": float(best_tau),
        "selected_by": "fallback_max_ratio",
        "min_bet_ratio": float(min_bet_ratio),
        "candidates_checked": len(taus),
        "stats": best_stats.__dict__,
    }


def apply_tau_rule(
    df: pd.DataFrame,
    tau: float,
    score_col: str = "EV_hat",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["buy_flag"] = []
        return out
    top = select_one_per_race(out, score_col=score_col)
    top = top[[c for c in ["race_id", "asof_ts", "horse_no", score_col] if c in top.columns]]
    top["buy_flag"] = (pd.to_numeric(top[score_col], errors="coerce") >= float(tau)).astype(int)
    keys = ["race_id", "horse_no"]
    if "asof_ts" in top.columns and "asof_ts" in out.columns:
        keys.append("asof_ts")
    out = out.merge(top[keys + ["buy_flag"]], on=keys, how="left")
    out["buy_flag"] = out["buy_flag"].fillna(0).astype(int)
    return out


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_bundle_payload(
    model_f: Any,
    model_m: Any,
    model_o: Any,
    meta_model: Any,
    tau_info: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    f_features = _resolve_feature_names(model_f, list(FUNDAMENTAL_FEATURES))
    m_features = _resolve_feature_names(model_m, list(MODEL_M_FEATURES))
    o_features = _resolve_feature_names(model_o, list(MODEL_O_FEATURES))
    return {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now().isoformat(),
        "models": {
            "F": model_f,
            "M": model_m,
            "O": model_o,
            "META": meta_model,
        },
        "feature_names": {
            "F": f_features,
            "M": m_features,
            "O": o_features,
            "META": list(META_FEATURES),
        },
        "tau": float(tau_info["tau"]),
        "tau_info": tau_info,
        "metadata": metadata,
    }


def dump_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_feature_names(model: Any, fallback: list[str]) -> list[str]:
    candidate = getattr(model, "feature_name_", None)
    if candidate is None:
        candidate = getattr(model, "feature_names", None)
    if candidate is None:
        return fallback
    return list(candidate)


def safe_log(value: float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except Exception:
        return math.nan
    if not math.isfinite(numeric) or numeric <= 0:
        return math.nan
    return math.log(numeric)
