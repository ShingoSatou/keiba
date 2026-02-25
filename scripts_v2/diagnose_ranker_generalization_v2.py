#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v2.train_ranker_v2 import (  # noqa: E402
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HOLDOUT_YEAR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_SEED,
    DEFAULT_TRAIN_WINDOW_YEARS,
    _assert_fold_integrity,
    _base_ranker_params,
    _categorical_features,
    _coerce_feature_matrix,
    _feature_columns,
    _group_sizes,
    _prepare_dataframe,
    build_rolling_year_folds,
)

logger = logging.getLogger(__name__)

NAME_LEAK_PATTERNS = (
    "target",
    "label",
    "finish",
    "result",
    "payout",
    "odds",
    "popularity",
    "final",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose ranker train-valid generalization gap for v2 features."
    )
    parser.add_argument("--input", default="data/features_v2.parquet")
    parser.add_argument("--output", default="data/oof/ranker_generalization_diagnostics.json")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--reg-num-leaves", type=int, default=31)
    parser.add_argument("--reg-min-data-in-leaf", type=int, default=120)
    parser.add_argument("--reg-lambda-l2", type=float, default=10.0)
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument("--shift-top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _bucket_distance(distance_m: float | int | None) -> str:
    if distance_m is None or not np.isfinite(distance_m):
        return "unknown"
    value = int(distance_m)
    if value < 1400:
        return "sprint(<1400m)"
    if value < 1800:
        return "mile(1400-1799m)"
    if value < 2200:
        return "middle(1800-2199m)"
    return "long(>=2200m)"


def _bucket_field_size(field_size: float | int | None) -> str:
    if field_size is None or not np.isfinite(field_size):
        return "unknown"
    value = int(field_size)
    if value <= 10:
        return "small(<=10)"
    if value <= 14:
        return "medium(11-14)"
    return "large(>=15)"


def _safe_ndcg_at3(frame: pd.DataFrame) -> float:
    values: list[float] = []
    for _, sub in frame.groupby("race_id", sort=False):
        if sub.empty:
            continue
        y_true = sub["target_label"].to_numpy(dtype=float)
        y_score = sub["ranker_score"].to_numpy(dtype=float)
        score = float(ndcg_score([y_true], [y_score], k=3))
        if np.isfinite(score):
            values.append(score)
    if not values:
        return float("nan")
    return float(np.mean(values))


def _two_sample_ks_statistic(train_values: np.ndarray, valid_values: np.ndarray) -> float:
    train = np.asarray(train_values, dtype=float)
    valid = np.asarray(valid_values, dtype=float)
    train = train[np.isfinite(train)]
    valid = valid[np.isfinite(valid)]
    if train.size == 0 or valid.size == 0:
        return float("nan")

    train = np.sort(train)
    valid = np.sort(valid)
    support = np.sort(np.unique(np.concatenate([train, valid])))
    train_cdf = np.searchsorted(train, support, side="right") / train.size
    valid_cdf = np.searchsorted(valid, support, side="right") / valid.size
    return float(np.max(np.abs(train_cdf - valid_cdf)))


def _psi_numeric(train_values: np.ndarray, valid_values: np.ndarray, bins: int = 10) -> float:
    train = np.asarray(train_values, dtype=float)
    valid = np.asarray(valid_values, dtype=float)
    train = train[np.isfinite(train)]
    valid = valid[np.isfinite(valid)]
    if train.size == 0 or valid.size == 0:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(train, quantiles))
    if edges.size < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_bins = pd.cut(train, bins=edges, include_lowest=True)
    valid_bins = pd.cut(valid, bins=edges, include_lowest=True)

    train_freq = (
        pd.Series(train_bins)
        .value_counts(sort=False, normalize=True)
        .reindex(train_bins.categories)
    )
    valid_freq = (
        pd.Series(valid_bins)
        .value_counts(sort=False, normalize=True)
        .reindex(train_bins.categories)
    )

    eps = 1e-6
    train_prob = np.clip(train_freq.to_numpy(dtype=float), eps, None)
    valid_prob = np.clip(valid_freq.to_numpy(dtype=float), eps, None)
    return float(np.sum((valid_prob - train_prob) * np.log(valid_prob / train_prob)))


def _feature_shift_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    top_k: int,
) -> dict:
    rows: list[dict] = []
    for col in feature_cols:
        train_values = pd.to_numeric(train_df[col], errors="coerce")
        valid_values = pd.to_numeric(valid_df[col], errors="coerce")

        psi = _psi_numeric(train_values.to_numpy(dtype=float), valid_values.to_numpy(dtype=float))
        ks = _two_sample_ks_statistic(
            train_values.to_numpy(dtype=float),
            valid_values.to_numpy(dtype=float),
        )
        missing_train = float(train_df[col].isna().mean())
        missing_valid = float(valid_df[col].isna().mean())

        rows.append(
            {
                "feature": col,
                "psi": psi,
                "ks": ks,
                "missing_rate_train": missing_train,
                "missing_rate_valid": missing_valid,
                "missing_rate_diff": float(missing_valid - missing_train),
            }
        )

    valid_rows = [r for r in rows if np.isfinite(r["psi"]) and np.isfinite(r["ks"])]
    psi_sorted = sorted(valid_rows, key=lambda x: x["psi"], reverse=True)
    ks_sorted = sorted(valid_rows, key=lambda x: x["ks"], reverse=True)

    return {
        "top_psi": psi_sorted[:top_k],
        "top_ks": ks_sorted[:top_k],
        "mean_psi": float(np.mean([r["psi"] for r in valid_rows])) if valid_rows else float("nan"),
        "mean_ks": float(np.mean([r["ks"] for r in valid_rows])) if valid_rows else float("nan"),
    }


def _leakage_candidate_summary(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    corr_threshold: float,
) -> dict:
    lowered_patterns = tuple(p.lower() for p in NAME_LEAK_PATTERNS)
    name_hits = [
        col for col in feature_cols if any(pattern in col.lower() for pattern in lowered_patterns)
    ]

    high_corr: list[dict] = []
    target = train_df["target_label"].astype(float)
    for col in feature_cols:
        series = pd.to_numeric(train_df[col], errors="coerce")
        pair = pd.DataFrame({"x": series, "y": target}).dropna()
        if len(pair) < 100:
            continue
        if pair["x"].nunique(dropna=True) <= 1 or pair["y"].nunique(dropna=True) <= 1:
            continue
        corr = float(pair["x"].corr(pair["y"], method="spearman"))
        if not np.isfinite(corr):
            continue
        if abs(corr) >= corr_threshold:
            high_corr.append(
                {
                    "feature": col,
                    "spearman_abs": float(abs(corr)),
                    "spearman": corr,
                    "n": int(len(pair)),
                }
            )
    high_corr.sort(key=lambda x: x["spearman_abs"], reverse=True)

    return {
        "suspicious_name_hits": name_hits,
        "high_target_corr_hits": high_corr[:20],
    }


def _prepare_scored_frame(base_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    scored = base_df[
        [
            "race_id",
            "race_date",
            "target_label",
            "going",
            "distance_m",
            "field_size",
        ]
    ].copy()
    scored["ranker_score"] = scores
    scored["distance_bucket"] = scored["distance_m"].map(_bucket_distance)
    scored["field_size_bucket"] = scored["field_size"].map(_bucket_field_size)
    return scored


def _segment_breakdown(valid_scored: pd.DataFrame, *, min_races: int = 30) -> dict:
    outputs: dict[str, list[dict]] = {}
    for key in ("going", "distance_bucket", "field_size_bucket"):
        rows: list[dict] = []
        for value, sub in valid_scored.groupby(key, sort=True):
            races = int(sub["race_id"].nunique())
            if races < min_races:
                continue
            rows.append(
                {
                    "segment": str(value),
                    "races": races,
                    "rows": int(len(sub)),
                    "ndcg_at_3": _safe_ndcg_at3(sub),
                    "positive_rate": float((sub["target_label"] > 0).mean()),
                }
            )
        rows.sort(key=lambda x: x["ndcg_at_3"], reverse=True)
        outputs[key] = rows
    return outputs


def _train_and_score(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    args: argparse.Namespace,
    fold_id: int,
    num_leaves: int,
    min_data_in_leaf: int,
    lambda_l2: float,
    seed_offset: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _assert_fold_integrity(train_df, valid_df, int(valid_df["year"].iloc[0]))

    train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )
    valid_df = valid_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )

    X_train = _coerce_feature_matrix(train_df, feature_cols)
    y_train = train_df["target_label"].astype(int)
    g_train = _group_sizes(train_df)

    X_valid = _coerce_feature_matrix(valid_df, feature_cols)
    y_valid = valid_df["target_label"].astype(int)
    g_valid = _group_sizes(valid_df)

    ranker_params = _base_ranker_params(
        args,
        n_estimators=args.num_boost_round,
        seed=args.seed + seed_offset + fold_id,
    )
    ranker_params["num_leaves"] = int(num_leaves)
    ranker_params["min_child_samples"] = int(min_data_in_leaf)
    ranker_params["reg_lambda"] = float(lambda_l2)
    ranker_params["verbose"] = -1

    model = LGBMRanker(**ranker_params)
    model.fit(
        X_train,
        y_train,
        group=g_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[g_valid],
        eval_at=[3],
        eval_metric="ndcg",
        categorical_feature=categorical_cols or "auto",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )

    best_iteration = int(getattr(model, "best_iteration_", 0) or args.num_boost_round)
    if best_iteration <= 0:
        best_iteration = int(args.num_boost_round)

    train_score = model.predict(X_train, num_iteration=best_iteration)
    valid_score = model.predict(X_valid, num_iteration=best_iteration)

    train_scored = _prepare_scored_frame(train_df, train_score)
    valid_scored = _prepare_scored_frame(valid_df, valid_score)
    train_ndcg = _safe_ndcg_at3(train_scored)
    valid_ndcg = _safe_ndcg_at3(valid_scored)

    metrics = {
        "best_iteration": best_iteration,
        "train_ndcg_at_3": train_ndcg,
        "valid_ndcg_at_3": valid_ndcg,
        "gap_train_minus_valid": float(train_ndcg - valid_ndcg),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "lambda_l2": float(lambda_l2),
    }
    return metrics, train_scored, valid_scored


def _to_finite_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        return _to_finite_float(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if args.num_boost_round <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if args.train_window_years <= 0:
        raise SystemExit("--train-window-years must be > 0")

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    frame = pd.read_parquet(input_path)
    frame = _prepare_dataframe(frame)
    frame = frame[frame["year"] < args.holdout_year].copy()
    if frame.empty:
        raise SystemExit("No trainable data found after holdout exclusion.")

    years = sorted(frame["year"].unique().tolist())
    folds = build_rolling_year_folds(
        years,
        train_window_years=args.train_window_years,
        holdout_year=args.holdout_year,
    )
    feature_cols = _feature_columns(frame)
    categorical_cols = _categorical_features(feature_cols)

    baseline_gaps: list[float] = []
    regularized_gaps: list[float] = []
    regularized_valid_deltas: list[float] = []

    fold_reports: list[dict] = []
    for fold in folds:
        train_df = frame[frame["year"].isin(fold.train_years)].copy()
        valid_df = frame[frame["year"] == fold.valid_year].copy()
        if train_df.empty or valid_df.empty:
            raise SystemExit(
                f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
            )

        logger.info(
            "fold=%s train=%s valid=%s train_rows=%s valid_rows=%s",
            fold.fold_id,
            f"{min(fold.train_years)}-{max(fold.train_years)}",
            fold.valid_year,
            len(train_df),
            len(valid_df),
        )

        baseline_metrics, _, baseline_valid = _train_and_score(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            args=args,
            fold_id=fold.fold_id,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            lambda_l2=args.lambda_l2,
            seed_offset=0,
        )
        regularized_metrics, _, regularized_valid = _train_and_score(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            args=args,
            fold_id=fold.fold_id,
            num_leaves=args.reg_num_leaves,
            min_data_in_leaf=args.reg_min_data_in_leaf,
            lambda_l2=args.reg_lambda_l2,
            seed_offset=1000,
        )

        leakage_candidates = _leakage_candidate_summary(
            train_df,
            feature_cols,
            corr_threshold=args.corr_threshold,
        )
        shift_summary = _feature_shift_summary(
            train_df,
            valid_df,
            feature_cols,
            top_k=args.shift_top_k,
        )

        baseline_segments = _segment_breakdown(baseline_valid)
        regularized_segments = _segment_breakdown(regularized_valid)

        baseline_gap = baseline_metrics["gap_train_minus_valid"]
        regularized_gap = regularized_metrics["gap_train_minus_valid"]
        valid_delta = regularized_metrics["valid_ndcg_at_3"] - baseline_metrics["valid_ndcg_at_3"]

        baseline_gaps.append(baseline_gap)
        regularized_gaps.append(regularized_gap)
        regularized_valid_deltas.append(valid_delta)

        fold_reports.append(
            {
                "fold_id": int(fold.fold_id),
                "train_years": list(fold.train_years),
                "valid_year": int(fold.valid_year),
                "rows": {
                    "train": int(len(train_df)),
                    "valid": int(len(valid_df)),
                    "train_races": int(train_df["race_id"].nunique()),
                    "valid_races": int(valid_df["race_id"].nunique()),
                },
                "baseline": baseline_metrics,
                "regularized": regularized_metrics,
                "delta_regularized_minus_baseline": {
                    "valid_ndcg_at_3": float(valid_delta),
                    "gap_train_minus_valid": float(regularized_gap - baseline_gap),
                },
                "segments": {
                    "baseline": baseline_segments,
                    "regularized": regularized_segments,
                },
                "feature_shift": shift_summary,
                "leakage_candidates": leakage_candidates,
            }
        )

    summary = {
        "folds": int(len(fold_reports)),
        "baseline_gap_mean": float(np.mean(baseline_gaps)),
        "regularized_gap_mean": float(np.mean(regularized_gaps)),
        "regularized_valid_delta_mean": float(np.mean(regularized_valid_deltas)),
        "regularized_valid_delta_min": float(np.min(regularized_valid_deltas)),
        "regularized_valid_delta_max": float(np.max(regularized_valid_deltas)),
    }

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input": str(input_path),
        "config": {
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "seed": int(args.seed),
            "corr_threshold": float(args.corr_threshold),
            "baseline_params": {
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.num_leaves),
                "min_data_in_leaf": int(args.min_data_in_leaf),
                "lambda_l2": float(args.lambda_l2),
            },
            "regularized_params": {
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.reg_num_leaves),
                "min_data_in_leaf": int(args.reg_min_data_in_leaf),
                "lambda_l2": float(args.reg_lambda_l2),
            },
        },
        "data_summary": {
            "rows": int(len(frame)),
            "races": int(frame["race_id"].nunique()),
            "years": years,
            "features": int(len(feature_cols)),
        },
        "summary": summary,
        "folds": fold_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_clean_for_json(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
