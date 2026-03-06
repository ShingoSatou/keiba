#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.train_binary_v3_common import (
    DEFAULT_CV_WINDOW_POLICY,
    DEFAULT_TRAIN_WINDOW_YEARS,
    attach_cv_policy_columns,
    build_cv_policy_payload,
    build_fixed_window_year_folds,
    compute_binary_metrics,
    hash_files,
    make_window_definition,
    resolve_path,
    save_json,
    select_recent_window_years,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_SEED = 42
METHOD_CHOICES = ("logreg", "isotonic")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train odds->win probability calibrators with fixed-length sliding yearly OOF (v3)."
        )
    )
    parser.add_argument("--input", default="data/features_v3.parquet")
    parser.add_argument("--score-cols", default="p_win_odds_t10_norm,p_win_odds_final_norm")
    parser.add_argument("--methods", default="logreg,isotonic")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        help="The v3 standard comparison condition is fixed_sliding with 4 years.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--clip-eps", type=float, default=1e-6)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--oof-output", default="data/oof/odds_win_calibration_oof.parquet")
    parser.add_argument("--metrics-output", default="data/oof/odds_win_calibration_cv_metrics.json")
    parser.add_argument("--model-output", default="models/odds_win_calibrators_v3.pkl")
    parser.add_argument("--meta-output", default="models/odds_win_calibrators_v3_meta.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _fit_calibrator(method: str, x: np.ndarray, y: np.ndarray, *, seed: int):
    if method == "logreg":
        if int(np.unique(y).shape[0]) <= 1:
            return {"constant_prob": float(y[0])}
        model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=int(seed))
        model.fit(x.reshape(-1, 1), y)
        return model

    if method == "isotonic":
        if int(np.unique(y).shape[0]) <= 1:
            return {"constant_prob": float(y[0])}
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(x, y)
        return model

    raise ValueError(f"Unknown method: {method}")


def _predict_calibrator(model, method: str, x: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(x), float(model["constant_prob"]), dtype=float)
    if method == "logreg":
        return model.predict_proba(x.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return model.predict(x)
    raise ValueError(f"Unknown method: {method}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not (0.0 < float(args.clip_eps) < 0.5):
        raise SystemExit("--clip-eps must be in (0, 0.5)")
    if int(args.n_bins) <= 1:
        raise SystemExit("--n-bins must be > 1")

    input_path = resolve_path(args.input)
    oof_output = resolve_path(args.oof_output)
    metrics_output = resolve_path(args.metrics_output)
    model_output = resolve_path(args.model_output)
    meta_output = resolve_path(args.meta_output)

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    score_cols = _parse_csv(args.score_cols)
    methods = _parse_csv(args.methods)
    if not score_cols:
        raise SystemExit("--score-cols must not be empty")
    for method in methods:
        if method not in METHOD_CHOICES:
            raise SystemExit(f"Unsupported method: {method}")

    frame = pd.read_parquet(input_path)
    required = {"race_id", "horse_id", "horse_no", "race_date", "y_win"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}")

    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame = frame[frame["race_date"].notna()].copy()
    frame["year"] = frame["race_date"].dt.year.astype(int)
    frame["race_id"] = pd.to_numeric(frame["race_id"], errors="coerce").astype("Int64")
    frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64")
    frame["y_win"] = pd.to_numeric(frame["y_win"], errors="coerce").fillna(0).astype(int)
    frame = frame[frame["race_id"].notna() & frame["horse_no"].notna()].copy()
    frame["race_id"] = frame["race_id"].astype(int)
    frame["horse_no"] = frame["horse_no"].astype(int)

    for col in score_cols:
        if col not in frame.columns:
            raise SystemExit(f"Score column not found: {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    train_frame = frame[frame["year"] < int(args.holdout_year)].copy()
    if train_frame.empty:
        raise SystemExit("No train rows after holdout exclusion")

    years = sorted(train_frame["year"].unique().tolist())
    folds = build_fixed_window_year_folds(
        years,
        window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        max_window = max(1, len(years) - 1)
        raise SystemExit(
            "No odds calibration CV folds available under the fixed_sliding policy "
            f"(available_years={years}, try --train-window-years <= {max_window})"
        )
    recent_years = select_recent_window_years(
        years,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    recent_train_frame = train_frame[train_frame["year"].isin(recent_years)].copy()

    oof = train_frame[["race_id", "horse_id", "horse_no", "race_date", "year", "y_win"]].copy()
    oof = oof.rename(columns={"year": "valid_year"})
    oof = attach_cv_policy_columns(
        oof,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
        cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        window_definition=make_window_definition(int(args.train_window_years)),
    )

    fold_metrics: list[dict[str, Any]] = []
    variant_models: dict[str, Any] = {}
    variant_cols: list[str] = []

    for score_col in score_cols:
        for method in methods:
            pred_col = f"{score_col}_cal_{method}"
            variant_cols.append(pred_col)
            oof[pred_col] = np.nan

            for fold in folds:
                train_df = train_frame[train_frame["year"].isin(fold.train_years)].copy()
                valid_df = train_frame[train_frame["year"] == fold.valid_year].copy()
                if train_df.empty or valid_df.empty:
                    continue

                train_sub = train_df[train_df[score_col].notna()].copy()
                valid_sub = valid_df[valid_df[score_col].notna()].copy()

                if train_sub.empty or valid_sub.empty:
                    fold_metrics.append(
                        {
                            "score_col": score_col,
                            "method": method,
                            "fold_id": int(fold.fold_id),
                            "valid_year": int(fold.valid_year),
                            "train_years": list(map(int, fold.train_years)),
                            "train_rows": int(len(train_sub)),
                            "valid_rows": int(len(valid_sub)),
                            "logloss": None,
                            "brier": None,
                            "auc": None,
                            "ece": None,
                            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
                            "train_window_years": int(args.train_window_years),
                            "holdout_year": int(args.holdout_year),
                            "window_definition": make_window_definition(
                                int(args.train_window_years)
                            ),
                        }
                    )
                    continue

                model = _fit_calibrator(
                    method,
                    train_sub[score_col].to_numpy(dtype=float),
                    train_sub["y_win"].to_numpy(dtype=int),
                    seed=int(args.seed) + int(fold.fold_id),
                )
                pred = _predict_calibrator(
                    model,
                    method,
                    valid_sub[score_col].to_numpy(dtype=float),
                )
                pred = np.clip(pred, float(args.clip_eps), 1.0 - float(args.clip_eps))

                valid_idx = valid_sub.index
                oof.loc[valid_idx, pred_col] = pred

                metrics = compute_binary_metrics(
                    valid_sub["y_win"].to_numpy(dtype=int),
                    pred,
                    n_bins=int(args.n_bins),
                )
                fold_metrics.append(
                    {
                        "score_col": score_col,
                        "method": method,
                        "fold_id": int(fold.fold_id),
                        "valid_year": int(fold.valid_year),
                        "train_years": list(map(int, fold.train_years)),
                        "train_rows": int(len(train_sub)),
                        "valid_rows": int(len(valid_sub)),
                        "logloss": metrics["logloss"],
                        "brier": metrics["brier"],
                        "auc": metrics["auc"],
                        "ece": metrics["ece"],
                        "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
                        "train_window_years": int(args.train_window_years),
                        "holdout_year": int(args.holdout_year),
                        "window_definition": make_window_definition(int(args.train_window_years)),
                        "reliability": metrics["reliability"],
                    }
                )

            full_sub = recent_train_frame[recent_train_frame[score_col].notna()].copy()
            if full_sub.empty:
                continue
            full_model = _fit_calibrator(
                method,
                full_sub[score_col].to_numpy(dtype=float),
                full_sub["y_win"].to_numpy(dtype=int),
                seed=int(args.seed) + 999,
            )
            variant_models[pred_col] = {
                "score_col": score_col,
                "method": method,
                "model": full_model,
            }

    oof = oof.sort_values(["race_id", "horse_no"], kind="mergesort")
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_output, index=False)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "score_cols": score_cols,
            "methods": methods,
            "clip_eps": float(args.clip_eps),
            "cv_policy": build_cv_policy_payload(
                folds,
                train_window_years=int(args.train_window_years),
                holdout_year=int(args.holdout_year),
                cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            ),
            "final_model_train_years": recent_years,
            "models": variant_models,
        },
        model_output,
    )

    summary_by_variant: dict[str, dict[str, float | None]] = {}
    for pred_col in variant_cols:
        variant_rows = [
            m for m in fold_metrics if f"{m['score_col']}_cal_{m['method']}" == pred_col
        ]
        for metric_name in ("logloss", "brier", "auc", "ece"):
            values = [m.get(metric_name) for m in variant_rows]
            vals = [float(v) for v in values if v is not None and np.isfinite(v)]
            if pred_col not in summary_by_variant:
                summary_by_variant[pred_col] = {}
            summary_by_variant[pred_col][metric_name] = float(np.mean(vals)) if vals else None

    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_path": str(input_path),
        "oof_path": str(oof_output),
        "cv_policy": build_cv_policy_payload(
            folds,
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        ),
        "holdout_year": int(args.holdout_year),
        "train_window_years": int(args.train_window_years),
        "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
        "score_cols": score_cols,
        "methods": methods,
        "clip_eps": float(args.clip_eps),
        "n_bins": int(args.n_bins),
        "data_summary": {
            "rows": int(len(train_frame)),
            "races": int(train_frame["race_id"].nunique()),
            "years": years,
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
        },
        "folds": fold_metrics,
        "summary_by_variant": summary_by_variant,
    }
    save_json(metrics_output, metrics_payload)

    meta_payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_path": str(input_path),
        "cv_policy": build_cv_policy_payload(
            folds,
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        ),
        "output_paths": {
            "oof": str(oof_output),
            "metrics": str(metrics_output),
            "model": str(model_output),
        },
        "score_cols": score_cols,
        "methods": methods,
        "model_columns": variant_cols,
        "final_model_train_years": recent_years,
        "code_hash": hash_files([Path(__file__)]),
    }
    save_json(meta_output, meta_payload)

    logger.info("wrote %s", oof_output)
    logger.info("wrote %s", metrics_output)
    logger.info("wrote %s", model_output)
    logger.info("wrote %s", meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
