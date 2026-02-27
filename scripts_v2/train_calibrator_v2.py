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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v2.calibration_v2_common import (  # noqa: E402
    assert_fold_integrity,
    build_calibration_features,
    build_calibration_walkforward_splits,
    compute_binary_metrics,
    fit_isotonic_calibrator,
    fit_logistic_calibrator,
    predict_top3_proba,
    resolve_path,
    save_json,
)

logger = logging.getLogger(__name__)

METHOD_CHOICES = ("logreg", "isotonic")
DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_SEED = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Top3 calibrator with walk-forward OOF (v2)."
    )
    parser.add_argument("--input", default="data/oof/ranker_stack_oof.parquet")
    parser.add_argument("--score-col", default="stack_score")
    parser.add_argument("--train-years", default="")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)

    parser.add_argument("--method", choices=list(METHOD_CHOICES), default="logreg")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="none")
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--clip-eps", type=float, default=1e-6)
    parser.add_argument("--n-bins", type=int, default=10)

    parser.add_argument("--oof-output", default="data/oof/top3_oof.parquet")
    parser.add_argument("--metrics-output", default="data/oof/calibration_metrics.json")
    parser.add_argument("--model-output", default="models/calibrator.pkl")
    parser.add_argument("--meta-output", default="models/calibrator_bundle_meta.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _parse_years(raw: str) -> list[int]:
    years = [int(token.strip()) for token in raw.split(",") if token.strip()]
    years = sorted(set(years))
    if not years:
        raise ValueError("No years parsed from --train-years")
    return years


def _resolve_class_weight(raw: str) -> str | None:
    if raw == "none":
        return None
    return raw


def _check_overwrite(outputs: list[Path], *, force: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists. pass --force to overwrite: {joined}")


def _fit_calibrator(
    method: str,
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    c_value: float,
    class_weight: str | None,
    max_iter: int,
    seed: int,
):
    if method == "logreg":
        return fit_logistic_calibrator(
            train_df,
            feature_cols=feature_cols,
            c_value=float(c_value),
            class_weight=class_weight,
            max_iter=int(max_iter),
            seed=int(seed),
        )
    if method == "isotonic":
        return fit_isotonic_calibrator(train_df)
    raise ValueError(f"Unknown method: {method}")


def _format_field_size_key(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.n_bins <= 1:
        raise SystemExit("--n-bins must be > 1")
    if not (0.0 < args.clip_eps < 0.5):
        raise SystemExit("--clip-eps must be in (0, 0.5)")
    if args.c_value <= 0:
        raise SystemExit("--c-value must be > 0")
    if args.max_iter <= 0:
        raise SystemExit("--max-iter must be > 0")

    input_path = resolve_path(args.input)
    oof_output = resolve_path(args.oof_output)
    metrics_output = resolve_path(args.metrics_output)
    model_output = resolve_path(args.model_output)
    meta_output = resolve_path(args.meta_output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    _check_overwrite([oof_output, metrics_output, model_output, meta_output], force=args.force)

    frame_raw = pd.read_parquet(input_path)
    frame_feat, feature_cols = build_calibration_features(frame_raw, score_col=args.score_col)
    frame_feat = frame_feat[frame_feat["valid_year"] < int(args.holdout_year)].copy()
    if frame_feat.empty:
        raise SystemExit(
            f"No rows remain after holdout-year filter (valid_year < {int(args.holdout_year)})."
        )

    available_years = sorted(frame_feat["valid_year"].unique().tolist())
    if args.train_years.strip():
        train_years = _parse_years(args.train_years)
        missing = sorted(set(train_years) - set(available_years))
        if missing:
            raise SystemExit(
                "train_years not found in input OOF: "
                f"missing={missing}, available={available_years}"
            )
    else:
        train_years = available_years

    if len(train_years) < 2:
        raise SystemExit(f"Need at least two years for walk-forward calibration. got={train_years}")

    frame = frame_feat[frame_feat["valid_year"].isin(train_years)].copy()
    if frame.empty:
        raise SystemExit("No rows for selected train_years.")

    splits = build_calibration_walkforward_splits(train_years)
    class_weight = _resolve_class_weight(args.class_weight)
    logger.info(
        "calibration method=%s score_col=%s years=%s splits=%s",
        args.method,
        args.score_col,
        train_years,
        splits,
    )

    fold_outputs: list[pd.DataFrame] = []
    fold_eval_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for calib_fold_id, (fold_train_years, valid_year) in enumerate(splits, start=1):
        train_df = frame[frame["valid_year"].isin(fold_train_years)].copy()
        valid_df = frame[frame["valid_year"] == int(valid_year)].copy()
        assert_fold_integrity(train_df, valid_df, int(valid_year))

        model = _fit_calibrator(
            args.method,
            train_df=train_df,
            feature_cols=feature_cols,
            c_value=float(args.c_value),
            class_weight=class_weight,
            max_iter=int(args.max_iter),
            seed=int(args.seed) + int(calib_fold_id),
        )
        raw_prob = predict_top3_proba(
            model,
            valid_df,
            feature_cols=feature_cols,
            method=args.method,
        )
        p_top3 = np.clip(raw_prob, args.clip_eps, 1.0 - args.clip_eps)

        pred_df = valid_df[
            ["race_id", "horse_id", "horse_no", "valid_year", "target_label", args.score_col]
        ].copy()
        pred_df = pred_df.rename(columns={args.score_col: "input_score"})
        pred_df["p_top3"] = p_top3
        pred_df = pred_df[
            [
                "race_id",
                "horse_id",
                "horse_no",
                "valid_year",
                "p_top3",
                "input_score",
                "target_label",
            ]
        ]
        fold_outputs.append(pred_df)

        fold_eval = valid_df[["valid_year", "race_id", "field_size", "is_top3"]].copy()
        fold_eval["p_top3"] = p_top3
        fold_eval["calib_fold_id"] = int(calib_fold_id)
        fold_eval_frames.append(fold_eval)

        fold_metric_values = compute_binary_metrics(
            valid_df["is_top3"].to_numpy(dtype=int),
            p_top3,
            n_bins=int(args.n_bins),
        )
        fold_metrics.append(
            {
                "calib_fold_id": int(calib_fold_id),
                "train_years": list(map(int, fold_train_years)),
                "valid_year": int(valid_year),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_races": int(train_df["race_id"].nunique()),
                "valid_races": int(valid_df["race_id"].nunique()),
                "logloss": fold_metric_values["logloss"],
                "brier": fold_metric_values["brier"],
                "ece": fold_metric_values["ece"],
                "base_rate": fold_metric_values["base_rate"],
                "reliability": fold_metric_values["reliability"],
            }
        )
        logger.info(
            "fold=%s valid_year=%s logloss=%s brier=%.6f ece=%.6f",
            calib_fold_id,
            valid_year,
            (
                f"{fold_metric_values['logloss']:.6f}"
                if fold_metric_values["logloss"] is not None
                else "None"
            ),
            float(fold_metric_values["brier"]),
            float(fold_metric_values["ece"]),
        )

    oof = pd.concat(fold_outputs, axis=0, ignore_index=True)
    oof = oof.sort_values(["race_id", "horse_no"], kind="mergesort")
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_output, index=False)

    eval_all = pd.concat(fold_eval_frames, axis=0, ignore_index=True)
    summary_metric_values = compute_binary_metrics(
        eval_all["is_top3"].to_numpy(dtype=int),
        eval_all["p_top3"].to_numpy(dtype=float),
        n_bins=int(args.n_bins),
    )

    by_year: dict[str, dict[str, Any]] = {}
    for year, group in eval_all.groupby("valid_year", sort=True):
        by_year[str(int(year))] = compute_binary_metrics(
            group["is_top3"].to_numpy(dtype=int),
            group["p_top3"].to_numpy(dtype=float),
            n_bins=int(args.n_bins),
        )

    by_field_size: dict[str, dict[str, Any]] = {}
    for size, group in eval_all.groupby("field_size", sort=True):
        key = _format_field_size_key(size)
        by_field_size[key] = compute_binary_metrics(
            group["is_top3"].to_numpy(dtype=int),
            group["p_top3"].to_numpy(dtype=float),
            n_bins=int(args.n_bins),
        )
        by_field_size[key]["n_races"] = int(group["race_id"].nunique())

    full_model = _fit_calibrator(
        args.method,
        train_df=frame,
        feature_cols=feature_cols,
        c_value=float(args.c_value),
        class_weight=class_weight,
        max_iter=int(args.max_iter),
        seed=int(args.seed) + 9999,
    )
    model_payload = {
        "method": args.method,
        "score_col": args.score_col,
        "feature_columns": feature_cols,
        "train_years": train_years,
        "holdout_year": int(args.holdout_year),
        "clip_eps": float(args.clip_eps),
        "class_weight": class_weight,
        "c_value": float(args.c_value),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "model": full_model,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, model_output)

    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": args.method,
        "score_col": args.score_col,
        "input_path": str(input_path),
        "feature_columns": feature_cols,
        "holdout_year": int(args.holdout_year),
        "train_years": train_years,
        "available_years": available_years,
        "oof_valid_years": sorted(oof["valid_year"].unique().tolist()),
        "n_bins": int(args.n_bins),
        "clip_eps": float(args.clip_eps),
        "class_weight": class_weight,
        "c_value": float(args.c_value),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "folds": fold_metrics,
        "summary": {
            "n_folds": int(len(fold_metrics)),
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
            "logloss": summary_metric_values["logloss"],
            "brier": summary_metric_values["brier"],
            "ece": summary_metric_values["ece"],
            "base_rate": summary_metric_values["base_rate"],
            "reliability": summary_metric_values["reliability"],
        },
        "metrics_by_year": by_year,
        "metrics_by_field_size": by_field_size,
    }
    save_json(metrics_output, metrics_payload)

    meta_payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": args.method,
        "score_col": args.score_col,
        "feature_columns": feature_cols,
        "holdout_year": int(args.holdout_year),
        "train_years": train_years,
        "oof_valid_years": sorted(oof["valid_year"].unique().tolist()),
        "input": {
            "path": str(input_path),
            "input_rows": int(len(frame_raw)),
            "eligible_rows": int(len(frame)),
            "eligible_races": int(frame["race_id"].nunique()),
        },
        "outputs": {
            "oof": str(oof_output),
            "metrics": str(metrics_output),
            "model": str(model_output),
        },
        "params": {
            "class_weight": class_weight,
            "c_value": float(args.c_value),
            "max_iter": int(args.max_iter),
            "seed": int(args.seed),
            "clip_eps": float(args.clip_eps),
            "n_bins": int(args.n_bins),
        },
        "folds": [
            {
                "calib_fold_id": int(item["calib_fold_id"]),
                "train_years": item["train_years"],
                "valid_year": int(item["valid_year"]),
                "train_rows": int(item["train_rows"]),
                "valid_rows": int(item["valid_rows"]),
                "train_races": int(item["train_races"]),
                "valid_races": int(item["valid_races"]),
            }
            for item in fold_metrics
        ],
    }
    save_json(meta_output, meta_payload)

    logger.info("wrote %s", oof_output)
    logger.info("wrote %s", metrics_output)
    logger.info("wrote %s", model_output)
    logger.info("wrote %s", meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
