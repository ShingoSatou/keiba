#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    build_calibration_features,
    compute_binary_metrics,
    predict_top3_proba,
    resolve_path,
    save_json,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Top3 calibrator on holdout year (one-shot, v2)."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--input", default="data/holdout/ranker_stack_2025_compare_preds.parquet")
    parser.add_argument("--model", default="models/calibrator.pkl")
    parser.add_argument("--stack-meta", default="models/ranker_stack_bundle_meta.json")
    parser.add_argument(
        "--score-col",
        default="",
        help="Optional holdout score column. If empty, inferred from --stack-meta method.",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--preds-output", default="data/holdout/top3_calibrated_2025.parquet")
    parser.add_argument(
        "--metrics-output",
        default="data/holdout/calibration_2025_metrics.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at: {path}")
    return obj


def _check_overwrite(outputs: list[Path], *, force: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(
            "holdout output already exists. one-shot rule blocks overwrite. "
            f"use --force only when intentionally rerunning. existing={joined}"
        )


def _infer_score_col(args: argparse.Namespace, holdout_df: pd.DataFrame) -> str:
    if args.score_col.strip():
        return args.score_col.strip()
    stack_meta_path = resolve_path(args.stack_meta)
    if not stack_meta_path.exists():
        raise SystemExit(
            "--score-col is empty and stack-meta file not found: "
            f"{stack_meta_path}"
        )
    stack_meta = _load_json(stack_meta_path)
    method = str(stack_meta.get("method", "")).strip()
    if not method:
        raise SystemExit(f"method is missing in stack-meta: {stack_meta_path}")
    inferred = f"stack_{method}_score"
    if inferred not in holdout_df.columns:
        raise SystemExit(
            f"Inferred score column not found: {inferred}. "
            f"Available stack columns={[c for c in holdout_df.columns if c.startswith('stack_')]}"
        )
    return inferred


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

    input_path = resolve_path(args.input)
    model_path = resolve_path(args.model)
    preds_output = resolve_path(args.preds_output)
    metrics_output = resolve_path(args.metrics_output)

    for path in (input_path, model_path):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

    _check_overwrite([preds_output, metrics_output], force=args.force)

    holdout_df = pd.read_parquet(input_path)
    score_col = _infer_score_col(args, holdout_df)
    if score_col not in holdout_df.columns:
        raise SystemExit(f"score column not found: {score_col}")

    if "race_date" in holdout_df.columns:
        race_date = pd.to_datetime(holdout_df["race_date"], errors="coerce")
        holdout_df = holdout_df[race_date.dt.year == int(args.year)].copy()
    if holdout_df.empty:
        raise SystemExit(f"No rows found for holdout year={int(args.year)}")
    holdout_df["valid_year"] = int(args.year)

    model_payload = joblib.load(model_path)
    if not isinstance(model_payload, dict):
        raise SystemExit(f"Invalid calibrator payload at: {model_path}")
    calibrator = model_payload.get("model")
    method = str(model_payload.get("method", "")).strip()
    if method not in {"logreg", "isotonic"}:
        raise SystemExit(f"Unsupported calibrator method: {method}")
    feature_cols = model_payload.get("feature_columns")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise SystemExit("feature_columns missing in calibrator payload.")
    clip_eps = float(model_payload.get("clip_eps", 1e-6))

    featured, _ = build_calibration_features(holdout_df, score_col=score_col)
    featured["valid_year"] = pd.to_numeric(featured["valid_year"], errors="coerce").fillna(
        int(args.year)
    )
    featured["valid_year"] = featured["valid_year"].astype(int)

    raw_prob = predict_top3_proba(
        calibrator,
        featured,
        feature_cols=feature_cols,
        method=method,
    )
    p_top3 = np.clip(raw_prob, clip_eps, 1.0 - clip_eps)

    preds = featured[
        ["race_id", "horse_id", "horse_no", "target_label", "field_size", score_col]
    ].copy()
    if "race_date" in featured.columns:
        preds["race_date"] = featured["race_date"]
    preds = preds.rename(columns={score_col: "input_score"})
    preds["valid_year"] = int(args.year)
    preds["p_top3"] = p_top3
    preds = preds[
        [
            "race_id",
            "horse_id",
            "horse_no",
            "race_date",
            "valid_year",
            "p_top3",
            "input_score",
            "target_label",
            "field_size",
        ]
    ]
    preds = preds.sort_values(["race_id", "horse_no"], kind="mergesort")
    preds_output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(preds_output, index=False)

    y_true = (preds["target_label"].astype(int) > 0).astype(int).to_numpy(dtype=int)
    summary = compute_binary_metrics(
        y_true,
        preds["p_top3"].to_numpy(dtype=float),
        n_bins=args.n_bins,
    )

    by_field_size: dict[str, dict[str, Any]] = {}
    for size, group in preds.groupby("field_size", sort=True):
        key = _format_field_size_key(size)
        y_group = (group["target_label"].astype(int) > 0).astype(int).to_numpy(dtype=int)
        by_field_size[key] = compute_binary_metrics(
            y_group,
            group["p_top3"].to_numpy(dtype=float),
            n_bins=args.n_bins,
        )
        by_field_size[key]["n_races"] = int(group["race_id"].nunique())

    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "year": int(args.year),
        "input_path": str(input_path),
        "calibrator_model_path": str(model_path),
        "calibrator_method": method,
        "score_col": score_col,
        "n_bins": int(args.n_bins),
        "n_rows": int(len(preds)),
        "n_races": int(preds["race_id"].nunique()),
        "summary": summary,
        "metrics_by_field_size": by_field_size,
        "outputs": {
            "preds": str(preds_output),
            "metrics": str(metrics_output),
        },
    }
    save_json(metrics_output, metrics)

    logger.info(
        "year=%s races=%s rows=%s logloss=%s brier=%.6f ece=%.6f",
        args.year,
        int(preds["race_id"].nunique()),
        int(len(preds)),
        (f"{summary['logloss']:.6f}" if summary["logloss"] is not None else "None"),
        float(summary["brier"]),
        float(summary["ece"]),
    )
    logger.info("wrote %s", preds_output)
    logger.info("wrote %s", metrics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
