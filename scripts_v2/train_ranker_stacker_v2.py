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

from scripts_v2.ranker_stacking_v2_common import (  # noqa: E402
    META_BASE_COLUMNS,
    META_METHOD_CHOICES,
    add_meta_features,
    add_rank_columns,
    build_meta_walkforward_splits,
    fit_lgbm_ranker,
    fit_logreg_multiclass,
    fit_ridge,
    load_json,
    merge_ranker_oofs,
    ndcg_at_3,
    ndcg_by_year,
    predict_convex,
    predict_lgbm_ranker,
    predict_logreg_expected,
    resolve_path,
    save_json,
)
from scripts_v2.train_ranker_v2 import _coerce_feature_matrix  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed stacker meta-model with OOF (v2).")
    parser.add_argument("--best-config", default="data/oof/ranker_stack_optuna_best.json")
    parser.add_argument("--method", choices=list(META_METHOD_CHOICES))
    parser.add_argument("--lgbm-oof", default="data/oof/ranker_oof.parquet")
    parser.add_argument("--xgb-oof", default="data/oof/ranker_xgb_oof.parquet")
    parser.add_argument("--cat-oof", default="data/oof/ranker_cat_oof.parquet")
    parser.add_argument("--train-years", default="")
    parser.add_argument("--oof-output", default="data/oof/ranker_stack_oof.parquet")
    parser.add_argument("--metrics-output", default="data/oof/ranker_stack_cv_metrics.json")
    parser.add_argument("--model-output", default="models/ranker_stack_meta.model")
    parser.add_argument("--meta-output", default="models/ranker_stack_bundle_meta.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _parse_years(raw: str) -> list[int]:
    years = [int(token.strip()) for token in raw.split(",") if token.strip()]
    years = sorted(set(years))
    if not years:
        raise ValueError("No train years specified.")
    return years


def _predict_method(
    method: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    if method == "convex":
        weights = np.asarray(params["weights"], dtype=float)
        preds = predict_convex(valid_df, weights)
        return preds, {"weights": weights.tolist()}, {}

    if method == "ridge":
        model = fit_ridge(train_df, feature_cols, alpha=float(params["alpha"]))
        X_valid = _coerce_feature_matrix(valid_df, feature_cols)
        return model.predict(X_valid), model, {}

    if method == "logreg_multiclass":
        model = fit_logreg_multiclass(
            train_df,
            feature_cols,
            c_value=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=int(params.get("max_iter", 3000)),
        )
        preds = predict_logreg_expected(model, valid_df, feature_cols)
        return preds, model, {}

    if method == "lgbm_ranker":
        model, best_iteration = fit_lgbm_ranker(
            train_df,
            valid_df,
            feature_cols,
            params=params,
            seed=int(seed),
            early_stopping_rounds=int(early_stopping_rounds),
        )
        preds = predict_lgbm_ranker(
            model,
            valid_df,
            feature_cols,
            best_iteration=int(best_iteration),
        )
        return preds, model, {"best_iteration": int(best_iteration)}

    raise ValueError(f"Unknown method: {method}")


def _fit_full_model(
    method: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    early_stopping_rounds: int,
):
    if method == "convex":
        return {"weights": list(map(float, params["weights"]))}, {}

    if method == "ridge":
        return fit_ridge(train_df, feature_cols, alpha=float(params["alpha"])), {}

    if method == "logreg_multiclass":
        model = fit_logreg_multiclass(
            train_df,
            feature_cols,
            c_value=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=int(params.get("max_iter", 3000)),
        )
        return model, {}

    if method == "lgbm_ranker":
        model, best_iteration = fit_lgbm_ranker(
            train_df,
            valid_df=None,
            feature_cols=feature_cols,
            params=params,
            seed=int(seed),
            early_stopping_rounds=int(early_stopping_rounds),
        )
        return model, {"best_iteration": int(best_iteration)}

    raise ValueError(f"Unknown method: {method}")


def _save_meta_model(method: str, model: Any, model_output: Path) -> str:
    model_output.parent.mkdir(parents=True, exist_ok=True)
    if method == "convex":
        save_json(model_output, {"method": "convex", "weights": model["weights"]})
        return "json"
    if method in {"ridge", "logreg_multiclass"}:
        joblib.dump(model, model_output)
        return "joblib"
    if method == "lgbm_ranker":
        model.booster_.save_model(str(model_output))
        return "lightgbm_text"
    raise ValueError(f"Unknown method: {method}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")

    best_config_path = resolve_path(args.best_config)
    if not best_config_path.exists():
        raise SystemExit(f"best-config not found: {best_config_path}")
    best_config = load_json(best_config_path)

    method = args.method or str(best_config.get("selected_method", ""))
    if method not in META_METHOD_CHOICES:
        raise SystemExit(f"Invalid or missing method: {method}")

    methods = best_config.get("methods", {})
    if not isinstance(methods, dict) or method not in methods:
        raise SystemExit(f"Method summary not found in best-config: {method}")
    method_summary = methods[method]
    params = method_summary.get("best_params", {})
    if not isinstance(params, dict):
        raise SystemExit("best_params is missing or invalid in best-config.")

    lgbm_oof = resolve_path(args.lgbm_oof)
    xgb_oof = resolve_path(args.xgb_oof)
    cat_oof = resolve_path(args.cat_oof)
    oof_output = resolve_path(args.oof_output)
    metrics_output = resolve_path(args.metrics_output)
    model_output = resolve_path(args.model_output)
    meta_output = resolve_path(args.meta_output)

    for path in (lgbm_oof, xgb_oof, cat_oof):
        if not path.exists():
            raise SystemExit(f"OOF file not found: {path}")

    merged = merge_ranker_oofs(lgbm_oof, xgb_oof, cat_oof)
    merged, feature_cols = add_meta_features(merged)

    if args.train_years.strip():
        train_years = _parse_years(args.train_years)
    else:
        default_years = best_config.get("config", {}).get("tune_years", [])
        select_year = best_config.get("config", {}).get("select_year")
        train_years = sorted({int(y) for y in default_years if y is not None})
        if select_year is not None:
            train_years.append(int(select_year))
        train_years = sorted(set(train_years))
    if len(train_years) < 2:
        raise SystemExit(f"Need at least two train years. got={train_years}")

    available_years = sorted(merged["valid_year"].unique().tolist())
    missing_years = sorted(set(train_years) - set(available_years))
    if missing_years:
        raise SystemExit(
            "train_years not found in merged OOF: "
            f"missing={missing_years}, available={available_years}"
        )

    frame = merged[merged["valid_year"].isin(train_years)].copy()
    splits = build_meta_walkforward_splits(train_years)
    logger.info("stack method=%s train_years=%s splits=%s", method, train_years, splits)

    fold_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_best_iterations: list[int] = []

    for fold_id, (fold_train_years, valid_year) in enumerate(splits, start=1):
        train_df = frame[frame["valid_year"].isin(fold_train_years)].copy()
        valid_df = frame[frame["valid_year"] == int(valid_year)].copy()
        if train_df.empty or valid_df.empty:
            raise SystemExit(
                f"empty split encountered train_years={fold_train_years} valid_year={valid_year}"
            )

        preds, _, extra = _predict_method(
            method,
            params,
            train_df,
            valid_df,
            feature_cols,
            seed=int(args.seed) + int(fold_id),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        scored = valid_df[
            META_BASE_COLUMNS
            + ["lgbm_score", "xgb_score", "cat_score"]
            + [
                "lgbm_percentile",
                "xgb_percentile",
                "cat_percentile",
            ]
        ].copy()
        scored["stack_score"] = preds
        scored = add_rank_columns(scored, score_col="stack_score", prefix="stack")
        scored["meta_fold_id"] = int(fold_id)
        scored["meta_method"] = method
        fold_frames.append(scored)

        eval_df = scored[["race_id", "target_label", "stack_score"]].rename(
            columns={"stack_score": "score"}
        )
        ndcg3 = ndcg_at_3(eval_df, score_col="score")
        metric = {
            "meta_fold_id": int(fold_id),
            "train_years": list(map(int, fold_train_years)),
            "valid_year": int(valid_year),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_races": int(train_df["race_id"].nunique()),
            "valid_races": int(valid_df["race_id"].nunique()),
            "ndcg_at_3": float(ndcg3),
        }
        if "best_iteration" in extra:
            metric["best_iteration"] = int(extra["best_iteration"])
            fold_best_iterations.append(int(extra["best_iteration"]))
        fold_metrics.append(metric)
        logger.info(
            "meta_fold=%s valid_year=%s ndcg@3=%.6f",
            fold_id,
            valid_year,
            ndcg3,
        )

    oof = pd.concat(fold_frames, axis=0, ignore_index=True)
    oof = oof.sort_values(["race_id", "horse_no"], kind="mergesort")
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_output, index=False)

    ndcg_values = [float(item["ndcg_at_3"]) for item in fold_metrics]
    oof_year_ndcg = ndcg_by_year(
        oof[["valid_year", "race_id", "target_label", "stack_score"]].rename(
            columns={"stack_score": "score"}
        ),
        score_col="score",
    )
    base_year_ndcg = {
        "lgbm": ndcg_by_year(
            oof[["valid_year", "race_id", "target_label", "lgbm_score"]].rename(
                columns={"lgbm_score": "score"}
            ),
            score_col="score",
        ),
        "xgb": ndcg_by_year(
            oof[["valid_year", "race_id", "target_label", "xgb_score"]].rename(
                columns={"xgb_score": "score"}
            ),
            score_col="score",
        ),
        "cat": ndcg_by_year(
            oof[["valid_year", "race_id", "target_label", "cat_score"]].rename(
                columns={"cat_score": "score"}
            ),
            score_col="score",
        ),
    }

    full_model_params = dict(params)
    if method == "lgbm_ranker" and fold_best_iterations:
        full_model_params["n_estimators"] = int(np.median(fold_best_iterations))
        logger.info(
            "full-model n_estimators=%s (median fold best_iteration)",
            int(full_model_params["n_estimators"]),
        )

    full_train_df = frame.copy()
    full_model, full_extra = _fit_full_model(
        method,
        full_model_params,
        full_train_df,
        feature_cols,
        seed=int(args.seed) + 7777,
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    model_format = _save_meta_model(method, full_model, model_output)

    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": method,
        "params": params,
        "full_model_params": full_model_params,
        "feature_columns": feature_cols,
        "train_years": train_years,
        "folds": fold_metrics,
        "summary": {
            "n_folds": int(len(fold_metrics)),
            "ndcg_at_3_mean": float(np.mean(ndcg_values)),
            "ndcg_at_3_std": float(np.std(ndcg_values, ddof=0)),
            "ndcg_at_3_min": float(np.min(ndcg_values)),
            "ndcg_at_3_max": float(np.max(ndcg_values)),
        },
        "oof_ndcg_by_year": oof_year_ndcg,
        "base_ndcg_by_year": base_year_ndcg,
        "best_iteration_median": (
            int(np.median(fold_best_iterations)) if fold_best_iterations else None
        ),
        "full_model_extra": full_extra,
    }
    save_json(metrics_output, metrics)

    meta_payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": method,
        "params": params,
        "full_model_params": full_model_params,
        "feature_columns": feature_cols,
        "train_years": train_years,
        "holdout_rule": "year 2025 is one-shot test only",
        "input_paths": {
            "best_config": str(best_config_path),
            "lgbm_oof": str(lgbm_oof),
            "xgb_oof": str(xgb_oof),
            "cat_oof": str(cat_oof),
        },
        "output_paths": {
            "oof": str(oof_output),
            "metrics": str(metrics_output),
            "model": str(model_output),
        },
        "model_format": model_format,
        "metrics_summary": metrics["summary"],
        "full_model_extra": full_extra,
    }
    save_json(meta_output, meta_payload)

    logger.info("wrote %s", oof_output)
    logger.info("wrote %s", metrics_output)
    logger.info("wrote %s", model_output)
    logger.info("wrote %s", meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
