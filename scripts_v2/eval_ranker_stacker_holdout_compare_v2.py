#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v2.ranker_stacking_v2_common import (  # noqa: E402
    KEY_COLUMNS,
    META_METHOD_CHOICES,
    add_meta_features,
    add_rank_columns,
    fit_lgbm_ranker,
    fit_logreg_multiclass,
    fit_ridge,
    load_json,
    merge_ranker_oofs,
    ndcg_at_3,
    predict_convex,
    predict_lgbm_ranker,
    predict_logreg_expected,
    resolve_path,
    save_json,
)
from scripts_v2.train_ranker_v2 import _coerce_feature_matrix, _prepare_dataframe  # noqa: E402

logger = logging.getLogger(__name__)

CORE_COLUMNS = ["race_id", "horse_id", "horse_no", "race_date", "target_label", "field_size"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stacker methods on a holdout year (one-shot, v2)."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--features-base", default="data/features_v2_2025.parquet")
    parser.add_argument("--features-te", default="data/features_v2_te_2025.parquet")

    parser.add_argument("--lgbm-model", default="models/ranker_lgbm.txt")
    parser.add_argument("--lgbm-meta", default="models/ranker_bundle_meta.json")
    parser.add_argument("--xgb-model", default="models/ranker_xgb.json")
    parser.add_argument("--xgb-meta", default="models/ranker_xgb_bundle_meta.json")
    parser.add_argument("--cat-model", default="models/ranker_cat.cbm")
    parser.add_argument("--cat-meta", default="models/ranker_cat_bundle_meta.json")

    parser.add_argument("--best-config", default="data/oof/ranker_stack_optuna_best.json")
    parser.add_argument("--lgbm-oof", default="data/oof/ranker_oof.parquet")
    parser.add_argument("--xgb-oof", default="data/oof/ranker_xgb_oof.parquet")
    parser.add_argument("--cat-oof", default="data/oof/ranker_cat_oof.parquet")
    parser.add_argument("--train-years", default="2021,2022,2023,2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)

    parser.add_argument(
        "--preds-output", default="data/holdout/ranker_stack_2025_compare_preds.parquet"
    )
    parser.add_argument(
        "--metrics-output",
        default="data/holdout/ranker_stack_2025_compare_metrics.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _parse_years(raw: str) -> list[int]:
    years = [int(token.strip()) for token in raw.split(",") if token.strip()]
    years = sorted(set(years))
    if not years:
        raise ValueError("No years parsed.")
    return years


def _infer_feature_set(input_path_str: str) -> str:
    value = input_path_str.lower()
    if "features_v2_te" in value:
        return "te"
    return "base"


def _load_year_frame(path: Path, year: int) -> pd.DataFrame:
    frame = _prepare_dataframe(pd.read_parquet(path))
    year_frame = frame[frame["year"] == int(year)].copy()
    return year_frame.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _prepare_core_frame(
    base_frame: pd.DataFrame, te_frame: pd.DataFrame, *, year: int
) -> pd.DataFrame:
    base_key = base_frame[KEY_COLUMNS].copy()
    te_key = te_frame[KEY_COLUMNS].copy()
    merged = base_key.merge(te_key, on=KEY_COLUMNS, how="inner")
    if len(merged) != len(base_frame) or len(merged) != len(te_frame):
        raise ValueError(
            "base and te feature rows are misaligned "
            f"base={len(base_frame)} te={len(te_frame)} inner={len(merged)}"
        )
    core = base_frame[CORE_COLUMNS].copy()
    core["t_race"] = pd.to_datetime(core["race_date"], errors="coerce")
    core["fold_id"] = np.nan
    core["valid_year"] = int(year)
    return core


def _predict_lgbm(model_path: Path, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_path))
    X = _coerce_feature_matrix(frame, feature_cols)
    return booster.predict(X)


def _predict_xgb(model_path: Path, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    from xgboost import XGBRanker

    model = XGBRanker()
    model.load_model(str(model_path))
    X = _coerce_feature_matrix(frame, feature_cols)
    return model.predict(X)


def _predict_cat(model_path: Path, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    from catboost import CatBoostRanker, Pool

    model = CatBoostRanker()
    model.load_model(str(model_path))
    X = _coerce_feature_matrix(frame, feature_cols)
    pool = Pool(X, group_id=frame["race_id"].to_numpy())
    return model.predict(pool)


def _select_model_frame(
    *,
    base_frame: pd.DataFrame,
    te_frame: pd.DataFrame,
    model_meta: dict[str, Any],
) -> pd.DataFrame:
    feature_set = _infer_feature_set(str(model_meta.get("input_path", "")))
    if feature_set == "te":
        return te_frame
    return base_frame


def _predict_base_models(
    core_frame: pd.DataFrame,
    *,
    base_frame: pd.DataFrame,
    te_frame: pd.DataFrame,
    lgbm_model_path: Path,
    lgbm_meta: dict[str, Any],
    xgb_model_path: Path,
    xgb_meta: dict[str, Any],
    cat_model_path: Path,
    cat_meta: dict[str, Any],
) -> pd.DataFrame:
    out = core_frame.copy()
    lgbm_frame = _select_model_frame(base_frame=base_frame, te_frame=te_frame, model_meta=lgbm_meta)
    xgb_frame = _select_model_frame(base_frame=base_frame, te_frame=te_frame, model_meta=xgb_meta)
    cat_frame = _select_model_frame(base_frame=base_frame, te_frame=te_frame, model_meta=cat_meta)

    lgbm_features = list(map(str, lgbm_meta.get("feature_columns", [])))
    xgb_features = list(map(str, xgb_meta.get("feature_columns", [])))
    cat_features = list(map(str, cat_meta.get("feature_columns", [])))
    if not lgbm_features or not xgb_features or not cat_features:
        raise ValueError("feature_columns missing in one of the base model meta files.")

    out["lgbm_score"] = _predict_lgbm(lgbm_model_path, lgbm_frame, lgbm_features)
    out["xgb_score"] = _predict_xgb(xgb_model_path, xgb_frame, xgb_features)
    out["cat_score"] = _predict_cat(cat_model_path, cat_frame, cat_features)

    out = add_rank_columns(out, score_col="lgbm_score", prefix="lgbm")
    out = add_rank_columns(out, score_col="xgb_score", prefix="xgb")
    out = add_rank_columns(out, score_col="cat_score", prefix="cat")
    return out


def _ndcg_from_scores(frame: pd.DataFrame, score_col: str) -> float:
    eval_df = frame[["race_id", "target_label", score_col]].rename(columns={score_col: "score"})
    return float(ndcg_at_3(eval_df, "score"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")

    preds_output = resolve_path(args.preds_output)
    metrics_output = resolve_path(args.metrics_output)
    if metrics_output.exists() and not args.force:
        raise SystemExit(
            "holdout metrics file already exists. "
            "One-shot rule blocks overwrite. Use --force only when intentionally rerunning."
        )

    features_base_path = resolve_path(args.features_base)
    features_te_path = resolve_path(args.features_te)
    lgbm_model_path = resolve_path(args.lgbm_model)
    lgbm_meta_path = resolve_path(args.lgbm_meta)
    xgb_model_path = resolve_path(args.xgb_model)
    xgb_meta_path = resolve_path(args.xgb_meta)
    cat_model_path = resolve_path(args.cat_model)
    cat_meta_path = resolve_path(args.cat_meta)
    best_config_path = resolve_path(args.best_config)
    lgbm_oof_path = resolve_path(args.lgbm_oof)
    xgb_oof_path = resolve_path(args.xgb_oof)
    cat_oof_path = resolve_path(args.cat_oof)

    for path in (
        features_base_path,
        features_te_path,
        lgbm_model_path,
        lgbm_meta_path,
        xgb_model_path,
        xgb_meta_path,
        cat_model_path,
        cat_meta_path,
        best_config_path,
        lgbm_oof_path,
        xgb_oof_path,
        cat_oof_path,
    ):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

    train_years = _parse_years(args.train_years)

    base_year = _load_year_frame(features_base_path, int(args.year))
    te_year = _load_year_frame(features_te_path, int(args.year))
    if base_year.empty or te_year.empty:
        raise SystemExit(
            f"No feature rows for holdout year={args.year}. "
            "Generate feature parquet including that year before holdout evaluation."
        )

    core_frame = _prepare_core_frame(base_year, te_year, year=int(args.year))
    logger.info(
        "holdout rows=%s races=%s year=%s",
        len(core_frame),
        int(core_frame["race_id"].nunique()),
        args.year,
    )

    lgbm_meta = load_json(lgbm_meta_path)
    xgb_meta = load_json(xgb_meta_path)
    cat_meta = load_json(cat_meta_path)

    scored = _predict_base_models(
        core_frame,
        base_frame=base_year,
        te_frame=te_year,
        lgbm_model_path=lgbm_model_path,
        lgbm_meta=lgbm_meta,
        xgb_model_path=xgb_model_path,
        xgb_meta=xgb_meta,
        cat_model_path=cat_model_path,
        cat_meta=cat_meta,
    )
    scored, _ = add_meta_features(scored)

    base_ndcg = {
        "lgbm": _ndcg_from_scores(scored, "lgbm_score"),
        "xgb": _ndcg_from_scores(scored, "xgb_score"),
        "cat": _ndcg_from_scores(scored, "cat_score"),
    }

    best_config = load_json(best_config_path)
    method_summaries = best_config.get("methods", {})
    if not isinstance(method_summaries, dict):
        raise SystemExit("best-config missing 'methods' dictionary.")

    train_merged = merge_ranker_oofs(lgbm_oof_path, xgb_oof_path, cat_oof_path)
    train_merged, feature_cols = add_meta_features(train_merged)
    train_df = train_merged[train_merged["valid_year"].isin(train_years)].copy()
    if train_df.empty:
        available = sorted(train_merged["valid_year"].unique().tolist())
        raise SystemExit(f"No meta train rows for train_years={train_years}. available={available}")

    method_scores: dict[str, float] = {}
    method_params: dict[str, Any] = {}

    for method in META_METHOD_CHOICES:
        summary = method_summaries.get(method)
        if not isinstance(summary, dict):
            raise SystemExit(f"best-config missing summary for method={method}")
        params = summary.get("best_params", {})
        if not isinstance(params, dict):
            raise SystemExit(f"best-config missing best_params for method={method}")
        method_params[method] = params

        if method == "convex":
            weights = np.asarray(params.get("weights", []), dtype=float)
            if weights.shape != (3,):
                raise SystemExit(f"invalid convex weights in best-config: {weights}")
            scored[f"stack_{method}_score"] = predict_convex(scored, weights)

        elif method == "ridge":
            model = fit_ridge(train_df, feature_cols, alpha=float(params["alpha"]))
            X_holdout = _coerce_feature_matrix(scored, feature_cols)
            scored[f"stack_{method}_score"] = model.predict(X_holdout)

        elif method == "logreg_multiclass":
            model = fit_logreg_multiclass(
                train_df,
                feature_cols,
                c_value=float(params["C"]),
                class_weight=params.get("class_weight"),
                max_iter=int(params.get("max_iter", 3000)),
            )
            scored[f"stack_{method}_score"] = predict_logreg_expected(model, scored, feature_cols)

        elif method == "lgbm_ranker":
            model, best_iter = fit_lgbm_ranker(
                train_df,
                valid_df=None,
                feature_cols=feature_cols,
                params=params,
                seed=int(args.seed) + 9999,
                early_stopping_rounds=int(args.early_stopping_rounds),
            )
            scored[f"stack_{method}_score"] = predict_lgbm_ranker(
                model,
                scored,
                feature_cols,
                best_iteration=int(best_iter),
            )

        else:
            raise SystemExit(f"unknown method: {method}")

        method_scores[method] = _ndcg_from_scores(scored, f"stack_{method}_score")

    best_method, best_score = max(method_scores.items(), key=lambda item: float(item[1]))

    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "holdout_year": int(args.year),
        "rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()),
        "meta_train_years": train_years,
        "base_ndcg_at_3": base_ndcg,
        "stack_ndcg_at_3": method_scores,
        "best_stacking_method": best_method,
        "best_stacking_ndcg_at_3": float(best_score),
        "stacking_params": method_params,
        "input_paths": {
            "features_base": str(features_base_path),
            "features_te": str(features_te_path),
            "lgbm_model": str(lgbm_model_path),
            "xgb_model": str(xgb_model_path),
            "cat_model": str(cat_model_path),
            "best_config": str(best_config_path),
            "lgbm_oof": str(lgbm_oof_path),
            "xgb_oof": str(xgb_oof_path),
            "cat_oof": str(cat_oof_path),
        },
    }

    preds_output.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "race_id",
        "horse_id",
        "horse_no",
        "race_date",
        "target_label",
        "field_size",
        "lgbm_score",
        "xgb_score",
        "cat_score",
    ] + [f"stack_{method}_score" for method in META_METHOD_CHOICES]
    scored[keep_cols].to_parquet(preds_output, index=False)
    save_json(metrics_output, metrics)
    logger.info("wrote %s", preds_output)
    logger.info("wrote %s", metrics_output)
    logger.info("best_method=%s ndcg@3=%.6f", best_method, float(best_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
