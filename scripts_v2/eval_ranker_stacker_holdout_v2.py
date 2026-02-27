#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v2.ranker_stacking_v2_common import (  # noqa: E402
    KEY_COLUMNS,
    add_meta_features,
    add_rank_columns,
    load_json,
    ndcg_at_3,
    predict_convex,
    predict_logreg_expected,
    resolve_path,
    save_json,
)
from scripts_v2.train_ranker_v2 import _coerce_feature_matrix, _prepare_dataframe  # noqa: E402

logger = logging.getLogger(__name__)

CORE_COLUMNS = ["race_id", "horse_id", "horse_no", "race_date", "target_label", "field_size"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranker stacker on holdout year (v2).")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--features-base", default="data/features_v2.parquet")
    parser.add_argument("--features-te", default="data/features_v2_te.parquet")
    parser.add_argument("--lgbm-model", default="models/ranker_lgbm.txt")
    parser.add_argument("--lgbm-meta", default="models/ranker_bundle_meta.json")
    parser.add_argument("--xgb-model", default="models/ranker_xgb.json")
    parser.add_argument("--xgb-meta", default="models/ranker_xgb_bundle_meta.json")
    parser.add_argument("--cat-model", default="models/ranker_cat.cbm")
    parser.add_argument("--cat-meta", default="models/ranker_cat_bundle_meta.json")
    parser.add_argument("--stack-model", default="models/ranker_stack_meta.model")
    parser.add_argument("--stack-meta", default="models/ranker_stack_bundle_meta.json")
    parser.add_argument("--preds-output", default="data/holdout/ranker_stack_2025_preds.parquet")
    parser.add_argument("--metrics-output", default="data/holdout/ranker_stack_2025_metrics.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _infer_feature_set(input_path_str: str) -> str:
    value = input_path_str.lower()
    if "features_v2_te" in value:
        return "te"
    return "base"


def _load_year_frame(path: Path, year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _prepare_dataframe(pd.read_parquet(path))
    year_frame = frame[frame["year"] == int(year)].copy()
    return year_frame.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


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


def _predict_stack_score(
    scored: pd.DataFrame,
    *,
    stack_method: str,
    stack_params: dict[str, Any],
    stack_model_path: Path,
    stack_feature_cols: list[str],
) -> np.ndarray:
    if stack_method == "convex":
        weights = np.asarray(stack_params["weights"], dtype=float)
        return predict_convex(scored, weights)
    if stack_method == "ridge":
        model = joblib.load(stack_model_path)
        X = _coerce_feature_matrix(scored, stack_feature_cols)
        return model.predict(X)
    if stack_method == "logreg_multiclass":
        model = joblib.load(stack_model_path)
        return predict_logreg_expected(model, scored, stack_feature_cols)
    if stack_method == "lgbm_ranker":
        booster = lgb.Booster(model_file=str(stack_model_path))
        X = _coerce_feature_matrix(scored, stack_feature_cols)
        return booster.predict(X)
    raise ValueError(f"Unknown stack method: {stack_method}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

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
    stack_model_path = resolve_path(args.stack_model)
    stack_meta_path = resolve_path(args.stack_meta)

    for path in (
        features_base_path,
        features_te_path,
        lgbm_model_path,
        lgbm_meta_path,
        xgb_model_path,
        xgb_meta_path,
        cat_model_path,
        cat_meta_path,
        stack_model_path,
        stack_meta_path,
    ):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

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
    stack_meta = load_json(stack_meta_path)

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

    stack_method = str(stack_meta.get("method", ""))
    stack_params = stack_meta.get("params", {})
    stack_feature_cols = list(map(str, stack_meta.get("feature_columns", [])))
    if not stack_feature_cols:
        raise SystemExit("stack feature_columns missing in stack-meta.")

    scored["stack_score"] = _predict_stack_score(
        scored,
        stack_method=stack_method,
        stack_params=stack_params,
        stack_model_path=stack_model_path,
        stack_feature_cols=stack_feature_cols,
    )
    scored = add_rank_columns(scored, score_col="stack_score", prefix="stack")
    scored["holdout_year"] = int(args.year)
    scored["stack_method"] = stack_method

    eval_stack = scored[["race_id", "target_label", "stack_score"]].rename(
        columns={"stack_score": "score"}
    )
    eval_lgbm = scored[["race_id", "target_label", "lgbm_score"]].rename(
        columns={"lgbm_score": "score"}
    )
    eval_xgb = scored[["race_id", "target_label", "xgb_score"]].rename(
        columns={"xgb_score": "score"}
    )
    eval_cat = scored[["race_id", "target_label", "cat_score"]].rename(
        columns={"cat_score": "score"}
    )

    holdout_metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "holdout_year": int(args.year),
        "rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()),
        "stack_method": stack_method,
        "stack_ndcg_at_3": float(ndcg_at_3(eval_stack, "score")),
        "base_ndcg_at_3": {
            "lgbm": float(ndcg_at_3(eval_lgbm, "score")),
            "xgb": float(ndcg_at_3(eval_xgb, "score")),
            "cat": float(ndcg_at_3(eval_cat, "score")),
        },
        "input_paths": {
            "features_base": str(features_base_path),
            "features_te": str(features_te_path),
            "lgbm_model": str(lgbm_model_path),
            "xgb_model": str(xgb_model_path),
            "cat_model": str(cat_model_path),
            "stack_model": str(stack_model_path),
            "stack_meta": str(stack_meta_path),
        },
    }

    preds_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(preds_output, index=False)
    save_json(metrics_output, holdout_metrics)
    logger.info("wrote %s", preds_output)
    logger.info("wrote %s", metrics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
