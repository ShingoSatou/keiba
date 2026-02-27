#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v2.train_ranker_v2 import (  # noqa: E402
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HOLDOUT_YEAR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_SEED,
    DEFAULT_TRAIN_WINDOW_YEARS,
    FoldSpec,
    _assert_fold_integrity,
    _coerce_feature_matrix,
    _feature_columns,
    _fold_ndcg_at_3,
    _hash_files,
    _prepare_dataframe,
    _rank_within_race,
    _save_json,
    build_rolling_year_folds,
)

logger = logging.getLogger(__name__)

ENTITY_ID_FEATURES = {"jockey_key", "trainer_key"}
LOSS_FUNCTION_CHOICES = ("YetiRank", "YetiRankPairwise")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 ranker (CatBoost).")
    parser.add_argument(
        "--params-json",
        help="Optional JSON file to set default input/drop flags and CatBoost params. "
        "Explicit CLI flags take precedence.",
    )
    parser.add_argument("--input", default="data/features_v2.parquet")
    parser.add_argument("--oof-output", default="data/oof/ranker_cat_oof.parquet")
    parser.add_argument("--metrics-output", default="data/oof/ranker_cat_cv_metrics.json")
    parser.add_argument("--model-output", default="models/ranker_cat.cbm")
    parser.add_argument("--all-years-model-output", default="models/ranker_cat_all_years.cbm")
    parser.add_argument("--meta-output", default="models/ranker_cat_bundle_meta.json")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--loss-function", default="YetiRank", choices=list(LOSS_FUNCTION_CHOICES))
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--rsm", type=float, default=1.0)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--leaf-estimation-iterations", type=int, default=5)
    parser.add_argument(
        "--drop-entity-id-features",
        action="store_true",
        help="Drop high-cardinality entity ID features (jockey_key/trainer_key).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    for token in argv:
        if token == flag or token.startswith(prefix):
            return True
    return False


def _apply_params_json(
    args: argparse.Namespace,
    params: dict[str, Any],
    *,
    argv: list[str],
) -> None:
    if (
        "input" in params
        and isinstance(params["input"], str)
        and not _argv_has_flag(argv, "--input")
    ):
        args.input = params["input"]

    if "drop_entity_id_features" in params and not _argv_has_flag(
        argv, "--drop-entity-id-features"
    ):
        args.drop_entity_id_features = bool(params["drop_entity_id_features"])

    cat_params = params.get("cat_params")
    if not isinstance(cat_params, dict):
        return

    mapping: dict[str, tuple[str, str]] = {
        "loss_function": ("loss_function", "--loss-function"),
        "learning_rate": ("learning_rate", "--learning-rate"),
        "depth": ("depth", "--depth"),
        "l2_leaf_reg": ("l2_leaf_reg", "--l2-leaf-reg"),
        "random_strength": ("random_strength", "--random-strength"),
        "bagging_temperature": ("bagging_temperature", "--bagging-temperature"),
        "rsm": ("rsm", "--rsm"),
        "min_data_in_leaf": ("min_data_in_leaf", "--min-data-in-leaf"),
        "leaf_estimation_iterations": (
            "leaf_estimation_iterations",
            "--leaf-estimation-iterations",
        ),
    }
    for key, (attr, flag) in mapping.items():
        if key not in cat_params or _argv_has_flag(argv, flag):
            continue
        setattr(args, attr, cat_params[key])


def _base_ranker_params(args: argparse.Namespace, *, iterations: int, seed: int) -> dict[str, Any]:
    return {
        "loss_function": str(args.loss_function),
        "iterations": int(iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "bagging_temperature": float(args.bagging_temperature),
        "rsm": float(args.rsm),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "leaf_estimation_iterations": int(args.leaf_estimation_iterations),
        "eval_metric": "NDCG:top=3",
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
    }


def _drop_entity_features(feature_cols: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return feature_cols
    return [col for col in feature_cols if col not in ENTITY_ID_FEATURES]


def _validate_args(args: argparse.Namespace) -> None:
    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if args.num_boost_round <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if args.train_window_years <= 0:
        raise SystemExit("--train-window-years must be > 0")
    if args.depth <= 0:
        raise SystemExit("--depth must be > 0")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be > 0")
    if args.l2_leaf_reg < 0:
        raise SystemExit("--l2-leaf-reg must be >= 0")
    if args.random_strength < 0:
        raise SystemExit("--random-strength must be >= 0")
    if args.bagging_temperature < 0:
        raise SystemExit("--bagging-temperature must be >= 0")
    if not (0.0 < float(args.rsm) <= 1.0):
        raise SystemExit("--rsm must be in (0, 1].")
    if args.min_data_in_leaf <= 0:
        raise SystemExit("--min-data-in-leaf must be > 0")
    if args.leaf_estimation_iterations <= 0:
        raise SystemExit("--leaf-estimation-iterations must be > 0")


def _train_single_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    fold: FoldSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from catboost import CatBoostRanker, Pool

    _assert_fold_integrity(train_df, valid_df, fold.valid_year)
    train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )
    valid_df = valid_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )

    X_train = _coerce_feature_matrix(train_df, feature_cols)
    y_train = train_df["target_label"].astype(int)
    X_valid = _coerce_feature_matrix(valid_df, feature_cols)
    y_valid = valid_df["target_label"].astype(int)

    train_pool = Pool(X_train, label=y_train, group_id=train_df["race_id"].to_numpy())
    valid_pool = Pool(X_valid, label=y_valid, group_id=valid_df["race_id"].to_numpy())

    model = CatBoostRanker(
        **_base_ranker_params(
            args,
            iterations=int(args.num_boost_round),
            seed=int(args.seed) + int(fold.fold_id),
        ),
        od_type="Iter",
        od_wait=int(args.early_stopping_rounds),
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    best_iteration = int(model.get_best_iteration())
    if best_iteration < 0:
        best_iteration = int(args.num_boost_round) - 1
    best_iteration += 1

    valid_score = model.predict(valid_pool)

    oof = valid_df[
        ["race_id", "horse_id", "horse_no", "race_date", "target_label", "field_size"]
    ].copy()
    oof["t_race"] = pd.to_datetime(oof["race_date"], errors="coerce")
    oof["ranker_score"] = valid_score
    oof["fold_id"] = int(fold.fold_id)
    oof["valid_year"] = int(fold.valid_year)
    oof = _rank_within_race(oof)
    ndcg3 = _fold_ndcg_at_3(oof)

    metrics = {
        "fold_id": int(fold.fold_id),
        "train_years": list(fold.train_years),
        "valid_year": int(fold.valid_year),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_races": int(train_df["race_id"].nunique()),
        "valid_races": int(valid_df["race_id"].nunique()),
        "ndcg_at_3": float(ndcg3),
        "best_iteration": int(best_iteration),
    }
    return oof, metrics


def _train_final_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    *,
    iterations: int,
    seed_offset: int,
):
    from catboost import CatBoostRanker, Pool

    train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )
    X_train = _coerce_feature_matrix(train_df, feature_cols)
    y_train = train_df["target_label"].astype(int)
    train_pool = Pool(X_train, label=y_train, group_id=train_df["race_id"].to_numpy())
    model = CatBoostRanker(
        **_base_ranker_params(
            args,
            iterations=int(iterations),
            seed=int(args.seed) + int(seed_offset),
        )
    )
    model.fit(train_pool)
    return model


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv_list)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("catboost is not installed. Run `uv sync --extra catboost`.") from exc

    if args.params_json:
        params_path = _resolve_path(args.params_json)
        if not params_path.exists():
            raise SystemExit(f"params-json not found: {params_path}")
        params = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise SystemExit("params-json must be a JSON object.")
        _apply_params_json(args, params, argv=argv_list)

    _validate_args(args)

    input_path = _resolve_path(args.input)
    oof_output = _resolve_path(args.oof_output)
    metrics_output = _resolve_path(args.metrics_output)
    model_output = _resolve_path(args.model_output)
    all_years_model_output = _resolve_path(args.all_years_model_output)
    meta_output = _resolve_path(args.meta_output)

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    frame = _prepare_dataframe(pd.read_parquet(input_path))
    frame = frame[frame["year"] < int(args.holdout_year)].copy()
    if frame.empty:
        raise SystemExit("No trainable data found after holdout exclusion.")

    years = sorted(frame["year"].unique().tolist())
    folds = build_rolling_year_folds(
        years,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    feature_cols = _drop_entity_features(
        _feature_columns(frame), bool(args.drop_entity_id_features)
    )
    if not feature_cols:
        raise SystemExit("No feature columns available for training.")

    logger.info(
        "cat ranker train years=%s folds=%s window=%s holdout_year>=%s",
        years,
        len(folds),
        args.train_window_years,
        args.holdout_year,
    )

    oof_list: list[pd.DataFrame] = []
    fold_metrics_list: list[dict[str, Any]] = []
    best_iterations: list[int] = []
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
        fold_oof, fold_metrics = _train_single_fold(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=feature_cols,
            args=args,
            fold=fold,
        )
        oof_list.append(fold_oof)
        fold_metrics_list.append(fold_metrics)
        best_iterations.append(int(fold_metrics["best_iteration"]))
        logger.info(
            "fold=%s ndcg@3=%.6f best_iteration=%s",
            fold.fold_id,
            fold_metrics["ndcg_at_3"],
            fold_metrics["best_iteration"],
        )

    if not oof_list:
        raise SystemExit("No OOF predictions generated.")

    final_iterations = int(np.median(best_iterations))
    final_iterations = max(1, min(final_iterations, int(args.num_boost_round)))

    oof = pd.concat(oof_list, axis=0, ignore_index=True)
    oof = oof[
        [
            "race_id",
            "horse_id",
            "horse_no",
            "t_race",
            "race_date",
            "target_label",
            "field_size",
            "ranker_score",
            "ranker_rank",
            "ranker_percentile",
            "fold_id",
            "valid_year",
        ]
    ].sort_values(["race_id", "horse_no"], kind="mergesort")

    recent_span = int(args.train_window_years) + 1
    recent_years = years[-recent_span:] if len(years) > recent_span else years
    recent_train_df = frame[frame["year"].isin(recent_years)].copy()
    all_train_df = frame.copy()

    logger.info(
        "final training iterations=%s recent_years=%s all_years=%s",
        final_iterations,
        recent_years,
        years,
    )
    main_model = _train_final_model(
        train_df=recent_train_df,
        feature_cols=feature_cols,
        args=args,
        iterations=final_iterations,
        seed_offset=1000,
    )
    all_years_model = _train_final_model(
        train_df=all_train_df,
        feature_cols=feature_cols,
        args=args,
        iterations=final_iterations,
        seed_offset=2000,
    )

    oof_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    all_years_model_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.parent.mkdir(parents=True, exist_ok=True)

    oof.to_parquet(oof_output, index=False)
    main_model.save_model(str(model_output))
    all_years_model.save_model(str(all_years_model_output))

    fold_ndcg = [float(x["ndcg_at_3"]) for x in fold_metrics_list]
    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "loss_function": str(args.loss_function),
            "learning_rate": float(args.learning_rate),
            "depth": int(args.depth),
            "l2_leaf_reg": float(args.l2_leaf_reg),
            "random_strength": float(args.random_strength),
            "bagging_temperature": float(args.bagging_temperature),
            "rsm": float(args.rsm),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "leaf_estimation_iterations": int(args.leaf_estimation_iterations),
            "drop_entity_id_features": bool(args.drop_entity_id_features),
            "seed": int(args.seed),
            "eval_at": [3],
        },
        "data_summary": {
            "rows": int(len(frame)),
            "races": int(frame["race_id"].nunique()),
            "years": years,
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
        },
        "folds": fold_metrics_list,
        "summary": {
            "n_folds": int(len(fold_metrics_list)),
            "ndcg_at_3_mean": float(np.mean(fold_ndcg)),
            "ndcg_at_3_std": float(np.std(fold_ndcg, ddof=0)),
            "ndcg_at_3_min": float(np.min(fold_ndcg)),
            "ndcg_at_3_max": float(np.max(fold_ndcg)),
            "best_iteration_median": int(final_iterations),
        },
    }
    _save_json(metrics_output, metrics)

    meta = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "holdout_rule": f"exclude year >= {args.holdout_year}",
        "main_model_id": "ranker_cat_recent_window",
        "input_path": str(input_path),
        "output_paths": {
            "oof": str(oof_output),
            "cv_metrics": str(metrics_output),
            "main_model": str(model_output),
            "all_years_model": str(all_years_model_output),
        },
        "feature_columns": feature_cols,
        "model_params": _base_ranker_params(
            args,
            iterations=final_iterations,
            seed=int(args.seed),
        ),
        "cv_summary": metrics["summary"],
        "final_train_summary": {
            "main_model_years": recent_years,
            "main_model_rows": int(len(recent_train_df)),
            "all_years_model_years": years,
            "all_years_model_rows": int(len(all_train_df)),
        },
        "code_hash": _hash_files([Path(__file__)]),
    }
    _save_json(meta_output, meta)

    logger.info("wrote %s", oof_output)
    logger.info("wrote %s", metrics_output)
    logger.info("wrote %s", model_output)
    logger.info("wrote %s", all_years_model_output)
    logger.info("wrote %s", meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
