#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

IDENTIFIER_COLUMNS = {"race_id", "horse_id", "horse_no", "race_date", "target_label"}
DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_TRAIN_WINDOW_YEARS = 5
DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_SEED = 42


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_years: tuple[int, ...]
    valid_year: int


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 ranker (LightGBM LambdaRank).")
    parser.add_argument("--input", default="data/features_v2.parquet")
    parser.add_argument("--oof-output", default="data/oof/ranker_oof.parquet")
    parser.add_argument("--metrics-output", default="data/oof/ranker_cv_metrics.json")
    parser.add_argument("--model-output", default="models/ranker_lgbm.txt")
    parser.add_argument("--all-years-model-output", default="models/ranker_lgbm_all_years.txt")
    parser.add_argument("--meta-output", default="models/ranker_bundle_meta.json")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="keiba-v2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode when --wandb is enabled.",
    )
    return parser.parse_args(argv)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "horse_id", "horse_no", "race_date", "target_label", "field_size"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    out = frame.copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    out = out[out["race_date"].notna()].copy()
    out["year"] = out["race_date"].dt.year.astype(int)
    out["target_label"] = pd.to_numeric(out["target_label"], errors="coerce").fillna(0).astype(int)
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["horse_no"].notna()].copy()
    out["horse_no"] = out["horse_no"].astype(int)
    out["field_size"] = pd.to_numeric(out["field_size"], errors="coerce")
    out["horse_id"] = out["horse_id"].astype(str)
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    return out


def build_rolling_year_folds(
    years: list[int],
    *,
    train_window_years: int,
    holdout_year: int,
) -> list[FoldSpec]:
    if train_window_years <= 0:
        raise ValueError("train_window_years must be > 0")
    trainable_years = sorted(y for y in years if y < holdout_year)
    if len(trainable_years) < train_window_years + 1:
        raise ValueError(
            "Not enough non-holdout years for rolling folds: "
            f"need >= {train_window_years + 1}, got {len(trainable_years)}"
        )
    folds: list[FoldSpec] = []
    for idx in range(train_window_years, len(trainable_years)):
        fold = FoldSpec(
            fold_id=len(folds) + 1,
            train_years=tuple(trainable_years[idx - train_window_years : idx]),
            valid_year=trainable_years[idx],
        )
        folds.append(fold)
    return folds


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in IDENTIFIER_COLUMNS and c != "year"]


def _coerce_feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    mat = frame[feature_cols].copy()
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(mat[col]):
            mat[col] = pd.to_numeric(mat[col], errors="coerce")
    return mat


def _categorical_features(feature_cols: list[str]) -> list[str]:
    preferred = ["jockey_key", "trainer_key"]
    return [col for col in preferred if col in feature_cols]


def _group_sizes(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=np.int32)
    groups = frame.groupby("race_id", sort=False).size().to_numpy(dtype=np.int32)
    if int(groups.sum()) != len(frame):
        raise ValueError("Invalid group sizes.")
    return groups


def _assert_fold_integrity(train_df: pd.DataFrame, valid_df: pd.DataFrame, valid_year: int) -> None:
    train_year_max = int(train_df["year"].max())
    if train_year_max >= valid_year:
        raise ValueError(
            f"Temporal leakage detected: train max year={train_year_max}, valid year={valid_year}"
        )
    overlap = set(train_df["race_id"].unique()) & set(valid_df["race_id"].unique())
    if overlap:
        raise ValueError(f"Race leakage detected across train/valid: {len(overlap)} races")


def _rank_within_race(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["race_id", "ranker_score", "horse_no"], ascending=[True, False, True])
    out["ranker_rank"] = out.groupby("race_id", sort=False).cumcount() + 1
    race_size = out.groupby("race_id", sort=False)["race_id"].transform("size").astype(float)
    out["ranker_percentile"] = np.where(
        race_size <= 1.0,
        1.0,
        1.0 - (out["ranker_rank"].astype(float) - 1.0) / (race_size - 1.0),
    )
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort")
    return out


def _fold_ndcg_at_3(valid_df: pd.DataFrame) -> float:
    values: list[float] = []
    for _, sub in valid_df.groupby("race_id", sort=False):
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


def _base_ranker_params(args: argparse.Namespace, *, n_estimators: int, seed: int) -> dict:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "n_estimators": int(n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "random_state": int(seed),
        "n_jobs": -1,
    }


def _train_single_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    args: argparse.Namespace,
    fold: FoldSpec,
) -> tuple[pd.DataFrame, dict]:
    _assert_fold_integrity(train_df, valid_df, fold.valid_year)

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

    model = LGBMRanker(
        **_base_ranker_params(
            args,
            n_estimators=args.num_boost_round,
            seed=args.seed + fold.fold_id,
        )
    )
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

    valid_score = model.predict(X_valid, num_iteration=best_iteration)
    oof = valid_df[
        ["race_id", "horse_id", "horse_no", "race_date", "target_label", "field_size"]
    ].copy()
    oof["t_race"] = pd.to_datetime(oof["race_date"], errors="coerce")
    oof["ranker_score"] = valid_score
    oof["fold_id"] = int(fold.fold_id)
    oof["valid_year"] = int(fold.valid_year)
    oof = _rank_within_race(oof)
    ndcg3 = _fold_ndcg_at_3(oof)

    fold_metrics = {
        "fold_id": int(fold.fold_id),
        "train_years": list(fold.train_years),
        "valid_year": int(fold.valid_year),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_races": int(train_df["race_id"].nunique()),
        "valid_races": int(valid_df["race_id"].nunique()),
        "ndcg_at_3": ndcg3,
        "best_iteration": best_iteration,
    }
    return oof, fold_metrics


def _train_final_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    args: argparse.Namespace,
    *,
    n_estimators: int,
    seed_offset: int,
) -> LGBMRanker:
    train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )
    X_train = _coerce_feature_matrix(train_df, feature_cols)
    y_train = train_df["target_label"].astype(int)
    g_train = _group_sizes(train_df)
    model = LGBMRanker(
        **_base_ranker_params(
            args,
            n_estimators=n_estimators,
            seed=args.seed + seed_offset,
        )
    )
    model.fit(
        X_train,
        y_train,
        group=g_train,
        eval_metric="ndcg",
        categorical_feature=categorical_cols or "auto",
    )
    return model


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_wandb_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _init_wandb_run(args: argparse.Namespace, *, config: dict[str, Any]):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("wandb is not installed. Run `uv sync --extra wandb`.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=_parse_wandb_tags(args.wandb_tags),
        mode=args.wandb_mode,
        config=config,
    )
    return run


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
    oof_output = _resolve_path(args.oof_output)
    metrics_output = _resolve_path(args.metrics_output)
    model_output = _resolve_path(args.model_output)
    all_years_model_output = _resolve_path(args.all_years_model_output)
    meta_output = _resolve_path(args.meta_output)

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
    if not feature_cols:
        raise SystemExit("No feature columns available for training.")
    categorical_cols = _categorical_features(feature_cols)

    logger.info(
        "ranker train years=%s folds=%s window=%s holdout_year>=%s",
        years,
        len(folds),
        args.train_window_years,
        args.holdout_year,
    )

    wandb_run = _init_wandb_run(
        args,
        config={
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "seed": int(args.seed),
            "objective": "lambdarank",
            "eval_at": [3],
            "rows": int(len(frame)),
            "races": int(frame["race_id"].nunique()),
            "years": years,
        },
    )
    if wandb_run is not None:
        wandb_run.summary["planned_folds"] = int(len(folds))

    oof_list: list[pd.DataFrame] = []
    fold_metrics_list: list[dict] = []
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
            categorical_cols=categorical_cols,
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
        if wandb_run is not None:
            wandb_run.log(
                {
                    "cv/fold_id": int(fold.fold_id),
                    "cv/valid_year": int(fold_metrics["valid_year"]),
                    "cv/ndcg_at_3": float(fold_metrics["ndcg_at_3"]),
                    "cv/best_iteration": int(fold_metrics["best_iteration"]),
                    "cv/train_rows": int(fold_metrics["train_rows"]),
                    "cv/valid_rows": int(fold_metrics["valid_rows"]),
                    "cv/train_races": int(fold_metrics["train_races"]),
                    "cv/valid_races": int(fold_metrics["valid_races"]),
                },
                step=int(fold.fold_id),
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

    recent_span = args.train_window_years + 1
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
        categorical_cols=categorical_cols,
        args=args,
        n_estimators=final_iterations,
        seed_offset=1000,
    )
    all_years_model = _train_final_model(
        train_df=all_train_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        args=args,
        n_estimators=final_iterations,
        seed_offset=2000,
    )

    oof_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    all_years_model_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_output, index=False)
    main_model.booster_.save_model(str(model_output))
    all_years_model.booster_.save_model(str(all_years_model_output))

    fold_ndcg = [float(x["ndcg_at_3"]) for x in fold_metrics_list]
    metrics = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "seed": int(args.seed),
            "objective": "lambdarank",
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
        "main_model_id": "ranker_lgbm_recent_window",
        "input_path": str(input_path),
        "output_paths": {
            "oof": str(oof_output),
            "cv_metrics": str(metrics_output),
            "main_model": str(model_output),
            "all_years_model": str(all_years_model_output),
        },
        "feature_columns": feature_cols,
        "categorical_features": categorical_cols,
        "model_params": _base_ranker_params(
            args,
            n_estimators=final_iterations,
            seed=args.seed,
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

    if wandb_run is not None:
        wandb_run.log(
            {
                "cv/ndcg_at_3_mean": float(metrics["summary"]["ndcg_at_3_mean"]),
                "cv/ndcg_at_3_std": float(metrics["summary"]["ndcg_at_3_std"]),
                "cv/ndcg_at_3_min": float(metrics["summary"]["ndcg_at_3_min"]),
                "cv/ndcg_at_3_max": float(metrics["summary"]["ndcg_at_3_max"]),
                "cv/best_iteration_median": int(metrics["summary"]["best_iteration_median"]),
                "oof/rows": int(metrics["data_summary"]["oof_rows"]),
                "oof/races": int(metrics["data_summary"]["oof_races"]),
                "model/main_rows": int(meta["final_train_summary"]["main_model_rows"]),
                "model/all_years_rows": int(meta["final_train_summary"]["all_years_model_rows"]),
            }
        )
        wandb_run.summary["oof_path"] = str(oof_output)
        wandb_run.summary["cv_metrics_path"] = str(metrics_output)
        wandb_run.summary["main_model_path"] = str(model_output)
        wandb_run.summary["all_years_model_path"] = str(all_years_model_output)
        wandb_run.summary["bundle_meta_path"] = str(meta_output)
        wandb_run.finish()

    logger.info("wrote %s", oof_output)
    logger.info("wrote %s", metrics_output)
    logger.info("wrote %s", model_output)
    logger.info("wrote %s", all_years_model_output)
    logger.info("wrote %s", meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
