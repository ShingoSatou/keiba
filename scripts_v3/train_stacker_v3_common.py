#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from scripts_v3.cv_policy_v3 import (  # noqa: E402
    DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    attach_cv_policy_columns,
    build_capped_expanding_year_folds,
    build_cv_policy_payload,
    make_capped_expanding_window_definition,
    select_recent_window_years,
)
from scripts_v3.feature_registry_v3 import (  # noqa: E402
    FEATURE_MANIFEST_VERSION,
    STACKER_REQUIRED_PRED_FEATURES_PLACE,
    STACKER_REQUIRED_PRED_FEATURES_WIN,
    STACKER_TASK_CHOICES,
    get_stacker_feature_columns,
)
from scripts_v3.train_binary_v3_common import (  # noqa: E402
    coerce_feature_matrix,
    compute_binary_metrics,
    fold_integrity,
    prepare_binary_frame,
)
from scripts_v3.v3_common import hash_files, resolve_path, save_json  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_SEED = 42
DEFAULT_CV_WINDOW_POLICY = "capped_expanding"


def _label_col(task: str) -> str:
    return "y_win" if str(task) == "win" else "y_place"


def _pred_col(task: str) -> str:
    return "p_win_stack" if str(task) == "win" else "p_place_stack"


def _required_pred_cols(task: str) -> list[str]:
    if str(task) == "win":
        return list(STACKER_REQUIRED_PRED_FEATURES_WIN)
    return list(STACKER_REQUIRED_PRED_FEATURES_PLACE)


def _default_base_oof_paths(task: str) -> dict[str, str]:
    prefix = "win" if str(task) == "win" else "place"
    return {
        "lgbm": f"data/oof/{prefix}_lgbm_oof.parquet",
        "xgb": f"data/oof/{prefix}_xgb_oof.parquet",
        "cat": f"data/oof/{prefix}_cat_oof.parquet",
    }


def _default_base_holdout_paths(task: str) -> dict[str, str]:
    prefix = "win" if str(task) == "win" else "place"
    return {
        "lgbm": f"data/holdout/{prefix}_lgbm_holdout_pred_v3.parquet",
        "xgb": f"data/holdout/{prefix}_xgb_holdout_pred_v3.parquet",
        "cat": f"data/holdout/{prefix}_cat_holdout_pred_v3.parquet",
    }


def parse_args(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
) -> argparse.Namespace:
    task = str(default_task or "win")
    oof_defaults = _default_base_oof_paths(task)
    holdout_defaults = _default_base_holdout_paths(task)

    parser = argparse.ArgumentParser(
        description=(
            "Train strict temporal v3 stacker with capped expanding yearly CV "
            "(min2, max4 by default)."
        )
    )
    parser.add_argument("--task", choices=list(STACKER_TASK_CHOICES), default=task)
    parser.add_argument("--features-input", default="data/features_v3.parquet")
    parser.add_argument("--holdout-input", default="")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=DEFAULT_STACKER_MIN_TRAIN_WINDOW_YEARS,
    )
    parser.add_argument(
        "--max-train-years",
        type=int,
        default=DEFAULT_STACKER_MAX_TRAIN_WINDOW_YEARS,
    )
    parser.add_argument("--lgbm-oof", default=oof_defaults["lgbm"])
    parser.add_argument("--xgb-oof", default=oof_defaults["xgb"])
    parser.add_argument("--cat-oof", default=oof_defaults["cat"])
    parser.add_argument("--lgbm-holdout", default=holdout_defaults["lgbm"])
    parser.add_argument("--xgb-holdout", default=holdout_defaults["xgb"])
    parser.add_argument("--cat-holdout", default=holdout_defaults["cat"])
    parser.add_argument("--oof-output", default="")
    parser.add_argument("--holdout-output", default="")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--model-output", default="")
    parser.add_argument("--all-years-model-output", default="")
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--feature-manifest-output", default="")
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-freq", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.holdout_year) <= 0:
        raise SystemExit("--holdout-year must be > 0")
    if int(args.min_train_years) <= 0:
        raise SystemExit("--min-train-years must be > 0")
    if int(args.max_train_years) < int(args.min_train_years):
        raise SystemExit("--max-train-years must be >= --min-train-years")
    if int(args.num_boost_round) <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if int(args.early_stopping_rounds) <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if int(args.n_bins) <= 1:
        raise SystemExit("--n-bins must be > 1")


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    task = str(args.task)
    defaults = {
        "oof": f"data/oof/{task}_stack_oof.parquet",
        "holdout": f"data/holdout/{task}_stack_holdout_pred_v3.parquet",
        "metrics": f"data/oof/{task}_stack_cv_metrics.json",
        "model": f"models/{task}_stack_v3.txt",
        "all_years_model": f"models/{task}_stack_all_years_v3.txt",
        "meta": f"models/{task}_stack_bundle_meta_v3.json",
        "feature_manifest": f"models/{task}_stack_feature_manifest_v3.json",
    }
    return {
        "oof": resolve_path(args.oof_output or defaults["oof"]),
        "holdout": resolve_path(args.holdout_output or defaults["holdout"]),
        "metrics": resolve_path(args.metrics_output or defaults["metrics"]),
        "model": resolve_path(args.model_output or defaults["model"]),
        "all_years_model": resolve_path(args.all_years_model_output or defaults["all_years_model"]),
        "meta": resolve_path(args.meta_output or defaults["meta"]),
        "feature_manifest": resolve_path(
            args.feature_manifest_output or defaults["feature_manifest"]
        ),
    }


def _load_prediction_frame(path: Path, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction input not found: {path}")
    frame = pd.read_parquet(path)
    required = {"race_id", "horse_no", pred_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")
    out = frame[["race_id", "horse_no", pred_col]].copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    if out.duplicated(["race_id", "horse_no"]).any():
        raise SystemExit(f"Duplicate keys in prediction input: {path}")
    return out


def _load_valid_years(path: Path) -> list[int]:
    if not path.exists():
        raise SystemExit(f"prediction input not found: {path}")
    frame = pd.read_parquet(path, columns=["valid_year"])
    if "valid_year" not in frame.columns:
        raise SystemExit(f"Missing valid_year column in {path}")
    years = (
        pd.to_numeric(frame["valid_year"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    return [int(year) for year in years]


def _prediction_paths(args: argparse.Namespace, *, holdout: bool) -> dict[str, Path]:
    if str(args.task) == "win":
        mapping = {
            "p_win_lgbm": args.lgbm_holdout if holdout else args.lgbm_oof,
            "p_win_xgb": args.xgb_holdout if holdout else args.xgb_oof,
            "p_win_cat": args.cat_holdout if holdout else args.cat_oof,
        }
    else:
        mapping = {
            "p_place_lgbm": args.lgbm_holdout if holdout else args.lgbm_oof,
            "p_place_xgb": args.xgb_holdout if holdout else args.xgb_oof,
            "p_place_cat": args.cat_holdout if holdout else args.cat_oof,
        }
    return {pred_col: resolve_path(path) for pred_col, path in mapping.items()}


def _merge_prediction_features(frame: pd.DataFrame, pred_paths: dict[str, Path]) -> pd.DataFrame:
    merged = frame.copy()
    for pred_col, path in pred_paths.items():
        pred_df = _load_prediction_frame(path, pred_col)
        merged = merged.merge(pred_df, on=["race_id", "horse_no"], how="left")
    return merged.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _fit_lgbm_fold(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, np.ndarray, int]:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_data_in_leaf),
        reg_alpha=float(args.lambda_l1),
        reg_lambda=float(args.lambda_l2),
        colsample_bytree=float(args.feature_fraction),
        subsample=float(args.bagging_fraction),
        subsample_freq=int(args.bagging_freq),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_names=["valid"],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(int(args.early_stopping_rounds), verbose=False)],
    )
    best_iteration = int(getattr(model, "best_iteration_", 0) or int(args.num_boost_round))
    pred = np.clip(
        np.asarray(model.predict_proba(x_valid, num_iteration=best_iteration)[:, 1], dtype=float),
        1e-8,
        1.0 - 1e-8,
    )
    return model, pred, best_iteration


def _fit_lgbm_final(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
    seed: int,
    n_estimators: int,
) -> Any:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_data_in_leaf),
        reg_alpha=float(args.lambda_l1),
        reg_lambda=float(args.lambda_l2),
        colsample_bytree=float(args.feature_fraction),
        subsample=float(args.bagging_fraction),
        subsample_freq=int(args.bagging_freq),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def _predict_lgbm(model: Any, x: pd.DataFrame, *, num_iteration: int | None = None) -> np.ndarray:
    pred = model.predict_proba(x, num_iteration=num_iteration)[:, 1]
    return np.clip(np.asarray(pred, dtype=float), 1e-8, 1.0 - 1e-8)


def _save_lgbm_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def _summary(values: list[float | None]) -> dict[str, float | None]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _oof_frame(
    valid_df: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    pred_values: np.ndarray,
    fold_id: int,
    valid_year: int,
    args: argparse.Namespace,
    window_definition: str,
) -> pd.DataFrame:
    out = valid_df[
        [
            c
            for c in [
                "race_id",
                "horse_id",
                "horse_no",
                "t_race",
                "race_date",
                "field_size",
                "target_label",
                label_col,
            ]
            if c in valid_df.columns
        ]
    ].copy()
    out[pred_col] = np.asarray(pred_values, dtype=float)
    out["fold_id"] = int(fold_id)
    out["valid_year"] = int(valid_year)
    out = attach_cv_policy_columns(
        out,
        train_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
        cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        window_definition=window_definition,
    )
    out["min_train_window_years"] = int(args.min_train_years)
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort")
    return out


def _feature_manifest_payload(
    *,
    args: argparse.Namespace,
    feature_cols: list[str],
    valid_years: list[int],
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": "lgbm",
        "pred_col": _pred_col(str(args.task)),
        "cv_policy": {
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "min_train_window_years": int(args.min_train_years),
            "train_window_years": int(args.max_train_years),
            "valid_years": list(map(int, valid_years)),
            "holdout_year": int(args.holdout_year),
            "window_definition": make_capped_expanding_window_definition(
                int(args.min_train_years),
                int(args.max_train_years),
            ),
        },
        "feature_columns": list(feature_cols),
        "forbidden_feature_check_passed": True,
        "feature_manifest_version": int(FEATURE_MANIFEST_VERSION),
    }


def _meta_payload(
    *,
    args: argparse.Namespace,
    feature_cols: list[str],
    base_valid_years: list[int],
    valid_years: list[int],
    recent_years: list[int],
    input_paths: dict[str, str],
    output_paths: dict[str, Path],
    holdout_rows: int,
    holdout_races: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": "lgbm",
        "pred_col": _pred_col(str(args.task)),
        "feature_columns": list(feature_cols),
        "cv_policy": {
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "min_train_window_years": int(args.min_train_years),
            "train_window_years": int(args.max_train_years),
            "valid_years": list(map(int, valid_years)),
            "holdout_year": int(args.holdout_year),
            "window_definition": make_capped_expanding_window_definition(
                int(args.min_train_years),
                int(args.max_train_years),
            ),
        },
        "base_oof_valid_years": list(map(int, base_valid_years)),
        "stacker_oof_valid_years": list(map(int, valid_years)),
        "recent_window_years": list(map(int, recent_years)),
        "input_paths": input_paths,
        "output_paths": {
            "oof": str(output_paths["oof"]),
            "holdout": str(output_paths["holdout"]),
            "metrics": str(output_paths["metrics"]),
            "main_model": str(output_paths["model"]),
            "all_years_model": str(output_paths["all_years_model"]),
            "feature_manifest": str(output_paths["feature_manifest"]),
        },
        "holdout_summary": {
            "rows": int(holdout_rows),
            "races": int(holdout_races),
        },
        "forbidden_feature_check_passed": True,
        "feature_manifest_version": int(FEATURE_MANIFEST_VERSION),
        "code_hash": hash_files(
            [
                Path(__file__),
                Path(resolve_path("scripts_v3/feature_registry_v3.py")),
                Path(resolve_path("scripts_v3/cv_policy_v3.py")),
            ]
        ),
    }


def main(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
) -> int:
    args = parse_args(argv, default_task=default_task)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    _validate_args(args)

    label_col = _label_col(str(args.task))
    pred_col = _pred_col(str(args.task))
    required_pred_cols = _required_pred_cols(str(args.task))
    outputs = _resolve_output_paths(args)
    window_definition = make_capped_expanding_window_definition(
        int(args.min_train_years),
        int(args.max_train_years),
    )

    features_path = resolve_path(args.features_input)
    if not features_path.exists():
        raise SystemExit(f"features input not found: {features_path}")
    raw = pd.read_parquet(features_path)
    base = prepare_binary_frame(raw, label_col=label_col)
    merged = _merge_prediction_features(base, _prediction_paths(args, holdout=False))
    feature_cols = get_stacker_feature_columns(merged, task=str(args.task))
    base_valid_years = sorted(
        set.intersection(
            *[
                set(_load_valid_years(path))
                for path in _prediction_paths(args, holdout=False).values()
            ]
        )
    )

    eligible = merged.copy()
    for col in required_pred_cols:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
    eligible = eligible[eligible["year"] < int(args.holdout_year)].copy()
    eligible = eligible[eligible[required_pred_cols].notna().all(axis=1)].copy()
    if eligible.empty:
        raise SystemExit("No eligible rows for stacker training after OOF filtering.")

    years = sorted(eligible["year"].unique().tolist())
    folds = build_capped_expanding_year_folds(
        years,
        min_window_years=int(args.min_train_years),
        max_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        raise SystemExit(
            "No stacker folds available under capped_expanding "
            f"(available_years={years}, min={args.min_train_years}, max={args.max_train_years})"
        )

    oof_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold in folds:
        train_df = eligible[eligible["year"].isin(fold.train_years)].copy()
        valid_df = eligible[eligible["year"] == fold.valid_year].copy()
        if train_df.empty or valid_df.empty:
            raise SystemExit(
                f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
            )
        fold_integrity(train_df, valid_df, int(fold.valid_year))

        x_train = coerce_feature_matrix(train_df, feature_cols)
        y_train = train_df[label_col].astype(int)
        x_valid = coerce_feature_matrix(valid_df, feature_cols)
        y_valid = valid_df[label_col].astype(int)

        _, pred_valid, best_iteration = _fit_lgbm_fold(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=args,
            seed=int(args.seed) + int(fold.fold_id),
        )
        best_iterations.append(int(best_iteration))

        metrics = compute_binary_metrics(
            y_valid.to_numpy(dtype=int),
            pred_valid,
            n_bins=int(args.n_bins),
        )
        fold_metrics.append(
            {
                "fold_id": int(fold.fold_id),
                "train_years": list(map(int, fold.train_years)),
                "valid_year": int(fold.valid_year),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_races": int(train_df["race_id"].nunique()),
                "valid_races": int(valid_df["race_id"].nunique()),
                "best_iteration": int(best_iteration),
                "logloss": metrics["logloss"],
                "brier": metrics["brier"],
                "auc": metrics["auc"],
                "ece": metrics["ece"],
                "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
                "min_train_window_years": int(args.min_train_years),
                "train_window_years": int(args.max_train_years),
                "holdout_year": int(args.holdout_year),
                "window_definition": window_definition,
            }
        )
        oof_frames.append(
            _oof_frame(
                valid_df,
                label_col=label_col,
                pred_col=pred_col,
                pred_values=pred_valid,
                fold_id=int(fold.fold_id),
                valid_year=int(fold.valid_year),
                args=args,
                window_definition=window_definition,
            )
        )

    oof = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(
        ["race_id", "horse_no"],
        kind="mergesort",
    )
    outputs["oof"].parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(outputs["oof"], index=False)

    final_iterations = int(round(float(np.median(best_iterations)))) if best_iterations else int(
        args.num_boost_round
    )
    recent_years = select_recent_window_years(
        years,
        train_window_years=int(args.max_train_years),
        holdout_year=int(args.holdout_year),
    )
    recent_df = eligible[eligible["year"].isin(recent_years)].copy()
    all_df = eligible.copy()

    main_model = _fit_lgbm_final(
        x_train=coerce_feature_matrix(recent_df, feature_cols),
        y_train=recent_df[label_col].astype(int),
        args=args,
        seed=int(args.seed) + 1000,
        n_estimators=int(final_iterations),
    )
    all_years_model = _fit_lgbm_final(
        x_train=coerce_feature_matrix(all_df, feature_cols),
        y_train=all_df[label_col].astype(int),
        args=args,
        seed=int(args.seed) + 2000,
        n_estimators=int(final_iterations),
    )
    _save_lgbm_model(main_model, outputs["model"])
    _save_lgbm_model(all_years_model, outputs["all_years_model"])

    holdout_rows = 0
    holdout_races = 0
    holdout_input = resolve_path(args.holdout_input) if args.holdout_input else None
    if holdout_input is not None and holdout_input.exists():
        holdout_raw = pd.read_parquet(holdout_input)
        if label_col not in holdout_raw.columns:
            holdout_raw[label_col] = 0
        holdout_base = prepare_binary_frame(holdout_raw, label_col=label_col)
        holdout_merged = _merge_prediction_features(
            holdout_base,
            _prediction_paths(args, holdout=True),
        )
        holdout_merged = holdout_merged[holdout_merged["year"] >= int(args.holdout_year)].copy()
        holdout_x = coerce_feature_matrix(holdout_merged, feature_cols)
        holdout_pred = holdout_merged[
            [
                c
                for c in [
                    "race_id",
                    "horse_id",
                    "horse_no",
                    "race_date",
                    "field_size",
                    label_col,
                ]
                if c in holdout_merged.columns
            ]
        ].copy()
        holdout_pred[pred_col] = _predict_lgbm(
            main_model,
            holdout_x,
            num_iteration=int(final_iterations),
        )
        holdout_pred["valid_year"] = holdout_merged["year"].astype(int).to_numpy()
        holdout_pred = attach_cv_policy_columns(
            holdout_pred,
            train_window_years=int(args.max_train_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            window_definition=window_definition,
        )
        holdout_pred["min_train_window_years"] = int(args.min_train_years)
        holdout_pred = holdout_pred.sort_values(["race_id", "horse_no"], kind="mergesort")
        outputs["holdout"].parent.mkdir(parents=True, exist_ok=True)
        holdout_pred.to_parquet(outputs["holdout"], index=False)
        holdout_rows = int(len(holdout_pred))
        holdout_races = int(holdout_pred["race_id"].nunique())

    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": "lgbm",
        "pred_col": pred_col,
        "required_pred_cols": required_pred_cols,
        "feature_columns": feature_cols,
        "cv_policy": build_cv_policy_payload(
            folds,
            train_window_years=int(args.max_train_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            window_definition=window_definition,
        ),
        "config": {
            "holdout_year": int(args.holdout_year),
            "min_train_window_years": int(args.min_train_years),
            "train_window_years": int(args.max_train_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
        },
        "data_summary": {
            "rows": int(len(eligible)),
            "races": int(eligible["race_id"].nunique()),
            "years": years,
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
            "holdout_rows": int(holdout_rows),
            "holdout_races": int(holdout_races),
        },
        "folds": fold_metrics,
        "summary": {
            "logloss": _summary([m.get("logloss") for m in fold_metrics]),
            "brier": _summary([m.get("brier") for m in fold_metrics]),
            "auc": _summary([m.get("auc") for m in fold_metrics]),
            "ece": _summary([m.get("ece") for m in fold_metrics]),
            "best_iteration": _summary([float(x) for x in best_iterations]),
        },
        "final_iterations": int(final_iterations),
    }
    save_json(outputs["metrics"], metrics_payload)

    input_paths = {
        "features_v3": str(features_path),
        **{f"base_oof_{k}": str(v) for k, v in _prediction_paths(args, holdout=False).items()},
    }
    if holdout_input is not None:
        input_paths["holdout_input"] = str(holdout_input)
        input_paths.update(
            {
                f"base_holdout_{k}": str(v)
                for k, v in _prediction_paths(args, holdout=True).items()
            }
        )

    save_json(
        outputs["feature_manifest"],
        _feature_manifest_payload(
            args=args,
            feature_cols=feature_cols,
            valid_years=[int(fold.valid_year) for fold in folds],
        ),
    )
    save_json(
        outputs["meta"],
        _meta_payload(
            args=args,
            feature_cols=feature_cols,
            base_valid_years=base_valid_years,
            valid_years=[int(fold.valid_year) for fold in folds],
            recent_years=recent_years,
            input_paths=input_paths,
            output_paths=outputs,
            holdout_rows=holdout_rows,
            holdout_races=holdout_races,
        ),
    )

    logger.info(
        "stacker task=%s years=%s folds=%s recent_years=%s holdout_rows=%s",
        args.task,
        years,
        len(folds),
        recent_years,
        holdout_rows,
    )
    logger.info("wrote %s", outputs["oof"])
    logger.info("wrote %s", outputs["metrics"])
    logger.info("wrote %s", outputs["model"])
    logger.info("wrote %s", outputs["all_years_model"])
    logger.info("wrote %s", outputs["feature_manifest"])
    logger.info("wrote %s", outputs["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
