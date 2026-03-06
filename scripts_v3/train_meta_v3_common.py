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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.train_binary_v3_common import (  # noqa: E402
    compute_binary_metrics,
    hash_files,
    resolve_path,
    save_json,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
TASK_CHOICES = ("win", "place")
CV_STRATEGY_CHOICES = ("auto", "stratified_group_kfold", "group_kfold")


def _label_col(task: str) -> str:
    return "y_win" if str(task) == "win" else "y_place"


def _pred_col(task: str) -> str:
    return "p_win_meta" if str(task) == "win" else "p_place_meta"


def _base_pred_cols(task: str) -> list[str]:
    prefix = "win" if str(task) == "win" else "place"
    return [f"p_{prefix}_lgbm", f"p_{prefix}_xgb", f"p_{prefix}_cat"]


def _default_raw_oof_paths(task: str) -> dict[str, str]:
    prefix = "win" if str(task) == "win" else "place"
    return {
        "lgbm": f"data/oof/{prefix}_lgbm_oof.parquet",
        "xgb": f"data/oof/{prefix}_xgb_oof.parquet",
        "cat": f"data/oof/{prefix}_cat_oof.parquet",
    }


def _default_raw_holdout_paths(task: str) -> dict[str, str]:
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
    oof_defaults = _default_raw_oof_paths(task)
    holdout_defaults = _default_raw_holdout_paths(task)

    parser = argparse.ArgumentParser(
        description=(
            "Train grouped-CV meta combiner for v3 base predictions. "
            "Meta OOF is reference-only and not strict temporal."
        )
    )
    parser.add_argument("--task", choices=list(TASK_CHOICES), default=task)
    parser.add_argument("--features-input", default="data/features_v3.parquet")
    parser.add_argument("--holdout-input", default="")
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--lgbm-oof", default=oof_defaults["lgbm"])
    parser.add_argument("--xgb-oof", default=oof_defaults["xgb"])
    parser.add_argument("--cat-oof", default=oof_defaults["cat"])
    parser.add_argument("--lgbm-holdout", default=holdout_defaults["lgbm"])
    parser.add_argument("--xgb-holdout", default=holdout_defaults["xgb"])
    parser.add_argument("--cat-holdout", default=holdout_defaults["cat"])
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--cv-strategy",
        choices=list(CV_STRATEGY_CHOICES),
        default="auto",
        help="Grouped CV strategy for reference-only meta OOF.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--oof-output", default="")
    parser.add_argument("--holdout-output", default="")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--model-output", default="")
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.holdout_year) <= 0:
        raise SystemExit("--holdout-year must be > 0")
    if int(args.n_splits) <= 1:
        raise SystemExit("--n-splits must be > 1")
    if int(args.n_bins) <= 1:
        raise SystemExit("--n-bins must be > 1")


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    task = str(args.task)
    defaults = {
        "oof": f"data/oof/{task}_meta_oof.parquet",
        "holdout": f"data/holdout/{task}_meta_holdout_pred_v3.parquet",
        "metrics": f"data/oof/{task}_meta_cv_metrics_reference.json",
        "model": f"models/{task}_meta_v3.pkl",
        "meta": f"models/{task}_meta_bundle_meta_v3.json",
    }
    return {
        "oof": resolve_path(args.oof_output or defaults["oof"]),
        "holdout": resolve_path(args.holdout_output or defaults["holdout"]),
        "metrics": resolve_path(args.metrics_output or defaults["metrics"]),
        "model": resolve_path(args.model_output or defaults["model"]),
        "meta": resolve_path(args.meta_output or defaults["meta"]),
    }


def _hash_path(path: Path) -> dict[str, str]:
    return {"path": str(path), "hash": hash_files([path])}


def _build_input_artifacts_payload(
    *,
    features_input: Path,
    raw_oof_inputs: dict[str, Path],
    holdout_input: Path | None,
    raw_holdout_inputs: dict[str, Path] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "features_input": _hash_path(features_input),
        "raw_oof_inputs": {name: _hash_path(path) for name, path in raw_oof_inputs.items()},
    }
    if holdout_input is not None and holdout_input.exists():
        payload["holdout_input"] = _hash_path(holdout_input)
    if raw_holdout_inputs:
        existing = {
            name: _hash_path(path) for name, path in raw_holdout_inputs.items() if path.exists()
        }
        if existing:
            payload["raw_holdout_inputs"] = existing
    return payload


def _prep_frame(
    frame: pd.DataFrame,
    *,
    label_col: str | None,
) -> pd.DataFrame:
    required = {"race_id", "horse_id", "horse_no", "race_date", "field_size"}
    if label_col is not None:
        required.add(label_col)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    out = frame.copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    out = out[out["race_date"].notna()].copy()
    out["year"] = out["race_date"].dt.year.astype(int)
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out["horse_id"] = out["horse_id"].astype(str)
    out["field_size"] = pd.to_numeric(out["field_size"], errors="coerce")
    if label_col is not None:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
    if "finish_pos" in out.columns:
        out["finish_pos"] = pd.to_numeric(out["finish_pos"], errors="coerce")
        out["y_top3"] = np.where(
            out["finish_pos"].notna(),
            (out["finish_pos"] <= 3).astype(int),
            np.nan,
        )
    if out.duplicated(["race_id", "horse_no"]).any():
        dup = out[out.duplicated(["race_id", "horse_no"], keep=False)][
            ["race_id", "horse_no"]
        ].head()
        raise SystemExit(f"Duplicate (race_id, horse_no): {dup.to_dict('records')}")
    return out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _load_prediction_frame(path: Path, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction input not found: {path}")
    df = pd.read_parquet(path)
    required = {"race_id", "horse_no", pred_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")
    out = df[["race_id", "horse_no", pred_col]].copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    if out.duplicated(["race_id", "horse_no"]).any():
        raise SystemExit(f"Duplicate keys in prediction input: {path}")
    return out


def _merge_prediction_features(
    frame: pd.DataFrame,
    *,
    pred_paths: dict[str, Path],
) -> pd.DataFrame:
    merged = frame.copy()
    for pred_col, path in pred_paths.items():
        pred_df = _load_prediction_frame(path, pred_col)
        merged = merged.merge(pred_df, on=["race_id", "horse_no"], how="left")
    return merged.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _resolve_cv_splits(
    frame: pd.DataFrame,
    *,
    label_col: str,
    n_splits: int,
    cv_strategy: str,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    groups = frame["race_id"].to_numpy(dtype=int)
    y = frame[label_col].to_numpy(dtype=int)
    unique_groups = np.unique(groups)
    effective_n_splits = min(int(n_splits), int(unique_groups.shape[0]))
    if effective_n_splits < 2:
        raise SystemExit("Grouped meta CV requires at least 2 distinct race_id groups.")

    requested = str(cv_strategy)
    if requested not in CV_STRATEGY_CHOICES:
        raise SystemExit(f"Unsupported cv_strategy={requested!r}")

    can_try_stratified = requested in {"auto", "stratified_group_kfold"} and (
        int(np.sum(y == 1)) > 0 and int(np.sum(y == 0)) > 0
    )

    if can_try_stratified:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            splitter = StratifiedGroupKFold(
                n_splits=effective_n_splits,
                shuffle=True,
                random_state=int(seed),
            )
            splits = list(splitter.split(np.zeros(len(frame)), y, groups))
            return splits, {
                "requested": requested,
                "used": "stratified_group_kfold",
                "n_splits": int(effective_n_splits),
            }
        except (ImportError, ValueError):
            if requested == "stratified_group_kfold":
                raise

    splitter = GroupKFold(n_splits=effective_n_splits)
    splits = list(splitter.split(np.zeros(len(frame)), y, groups))
    return splits, {
        "requested": requested,
        "used": "group_kfold",
        "n_splits": int(effective_n_splits),
    }


def _fit_model(x_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> LogisticRegression:
    uniques = np.unique(y_train)
    if int(uniques.shape[0]) <= 1:
        raise SystemExit("Meta training requires both positive and negative labels.")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        random_state=int(seed),
    )
    model.fit(x_train, y_train)
    return model


def _predict_meta(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    pred = model.predict_proba(x)[:, 1]
    return np.clip(np.asarray(pred, dtype=float), 1e-8, 1.0 - 1e-8)


def _summary_stats(values: list[float | None]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _build_oof_frame(
    valid_df: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    pred_values: np.ndarray,
    fold_id: int,
    cv_used: str,
    n_splits: int,
) -> pd.DataFrame:
    out_cols = [
        "race_id",
        "horse_id",
        "horse_no",
        "t_race",
        "race_date",
        "field_size",
        "target_label",
        label_col,
        "year",
    ]
    out = valid_df[[col for col in out_cols if col in valid_df.columns]].copy()
    out[pred_col] = np.asarray(pred_values, dtype=float)
    out["fold_id"] = int(fold_id)
    out["valid_year"] = out["year"].astype(int)
    out["cv_strategy"] = str(cv_used)
    out["cv_is_temporal"] = False
    out["group_key"] = "race_id"
    out["n_splits"] = int(n_splits)
    return out.sort_values(["race_id", "horse_no"], kind="mergesort")


def _build_holdout_frame(
    holdout_df: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    pred_values: np.ndarray,
    cv_used: str,
    n_splits: int,
) -> pd.DataFrame:
    out_cols = [
        "race_id",
        "horse_id",
        "horse_no",
        "race_date",
        "field_size",
        "target_label",
        "finish_pos",
        label_col,
        "y_top3",
        "year",
    ]
    out = holdout_df[[col for col in out_cols if col in holdout_df.columns]].copy()
    out[pred_col] = np.asarray(pred_values, dtype=float)
    out["valid_year"] = out["year"].astype(int)
    out["cv_strategy"] = str(cv_used)
    out["cv_is_temporal"] = False
    out["group_key"] = "race_id"
    out["n_splits"] = int(n_splits)
    return out.sort_values(["race_id", "horse_no"], kind="mergesort")


def _build_reference_summary(
    frame: pd.DataFrame,
    *,
    label_col: str,
    pred_col: str,
    n_bins: int,
) -> dict[str, Any]:
    sub = frame[frame[pred_col].notna()].copy()
    if sub.empty:
        return {
            "rows": 0,
            "races": 0,
            "metrics": {"logloss": None, "brier": None, "auc": None, "ece": None},
        }
    metrics = compute_binary_metrics(
        sub[label_col].to_numpy(dtype=int),
        sub[pred_col].to_numpy(dtype=float),
        n_bins=int(n_bins),
    )
    return {
        "rows": int(len(sub)),
        "races": int(sub["race_id"].nunique()),
        "metrics": {
            "logloss": metrics["logloss"],
            "brier": metrics["brier"],
            "auc": metrics["auc"],
            "ece": metrics["ece"],
        },
    }


def main(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
) -> int:
    args = parse_args(argv, default_task=default_task)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    _validate_args(args)

    task = str(args.task)
    label_col = _label_col(task)
    pred_col = _pred_col(task)
    feature_columns = _base_pred_cols(task)
    outputs = _resolve_output_paths(args)

    features_input = resolve_path(args.features_input)
    holdout_input = resolve_path(args.holdout_input) if args.holdout_input else None
    raw_oof_paths = {
        feature_columns[0]: resolve_path(args.lgbm_oof),
        feature_columns[1]: resolve_path(args.xgb_oof),
        feature_columns[2]: resolve_path(args.cat_oof),
    }
    raw_holdout_paths = {
        feature_columns[0]: resolve_path(args.lgbm_holdout),
        feature_columns[1]: resolve_path(args.xgb_holdout),
        feature_columns[2]: resolve_path(args.cat_holdout),
    }

    if not features_input.exists():
        raise SystemExit(f"features input not found: {features_input}")

    train_base = _prep_frame(pd.read_parquet(features_input), label_col=label_col)
    train_base = train_base[train_base["year"] < int(args.holdout_year)].copy()
    if train_base.empty:
        raise SystemExit("No trainable rows after holdout exclusion.")

    merged_train = _merge_prediction_features(train_base, pred_paths=raw_oof_paths)
    for col in feature_columns:
        merged_train[col] = pd.to_numeric(merged_train[col], errors="coerce")
    eligible = merged_train[merged_train[feature_columns].notna().all(axis=1)].copy()
    if eligible.empty:
        raise SystemExit("No eligible rows for meta training after raw OOF merge.")

    splits, cv_payload = _resolve_cv_splits(
        eligible,
        label_col=label_col,
        n_splits=int(args.n_splits),
        cv_strategy=str(args.cv_strategy),
        seed=int(args.seed),
    )
    logger.info(
        "meta-%s rows=%s races=%s years=%s cv_strategy=%s n_splits=%s",
        task,
        len(eligible),
        eligible["race_id"].nunique(),
        sorted(eligible["year"].unique().tolist()),
        cv_payload["used"],
        cv_payload["n_splits"],
    )

    oof_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    x_all = eligible[feature_columns].to_numpy(dtype=float)
    y_all = eligible[label_col].to_numpy(dtype=int)

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_df = eligible.iloc[train_idx].copy()
        valid_df = eligible.iloc[valid_idx].copy()
        if set(train_df["race_id"].unique()) & set(valid_df["race_id"].unique()):
            raise SystemExit("Grouped meta CV leakage detected across train/valid race_id.")

        model = _fit_model(x_all[train_idx], y_all[train_idx], seed=int(args.seed) + int(fold_id))
        pred_valid = _predict_meta(model, x_all[valid_idx])
        oof_parts.append(
            _build_oof_frame(
                valid_df,
                label_col=label_col,
                pred_col=pred_col,
                pred_values=pred_valid,
                fold_id=fold_id,
                cv_used=str(cv_payload["used"]),
                n_splits=int(cv_payload["n_splits"]),
            )
        )

        metrics = compute_binary_metrics(
            valid_df[label_col].to_numpy(dtype=int),
            pred_valid,
            n_bins=int(args.n_bins),
        )
        fold_metrics.append(
            {
                "fold_id": int(fold_id),
                "cv_strategy": str(cv_payload["used"]),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "train_races": int(train_df["race_id"].nunique()),
                "valid_races": int(valid_df["race_id"].nunique()),
                "valid_years": sorted(valid_df["year"].astype(int).unique().tolist()),
                "logloss": metrics["logloss"],
                "brier": metrics["brier"],
                "auc": metrics["auc"],
                "ece": metrics["ece"],
                "base_rate": metrics["base_rate"],
                "reliability": metrics["reliability"],
            }
        )

    oof = pd.concat(oof_parts, axis=0, ignore_index=True).sort_values(
        ["race_id", "horse_no"], kind="mergesort"
    )
    outputs["oof"].parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(outputs["oof"], index=False)

    final_model = _fit_model(x_all, y_all, seed=int(args.seed) + 999)
    holdout_payload: dict[str, Any] | None = None
    if holdout_input is not None and holdout_input.exists():
        holdout_base = _prep_frame(pd.read_parquet(holdout_input), label_col=None)
        holdout_base = holdout_base[holdout_base["year"] >= int(args.holdout_year)].copy()
        merged_holdout = _merge_prediction_features(holdout_base, pred_paths=raw_holdout_paths)
        for col in feature_columns:
            merged_holdout[col] = pd.to_numeric(merged_holdout[col], errors="coerce")
        merged_holdout = merged_holdout[merged_holdout[feature_columns].notna().all(axis=1)].copy()
        if not merged_holdout.empty:
            pred_holdout = _predict_meta(
                final_model, merged_holdout[feature_columns].to_numpy(dtype=float)
            )
            holdout_frame = _build_holdout_frame(
                merged_holdout,
                label_col=label_col,
                pred_col=pred_col,
                pred_values=pred_holdout,
                cv_used=str(cv_payload["used"]),
                n_splits=int(cv_payload["n_splits"]),
            )
            outputs["holdout"].parent.mkdir(parents=True, exist_ok=True)
            holdout_frame.to_parquet(outputs["holdout"], index=False)
            if label_col in merged_holdout.columns:
                metric_source = merged_holdout.assign(**{pred_col: pred_holdout})
                holdout_payload = _build_reference_summary(
                    metric_source,
                    label_col=label_col,
                    pred_col=pred_col,
                    n_bins=int(args.n_bins),
                )
                holdout_payload["years"] = sorted(
                    merged_holdout["year"].astype(int).unique().tolist()
                )
        else:
            outputs["holdout"].parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["race_id", "horse_no", pred_col]).to_parquet(
                outputs["holdout"], index=False
            )

    input_artifacts = _build_input_artifacts_payload(
        features_input=features_input,
        raw_oof_inputs=raw_oof_paths,
        holdout_input=holdout_input,
        raw_holdout_inputs=raw_holdout_paths if holdout_input is not None else None,
    )
    train_summary = {
        "rows": int(len(eligible)),
        "races": int(eligible["race_id"].nunique()),
        "years": sorted(eligible["year"].astype(int).unique().tolist()),
        "base_rate": float(np.mean(eligible[label_col].to_numpy(dtype=int))),
    }
    metrics_summary = {
        "logloss": _summary_stats([item.get("logloss") for item in fold_metrics]),
        "brier": _summary_stats([item.get("brier") for item in fold_metrics]),
        "auc": _summary_stats([item.get("auc") for item in fold_metrics]),
        "ece": _summary_stats([item.get("ece") for item in fold_metrics]),
    }
    metrics_payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": task,
        "label_col": label_col,
        "pred_col": pred_col,
        "feature_columns": feature_columns,
        "cv_strategy": str(cv_payload["used"]),
        "cv_strategy_requested": str(cv_payload["requested"]),
        "cv_is_temporal": False,
        "group_key": "race_id",
        "n_splits": int(cv_payload["n_splits"]),
        "metrics": metrics_summary,
        "config": {
            "holdout_year": int(args.holdout_year),
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "n_bins": int(args.n_bins),
            "model": {
                "class": "LogisticRegression",
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 3000,
                "random_state": int(args.seed),
            },
        },
        "train_summary": train_summary,
        "input_artifacts": input_artifacts,
        "meta_oof_is_strict_temporal": False,
        "meta_metrics_are_reference_only": True,
        "data_summary": {
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
        },
        "folds": fold_metrics,
    }
    if holdout_payload is not None:
        metrics_payload["holdout_summary"] = holdout_payload
    save_json(outputs["metrics"], metrics_payload)

    artifact = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_type": "logreg_meta_v3",
        "task": task,
        "label_col": label_col,
        "pred_col": pred_col,
        "feature_columns": list(feature_columns),
        "model": final_model,
        "coef": np.asarray(final_model.coef_, dtype=float).tolist(),
        "intercept": np.asarray(final_model.intercept_, dtype=float).tolist(),
        "preprocess": {"type": "identity"},
        "config": metrics_payload["config"],
        "train_summary": train_summary,
        "input_artifacts": input_artifacts,
        "cv_strategy": str(cv_payload["used"]),
        "cv_is_temporal": False,
        "group_key": "race_id",
        "n_splits": int(cv_payload["n_splits"]),
        "meta_oof_is_strict_temporal": False,
        "meta_metrics_are_reference_only": True,
    }
    outputs["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, outputs["model"])

    meta_payload: dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": task,
        "label_col": label_col,
        "pred_col": pred_col,
        "feature_columns": feature_columns,
        "cv_strategy": str(cv_payload["used"]),
        "cv_strategy_requested": str(cv_payload["requested"]),
        "cv_is_temporal": False,
        "group_key": "race_id",
        "n_splits": int(cv_payload["n_splits"]),
        "metrics": metrics_summary,
        "config": metrics_payload["config"],
        "train_summary": train_summary,
        "input_artifacts": input_artifacts,
        "meta_oof_is_strict_temporal": False,
        "meta_metrics_are_reference_only": True,
        "output_paths": {
            "oof": str(outputs["oof"]),
            "holdout": str(outputs["holdout"]) if holdout_input is not None else None,
            "metrics": str(outputs["metrics"]),
            "model": str(outputs["model"]),
        },
        "code_hash": hash_files([Path(__file__)]),
    }
    if holdout_payload is not None:
        meta_payload["holdout_summary"] = holdout_payload
    save_json(outputs["meta"], meta_payload)

    logger.info("wrote %s", outputs["oof"])
    if holdout_input is not None:
        logger.info("wrote %s", outputs["holdout"])
    logger.info("wrote %s", outputs["metrics"])
    logger.info("wrote %s", outputs["model"])
    logger.info("wrote %s", outputs["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
