#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
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
    _group_sizes,
    _init_wandb_run,
    _prepare_dataframe,
    build_rolling_year_folds,
)

logger = logging.getLogger(__name__)

ENTITY_ID_FEATURES = {"jockey_key", "trainer_key"}

FEATURE_SET_CHOICES = ("base", "te")
OBJECTIVE_CHOICES = ("rank:ndcg", "rank:pairwise")

LR_RANGE = (0.01, 0.20)
MAX_DEPTH_RANGE = (2, 10)
MIN_CHILD_WEIGHT_RANGE = (0.1, 100.0)
GAMMA_RANGE = (1e-8, 10.0)
SUBSAMPLE_RANGE = (0.6, 1.0)
COLSAMPLE_BYTREE_RANGE = (0.6, 1.0)
REG_LAMBDA_RANGE = (1e-3, 1000.0)
REG_ALPHA_RANGE = (1e-3, 10.0)


@dataclass(frozen=True)
class TrialResult:
    trial_number: int
    state: str
    value_mean_ndcg_at_3: float | None
    feature_set: str
    params: dict[str, Any]
    fold_ndcg_at_3: dict[int, float]
    fold_best_iteration: dict[int, int]


def drop_entity_id_features(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c not in ENTITY_ID_FEATURES]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune v2 ranker (XGBoost) with Optuna on rolling CV (year-wise)."
    )
    parser.add_argument("--input-base", default="data/features_v2.parquet")
    parser.add_argument("--input-te", default="data/features_v2_te.parquet")
    parser.add_argument("--trials-output", default="data/oof/ranker_xgb_optuna_trials.parquet")
    parser.add_argument("--best-output", default="data/oof/ranker_xgb_optuna_best.json")
    parser.add_argument(
        "--best-params-output", default="data/oof/ranker_xgb_optuna_best_params.json"
    )
    parser.add_argument("--storage", default="data/optuna/ranker_xgb_optuna.sqlite3")
    parser.add_argument("--study-name", default="ranker_xgb_v2")
    parser.add_argument("--n-trials", type=int, default=300, help="Target total trial count.")
    parser.add_argument(
        "--timeout", type=int, default=0, help="Optional timeout seconds (0=disabled)."
    )
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
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


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def suggest_feature_set(trial) -> str:
    return str(trial.suggest_categorical("feature_set", list(FEATURE_SET_CHOICES)))


def suggest_xgb_params(trial) -> dict[str, Any]:
    return {
        "objective": str(trial.suggest_categorical("objective", list(OBJECTIVE_CHOICES))),
        "learning_rate": float(trial.suggest_float("learning_rate", *LR_RANGE, log=True)),
        "max_depth": int(trial.suggest_int("max_depth", *MAX_DEPTH_RANGE)),
        "min_child_weight": float(
            trial.suggest_float("min_child_weight", *MIN_CHILD_WEIGHT_RANGE, log=True)
        ),
        "gamma": float(trial.suggest_float("gamma", *GAMMA_RANGE, log=True)),
        "subsample": float(trial.suggest_float("subsample", *SUBSAMPLE_RANGE)),
        "colsample_bytree": float(trial.suggest_float("colsample_bytree", *COLSAMPLE_BYTREE_RANGE)),
        "reg_lambda": float(trial.suggest_float("reg_lambda", *REG_LAMBDA_RANGE, log=True)),
        "reg_alpha": float(trial.suggest_float("reg_alpha", *REG_ALPHA_RANGE, log=True)),
    }


def _trial_results_from_study(study) -> list[TrialResult]:
    results: list[TrialResult] = []
    for trial in study.trials:
        fold_ndcg = trial.user_attrs.get("fold_ndcg_at_3", {})
        fold_best_iter = trial.user_attrs.get("fold_best_iteration", {})
        results.append(
            TrialResult(
                trial_number=int(trial.number),
                state=str(trial.state.name),
                value_mean_ndcg_at_3=None if trial.value is None else float(trial.value),
                feature_set=str(trial.params.get("feature_set", "")),
                params={k: v for k, v in trial.params.items() if k != "feature_set"},
                fold_ndcg_at_3={int(k): float(v) for k, v in (fold_ndcg or {}).items()},
                fold_best_iteration={int(k): int(v) for k, v in (fold_best_iter or {}).items()},
            )
        )
    return results


def _trial_results_to_frame(results: list[TrialResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for res in results:
        row: dict[str, Any] = {
            "trial_number": int(res.trial_number),
            "state": res.state,
            "value_mean_ndcg_at_3": res.value_mean_ndcg_at_3,
            "feature_set": res.feature_set,
        }
        for key, value in res.params.items():
            row[f"param/{key}"] = value
        for fold_id, value in res.fold_ndcg_at_3.items():
            row[f"fold/{fold_id}/ndcg_at_3"] = value
        for fold_id, value in res.fold_best_iteration.items():
            row[f"fold/{fold_id}/best_iteration"] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trial_number"], kind="mergesort")


def _xgb_best_iteration(model, *, fallback: int) -> int:
    best = getattr(model, "best_iteration", None)
    if best is None:
        best = getattr(model, "best_iteration_", None)
    if best is None:
        best = getattr(getattr(model, "get_booster", lambda: None)(), "best_iteration", None)
    if best is None:
        best = fallback - 1
    return int(best) + 1  # 1-based


def _xgb_predict(model, X: pd.DataFrame, *, best_iteration: int) -> np.ndarray:
    # Prefer iteration_range (xgboost>=1.6). Fall back to ntree_limit.
    try:
        return model.predict(X, iteration_range=(0, int(best_iteration)))
    except TypeError:
        return model.predict(X, ntree_limit=int(best_iteration))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("optuna is not installed. Run `uv sync --extra optuna`.") from exc
    try:
        from xgboost import XGBRanker
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("xgboost is not installed. Run `uv sync --extra xgboost`.") from exc

    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be > 0")
    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if args.num_boost_round <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if args.train_window_years <= 0:
        raise SystemExit("--train-window-years must be > 0")
    if args.timeout < 0:
        raise SystemExit("--timeout must be >= 0")

    input_base_path = _resolve_path(args.input_base)
    input_te_path = _resolve_path(args.input_te)
    trials_output = _resolve_path(args.trials_output)
    best_output = _resolve_path(args.best_output)
    best_params_output = _resolve_path(args.best_params_output)
    storage_path = _resolve_path(args.storage)

    if not input_base_path.exists():
        raise SystemExit(f"input-base not found: {input_base_path}")
    if not input_te_path.exists():
        raise SystemExit(f"input-te not found: {input_te_path}")

    logger.info("loading base features: %s", input_base_path)
    base_frame = _prepare_dataframe(pd.read_parquet(input_base_path))
    base_frame = base_frame[base_frame["year"] < int(args.holdout_year)].copy()

    logger.info("loading te features: %s", input_te_path)
    te_frame = _prepare_dataframe(pd.read_parquet(input_te_path))
    te_frame = te_frame[te_frame["year"] < int(args.holdout_year)].copy()

    years_base = sorted(base_frame["year"].unique().tolist())
    years_te = sorted(te_frame["year"].unique().tolist())
    if years_base != years_te:
        raise SystemExit(
            f"year mismatch between base and te inputs: base={years_base} te={years_te}"
        )

    folds: list[FoldSpec] = build_rolling_year_folds(
        years_base,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    valid_years = [int(fold.valid_year) for fold in folds]
    logger.info(
        "optuna tuning folds=%s valid_years=%s window=%s holdout_year>=%s",
        len(folds),
        valid_years,
        args.train_window_years,
        args.holdout_year,
    )

    base_feature_cols = drop_entity_id_features(_feature_columns(base_frame))
    te_feature_cols = drop_entity_id_features(_feature_columns(te_frame))
    if any(col in base_feature_cols for col in ENTITY_ID_FEATURES):
        raise SystemExit("entity ID features unexpectedly present in base feature columns.")
    if any(col in te_feature_cols for col in ENTITY_ID_FEATURES):
        raise SystemExit("entity ID features unexpectedly present in te feature columns.")

    wandb_run = _init_wandb_run(
        args,
        config={
            "model": "xgboost",
            "eval_at": [3],
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "n_trials_target": int(args.n_trials),
            "drop_entity_id_features": True,
            "seed": int(args.seed),
        },
    )
    if wandb_run is not None:
        wandb_run.define_metric("optuna/trial")
        wandb_run.define_metric("optuna/value_mean_ndcg_at_3", step_metric="optuna/trial")

    def objective(trial) -> float:
        feature_set = suggest_feature_set(trial)
        xgb_params = suggest_xgb_params(trial)

        if feature_set == "te":
            frame = te_frame
            feature_cols = te_feature_cols
            input_path = str(args.input_te)
        else:
            frame = base_frame
            feature_cols = base_feature_cols
            input_path = str(args.input_base)

        params = {
            "n_estimators": int(args.num_boost_round),
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            "eval_metric": "ndcg@3",
            "early_stopping_rounds": int(args.early_stopping_rounds),
            **xgb_params,
        }

        fold_ndcg: dict[int, float] = {}
        fold_best_iter: dict[int, int] = {}
        values: list[float] = []
        for fold in folds:
            train_df = frame[frame["year"].isin(fold.train_years)].copy()
            valid_df = frame[frame["year"] == fold.valid_year].copy()
            if train_df.empty or valid_df.empty:
                raise RuntimeError(
                    f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
                )
            train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
                drop=True
            )
            valid_df = valid_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
                drop=True
            )
            _assert_fold_integrity(train_df, valid_df, fold.valid_year)

            X_train = _coerce_feature_matrix(train_df, feature_cols)
            y_train = train_df["target_label"].astype(int)
            g_train = _group_sizes(train_df)

            X_valid = _coerce_feature_matrix(valid_df, feature_cols)
            y_valid = valid_df["target_label"].astype(int)
            g_valid = _group_sizes(valid_df)

            fold_params = dict(params)
            fold_params["random_state"] = int(args.seed) + int(fold.fold_id)
            model = XGBRanker(**fold_params)
            model.fit(
                X_train,
                y_train,
                group=g_train,
                eval_set=[(X_valid, y_valid)],
                eval_group=[g_valid],
                verbose=False,
            )

            best_iteration = _xgb_best_iteration(model, fallback=int(args.num_boost_round))
            valid_score = _xgb_predict(model, X_valid, best_iteration=best_iteration)
            eval_frame = valid_df[["race_id", "target_label"]].copy()
            eval_frame["ranker_score"] = valid_score
            ndcg3 = float(_fold_ndcg_at_3(eval_frame))

            fold_id = int(fold.fold_id)
            fold_ndcg[fold_id] = ndcg3
            fold_best_iter[fold_id] = best_iteration
            values.append(ndcg3)

            running_mean = float(np.mean(values))
            trial.report(running_mean, step=fold_id)
            if trial.should_prune():
                trial.set_user_attr("input_path", input_path)
                trial.set_user_attr("fold_ndcg_at_3", fold_ndcg)
                trial.set_user_attr("fold_best_iteration", fold_best_iter)
                raise optuna.TrialPruned()

        mean_value = float(np.mean(values))
        trial.set_user_attr("input_path", input_path)
        trial.set_user_attr("fold_ndcg_at_3", fold_ndcg)
        trial.set_user_attr("fold_best_iteration", fold_best_iter)

        if wandb_run is not None:
            log_payload: dict[str, Any] = {
                "optuna/trial": int(trial.number),
                "optuna/value_mean_ndcg_at_3": float(mean_value),
                "optuna/feature_set_is_te": 1 if feature_set == "te" else 0,
            }
            for fold_id in sorted(fold_ndcg):
                log_payload[f"optuna/fold{fold_id}_ndcg_at_3"] = float(fold_ndcg[fold_id])
                log_payload[f"optuna/fold{fold_id}_best_iteration"] = int(fold_best_iter[fold_id])
            for key, value in xgb_params.items():
                log_payload[f"optuna/params/{key}"] = value
            wandb_run.log(log_payload, step=int(trial.number))
        return mean_value

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(
        direction="maximize",
        study_name=str(args.study_name),
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    existing_trials = len(study.trials)
    additional_trials = max(0, int(args.n_trials) - existing_trials)
    logger.info(
        "study=%s storage=%s existing_trials=%s target_trials=%s additional=%s",
        study.study_name,
        storage_path,
        existing_trials,
        args.n_trials,
        additional_trials,
    )
    timeout = None if int(args.timeout) == 0 else int(args.timeout)
    if additional_trials > 0:
        study.optimize(objective, n_trials=additional_trials, timeout=timeout, catch=(Exception,))

    results = _trial_results_from_study(study)
    trials_frame = _trial_results_to_frame(results)
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    trials_frame.to_parquet(trials_output, index=False)
    logger.info("wrote %s rows=%s", trials_output, len(trials_frame))

    best_trial = study.best_trial
    best_feature_set = str(best_trial.params.get("feature_set", ""))
    best_input_path = str(best_trial.user_attrs.get("input_path", ""))
    best_summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "study_name": str(study.study_name),
        "storage": str(storage_path),
        "direction": "maximize",
        "target_trials": int(args.n_trials),
        "total_trials": int(len(study.trials)),
        "best_trial_number": int(best_trial.number),
        "best_value_mean_ndcg_at_3": float(study.best_value),
        "best_feature_set": best_feature_set,
        "best_input": best_input_path,
        "best_params": best_trial.params,
        "best_fold_ndcg_at_3": best_trial.user_attrs.get("fold_ndcg_at_3", {}),
        "best_fold_best_iteration": best_trial.user_attrs.get("fold_best_iteration", {}),
        "fixed_config": {
            "eval_at": [3],
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "drop_entity_id_features": True,
        },
    }
    _save_json(best_output, best_summary)
    logger.info("wrote %s", best_output)

    best_params = {
        "input": best_input_path
        or (str(args.input_te) if best_feature_set == "te" else str(args.input_base)),
        "drop_entity_id_features": True,
        "xgb_params": {k: v for k, v in best_trial.params.items() if k != "feature_set"},
    }
    _save_json(best_params_output, best_params)
    logger.info("wrote %s", best_params_output)

    if wandb_run is not None:
        wandb_run.summary["best_trial_number"] = int(best_trial.number)
        wandb_run.summary["best_value_mean_ndcg_at_3"] = float(study.best_value)
        wandb_run.summary["best_feature_set"] = best_feature_set
        wandb_run.summary["best_input"] = best_input_path
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
