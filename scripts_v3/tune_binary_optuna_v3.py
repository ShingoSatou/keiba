#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.train_binary_model_v3 import (  # noqa: E402
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HOLDOUT_YEAR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_SEED,
    MODEL_CHOICES,
    TASK_CHOICES,
    _evaluate_cv_fold,
    _label_col,
    _pred_col,
    _resolve_binary_feature_columns,
)
from scripts_v3.train_binary_model_v3 import (
    parse_args as parse_binary_train_args,
)
from scripts_v3.train_binary_v3_common import (  # noqa: E402
    DEFAULT_CV_WINDOW_POLICY,
    build_fixed_window_year_folds,
    prepare_binary_frame,
    resolve_path,
    save_json,
)

logger = logging.getLogger(__name__)

FEATURE_SET_CHOICES = ("base", "te")

FIXED_TRAIN_WINDOW_YEARS = 4
FIXED_OPERATIONAL_MODE = "t10_only"
FIXED_INCLUDE_ENTITY_ID_FEATURES = False

LGBM_LR_RANGE = (0.01, 0.10)
LGBM_NUM_LEAVES_RANGE = (15, 255)
LGBM_MIN_DATA_IN_LEAF_RANGE = (20, 1000)
LGBM_LAMBDA_L1_RANGE = (1e-3, 10.0)
LGBM_LAMBDA_L2_RANGE = (1e-3, 1000.0)
LGBM_FEATURE_FRACTION_RANGE = (0.6, 1.0)
LGBM_BAGGING_FRACTION_RANGE = (0.6, 1.0)
LGBM_BAGGING_FREQ_RANGE = (0, 10)

XGB_LR_RANGE = (0.01, 0.20)
XGB_MAX_DEPTH_RANGE = (2, 10)
XGB_MIN_CHILD_WEIGHT_RANGE = (0.1, 100.0)
XGB_GAMMA_RANGE = (1e-8, 10.0)
XGB_SUBSAMPLE_RANGE = (0.6, 1.0)
XGB_COLSAMPLE_BYTREE_RANGE = (0.6, 1.0)
XGB_REG_ALPHA_RANGE = (1e-3, 10.0)
XGB_REG_LAMBDA_RANGE = (1e-3, 1000.0)

CAT_LR_RANGE = (0.01, 0.20)
CAT_DEPTH_RANGE = (4, 10)
CAT_L2_LEAF_REG_RANGE = (1e-3, 1000.0)
CAT_RANDOM_STRENGTH_RANGE = (0.0, 5.0)
CAT_BAGGING_TEMPERATURE_RANGE = (0.0, 10.0)
CAT_RSM_RANGE = (0.6, 1.0)
CAT_MIN_DATA_IN_LEAF_RANGE = (1, 1000)

WIN_BENTER_MEDIAN_TOLERANCE = 0.002
WIN_BENTER_MIN_TOLERANCE = 0.01

MODEL_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "lgbm": (
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "lambda_l1",
        "lambda_l2",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
    ),
    "xgb": (
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ),
    "cat": (
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "bagging_temperature",
        "rsm",
        "min_data_in_leaf",
    ),
}


@dataclass(frozen=True)
class CvEvalResult:
    value_mean_logloss: float
    fold_logloss: dict[int, float]
    fold_best_iteration: dict[int, int]
    fold_metrics: list[dict[str, Any]]
    fold_benter_r2_valid: dict[int, float] | None
    benter_r2_valid_mean: float | None
    benter_r2_valid_median: float | None
    benter_r2_valid_min: float | None


@dataclass(frozen=True)
class TrialResult:
    trial_number: int
    state: str
    value_mean_logloss: float | None
    feature_set: str
    input_path: str
    params: dict[str, Any]
    fold_logloss: dict[int, float]
    fold_best_iteration: dict[int, int]
    fold_benter_r2_valid: dict[int, float] | None
    benter_r2_valid_mean: float | None
    benter_r2_valid_median: float | None
    benter_r2_valid_min: float | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune v3 binary models with Optuna on fixed-sliding yearly CV. "
            "Fixed contract: train_window_years=4, operational_mode=t10_only, "
            "include_entity_id_features=false."
        )
    )
    parser.add_argument("--task", choices=list(TASK_CHOICES), default="win")
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default="lgbm")
    parser.add_argument("--input-base", default="data/features_v3.parquet")
    parser.add_argument("--input-te", default="data/features_v3_te.parquet")
    parser.add_argument("--trials-output", default="")
    parser.add_argument("--best-output", default="")
    parser.add_argument("--best-params-output", default="")
    parser.add_argument("--storage", default="")
    parser.add_argument("--study-name", default="")
    parser.add_argument("--n-trials", type=int, default=300, help="Target total trial count.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional timeout seconds (0=disabled).",
    )
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--benter-eps", type=float, default=1e-6)
    parser.add_argument("--benter-beta-min", type=float, default=0.01)
    parser.add_argument("--benter-beta-max", type=float, default=100.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.n_trials) <= 0:
        raise SystemExit("--n-trials must be > 0")
    if int(args.timeout) < 0:
        raise SystemExit("--timeout must be >= 0")
    if int(args.num_boost_round) <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if int(args.early_stopping_rounds) <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if int(args.n_bins) <= 1:
        raise SystemExit("--n-bins must be > 1")
    if not (0.0 < float(args.benter_eps) < 0.5):
        raise SystemExit("--benter-eps must be in (0, 0.5)")
    if not (0.0 < float(args.benter_beta_min) < float(args.benter_beta_max)):
        raise SystemExit("--benter-beta-min must be > 0 and < --benter-beta-max")


def _default_study_name(task: str, model: str) -> str:
    return f"binary_v3_{task}_{model}"


def _resolve_output_paths(args: argparse.Namespace, *, study_name: str) -> dict[str, Path]:
    defaults = {
        "trials": f"data/oof/{study_name}_trials.parquet",
        "best": f"data/oof/{study_name}_best.json",
        "best_params": f"data/oof/{study_name}_best_params.json",
        "storage": f"data/optuna/{study_name}.sqlite3",
    }
    return {
        "trials": resolve_path(args.trials_output or defaults["trials"]),
        "best": resolve_path(args.best_output or defaults["best"]),
        "best_params": resolve_path(args.best_params_output or defaults["best_params"]),
        "storage": resolve_path(args.storage or defaults["storage"]),
    }


def _fixed_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
        "train_window_years": FIXED_TRAIN_WINDOW_YEARS,
        "operational_mode": FIXED_OPERATIONAL_MODE,
        "include_entity_id_features": FIXED_INCLUDE_ENTITY_ID_FEATURES,
        "num_boost_round": int(args.num_boost_round),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "n_bins": int(args.n_bins),
        "seed": int(args.seed),
        "benter_eps": float(args.benter_eps),
        "benter_beta_min": float(args.benter_beta_min),
        "benter_beta_max": float(args.benter_beta_max),
    }


def _study_config_payload(
    args: argparse.Namespace,
    *,
    input_base_path: Path,
    input_te_path: Path,
) -> dict[str, Any]:
    return {
        "task": str(args.task),
        "model": str(args.model),
        "holdout_year": int(args.holdout_year),
        "input_base": str(input_base_path),
        "input_te": str(input_te_path),
        "fixed_config": _fixed_config_payload(args),
    }


def _assert_study_compatible(study, *, expected_config: dict[str, Any]) -> None:
    existing = study.user_attrs.get("study_config")
    if existing is None:
        if study.trials:
            raise SystemExit(
                "Existing study has trials but no stored study_config. "
                "Use a new study name or clear the storage."
            )
        study.set_user_attr("study_config", expected_config)
        return
    if existing != expected_config:
        raise SystemExit(
            "Study configuration mismatch for resume. "
            f"expected={expected_config} existing={existing}"
        )


def _trial_input_display(args: argparse.Namespace, *, feature_set: str) -> str:
    return str(args.input_te) if feature_set == "te" else str(args.input_base)


def _build_model_defaults(base_train_args: argparse.Namespace, *, model: str) -> dict[str, Any]:
    return {key: getattr(base_train_args, key) for key in MODEL_PARAM_KEYS[model]}


def _build_train_args(
    *,
    base_train_args: argparse.Namespace,
    tune_args: argparse.Namespace,
    params: dict[str, Any],
) -> argparse.Namespace:
    out = argparse.Namespace(**vars(base_train_args))
    out.task = str(tune_args.task)
    out.model = str(tune_args.model)
    out.holdout_year = int(tune_args.holdout_year)
    out.train_window_years = FIXED_TRAIN_WINDOW_YEARS
    out.num_boost_round = int(tune_args.num_boost_round)
    out.early_stopping_rounds = int(tune_args.early_stopping_rounds)
    out.seed = int(tune_args.seed)
    out.n_bins = int(tune_args.n_bins)
    out.operational_mode = FIXED_OPERATIONAL_MODE
    out.include_entity_id_features = FIXED_INCLUDE_ENTITY_ID_FEATURES
    out.drop_entity_id_features = False
    out.benter_eps = float(tune_args.benter_eps)
    out.benter_beta_min = float(tune_args.benter_beta_min)
    out.benter_beta_max = float(tune_args.benter_beta_max)
    for key, value in params.items():
        setattr(out, key, value)
    return out


def suggest_feature_set(trial) -> str:
    return str(trial.suggest_categorical("feature_set", list(FEATURE_SET_CHOICES)))


def suggest_model_params(trial, *, model: str) -> dict[str, Any]:
    if model == "lgbm":
        return {
            "learning_rate": float(trial.suggest_float("learning_rate", *LGBM_LR_RANGE, log=True)),
            "num_leaves": int(trial.suggest_int("num_leaves", *LGBM_NUM_LEAVES_RANGE)),
            "min_data_in_leaf": int(
                trial.suggest_int("min_data_in_leaf", *LGBM_MIN_DATA_IN_LEAF_RANGE)
            ),
            "lambda_l1": float(trial.suggest_float("lambda_l1", *LGBM_LAMBDA_L1_RANGE, log=True)),
            "lambda_l2": float(trial.suggest_float("lambda_l2", *LGBM_LAMBDA_L2_RANGE, log=True)),
            "feature_fraction": float(
                trial.suggest_float("feature_fraction", *LGBM_FEATURE_FRACTION_RANGE)
            ),
            "bagging_fraction": float(
                trial.suggest_float("bagging_fraction", *LGBM_BAGGING_FRACTION_RANGE)
            ),
            "bagging_freq": int(trial.suggest_int("bagging_freq", *LGBM_BAGGING_FREQ_RANGE)),
        }
    if model == "xgb":
        return {
            "learning_rate": float(trial.suggest_float("learning_rate", *XGB_LR_RANGE, log=True)),
            "max_depth": int(trial.suggest_int("max_depth", *XGB_MAX_DEPTH_RANGE)),
            "min_child_weight": float(
                trial.suggest_float("min_child_weight", *XGB_MIN_CHILD_WEIGHT_RANGE, log=True)
            ),
            "gamma": float(trial.suggest_float("gamma", *XGB_GAMMA_RANGE, log=True)),
            "subsample": float(trial.suggest_float("subsample", *XGB_SUBSAMPLE_RANGE)),
            "colsample_bytree": float(
                trial.suggest_float("colsample_bytree", *XGB_COLSAMPLE_BYTREE_RANGE)
            ),
            "reg_alpha": float(trial.suggest_float("reg_alpha", *XGB_REG_ALPHA_RANGE, log=True)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", *XGB_REG_LAMBDA_RANGE, log=True)),
        }
    if model == "cat":
        return {
            "learning_rate": float(trial.suggest_float("learning_rate", *CAT_LR_RANGE, log=True)),
            "depth": int(trial.suggest_int("depth", *CAT_DEPTH_RANGE)),
            "l2_leaf_reg": float(
                trial.suggest_float("l2_leaf_reg", *CAT_L2_LEAF_REG_RANGE, log=True)
            ),
            "random_strength": float(
                trial.suggest_float("random_strength", *CAT_RANDOM_STRENGTH_RANGE)
            ),
            "bagging_temperature": float(
                trial.suggest_float("bagging_temperature", *CAT_BAGGING_TEMPERATURE_RANGE)
            ),
            "rsm": float(trial.suggest_float("rsm", *CAT_RSM_RANGE)),
            "min_data_in_leaf": int(
                trial.suggest_int("min_data_in_leaf", *CAT_MIN_DATA_IN_LEAF_RANGE)
            ),
        }
    raise ValueError(f"Unknown model: {model}")


def _summary_with_median(values: list[float | None]) -> dict[str, float | None]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _benter_summary_from_fold_values(
    fold_benter_r2_valid: dict[int, float],
) -> tuple[float | None, float | None, float | None]:
    finite = [float(value) for value in fold_benter_r2_valid.values() if np.isfinite(value)]
    if not finite:
        return None, None, None
    arr = np.asarray(finite, dtype=float)
    return float(np.mean(arr)), float(np.median(arr)), float(np.min(arr))


def _set_trial_user_attrs(
    trial,
    *,
    feature_set: str,
    input_path: str,
    fold_logloss: dict[int, float],
    fold_best_iteration: dict[int, int],
    fold_benter_r2_valid: dict[int, float] | None,
) -> None:
    trial.set_user_attr("input_path", str(input_path))
    trial.set_user_attr("feature_set", str(feature_set))
    trial.set_user_attr(
        "fold_logloss",
        {int(fold_id): float(value) for fold_id, value in fold_logloss.items()},
    )
    trial.set_user_attr(
        "fold_best_iteration",
        {int(fold_id): int(value) for fold_id, value in fold_best_iteration.items()},
    )
    if fold_benter_r2_valid is not None:
        mean_value, median_value, min_value = _benter_summary_from_fold_values(fold_benter_r2_valid)
        trial.set_user_attr(
            "fold_benter_r2_valid",
            {int(fold_id): float(value) for fold_id, value in fold_benter_r2_valid.items()},
        )
        trial.set_user_attr("benter_r2_valid_mean", mean_value)
        trial.set_user_attr("benter_r2_valid_median", median_value)
        trial.set_user_attr("benter_r2_valid_min", min_value)


def _evaluate_binary_cv(
    *,
    frame: pd.DataFrame,
    folds: list,
    feat_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    pred_col: str,
    train_args: argparse.Namespace,
    progress_callback: Callable[
        [int, float, dict[int, float], dict[int, int], dict[int, float] | None],
        None,
    ]
    | None = None,
) -> CvEvalResult:
    fold_metrics: list[dict[str, Any]] = []
    fold_logloss: dict[int, float] = {}
    fold_best_iteration: dict[int, int] = {}
    fold_benter_r2_valid: dict[int, float] | None = {} if str(train_args.task) == "win" else None
    running_logloss: list[float] = []

    for fold in folds:
        fold_metric, _ = _evaluate_cv_fold(
            frame=frame,
            fold=fold,
            feat_cols=feat_cols,
            categorical_cols=categorical_cols,
            label_col=label_col,
            pred_col=pred_col,
            args=train_args,
            emit_oof=False,
        )
        logloss_value = fold_metric.get("logloss")
        if logloss_value is None or not np.isfinite(logloss_value):
            raise RuntimeError(
                f"fold={fold.fold_id} produced invalid logloss for task={train_args.task}"
            )

        fold_id = int(fold_metric["fold_id"])
        fold_logloss[fold_id] = float(logloss_value)
        fold_best_iteration[fold_id] = int(fold_metric["best_iteration"])
        if fold_benter_r2_valid is not None and isinstance(fold_metric.get("benter"), dict):
            benter_value = (fold_metric["benter"] or {}).get("benter_r2_valid")
            if benter_value is not None and np.isfinite(benter_value):
                fold_benter_r2_valid[fold_id] = float(benter_value)

        running_logloss.append(float(logloss_value))
        fold_metrics.append(fold_metric)

        if progress_callback is not None:
            progress_callback(
                fold_id,
                float(np.mean(running_logloss)),
                dict(fold_logloss),
                dict(fold_best_iteration),
                None if fold_benter_r2_valid is None else dict(fold_benter_r2_valid),
            )

    if not running_logloss:
        raise RuntimeError("No fold results were produced during CV evaluation.")

    benter_mean, benter_median, benter_min = (None, None, None)
    if fold_benter_r2_valid is not None:
        benter_mean, benter_median, benter_min = _benter_summary_from_fold_values(
            fold_benter_r2_valid
        )

    return CvEvalResult(
        value_mean_logloss=float(np.mean(np.asarray(running_logloss, dtype=float))),
        fold_logloss=fold_logloss,
        fold_best_iteration=fold_best_iteration,
        fold_metrics=fold_metrics,
        fold_benter_r2_valid=fold_benter_r2_valid,
        benter_r2_valid_mean=benter_mean,
        benter_r2_valid_median=benter_median,
        benter_r2_valid_min=benter_min,
    )


def _trial_results_from_study(study) -> list[TrialResult]:
    results: list[TrialResult] = []
    for trial in study.trials:
        fold_logloss = trial.user_attrs.get("fold_logloss", {})
        fold_best_iteration = trial.user_attrs.get("fold_best_iteration", {})
        fold_benter_r2_valid = trial.user_attrs.get("fold_benter_r2_valid", {})
        results.append(
            TrialResult(
                trial_number=int(trial.number),
                state=str(trial.state.name),
                value_mean_logloss=None if trial.value is None else float(trial.value),
                feature_set=str(
                    trial.user_attrs.get("feature_set", trial.params.get("feature_set", ""))
                ),
                input_path=str(trial.user_attrs.get("input_path", "")),
                params={key: value for key, value in trial.params.items() if key != "feature_set"},
                fold_logloss={
                    int(key): float(value) for key, value in (fold_logloss or {}).items()
                },
                fold_best_iteration={
                    int(key): int(value) for key, value in (fold_best_iteration or {}).items()
                },
                fold_benter_r2_valid=(
                    None
                    if not fold_benter_r2_valid
                    else {
                        int(key): float(value)
                        for key, value in (fold_benter_r2_valid or {}).items()
                    }
                ),
                benter_r2_valid_mean=_optional_float(trial.user_attrs.get("benter_r2_valid_mean")),
                benter_r2_valid_median=_optional_float(
                    trial.user_attrs.get("benter_r2_valid_median")
                ),
                benter_r2_valid_min=_optional_float(trial.user_attrs.get("benter_r2_valid_min")),
            )
        )
    return results


def _optional_float(value: Any) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _trial_results_to_frame(results: list[TrialResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "trial_number": int(result.trial_number),
            "state": result.state,
            "value_mean_logloss": result.value_mean_logloss,
            "feature_set": result.feature_set,
            "input": result.input_path,
            "benter_r2_valid_mean": result.benter_r2_valid_mean,
            "benter_r2_valid_median": result.benter_r2_valid_median,
            "benter_r2_valid_min": result.benter_r2_valid_min,
        }
        for key, value in result.params.items():
            row[f"param/{key}"] = value
        for fold_id, value in result.fold_logloss.items():
            row[f"fold/{fold_id}/logloss"] = value
        for fold_id, value in result.fold_best_iteration.items():
            row[f"fold/{fold_id}/best_iteration"] = value
        if result.fold_benter_r2_valid is not None:
            for fold_id, value in result.fold_benter_r2_valid.items():
                row[f"fold/{fold_id}/benter_r2_valid"] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trial_number"], kind="mergesort")


def _complete_trial_results(results: list[TrialResult]) -> list[TrialResult]:
    return [
        result
        for result in results
        if result.state == "COMPLETE" and result.value_mean_logloss is not None
    ]


def _select_lowest_logloss_trial(results: list[TrialResult]) -> TrialResult:
    return min(
        results,
        key=lambda result: (float(result.value_mean_logloss), int(result.trial_number)),
    )


def select_best_trial_result(
    *,
    results: list[TrialResult],
    task: str,
    baseline_summary: dict[str, Any],
    study_best_trial_number: int | None,
) -> tuple[TrialResult, bool, str]:
    complete_results = _complete_trial_results(results)
    if not complete_results:
        raise ValueError("No COMPLETE trials available.")

    if task == "place":
        return _select_lowest_logloss_trial(complete_results), False, "min_complete"

    baseline_median = _optional_float(baseline_summary.get("benter_r2_valid_median"))
    baseline_min = _optional_float(baseline_summary.get("benter_r2_valid_min"))
    constrained: list[TrialResult] = []
    if baseline_median is not None and baseline_min is not None:
        constrained = [
            result
            for result in complete_results
            if result.benter_r2_valid_median is not None
            and result.benter_r2_valid_min is not None
            and result.benter_r2_valid_median >= baseline_median - WIN_BENTER_MEDIAN_TOLERANCE
            and result.benter_r2_valid_min >= baseline_min - WIN_BENTER_MIN_TOLERANCE
        ]
    if constrained:
        return _select_lowest_logloss_trial(constrained), True, "constrained_logloss"

    if study_best_trial_number is not None:
        for result in complete_results:
            if int(result.trial_number) == int(study_best_trial_number):
                return result, False, "study_best_fallback"

    return _select_lowest_logloss_trial(complete_results), False, "min_complete_fallback"


def _median_best_iteration(fold_best_iteration: dict[int, int], *, upper_bound: int) -> int:
    if not fold_best_iteration:
        raise ValueError("fold_best_iteration must not be empty")
    values = np.asarray(list(fold_best_iteration.values()), dtype=float)
    median_value = int(np.median(values))
    return max(1, min(median_value, int(upper_bound)))


def _model_params_output_key(model: str) -> str:
    return f"{model}_params"


def _final_iteration_output_key(model: str) -> str:
    return "final_iterations" if model == "cat" else "final_num_boost_round"


def _build_baseline_summary(
    *,
    args: argparse.Namespace,
    eval_result: CvEvalResult,
    input_path: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input": str(input_path),
        "feature_set": "base",
        "value_mean_logloss": float(eval_result.value_mean_logloss),
        "fold_logloss": eval_result.fold_logloss,
        "fold_best_iteration": eval_result.fold_best_iteration,
        "logloss_summary": _summary_with_median(list(eval_result.fold_logloss.values())),
    }
    if str(args.task) == "win":
        summary["benter_r2_valid_mean"] = eval_result.benter_r2_valid_mean
        summary["benter_r2_valid_median"] = eval_result.benter_r2_valid_median
        summary["benter_r2_valid_min"] = eval_result.benter_r2_valid_min
        summary["fold_benter_r2_valid"] = eval_result.fold_benter_r2_valid or {}
    return summary


def _build_best_output(
    *,
    args: argparse.Namespace,
    outputs: dict[str, Path],
    study_name: str,
    total_trials: int,
    selected_trial: TrialResult,
    baseline_summary: dict[str, Any],
    fixed_config: dict[str, Any],
    constraint_passed: bool,
    selection_mode: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "study_name": study_name,
        "storage": str(outputs["storage"]),
        "direction": "minimize",
        "total_trials": int(total_trials),
        "best_trial_number": int(selected_trial.trial_number),
        "best_feature_set": selected_trial.feature_set,
        "best_input": selected_trial.input_path,
        "best_value_mean_logloss": float(selected_trial.value_mean_logloss),
        "best_fold_logloss": selected_trial.fold_logloss,
        "best_fold_best_iteration": selected_trial.fold_best_iteration,
        "best_params": {
            "feature_set": selected_trial.feature_set,
            **selected_trial.params,
        },
        "fixed_config": fixed_config,
        "task": str(args.task),
        "model": str(args.model),
        "operational_mode": FIXED_OPERATIONAL_MODE,
        "include_entity_id_features": FIXED_INCLUDE_ENTITY_ID_FEATURES,
        "train_window_years": FIXED_TRAIN_WINDOW_YEARS,
        "holdout_year": int(args.holdout_year),
        "baseline_summary": baseline_summary,
        "selection_mode": selection_mode,
    }
    if str(args.task) == "win":
        payload["constraint_passed"] = bool(constraint_passed)
        payload["best_benter_summary"] = {
            "benter_r2_valid_mean": selected_trial.benter_r2_valid_mean,
            "benter_r2_valid_median": selected_trial.benter_r2_valid_median,
            "benter_r2_valid_min": selected_trial.benter_r2_valid_min,
        }
    return payload


def _build_best_params_output(
    *,
    args: argparse.Namespace,
    selected_trial: TrialResult,
) -> dict[str, Any]:
    final_iterations = _median_best_iteration(
        selected_trial.fold_best_iteration,
        upper_bound=int(args.num_boost_round),
    )
    payload: dict[str, Any] = {
        "input": selected_trial.input_path,
        "feature_set": selected_trial.feature_set,
        "task": str(args.task),
        "model": str(args.model),
        "operational_mode": FIXED_OPERATIONAL_MODE,
        "include_entity_id_features": FIXED_INCLUDE_ENTITY_ID_FEATURES,
        "train_window_years": FIXED_TRAIN_WINDOW_YEARS,
        _model_params_output_key(str(args.model)): dict(selected_trial.params),
        _final_iteration_output_key(str(args.model)): int(final_iterations),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    _validate_args(args)

    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("optuna is not installed. Run `uv sync --extra optuna`.") from exc

    study_name = str(args.study_name or _default_study_name(str(args.task), str(args.model)))
    outputs = _resolve_output_paths(args, study_name=study_name)

    input_base_path = resolve_path(args.input_base)
    input_te_path = resolve_path(args.input_te)
    if not input_base_path.exists():
        raise SystemExit(f"input-base not found: {input_base_path}")
    if not input_te_path.exists():
        raise SystemExit(f"input-te not found: {input_te_path}")

    label_col = _label_col(str(args.task))
    pred_col = _pred_col(str(args.task), str(args.model))
    fixed_config = _fixed_config_payload(args)
    base_train_args = parse_binary_train_args(
        ["--task", str(args.task), "--model", str(args.model)]
    )

    logger.info("loading base features: %s", input_base_path)
    base_frame = prepare_binary_frame(pd.read_parquet(input_base_path), label_col=label_col)
    base_frame = base_frame[base_frame["year"] < int(args.holdout_year)].copy()

    logger.info("loading te features: %s", input_te_path)
    te_frame = prepare_binary_frame(pd.read_parquet(input_te_path), label_col=label_col)
    te_frame = te_frame[te_frame["year"] < int(args.holdout_year)].copy()

    if base_frame.empty or te_frame.empty:
        raise SystemExit("No trainable rows found after holdout exclusion.")

    years_base = sorted(base_frame["year"].unique().tolist())
    years_te = sorted(te_frame["year"].unique().tolist())
    if years_base != years_te:
        raise SystemExit(
            f"year mismatch between base and te inputs: base={years_base} te={years_te}"
        )

    folds = build_fixed_window_year_folds(
        years_base,
        window_years=FIXED_TRAIN_WINDOW_YEARS,
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        raise SystemExit(
            "No binary tuning folds available under fixed_sliding "
            f"window={FIXED_TRAIN_WINDOW_YEARS} for years={years_base}"
        )

    base_feat_cols, base_categorical_cols, _ = _resolve_binary_feature_columns(
        frame=base_frame,
        input_path=input_base_path,
        include_entity_id_features=FIXED_INCLUDE_ENTITY_ID_FEATURES,
        operational_mode=FIXED_OPERATIONAL_MODE,
        feature_set_override="base",
    )
    te_feat_cols, te_categorical_cols, _ = _resolve_binary_feature_columns(
        frame=te_frame,
        input_path=input_te_path,
        include_entity_id_features=FIXED_INCLUDE_ENTITY_ID_FEATURES,
        operational_mode=FIXED_OPERATIONAL_MODE,
        feature_set_override="te",
    )

    frames_by_feature_set = {
        "base": base_frame,
        "te": te_frame,
    }
    feature_cols_by_feature_set = {
        "base": base_feat_cols,
        "te": te_feat_cols,
    }
    categorical_cols_by_feature_set = {
        "base": base_categorical_cols,
        "te": te_categorical_cols,
    }

    logger.info(
        (
            "binary optuna task=%s model=%s years=%s folds=%s "
            "valid_years=%s base_features=%s te_features=%s"
        ),
        args.task,
        args.model,
        years_base,
        len(folds),
        [int(fold.valid_year) for fold in folds],
        len(base_feat_cols),
        len(te_feat_cols),
    )

    baseline_defaults = _build_model_defaults(base_train_args, model=str(args.model))
    baseline_train_args = _build_train_args(
        base_train_args=base_train_args,
        tune_args=args,
        params=baseline_defaults,
    )
    baseline_eval = _evaluate_binary_cv(
        frame=base_frame,
        folds=folds,
        feat_cols=base_feat_cols,
        categorical_cols=base_categorical_cols,
        label_col=label_col,
        pred_col=pred_col,
        train_args=baseline_train_args,
    )
    baseline_summary = _build_baseline_summary(
        args=args,
        eval_result=baseline_eval,
        input_path=str(args.input_base),
    )

    outputs["storage"].parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{outputs['storage']}"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )
    _assert_study_compatible(
        study,
        expected_config=_study_config_payload(
            args,
            input_base_path=input_base_path,
            input_te_path=input_te_path,
        ),
    )

    def objective(trial) -> float:
        feature_set = suggest_feature_set(trial)
        model_params = suggest_model_params(trial, model=str(args.model))
        input_display = _trial_input_display(args, feature_set=feature_set)
        train_args = _build_train_args(
            base_train_args=base_train_args,
            tune_args=args,
            params=model_params,
        )

        def on_progress(
            fold_id: int,
            running_mean: float,
            fold_logloss: dict[int, float],
            fold_best_iteration: dict[int, int],
            fold_benter_r2_valid: dict[int, float] | None,
        ) -> None:
            trial.report(float(running_mean), step=int(fold_id))
            _set_trial_user_attrs(
                trial,
                feature_set=feature_set,
                input_path=input_display,
                fold_logloss=fold_logloss,
                fold_best_iteration=fold_best_iteration,
                fold_benter_r2_valid=fold_benter_r2_valid,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

        eval_result = _evaluate_binary_cv(
            frame=frames_by_feature_set[feature_set],
            folds=folds,
            feat_cols=feature_cols_by_feature_set[feature_set],
            categorical_cols=categorical_cols_by_feature_set[feature_set],
            label_col=label_col,
            pred_col=pred_col,
            train_args=train_args,
            progress_callback=on_progress,
        )
        _set_trial_user_attrs(
            trial,
            feature_set=feature_set,
            input_path=input_display,
            fold_logloss=eval_result.fold_logloss,
            fold_best_iteration=eval_result.fold_best_iteration,
            fold_benter_r2_valid=eval_result.fold_benter_r2_valid,
        )
        return float(eval_result.value_mean_logloss)

    existing_trials = len(study.trials)
    additional_trials = max(0, int(args.n_trials) - existing_trials)
    logger.info(
        "study=%s storage=%s existing_trials=%s target_trials=%s additional=%s",
        study.study_name,
        outputs["storage"],
        existing_trials,
        args.n_trials,
        additional_trials,
    )
    timeout = None if int(args.timeout) == 0 else int(args.timeout)
    if additional_trials > 0:
        study.optimize(objective, n_trials=additional_trials, timeout=timeout)

    results = _trial_results_from_study(study)
    trials_frame = _trial_results_to_frame(results)
    outputs["trials"].parent.mkdir(parents=True, exist_ok=True)
    trials_frame.to_parquet(outputs["trials"], index=False)
    logger.info("wrote %s rows=%s", outputs["trials"], len(trials_frame))

    study_best_trial_number = None
    if study.best_trial is not None:
        study_best_trial_number = int(study.best_trial.number)

    selected_trial, constraint_passed, selection_mode = select_best_trial_result(
        results=results,
        task=str(args.task),
        baseline_summary=baseline_summary,
        study_best_trial_number=study_best_trial_number,
    )

    best_output = _build_best_output(
        args=args,
        outputs=outputs,
        study_name=study_name,
        total_trials=len(study.trials),
        selected_trial=selected_trial,
        baseline_summary=baseline_summary,
        fixed_config=fixed_config,
        constraint_passed=constraint_passed,
        selection_mode=selection_mode,
    )
    save_json(outputs["best"], best_output)
    logger.info("wrote %s", outputs["best"])

    best_params_output = _build_best_params_output(
        args=args,
        selected_trial=selected_trial,
    )
    save_json(outputs["best_params"], best_params_output)
    logger.info("wrote %s", outputs["best_params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
