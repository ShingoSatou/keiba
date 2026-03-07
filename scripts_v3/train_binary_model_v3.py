#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.feature_registry_v3 import (  # noqa: E402
    BINARY_ENTITY_ID_FEATURES,
    FEATURE_MANIFEST_VERSION,
    OPERATIONAL_MODE_CHOICES,
    get_binary_feature_columns,
    validate_feature_contract,
)
from scripts_v3.metrics_benter_v3_common import (  # noqa: E402
    benter_nll_and_null,
    benter_r2,
    fit_beta_by_nll,
    logit_clip,
    race_softmax,
)
from scripts_v3.train_binary_v3_common import (  # noqa: E402
    DEFAULT_CV_WINDOW_POLICY,
    DEFAULT_TRAIN_WINDOW_YEARS,
    build_cv_policy_payload,
    build_fixed_window_year_folds,
    build_oof_frame,
    coerce_feature_matrix,
    compute_binary_metrics,
    fold_integrity,
    hash_files,
    make_window_definition,
    prepare_binary_frame,
    resolve_path,
    save_json,
    select_recent_window_years,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_SEED = 42

TASK_CHOICES = ("win", "place")
MODEL_CHOICES = ("lgbm", "xgb", "cat")
MODEL_PARAM_CLI_FLAGS: dict[str, dict[str, str]] = {
    "lgbm": {
        "learning_rate": "--learning-rate",
        "num_leaves": "--num-leaves",
        "min_data_in_leaf": "--min-data-in-leaf",
        "lambda_l1": "--lambda-l1",
        "lambda_l2": "--lambda-l2",
        "feature_fraction": "--feature-fraction",
        "bagging_fraction": "--bagging-fraction",
        "bagging_freq": "--bagging-freq",
    },
    "xgb": {
        "learning_rate": "--learning-rate",
        "max_depth": "--max-depth",
        "min_child_weight": "--min-child-weight",
        "gamma": "--gamma",
        "subsample": "--subsample",
        "colsample_bytree": "--colsample-bytree",
        "reg_alpha": "--reg-alpha",
        "reg_lambda": "--reg-lambda",
    },
    "cat": {
        "learning_rate": "--learning-rate",
        "depth": "--depth",
        "l2_leaf_reg": "--l2-leaf-reg",
        "random_strength": "--random-strength",
        "bagging_temperature": "--bagging-temperature",
        "rsm": "--rsm",
        "min_data_in_leaf": "--min-data-in-leaf",
    },
}


def _label_col(task: str) -> str:
    return "y_win" if task == "win" else "y_place"


def _pred_col(task: str, model: str) -> str:
    return f"p_{task}_{model}"


def _default_ext(model: str) -> str:
    if model == "lgbm":
        return "txt"
    if model == "xgb":
        return "json"
    if model == "cat":
        return "cbm"
    raise ValueError(f"Unknown model: {model}")


def _model_thread_count() -> int:
    raw = str(os.environ.get("V3_MODEL_THREADS", "")).strip()
    if not raw:
        return -1
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid V3_MODEL_THREADS=%s; fallback to -1", raw)
        return -1
    if value == 0:
        logger.warning("Ignoring V3_MODEL_THREADS=0; fallback to -1")
        return -1
    return value


def parse_args(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
    default_model: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v3 binary models with fixed-length sliding yearly CV."
    )
    parser.add_argument("--task", choices=list(TASK_CHOICES), default=default_task or "win")
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default=default_model or "lgbm")
    parser.add_argument(
        "--params-json",
        default="",
        help=(
            "Optional JSON file to set default input and model params. "
            "If omitted, automatically loads data/oof/binary_v3_{task}_{model}_best_params.json "
            "when present. Explicit CLI flags take precedence."
        ),
    )
    parser.add_argument("--input", default="data/features_v3.parquet")
    parser.add_argument("--oof-output", default="")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--model-output", default="")
    parser.add_argument("--all-years-model-output", default="")
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--holdout-input", default="")
    parser.add_argument("--holdout-output", default="")

    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        help=(
            "Training window in years. "
            "The v3 standard comparison condition is fixed_sliding with 4 years."
        ),
    )
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--operational-mode",
        choices=list(OPERATIONAL_MODE_CHOICES),
        default="t10_only",
        help="Feature contract profile. Default is t10_only; includes_final is validation-only.",
    )
    parser.add_argument(
        "--include-entity-id-features",
        action="store_true",
        help="Opt in to raw entity ID features (jockey_key/trainer_key). Default is OFF.",
    )
    parser.add_argument(
        "--drop-entity-id-features",
        action="store_true",
        help="Deprecated: entity ID features are excluded by default. "
        "Use --include-entity-id-features to opt in.",
    )

    parser.add_argument("--learning-rate", type=float, default=0.05)

    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-fraction", type=float, default=1.0)
    parser.add_argument("--bagging-freq", type=int, default=0)

    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)

    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--rsm", type=float, default=1.0)
    parser.add_argument("--leaf-estimation-iterations", type=int, default=5)

    parser.add_argument("--benter-eps", type=float, default=1e-6)
    parser.add_argument("--benter-beta-min", type=float, default=0.01)
    parser.add_argument("--benter-beta-max", type=float, default=100.0)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_window_years <= 0:
        raise SystemExit("--train-window-years must be > 0")
    if args.num_boost_round <= 0:
        raise SystemExit("--num-boost-round must be > 0")
    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")
    if args.n_bins <= 1:
        raise SystemExit("--n-bins must be > 1")
    if not (0.0 < float(args.benter_eps) < 0.5):
        raise SystemExit("--benter-eps must be in (0, 0.5)")
    if bool(args.include_entity_id_features) and bool(args.drop_entity_id_features):
        raise SystemExit(
            "--include-entity-id-features and --drop-entity-id-features cannot be used together"
        )


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    for token in argv:
        if token == flag or token.startswith(prefix):
            return True
    return False


def _default_params_json_path(task: str, model: str) -> Path:
    return resolve_path(f"data/oof/binary_v3_{task}_{model}_best_params.json")


def _resolve_params_json_path(args: argparse.Namespace) -> Path | None:
    if str(args.params_json).strip():
        params_path = resolve_path(str(args.params_json))
        if not params_path.exists():
            raise SystemExit(f"params-json not found: {params_path}")
        return params_path

    default_path = _default_params_json_path(str(args.task), str(args.model))
    if default_path.exists():
        return default_path
    return None


def _derive_holdout_input_from_input(input_value: str, *, holdout_year: int) -> str | None:
    input_path = Path(str(input_value))
    if input_path.suffix != ".parquet":
        return None
    candidate = input_path.with_name(f"{input_path.stem}_{int(holdout_year)}{input_path.suffix}")
    if resolve_path(str(candidate)).exists():
        return str(candidate)
    return None


def _params_model_key(model: str) -> str:
    return f"{model}_params"


def _params_final_iterations_key(model: str) -> str:
    return "final_iterations" if model == "cat" else "final_num_boost_round"


def _apply_params_json(
    args: argparse.Namespace,
    params: dict[str, Any],
    *,
    argv: list[str],
    params_path: Path,
) -> None:
    if not hasattr(args, "_tuned_final_iterations"):
        args._tuned_final_iterations = None  # type: ignore[attr-defined]
    if not hasattr(args, "_applied_params_json"):
        args._applied_params_json = None  # type: ignore[attr-defined]
    if not hasattr(args, "_applied_params_feature_set"):
        args._applied_params_feature_set = None  # type: ignore[attr-defined]

    expected_task = str(args.task)
    expected_model = str(args.model)
    params_task = params.get("task")
    params_model = params.get("model")
    if params_task is not None and str(params_task) != expected_task:
        raise SystemExit(
            f"params-json task mismatch: expected={expected_task} actual={params_task}"
        )
    if params_model is not None and str(params_model) != expected_model:
        raise SystemExit(
            f"params-json model mismatch: expected={expected_model} actual={params_model}"
        )

    input_was_applied = False
    if (
        "input" in params
        and isinstance(params["input"], str)
        and not _argv_has_flag(argv, "--input")
    ):
        args.input = str(params["input"])
        input_was_applied = True

    if (
        input_was_applied
        and not _argv_has_flag(argv, "--holdout-input")
        and not str(args.holdout_input).strip()
    ):
        derived_holdout = _derive_holdout_input_from_input(
            str(args.input),
            holdout_year=int(args.holdout_year),
        )
        if derived_holdout is not None:
            args.holdout_input = derived_holdout

    if "train_window_years" in params and not _argv_has_flag(argv, "--train-window-years"):
        args.train_window_years = int(params["train_window_years"])
    if "operational_mode" in params and not _argv_has_flag(argv, "--operational-mode"):
        args.operational_mode = str(params["operational_mode"])
    if not _argv_has_flag(argv, "--include-entity-id-features") and not _argv_has_flag(
        argv, "--drop-entity-id-features"
    ):
        if "include_entity_id_features" in params:
            args.include_entity_id_features = bool(params["include_entity_id_features"])
            args.drop_entity_id_features = False

    model_params = params.get(_params_model_key(expected_model))
    if isinstance(model_params, dict):
        for key, flag in MODEL_PARAM_CLI_FLAGS[expected_model].items():
            if key in model_params and not _argv_has_flag(argv, flag):
                setattr(args, key, model_params[key])

    if not _argv_has_flag(argv, "--num-boost-round"):
        tuned_final_iterations = params.get(_params_final_iterations_key(expected_model))
        if tuned_final_iterations is not None:
            args._tuned_final_iterations = int(tuned_final_iterations)  # type: ignore[attr-defined]

    feature_set_value = params.get("feature_set")
    args._applied_params_json = str(params_path)  # type: ignore[attr-defined]
    args._applied_params_feature_set = (  # type: ignore[attr-defined]
        None if feature_set_value is None else str(feature_set_value)
    )


def _include_entity_id_features(args: argparse.Namespace) -> bool:
    if bool(args.drop_entity_id_features):
        logger.warning(
            "--drop-entity-id-features is deprecated; entity IDs are already excluded by default."
        )
        return False
    return bool(args.include_entity_id_features)


def _infer_feature_set_from_input_path(input_path: Path | str) -> str:
    basename = Path(str(input_path)).name.lower()
    return "te" if "_te" in basename else "base"


def _resolve_binary_feature_columns(
    *,
    frame: pd.DataFrame,
    input_path: Path | str,
    include_entity_id_features: bool,
    operational_mode: str,
    feature_set_override: str | None = None,
) -> tuple[list[str], list[str], bool]:
    feature_set = (
        str(feature_set_override)
        if feature_set_override is not None
        else _infer_feature_set_from_input_path(input_path)
    )
    feat_cols = get_binary_feature_columns(
        frame,
        include_entity_ids=include_entity_id_features,
        operational_mode=str(operational_mode),
        include_te_features=feature_set == "te",
    )
    if not feat_cols:
        raise SystemExit("No feature columns available")
    validate_feature_contract(
        feat_cols,
        operational_mode=str(operational_mode),
        stage="binary",
    )
    categorical_cols = (
        [c for c in BINARY_ENTITY_ID_FEATURES if c in feat_cols]
        if include_entity_id_features
        else []
    )
    forbidden_feature_check_passed = True
    return feat_cols, categorical_cols, forbidden_feature_check_passed


def _resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    task = str(args.task)
    model = str(args.model)
    ext = _default_ext(model)
    defaults = {
        "oof": f"data/oof/{task}_{model}_oof.parquet",
        "metrics": f"data/oof/{task}_{model}_cv_metrics.json",
        "model": f"models/{task}_{model}_v3.{ext}",
        "all_years_model": f"models/{task}_{model}_all_years_v3.{ext}",
        "meta": f"models/{task}_{model}_bundle_meta_v3.json",
        "feature_manifest": f"models/{task}_{model}_feature_manifest_v3.json",
        "holdout": f"data/holdout/{task}_{model}_holdout_pred_v3.parquet",
    }
    return {
        "oof": resolve_path(args.oof_output or defaults["oof"]),
        "metrics": resolve_path(args.metrics_output or defaults["metrics"]),
        "model": resolve_path(args.model_output or defaults["model"]),
        "all_years_model": resolve_path(args.all_years_model_output or defaults["all_years_model"]),
        "meta": resolve_path(args.meta_output or defaults["meta"]),
        "feature_manifest": resolve_path(defaults["feature_manifest"]),
        "holdout": resolve_path(args.holdout_output or defaults["holdout"]),
    }


def _lgbm_fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
    seed: int,
    categorical_cols: list[str],
) -> tuple[Any, np.ndarray, np.ndarray, int]:
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
        n_jobs=_model_thread_count(),
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_names=["valid"],
        eval_metric="binary_logloss",
        categorical_feature=categorical_cols or "auto",
        callbacks=[lgb.early_stopping(int(args.early_stopping_rounds), verbose=False)],
    )
    best_iteration = int(getattr(model, "best_iteration_", 0) or int(args.num_boost_round))
    p_train = model.predict_proba(x_train, num_iteration=best_iteration)[:, 1]
    p_valid = model.predict_proba(x_valid, num_iteration=best_iteration)[:, 1]
    return model, p_train, p_valid, best_iteration


def _xgb_best_iteration(model: Any, *, fallback: int) -> int:
    best = getattr(model, "best_iteration", None)
    if best is None:
        best = getattr(model, "best_iteration_", None)
    if best is None:
        best = getattr(getattr(model, "get_booster", lambda: None)(), "best_iteration", None)
    if best is None:
        best = int(fallback) - 1
    return int(best) + 1


def _xgb_predict_proba(model: Any, x: pd.DataFrame, *, best_iteration: int) -> np.ndarray:
    try:
        return model.predict_proba(x, iteration_range=(0, int(best_iteration)))[:, 1]
    except TypeError:
        return model.predict_proba(x, ntree_limit=int(best_iteration))[:, 1]


def _xgb_fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("xgboost is not installed. Run `uv sync --extra xgboost`.") from exc

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight),
        gamma=float(args.gamma),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        reg_alpha=float(args.reg_alpha),
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(seed),
        early_stopping_rounds=int(args.early_stopping_rounds),
        n_jobs=_model_thread_count(),
        verbosity=0,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    best_iteration = _xgb_best_iteration(model, fallback=int(args.num_boost_round))
    p_train = _xgb_predict_proba(model, x_train, best_iteration=best_iteration)
    p_valid = _xgb_predict_proba(model, x_valid, best_iteration=best_iteration)
    return model, p_train, p_valid, best_iteration


def _cat_fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    try:
        from catboost import CatBoostClassifier, Pool
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("catboost is not installed. Run `uv sync --extra catboost`.") from exc

    train_pool = Pool(x_train, label=y_train)
    valid_pool = Pool(x_valid, label=y_valid)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=int(args.num_boost_round),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_strength=float(args.random_strength),
        bagging_temperature=float(args.bagging_temperature),
        rsm=float(args.rsm),
        min_data_in_leaf=int(args.min_data_in_leaf),
        leaf_estimation_iterations=int(args.leaf_estimation_iterations),
        random_seed=int(seed),
        allow_writing_files=False,
        od_type="Iter",
        od_wait=int(args.early_stopping_rounds),
        thread_count=_model_thread_count(),
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    best_iteration = int(model.get_best_iteration())
    if best_iteration < 0:
        best_iteration = int(args.num_boost_round) - 1
    best_iteration += 1

    p_train = model.predict_proba(train_pool)[:, 1]
    p_valid = model.predict_proba(valid_pool)[:, 1]
    return model, p_train, p_valid, best_iteration


def _fit_predict_fold(
    *,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
    seed: int,
    categorical_cols: list[str],
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    if model_name == "lgbm":
        return _lgbm_fit_predict(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=args,
            seed=seed,
            categorical_cols=categorical_cols,
        )
    if model_name == "xgb":
        return _xgb_fit_predict(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=args,
            seed=seed,
        )
    if model_name == "cat":
        return _cat_fit_predict(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=args,
            seed=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _fit_final_model(
    *,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
    seed: int,
    n_estimators: int,
    categorical_cols: list[str],
):
    if model_name == "lgbm":
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
            n_jobs=_model_thread_count(),
            verbosity=-1,
        )
        model.fit(x_train, y_train, categorical_feature=categorical_cols or "auto")
        return model

    if model_name == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=int(n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
            min_child_weight=float(args.min_child_weight),
            gamma=float(args.gamma),
            subsample=float(args.subsample),
            colsample_bytree=float(args.colsample_bytree),
            reg_lambda=float(args.reg_lambda),
            reg_alpha=float(args.reg_alpha),
            eval_metric="logloss",
            tree_method="hist",
            random_state=int(seed),
            n_jobs=_model_thread_count(),
            verbosity=0,
        )
        model.fit(x_train, y_train, verbose=False)
        return model

    if model_name == "cat":
        from catboost import CatBoostClassifier, Pool

        train_pool = Pool(x_train, label=y_train)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=int(n_estimators),
            learning_rate=float(args.learning_rate),
            depth=int(args.depth),
            l2_leaf_reg=float(args.l2_leaf_reg),
            random_strength=float(args.random_strength),
            bagging_temperature=float(args.bagging_temperature),
            rsm=float(args.rsm),
            min_data_in_leaf=int(args.min_data_in_leaf),
            leaf_estimation_iterations=int(args.leaf_estimation_iterations),
            random_seed=int(seed),
            allow_writing_files=False,
            thread_count=_model_thread_count(),
            verbose=False,
        )
        model.fit(train_pool)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def _predict_proba(
    model_name: str,
    model: Any,
    x: pd.DataFrame,
    *,
    best_iteration: int | None = None,
) -> np.ndarray:
    if model_name == "lgbm":
        return model.predict_proba(x, num_iteration=best_iteration)[:, 1]
    if model_name == "xgb":
        if best_iteration is None:
            return model.predict_proba(x)[:, 1]
        return _xgb_predict_proba(model, x, best_iteration=best_iteration)
    if model_name == "cat":
        return model.predict_proba(x)[:, 1]
    raise ValueError(f"Unknown model: {model_name}")


def _save_model(model_name: str, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "lgbm":
        model.booster_.save_model(str(path))
        return
    if model_name == "xgb":
        model.save_model(str(path))
        return
    if model_name == "cat":
        model.save_model(str(path))
        return
    raise ValueError(f"Unknown model: {model_name}")


def _summary_stats(values: list[float | None]) -> dict[str, float | None]:
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


def _build_feature_manifest_payload(
    *,
    args: argparse.Namespace,
    feat_cols: list[str],
    categorical_cols: list[str],
    include_entity_id_features: bool,
    forbidden_feature_check_passed: bool,
    valid_years: list[int],
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": str(args.model),
        "cv_policy": {
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "train_window_years": int(args.train_window_years),
            "valid_years": list(map(int, valid_years)),
            "holdout_year": int(args.holdout_year),
            "window_definition": make_window_definition(int(args.train_window_years)),
        },
        "operational_mode": str(args.operational_mode),
        "include_entity_id_features": bool(include_entity_id_features),
        "feature_columns": list(feat_cols),
        "categorical_features": list(categorical_cols),
        "forbidden_feature_check_passed": bool(forbidden_feature_check_passed),
        "feature_manifest_version": int(FEATURE_MANIFEST_VERSION),
    }


def _add_benter_for_fold(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    p_train: np.ndarray,
    p_valid: np.ndarray,
    pred_col: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_scores = logit_clip(p_train, eps=float(args.benter_eps))
    beta_hat = fit_beta_by_nll(
        train_df["race_id"].to_numpy(),
        train_df["y_win"].to_numpy(),
        train_df["field_size"].to_numpy(dtype=float),
        train_scores,
        beta_min=float(args.benter_beta_min),
        beta_max=float(args.benter_beta_max),
    )

    valid_scores = logit_clip(p_valid, eps=float(args.benter_eps))
    c_beta1 = race_softmax(valid_scores, valid_df["race_id"].to_numpy(), beta=1.0)
    c_betahat = race_softmax(valid_scores, valid_df["race_id"].to_numpy(), beta=beta_hat)

    nll_valid, nll_null_valid, n_races_valid = benter_nll_and_null(
        valid_df["race_id"].to_numpy(),
        valid_df["y_win"].to_numpy(),
        valid_df["field_size"].to_numpy(dtype=float),
        c_betahat,
    )
    nll_valid_beta1, nll_null_valid_beta1, _ = benter_nll_and_null(
        valid_df["race_id"].to_numpy(),
        valid_df["y_win"].to_numpy(),
        valid_df["field_size"].to_numpy(dtype=float),
        c_beta1,
    )

    benter_payload = {
        "benter_beta_hat": float(beta_hat),
        "benter_nll_valid": float(nll_valid),
        "benter_nll_null_valid": float(nll_null_valid),
        "benter_r2_valid": float(benter_r2(nll_valid, nll_null_valid)),
        "benter_r2_valid_beta1": float(benter_r2(nll_valid_beta1, nll_null_valid_beta1)),
        "benter_num_races_valid": int(n_races_valid),
    }
    extra_cols = {
        f"score_{pred_col}": valid_scores,
        f"c_{pred_col}_beta1": c_beta1,
        f"c_{pred_col}_betahat": c_betahat,
    }
    return benter_payload, extra_cols


def _evaluate_cv_fold(
    *,
    frame: pd.DataFrame,
    fold,
    feat_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    pred_col: str,
    args: argparse.Namespace,
    emit_oof: bool,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    train_df = frame[frame["year"].isin(fold.train_years)].copy()
    valid_df = frame[frame["year"] == fold.valid_year].copy()
    if train_df.empty or valid_df.empty:
        raise SystemExit(
            f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
        )
    fold_integrity(train_df, valid_df, fold.valid_year)

    train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )
    valid_df = valid_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
        drop=True
    )

    x_train = coerce_feature_matrix(train_df, feat_cols)
    y_train = train_df[label_col].astype(int)
    x_valid = coerce_feature_matrix(valid_df, feat_cols)
    y_valid = valid_df[label_col].astype(int)

    _, p_train, p_valid, best_iteration = _fit_predict_fold(
        model_name=str(args.model),
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        args=args,
        seed=int(args.seed) + int(fold.fold_id),
        categorical_cols=categorical_cols,
    )

    metrics = compute_binary_metrics(
        y_valid.to_numpy(dtype=int),
        p_valid,
        n_bins=int(args.n_bins),
    )
    window_definition = make_window_definition(int(args.train_window_years))
    fold_metric: dict[str, Any] = {
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
        "base_rate": metrics["base_rate"],
        "reliability": metrics["reliability"],
        "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
        "train_window_years": int(args.train_window_years),
        "holdout_year": int(args.holdout_year),
        "window_definition": window_definition,
    }

    oof: pd.DataFrame | None = None
    if emit_oof:
        oof = build_oof_frame(
            valid_df,
            label_col=label_col,
            pred_col=pred_col,
            pred_values=p_valid,
            fold_id=int(fold.fold_id),
            valid_year=int(fold.valid_year),
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            window_definition=window_definition,
        )

    if str(args.task) == "win":
        benter_payload, extra_cols = _add_benter_for_fold(
            train_df=train_df,
            valid_df=valid_df,
            p_train=p_train,
            p_valid=p_valid,
            pred_col=pred_col,
            args=args,
        )
        fold_metric["benter"] = benter_payload
        if oof is not None:
            for col_name, values in extra_cols.items():
                oof[col_name] = values

    logger.info(
        "fold=%s valid_year=%s logloss=%s brier=%.6f auc=%s ece=%.6f",
        fold.fold_id,
        fold.valid_year,
        (f"{metrics['logloss']:.6f}" if metrics["logloss"] is not None else "None"),
        float(metrics["brier"]),
        (f"{metrics['auc']:.6f}" if metrics["auc"] is not None else "None"),
        float(metrics["ece"]),
    )
    return fold_metric, oof


def _run_cv_loop(
    *,
    frame: pd.DataFrame,
    folds: list,
    feat_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    pred_col: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[int]]:
    """CV loop を実行し、OOF / fold metrics / best_iterations を返す。"""
    oof_list: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold in folds:
        fold_metric, oof = _evaluate_cv_fold(
            frame=frame,
            fold=fold,
            feat_cols=feat_cols,
            categorical_cols=categorical_cols,
            label_col=label_col,
            pred_col=pred_col,
            args=args,
            emit_oof=True,
        )
        best_iterations.append(int(fold_metric["best_iteration"]))
        if oof is None:
            raise SystemExit("emit_oof=True but no OOF predictions were generated")
        oof_list.append(oof)
        fold_metrics.append(fold_metric)

    if not oof_list:
        raise SystemExit("No OOF predictions generated")

    oof = pd.concat(oof_list, axis=0, ignore_index=True)
    oof = oof.sort_values(["race_id", "horse_no"], kind="mergesort")
    return oof, fold_metrics, best_iterations


def _train_final_models(
    *,
    frame: pd.DataFrame,
    years: list[int],
    feat_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    args: argparse.Namespace,
    final_iterations: int,
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame]:
    """recent window と all-years の最終モデルを学習する。"""
    recent_years = select_recent_window_years(
        years,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    recent_df = frame[frame["year"].isin(recent_years)].copy()
    all_df = frame.copy()

    x_recent = coerce_feature_matrix(recent_df, feat_cols)
    y_recent = recent_df[label_col].astype(int)
    x_all = coerce_feature_matrix(all_df, feat_cols)
    y_all = all_df[label_col].astype(int)

    main_model = _fit_final_model(
        model_name=str(args.model),
        x_train=x_recent,
        y_train=y_recent,
        args=args,
        seed=int(args.seed) + 1000,
        n_estimators=int(final_iterations),
        categorical_cols=categorical_cols,
    )
    all_model = _fit_final_model(
        model_name=str(args.model),
        x_train=x_all,
        y_train=y_all,
        args=args,
        seed=int(args.seed) + 2000,
        n_estimators=int(final_iterations),
        categorical_cols=categorical_cols,
    )
    return main_model, all_model, recent_df, all_df


def _run_holdout_prediction(
    *,
    holdout_input_path: Path | None,
    main_model: Any,
    feat_cols: list[str],
    label_col: str,
    pred_col: str,
    args: argparse.Namespace,
    final_iterations: int,
    output_path: Path,
) -> tuple[int, int]:
    """holdout 入力があれば推論を実行し、行数・レース数を返す。"""
    if holdout_input_path is None or not holdout_input_path.exists():
        return 0, 0

    holdout_raw = pd.read_parquet(holdout_input_path)
    if label_col not in holdout_raw.columns:
        holdout_raw[label_col] = 0
    holdout_df = prepare_binary_frame(holdout_raw, label_col=label_col)
    holdout_df = holdout_df[holdout_df["year"] >= int(args.holdout_year)].copy()
    if holdout_df.empty:
        return 0, 0

    x_hold = coerce_feature_matrix(holdout_df, feat_cols)
    p_hold = _predict_proba(
        str(args.model),
        main_model,
        x_hold,
        best_iteration=(
            int(final_iterations) if str(args.model) == "xgb" else int(final_iterations)
        ),
    )
    holdout_pred = holdout_df[
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
            if c in holdout_df.columns
        ]
    ].copy()
    holdout_pred[pred_col] = p_hold
    holdout_pred["valid_year"] = holdout_df["year"].astype(int).to_numpy()
    holdout_pred["cv_window_policy"] = DEFAULT_CV_WINDOW_POLICY
    holdout_pred["train_window_years"] = int(args.train_window_years)
    holdout_pred["holdout_year"] = int(args.holdout_year)
    holdout_pred["window_definition"] = make_window_definition(int(args.train_window_years))
    holdout_pred = holdout_pred.sort_values(["race_id", "horse_no"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_pred.to_parquet(output_path, index=False)
    logger.info("wrote %s", output_path)
    return int(len(holdout_pred)), int(holdout_pred["race_id"].nunique())


def _build_metrics_payload(
    *,
    folds: list,
    fold_metrics: list[dict[str, Any]],
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    holdout_rows: int,
    holdout_races: int,
    years: list[int],
    final_iterations: int,
    cv_best_iteration_median: int,
    final_iteration_source: str,
    args: argparse.Namespace,
    label_col: str,
    pred_col: str,
    include_entity_id_features: bool,
) -> dict[str, Any]:
    """CV メトリクスの集計辞書を組み立てる。"""
    logloss_stats = _summary_stats([m.get("logloss") for m in fold_metrics])
    brier_stats = _summary_stats([m.get("brier") for m in fold_metrics])
    auc_stats = _summary_stats([m.get("auc") for m in fold_metrics])
    ece_stats = _summary_stats([m.get("ece") for m in fold_metrics])
    cv_policy = build_cv_policy_payload(
        folds,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
        cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": str(args.model),
        "label_col": label_col,
        "pred_col": pred_col,
        "cv_policy": cv_policy,
        "config": {
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "seed": int(args.seed),
            "n_bins": int(args.n_bins),
            "operational_mode": str(args.operational_mode),
            "include_entity_id_features": bool(include_entity_id_features),
            "params_json": getattr(args, "_applied_params_json", None),
            "params_json_feature_set": getattr(args, "_applied_params_feature_set", None),
            "final_iteration_source": final_iteration_source,
        },
        "data_summary": {
            "rows": int(len(frame)),
            "races": int(frame["race_id"].nunique()),
            "years": years,
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
            "holdout_rows": int(holdout_rows),
            "holdout_races": int(holdout_races),
        },
        "folds": fold_metrics,
        "summary": {
            "n_folds": int(len(fold_metrics)),
            "logloss": logloss_stats,
            "brier": brier_stats,
            "auc": auc_stats,
            "ece": ece_stats,
            "best_iteration_median": int(cv_best_iteration_median),
            "final_iterations": int(final_iterations),
        },
    }

    if str(args.task) == "win":
        benter_r2_stats = _summary_stats(
            [
                (f.get("benter") or {}).get("benter_r2_valid")
                for f in fold_metrics
                if isinstance(f.get("benter"), dict)
            ]
        )
        payload["summary"]["benter_r2_valid"] = benter_r2_stats

    return payload


def _build_meta_payload(
    *,
    args: argparse.Namespace,
    folds: list,
    label_col: str,
    pred_col: str,
    input_path: Path,
    feat_cols: list[str],
    categorical_cols: list[str],
    outputs: dict[str, Path],
    metrics_summary: dict[str, Any],
    recent_df: pd.DataFrame,
    all_df: pd.DataFrame,
    years: list[int],
    include_entity_id_features: bool,
    forbidden_feature_check_passed: bool,
) -> dict[str, Any]:
    """モデルバンドルメタ辞書を組み立てる。"""
    recent_years = sorted(recent_df["year"].unique().tolist())
    cv_policy = build_cv_policy_payload(
        folds,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
        cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
    )
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": str(args.model),
        "label_col": label_col,
        "pred_col": pred_col,
        "holdout_rule": f"exclude year >= {args.holdout_year}",
        "cv_policy": cv_policy,
        "input_path": str(input_path),
        "operational_mode": str(args.operational_mode),
        "include_entity_id_features": bool(include_entity_id_features),
        "feature_columns": feat_cols,
        "categorical_features": categorical_cols,
        "forbidden_feature_check_passed": bool(forbidden_feature_check_passed),
        "feature_manifest_version": int(FEATURE_MANIFEST_VERSION),
        "output_paths": {
            "oof": str(outputs["oof"]),
            "cv_metrics": str(outputs["metrics"]),
            "main_model": str(outputs["model"]),
            "all_years_model": str(outputs["all_years_model"]),
            "feature_manifest": str(outputs["feature_manifest"]),
            "holdout": str(outputs["holdout"]),
        },
        "tuned_defaults": {
            "params_json": getattr(args, "_applied_params_json", None),
            "feature_set": getattr(args, "_applied_params_feature_set", None),
        },
        "cv_summary": metrics_summary,
        "final_train_summary": {
            "main_model_years": recent_years,
            "main_model_rows": int(len(recent_df)),
            "main_model_window_type": "fixed_sliding_recent_window",
            "all_years_model_years": years,
            "all_years_model_rows": int(len(all_df)),
            "all_years_model_analysis_only": True,
        },
        "code_hash": hash_files(
            [
                Path(__file__),
                Path(resolve_path("scripts_v3/feature_registry_v3.py")),
            ]
        ),
    }


def main(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
    default_model: str | None = None,
) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv_list, default_task=default_task, default_model=default_model)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    args._tuned_final_iterations = None  # type: ignore[attr-defined]
    args._applied_params_json = None  # type: ignore[attr-defined]
    args._applied_params_feature_set = None  # type: ignore[attr-defined]
    params_path = _resolve_params_json_path(args)
    if params_path is not None:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise SystemExit("params-json must be a JSON object.")
        _apply_params_json(args, params, argv=argv_list, params_path=params_path)
        logger.info(
            "applied params-json=%s feature_set=%s input=%s",
            params_path,
            getattr(args, "_applied_params_feature_set", None),
            args.input,
        )

    _validate_args(args)

    label_col = _label_col(str(args.task))
    pred_col = _pred_col(str(args.task), str(args.model))

    input_path = resolve_path(args.input)
    outputs = _resolve_output_paths(args)
    holdout_input_path = resolve_path(args.holdout_input) if args.holdout_input else None

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    frame = prepare_binary_frame(pd.read_parquet(input_path), label_col=label_col)
    frame = frame[frame["year"] < int(args.holdout_year)].copy()
    if frame.empty:
        raise SystemExit("No trainable rows after holdout exclusion")

    years = sorted(frame["year"].unique().tolist())
    folds = build_fixed_window_year_folds(
        years,
        window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        max_window = max(1, len(years) - 1)
        raise SystemExit(
            "No binary CV folds available under the fixed_sliding policy "
            f"(available_years={years}, try --train-window-years <= {max_window})"
        )

    include_entity_id_features = _include_entity_id_features(args)
    feat_cols, categorical_cols, forbidden_feature_check_passed = _resolve_binary_feature_columns(
        frame=frame,
        input_path=input_path,
        include_entity_id_features=include_entity_id_features,
        operational_mode=str(args.operational_mode),
    )

    logger.info(
        "%s-%s train years=%s folds=%s window=%s cv_policy=%s holdout_year>=%s",
        args.task,
        args.model,
        years,
        len(folds),
        args.train_window_years,
        DEFAULT_CV_WINDOW_POLICY,
        args.holdout_year,
    )

    # --- CV loop ---
    oof, fold_metrics, best_iterations = _run_cv_loop(
        frame=frame,
        folds=folds,
        feat_cols=feat_cols,
        categorical_cols=categorical_cols,
        label_col=label_col,
        pred_col=pred_col,
        args=args,
    )

    # --- 最終モデル学習 ---
    cv_best_iteration_median = int(np.median(best_iterations))
    cv_best_iteration_median = max(1, min(cv_best_iteration_median, int(args.num_boost_round)))
    final_iterations = cv_best_iteration_median
    final_iteration_source = "cv_best_iteration_median"
    tuned_final_iterations = getattr(args, "_tuned_final_iterations", None)
    if tuned_final_iterations is not None:
        final_iterations = max(1, min(int(tuned_final_iterations), int(args.num_boost_round)))
        final_iteration_source = "params_json"

    main_model, all_model, recent_df, all_df = _train_final_models(
        frame=frame,
        years=years,
        feat_cols=feat_cols,
        categorical_cols=categorical_cols,
        label_col=label_col,
        args=args,
        final_iterations=final_iterations,
    )

    # --- OOF / モデル保存 ---
    outputs["oof"].parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(outputs["oof"], index=False)
    _save_model(str(args.model), main_model, outputs["model"])
    _save_model(str(args.model), all_model, outputs["all_years_model"])

    # --- holdout 推論 ---
    holdout_rows, holdout_races = _run_holdout_prediction(
        holdout_input_path=holdout_input_path,
        main_model=main_model,
        feat_cols=feat_cols,
        label_col=label_col,
        pred_col=pred_col,
        args=args,
        final_iterations=final_iterations,
        output_path=outputs["holdout"],
    )

    # --- メトリクス / メタ保存 ---
    metrics_payload = _build_metrics_payload(
        folds=folds,
        fold_metrics=fold_metrics,
        frame=frame,
        oof=oof,
        holdout_rows=holdout_rows,
        holdout_races=holdout_races,
        years=years,
        final_iterations=final_iterations,
        cv_best_iteration_median=cv_best_iteration_median,
        final_iteration_source=final_iteration_source,
        args=args,
        label_col=label_col,
        pred_col=pred_col,
        include_entity_id_features=include_entity_id_features,
    )
    save_json(outputs["metrics"], metrics_payload)

    feature_manifest_payload = _build_feature_manifest_payload(
        args=args,
        feat_cols=feat_cols,
        categorical_cols=categorical_cols,
        include_entity_id_features=include_entity_id_features,
        forbidden_feature_check_passed=forbidden_feature_check_passed,
        valid_years=[int(fold.valid_year) for fold in folds],
    )
    save_json(outputs["feature_manifest"], feature_manifest_payload)

    meta_payload = _build_meta_payload(
        args=args,
        folds=folds,
        label_col=label_col,
        pred_col=pred_col,
        input_path=input_path,
        feat_cols=feat_cols,
        categorical_cols=categorical_cols,
        outputs=outputs,
        metrics_summary=metrics_payload["summary"],
        recent_df=recent_df,
        all_df=all_df,
        years=years,
        include_entity_id_features=include_entity_id_features,
        forbidden_feature_check_passed=forbidden_feature_check_passed,
    )
    save_json(outputs["meta"], meta_payload)

    logger.info("wrote %s", outputs["oof"])
    logger.info("wrote %s", outputs["metrics"])
    logger.info("wrote %s", outputs["model"])
    logger.info("wrote %s", outputs["all_years_model"])
    logger.info("wrote %s", outputs["feature_manifest"])
    logger.info("wrote %s", outputs["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
