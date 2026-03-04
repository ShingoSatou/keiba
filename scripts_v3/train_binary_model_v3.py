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

from scripts_v3.metrics_benter_v3_common import (  # noqa: E402
    benter_nll_and_null,
    benter_r2,
    fit_beta_by_nll,
    logit_clip,
    race_softmax,
)
from scripts_v3.train_binary_v3_common import (  # noqa: E402
    build_oof_frame,
    build_rolling_year_folds,
    coerce_feature_matrix,
    compute_binary_metrics,
    feature_columns,
    fold_integrity,
    hash_files,
    prepare_binary_frame,
    resolve_path,
    save_json,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_TRAIN_WINDOW_YEARS = 5
DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_SEED = 42

TASK_CHOICES = ("win", "place")
MODEL_CHOICES = ("lgbm", "xgb", "cat")
ENTITY_ID_FEATURES = {"jockey_key", "trainer_key"}


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


def parse_args(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
    default_model: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v3 binary models with rolling yearly CV.")
    parser.add_argument("--task", choices=list(TASK_CHOICES), default=default_task or "win")
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default=default_model or "lgbm")
    parser.add_argument("--input", default="data/features_v3.parquet")
    parser.add_argument("--oof-output", default="")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--model-output", default="")
    parser.add_argument("--all-years-model-output", default="")
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--holdout-input", default="")
    parser.add_argument("--holdout-output", default="")

    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS)
    parser.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--drop-entity-id-features", action="store_true")

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


def _drop_entity_features(cols: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return cols
    return [c for c in cols if c not in ENTITY_ID_FEATURES]


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
        "holdout": f"data/holdout/{task}_{model}_holdout_pred_v3.parquet",
    }
    return {
        "oof": resolve_path(args.oof_output or defaults["oof"]),
        "metrics": resolve_path(args.metrics_output or defaults["metrics"]),
        "model": resolve_path(args.model_output or defaults["model"]),
        "all_years_model": resolve_path(args.all_years_model_output or defaults["all_years_model"]),
        "meta": resolve_path(args.meta_output or defaults["meta"]),
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
        n_jobs=-1,
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
        n_jobs=-1,
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
        thread_count=-1,
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
            n_jobs=-1,
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
            n_jobs=-1,
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
            thread_count=-1,
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

        model, p_train, p_valid, best_iteration = _fit_predict_fold(
            model_name=str(args.model),
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            args=args,
            seed=int(args.seed) + int(fold.fold_id),
            categorical_cols=categorical_cols,
        )
        best_iterations.append(int(best_iteration))

        metrics = compute_binary_metrics(
            y_valid.to_numpy(dtype=int),
            p_valid,
            n_bins=int(args.n_bins),
        )
        fold_metric = {
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
        }

        oof = build_oof_frame(
            valid_df,
            label_col=label_col,
            pred_col=pred_col,
            pred_values=p_valid,
            fold_id=int(fold.fold_id),
            valid_year=int(fold.valid_year),
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
            for col_name, values in extra_cols.items():
                oof[col_name] = values

        oof_list.append(oof)
        fold_metrics.append(fold_metric)
        logger.info(
            "fold=%s valid_year=%s logloss=%s brier=%.6f auc=%s ece=%.6f",
            fold.fold_id,
            fold.valid_year,
            (f"{metrics['logloss']:.6f}" if metrics["logloss"] is not None else "None"),
            float(metrics["brier"]),
            (f"{metrics['auc']:.6f}" if metrics["auc"] is not None else "None"),
            float(metrics["ece"]),
        )

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
    recent_span = int(args.train_window_years) + 1
    recent_years = years[-recent_span:] if len(years) > recent_span else years
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
    holdout_pred = holdout_pred.sort_values(["race_id", "horse_no"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_pred.to_parquet(output_path, index=False)
    logger.info("wrote %s", output_path)
    return int(len(holdout_pred)), int(holdout_pred["race_id"].nunique())


def _build_metrics_payload(
    *,
    fold_metrics: list[dict[str, Any]],
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    holdout_rows: int,
    holdout_races: int,
    years: list[int],
    final_iterations: int,
    args: argparse.Namespace,
    label_col: str,
    pred_col: str,
) -> dict[str, Any]:
    """CV メトリクスの集計辞書を組み立てる。"""
    logloss_stats = _summary_stats([m.get("logloss") for m in fold_metrics])
    brier_stats = _summary_stats([m.get("brier") for m in fold_metrics])
    auc_stats = _summary_stats([m.get("auc") for m in fold_metrics])
    ece_stats = _summary_stats([m.get("ece") for m in fold_metrics])

    payload: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": str(args.model),
        "label_col": label_col,
        "pred_col": pred_col,
        "config": {
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "num_boost_round": int(args.num_boost_round),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "seed": int(args.seed),
            "n_bins": int(args.n_bins),
            "drop_entity_id_features": bool(args.drop_entity_id_features),
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
            "best_iteration_median": int(final_iterations),
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
) -> dict[str, Any]:
    """モデルバンドルメタ辞書を組み立てる。"""
    recent_years = sorted(recent_df["year"].unique().tolist())
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": str(args.task),
        "model": str(args.model),
        "label_col": label_col,
        "pred_col": pred_col,
        "holdout_rule": f"exclude year >= {args.holdout_year}",
        "input_path": str(input_path),
        "feature_columns": feat_cols,
        "categorical_features": categorical_cols,
        "output_paths": {
            "oof": str(outputs["oof"]),
            "cv_metrics": str(outputs["metrics"]),
            "main_model": str(outputs["model"]),
            "all_years_model": str(outputs["all_years_model"]),
            "holdout": str(outputs["holdout"]),
        },
        "cv_summary": metrics_summary,
        "final_train_summary": {
            "main_model_years": recent_years,
            "main_model_rows": int(len(recent_df)),
            "all_years_model_years": years,
            "all_years_model_rows": int(len(all_df)),
        },
        "code_hash": hash_files([Path(__file__)]),
    }


def main(
    argv: list[str] | None = None,
    *,
    default_task: str | None = None,
    default_model: str | None = None,
) -> int:
    args = parse_args(argv, default_task=default_task, default_model=default_model)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
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
    folds = build_rolling_year_folds(
        years,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )

    drop_cols = {label_col}
    feat_cols = feature_columns(frame, extra_drop=drop_cols)
    feat_cols = _drop_entity_features(feat_cols, bool(args.drop_entity_id_features))
    if not feat_cols:
        raise SystemExit("No feature columns available")
    categorical_cols = [c for c in ["jockey_key", "trainer_key"] if c in feat_cols]

    logger.info(
        "%s-%s train years=%s folds=%s window=%s holdout_year>=%s",
        args.task,
        args.model,
        years,
        len(folds),
        args.train_window_years,
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
    final_iterations = int(np.median(best_iterations))
    final_iterations = max(1, min(final_iterations, int(args.num_boost_round)))

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
        fold_metrics=fold_metrics,
        frame=frame,
        oof=oof,
        holdout_rows=holdout_rows,
        holdout_races=holdout_races,
        years=years,
        final_iterations=final_iterations,
        args=args,
        label_col=label_col,
        pred_col=pred_col,
    )
    save_json(outputs["metrics"], metrics_payload)

    meta_payload = _build_meta_payload(
        args=args,
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
    )
    save_json(outputs["meta"], meta_payload)

    logger.info("wrote %s", outputs["oof"])
    logger.info("wrote %s", outputs["metrics"])
    logger.info("wrote %s", outputs["model"])
    logger.info("wrote %s", outputs["all_years_model"])
    logger.info("wrote %s", outputs["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
