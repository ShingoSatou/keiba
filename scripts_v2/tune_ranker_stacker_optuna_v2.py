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

from scripts_v2.ranker_stacking_v2_common import (  # noqa: E402
    META_METHOD_CHOICES,
    add_meta_features,
    build_meta_walkforward_splits,
    fit_lgbm_ranker,
    fit_logreg_multiclass,
    fit_ridge,
    merge_ranker_oofs,
    ndcg_at_3,
    ndcg_by_year,
    predict_convex,
    predict_lgbm_ranker,
    predict_logreg_expected,
    resolve_path,
    save_json,
    softmax_weights,
)
from scripts_v2.train_ranker_v2 import _coerce_feature_matrix  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune ranker stacker meta-models with Optuna (v2)."
    )
    parser.add_argument("--lgbm-oof", default="data/oof/ranker_oof.parquet")
    parser.add_argument("--xgb-oof", default="data/oof/ranker_xgb_oof.parquet")
    parser.add_argument("--cat-oof", default="data/oof/ranker_cat_oof.parquet")
    parser.add_argument("--trials-output", default="data/oof/ranker_stack_optuna_trials.parquet")
    parser.add_argument("--best-output", default="data/oof/ranker_stack_optuna_best.json")
    parser.add_argument("--storage-dir", default="data/optuna")
    parser.add_argument("--study-prefix", default="ranker_stack_v2")
    parser.add_argument("--n-trials-per-model", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--tune-years", default="2021,2022,2023")
    parser.add_argument("--select-year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _parse_years(raw: str) -> list[int]:
    years: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    uniq = sorted(set(years))
    if not uniq:
        raise ValueError("No years parsed.")
    return uniq


def _params_from_trial(method: str, trial) -> dict[str, Any]:
    if method == "convex":
        logits = [
            float(trial.suggest_float("w_lgbm_logit", -4.0, 4.0)),
            float(trial.suggest_float("w_xgb_logit", -4.0, 4.0)),
            float(trial.suggest_float("w_cat_logit", -4.0, 4.0)),
        ]
        weights = softmax_weights(logits).tolist()
        return {
            "weights": weights,
            "weights_logits": logits,
        }
    if method == "ridge":
        return {"alpha": float(trial.suggest_float("alpha", 1e-4, 1e3, log=True))}
    if method == "logreg_multiclass":
        class_weight_raw = str(trial.suggest_categorical("class_weight", ["none", "balanced"]))
        class_weight = None if class_weight_raw == "none" else "balanced"
        return {
            "C": float(trial.suggest_float("C", 1e-3, 1e3, log=True)),
            "class_weight": class_weight,
            "max_iter": 3000,
        }
    if method == "lgbm_ranker":
        return {
            "n_estimators": int(trial.suggest_int("n_estimators", 200, 2000)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.20, log=True)),
            "num_leaves": int(trial.suggest_int("num_leaves", 15, 127)),
            "min_child_samples": int(trial.suggest_int("min_child_samples", 20, 1000)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)),
            "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)),
            "feature_fraction": float(trial.suggest_float("feature_fraction", 0.6, 1.0)),
            "bagging_fraction": float(trial.suggest_float("bagging_fraction", 0.6, 1.0)),
            "bagging_freq": int(trial.suggest_int("bagging_freq", 0, 10)),
        }
    raise ValueError(f"Unknown method: {method}")


def _predict_method(
    method: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "convex":
        weights = np.asarray(params["weights"], dtype=float)
        return predict_convex(valid_df, weights), {}
    if method == "ridge":
        model = fit_ridge(train_df, feature_cols, alpha=float(params["alpha"]))
        X_valid = _coerce_feature_matrix(valid_df, feature_cols)
        return model.predict(X_valid), {}
    if method == "logreg_multiclass":
        model = fit_logreg_multiclass(
            train_df,
            feature_cols,
            c_value=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=int(params["max_iter"]),
        )
        return predict_logreg_expected(model, valid_df, feature_cols), {}
    if method == "lgbm_ranker":
        model, best_iteration = fit_lgbm_ranker(
            train_df,
            valid_df,
            feature_cols,
            params=params,
            seed=int(seed),
            early_stopping_rounds=int(early_stopping_rounds),
        )
        preds = predict_lgbm_ranker(
            model,
            valid_df,
            feature_cols,
            best_iteration=int(best_iteration),
        )
        return preds, {"best_iteration": int(best_iteration)}
    raise ValueError(f"Unknown method: {method}")


def _evaluate_on_select_year(
    method: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_years: list[int],
    select_year: int,
    seed: int,
    early_stopping_rounds: int,
) -> float:
    effective_train_years = [int(year) for year in train_years if int(year) != int(select_year)]
    train_df = frame[frame["valid_year"].isin(effective_train_years)].copy()
    valid_df = frame[frame["valid_year"] == int(select_year)].copy()
    if train_df.empty or valid_df.empty:
        return float("nan")
    preds, _ = _predict_method(
        method,
        params,
        train_df,
        valid_df,
        feature_cols,
        seed=int(seed) + int(select_year),
        early_stopping_rounds=int(early_stopping_rounds),
    )
    eval_df = valid_df[["race_id", "target_label"]].copy()
    eval_df["stack_score"] = preds
    return ndcg_at_3(eval_df, "stack_score")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("optuna is not installed. Run `uv sync --extra optuna`.") from exc

    if args.n_trials_per_model <= 0:
        raise SystemExit("--n-trials-per-model must be > 0")
    if args.timeout < 0:
        raise SystemExit("--timeout must be >= 0")
    if args.early_stopping_rounds <= 0:
        raise SystemExit("--early-stopping-rounds must be > 0")

    lgbm_oof = resolve_path(args.lgbm_oof)
    xgb_oof = resolve_path(args.xgb_oof)
    cat_oof = resolve_path(args.cat_oof)
    trials_output = resolve_path(args.trials_output)
    best_output = resolve_path(args.best_output)
    storage_dir = resolve_path(args.storage_dir)

    for path in (lgbm_oof, xgb_oof, cat_oof):
        if not path.exists():
            raise SystemExit(f"OOF file not found: {path}")

    merged = merge_ranker_oofs(lgbm_oof, xgb_oof, cat_oof)
    merged, feature_cols = add_meta_features(merged)

    available_years = sorted(merged["valid_year"].unique().tolist())
    tune_years = _parse_years(args.tune_years)
    if int(args.select_year) in tune_years:
        raise SystemExit("--tune-years must not include --select-year.")
    missing_tune_years = sorted(set(tune_years) - set(available_years))
    if missing_tune_years:
        raise SystemExit(
            "tune_years not found in available_valid_years: "
            f"missing={missing_tune_years}, available={available_years}"
        )
    tune_frame = merged[merged["valid_year"].isin(tune_years)].copy()
    if tune_frame.empty:
        raise SystemExit(
            f"No rows for tune_years={tune_years}. available_valid_years={available_years}"
        )
    if int(args.select_year) not in available_years:
        raise SystemExit(
            f"select_year={args.select_year} not in available_valid_years={available_years}"
        )

    tune_splits = build_meta_walkforward_splits(tune_years)
    logger.info(
        "stack tuning methods=%s tune_years=%s select_year=%s splits=%s",
        list(META_METHOD_CHOICES),
        tune_years,
        args.select_year,
        [(a, b) for a, b in tune_splits],
    )

    timeout = None if int(args.timeout) == 0 else int(args.timeout)
    storage_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, Any]] = []
    method_summaries: dict[str, Any] = {}

    for method in META_METHOD_CHOICES:
        storage_path = storage_dir / f"{args.study_prefix}_{method}.sqlite3"
        storage_url = f"sqlite:///{storage_path}"
        sampler = optuna.samplers.TPESampler(seed=int(args.seed))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20, interval_steps=1)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{args.study_prefix}_{method}",
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True,
        )

        def objective(trial, *, _method: str = method) -> float:
            params = _params_from_trial(_method, trial)
            fold_scores: dict[int, float] = {}
            fold_best_iter: dict[int, int] = {}
            values: list[float] = []
            for split_idx, (train_years, valid_year) in enumerate(tune_splits, start=1):
                train_df = tune_frame[tune_frame["valid_year"].isin(train_years)].copy()
                valid_df = tune_frame[tune_frame["valid_year"] == valid_year].copy()
                if train_df.empty or valid_df.empty:
                    raise RuntimeError(
                        "empty split during tuning "
                        f"method={_method} train_years={train_years} valid_year={valid_year}"
                    )

                preds, extra = _predict_method(
                    _method,
                    params,
                    train_df,
                    valid_df,
                    feature_cols,
                    seed=int(args.seed) + int(trial.number) + split_idx,
                    early_stopping_rounds=int(args.early_stopping_rounds),
                )
                eval_df = valid_df[["race_id", "target_label"]].copy()
                eval_df["stack_score"] = preds
                score = ndcg_at_3(eval_df, "stack_score")
                fold_scores[int(valid_year)] = float(score)
                if "best_iteration" in extra:
                    fold_best_iter[int(valid_year)] = int(extra["best_iteration"])
                values.append(float(score))
                trial.report(float(np.mean(values)), step=split_idx)
                if trial.should_prune():
                    trial.set_user_attr("params", params)
                    trial.set_user_attr("fold_ndcg_at_3", fold_scores)
                    trial.set_user_attr("fold_best_iteration", fold_best_iter)
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(values))
            trial.set_user_attr("params", params)
            trial.set_user_attr("fold_ndcg_at_3", fold_scores)
            trial.set_user_attr("fold_best_iteration", fold_best_iter)
            return mean_score

        existing = len(study.trials)
        add_trials = max(0, int(args.n_trials_per_model) - existing)
        logger.info(
            "method=%s existing_trials=%s target=%s additional=%s",
            method,
            existing,
            args.n_trials_per_model,
            add_trials,
        )
        if add_trials > 0:
            study.optimize(objective, n_trials=add_trials, timeout=timeout, catch=(Exception,))

        for trial in study.trials:
            row: dict[str, Any] = {
                "method": method,
                "trial_number": int(trial.number),
                "state": str(trial.state.name),
                "value_mean_ndcg_at_3": None if trial.value is None else float(trial.value),
            }
            trial_params = trial.user_attrs.get("params", {})
            for key, value in trial_params.items():
                if isinstance(value, list):
                    row[f"param/{key}"] = json.dumps(value, ensure_ascii=False)
                else:
                    row[f"param/{key}"] = value
            for year, value in (trial.user_attrs.get("fold_ndcg_at_3") or {}).items():
                row[f"fold/{int(year)}/ndcg_at_3"] = float(value)
            for year, value in (trial.user_attrs.get("fold_best_iteration") or {}).items():
                row[f"fold/{int(year)}/best_iteration"] = int(value)
            trial_rows.append(row)

        best_trial = study.best_trial
        best_params = best_trial.user_attrs.get("params", {})
        select_score = _evaluate_on_select_year(
            method,
            best_params,
            merged,
            feature_cols,
            train_years=tune_years,
            select_year=int(args.select_year),
            seed=int(args.seed) + 9999,
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        method_summaries[method] = {
            "study_name": study.study_name,
            "storage": str(storage_path),
            "target_trials": int(args.n_trials_per_model),
            "total_trials": int(len(study.trials)),
            "best_trial_number": int(best_trial.number),
            "best_value_mean_ndcg_at_3": float(study.best_value),
            "best_params": best_params,
            "best_fold_ndcg_at_3": best_trial.user_attrs.get("fold_ndcg_at_3", {}),
            "best_fold_best_iteration": best_trial.user_attrs.get("fold_best_iteration", {}),
            "select_year": int(args.select_year),
            "select_year_ndcg_at_3": float(select_score),
        }

    trials_frame = pd.DataFrame(trial_rows).sort_values(
        ["method", "trial_number"], kind="mergesort"
    )
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    trials_frame.to_parquet(trials_output, index=False)
    logger.info("wrote %s rows=%s", trials_output, len(trials_frame))

    ranked_methods = sorted(
        method_summaries.items(),
        key=lambda item: float(item[1]["select_year_ndcg_at_3"]),
        reverse=True,
    )
    selected_method = ranked_methods[0][0]
    selected_summary = ranked_methods[0][1]

    base_select_frame = merged[merged["valid_year"] == int(args.select_year)].copy()
    base_model_scores = {
        "lgbm": ndcg_at_3(
            base_select_frame[["race_id", "target_label", "lgbm_score"]].rename(
                columns={"lgbm_score": "score"}
            ),
            "score",
        ),
        "xgb": ndcg_at_3(
            base_select_frame[["race_id", "target_label", "xgb_score"]].rename(
                columns={"xgb_score": "score"}
            ),
            "score",
        ),
        "cat": ndcg_at_3(
            base_select_frame[["race_id", "target_label", "cat_score"]].rename(
                columns={"cat_score": "score"}
            ),
            "score",
        ),
    }

    best_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": {
            "lgbm_oof": str(lgbm_oof),
            "xgb_oof": str(xgb_oof),
            "cat_oof": str(cat_oof),
        },
        "config": {
            "methods": list(META_METHOD_CHOICES),
            "n_trials_per_model": int(args.n_trials_per_model),
            "timeout": int(args.timeout),
            "tune_years": tune_years,
            "select_year": int(args.select_year),
            "seed": int(args.seed),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "feature_columns": feature_cols,
        },
        "data_summary": {
            "rows": int(len(merged)),
            "races": int(merged["race_id"].nunique()),
            "valid_years": available_years,
            "rows_by_valid_year": {
                str(year): int(count)
                for year, count in merged.groupby("valid_year", sort=True).size().items()
            },
        },
        "methods": method_summaries,
        "selected_method": selected_method,
        "selected_method_summary": selected_summary,
        "base_model_select_year_ndcg_at_3": base_model_scores,
        "base_model_ndcg_by_year": {
            "lgbm": ndcg_by_year(
                merged[["valid_year", "race_id", "target_label", "lgbm_score"]].rename(
                    columns={"lgbm_score": "score"}
                ),
                "score",
            ),
            "xgb": ndcg_by_year(
                merged[["valid_year", "race_id", "target_label", "xgb_score"]].rename(
                    columns={"xgb_score": "score"}
                ),
                "score",
            ),
            "cat": ndcg_by_year(
                merged[["valid_year", "race_id", "target_label", "cat_score"]].rename(
                    columns={"cat_score": "score"}
                ),
                "score",
            ),
        },
    }
    save_json(best_output, best_payload)
    logger.info("wrote %s", best_output)
    logger.info(
        "selected_method=%s select_year_ndcg@3=%.6f",
        selected_method,
        float(selected_summary["select_year_ndcg_at_3"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
