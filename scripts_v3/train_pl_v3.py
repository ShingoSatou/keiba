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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_v3.feature_registry_v3 import (  # noqa: E402
    PL_REQUIRED_PRED_FEATURES,
    get_pl_feature_columns,
)
from scripts_v3.pl_v3_common import (  # noqa: E402
    PLTrainConfig,
    build_group_indices,
    estimate_p_top3_by_race,
    estimate_p_wide_by_race,
    fit_pl_linear_torch,
    pl_nll_numpy,
    predict_linear_scores,
)
from scripts_v3.train_binary_v3_common import (  # noqa: E402
    DEFAULT_CV_WINDOW_POLICY,
    DEFAULT_TRAIN_WINDOW_YEARS,
    attach_cv_policy_columns,
    build_cv_policy_payload,
    build_fixed_window_year_folds,
    compute_binary_metrics,
    fold_integrity,
    hash_files,
    make_window_definition,
    resolve_path,
    save_json,
    select_recent_window_years,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.05
DEFAULT_L2 = 1e-5
DEFAULT_SEED = 42
DEFAULT_MC_SAMPLES = 10_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train v3 PL(uなし) layer with fixed-length sliding yearly CV "
            "using OOF predictions."
        )
    )
    parser.add_argument("--features-input", default="data/features_v3.parquet")
    parser.add_argument("--win-lgbm-oof", default="data/oof/win_lgbm_oof.parquet")
    parser.add_argument("--win-xgb-oof", default="data/oof/win_xgb_oof.parquet")
    parser.add_argument("--win-cat-oof", default="data/oof/win_cat_oof.parquet")
    parser.add_argument("--place-lgbm-oof", default="data/oof/place_lgbm_oof.parquet")
    parser.add_argument("--place-xgb-oof", default="data/oof/place_xgb_oof.parquet")
    parser.add_argument("--place-cat-oof", default="data/oof/place_cat_oof.parquet")
    parser.add_argument(
        "--odds-cal-oof",
        default="data/oof/odds_win_calibration_oof.parquet",
        help="Optional. If file exists, *_cal_* columns are merged.",
    )
    parser.add_argument(
        "--odds-cal-cols",
        default="",
        help=(
            "Comma-separated calibration columns to use. Empty means auto-detect from odds-cal-oof."
        ),
    )
    parser.add_argument(
        "--include-final-odds-features",
        action="store_true",
        help=(
            "Include final-odds columns (検証専用). "
            "Default is t10-only features for operational parity."
        ),
    )

    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        help="The v3 standard comparison condition is fixed_sliding with 4 years.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)

    parser.add_argument("--oof-output", default="data/oof/pl_v3_oof.parquet")
    parser.add_argument("--wide-oof-output", default="data/oof/pl_v3_wide_oof.parquet")
    parser.add_argument("--emit-wide-oof", action="store_true")
    parser.add_argument("--metrics-output", default="data/oof/pl_v3_cv_metrics.json")
    parser.add_argument("--model-output", default="models/pl_v3_recent_window.joblib")
    parser.add_argument("--all-years-model-output", default="models/pl_v3_all_years.joblib")
    parser.add_argument("--meta-output", default="models/pl_v3_bundle_meta.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.holdout_year) <= 0:
        raise SystemExit("--holdout-year must be > 0")
    if int(args.train_window_years) <= 0:
        raise SystemExit("--train-window-years must be > 0")
    if int(args.epochs) <= 0:
        raise SystemExit("--epochs must be > 0")
    if float(args.lr) <= 0.0:
        raise SystemExit("--lr must be > 0")
    if float(args.l2) < 0.0:
        raise SystemExit("--l2 must be >= 0")
    if int(args.mc_samples) <= 0:
        raise SystemExit("--mc-samples must be > 0")


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _dedupe_preserve_order(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def _operational_mode(include_final_odds_features: bool) -> str:
    return "includes_final" if bool(include_final_odds_features) else "t10_only"


def _prep_base_frame(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise SystemExit(f"features input not found: {features_path}")
    frame = pd.read_parquet(features_path)
    required = {
        "race_id",
        "horse_id",
        "horse_no",
        "race_date",
        "finish_pos",
        "field_size",
        "y_win",
        "y_place",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in features_v3: {missing}")

    out = frame.copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out["horse_id"] = out["horse_id"].astype(str)

    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    out = out[out["race_date"].notna()].copy()
    out["year"] = out["race_date"].dt.year.astype(int)
    out["finish_pos"] = pd.to_numeric(out["finish_pos"], errors="coerce")
    out["field_size"] = pd.to_numeric(out["field_size"], errors="coerce")
    out["y_win"] = pd.to_numeric(out["y_win"], errors="coerce").fillna(0).astype(int)
    out["y_place"] = pd.to_numeric(out["y_place"], errors="coerce").fillna(0).astype(int)
    out["y_top3"] = np.where(
        out["finish_pos"].notna(),
        (out["finish_pos"] <= 3).astype(int),
        np.nan,
    )

    if out.duplicated(["race_id", "horse_no"]).any():
        dup = out[out.duplicated(["race_id", "horse_no"], keep=False)][
            ["race_id", "horse_no"]
        ].head()
        raise SystemExit(f"Duplicate (race_id, horse_no) in features_v3: {dup.to_dict('records')}")
    return out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def _load_single_oof_prediction(path: Path, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"OOF file not found: {path}")
    df = pd.read_parquet(path)
    required = {"race_id", "horse_no", pred_col, "valid_year"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")

    out = df[["race_id", "horse_no", pred_col, "valid_year"]].copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out["valid_year"] = pd.to_numeric(out["valid_year"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    if out.duplicated(["race_id", "horse_no"]).any():
        raise SystemExit(f"Duplicate keys in OOF file: {path}")
    return out[["race_id", "horse_no", pred_col]]


def _load_odds_calibration_oof(path: Path, cols_hint: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return pd.DataFrame(columns=["race_id", "horse_no"]), []
    df = pd.read_parquet(path)
    base_required = {"race_id", "horse_no", "valid_year"}
    missing = sorted(base_required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns in odds calibration OOF {path}: {missing}")

    if cols_hint:
        cal_cols = [col for col in cols_hint if col in df.columns]
        missing_hint = sorted(set(cols_hint) - set(cal_cols))
        if missing_hint:
            raise SystemExit(f"--odds-cal-cols not found in {path}: {missing_hint}")
    else:
        cal_cols = sorted(
            col for col in df.columns if "_cal_" in col and (("_t10_" in col) or ("_final_" in col))
        )
    if not cal_cols:
        return pd.DataFrame(columns=["race_id", "horse_no"]), []

    out = df[["race_id", "horse_no", "valid_year", *cal_cols]].copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    if out.duplicated(["race_id", "horse_no"]).any():
        raise SystemExit(f"Duplicate keys in odds calibration OOF: {path}")
    for col in cal_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["race_id", "horse_no", *cal_cols]], cal_cols


def _merge_prediction_features(
    args: argparse.Namespace,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    merged = frame.copy()

    win_paths = {
        "lgbm": resolve_path(args.win_lgbm_oof),
        "xgb": resolve_path(args.win_xgb_oof),
        "cat": resolve_path(args.win_cat_oof),
    }
    place_paths = {
        "lgbm": resolve_path(args.place_lgbm_oof),
        "xgb": resolve_path(args.place_xgb_oof),
        "cat": resolve_path(args.place_cat_oof),
    }
    pred_paths = {
        "p_win_lgbm": win_paths["lgbm"],
        "p_win_xgb": win_paths["xgb"],
        "p_win_cat": win_paths["cat"],
        "p_place_lgbm": place_paths["lgbm"],
        "p_place_xgb": place_paths["xgb"],
        "p_place_cat": place_paths["cat"],
    }

    for pred_col in PL_REQUIRED_PRED_FEATURES:
        pred_df = _load_single_oof_prediction(pred_paths[pred_col], pred_col)
        merged = merged.merge(pred_df, on=["race_id", "horse_no"], how="left")

    cal_cols_hint = _parse_csv(args.odds_cal_cols)
    cal_df, cal_cols = _load_odds_calibration_oof(resolve_path(args.odds_cal_oof), cal_cols_hint)
    if cal_cols:
        merged = merged.merge(cal_df, on=["race_id", "horse_no"], how="left")
    required_pred_cols = [col for col in PL_REQUIRED_PRED_FEATURES if col in merged.columns]
    for col in sorted(cal_cols):
        if "_t10_" in col or bool(args.include_final_odds_features):
            required_pred_cols.append(col)
    return merged, _dedupe_preserve_order(required_pred_cols)


def _collect_pl_feature_columns(
    frame: pd.DataFrame,
    *,
    required_pred_cols: list[str],
    include_final_odds: bool,
    operational_mode: str,
) -> list[str]:
    return get_pl_feature_columns(
        frame,
        required_pred_cols=required_pred_cols,
        include_context=True,
        include_final_odds=include_final_odds,
        operational_mode=operational_mode,
    )


def _build_matrix_with_stats(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    stats: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    mat = frame[feature_cols].copy()
    for col in feature_cols:
        mat[col] = pd.to_numeric(mat[col], errors="coerce")

    if stats is None:
        medians = mat.median(axis=0, skipna=True).fillna(0.0)
        mat = mat.fillna(medians).fillna(0.0)
        means = mat.mean(axis=0).fillna(0.0)
        stds = mat.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        stats_out = {
            "median": {col: float(medians[col]) for col in feature_cols},
            "mean": {col: float(means[col]) for col in feature_cols},
            "std": {col: float(stds[col]) for col in feature_cols},
        }
    else:
        medians = pd.Series({col: float(stats["median"][col]) for col in feature_cols})
        means = pd.Series({col: float(stats["mean"][col]) for col in feature_cols})
        stds = pd.Series({col: float(stats["std"][col]) for col in feature_cols}).replace(0.0, 1.0)
        mat = mat.fillna(medians).fillna(0.0)
        stats_out = stats

    mat = (mat - means) / stds
    return mat.to_numpy(dtype=float), stats_out


def _fit_pl_on_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, float]]:
    train_df = frame.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    x_train, stats = _build_matrix_with_stats(train_df, feature_cols, stats=None)
    groups = build_group_indices(
        train_df,
        race_col="race_id",
        finish_col="finish_pos",
        horse_no_col="horse_no",
    )
    if not groups:
        raise SystemExit(
            "No valid race groups found for PL training (need finish_pos and >=2 horses)."
        )

    config = PLTrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed) + int(seed_offset),
    )
    weights, train_info = fit_pl_linear_torch(x_train, groups, config=config)
    train_nll = pl_nll_numpy(predict_linear_scores(x_train, weights), groups)
    info = {"train_nll": float(train_nll), "optimizer_final_nll": float(train_info["train_nll"])}
    return weights, stats, info


def _score_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    weights: np.ndarray,
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    x, _ = _build_matrix_with_stats(frame, feature_cols, stats=stats)
    return predict_linear_scores(x, weights)


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


def _artifact_from_fit(
    *,
    feature_cols: list[str],
    weights: np.ndarray,
    stats: dict[str, dict[str, float]],
    args: argparse.Namespace,
    train_years: list[int],
    train_rows: int,
    train_races: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_type": "pl_linear_torch_v3",
        "feature_columns": list(feature_cols),
        "weights": np.asarray(weights, dtype=float).tolist(),
        "preprocess": stats,
        "config": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "seed": int(args.seed),
            "mc_samples_default": int(args.mc_samples),
            "top_k": 3,
        },
        "operational_mode": _operational_mode(bool(args.include_final_odds_features)),
        "cv_policy": {
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "train_window_years": int(args.train_window_years),
            "valid_years": [],
            "holdout_year": int(args.holdout_year),
            "window_definition": make_window_definition(int(args.train_window_years)),
        },
        "train_summary": {
            "years": list(map(int, train_years)),
            "rows": int(train_rows),
            "races": int(train_races),
        },
    }


def _run_pl_cv_loop(
    *,
    eligible: pd.DataFrame,
    folds: list,
    pl_feature_cols: list[str],
    required_pred_cols: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]]]:
    """PL の CV loop を実行し、OOF / wide_oof / fold metrics を返す。"""
    oof_list: list[pd.DataFrame] = []
    wide_oof_list: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    window_definition = make_window_definition(int(args.train_window_years))

    for fold in folds:
        train_df = eligible[eligible["year"].isin(fold.train_years)].copy()
        valid_df = eligible[eligible["year"] == fold.valid_year].copy()
        if train_df.empty or valid_df.empty:
            raise SystemExit(
                f"fold={fold.fold_id} empty split: train={len(train_df)} valid={len(valid_df)}"
            )
        fold_integrity(train_df, valid_df, int(fold.valid_year))

        train_df = train_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
            drop=True
        )
        valid_df = valid_df.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(
            drop=True
        )

        weights, stats, train_info = _fit_pl_on_frame(
            train_df,
            feature_cols=pl_feature_cols,
            args=args,
            seed_offset=int(fold.fold_id),
        )
        valid_scores = _score_frame(
            valid_df,
            feature_cols=pl_feature_cols,
            weights=weights,
            stats=stats,
        )

        valid_groups = build_group_indices(
            valid_df,
            race_col="race_id",
            finish_col="finish_pos",
            horse_no_col="horse_no",
        )
        valid_nll = pl_nll_numpy(valid_scores, valid_groups)

        scored = valid_df[
            [
                c
                for c in [
                    "race_id",
                    "horse_id",
                    "horse_no",
                    "race_date",
                    "t_race",
                    "field_size",
                    "target_label",
                    "finish_pos",
                    "y_win",
                    "y_place",
                    "y_top3",
                ]
                if c in valid_df.columns
            ]
        ].copy()
        scored["pl_score"] = valid_scores

        p_top3 = estimate_p_top3_by_race(
            scored[["race_id", "horse_no", "pl_score"]],
            score_col="pl_score",
            mc_samples=int(args.mc_samples),
            seed=int(args.seed) + int(fold.fold_id),
            top_k=3,
        )
        scored = scored.merge(p_top3, on=["race_id", "horse_no"], how="left")
        scored["fold_id"] = int(fold.fold_id)
        scored["valid_year"] = int(fold.valid_year)
        scored = attach_cv_policy_columns(
            scored,
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
            window_definition=window_definition,
        )
        scored = scored.sort_values(["race_id", "horse_no"], kind="mergesort")
        oof_list.append(scored)

        valid_sub = scored[scored["p_top3"].notna() & scored["y_top3"].notna()].copy()
        top3_metrics: dict[str, Any]
        if valid_sub.empty:
            top3_metrics = {"logloss": None, "brier": None, "auc": None, "ece": None}
        else:
            m = compute_binary_metrics(
                valid_sub["y_top3"].to_numpy(dtype=int),
                valid_sub["p_top3"].to_numpy(dtype=float),
                n_bins=10,
            )
            top3_metrics = {
                "logloss": m["logloss"],
                "brier": m["brier"],
                "auc": m["auc"],
                "ece": m["ece"],
            }

        fold_metric: dict[str, Any] = {
            "fold_id": int(fold.fold_id),
            "train_years": list(map(int, fold.train_years)),
            "valid_year": int(fold.valid_year),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_races": int(train_df["race_id"].nunique()),
            "valid_races": int(valid_df["race_id"].nunique()),
            "feature_count": int(len(pl_feature_cols)),
            "pl_nll_valid": float(valid_nll) if np.isfinite(valid_nll) else None,
            "train_nll": float(train_info["train_nll"]),
            "optimizer_final_nll": float(train_info["optimizer_final_nll"]),
            "top3_logloss": top3_metrics["logloss"],
            "top3_brier": top3_metrics["brier"],
            "top3_auc": top3_metrics["auc"],
            "top3_ece": top3_metrics["ece"],
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "train_window_years": int(args.train_window_years),
            "holdout_year": int(args.holdout_year),
            "window_definition": window_definition,
        }
        fold_metrics.append(fold_metric)
        logger.info(
            "fold=%s valid_year=%s pl_nll=%s top3_logloss=%s",
            fold.fold_id,
            fold.valid_year,
            (
                f"{fold_metric['pl_nll_valid']:.6f}"
                if fold_metric["pl_nll_valid"] is not None
                else "None"
            ),
            (
                f"{fold_metric['top3_logloss']:.6f}"
                if fold_metric["top3_logloss"] is not None
                else "None"
            ),
        )

        if bool(args.emit_wide_oof):
            wide = estimate_p_wide_by_race(
                scored[["race_id", "horse_no", "pl_score"]],
                score_col="pl_score",
                mc_samples=int(args.mc_samples),
                seed=int(args.seed) + int(fold.fold_id),
                top_k=3,
            )
            if not wide.empty:
                wide["fold_id"] = int(fold.fold_id)
                wide["valid_year"] = int(fold.valid_year)
                wide = attach_cv_policy_columns(
                    wide,
                    train_window_years=int(args.train_window_years),
                    holdout_year=int(args.holdout_year),
                    cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
                    window_definition=window_definition,
                )
                wide_oof_list.append(wide)

    if not oof_list:
        raise SystemExit("No PL OOF predictions generated.")

    oof = pd.concat(oof_list, axis=0, ignore_index=True).sort_values(
        ["race_id", "horse_no"], kind="mergesort"
    )
    return oof, wide_oof_list, fold_metrics


def _train_pl_final_models(
    *,
    eligible: pd.DataFrame,
    years: list[int],
    pl_feature_cols: list[str],
    args: argparse.Namespace,
    cv_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """recent window と all-years の PL 最終モデルを学習する。"""
    recent_years = select_recent_window_years(
        years,
        train_window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    recent_df = eligible[eligible["year"].isin(recent_years)].copy()
    all_df = eligible.copy()

    w_recent, stats_recent, _ = _fit_pl_on_frame(
        recent_df,
        feature_cols=pl_feature_cols,
        args=args,
        seed_offset=1000,
    )
    artifact_recent = _artifact_from_fit(
        feature_cols=pl_feature_cols,
        weights=w_recent,
        stats=stats_recent,
        args=args,
        train_years=recent_years,
        train_rows=int(len(recent_df)),
        train_races=int(recent_df["race_id"].nunique()),
    )
    artifact_recent["cv_policy"] = dict(cv_policy)

    w_all, stats_all, _ = _fit_pl_on_frame(
        all_df,
        feature_cols=pl_feature_cols,
        args=args,
        seed_offset=2000,
    )
    artifact_all = _artifact_from_fit(
        feature_cols=pl_feature_cols,
        weights=w_all,
        stats=stats_all,
        args=args,
        train_years=years,
        train_rows=int(len(all_df)),
        train_races=int(all_df["race_id"].nunique()),
    )
    artifact_all["cv_policy"] = dict(cv_policy)
    artifact_all["analysis_only"] = True

    return artifact_recent, artifact_all, recent_df, all_df


def _build_pl_metrics_payload(
    *,
    folds: list,
    fold_metrics: list[dict[str, Any]],
    eligible: pd.DataFrame,
    oof: pd.DataFrame,
    years: list[int],
    required_pred_cols: list[str],
    pl_feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """PL メトリクスの集計辞書を組み立てる。"""
    summary = {
        "pl_nll_valid": _summary([f.get("pl_nll_valid") for f in fold_metrics]),
        "top3_logloss": _summary([f.get("top3_logloss") for f in fold_metrics]),
        "top3_brier": _summary([f.get("top3_brier") for f in fold_metrics]),
        "top3_auc": _summary([f.get("top3_auc") for f in fold_metrics]),
        "top3_ece": _summary([f.get("top3_ece") for f in fold_metrics]),
    }
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cv_policy": build_cv_policy_payload(
            folds,
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        ),
        "config": {
            "holdout_year": int(args.holdout_year),
            "train_window_years": int(args.train_window_years),
            "cv_window_policy": DEFAULT_CV_WINDOW_POLICY,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "seed": int(args.seed),
            "mc_samples": int(args.mc_samples),
            "operational_mode": _operational_mode(bool(args.include_final_odds_features)),
            "include_final_odds_features": bool(args.include_final_odds_features),
        },
        "data_summary": {
            "rows": int(len(eligible)),
            "races": int(eligible["race_id"].nunique()),
            "years": years,
            "oof_rows": int(len(oof)),
            "oof_races": int(oof["race_id"].nunique()),
        },
        "required_pred_cols": required_pred_cols,
        "feature_columns": pl_feature_cols,
        "folds": fold_metrics,
        "summary": summary,
    }


def _build_pl_meta_payload(
    *,
    args: argparse.Namespace,
    folds: list,
    features_path: Path,
    oof_output: Path,
    wide_oof_output: Path,
    metrics_output: Path,
    model_output: Path,
    all_years_model_output: Path,
    required_pred_cols: list[str],
    pl_feature_cols: list[str],
    cv_summary: dict[str, Any],
    recent_df: pd.DataFrame,
    all_df: pd.DataFrame,
    years: list[int],
) -> dict[str, Any]:
    """PL バンドルメタ辞書を組み立てる。"""
    recent_years = sorted(recent_df["year"].unique().tolist())
    return {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "holdout_rule": f"exclude year >= {args.holdout_year}",
        "cv_policy": build_cv_policy_payload(
            folds,
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        ),
        "operational_mode": _operational_mode(bool(args.include_final_odds_features)),
        "input_paths": {
            "features_v3": str(features_path),
            "win_lgbm_oof": str(resolve_path(args.win_lgbm_oof)),
            "win_xgb_oof": str(resolve_path(args.win_xgb_oof)),
            "win_cat_oof": str(resolve_path(args.win_cat_oof)),
            "place_lgbm_oof": str(resolve_path(args.place_lgbm_oof)),
            "place_xgb_oof": str(resolve_path(args.place_xgb_oof)),
            "place_cat_oof": str(resolve_path(args.place_cat_oof)),
            "odds_cal_oof": str(resolve_path(args.odds_cal_oof)),
        },
        "output_paths": {
            "oof": str(oof_output),
            "wide_oof": str(wide_oof_output) if bool(args.emit_wide_oof) else None,
            "metrics": str(metrics_output),
            "main_model": str(model_output),
            "all_years_model": str(all_years_model_output),
        },
        "required_pred_cols": required_pred_cols,
        "feature_columns": pl_feature_cols,
        "cv_summary": cv_summary,
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
                Path(resolve_path("scripts_v3/pl_v3_common.py")),
                Path(resolve_path("scripts_v3/feature_registry_v3.py")),
            ]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    _validate_args(args)

    features_path = resolve_path(args.features_input)
    oof_output = resolve_path(args.oof_output)
    wide_oof_output = resolve_path(args.wide_oof_output)
    metrics_output = resolve_path(args.metrics_output)
    model_output = resolve_path(args.model_output)
    all_years_model_output = resolve_path(args.all_years_model_output)
    meta_output = resolve_path(args.meta_output)
    operational_mode = _operational_mode(bool(args.include_final_odds_features))

    base = _prep_base_frame(features_path)
    merged, required_pred_cols = _merge_prediction_features(args, base)
    if not required_pred_cols:
        raise SystemExit("No required OOF prediction columns found for PL.")

    missing_req = [col for col in required_pred_cols if col not in merged.columns]
    if missing_req:
        raise SystemExit(f"Missing required OOF columns after merge: {missing_req}")

    pl_feature_cols = _collect_pl_feature_columns(
        merged,
        required_pred_cols=required_pred_cols,
        include_final_odds=bool(args.include_final_odds_features),
        operational_mode=operational_mode,
    )
    if not pl_feature_cols:
        raise SystemExit("No PL feature columns available.")

    # OOF予測が揃わない行は後段PL学習対象から除外（OOF stacking原則）
    eligible = merged.copy()
    for col in required_pred_cols:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
    eligible = eligible[eligible[required_pred_cols].notna().all(axis=1)].copy()
    eligible = eligible[eligible["year"] < int(args.holdout_year)].copy()
    if eligible.empty:
        raise SystemExit("No eligible rows for PL training after OOF/holdout filtering.")

    years = sorted(eligible["year"].unique().tolist())
    folds = build_fixed_window_year_folds(
        years,
        window_years=int(args.train_window_years),
        holdout_year=int(args.holdout_year),
    )
    if not folds:
        max_window = max(1, len(years) - 1)
        raise SystemExit(
            "No PL CV folds available under the fixed_sliding policy "
            f"(available_years={years}, try --train-window-years <= {max_window})"
        )
    logger.info(
        "pl-v3 train years=%s folds=%s window=%s cv_policy=%s holdout_year>=%s rows=%s races=%s",
        years,
        len(folds),
        args.train_window_years,
        DEFAULT_CV_WINDOW_POLICY,
        args.holdout_year,
        len(eligible),
        eligible["race_id"].nunique(),
    )

    # --- CV loop ---
    oof, wide_oof_list, fold_metrics = _run_pl_cv_loop(
        eligible=eligible,
        folds=folds,
        pl_feature_cols=pl_feature_cols,
        required_pred_cols=required_pred_cols,
        args=args,
    )

    # --- OOF 保存 ---
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_output, index=False)

    if bool(args.emit_wide_oof):
        wide_oof_output.parent.mkdir(parents=True, exist_ok=True)
        wide_oof = (
            pd.concat(wide_oof_list, axis=0, ignore_index=True)
            if wide_oof_list
            else pd.DataFrame(
                columns=[
                    "race_id",
                    "horse_no_1",
                    "horse_no_2",
                    "kumiban",
                    "p_wide",
                    "p_top3_1",
                    "p_top3_2",
                    "fold_id",
                    "valid_year",
                    "cv_window_policy",
                    "train_window_years",
                    "holdout_year",
                    "window_definition",
                ]
            )
        )
        wide_oof.to_parquet(wide_oof_output, index=False)

    # --- 最終モデル学習 ---
    artifact_recent, artifact_all, recent_df, all_df = _train_pl_final_models(
        eligible=eligible,
        years=years,
        pl_feature_cols=pl_feature_cols,
        args=args,
        cv_policy=build_cv_policy_payload(
            folds,
            train_window_years=int(args.train_window_years),
            holdout_year=int(args.holdout_year),
            cv_window_policy=DEFAULT_CV_WINDOW_POLICY,
        ),
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    all_years_model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_recent, model_output)
    joblib.dump(artifact_all, all_years_model_output)

    # --- メトリクス / メタ保存 ---
    metrics_payload = _build_pl_metrics_payload(
        folds=folds,
        fold_metrics=fold_metrics,
        eligible=eligible,
        oof=oof,
        years=years,
        required_pred_cols=required_pred_cols,
        pl_feature_cols=pl_feature_cols,
        args=args,
    )
    save_json(metrics_output, metrics_payload)

    meta_payload = _build_pl_meta_payload(
        args=args,
        folds=folds,
        features_path=features_path,
        oof_output=oof_output,
        wide_oof_output=wide_oof_output,
        metrics_output=metrics_output,
        model_output=model_output,
        all_years_model_output=all_years_model_output,
        required_pred_cols=required_pred_cols,
        pl_feature_cols=pl_feature_cols,
        cv_summary=metrics_payload["summary"],
        recent_df=recent_df,
        all_df=all_df,
        years=years,
    )
    save_json(meta_output, meta_payload)

    logger.info("wrote %s", oof_output)
    if bool(args.emit_wide_oof):
        logger.info("wrote %s", wide_oof_output)
    logger.info("wrote %s", metrics_output)
    logger.info("wrote %s", model_output)
    logger.info("wrote %s", all_years_model_output)
    logger.info("wrote %s", meta_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
