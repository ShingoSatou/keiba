from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.t5_modeling import (
    FUNDAMENTAL_FEATURES,
    META_FEATURES,
    MODEL_M_FEATURES,
    MODEL_O_FEATURES,
    apply_tau_rule,
    binary_metrics,
    choose_tau,
    coerce_model_matrix,
    evaluate_betting,
    predict_proba,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _detect_git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load_dataset(input_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(input_path)
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame = frame[frame["race_date"].notna()].copy()
    frame["year"] = frame["race_date"].dt.year.astype(int)
    frame["is_win"] = frame["is_win"].astype(int)
    frame["odds_win_final"] = pd.to_numeric(frame.get("odds_win_final"), errors="coerce")
    frame["target_log_odds_final"] = pd.to_numeric(
        frame.get("target_log_odds_final"), errors="coerce"
    )
    frame["market_available_flag"] = (
        pd.to_numeric(frame.get("market_available_flag"), errors="coerce").fillna(0).astype(int)
    )
    return frame.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)


def _build_outer_folds(
    frame: pd.DataFrame,
    min_train_years: int = 3,
    holdout_years: int = 1,
) -> list[dict[str, Any]]:
    years = sorted(frame["year"].dropna().unique().tolist())
    if len(years) < min_train_years + holdout_years + 1:
        raise ValueError(
            "not enough years for walk-forward; "
            f"need >= {min_train_years + holdout_years + 1}, got {len(years)}"
        )

    last_valid_index = len(years) - holdout_years
    folds: list[dict[str, Any]] = []
    for valid_index in range(min_train_years, last_valid_index):
        train_years = years[:valid_index]
        valid_year = years[valid_index]
        folds.append(
            {
                "fold_id": len(folds) + 1,
                "train_years": train_years,
                "valid_year": valid_year,
            }
        )
    return folds


def _split_inner(
    outer_train: pd.DataFrame,
    es_days: int,
    calib_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_date = outer_train["race_date"].max()
    calib_start = max_date - timedelta(days=calib_days - 1)
    es_start = calib_start - timedelta(days=es_days)

    train_fit = outer_train[outer_train["race_date"] < es_start].copy()
    es_valid = outer_train[
        (outer_train["race_date"] >= es_start) & (outer_train["race_date"] < calib_start)
    ].copy()
    calib = outer_train[outer_train["race_date"] >= calib_start].copy()

    if train_fit.empty or es_valid.empty or calib.empty:
        race_dates = sorted(outer_train["race_date"].dropna().unique().tolist())
        n_dates = len(race_dates)
        train_end = max(int(n_dates * 0.7), 1)
        es_end = max(int(n_dates * 0.85), train_end + 1)
        train_dates = set(race_dates[:train_end])
        es_dates = set(race_dates[train_end:es_end])
        calib_dates = set(race_dates[es_end:])
        train_fit = outer_train[outer_train["race_date"].isin(train_dates)].copy()
        es_valid = outer_train[outer_train["race_date"].isin(es_dates)].copy()
        calib = outer_train[outer_train["race_date"].isin(calib_dates)].copy()
    return train_fit, es_valid, calib


def _train_lgb_binary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    num_boost_round: int,
    early_stopping_rounds: int,
    random_state: int,
) -> lgb.LGBMClassifier:
    y_train = train_df["is_win"].astype(int)
    if y_train.nunique() < 2:
        dummy = DummyClassifier(strategy="constant", constant=int(y_train.iloc[0]))
        dummy.fit(coerce_model_matrix(train_df, feature_names), y_train)
        return dummy

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=num_boost_round,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    X_train = coerce_model_matrix(train_df, feature_names)
    X_valid = coerce_model_matrix(valid_df, feature_names)
    y_train = train_df["is_win"].astype(int)
    y_valid = valid_df["is_win"].astype(int)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def _train_lgb_regression(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    num_boost_round: int,
    early_stopping_rounds: int,
    random_state: int,
) -> lgb.LGBMRegressor:
    y_train = train_df["target_log_odds_final"].astype(float)
    if len(train_df) < 2 or y_train.nunique() < 2:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(coerce_model_matrix(train_df, feature_names), y_train)
        return dummy

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=num_boost_round,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    X_train = coerce_model_matrix(train_df, feature_names)
    X_valid = coerce_model_matrix(valid_df, feature_names)
    y_train = train_df["target_log_odds_final"].astype(float)
    y_valid = valid_df["target_log_odds_final"].astype(float)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def _build_meta_frame(
    frame: pd.DataFrame,
    model_f: Any,
    model_m: Any,
    model_o: Any,
) -> pd.DataFrame:
    out = frame.copy()
    out["pF_raw"] = predict_proba(model_f, coerce_model_matrix(out, FUNDAMENTAL_FEATURES))
    out["pM_raw"] = predict_proba(model_m, coerce_model_matrix(out, MODEL_M_FEATURES))
    o_matrix = coerce_model_matrix(out, MODEL_O_FEATURES)
    o_pred_log = np.asarray(model_o.predict(o_matrix), dtype=float)
    out["o_close_hat"] = np.exp(o_pred_log)
    out["o_close_hat"] = np.where(np.isfinite(out["o_close_hat"]), out["o_close_hat"], np.nan)
    out["o_close_hat"] = np.clip(out["o_close_hat"], 1.0, None)
    out["log_o_close_hat"] = np.log(out["o_close_hat"])
    return out


def _train_meta_model(calib_frame: pd.DataFrame) -> Any:
    X = coerce_model_matrix(calib_frame, META_FEATURES)
    y = calib_frame["is_win"].astype(int)
    if y.nunique() < 2:
        dummy = DummyClassifier(strategy="constant", constant=int(y.iloc[0]))
        dummy.fit(X, y)
        return dummy
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    return model


def _finalize_predictions(frame: pd.DataFrame, meta_model: Any) -> pd.DataFrame:
    out = frame.copy()
    meta_matrix = coerce_model_matrix(out, META_FEATURES)
    out["p_final"] = predict_proba(meta_model, meta_matrix)
    out["EV_hat"] = out["p_final"] * out["o_close_hat"] - 1.0
    out["EV_true"] = out["p_final"] * out["odds_win_final"] - 1.0
    out["decision_consistency"] = (
        np.sign(pd.to_numeric(out["EV_hat"], errors="coerce"))
        == np.sign(pd.to_numeric(out["EV_true"], errors="coerce"))
    ).astype(int)
    return out


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def run_walk_forward(
    frame: pd.DataFrame,
    es_days: int,
    calib_days: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    min_bet_ratio: float,
    holdout_years: int,
    seed: int,
) -> dict[str, Any]:
    folds = _build_outer_folds(frame, min_train_years=3, holdout_years=holdout_years)
    fold_reports: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = fold["fold_id"]
        train_years = fold["train_years"]
        valid_year = fold["valid_year"]

        outer_train = frame[frame["year"].isin(train_years)].copy()
        outer_valid = frame[frame["year"] == valid_year].copy()
        train_fit, es_valid, calib = _split_inner(
            outer_train,
            es_days=es_days,
            calib_days=calib_days,
        )

        train_fit_market = train_fit[train_fit["market_available_flag"] == 1].copy()
        es_valid_market = es_valid[es_valid["market_available_flag"] == 1].copy()
        calib_market = calib[calib["market_available_flag"] == 1].copy()
        outer_valid_market = outer_valid[outer_valid["market_available_flag"] == 1].copy()

        train_fit_o = train_fit_market[train_fit_market["target_log_odds_final"].notna()].copy()
        es_valid_o = es_valid_market[es_valid_market["target_log_odds_final"].notna()].copy()
        calib_o = calib_market[calib_market["target_log_odds_final"].notna()].copy()
        outer_valid_o = outer_valid_market[
            outer_valid_market["target_log_odds_final"].notna()
        ].copy()

        if train_fit.empty or es_valid.empty or calib.empty or outer_valid.empty:
            logger.warning("fold=%s skipped: not enough rows", fold_id)
            continue
        if (
            train_fit_market.empty
            or es_valid_market.empty
            or calib_market.empty
            or outer_valid_market.empty
        ):
            logger.warning("fold=%s skipped: market rows are insufficient", fold_id)
            continue
        if train_fit_o.empty or es_valid_o.empty or calib_o.empty or outer_valid_o.empty:
            logger.warning("fold=%s skipped: O-model rows are insufficient", fold_id)
            continue

        logger.info(
            "fold=%s train=%s valid=%s train_fit=%s es=%s calib=%s outer_valid=%s",
            fold_id,
            f"{min(train_years)}-{max(train_years)}",
            valid_year,
            len(train_fit),
            len(es_valid),
            len(calib),
            len(outer_valid),
        )

        model_f = _train_lgb_binary(
            train_df=train_fit,
            valid_df=es_valid,
            feature_names=list(FUNDAMENTAL_FEATURES),
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            random_state=seed + fold_id,
        )
        model_m = _train_lgb_binary(
            train_df=train_fit_market,
            valid_df=es_valid_market,
            feature_names=list(MODEL_M_FEATURES),
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            random_state=seed + 100 + fold_id,
        )
        model_o = _train_lgb_regression(
            train_df=train_fit_o,
            valid_df=es_valid_o,
            feature_names=list(MODEL_O_FEATURES),
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            random_state=seed + 200 + fold_id,
        )

        calib_meta = _build_meta_frame(
            calib_market,
            model_f=model_f,
            model_m=model_m,
            model_o=model_o,
        )
        meta_model = _train_meta_model(calib_meta)
        calib_meta = _finalize_predictions(calib_meta, meta_model=meta_model)

        tau_info = choose_tau(calib_meta, min_bet_ratio=min_bet_ratio, score_col="EV_hat")
        tau = float(tau_info["tau"])

        outer_full = outer_valid.copy()
        outer_full["pF_raw"] = predict_proba(
            model_f, coerce_model_matrix(outer_full, list(FUNDAMENTAL_FEATURES))
        )
        f_metrics = binary_metrics(outer_full["is_win"], outer_full["pF_raw"])

        outer_meta = _build_meta_frame(
            outer_valid_market,
            model_f=model_f,
            model_m=model_m,
            model_o=model_o,
        )
        outer_meta = _finalize_predictions(outer_meta, meta_model=meta_model)
        outer_meta = apply_tau_rule(outer_meta, tau=tau, score_col="EV_hat")

        m_metrics = binary_metrics(outer_meta["is_win"], outer_meta["pM_raw"])
        final_metrics = binary_metrics(outer_meta["is_win"], outer_meta["p_final"])
        betting = evaluate_betting(outer_meta, threshold=tau, score_col="EV_hat")

        o_true_log = np.log(pd.to_numeric(outer_meta["odds_win_final"], errors="coerce"))
        o_pred_log = np.log(pd.to_numeric(outer_meta["o_close_hat"], errors="coerce"))
        o_diff = o_pred_log - o_true_log
        o_diff = o_diff[np.isfinite(o_diff)]
        o_metrics = {
            "mae_log": float(np.abs(o_diff).mean()) if len(o_diff) > 0 else None,
            "rmse_log": float(np.sqrt((o_diff**2).mean())) if len(o_diff) > 0 else None,
            "decision_consistency": float(outer_meta["decision_consistency"].mean())
            if len(outer_meta) > 0
            else None,
        }

        fold_reports.append(
            {
                "fold_id": fold_id,
                "train_years": train_years,
                "outer_valid_year": int(valid_year),
                "counts": {
                    "outer_train_rows": int(len(outer_train)),
                    "outer_valid_rows": int(len(outer_valid)),
                    "train_fit_rows": int(len(train_fit)),
                    "es_valid_rows": int(len(es_valid)),
                    "calib_rows": int(len(calib)),
                    "outer_valid_market_rows": int(len(outer_meta)),
                },
                "inner_split": {
                    "train_fit_end": str(train_fit["race_date"].max().date())
                    if not train_fit.empty
                    else None,
                    "es_valid_start": str(es_valid["race_date"].min().date())
                    if not es_valid.empty
                    else None,
                    "es_valid_end": str(es_valid["race_date"].max().date())
                    if not es_valid.empty
                    else None,
                    "calib_start": (
                        str(calib["race_date"].min().date()) if not calib.empty else None
                    ),
                },
                "tau": tau,
                "tau_info": tau_info,
                "metrics": {
                    "F": f_metrics,
                    "M": m_metrics,
                    "META": final_metrics,
                    "O": o_metrics,
                },
                "betting": betting.__dict__,
            }
        )

    summary = {
        "n_folds": len(fold_reports),
        "mean_logloss_F": _safe_mean(
            [report["metrics"]["F"]["logloss"] for report in fold_reports]
        ),
        "mean_logloss_META": _safe_mean(
            [report["metrics"]["META"]["logloss"] for report in fold_reports]
        ),
        "mean_auc_META": _safe_mean([report["metrics"]["META"]["auc"] for report in fold_reports]),
        "mean_roi": _safe_mean([report["betting"]["roi"] for report in fold_reports]),
        "mean_bet_ratio": _safe_mean([report["betting"]["bet_ratio"] for report in fold_reports]),
    }

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "code_version": _detect_git_version(),
        "config": {
            "es_days": es_days,
            "calib_days": calib_days,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
            "min_bet_ratio": min_bet_ratio,
            "holdout_years": holdout_years,
            "seed": seed,
        },
        "summary": summary,
        "folds": fold_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward 検証 (F/M/O + stacking)")
    parser.add_argument("--input", default="data/train_t5.parquet", help="入力parquet")
    parser.add_argument("--output-json", default="data/reports/walk_forward_report.json")
    parser.add_argument("--output-csv", default="data/reports/walk_forward_folds.csv")
    parser.add_argument("--es-days", type=int, default=90)
    parser.add_argument("--calib-days", type=int, default=60)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--min-bet-ratio", type=float, default=0.10)
    parser.add_argument("--holdout-years", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    frame = _load_dataset(input_path)
    report = run_walk_forward(
        frame=frame,
        es_days=args.es_days,
        calib_days=args.calib_days,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        min_bet_ratio=args.min_bet_ratio,
        holdout_years=args.holdout_years,
        seed=args.seed,
    )

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(
        [
            {
                "fold_id": fold["fold_id"],
                "outer_valid_year": fold["outer_valid_year"],
                "tau": fold["tau"],
                "meta_logloss": fold["metrics"]["META"]["logloss"],
                "meta_auc": fold["metrics"]["META"]["auc"],
                "meta_brier": fold["metrics"]["META"]["brier"],
                "roi": fold["betting"]["roi"],
                "bet_ratio": fold["betting"]["bet_ratio"],
                "bets": fold["betting"]["bets"],
                "races": fold["betting"]["races"],
                "max_drawdown": fold["betting"]["max_drawdown"],
            }
            for fold in report["folds"]
        ]
    )
    folds.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("saved report: %s", output_json)
    logger.info("saved folds: %s", output_csv)
    logger.info("summary: %s", json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
