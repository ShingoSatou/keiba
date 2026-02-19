from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.t5_modeling import (
    FUNDAMENTAL_FEATURES,
    MODEL_M_FEATURES,
    MODEL_O_FEATURES,
    binary_metrics,
    build_bundle_payload,
    choose_tau,
    dump_json,
    evaluate_betting,
    save_bundle,
)
from scripts.train_walk_forward import (
    _build_meta_frame,
    _finalize_predictions,
    _load_dataset,
    _split_inner,
    _train_lgb_binary,
    _train_lgb_regression,
    _train_meta_model,
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


def _split_production(
    frame: pd.DataFrame,
    es_days: int,
    calib_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_fit, es_valid, calib = _split_inner(frame, es_days=es_days, calib_days=calib_days)
    if train_fit.empty or es_valid.empty or calib.empty:
        max_date = frame["race_date"].max()
        calib_start = max_date - timedelta(days=calib_days - 1)
        es_start = calib_start - timedelta(days=es_days)
        train_fit = frame[frame["race_date"] < es_start].copy()
        es_valid = frame[
            (frame["race_date"] >= es_start) & (frame["race_date"] < calib_start)
        ].copy()
        calib = frame[frame["race_date"] >= calib_start].copy()
    return train_fit, es_valid, calib


def _split_by_race_fraction(
    frame: pd.DataFrame,
    train_ratio: float = 0.6,
    es_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()
    races = (
        frame[["race_id", "race_date"]]
        .drop_duplicates()
        .sort_values(["race_date", "race_id"])
        .reset_index(drop=True)
    )
    n_races = len(races)
    n_train = max(int(n_races * train_ratio), 1)
    n_es = max(int(n_races * es_ratio), 1)
    if n_train + n_es >= n_races:
        n_train = max(n_races - 2, 1)
        n_es = 1
    train_races = set(races.iloc[:n_train]["race_id"])
    es_races = set(races.iloc[n_train : n_train + n_es]["race_id"])
    calib_races = set(races.iloc[n_train + n_es :]["race_id"])
    if not calib_races:
        calib_races = es_races
    train_fit = frame[frame["race_id"].isin(train_races)].copy()
    es_valid = frame[frame["race_id"].isin(es_races)].copy()
    calib = frame[frame["race_id"].isin(calib_races)].copy()
    return train_fit, es_valid, calib


def train_bundle(
    frame: pd.DataFrame,
    es_days: int,
    calib_days: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    min_bet_ratio: float,
    seed: int,
) -> tuple[dict, dict]:
    train_fit, es_valid, calib = _split_production(frame, es_days=es_days, calib_days=calib_days)

    train_fit_market = train_fit[train_fit["market_available_flag"] == 1].copy()
    es_valid_market = es_valid[es_valid["market_available_flag"] == 1].copy()
    calib_market = calib[calib["market_available_flag"] == 1].copy()

    if train_fit_market.empty or es_valid_market.empty or calib_market.empty:
        logger.warning(
            "market split is sparse; fallback to race-fraction split on market rows only"
        )
        market_all = frame[frame["market_available_flag"] == 1].copy()
        train_fit_market, es_valid_market, calib_market = _split_by_race_fraction(market_all)

    train_fit_o = train_fit_market[train_fit_market["target_log_odds_final"].notna()].copy()
    es_valid_o = es_valid_market[es_valid_market["target_log_odds_final"].notna()].copy()
    calib_o = calib_market[calib_market["target_log_odds_final"].notna()].copy()

    if train_fit.empty or es_valid.empty or calib.empty:
        raise ValueError("production split failed: train_fit/es_valid/calib has empty partition")
    if train_fit_market.empty or es_valid_market.empty or calib_market.empty:
        raise ValueError("market rows are not enough to train M/META models")
    if train_fit_o.empty or es_valid_o.empty or calib_o.empty:
        raise ValueError("odds-final rows are not enough to train O model")

    model_f = _train_lgb_binary(
        train_df=train_fit,
        valid_df=es_valid,
        feature_names=list(FUNDAMENTAL_FEATURES),
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
    )
    model_m = _train_lgb_binary(
        train_df=train_fit_market,
        valid_df=es_valid_market,
        feature_names=list(MODEL_M_FEATURES),
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed + 101,
    )
    model_o = _train_lgb_regression(
        train_df=train_fit_o,
        valid_df=es_valid_o,
        feature_names=list(MODEL_O_FEATURES),
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed + 202,
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

    calib_betting = evaluate_betting(calib_meta, threshold=tau, score_col="EV_hat")
    calib_f = _build_meta_frame(calib, model_f, model_m, model_o)
    metrics = {
        "F": binary_metrics(calib["is_win"], calib_f["pF_raw"]),
        "M": binary_metrics(calib_meta["is_win"], calib_meta["pM_raw"]),
        "META": binary_metrics(calib_meta["is_win"], calib_meta["p_final"]),
    }

    metadata = {
        "code_version": _detect_git_version(),
        "dataset_rows": int(len(frame)),
        "dataset_races": int(frame["race_id"].nunique()),
        "split": {
            "train_fit_rows": int(len(train_fit)),
            "es_valid_rows": int(len(es_valid)),
            "calib_rows": int(len(calib)),
            "train_fit_end": str(train_fit["race_date"].max().date()),
            "es_valid_start": str(es_valid["race_date"].min().date()),
            "es_valid_end": str(es_valid["race_date"].max().date()),
            "calib_start": str(calib["race_date"].min().date()),
            "calib_end": str(calib["race_date"].max().date()),
        },
        "metrics_on_calib": metrics,
        "calib_betting": calib_betting.__dict__,
        "tau_info": tau_info,
        "config": {
            "es_days": es_days,
            "calib_days": calib_days,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
            "min_bet_ratio": min_bet_ratio,
            "seed": seed,
        },
    }
    bundle = build_bundle_payload(
        model_f=model_f,
        model_m=model_m,
        model_o=model_o,
        meta_model=meta_model,
        tau_info=tau_info,
        metadata=metadata,
    )
    return bundle, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="本番用T5モデルバンドル学習")
    parser.add_argument("--input", default="data/train_t5.parquet")
    parser.add_argument("--bundle-out", default="models/t5_bundle.pkl")
    parser.add_argument("--meta-out", default="models/t5_bundle_meta.json")
    parser.add_argument("--es-days", type=int, default=90)
    parser.add_argument("--calib-days", type=int, default=60)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--min-bet-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    frame = _load_dataset(input_path)
    bundle, metadata = train_bundle(
        frame=frame,
        es_days=args.es_days,
        calib_days=args.calib_days,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        min_bet_ratio=args.min_bet_ratio,
        seed=args.seed,
    )

    bundle_out = Path(args.bundle_out)
    if not bundle_out.is_absolute():
        bundle_out = PROJECT_ROOT / bundle_out
    save_bundle(bundle, bundle_out)

    meta_out = Path(args.meta_out)
    if not meta_out.is_absolute():
        meta_out = PROJECT_ROOT / meta_out
    dump_json(metadata, meta_out)

    logger.info("saved bundle: %s", bundle_out)
    logger.info("saved metadata: %s", meta_out)
    logger.info("tau=%.6f", float(bundle["tau"]))
    logger.info("summary=%s", json.dumps(metadata["calib_betting"], ensure_ascii=False))


if __name__ == "__main__":
    main()
