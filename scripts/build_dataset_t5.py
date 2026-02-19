from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from scripts.build_dataset import build_dataset as build_fundamental_dataset
from scripts.market_features import add_market_features
from scripts.t5_modeling import MARKET_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_date(value: str | None):
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _fetch_latest_snapshot_rows(
    db: Database,
    feature_set: str,
    from_date=None,
    to_date=None,
) -> pd.DataFrame:
    where_clauses = ["s.feature_set = %(feature_set)s"]
    params: dict[str, object] = {"feature_set": feature_set}
    if from_date is not None:
        where_clauses.append("s.race_date >= %(from_date)s")
        params["from_date"] = from_date
    if to_date is not None:
        where_clauses.append("s.race_date <= %(to_date)s")
        params["to_date"] = to_date
    where_sql = " AND ".join(where_clauses)

    query = f"""
    WITH ranked AS (
        SELECT
            s.race_id,
            s.horse_id,
            s.horse_no,
            s.race_date,
            s.asof_ts,
            s.feature_set,
            s.o1_data_kbn,
            s.o1_announce_mmddhhmi,
            s.odds_win_t5,
            s.odds_rank_t5,
            s.win_pool_total_100yen_t5,
            s.odds_snapshot_age_sec,
            s.odds_missing_flag,
            s.odds_win_final,
            s.pop_win_final,
            ROW_NUMBER() OVER (
                PARTITION BY s.race_id, s.horse_no
                ORDER BY s.asof_ts DESC
            ) AS rn
        FROM mart.t5_runner_snapshot s
        WHERE {where_sql}
    )
    SELECT
        race_id,
        horse_id,
        horse_no,
        race_date,
        asof_ts,
        feature_set,
        o1_data_kbn,
        o1_announce_mmddhhmi,
        odds_win_t5,
        odds_rank_t5,
        win_pool_total_100yen_t5,
        odds_snapshot_age_sec,
        odds_missing_flag,
        odds_win_final,
        pop_win_final
    FROM ranked
    WHERE rn = 1
    """
    rows = db.fetch_all(query, params)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["asof_ts"] = pd.to_datetime(frame["asof_ts"], errors="coerce")
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.date
    return frame


def build_dataset_t5(
    db: Database,
    feature_set: str,
    from_date=None,
    to_date=None,
) -> pd.DataFrame:
    fundamental = build_fundamental_dataset(db, from_date=from_date, to_date=to_date)
    if fundamental.empty:
        return fundamental

    snapshot = _fetch_latest_snapshot_rows(
        db,
        feature_set=feature_set,
        from_date=from_date,
        to_date=to_date,
    )
    if snapshot.empty:
        logger.warning("mart.t5_runner_snapshot has no rows for feature_set=%s", feature_set)
        merged = fundamental.copy()
        merged["asof_ts"] = pd.NaT
        merged["odds_win_t5"] = np.nan
        merged["odds_rank_t5"] = np.nan
        merged["win_pool_total_100yen_t5"] = np.nan
        merged["odds_snapshot_age_sec"] = np.nan
        merged["odds_missing_flag"] = True
        merged["odds_win_final"] = np.nan
        merged["target_log_odds_final"] = np.nan
        merged["market_available_flag"] = 0
        for column in MARKET_FEATURES:
            if column not in merged.columns:
                merged[column] = np.nan
        return merged

    merged = fundamental.merge(
        snapshot,
        on=["race_id", "horse_id"],
        how="left",
        suffixes=("", "_snap"),
    )
    merged["horse_no"] = merged["horse_no"].fillna(merged.get("horse_no_snap"))
    merged = merged.drop(columns=["horse_no_snap"], errors="ignore")

    merged = add_market_features(db, merged)

    merged["odds_win_final"] = pd.to_numeric(merged["odds_win_final"], errors="coerce")
    merged["target_log_odds_final"] = np.where(
        merged["odds_win_final"] > 0,
        np.log(merged["odds_win_final"]),
        np.nan,
    )
    merged["market_available_flag"] = (
        (pd.to_numeric(merged["M_odds_win_t5"], errors="coerce") > 0)
        & (pd.to_numeric(merged["odds_win_final"], errors="coerce") > 0)
    ).astype(int)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="T-5準拠データセット生成")
    parser.add_argument("--feature-set", default="backfillable", help="snapshot feature_set")
    parser.add_argument("--from-date", type=str, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/train_t5.parquet", help="出力パス")
    args = parser.parse_args()

    from_date = _parse_date(args.from_date)
    to_date = _parse_date(args.to_date)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Database() as db:
        dataset = build_dataset_t5(
            db,
            feature_set=args.feature_set,
            from_date=from_date,
            to_date=to_date,
        )

    if dataset.empty:
        logger.error("dataset is empty")
        return

    dataset.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(
        "saved: %s rows=%s races=%s",
        output_path,
        len(dataset),
        dataset["race_id"].nunique(),
    )
    if "market_available_flag" in dataset.columns:
        logger.info("market available ratio=%.4f", float(dataset["market_available_flag"].mean()))


if __name__ == "__main__":
    main()
