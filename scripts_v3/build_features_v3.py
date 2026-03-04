#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from scripts_v2.build_features_v2 import assert_sorted
from scripts_v3.odds_v3_common import (
    assert_t10_no_future_reference,
    load_o1_odds_long,
    merge_odds_features,
)
from scripts_v3.train_binary_v3_common import hash_files, resolve_path, save_json

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3 features from v2 features + odds snapshots(final/t10)."
    )
    parser.add_argument("--input", default="data/features_v2.parquet")
    parser.add_argument("--output", default="data/features_v3.parquet")
    parser.add_argument("--meta-output", default="data/features_v3_meta.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _load_finish_positions(db: Database, race_ids: list[int]) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame(columns=["race_id", "horse_no", "finish_pos"])
    query = """
    SELECT
        run.race_id,
        run.horse_no,
        res.finish_pos
    FROM core.runner run
    JOIN core.result res
      ON res.race_id = run.race_id
     AND res.horse_id = run.horse_id
    WHERE run.race_id = ANY(%(race_ids)s)
    """
    rows = db.fetch_all(query, {"race_ids": race_ids})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["race_id"] = pd.to_numeric(frame["race_id"], errors="coerce").astype("Int64")
    frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64")
    frame["finish_pos"] = pd.to_numeric(frame["finish_pos"], errors="coerce")
    frame = frame.dropna(subset=["race_id", "horse_no"]).copy()
    frame["race_id"] = frame["race_id"].astype(int)
    frame["horse_no"] = frame["horse_no"].astype(int)
    frame = frame.sort_values(["race_id", "horse_no"], kind="mergesort")
    return frame[["race_id", "horse_no", "finish_pos"]]


def build_features_v3(input_df: pd.DataFrame) -> pd.DataFrame:
    frame = input_df.copy()
    required = {"race_id", "horse_no", "target_label", "race_date", "field_size"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in input features: {missing}")

    frame["race_id"] = pd.to_numeric(frame["race_id"], errors="coerce").astype("Int64")
    frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64")
    frame["target_label"] = pd.to_numeric(frame["target_label"], errors="coerce")
    frame = frame.dropna(subset=["race_id", "horse_no", "target_label"]).copy()
    frame["race_id"] = frame["race_id"].astype(int)
    frame["horse_no"] = frame["horse_no"].astype(int)
    frame["target_label"] = frame["target_label"].astype(int)

    frame["y_win"] = (frame["target_label"] == 3).astype(int)
    frame["y_place"] = (frame["target_label"] >= 1).astype(int)
    frame = frame.sort_values(["race_id", "horse_no"], kind="mergesort")

    race_ids = sorted(frame["race_id"].unique().tolist())
    with Database() as db:
        finish_df = _load_finish_positions(db, race_ids)
        odds_long = load_o1_odds_long(db, race_ids)

    if not finish_df.empty:
        frame = frame.merge(finish_df, on=["race_id", "horse_no"], how="left")
    else:
        frame["finish_pos"] = pd.NA

    frame = merge_odds_features(frame, odds_long)
    assert_t10_no_future_reference(frame)

    frame = frame.sort_values(["race_id", "horse_no"], kind="mergesort")
    assert_sorted(frame[["race_id", "horse_no"]].copy())
    return frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    meta_path = resolve_path(args.meta_output)

    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    features_v2 = pd.read_parquet(input_path)
    features_v3 = build_features_v3(features_v2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_v3.to_parquet(output_path, index=False)

    meta = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(features_v3)),
        "races": int(features_v3["race_id"].nunique()) if not features_v3.empty else 0,
        "columns": features_v3.columns.tolist(),
        "coverage": {
            "finish_pos_notna_rate": float(features_v3["finish_pos"].notna().mean()),
            "odds_win_final_notna_rate": float(features_v3["odds_win_final"].notna().mean()),
            "odds_win_t10_notna_rate": float(features_v3["odds_win_t10"].notna().mean()),
            "p_win_odds_final_norm_notna_rate": float(
                features_v3["p_win_odds_final_norm"].notna().mean()
            ),
            "p_win_odds_t10_norm_notna_rate": float(
                features_v3["p_win_odds_t10_norm"].notna().mean()
            ),
        },
        "code_hash": hash_files(
            [
                Path(__file__),
                Path(resolve_path("scripts_v3/odds_v3_common.py")),
            ]
        ),
    }
    save_json(meta_path, meta)

    logger.info("features_v3 rows=%s races=%s", len(features_v3), features_v3["race_id"].nunique())
    logger.info("wrote %s", output_path)
    logger.info("wrote %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
