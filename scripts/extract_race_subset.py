"""
RACE JSONL から指定レースだけを抽出するユーティリティ。

0B41 のサンプル検証用に、巨大な RACE ファイルから
特定開催日・場コード・レース番号の RA/SE/HR/O1 を切り出す。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.parsers import OddsRecord, PayoutRecord, RaceRecord, RunnerRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_REC_IDS = {"RA", "SE", "HR", "O1"}


def parse_race_nos(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1 or value > 12:
            raise ValueError(f"race_no out of range: {value}")
        values.append(value)
    if not values:
        raise ValueError("race_nos is empty")
    return sorted(set(values))


def infer_race_id(rec_id: str, payload: str) -> int:
    if rec_id == "RA":
        return RaceRecord.parse(payload).race_id
    if rec_id == "SE":
        return RunnerRecord.parse(payload).race_id
    if rec_id == "HR":
        payouts = PayoutRecord.parse(payload)
        return payouts[0].race_id if payouts else 0
    if rec_id == "O1":
        odds = OddsRecord.parse(payload)
        return odds[0].race_id if odds else 0
    return 0


def main():
    parser = argparse.ArgumentParser(description="RACE JSONL サブセット抽出")
    parser.add_argument("--input", type=Path, required=True, help="入力 JSONL (RACE_*.jsonl)")
    parser.add_argument("--race-date", type=str, required=True, help="対象開催日 (YYYY-MM-DD)")
    parser.add_argument("--track-code", type=int, required=True, help="対象場コード (01-10)")
    parser.add_argument(
        "--race-nos",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="対象レース番号CSV (例: 1,2,3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力 JSONL。未指定時は data/RACE_subset_YYYYMMDD_TT.jsonl",
    )
    args = parser.parse_args()

    race_date = date.fromisoformat(args.race_date)
    race_nos = parse_race_nos(args.race_nos)
    track_code = int(args.track_code)

    if track_code < 1 or track_code > 99:
        raise ValueError("--track-code must be between 1 and 99")

    date_int = race_date.year * 10000 + race_date.month * 100 + race_date.day
    target_race_ids = {date_int * 10000 + track_code * 100 + race_no for race_no in race_nos}

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "data" / f"RACE_subset_{race_date:%Y%m%d}_{track_code:02d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("input=%s", args.input)
    logger.info("output=%s", output)
    logger.info("target_race_ids=%s", sorted(target_race_ids))

    stats = {
        "lines_total": 0,
        "lines_written": 0,
        "parse_errors": 0,
    }
    rec_counts: dict[str, int] = {}

    with args.input.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            stats["lines_total"] += 1
            record = json.loads(line)
            rec_id = record.get("rec_id", "")
            if rec_id not in TARGET_REC_IDS:
                continue

            payload = record.get("payload", "")
            try:
                race_id = infer_race_id(rec_id, payload)
            except Exception:
                stats["parse_errors"] += 1
                continue

            if race_id in target_race_ids:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["lines_written"] += 1
                rec_counts[rec_id] = rec_counts.get(rec_id, 0) + 1

    logger.info(
        "total=%s written=%s parse_errors=%s",
        stats["lines_total"],
        stats["lines_written"],
        stats["parse_errors"],
    )
    for rec_id in sorted(rec_counts):
        logger.info("written[%s]=%s", rec_id, rec_counts[rec_id])


if __name__ == "__main__":
    main()
