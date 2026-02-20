"""
core.race バックフィルスクリプト

raw.jv_raw の RA レコードを再パースして core.race に再投入する。
stub/距離異常/0埋め残りの修復を目的とする。

使用例:
    uv run python scripts/backfill_core_race.py --from-date 2016-01-01 --only-bad
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from app.infrastructure.parsers import RaceRecord
from scripts.load_to_db import upsert_race

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _needs_fix(db: Database, race_id: int) -> bool:
    row = db.fetch_one(
        """
        SELECT surface, distance_m, going, field_size, class_code
        FROM core.race
        WHERE race_id = %s
        """,
        (race_id,),
    )
    if row is None:
        return True

    surface = row.get("surface")
    distance_m = row.get("distance_m")
    going = row.get("going")
    field_size = row.get("field_size")
    class_code = row.get("class_code")

    return (
        surface in (None, 0)
        or distance_m in (None, 0)
        or (distance_m is not None and distance_m < 800)
        or going == 0
        or field_size == 0
        or class_code is None
    )


def _iter_ra_rows(db: Database, from_date: date, to_date: date | None, batch_size: int):
    from_yyyymmdd = from_date.strftime("%Y%m%d")
    to_yyyymmdd = to_date.strftime("%Y%m%d") if to_date else None

    last_id = 0
    while True:
        params: list = [last_id]
        where_clauses = [
            "rec_id = 'RA'",
            "dataspec = 'RACE'",
            "id > %s",
            "substring(payload from 4 for 8) >= %s",
        ]
        params.append(from_yyyymmdd)
        if to_yyyymmdd:
            where_clauses.append("substring(payload from 4 for 8) <= %s")
            params.append(to_yyyymmdd)
        params.append(batch_size)

        rows = db.fetch_all(
            f"""
            SELECT id, payload
            FROM raw.jv_raw
            WHERE {" AND ".join(where_clauses)}
            ORDER BY id
            LIMIT %s
            """,
            tuple(params),
        )
        if not rows:
            return
        yield from rows
        last_id = rows[-1]["id"]


def backfill_core_race(
    db: Database,
    from_date: date,
    to_date: date | None = None,
    batch_size: int = 5000,
    commit_interval: int = 1000,
    only_bad: bool = False,
) -> dict[str, int]:
    stats = {
        "seen": 0,
        "parsed": 0,
        "skipped_out_of_range": 0,
        "skipped_healthy": 0,
        "upserted": 0,
        "errors": 0,
    }

    for row in _iter_ra_rows(db, from_date=from_date, to_date=to_date, batch_size=batch_size):
        stats["seen"] += 1
        if stats["seen"] % 10000 == 0:
            logger.info("scan progress: %s records", stats["seen"])

        try:
            race = RaceRecord.parse(row["payload"])
        except Exception:
            stats["errors"] += 1
            continue

        if race.race_date is None or race.race_id <= 0:
            stats["skipped_out_of_range"] += 1
            continue
        if race.race_date < from_date or (to_date is not None and race.race_date > to_date):
            stats["skipped_out_of_range"] += 1
            continue

        stats["parsed"] += 1

        if only_bad and not _needs_fix(db, race.race_id):
            stats["skipped_healthy"] += 1
            continue

        try:
            upsert_race(db, race)
            stats["upserted"] += 1
        except Exception:
            db.connect().rollback()
            stats["errors"] += 1
            continue

        if stats["upserted"] % commit_interval == 0:
            db.connect().commit()
            logger.info("upsert progress: %s races", stats["upserted"])

    db.connect().commit()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="raw.jv_raw(RA) から core.race を再投入")
    parser.add_argument("--from-date", type=str, default="2016-01-01", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=5000, help="raw 読み込みバッチサイズ")
    parser.add_argument("--commit-interval", type=int, default=1000, help="コミット間隔")
    parser.add_argument(
        "--only-bad",
        action="store_true",
        help="既存 core.race が不良と判定された race_id のみ再投入",
    )
    args = parser.parse_args()

    from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date() if args.to_date else None

    with Database() as db:
        stats = backfill_core_race(
            db,
            from_date=from_date,
            to_date=to_date,
            batch_size=args.batch_size,
            commit_interval=args.commit_interval,
            only_bad=args.only_bad,
        )

    logger.info("backfill finished: %s", stats)


if __name__ == "__main__":
    main()
