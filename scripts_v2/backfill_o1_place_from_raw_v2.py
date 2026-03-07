#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from app.infrastructure.parsers import OddsTimeSeriesRecord  # noqa: E402
from scripts_v2.load_to_db import is_central_race, upsert_o1_timeseries_bulk  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill core.o1_place from raw.jv_raw O1 payloads (v2)."
    )
    parser.add_argument("--from-date", required=True, help="開始日 (YYYYMMDD)")
    parser.add_argument("--to-date", required=True, help="終了日 (YYYYMMDD)")
    parser.add_argument(
        "--dataspecs",
        default="0B41,RACE",
        help="対象 dataspec のカンマ区切り (既定: 0B41,RACE)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="raw.jv_raw の取得件数 / 1 クエリ",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=50000,
        help="core.o1_place へ upsert する O1 行数 / 1 回",
    )
    parser.add_argument(
        "--include-non-central",
        action="store_true",
        help="中央競馬（場コード01-10）以外も対象に含める",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="解析と件数確認のみ行い、DB 書き込みはしない",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _validate_yyyymmdd(value: str, *, label: str) -> str:
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise SystemExit(f"{label} must be YYYYMMDD: {value}")
    return text


def _race_date_ymd_from_race_id(race_id: int) -> str:
    return f"{int(race_id) // 10000:08d}"


def _within_date_range(race_id: int, from_date: str, to_date: str) -> bool:
    race_date = _race_date_ymd_from_race_id(race_id)
    return str(from_date) <= race_date <= str(to_date)


def _count_unique_rows(records: list[OddsTimeSeriesRecord]) -> tuple[int, int]:
    win_keys = {
        (row.race_id, row.data_kbn, row.announce_mmddhhmi, row.horse_no)
        for row in records
        if row.has_win_block
    }
    place_keys = {
        (row.race_id, row.data_kbn, row.announce_mmddhhmi, row.horse_no)
        for row in records
        if row.has_place_block
    }
    return (len(win_keys), len(place_keys))


def _fetch_raw_batch(
    db: Database,
    *,
    last_id: int,
    batch_size: int,
    dataspecs: list[str],
) -> list[dict]:
    return db.fetch_all(
        """
        SELECT id, dataspec, payload
        FROM raw.jv_raw
        WHERE rec_id = 'O1'
          AND dataspec = ANY(%(dataspecs)s)
          AND id > %(last_id)s
        ORDER BY id
        LIMIT %(limit)s
        """,
        {"dataspecs": dataspecs, "last_id": int(last_id), "limit": int(batch_size)},
    )


def _flush_place_rows(
    db: Database,
    rows: list[OddsTimeSeriesRecord],
    *,
    dry_run: bool,
) -> tuple[int, int]:
    if not rows:
        return (0, 0)
    if dry_run:
        return _count_unique_rows(rows)
    result = upsert_o1_timeseries_bulk(
        db,
        rows,
        race_stub_cache=set(),
        include_win_details=False,
        include_place_details=True,
    )
    db.connect().commit()
    return result


def backfill_o1_place_from_raw(
    db: Database,
    *,
    from_date: str,
    to_date: str,
    dataspecs: list[str],
    batch_size: int,
    upsert_batch_size: int,
    include_non_central: bool,
    dry_run: bool,
) -> dict[str, int]:
    stats = {
        "raw_rows": 0,
        "matched_snapshots": 0,
        "o1_rows_parsed": 0,
        "win_rows_seen": 0,
        "place_rows_written": 0,
        "skipped_non_central": 0,
        "skipped_out_of_range": 0,
    }

    last_id = 0
    pending: list[OddsTimeSeriesRecord] = []

    while True:
        raw_rows = _fetch_raw_batch(
            db,
            last_id=last_id,
            batch_size=batch_size,
            dataspecs=dataspecs,
        )
        if not raw_rows:
            break

        last_id = int(raw_rows[-1]["id"])
        stats["raw_rows"] += len(raw_rows)

        for row in raw_rows:
            payload = str(row.get("payload") or "")
            records = OddsTimeSeriesRecord.parse(payload)
            if not records:
                continue

            race_id = int(records[0].race_id)
            if not include_non_central and not is_central_race(race_id):
                stats["skipped_non_central"] += 1
                continue
            if not _within_date_range(race_id, from_date, to_date):
                stats["skipped_out_of_range"] += 1
                continue

            stats["matched_snapshots"] += 1
            stats["o1_rows_parsed"] += len(records)
            stats["win_rows_seen"] += sum(1 for record in records if record.has_win_block)
            pending.extend(records)

            if len(pending) >= upsert_batch_size:
                _, place_rows = _flush_place_rows(db, pending, dry_run=dry_run)
                stats["place_rows_written"] += place_rows
                pending = []

        logger.info(
            "progress raw_rows=%s matched_snapshots=%s place_rows=%s last_id=%s",
            f"{stats['raw_rows']:,}",
            f"{stats['matched_snapshots']:,}",
            f"{stats['place_rows_written']:,}",
            last_id,
        )

    if pending:
        _, place_rows = _flush_place_rows(db, pending, dry_run=dry_run)
        stats["place_rows_written"] += place_rows

    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from_date = _validate_yyyymmdd(args.from_date, label="from-date")
    to_date = _validate_yyyymmdd(args.to_date, label="to-date")
    if from_date > to_date:
        raise SystemExit("--from-date must be <= --to-date")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.upsert_batch_size <= 0:
        raise SystemExit("--upsert-batch-size must be > 0")

    dataspecs = [part.strip() for part in str(args.dataspecs).split(",") if part.strip()]
    if not dataspecs:
        raise SystemExit("--dataspecs must not be empty")

    with Database() as db:
        stats = backfill_o1_place_from_raw(
            db,
            from_date=from_date,
            to_date=to_date,
            dataspecs=dataspecs,
            batch_size=int(args.batch_size),
            upsert_batch_size=int(args.upsert_batch_size),
            include_non_central=bool(args.include_non_central),
            dry_run=bool(args.dry_run),
        )

    print("=" * 80)
    print("backfill core.o1_place summary")
    print("=" * 80)
    print(f"dry_run             : {args.dry_run}")
    print(f"from_date           : {from_date}")
    print(f"to_date             : {to_date}")
    print(f"dataspecs           : {','.join(dataspecs)}")
    print(f"raw_rows            : {stats['raw_rows']:,}")
    print(f"matched_snapshots   : {stats['matched_snapshots']:,}")
    print(f"o1_rows_parsed      : {stats['o1_rows_parsed']:,}")
    print(f"win_rows_seen       : {stats['win_rows_seen']:,}")
    print(f"place_rows_written  : {stats['place_rows_written']:,}")
    print(f"skipped_non_central : {stats['skipped_non_central']:,}")
    print(f"skipped_out_of_range: {stats['skipped_out_of_range']:,}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
