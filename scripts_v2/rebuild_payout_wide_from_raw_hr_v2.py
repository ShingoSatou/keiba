#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from app.infrastructure.parsers import PayoutRecord  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild core.payout(bet_type=5) from raw HR payloads (v2)."
    )
    parser.add_argument(
        "--truncate-wide",
        action="store_true",
        help="Delete existing bet_type=5 rows before rebuilding.",
    )
    parser.add_argument("--from-year", type=int, default=None, help="Inclusive start year filter.")
    parser.add_argument("--to-year", type=int, default=None, help="Inclusive end year filter.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="UPSERT batch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report stats only, without writing DB.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _race_year_from_id(race_id: int) -> int:
    return int(race_id) // 100000000


def _within_year_range(race_id: int, from_year: int | None, to_year: int | None) -> bool:
    year = _race_year_from_id(int(race_id))
    if from_year is not None and year < int(from_year):
        return False
    if to_year is not None and year > int(to_year):
        return False
    return True


def _flush_rows(db: Database, rows: list[dict]) -> int:
    if not rows:
        return 0
    db.execute_many(
        """
        INSERT INTO core.payout (race_id, bet_type, selection, payout_yen, popularity)
        VALUES (%(race_id)s, %(bet_type)s, %(selection)s, %(payout_yen)s, %(popularity)s)
        ON CONFLICT (race_id, bet_type, selection) DO UPDATE SET
            payout_yen = EXCLUDED.payout_yen,
            popularity = EXCLUDED.popularity
        """,
        rows,
    )
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.from_year and args.to_year and args.from_year > args.to_year:
        raise SystemExit("--from-year must be <= --to-year")

    with Database() as db:
        if args.truncate_wide and not args.dry_run:
            logger.info("deleting existing core.payout rows where bet_type=5")
            db.execute("DELETE FROM core.payout WHERE bet_type = 5")

        raw_rows = db.fetch_all(
            "SELECT id, payload FROM raw.jv_raw WHERE rec_id = 'HR' ORDER BY id"
        )

        parsed_hr_count = 0
        race_count_map: dict[int, int] = {}
        to_upsert: list[dict] = []
        written_rows = 0

        for row in raw_rows:
            payload = row.get("payload")
            if not payload:
                continue
            payouts = PayoutRecord.parse(payload)
            if not payouts:
                continue

            race_id = int(payouts[0].race_id)
            if not _within_year_range(race_id, args.from_year, args.to_year):
                continue

            parsed_hr_count += 1
            wide_rows = [item for item in payouts if int(item.bet_type) == 5]
            if not wide_rows:
                continue

            race_count_map[race_id] = len(wide_rows)
            for payout in wide_rows:
                to_upsert.append(
                    {
                        "race_id": int(payout.race_id),
                        "bet_type": 5,
                        "selection": str(payout.selection),
                        "payout_yen": int(payout.payout_yen),
                        "popularity": payout.popularity,
                    }
                )

            if not args.dry_run and len(to_upsert) >= int(args.batch_size):
                written_rows += _flush_rows(db, to_upsert)
                to_upsert = []

        if not args.dry_run and to_upsert:
            written_rows += _flush_rows(db, to_upsert)

    race_counter = Counter(race_count_map.values())
    total_races = len(race_count_map)
    non_standard_races = sum(v for k, v in race_counter.items() if k != 3)

    print("=" * 80)
    print("rebuild core.payout wide summary")
    print("=" * 80)
    print(f"dry_run                : {args.dry_run}")
    print(f"raw_hr_rows            : {len(raw_rows):,}")
    print(f"parsed_hr_rows         : {parsed_hr_count:,}")
    print(f"wide_races             : {total_races:,}")
    print(f"wide_rows_written      : {written_rows:,}")
    print(f"from_year              : {args.from_year}")
    print(f"to_year                : {args.to_year}")
    print(f"races_with_non3_wide   : {non_standard_races:,}")
    print("wide_rows_per_race_dist:")
    for key in sorted(race_counter):
        print(f"  {key}: {race_counter[key]:,}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
