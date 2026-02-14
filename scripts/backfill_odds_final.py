"""
core.odds_final backfill script.

raw.jv_raw に保存済みの O1 レコードから core.odds_final を再構築する。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from app.infrastructure.parsers import OddsRecord  # noqa: E402
from scripts.load_to_db import upsert_odds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def backfill_odds_final(
    db: Database,
    dataspec: str,
    start_id: int,
    batch_size: int,
    limit: int | None,
    commit_every: int,
    log_missing_samples: int,
) -> dict[str, int]:
    stats = {
        "raw_processed": 0,
        "odds_parsed": 0,
        "odds_upserted": 0,
        "odds_missing_runner": 0,
        "odds_skipped_bracket": 0,
        "errors": 0,
        "last_id": start_id,
    }

    missing_log_count = 0
    should_stop = False

    while True:
        rows = db.fetch_all(
            """
            SELECT id, payload
            FROM raw.jv_raw
            WHERE dataspec = %(dataspec)s
              AND rec_id = 'O1'
              AND id > %(last_id)s
            ORDER BY id
            LIMIT %(batch_size)s
            """,
            {"dataspec": dataspec, "last_id": stats["last_id"], "batch_size": batch_size},
        )

        if not rows:
            break

        for row in rows:
            if limit is not None and stats["raw_processed"] >= limit:
                should_stop = True
                break

            raw_id = row["id"]
            payload = row["payload"]
            stats["last_id"] = raw_id

            try:
                odds_list = OddsRecord.parse(payload)
                stats["odds_parsed"] += len(odds_list)

                for odds in odds_list:
                    if odds.bet_type == 3:
                        stats["odds_skipped_bracket"] += 1
                        continue
                    if odds.bet_type not in (1, 2):
                        continue

                    affected = upsert_odds(db, odds)
                    stats["odds_upserted"] += affected
                    if affected == 0:
                        stats["odds_missing_runner"] += 1
                        if missing_log_count < log_missing_samples:
                            logger.warning(
                                "O1 odds skipped (runner not found): raw_id=%s race_id=%s "
                                "horse_no=%s bet_type=%s",
                                raw_id,
                                odds.race_id,
                                odds.horse_no,
                                odds.bet_type,
                            )
                            missing_log_count += 1
            except Exception as exc:
                db.connect().rollback()
                if stats["errors"] < 20:
                    logger.warning("Backfill error at raw_id=%s: %s", raw_id, exc)
                stats["errors"] += 1
                continue

            stats["raw_processed"] += 1
            if stats["raw_processed"] % commit_every == 0:
                db.connect().commit()
                logger.info(
                    "processed=%s last_id=%s upserted=%s missing_runner=%s",
                    stats["raw_processed"],
                    stats["last_id"],
                    stats["odds_upserted"],
                    stats["odds_missing_runner"],
                )

        if should_stop:
            break

    db.connect().commit()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill core.odds_final from raw.jv_raw O1")
    parser.add_argument(
        "--dataspec",
        type=str,
        default="RACE",
        help="target dataspec (default: RACE)",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="start from raw.jv_raw.id > start-id",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="number of raw rows fetched per batch",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max number of raw O1 rows to process",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="commit every N processed raw rows",
    )
    parser.add_argument(
        "--log-missing-samples",
        type=int,
        default=20,
        help="max warning samples for missing runner",
    )
    args = parser.parse_args()

    with Database() as db:
        stats = backfill_odds_final(
            db=db,
            dataspec=args.dataspec,
            start_id=args.start_id,
            batch_size=args.batch_size,
            limit=args.limit,
            commit_every=args.commit_every,
            log_missing_samples=args.log_missing_samples,
        )

    print("=" * 60)
    print("Backfill core.odds_final 完了")
    print("=" * 60)
    print(f"dataspec: {args.dataspec}")
    print(f"raw_processed: {stats['raw_processed']:,}")
    print(f"odds_parsed: {stats['odds_parsed']:,}")
    print(f"odds_upserted: {stats['odds_upserted']:,}")
    print(f"odds_missing_runner: {stats['odds_missing_runner']:,}")
    print(f"odds_skipped_bracket: {stats['odds_skipped_bracket']:,}")
    print(f"errors: {stats['errors']:,}")
    print(f"last_id: {stats['last_id']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
