"""
当日運用 RT JSONL -> DB 投入スクリプト。

dataspec 別の投入ポリシー（stop / warn / continue）を適用し、
投入件数サマリをログと JSON で出力する。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from scripts.load_to_db import process_file
from scripts.rt_common import DATA_DIR, OPS_DEFAULT_DATASPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATE_KEY_DATASPECS = {"0B11", "0B13", "0B14", "0B17"}

DATASPEC_REQUIRED_TABLES: dict[str, list[str]] = {
    "0B41": ["core.o1_header", "core.o1_win"],
    "0B11": ["core.wh_header", "core.wh_detail"],
    "0B14": ["core.event_change"],
    "0B13": ["core.rt_mining_dm"],
    "0B17": ["core.rt_mining_tm"],
}


def _parse_dataspecs(raw: str) -> list[str]:
    return [value.strip().upper() for value in raw.split(",") if value.strip()]


def _table_exists(db: Database, qualified_name: str) -> bool:
    row = db.fetch_one("SELECT to_regclass(%(name)s) AS regclass", {"name": qualified_name})
    return bool(row and row.get("regclass"))


def _policy_for_dataspec(dataspec: str, policy: str) -> str:
    if policy in {"strict", "warn", "continue"}:
        return policy
    if dataspec == "0B14":
        return "strict"
    if dataspec in {"0B41", "0B11"}:
        return "warn"
    if dataspec in {"0B13", "0B17"}:
        return "continue"
    return "warn"


def _pattern_for_dataspec(dataspec: str, race_date: str) -> str:
    if dataspec in DATE_KEY_DATASPECS:
        return f"{dataspec}_{race_date}_*.jsonl"
    return f"{dataspec}_{race_date}????_*.jsonl"


def _discover_files(input_dir: Path, race_date: str, dataspec: str) -> list[Path]:
    return sorted(input_dir.glob(_pattern_for_dataspec(dataspec, race_date)))


def _sum_stats(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def run_ops_rt_load(
    race_date: str,
    input_dir: Path,
    dataspecs: list[str],
    policy: str,
    dry_run: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "race_date": race_date,
        "input_dir": str(input_dir),
        "policy": policy,
        "generated_at": datetime.now().isoformat(),
        "dataspecs": {},
    }

    with Database() as db:
        for dataspec in dataspecs:
            ds_policy = _policy_for_dataspec(dataspec, policy)
            files = _discover_files(input_dir, race_date, dataspec)
            required_tables = DATASPEC_REQUIRED_TABLES.get(dataspec, [])
            missing_tables = [table for table in required_tables if not _table_exists(db, table)]

            item: dict[str, Any] = {
                "policy": ds_policy,
                "pattern": _pattern_for_dataspec(dataspec, race_date),
                "files": [str(path) for path in files],
                "file_count": len(files),
                "missing_tables": missing_tables,
                "loaded_file_count": 0,
                "stats": {},
                "status": "ok",
            }
            summary["dataspecs"][dataspec] = item

            logger.info(
                "dataspec=%s policy=%s files=%s pattern=%s",
                dataspec,
                ds_policy,
                len(files),
                item["pattern"],
            )

            if missing_tables:
                message = f"required tables missing for {dataspec}: {','.join(missing_tables)}"
                if ds_policy == "strict":
                    raise RuntimeError(message)
                item["status"] = "skipped_missing_table"
                logger.warning(message)
                continue

            if not files:
                message = f"missing input files for {dataspec}"
                if ds_policy == "strict":
                    raise RuntimeError(message)
                if ds_policy == "warn":
                    item["status"] = "warn_missing_file"
                    logger.warning("%s (continue)", message)
                else:
                    item["status"] = "continue_missing_file"
                    logger.info("%s (continue)", message)
                continue

            if dry_run:
                item["status"] = "dry_run"
                continue

            ds_stats: dict[str, int] = {}
            for file_path in files:
                logger.info("load file: dataspec=%s file=%s", dataspec, file_path.name)
                file_stats = process_file(db, file_path)
                ds_stats = _sum_stats(ds_stats, file_stats)
                item["loaded_file_count"] += 1
                logger.info(
                    "loaded: dataspec=%s file=%s stats=%s", dataspec, file_path.name, file_stats
                )

            item["stats"] = ds_stats
            logger.info(
                "dataspec summary: %s files=%s loaded=%s stats=%s",
                dataspec,
                item["file_count"],
                item["loaded_file_count"],
                ds_stats,
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="当日運用 RT JSONL -> DB投入")
    parser.add_argument(
        "--race-date",
        default=None,
        help="対象日 (YYYYMMDD, デフォルト: 当日)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help=f"入力ディレクトリ (デフォルト: {DATA_DIR})",
    )
    parser.add_argument(
        "--dataspecs",
        default=",".join(OPS_DEFAULT_DATASPECS),
        help="投入対象 dataspec (カンマ区切り)",
    )
    parser.add_argument(
        "--policy",
        choices=["auto", "strict", "warn", "continue"],
        default="auto",
        help="不足ファイル・不足テーブル時の扱い",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="投入サマリJSONの出力先",
    )
    parser.add_argument("--dry-run", action="store_true", help="DRY-RUN")
    args = parser.parse_args()

    race_date = args.race_date or datetime.now().strftime("%Y%m%d")
    dataspecs = _parse_dataspecs(args.dataspecs)

    summary = run_ops_rt_load(
        race_date=race_date,
        input_dir=args.input_dir,
        dataspecs=dataspecs,
        policy=args.policy,
        dry_run=args.dry_run,
    )

    output_json = args.output_json
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("summary json: %s", output_json)

    logger.info("ops_rt_load done: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
