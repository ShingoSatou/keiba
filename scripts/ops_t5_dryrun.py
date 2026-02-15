"""
T-5 as-of 運用ドライラン用オーケストレーション。

対象日について、以下を順に実行する:
1) (任意) リアルタイム取得
2) JSONL -> DB ロード
3) 特徴量生成 (対象日)
4) T-5 スナップ生成
5) スナップを CSV / JSON / HTML に出力
6) T-5スナップ入力の推論を実行して結果/監査を出力
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from scripts.rt_common import DATA_DIR, OPS_DEFAULT_DATASPECS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SNAPSHOT_COLUMNS = [
    "race_id",
    "race_date",
    "track_code",
    "race_no",
    "horse_no",
    "horse_id",
    "horse_name",
    "post_time",
    "asof_ts",
    "bw_source",
    "o1_announce_mmddhhmi",
    "odds_win_t5",
    "pop_win_t5",
    "odds_rank_t5",
    "odds_snapshot_age_sec",
    "odds_missing_flag",
    "wh_announce_mmddhhmi",
    "event_change_keys",
    "dm_kbn",
    "dm_create_time",
    "tm_kbn",
    "tm_create_time",
    "odds_stale_flag",
]

DATASPEC_REQUIRED_TABLES: dict[str, list[str]] = {
    "0B41": ["core.o1_header", "core.o1_win"],
    "0B11": ["core.wh_header", "core.wh_detail"],
    "0B14": ["core.event_change"],
    "0B16": ["core.event_change"],  # JVWatchEvent の event key 前提（通常は 0B14 を使用）
    "0B13": ["core.rt_mining_dm"],
    "0B17": ["core.rt_mining_tm"],
}


def _run_command(cmd: list[str], dry_run: bool) -> None:
    logger.info("run: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _parse_dataspecs(dataspecs_raw: str) -> list[str]:
    return [value.strip().upper() for value in dataspecs_raw.split(",") if value.strip()]


def _discover_rt_files(
    input_dir: Path,
    race_date: str,
    dataspecs: list[str],
) -> dict[str, list[Path]]:
    return {
        dataspec: sorted(input_dir.glob(f"{dataspec}_{race_date}*.jsonl")) for dataspec in dataspecs
    }


def _validate_rt_files(
    file_map: dict[str, list[Path]],
    policy: str,
    context: str,
) -> None:
    missing = [dataspec for dataspec, files in file_map.items() if not files]
    if not missing:
        return
    message = f"missing dataspec files ({context}): {','.join(missing)}"
    if policy == "strict":
        raise RuntimeError(message)
    if policy == "warn":
        logger.warning(message)


def _load_rt_jsonl(
    input_dir: Path,
    race_date: str,
    dataspecs: list[str],
    policy: str,
    dry_run: bool,
) -> None:
    file_map = _discover_rt_files(input_dir, race_date, dataspecs)
    for dataspec, files in file_map.items():
        logger.info("input files: dataspec=%s count=%s dir=%s", dataspec, len(files), input_dir)
    if not dry_run:
        _validate_rt_files(file_map, policy=policy, context=f"load input dir={input_dir}")
    for dataspec, files in file_map.items():
        if not files:
            continue
        pattern = str(input_dir / f"{dataspec}_{race_date}*.jsonl")
        _run_command(
            ["uv", "run", "python", "scripts/load_to_db.py", "--input", pattern],
            dry_run=dry_run,
        )


def _ensure_required_files(output_dir: Path, filenames: list[str]) -> None:
    missing = [name for name in filenames if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"required output files missing: {missing}")


def _fetch_count(db: Database, sql: str, params: dict[str, Any]) -> int:
    row = db.fetch_one(sql, params)
    return int(row["n"]) if row and row.get("n") is not None else 0


def _log_day_data_health(race_date_iso: str, feature_set: str, label: str) -> dict[str, int]:
    ymd = int(race_date_iso.replace("-", ""))
    race_id_from = ymd * 10000
    race_id_to = race_id_from + 1099
    params = {
        "race_date": race_date_iso,
        "feature_set": feature_set,
        "race_id_from": race_id_from,
        "race_id_to": race_id_to,
    }
    with Database() as db:
        races = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.race
            WHERE race_date = %(race_date)s
              AND track_code BETWEEN 1 AND 10
              AND race_no BETWEEN 1 AND 12
            """,
            params,
        )
        missing_start_time = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.race
            WHERE race_date = %(race_date)s
              AND track_code BETWEEN 1 AND 10
              AND race_no BETWEEN 1 AND 12
              AND start_time IS NULL
            """,
            params,
        )
        stub_race_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.race
            WHERE race_date = %(race_date)s
              AND track_code BETWEEN 1 AND 10
              AND race_no BETWEEN 1 AND 12
              AND (surface = 0 OR distance_m = 0)
            """,
            params,
        )
        wh_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.wh_header
            WHERE race_id BETWEEN %(race_id_from)s AND %(race_id_to)s
            """,
            params,
        )
        event_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.event_change
            WHERE race_id BETWEEN %(race_id_from)s AND %(race_id_to)s
            """,
            params,
        )
        rt_dm_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.rt_mining_dm
            WHERE race_id BETWEEN %(race_id_from)s AND %(race_id_to)s
            """,
            params,
        )
        rt_tm_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.rt_mining_tm
            WHERE race_id BETWEEN %(race_id_from)s AND %(race_id_to)s
            """,
            params,
        )
        snapshot_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM mart.t5_runner_snapshot
            WHERE race_date = %(race_date)s
              AND feature_set = %(feature_set)s
            """,
            params,
        )
    logger.info(
        (
            "day health[%s]: races=%s start_time_missing=%s race_stub=%s wh=%s event_change=%s "
            "rt_dm=%s rt_tm=%s t5_snapshot=%s"
        ),
        label,
        races,
        missing_start_time,
        stub_race_rows,
        wh_rows,
        event_rows,
        rt_dm_rows,
        rt_tm_rows,
        snapshot_rows,
    )
    return {
        "races": races,
        "start_time_missing": missing_start_time,
        "race_stub": stub_race_rows,
        "wh": wh_rows,
        "event_change": event_rows,
        "rt_dm": rt_dm_rows,
        "rt_tm": rt_tm_rows,
        "t5_snapshot": snapshot_rows,
    }


def _table_exists(db: Database, qualified_name: str) -> bool:
    row = db.fetch_one("SELECT to_regclass(%(name)s) AS regclass", {"name": qualified_name})
    return bool(row and row.get("regclass"))


def _resolve_usable_dataspecs(dataspecs: list[str], policy: str, dry_run: bool) -> list[str]:
    if dry_run:
        return dataspecs
    missing_by_dataspec: dict[str, list[str]] = {}
    with Database() as db:
        for dataspec in dataspecs:
            required_tables = DATASPEC_REQUIRED_TABLES.get(dataspec, [])
            missing_tables = [table for table in required_tables if not _table_exists(db, table)]
            if missing_tables:
                missing_by_dataspec[dataspec] = missing_tables
    if not missing_by_dataspec:
        return dataspecs
    parts = [f"{dataspec}:{','.join(tables)}" for dataspec, tables in missing_by_dataspec.items()]
    message = f"required tables missing for dataspec load: {'; '.join(parts)}"
    if policy == "strict":
        raise RuntimeError(message)
    logger.warning("%s; skip dataspecs=%s", message, ",".join(sorted(missing_by_dataspec.keys())))
    return [dataspec for dataspec in dataspecs if dataspec not in missing_by_dataspec]


def _export_snapshot(
    race_date_iso: str,
    feature_set: str,
    output_dir: Path,
    odds_stale_sec: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    sql = """
    SELECT
        s.race_id,
        s.race_date,
        s.track_code,
        s.race_no,
        s.horse_no,
        s.horse_id,
        h.horse_name,
        s.post_time,
        s.asof_ts,
        s.bw_source,
        s.o1_announce_mmddhhmi,
        s.odds_win_t5,
        s.pop_win_t5,
        s.odds_rank_t5,
        s.odds_snapshot_age_sec,
        s.odds_missing_flag,
        s.wh_announce_mmddhhmi,
        s.event_change_keys,
        s.dm_kbn,
        s.dm_create_time,
        s.tm_kbn,
        s.tm_create_time
    FROM mart.t5_runner_snapshot s
    LEFT JOIN core.horse h ON h.horse_id = s.horse_id
    WHERE s.race_date = %(race_date)s
      AND s.feature_set = %(feature_set)s
    ORDER BY s.race_id, s.horse_no
    """
    with Database() as db:
        rows = db.fetch_all(sql, {"race_date": race_date_iso, "feature_set": feature_set})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    if not rows:
        logger.warning(
            "snapshot rows are empty (date=%s feature_set=%s)",
            race_date_iso,
            feature_set,
        )
    if "odds_snapshot_age_sec" not in df.columns:
        df["odds_snapshot_age_sec"] = pd.Series(dtype="float64")
    if "odds_missing_flag" not in df.columns:
        df["odds_missing_flag"] = pd.Series(dtype="bool")
    df["odds_stale_flag"] = (
        pd.to_numeric(df["odds_snapshot_age_sec"], errors="coerce").fillna(10**9).astype(float)
        > float(odds_stale_sec)
    ).astype(int)
    df = df.reindex(columns=SNAPSHOT_COLUMNS)

    csv_path = output_dir / "t5_snapshot.csv"
    json_path = output_dir / "t5_snapshot.json"
    html_path = output_dir / "t5_snapshot.html"
    meta_path = output_dir / "run_meta.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    html = (
        "<html><head><meta charset='utf-8'><title>T5 Snapshot</title></head><body>"
        f"<h1>T5 Snapshot ({race_date_iso}, feature_set={feature_set})</h1>"
        + df.to_html(index=False, escape=False)
        + "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")

    meta = {
        "race_date": race_date_iso,
        "feature_set": feature_set,
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
        "odds_missing": int(df["odds_missing_flag"].fillna(False).astype(bool).sum()),
        "odds_stale": int(df["odds_stale_flag"].fillna(0).astype(int).sum()),
        "generated_at": datetime.now().isoformat(),
        "files": {
            "csv": str(csv_path),
            "json": str(json_path),
            "html": str(html_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "exported snapshot: rows=%s races=%s odds_missing=%s odds_stale=%s",
        meta["rows"],
        meta["races"],
        meta["odds_missing"],
        meta["odds_stale"],
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="T-5 as-of 運用ドライラン")
    parser.add_argument("--race-date", required=True, help="対象日 (YYYYMMDD)")
    parser.add_argument("--feature-set", default="realtime", help="snapshot feature_set")
    parser.add_argument("--input-dir", default="data", help="load_to_db 対象ディレクトリ")
    parser.add_argument(
        "--rt-data-dir",
        default=str(DATA_DIR),
        help="fetch-rt時のJSONL出力/ロード対象ディレクトリ",
    )
    parser.add_argument(
        "--rt-dataspecs",
        default=",".join(OPS_DEFAULT_DATASPECS),
        help="当日リアルタイム投入dataspec (カンマ区切り)",
    )
    parser.add_argument(
        "--fetch-rt-policy",
        choices=["strict", "warn", "continue"],
        default="strict",
        help="fetch-rt後の不足dataspec扱い",
    )
    parser.add_argument("--output-dir", default="", help="出力先 (未指定時は data/ops/<date>)")
    parser.add_argument("--fetch-rt", action="store_true", help="ops_rt.py を先に実行する")
    parser.add_argument(
        "--odds-stale-sec", type=int, default=900, help="オッズ古さ判定しきい値(秒)"
    )
    parser.add_argument("--slippage", type=float, default=0.15, help="スリッページ率")
    parser.add_argument("--min-prob", type=float, default=0.03, help="最低確率閾値")
    parser.add_argument("--bet-amount", type=int, default=500, help="賭け金")
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="snapshot rows=0 の場合に非0終了する",
    )
    parser.add_argument("--dry-run", action="store_true", help="コマンド表示のみ")
    args = parser.parse_args()

    race_date = args.race_date
    race_date_iso = f"{race_date[:4]}-{race_date[4:6]}-{race_date[6:8]}"
    rt_dataspecs = _parse_dataspecs(args.rt_dataspecs)
    rt_data_dir = Path(args.rt_data_dir)

    output_dir = (
        Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data" / "ops" / race_date)
    )

    if args.fetch_rt:
        fetch_cmd = [
            "uv",
            "run",
            "python",
            "scripts/ops_rt.py",
            "--race-date",
            race_date,
            "--output-dir",
            str(rt_data_dir),
            "--dataspecs",
            ",".join(rt_dataspecs),
        ]
        if args.fetch_rt_policy == "strict":
            fetch_cmd.append("--fail-on-error")
        _run_command(fetch_cmd, dry_run=args.dry_run)
        discovered = _discover_rt_files(rt_data_dir, race_date, rt_dataspecs)
        for dataspec, files in discovered.items():
            logger.info("fetch output files: dataspec=%s count=%s", dataspec, len(files))
        if not args.dry_run:
            _validate_rt_files(
                discovered,
                policy=args.fetch_rt_policy,
                context=f"fetch dir={rt_data_dir}",
            )

    load_input_dir = rt_data_dir if args.fetch_rt else Path(args.input_dir)
    usable_dataspecs = _resolve_usable_dataspecs(
        dataspecs=rt_dataspecs,
        policy=args.fetch_rt_policy,
        dry_run=args.dry_run,
    )
    _load_rt_jsonl(
        input_dir=load_input_dir,
        race_date=race_date,
        dataspecs=usable_dataspecs,
        policy=args.fetch_rt_policy,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        _log_day_data_health(
            race_date_iso=race_date_iso,
            feature_set=args.feature_set,
            label="before_build",
        )
    _run_command(
        ["uv", "run", "python", "scripts/build_features.py", "--date", race_date_iso],
        dry_run=args.dry_run,
    )
    _run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/build_t5_snapshot.py",
            "--date",
            race_date_iso,
            "--feature-set",
            args.feature_set,
        ],
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        _log_day_data_health(
            race_date_iso=race_date_iso,
            feature_set=args.feature_set,
            label="after_snapshot",
        )

    if args.dry_run:
        logger.info("dry-run completed")
        return

    meta = _export_snapshot(
        race_date_iso=race_date_iso,
        feature_set=args.feature_set,
        output_dir=output_dir,
        odds_stale_sec=args.odds_stale_sec,
    )
    _ensure_required_files(
        output_dir,
        ["t5_snapshot.csv", "t5_snapshot.json", "t5_snapshot.html", "run_meta.json"],
    )
    if args.fail_on_empty and int(meta.get("rows", 0)) == 0:
        logger.error(
            "fail-on-empty: snapshot rows=0 (date=%s feature_set=%s)",
            race_date_iso,
            args.feature_set,
        )
        sys.exit(1)

    _run_command(
        [
            "uv",
            "run",
            "python",
            "scripts/predict_t5.py",
            "--race-date",
            race_date_iso,
            "--feature-set",
            args.feature_set,
            "--output-dir",
            str(output_dir),
            "--odds-stale-sec",
            str(args.odds_stale_sec),
            "--slippage",
            str(args.slippage),
            "--min-prob",
            str(args.min_prob),
            "--bet-amount",
            str(args.bet_amount),
        ],
        dry_run=False,
    )

    run_meta_path = output_dir / "run_meta.json"
    if run_meta_path.exists():
        merged_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    else:
        merged_meta = meta
    merged_meta["prediction"] = {
        "slippage": args.slippage,
        "min_prob": args.min_prob,
        "bet_amount": args.bet_amount,
        "files": {
            "csv": str(output_dir / "t5_predictions.csv"),
            "json": str(output_dir / "t5_predictions.json"),
            "html": str(output_dir / "t5_predictions.html"),
            "audit": str(output_dir / "t5_audit.json"),
        },
    }
    run_meta_path.write_text(
        json.dumps(merged_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _ensure_required_files(
        output_dir,
        [
            "t5_snapshot.csv",
            "t5_snapshot.json",
            "t5_snapshot.html",
            "t5_predictions.csv",
            "t5_predictions.json",
            "t5_predictions.html",
            "t5_audit.json",
            "run_meta.json",
        ],
    )
    logger.info("done: %s", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
