"""
T-5 as-of 運用ドライラン用オーケストレーション。

対象日について、以下を順に実行する:
1) (任意) 当日RACE前提データを補完
2) (任意) リアルタイム取得
3) JSONL -> DB ロード
4) 特徴量生成 (対象日)
5) T-5 スナップ生成
6) スナップを CSV / JSON / HTML に出力
7) T-5スナップ入力の推論を実行して結果/監査を出力
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
    "0B16": ["core.event_change"],
    "0B13": ["core.rt_mining_dm"],
    "0B17": ["core.rt_mining_tm"],
}

DATE_KEY_DATASPECS = {"0B11", "0B13", "0B14", "0B17"}


def _run_command(cmd: list[str], dry_run: bool) -> None:
    logger.info("run: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _parse_dataspecs(dataspecs_raw: str) -> list[str]:
    return [value.strip().upper() for value in dataspecs_raw.split(",") if value.strip()]


def _resolve_policy_for_dataspec(dataspec: str, policy: str) -> str:
    if policy in {"strict", "warn", "continue"}:
        return policy
    if dataspec == "0B14":
        return "strict"
    if dataspec in {"0B41", "0B11"}:
        return "warn"
    if dataspec in {"0B13", "0B17"}:
        return "continue"
    return "warn"


def _dataspec_pattern(dataspec: str, race_date: str) -> str:
    if dataspec in DATE_KEY_DATASPECS:
        return f"{dataspec}_{race_date}_*.jsonl"
    return f"{dataspec}_{race_date}????_*.jsonl"


def _discover_rt_files(
    input_dir: Path,
    race_date: str,
    dataspecs: list[str],
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    file_map: dict[str, list[Path]] = {}
    pattern_map: dict[str, str] = {}
    for dataspec in dataspecs:
        pattern = _dataspec_pattern(dataspec, race_date)
        pattern_map[dataspec] = pattern
        file_map[dataspec] = sorted(input_dir.glob(pattern))
    return file_map, pattern_map


def _validate_dataspec_files(
    file_map: dict[str, list[Path]],
    policy_by_dataspec: dict[str, str],
    context: str,
) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for dataspec, files in file_map.items():
        policy = policy_by_dataspec[dataspec]
        if files:
            status[dataspec] = {
                "policy": policy,
                "status": "ok",
                "file_count": len(files),
            }
            continue

        message = f"missing dataspec files ({context}): {dataspec}"
        if policy == "strict":
            raise RuntimeError(message)
        if policy == "warn":
            logger.warning("%s (continue)", message)
            status[dataspec] = {
                "policy": policy,
                "status": "warn_missing_file",
                "file_count": 0,
            }
            continue
        logger.info("%s (continue)", message)
        status[dataspec] = {
            "policy": policy,
            "status": "continue_missing_file",
            "file_count": 0,
        }
    return status


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
        runner_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.runner
            WHERE race_id BETWEEN %(race_id_from)s AND %(race_id_to)s
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
        o1_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM core.o1_header
            WHERE race_id BETWEEN %(race_id_from)s AND %(race_id_to)s
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
        snapshot_odds_missing_rows = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM mart.t5_runner_snapshot
            WHERE race_date = %(race_date)s
              AND feature_set = %(feature_set)s
              AND COALESCE(odds_missing_flag, FALSE)
            """,
            params,
        )
        snapshot_odds_missing_races = _fetch_count(
            db,
            """
            SELECT COUNT(*) AS n
            FROM (
                SELECT race_id, asof_ts
                FROM mart.t5_runner_snapshot
                WHERE race_date = %(race_date)s
                  AND feature_set = %(feature_set)s
                GROUP BY race_id, asof_ts
                HAVING BOOL_AND(COALESCE(odds_missing_flag, FALSE))
            ) z
            """,
            params,
        )

    health = {
        "races": races,
        "runner_rows": runner_rows,
        "missing_start_time": missing_start_time,
        "race_stub": stub_race_rows,
        "o1_rows": o1_rows,
        "wh_rows": wh_rows,
        "event_rows": event_rows,
        "rt_dm_rows": rt_dm_rows,
        "rt_tm_rows": rt_tm_rows,
        "t5_snapshot_rows": snapshot_rows,
        "snapshot_odds_missing_rows": snapshot_odds_missing_rows,
        "snapshot_odds_missing_races": snapshot_odds_missing_races,
    }
    logger.info(
        (
            "day health[%s]: races=%s runners=%s missing_start_time=%s race_stub=%s "
            "o1_rows=%s wh_rows=%s event_rows=%s rt_dm_rows=%s rt_tm_rows=%s "
            "t5_snapshot_rows=%s odds_missing_rows=%s odds_missing_races=%s"
        ),
        label,
        health["races"],
        health["runner_rows"],
        health["missing_start_time"],
        health["race_stub"],
        health["o1_rows"],
        health["wh_rows"],
        health["event_rows"],
        health["rt_dm_rows"],
        health["rt_tm_rows"],
        health["t5_snapshot_rows"],
        health["snapshot_odds_missing_rows"],
        health["snapshot_odds_missing_races"],
    )
    return health


def _table_exists(db: Database, qualified_name: str) -> bool:
    row = db.fetch_one("SELECT to_regclass(%(name)s) AS regclass", {"name": qualified_name})
    return bool(row and row.get("regclass"))


def _resolve_usable_dataspecs(
    dataspecs: list[str],
    policy_by_dataspec: dict[str, str],
    dry_run: bool,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if dry_run:
        return dataspecs, {}

    status: dict[str, dict[str, Any]] = {}
    usable: list[str] = []
    with Database() as db:
        for dataspec in dataspecs:
            required_tables = DATASPEC_REQUIRED_TABLES.get(dataspec, [])
            missing_tables = [table for table in required_tables if not _table_exists(db, table)]
            policy = policy_by_dataspec[dataspec]
            if missing_tables:
                message = f"required tables missing for {dataspec}: {','.join(missing_tables)}"
                if policy == "strict":
                    raise RuntimeError(message)
                logger.warning("%s; skip dataspec=%s", message, dataspec)
                status[dataspec] = {
                    "policy": policy,
                    "status": "skip_missing_table",
                    "missing_tables": missing_tables,
                }
                continue
            status[dataspec] = {
                "policy": policy,
                "status": "ok",
                "missing_tables": [],
            }
            usable.append(dataspec)
    return usable, status


def _load_rt_jsonl(
    input_dir: Path,
    race_date: str,
    dataspecs: list[str],
    policy_by_dataspec: dict[str, str],
    dry_run: bool,
) -> tuple[dict[str, list[Path]], dict[str, str], dict[str, dict[str, Any]]]:
    file_map, pattern_map = _discover_rt_files(input_dir, race_date, dataspecs)
    for dataspec, files in file_map.items():
        logger.info(
            "input files: dataspec=%s policy=%s count=%s pattern=%s dir=%s",
            dataspec,
            policy_by_dataspec[dataspec],
            len(files),
            pattern_map[dataspec],
            input_dir,
        )

    status_map = _validate_dataspec_files(
        file_map,
        policy_by_dataspec=policy_by_dataspec,
        context=f"load input dir={input_dir}",
    )
    for dataspec, files in file_map.items():
        if not files:
            continue
        pattern = str(input_dir / pattern_map[dataspec])
        _run_command(
            ["uv", "run", "python", "scripts/load_to_db.py", "--input", pattern],
            dry_run=dry_run,
        )
    return file_map, pattern_map, status_map


def _export_snapshot(
    race_date_iso: str,
    feature_set: str,
    output_dir: Path,
    odds_stale_sec: int,
) -> dict[str, Any]:
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
            "snapshot rows are empty (date=%s feature_set=%s)", race_date_iso, feature_set
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

    meta: dict[str, Any] = {
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


def _evaluate_health(
    health: dict[str, int],
    dataspec_file_status: dict[str, dict[str, Any]],
    include_snapshot_quality: bool = True,
) -> dict[str, Any]:
    stop_reasons: list[str] = []
    warn_reasons: list[str] = []

    races = int(health["races"])
    runners = int(health["runner_rows"])
    missing_start_time = int(health["missing_start_time"])
    race_stub = int(health["race_stub"])
    snapshot_rows = int(health.get("t5_snapshot_rows", 0))
    snapshot_odds_missing_rows = int(health.get("snapshot_odds_missing_rows", 0))
    snapshot_odds_missing_races = int(health.get("snapshot_odds_missing_races", 0))

    if races == 0:
        stop_reasons.append("core.race rows=0")
    if runners == 0:
        stop_reasons.append("core.runner rows=0")

    if races > 0 and missing_start_time * 2 >= races:
        stop_reasons.append(
            f"missing_start_time ratio too high ({missing_start_time}/{races}, threshold=0.50)"
        )
    elif missing_start_time > 0:
        warn_reasons.append(f"missing_start_time={missing_start_time}")

    if races > 0 and race_stub * 2 >= races:
        stop_reasons.append(f"race_stub ratio too high ({race_stub}/{races}, threshold=0.50)")
    elif race_stub > 0:
        warn_reasons.append(f"race_stub={race_stub}")

    if int(health["o1_rows"]) == 0:
        warn_reasons.append("o1_rows=0")
    if int(health["wh_rows"]) == 0:
        warn_reasons.append("wh_rows=0")
    if int(health["event_rows"]) == 0:
        warn_reasons.append("event_rows=0")
    if int(health["rt_dm_rows"]) == 0:
        warn_reasons.append("rt_dm_rows=0")
    if int(health["rt_tm_rows"]) == 0:
        warn_reasons.append("rt_tm_rows=0")

    if include_snapshot_quality and snapshot_rows > 0:
        if snapshot_odds_missing_rows * 2 >= snapshot_rows:
            warn_reasons.append(
                "snapshot odds_missing ratio high "
                f"({snapshot_odds_missing_rows}/{snapshot_rows}, threshold=0.50)"
            )
        elif snapshot_odds_missing_rows > 0:
            warn_reasons.append(f"snapshot odds_missing_rows={snapshot_odds_missing_rows}")

        if races > 0 and snapshot_odds_missing_races * 2 >= races:
            warn_reasons.append(
                "snapshot all-odds-missing races ratio high "
                f"({snapshot_odds_missing_races}/{races}, threshold=0.50)"
            )
        elif snapshot_odds_missing_races > 0:
            warn_reasons.append(f"snapshot odds_missing_races={snapshot_odds_missing_races}")

    for dataspec, ds_status in dataspec_file_status.items():
        status = ds_status.get("status")
        if status == "warn_missing_file":
            warn_reasons.append(f"{dataspec} files missing (warn)")
        if status == "continue_missing_file":
            warn_reasons.append(f"{dataspec} files missing (continue)")
        if status == "skip_missing_table":
            warn_reasons.append(f"{dataspec} skipped due to missing tables")

    status = "ok"
    if stop_reasons:
        status = "stop"
    elif warn_reasons:
        status = "warn"

    return {
        "status": status,
        "stop_reasons": stop_reasons,
        "warn_reasons": warn_reasons,
        "metrics": health,
    }


def _build_empty_snapshot_diagnostics(
    race_date: str,
    race_date_iso: str,
    feature_set: str,
    load_input_dir: Path,
    dataspecs: list[str],
    policy_by_dataspec: dict[str, str],
) -> dict[str, Any]:
    health = _log_day_data_health(
        race_date_iso=race_date_iso,
        feature_set=feature_set,
        label="empty_snapshot_diagnostic",
    )
    file_map, pattern_map = _discover_rt_files(load_input_dir, race_date, dataspecs)
    return {
        "health": health,
        "files": {
            dataspec: {
                "policy": policy_by_dataspec[dataspec],
                "pattern": pattern_map[dataspec],
                "count": len(files),
            }
            for dataspec, files in file_map.items()
        },
        "next_actions": [
            f"uv run python scripts/ops_race.py --race-date {race_date}",
            f"uv run python scripts/ops_rt.py --race-date {race_date}",
            f"uv run python scripts/ops_rt_load.py --race-date {race_date}",
            (
                "uv run python scripts/build_t5_snapshot.py "
                f"--date {race_date_iso} --feature-set {feature_set}"
            ),
        ],
    }


def _ensure_race_prerequisite(race_date: str, race_date_iso: str, dry_run: bool) -> dict[str, Any]:
    before = _log_day_data_health(
        race_date_iso=race_date_iso,
        feature_set="realtime",
        label="ensure_race_before",
    )
    needs_bootstrap = (
        before["races"] == 0
        or before["runner_rows"] == 0
        or (before["races"] > 0 and before["missing_start_time"] == before["races"])
    )
    if not needs_bootstrap:
        return {"executed": False, "before": before, "after": before}

    _run_command(
        ["uv", "run", "python", "scripts/ops_race.py", "--race-date", race_date],
        dry_run=dry_run,
    )
    if dry_run:
        return {"executed": True, "before": before, "after": before}

    after = _log_day_data_health(
        race_date_iso=race_date_iso,
        feature_set="realtime",
        label="ensure_race_after",
    )
    if after["races"] == 0 or after["runner_rows"] == 0:
        raise RuntimeError(
            "race prerequisite failed: core.race/core.runner is still empty after ops_race"
        )
    return {"executed": True, "before": before, "after": after}


def _fail(message: str) -> None:
    logger.error(message)
    sys.exit(1)


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
        choices=["auto", "strict", "warn", "continue"],
        default="auto",
        help="fetch-rt後の不足dataspec扱い",
    )
    parser.add_argument("--output-dir", default="", help="出力先 (未指定時は data/ops/<date>)")
    parser.add_argument("--fetch-rt", action="store_true", help="ops_rt.py を先に実行する")
    parser.add_argument(
        "--ensure-race",
        dest="ensure_race",
        action="store_true",
        help="当日RACEを事前チェックし不足時に ops_race.py で補完する",
    )
    parser.add_argument(
        "--no-ensure-race",
        dest="ensure_race",
        action="store_false",
        help="当日RACEの事前補完を行わない",
    )
    parser.set_defaults(ensure_race=True)
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
    policy_by_dataspec = {
        dataspec: _resolve_policy_for_dataspec(dataspec, args.fetch_rt_policy)
        for dataspec in rt_dataspecs
    }

    rt_data_dir = Path(args.rt_data_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data" / "ops" / race_date)
    )

    race_bootstrap: dict[str, Any] = {"executed": False}
    if args.ensure_race:
        race_bootstrap = _ensure_race_prerequisite(
            race_date=race_date,
            race_date_iso=race_date_iso,
            dry_run=args.dry_run,
        )

    fetch_status: dict[str, dict[str, Any]] = {}
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

        discovered, patterns = _discover_rt_files(rt_data_dir, race_date, rt_dataspecs)
        for dataspec, files in discovered.items():
            logger.info(
                "fetch output: dataspec=%s policy=%s count=%s pattern=%s",
                dataspec,
                policy_by_dataspec[dataspec],
                len(files),
                patterns[dataspec],
            )
        fetch_status = _validate_dataspec_files(
            discovered,
            policy_by_dataspec=policy_by_dataspec,
            context=f"fetch dir={rt_data_dir}",
        )

    load_input_dir = rt_data_dir if args.fetch_rt else Path(args.input_dir)
    usable_dataspecs, table_status = _resolve_usable_dataspecs(
        dataspecs=rt_dataspecs,
        policy_by_dataspec=policy_by_dataspec,
        dry_run=args.dry_run,
    )

    _, _, load_status = _load_rt_jsonl(
        input_dir=load_input_dir,
        race_date=race_date,
        dataspecs=usable_dataspecs,
        policy_by_dataspec=policy_by_dataspec,
        dry_run=args.dry_run,
    )

    dataspec_status: dict[str, dict[str, Any]] = {}
    for dataspec in rt_dataspecs:
        dataspec_status[dataspec] = {
            "policy": policy_by_dataspec[dataspec],
            "fetch": fetch_status.get(dataspec, {"status": "not_checked"}),
            "load": load_status.get(dataspec, {"status": "not_loaded"}),
            "table": table_status.get(dataspec, {"status": "ok", "missing_tables": []}),
        }

    before_build_health: dict[str, int] = {}
    before_build_eval: dict[str, Any] = {
        "status": "ok",
        "stop_reasons": [],
        "warn_reasons": [],
        "metrics": {},
    }
    if not args.dry_run:
        before_build_health = _log_day_data_health(
            race_date_iso=race_date_iso,
            feature_set=args.feature_set,
            label="before_build",
        )
        before_build_eval = _evaluate_health(
            before_build_health,
            {
                ds: {
                    "status": dataspec_status[ds]["load"].get("status"),
                    "policy": dataspec_status[ds]["policy"],
                }
                for ds in dataspec_status
            },
            include_snapshot_quality=False,
        )
        if before_build_eval["status"] == "stop":
            _fail(f"health stop before build: {before_build_eval['stop_reasons']}")

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

    after_snapshot_health: dict[str, int] = {}
    after_snapshot_eval: dict[str, Any] = {
        "status": "ok",
        "stop_reasons": [],
        "warn_reasons": [],
        "metrics": {},
    }
    if not args.dry_run:
        after_snapshot_health = _log_day_data_health(
            race_date_iso=race_date_iso,
            feature_set=args.feature_set,
            label="after_snapshot",
        )
        after_snapshot_eval = _evaluate_health(
            after_snapshot_health,
            {
                ds: {
                    "status": dataspec_status[ds]["load"].get("status"),
                    "policy": dataspec_status[ds]["policy"],
                }
                for ds in dataspec_status
            },
            include_snapshot_quality=True,
        )
        if after_snapshot_eval["status"] == "stop":
            _fail(f"health stop after snapshot: {after_snapshot_eval['stop_reasons']}")

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

    run_meta_path = output_dir / "run_meta.json"
    merged_meta = (
        json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else meta
    )
    merged_meta["rt"] = {
        "policy": args.fetch_rt_policy,
        "policy_by_dataspec": policy_by_dataspec,
        "load_input_dir": str(load_input_dir),
        "dataspecs": dataspec_status,
    }
    merged_meta["health"] = {
        "before_build": before_build_eval,
        "after_snapshot": after_snapshot_eval,
        "race_bootstrap": race_bootstrap,
    }

    if args.fail_on_empty and int(meta.get("rows", 0)) == 0:
        diagnostic = _build_empty_snapshot_diagnostics(
            race_date=race_date,
            race_date_iso=race_date_iso,
            feature_set=args.feature_set,
            load_input_dir=load_input_dir,
            dataspecs=rt_dataspecs,
            policy_by_dataspec=policy_by_dataspec,
        )
        merged_meta["empty_snapshot_diagnostic"] = diagnostic
        merged_meta["health"]["after_snapshot"]["status"] = "stop"
        merged_meta["health"]["after_snapshot"].setdefault("stop_reasons", []).append(
            "t5_snapshot_rows=0 with fail-on-empty"
        )
        run_meta_path.write_text(
            json.dumps(merged_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _fail(
            "fail-on-empty: snapshot rows=0 diagnostics="
            f"{json.dumps(diagnostic, ensure_ascii=False)}"
        )

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
        json.dumps(merged_meta, ensure_ascii=False, indent=2), encoding="utf-8"
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
    logger.info("done: %s", json.dumps(merged_meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
