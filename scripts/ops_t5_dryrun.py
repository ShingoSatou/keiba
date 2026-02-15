"""
T-5 as-of 運用ドライラン用オーケストレーション。

対象日について、以下を順に実行する:
1) (任意) リアルタイム取得
2) JSONL -> DB ロード
3) 特徴量生成 (対象日)
4) T-5 スナップ生成
5) スナップを CSV / JSON / HTML に出力
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_command(cmd: list[str], dry_run: bool) -> None:
    logger.info("run: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


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

    if not rows:
        logger.warning(
            "snapshot rows are empty (date=%s feature_set=%s)", race_date_iso, feature_set
        )
        return {"rows": 0}

    df = pd.DataFrame(rows)
    df["odds_stale_flag"] = (
        df["odds_snapshot_age_sec"].fillna(10**9).astype(float) > float(odds_stale_sec)
    ).astype(int)

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
        "races": int(df["race_id"].nunique()),
        "odds_missing": int(df["odds_missing_flag"].sum()),
        "odds_stale": int(df["odds_stale_flag"].sum()),
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
    parser.add_argument("--output-dir", default="", help="出力先 (未指定時は data/ops/<date>)")
    parser.add_argument("--fetch-rt", action="store_true", help="ops_rt.py を先に実行する")
    parser.add_argument(
        "--odds-stale-sec", type=int, default=900, help="オッズ古さ判定しきい値(秒)"
    )
    parser.add_argument("--dry-run", action="store_true", help="コマンド表示のみ")
    args = parser.parse_args()

    race_date = args.race_date
    race_date_iso = f"{race_date[:4]}-{race_date[4:6]}-{race_date[6:8]}"

    output_dir = (
        Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data" / "ops" / race_date)
    )

    if args.fetch_rt:
        _run_command(
            ["uv", "run", "python", "scripts/ops_rt.py", "--race-date", race_date],
            dry_run=args.dry_run,
        )

    _run_command(
        ["uv", "run", "python", "scripts/load_to_db.py", "--input-dir", args.input_dir],
        dry_run=args.dry_run,
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

    if args.dry_run:
        logger.info("dry-run completed")
        return

    meta = _export_snapshot(
        race_date_iso=race_date_iso,
        feature_set=args.feature_set,
        output_dir=output_dir,
        odds_stale_sec=args.odds_stale_sec,
    )
    logger.info("done: %s", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
