"""
T-5(as-of)スナップを mart に生成するバッチ。

初回スコープ:
- O1(0B41)の data_kbn=1 から T-5 採用
- 最終オッズ(core.odds_final)を付与
- 体重は WH があれば優先、なければ SE(body_weight) にフォールバック
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def detect_git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_snapshot(
    db: Database,
    from_date: date,
    to_date: date,
    feature_set: str,
    code_version: str,
) -> None:
    missing_start_time = db.fetch_one(
        """
        SELECT COUNT(*) AS n
        FROM core.race
        WHERE race_date BETWEEN %(from_date)s AND %(to_date)s
          AND track_code BETWEEN 1 AND 10
          AND start_time IS NULL
        """,
        {"from_date": from_date, "to_date": to_date},
    )
    total_target_races = db.fetch_one(
        """
        SELECT COUNT(*) AS n
        FROM core.race
        WHERE race_date BETWEEN %(from_date)s AND %(to_date)s
          AND track_code BETWEEN 1 AND 10
        """,
        {"from_date": from_date, "to_date": to_date},
    )
    logger.info(
        "target races=%s (missing start_time=%s)",
        int(total_target_races["n"]) if total_target_races else 0,
        int(missing_start_time["n"]) if missing_start_time else 0,
    )

    sql = """
    WITH target_races AS (
        SELECT
            r.race_id,
            r.race_date,
            r.track_code,
            r.race_no,
            r.start_time AS post_time,
            (r.race_date::timestamp + r.start_time - INTERVAL '5 minutes') AS asof_ts
        FROM core.race r
        WHERE r.race_date BETWEEN %(from_date)s AND %(to_date)s
          AND r.track_code BETWEEN 1 AND 10
          AND r.start_time IS NOT NULL
    ),
    runner_base AS (
        SELECT
            tr.race_id,
            tr.race_date,
            tr.track_code,
            tr.race_no,
            tr.post_time,
            tr.asof_ts,
            run.horse_id,
            run.horse_no,
            run.gate,
            run.jockey_id,
            run.trainer_id,
            run.carried_weight,
            run.body_weight AS se_body_weight,
            run.body_weight_diff AS se_body_weight_diff
        FROM target_races tr
        JOIN core.runner run ON run.race_id = tr.race_id
        WHERE run.scratch_flag = FALSE
    ),
    o1_header_selected AS (
        SELECT
            rb.race_id,
            rb.asof_ts,
            picked.data_kbn,
            picked.announce_mmddhhmi,
            picked.win_pool_total_100yen
        FROM (SELECT DISTINCT race_id, asof_ts FROM runner_base) rb
        LEFT JOIN LATERAL (
            SELECT
                h.data_kbn,
                h.announce_mmddhhmi,
                h.win_pool_total_100yen
            FROM core.o1_header h
            WHERE h.race_id = rb.race_id
              AND h.data_kbn = 1
              AND h.announce_mmddhhmi <= to_char(rb.asof_ts, 'MMDDHH24MI')
            ORDER BY h.announce_mmddhhmi DESC
            LIMIT 1
        ) picked ON TRUE
    ),
    wh_header_selected AS (
        SELECT
            rb.race_id,
            rb.asof_ts,
            picked.data_kbn,
            picked.announce_mmddhhmi
        FROM (SELECT DISTINCT race_id, asof_ts FROM runner_base) rb
        LEFT JOIN LATERAL (
            SELECT
                h.data_kbn,
                h.announce_mmddhhmi
            FROM core.wh_header h
            WHERE h.race_id = rb.race_id
              AND h.announce_mmddhhmi <= to_char(rb.asof_ts, 'MMDDHH24MI')
            ORDER BY h.announce_mmddhhmi DESC
            LIMIT 1
        ) picked ON TRUE
    ),
    joined AS (
        SELECT
            rb.race_id,
            rb.race_date,
            rb.track_code,
            rb.race_no,
            rb.horse_id,
            rb.horse_no,
            rb.gate,
            rb.jockey_id,
            rb.trainer_id,
            rb.carried_weight,
            COALESCE(wd.body_weight_kg, rb.se_body_weight) AS body_weight_asof,
            CASE
                WHEN wd.horse_no IS NOT NULL THEN
                    CASE wd.diff_sign
                        WHEN '-' THEN -COALESCE(wd.diff_kg, 0)
                        WHEN '+' THEN COALESCE(wd.diff_kg, 0)
                        ELSE COALESCE(wd.diff_kg, 0)
                    END
                ELSE rb.se_body_weight_diff
            END AS body_weight_diff_asof,
            CASE WHEN wd.horse_no IS NOT NULL THEN 'WH' ELSE 'SE' END AS bw_source,
            rb.post_time,
            rb.asof_ts,
            ohs.data_kbn AS o1_data_kbn,
            ohs.announce_mmddhhmi AS o1_announce_mmddhhmi,
            CASE
                WHEN o1w.win_odds_x10 IS NULL THEN NULL
                ELSE ROUND(o1w.win_odds_x10::numeric / 10.0, 2)
            END AS odds_win_t5,
            o1w.win_popularity AS pop_win_t5,
            ohs.win_pool_total_100yen AS win_pool_total_100yen_t5,
            CASE
                WHEN ohs.announce_mmddhhmi IS NULL THEN NULL
                ELSE EXTRACT(
                    EPOCH FROM (
                        rb.asof_ts
                        - to_timestamp(
                            to_char(rb.race_date, 'YYYY') || ohs.announce_mmddhhmi,
                            'YYYYMMDDHH24MI'
                        )
                    )
                )::INT
            END AS odds_snapshot_age_sec,
            (o1w.win_odds_x10 IS NULL) AS odds_missing_flag,
            final.odds_win AS odds_win_final,
            final.pop_win AS pop_win_final,
            whs.announce_mmddhhmi AS wh_announce_mmddhhmi
        FROM runner_base rb
        LEFT JOIN o1_header_selected ohs
            ON ohs.race_id = rb.race_id
           AND ohs.asof_ts = rb.asof_ts
        LEFT JOIN core.o1_win o1w
            ON o1w.race_id = rb.race_id
           AND o1w.data_kbn = ohs.data_kbn
           AND o1w.announce_mmddhhmi = ohs.announce_mmddhhmi
           AND o1w.horse_no = rb.horse_no
        LEFT JOIN wh_header_selected whs
            ON whs.race_id = rb.race_id
           AND whs.asof_ts = rb.asof_ts
        LEFT JOIN core.wh_detail wd
            ON wd.race_id = rb.race_id
           AND wd.data_kbn = whs.data_kbn
           AND wd.announce_mmddhhmi = whs.announce_mmddhhmi
           AND wd.horse_no = rb.horse_no
        LEFT JOIN core.odds_final final
            ON final.race_id = rb.race_id
           AND final.horse_id = rb.horse_id
    ),
    ranked AS (
        SELECT
            j.*,
            CASE
                WHEN j.odds_win_t5 IS NULL THEN NULL
                ELSE RANK() OVER (
                    PARTITION BY j.race_id, j.asof_ts
                    ORDER BY j.odds_win_t5 ASC, j.horse_no ASC
                )::SMALLINT
            END AS odds_rank_t5
        FROM joined j
    )
    INSERT INTO mart.t5_runner_snapshot (
        race_id,
        race_date,
        track_code,
        race_no,
        horse_id,
        horse_no,
        gate,
        jockey_id,
        trainer_id,
        carried_weight,
        body_weight_asof,
        body_weight_diff_asof,
        bw_source,
        post_time,
        asof_ts,
        feature_set,
        o1_data_kbn,
        o1_announce_mmddhhmi,
        odds_win_t5,
        pop_win_t5,
        odds_rank_t5,
        win_pool_total_100yen_t5,
        odds_snapshot_age_sec,
        odds_missing_flag,
        odds_win_final,
        pop_win_final,
        wh_announce_mmddhhmi,
        event_change_keys,
        dm_kbn,
        dm_create_time,
        tm_kbn,
        tm_create_time,
        code_version
    )
    SELECT
        race_id,
        race_date,
        track_code,
        race_no,
        horse_id,
        horse_no,
        gate,
        jockey_id,
        trainer_id,
        carried_weight,
        body_weight_asof,
        body_weight_diff_asof,
        bw_source,
        post_time,
        asof_ts,
        %(feature_set)s,
        o1_data_kbn,
        o1_announce_mmddhhmi,
        odds_win_t5,
        pop_win_t5,
        odds_rank_t5,
        win_pool_total_100yen_t5,
        odds_snapshot_age_sec,
        odds_missing_flag,
        odds_win_final,
        pop_win_final,
        wh_announce_mmddhhmi,
        NULL::jsonb,
        NULL::smallint,
        NULL::char(12),
        NULL::smallint,
        NULL::char(12),
        %(code_version)s
    FROM ranked
    ON CONFLICT (race_id, horse_no, asof_ts) DO UPDATE SET
        race_date = EXCLUDED.race_date,
        track_code = EXCLUDED.track_code,
        race_no = EXCLUDED.race_no,
        horse_id = EXCLUDED.horse_id,
        gate = EXCLUDED.gate,
        jockey_id = EXCLUDED.jockey_id,
        trainer_id = EXCLUDED.trainer_id,
        carried_weight = EXCLUDED.carried_weight,
        body_weight_asof = EXCLUDED.body_weight_asof,
        body_weight_diff_asof = EXCLUDED.body_weight_diff_asof,
        bw_source = EXCLUDED.bw_source,
        post_time = EXCLUDED.post_time,
        feature_set = EXCLUDED.feature_set,
        o1_data_kbn = EXCLUDED.o1_data_kbn,
        o1_announce_mmddhhmi = EXCLUDED.o1_announce_mmddhhmi,
        odds_win_t5 = EXCLUDED.odds_win_t5,
        pop_win_t5 = EXCLUDED.pop_win_t5,
        odds_rank_t5 = EXCLUDED.odds_rank_t5,
        win_pool_total_100yen_t5 = EXCLUDED.win_pool_total_100yen_t5,
        odds_snapshot_age_sec = EXCLUDED.odds_snapshot_age_sec,
        odds_missing_flag = EXCLUDED.odds_missing_flag,
        odds_win_final = EXCLUDED.odds_win_final,
        pop_win_final = EXCLUDED.pop_win_final,
        wh_announce_mmddhhmi = EXCLUDED.wh_announce_mmddhhmi,
        event_change_keys = EXCLUDED.event_change_keys,
        dm_kbn = EXCLUDED.dm_kbn,
        dm_create_time = EXCLUDED.dm_create_time,
        tm_kbn = EXCLUDED.tm_kbn,
        tm_create_time = EXCLUDED.tm_create_time,
        code_version = EXCLUDED.code_version,
        updated_at = NOW()
    """
    params = {
        "from_date": from_date,
        "to_date": to_date,
        "feature_set": feature_set,
        "code_version": code_version,
    }
    db.execute(sql, params)

    stats = db.fetch_one(
        """
        SELECT
            COUNT(*) AS n_rows,
            COUNT(*) FILTER (WHERE odds_missing_flag) AS n_odds_missing,
            COUNT(*) FILTER (WHERE bw_source = 'SE') AS n_bw_from_se,
            COUNT(*) FILTER (WHERE bw_source = 'WH') AS n_bw_from_wh
        FROM mart.t5_runner_snapshot
        WHERE race_date BETWEEN %(from_date)s AND %(to_date)s
          AND feature_set = %(feature_set)s
        """,
        {"from_date": from_date, "to_date": to_date, "feature_set": feature_set},
    )
    logger.info(
        "snapshot rows=%s odds_missing=%s bw_source(SE)=%s bw_source(WH)=%s",
        int(stats["n_rows"]) if stats else 0,
        int(stats["n_odds_missing"]) if stats else 0,
        int(stats["n_bw_from_se"]) if stats else 0,
        int(stats["n_bw_from_wh"]) if stats else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="T-5 as-of スナップ生成")
    parser.add_argument("--date", type=str, help="単日 (YYYY-MM-DD)")
    parser.add_argument("--from-date", type=str, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="backfillable",
        help="feature_set 列に入れる識別子",
    )
    parser.add_argument(
        "--code-version",
        type=str,
        default="",
        help="コードバージョン。未指定時は git short hash を自動設定",
    )
    args = parser.parse_args()

    if args.date:
        from_date = date.fromisoformat(args.date)
        to_date = from_date
    else:
        if not args.from_date or not args.to_date:
            parser.error("--date または --from-date/--to-date を指定してください")
        from_date = date.fromisoformat(args.from_date)
        to_date = date.fromisoformat(args.to_date)

    if from_date > to_date:
        parser.error("from_date must be <= to_date")

    code_version = args.code_version.strip() or detect_git_version()
    logger.info(
        "build snapshot: from=%s to=%s feature_set=%s code_version=%s",
        from_date,
        to_date,
        args.feature_set,
        code_version,
    )

    with Database() as db:
        build_snapshot(
            db=db,
            from_date=from_date,
            to_date=to_date,
            feature_set=args.feature_set,
            code_version=code_version,
        )
        db.connect().commit()


if __name__ == "__main__":
    main()
