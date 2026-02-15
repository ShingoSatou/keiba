"""
JSONL → PostgreSQL ロードスクリプト (64bit Python用)

extract_jvlink.py で出力したJSONLファイルをPostgreSQLに投入します。
64bit Python環境 (.venv) で実行する必要があります。

使用方法:
    uv run python scripts/load_to_db.py --input data/RACE_20260203_123456.jsonl

    # 全ファイルを一括処理
    uv run python scripts/load_to_db.py --input-dir data/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from app.infrastructure.parsers import (  # noqa: E402
    CKRecord,
    DMRecord,
    EventChangeRecord,
    HCRecord,
    HorseExclusionRecord,
    HorseMasterRecord,
    JockeyRecord,
    OddsRecord,
    OddsTimeSeriesRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
    TMRecord,
    TrainerRecord,
    WCRecord,
    WHRecord,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAINING_BATCH_SIZE = 10000
MINING_BATCH_SIZE = 5000
CK_BATCH_SIZE = 2000

HORSE_STUB_SQL = """
    INSERT INTO core.horse (horse_id)
    VALUES (%(horse_id)s)
    ON CONFLICT (horse_id) DO NOTHING
"""

TRAINING_SLOP_SQL = """
    INSERT INTO core.training_slop (
        horse_id, training_date, data_kbn,
        training_center, training_time,
        total_4f, lap_4f,
        total_3f, lap_3f,
        total_2f, lap_2f,
        lap_1f,
        payload_raw
    ) VALUES (
        %(horse_id)s, %(training_date)s,
        %(data_kbn)s, %(training_center)s,
        %(training_time)s,
        %(total_4f)s, %(lap_4f)s,
        %(total_3f)s, %(lap_3f)s,
        %(total_2f)s, %(lap_2f)s,
        %(lap_1f)s,
        %(payload_raw)s
    )
    ON CONFLICT (
        horse_id, training_date,
        training_center, total_4f
    ) DO NOTHING
"""

TRAINING_WOOD_SQL = """
    INSERT INTO core.training_wood (
        horse_id, training_date, data_kbn,
        training_center, training_time,
        course, direction,
        total_10f, lap_10f,
        total_9f, lap_9f,
        total_8f, lap_8f,
        total_7f, lap_7f,
        total_6f, lap_6f,
        total_5f, lap_5f,
        total_4f, lap_4f,
        total_3f, lap_3f,
        total_2f, lap_2f,
        lap_1f,
        payload_raw
    ) VALUES (
        %(horse_id)s, %(training_date)s,
        %(data_kbn)s, %(training_center)s,
        %(training_time)s,
        %(course)s, %(direction)s,
        %(total_10f)s, %(lap_10f)s,
        %(total_9f)s, %(lap_9f)s,
        %(total_8f)s, %(lap_8f)s,
        %(total_7f)s, %(lap_7f)s,
        %(total_6f)s, %(lap_6f)s,
        %(total_5f)s, %(lap_5f)s,
        %(total_4f)s, %(lap_4f)s,
        %(total_3f)s, %(lap_3f)s,
        %(total_2f)s, %(lap_2f)s,
        %(lap_1f)s,
        %(payload_raw)s
    )
    ON CONFLICT (
        horse_id, training_date,
        training_center, training_time
    ) DO NOTHING
"""

MINING_DM_SQL = """
    INSERT INTO core.mining_dm (
        race_id, horse_no, data_kbn,
        dm_time_x10, dm_rank,
        payload_raw
    )
    VALUES (
        %(race_id)s, %(horse_no)s, %(data_kbn)s,
        %(dm_time_x10)s, %(dm_rank)s,
        %(payload_raw)s
    )
    ON CONFLICT (race_id, horse_no) DO UPDATE SET
        data_kbn = EXCLUDED.data_kbn,
        dm_time_x10 = EXCLUDED.dm_time_x10,
        dm_rank = EXCLUDED.dm_rank,
        payload_raw = EXCLUDED.payload_raw
"""

MINING_TM_SQL = """
    INSERT INTO core.mining_tm (
        race_id, horse_no, data_kbn,
        tm_score, tm_rank,
        payload_raw
    )
    VALUES (
        %(race_id)s, %(horse_no)s, %(data_kbn)s,
        %(tm_score)s, %(tm_rank)s,
        %(payload_raw)s
    )
    ON CONFLICT (race_id, horse_no) DO UPDATE SET
        data_kbn = EXCLUDED.data_kbn,
        tm_score = EXCLUDED.tm_score,
        tm_rank = EXCLUDED.tm_rank,
        payload_raw = EXCLUDED.payload_raw
"""

CK_RAW_SQL = """
    INSERT INTO raw.jv_ck_event (
        dataspec, record_type, data_kbn, data_create_ymd,
        kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi,
        race_no, horse_id, horse_name, payload, payload_sha256
    ) VALUES (
        %(dataspec)s, 'CK', %(data_kbn)s, %(data_create_ymd)s,
        %(kaisai_year)s, %(kaisai_md)s, %(track_cd)s, %(kaisai_kai)s, %(kaisai_nichi)s,
        %(race_no)s, %(horse_id)s, %(horse_name)s, %(payload)s, %(sha256)s
    )
    ON CONFLICT (
        dataspec, data_create_ymd, kaisai_year, kaisai_md,
        track_cd, kaisai_kai, kaisai_nichi, race_no,
        horse_id, payload_sha256
    )
    DO NOTHING
"""

CK_CORE_SQL = """
    INSERT INTO core.ck_runner_event (
        dataspec, data_create_ymd,
        kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi,
        race_no, horse_id, horse_name,
        finish_counts, style_counts, registered_races_n,
        jockey_cd, trainer_cd, owner_cd, breeder_cd,
        entity_prize
    ) VALUES (
        %(dataspec)s, %(dc_ymd)s,
        %(kaisai_year)s, %(kaisai_md)s, %(track_cd)s, %(kaisai_kai)s, %(kaisai_nichi)s,
        %(race_no)s, %(horse_id)s, %(horse_name)s,
        %(finish_counts)s, %(style_counts)s, %(reg_races)s,
        %(jockey_cd)s, %(trainer_cd)s, %(owner_cd)s, %(breeder_cd)s,
        %(entity_prize)s
    )
    ON CONFLICT (
        dataspec, data_create_ymd, kaisai_year,
        kaisai_md, track_cd, kaisai_kai,
        kaisai_nichi, race_no, horse_id
    )
    DO UPDATE SET
        horse_name = EXCLUDED.horse_name,
        finish_counts = EXCLUDED.finish_counts,
        style_counts = EXCLUDED.style_counts,
        registered_races_n = EXCLUDED.registered_races_n,
        jockey_cd = EXCLUDED.jockey_cd,
        trainer_cd = EXCLUDED.trainer_cd,
        owner_cd = EXCLUDED.owner_cd,
        breeder_cd = EXCLUDED.breeder_cd,
        entity_prize = EXCLUDED.entity_prize
"""

CK_FEAT_SQL = """
    INSERT INTO mart.feat_ck_win (
        kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id,
        dataspec, data_create_ymd,
        h_total_starts, h_total_wins, h_total_top3, h_total_top5, h_total_out,
        h_central_starts, h_central_wins, h_central_top3,
        h_turf_right_starts, h_turf_left_starts, h_turf_straight_starts,
        h_dirt_right_starts, h_dirt_left_starts, h_dirt_straight_starts,
        h_turf_good_starts, h_turf_soft_starts, h_turf_heavy_starts, h_turf_bad_starts,
        h_dirt_good_starts, h_dirt_soft_starts, h_dirt_heavy_starts, h_dirt_bad_starts,
        h_turf_16down_starts, h_turf_22down_starts, h_turf_22up_starts,
        h_dirt_16down_starts, h_dirt_22down_starts, h_dirt_22up_starts,
        h_style_nige_cnt, h_style_senko_cnt, h_style_sashi_cnt, h_style_oikomi_cnt,
        h_registered_races_n,
        j_year_flat_prize_total, j_year_ob_prize_total, j_cum_flat_prize_total,
        j_cum_ob_prize_total,
        t_year_flat_prize_total, t_year_ob_prize_total, t_cum_flat_prize_total,
        t_cum_ob_prize_total,
        o_year_prize_total, o_cum_prize_total, b_year_prize_total, b_cum_prize_total
    ) VALUES (
        %(kaisai_year)s, %(kaisai_md)s, %(track_cd)s, %(kaisai_kai)s,
        %(kaisai_nichi)s, %(race_no)s, %(horse_id)s,
        %(dataspec)s, %(data_create_ymd)s,
        %(h_total_starts)s, %(h_total_wins)s, %(h_total_top3)s,
        %(h_total_top5)s, %(h_total_out)s,
        %(h_central_starts)s, %(h_central_wins)s, %(h_central_top3)s,
        %(h_turf_right_starts)s, %(h_turf_left_starts)s, %(h_turf_straight_starts)s,
        %(h_dirt_right_starts)s, %(h_dirt_left_starts)s, %(h_dirt_straight_starts)s,
        %(h_turf_good_starts)s, %(h_turf_soft_starts)s,
        %(h_turf_heavy_starts)s, %(h_turf_bad_starts)s,
        %(h_dirt_good_starts)s, %(h_dirt_soft_starts)s,
        %(h_dirt_heavy_starts)s, %(h_dirt_bad_starts)s,
        %(h_turf_16down_starts)s, %(h_turf_22down_starts)s, %(h_turf_22up_starts)s,
        %(h_dirt_16down_starts)s, %(h_dirt_22down_starts)s, %(h_dirt_22up_starts)s,
        %(h_style_nige_cnt)s, %(h_style_senko_cnt)s,
        %(h_style_sashi_cnt)s, %(h_style_oikomi_cnt)s,
        %(h_registered_races_n)s,
        %(j_year_flat_prize_total)s, %(j_year_ob_prize_total)s,
        %(j_cum_flat_prize_total)s, %(j_cum_ob_prize_total)s,
        %(t_year_flat_prize_total)s, %(t_year_ob_prize_total)s,
        %(t_cum_flat_prize_total)s, %(t_cum_ob_prize_total)s,
        %(o_year_prize_total)s, %(o_cum_prize_total)s,
        %(b_year_prize_total)s, %(b_cum_prize_total)s
    )
    ON CONFLICT (
        kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id,
        dataspec, data_create_ymd
    )
    DO UPDATE SET
        h_total_starts = EXCLUDED.h_total_starts,
        h_total_wins = EXCLUDED.h_total_wins,
        h_total_top3 = EXCLUDED.h_total_top3,
        h_total_top5 = EXCLUDED.h_total_top5,
        h_total_out = EXCLUDED.h_total_out,
        h_central_starts = EXCLUDED.h_central_starts,
        h_central_wins = EXCLUDED.h_central_wins,
        h_central_top3 = EXCLUDED.h_central_top3,
        h_turf_right_starts = EXCLUDED.h_turf_right_starts,
        h_turf_left_starts = EXCLUDED.h_turf_left_starts,
        h_turf_straight_starts = EXCLUDED.h_turf_straight_starts,
        h_dirt_right_starts = EXCLUDED.h_dirt_right_starts,
        h_dirt_left_starts = EXCLUDED.h_dirt_left_starts,
        h_dirt_straight_starts = EXCLUDED.h_dirt_straight_starts,
        h_turf_good_starts = EXCLUDED.h_turf_good_starts,
        h_turf_soft_starts = EXCLUDED.h_turf_soft_starts,
        h_turf_heavy_starts = EXCLUDED.h_turf_heavy_starts,
        h_turf_bad_starts = EXCLUDED.h_turf_bad_starts,
        h_dirt_good_starts = EXCLUDED.h_dirt_good_starts,
        h_dirt_soft_starts = EXCLUDED.h_dirt_soft_starts,
        h_dirt_heavy_starts = EXCLUDED.h_dirt_heavy_starts,
        h_dirt_bad_starts = EXCLUDED.h_dirt_bad_starts,
        h_turf_16down_starts = EXCLUDED.h_turf_16down_starts,
        h_turf_22down_starts = EXCLUDED.h_turf_22down_starts,
        h_turf_22up_starts = EXCLUDED.h_turf_22up_starts,
        h_dirt_16down_starts = EXCLUDED.h_dirt_16down_starts,
        h_dirt_22down_starts = EXCLUDED.h_dirt_22down_starts,
        h_dirt_22up_starts = EXCLUDED.h_dirt_22up_starts,
        h_style_nige_cnt = EXCLUDED.h_style_nige_cnt,
        h_style_senko_cnt = EXCLUDED.h_style_senko_cnt,
        h_style_sashi_cnt = EXCLUDED.h_style_sashi_cnt,
        h_style_oikomi_cnt = EXCLUDED.h_style_oikomi_cnt,
        h_registered_races_n = EXCLUDED.h_registered_races_n,
        j_year_flat_prize_total = EXCLUDED.j_year_flat_prize_total,
        j_year_ob_prize_total = EXCLUDED.j_year_ob_prize_total,
        j_cum_flat_prize_total = EXCLUDED.j_cum_flat_prize_total,
        j_cum_ob_prize_total = EXCLUDED.j_cum_ob_prize_total,
        t_year_flat_prize_total = EXCLUDED.t_year_flat_prize_total,
        t_year_ob_prize_total = EXCLUDED.t_year_ob_prize_total,
        t_cum_flat_prize_total = EXCLUDED.t_cum_flat_prize_total,
        t_cum_ob_prize_total = EXCLUDED.t_cum_ob_prize_total,
        o_year_prize_total = EXCLUDED.o_year_prize_total,
        o_cum_prize_total = EXCLUDED.o_cum_prize_total,
        b_year_prize_total = EXCLUDED.b_year_prize_total,
        b_cum_prize_total = EXCLUDED.b_cum_prize_total
"""


def calc_payload_hash(payload: str) -> bytes:
    """payload(TEXT) のSHA256ハッシュを計算（raw.jv_raw の重複排除用）"""
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).digest()


def ensure_race_stub(db: Database, race_id: int, cache: set[int] | None = None) -> None:
    """core.race に race_id のstub行がなければ作成する（FK違反回避）"""
    if not race_id:
        return

    try:
        r_id = int(race_id)
    except (TypeError, ValueError):
        return

    if r_id <= 0:
        return
    if cache is not None and r_id in cache:
        return

    race_no = r_id % 100
    track_code = (r_id // 100) % 100
    date_int = int(r_id // 10000)
    y = date_int // 10000
    m = (date_int // 100) % 100
    d = date_int % 100
    try:
        race_date = date(y, m, d).isoformat()
    except ValueError:
        race_date = "2000-01-01"

    sql = """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no,
            surface, distance_m
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s,
            0, 0
        )
        ON CONFLICT (race_id) DO NOTHING
    """
    db.execute(
        sql,
        {
            "race_id": r_id,
            "race_date": race_date,
            "track_code": track_code,
            "race_no": race_no,
        },
    )
    if cache is not None:
        cache.add(r_id)


def load_jsonl(file_path: Path):
    """JSONLファイルを読み込み"""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def insert_raw_record(db: Database, dataspec: str, record: dict) -> None:
    """raw.jv_rawにレコードを挿入（単一レコード版）"""
    payload_hash = calc_payload_hash(record["payload"])
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload, payload_hash)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (dataspec, rec_id, payload_hash) DO NOTHING
    """
    db.execute(
        sql,
        (
            dataspec,
            record["rec_id"],
            record["filename"],
            record["payload"],
            payload_hash,
        ),
    )


def insert_raw_records_batch(db: Database, dataspec: str, records: list[dict]) -> int:
    """raw.jv_rawにレコードをバッチ挿入（高速版）"""
    if not records:
        return 0
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload, payload_hash)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (dataspec, rec_id, payload_hash) DO NOTHING
    """
    values = [
        (dataspec, r["rec_id"], r["filename"], r["payload"], calc_payload_hash(r["payload"]))
        for r in records
    ]
    cursor = db.connect().cursor()
    cursor.executemany(sql, values)
    return len(records)


def upsert_race(db: Database, race: RaceRecord) -> None:
    """core.raceにレースを挿入/更新"""
    sql = """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface,
            distance_m, going, weather, class_code, field_size, start_time
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, %(surface)s,
            %(distance_m)s, %(going)s, %(weather)s,
            %(class_code)s, %(field_size)s, %(start_time)s
        )
        ON CONFLICT (race_id) DO UPDATE SET
            going = EXCLUDED.going,
            weather = EXCLUDED.weather,
            field_size = EXCLUDED.field_size,
            start_time = EXCLUDED.start_time,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "track_code": race.track_code,
            "race_no": race.race_no,
            "surface": race.surface,
            "distance_m": race.distance_m,
            "going": race.going,
            "weather": race.weather,
            "class_code": race.class_code,
            "field_size": race.field_size,
            "start_time": race.start_time,
        },
    )


def prepare_master_data_cache(db: Database) -> tuple[set[int], set[int]]:
    """マスタデータのIDセットをキャッシュとして取得"""
    jockeys = {row["jockey_id"] for row in db.fetch_all("SELECT jockey_id FROM core.jockey")}
    trainers = {row["trainer_id"] for row in db.fetch_all("SELECT trainer_id FROM core.trainer")}
    return jockeys, trainers


def upsert_runner(
    db: Database,
    runner: RunnerRecord,
    master_jockeys: set[int] = None,
    master_trainers: set[int] = None,
) -> None:
    """core.runner と core.result にデータを挿入/更新"""

    # FK制御ロジック
    # 1. 地方競馬(A)や海外(B2)はマスタ参照できないためNULL
    # 2. それ以外でもマスタに存在しない場合はNULL (エラー回避)

    # マスタキャッシュが渡されていない場合は空集合として扱う（安全策）
    if master_jockeys is None:
        master_jockeys = set()
    if master_trainers is None:
        master_trainers = set()

    safe_jockey_id = runner.jockey_id
    safe_trainer_id = runner.trainer_id

    # データ区分によるフィルタ
    # 'A': 地方, 'B2': 海外
    # これらは仕様上コードが入っていてもマスタとリンクしない
    if runner.data_kubun in ("A", "B2"):
        safe_jockey_id = None
        safe_trainer_id = None
    else:
        # マスタ存在チェック (中央系でもマスタ欠けがある場合への対処)
        if safe_jockey_id is not None and safe_jockey_id not in master_jockeys:
            # logger.warning(f"Missing Jockey Master: {safe_jockey_id} (Race: {runner.race_id})")
            safe_jockey_id = None

        if safe_trainer_id is not None and safe_trainer_id not in master_trainers:
            # logger.warning(f"Missing Trainer Master: {safe_trainer_id} (Race: {runner.race_id})")
            safe_trainer_id = None

    # Missing Horse対策: 馬マスタになければ、SEレコードの情報で自動登録する
    # (地方馬や、UMレコード未着のケースに対応)
    horse_sql = """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO NOTHING
    """
    db.execute(horse_sql, {"horse_id": runner.horse_id, "horse_name": runner.horse_name})

    # FK違反回避のため、core.race のstubを先に用意する
    ensure_race_stub(db, runner.race_id)

    # runner (出走表)
    runner_sql = """
        INSERT INTO core.runner (
            race_id, horse_id, horse_no, gate, jockey_id,
            trainer_id, carried_weight, body_weight, body_weight_diff
        ) VALUES (
            %(race_id)s, %(horse_id)s, %(horse_no)s, %(gate)s, %(jockey_id)s,
            %(trainer_id)s, %(carried_weight)s, %(body_weight)s, %(body_weight_diff)s
        )
        ON CONFLICT (race_id, horse_id) DO UPDATE SET
            horse_no = EXCLUDED.horse_no,
            gate = EXCLUDED.gate,
            jockey_id = EXCLUDED.jockey_id,
            trainer_id = EXCLUDED.trainer_id,
            carried_weight = EXCLUDED.carried_weight,
            body_weight = EXCLUDED.body_weight,
            body_weight_diff = EXCLUDED.body_weight_diff,
            updated_at = now()
    """
    db.execute(
        runner_sql,
        {
            "race_id": runner.race_id,
            "horse_id": runner.horse_id,
            "horse_no": runner.horse_no,
            "gate": runner.gate,
            "jockey_id": safe_jockey_id,
            "trainer_id": safe_trainer_id,
            "carried_weight": runner.carried_weight,
            "body_weight": runner.body_weight,
            "body_weight_diff": runner.body_weight_diff,
        },
    )

    # result (結果) - 着順がある場合のみ
    if runner.finish_pos and runner.finish_pos > 0:
        result_sql = """
            INSERT INTO core.result (
                race_id, horse_id, finish_pos, time_sec,
                margin, final3f_sec, corner1_pos, corner2_pos, corner3_pos, corner4_pos
            ) VALUES (
                %(race_id)s, %(horse_id)s, %(finish_pos)s, %(time_sec)s,
                %(margin)s, %(final3f_sec)s, %(corner1_pos)s, %(corner2_pos)s,
                %(corner3_pos)s, %(corner4_pos)s
            )
            ON CONFLICT (race_id, horse_id) DO UPDATE SET
                finish_pos = EXCLUDED.finish_pos,
                time_sec = EXCLUDED.time_sec,
                margin = EXCLUDED.margin,
                final3f_sec = EXCLUDED.final3f_sec,
                corner1_pos = EXCLUDED.corner1_pos,
                corner2_pos = EXCLUDED.corner2_pos,
                corner3_pos = EXCLUDED.corner3_pos,
                corner4_pos = EXCLUDED.corner4_pos,
                updated_at = now()
        """
        db.execute(
            result_sql,
            {
                "race_id": runner.race_id,
                "horse_id": runner.horse_id,
                "finish_pos": runner.finish_pos,
                "time_sec": runner.time_sec,
                "margin": runner.margin,
                "final3f_sec": runner.final3f_sec,
                "corner1_pos": runner.corner1_pos,
                "corner2_pos": runner.corner2_pos,
                "corner3_pos": runner.corner3_pos,
                "corner4_pos": runner.corner4_pos,
            },
        )


def upsert_payout(db: Database, payout: PayoutRecord) -> None:
    """core.payoutにデータを挿入/更新"""
    sql = """
        INSERT INTO core.payout (race_id, bet_type, selection, payout_yen, popularity)
        VALUES (%(race_id)s, %(bet_type)s, %(selection)s, %(payout_yen)s, %(popularity)s)
        ON CONFLICT (race_id, bet_type, selection) DO UPDATE SET
            payout_yen = EXCLUDED.payout_yen,
            popularity = EXCLUDED.popularity
    """
    db.execute(
        sql,
        {
            "race_id": payout.race_id,
            "bet_type": payout.bet_type,
            "selection": payout.selection,
            "payout_yen": payout.payout_yen,
            "popularity": payout.popularity,
        },
    )


def upsert_odds(db: Database, odds: OddsRecord) -> int:
    """core.odds_finalにデータを挿入/更新

    Note: O1レコードにはhorse_idがないため、core.runnerからhorse_noで引いて解決する
    bet_type=1 (Win), bet_type=2 (Place) のみ処理
    """
    if odds.bet_type == 1:  # Win
        sql = """
            WITH upserted AS (
                INSERT INTO core.odds_final (race_id, horse_id, odds_win, pop_win)
                SELECT %(race_id)s, horse_id, %(odds_win)s, %(pop_win)s
                FROM core.runner
                WHERE race_id = %(race_id)s AND horse_no = %(horse_no)s
                ON CONFLICT (race_id, horse_id) DO UPDATE SET
                    odds_win = EXCLUDED.odds_win,
                    pop_win = EXCLUDED.pop_win,
                    updated_at = now()
                RETURNING 1
            )
            SELECT COUNT(*) AS n FROM upserted
        """
        result = db.fetch_one(
            sql,
            {
                "race_id": odds.race_id,
                "horse_no": int(odds.horse_no) if isinstance(odds.horse_no, str) else odds.horse_no,
                "odds_win": odds.odds_1,
                "pop_win": odds.popularity,
            },
        )
        return int(result["n"]) if result else 0
    elif odds.bet_type == 2:  # Place
        sql = """
            WITH upserted AS (
                INSERT INTO core.odds_final
                    (race_id, horse_id, odds_place_low, odds_place_high, pop_place)
                SELECT %(race_id)s, horse_id, %(odds_place_low)s,
                    %(odds_place_high)s, %(pop_place)s
                FROM core.runner
                WHERE race_id = %(race_id)s AND horse_no = %(horse_no)s
                ON CONFLICT (race_id, horse_id) DO UPDATE SET
                    odds_place_low = EXCLUDED.odds_place_low,
                    odds_place_high = EXCLUDED.odds_place_high,
                    pop_place = EXCLUDED.pop_place,
                    updated_at = now()
                RETURNING 1
            )
            SELECT COUNT(*) AS n FROM upserted
        """
        result = db.fetch_one(
            sql,
            {
                "race_id": odds.race_id,
                "horse_no": int(odds.horse_no) if isinstance(odds.horse_no, str) else odds.horse_no,
                "odds_place_low": odds.odds_1,
                "odds_place_high": odds.odds_2,
                "pop_place": odds.popularity,
            },
        )
        return int(result["n"]) if result else 0
    # bet_type=3 (Bracket) は現状スキップ（horse_idへの変換が困難）
    return 0


def upsert_horse(db: Database, horse: HorseExclusionRecord) -> None:
    """core.horseにデータを挿入/更新

    JGレコードは競走馬除外情報なので、馬名のみ更新。
    """
    sql = """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = EXCLUDED.horse_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "horse_id": horse.horse_id,
            "horse_name": horse.horse_name,
        },
    )


def upsert_horse_master(db: Database, horse: HorseMasterRecord) -> None:
    """core.horseにデータを挿入/更新

    UMレコード（馬マスタ）から馬情報を登録。
    """
    sql = """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = EXCLUDED.horse_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "horse_id": horse.horse_id,
            "horse_name": horse.horse_name,
        },
    )


def upsert_jockey(db: Database, jockey: JockeyRecord) -> None:
    """core.jockeyにデータを挿入/更新"""
    sql = """
        INSERT INTO core.jockey (jockey_id, jockey_name)
        VALUES (%(jockey_id)s, %(jockey_name)s)
        ON CONFLICT (jockey_id) DO UPDATE SET
            jockey_name = EXCLUDED.jockey_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "jockey_id": jockey.jockey_id,
            "jockey_name": jockey.jockey_name,
        },
    )


def upsert_trainer(db: Database, trainer: TrainerRecord) -> None:
    """core.trainerにデータを挿入/更新"""
    sql = """
        INSERT INTO core.trainer (trainer_id, trainer_name)
        VALUES (%(trainer_id)s, %(trainer_name)s)
        ON CONFLICT (trainer_id) DO UPDATE SET
            trainer_name = EXCLUDED.trainer_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "trainer_id": trainer.trainer_id,
            "trainer_name": trainer.trainer_name,
        },
    )


def upsert_o1_timeseries(
    db: Database, records: list[OddsTimeSeriesRecord], race_stub_cache: set[int] | None = None
) -> int:
    """core.o1_header / core.o1_win に時系列オッズを挿入/更新"""
    if not records:
        return 0

    # ヘッダー情報は最初のレコードから取得
    first = records[0]
    # FK (core.o1_header -> core.race) 回避のため、stub race を用意
    ensure_race_stub(db, first.race_id, cache=race_stub_cache)
    header_sql = """
        INSERT INTO core.o1_header (race_id, data_kbn, announce_mmddhhmi, win_pool_total_100yen)
        VALUES (%(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s, %(win_pool_total_100yen)s)
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO NOTHING
    """
    db.execute(
        header_sql,
        {
            "race_id": first.race_id,
            "data_kbn": first.data_kbn,
            "announce_mmddhhmi": first.announce_mmddhhmi,
            "win_pool_total_100yen": first.win_pool_total_100yen,
        },
    )

    detail_sql = """
        INSERT INTO core.o1_win (
            race_id, data_kbn, announce_mmddhhmi,
            horse_no, win_odds_x10, win_popularity
        ) VALUES (
            %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
            %(horse_no)s, %(win_odds_x10)s, %(win_popularity)s
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, horse_no)
        DO UPDATE SET
            win_odds_x10 = EXCLUDED.win_odds_x10,
            win_popularity = EXCLUDED.win_popularity
    """
    detail_params = [
        {
            "race_id": r.race_id,
            "data_kbn": r.data_kbn,
            "announce_mmddhhmi": r.announce_mmddhhmi,
            "horse_no": r.horse_no,
            "win_odds_x10": r.win_odds_x10,
            "win_popularity": r.win_popularity,
        }
        for r in records
    ]
    db.execute_many(detail_sql, detail_params)
    return len(detail_params)


def upsert_o1_timeseries_bulk(
    db: Database, records: list[OddsTimeSeriesRecord], race_stub_cache: set[int] | None = None
) -> int:
    """core.o1_header / core.o1_win をまとめて挿入/更新（0B41高速処理向け）"""
    if not records:
        return 0

    header_map: dict[tuple[int, int, str], dict] = {}
    for r in records:
        key = (r.race_id, r.data_kbn, r.announce_mmddhhmi)
        # 同一キーが複数回現れる場合は最後の値を採用
        header_map[key] = {
            "race_id": r.race_id,
            "data_kbn": r.data_kbn,
            "announce_mmddhhmi": r.announce_mmddhhmi,
            "win_pool_total_100yen": r.win_pool_total_100yen,
        }

    for header in header_map.values():
        ensure_race_stub(db, header["race_id"], cache=race_stub_cache)

    header_sql = """
        INSERT INTO core.o1_header (race_id, data_kbn, announce_mmddhhmi, win_pool_total_100yen)
        VALUES (%(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s, %(win_pool_total_100yen)s)
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
            win_pool_total_100yen = EXCLUDED.win_pool_total_100yen
    """
    db.execute_many(header_sql, list(header_map.values()))

    detail_map: dict[tuple[int, int, str, int], dict] = {}
    for r in records:
        detail_map[(r.race_id, r.data_kbn, r.announce_mmddhhmi, r.horse_no)] = {
            "race_id": r.race_id,
            "data_kbn": r.data_kbn,
            "announce_mmddhhmi": r.announce_mmddhhmi,
            "horse_no": r.horse_no,
            "win_odds_x10": r.win_odds_x10,
            "win_popularity": r.win_popularity,
        }
    detail_rows = list(detail_map.values())
    detail_sql = """
        INSERT INTO core.o1_win (
            race_id, data_kbn, announce_mmddhhmi,
            horse_no, win_odds_x10, win_popularity
        )
        SELECT
            x.race_id, x.data_kbn, x.announce_mmddhhmi,
            x.horse_no, x.win_odds_x10, x.win_popularity
        FROM jsonb_to_recordset(%(rows_json)s::jsonb) AS x(
            race_id bigint,
            data_kbn integer,
            announce_mmddhhmi text,
            horse_no integer,
            win_odds_x10 integer,
            win_popularity integer
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, horse_no)
        DO UPDATE SET
            win_odds_x10 = EXCLUDED.win_odds_x10,
            win_popularity = EXCLUDED.win_popularity
    """
    db.execute(detail_sql, {"rows_json": json.dumps(detail_rows, ensure_ascii=False)})
    return len(detail_rows)


def upsert_wh_records(db: Database, records: list[WHRecord]) -> int:
    """core.wh_header / core.wh_detail に馬体重を挿入/更新"""
    if not records:
        return 0

    first = records[0]
    header_sql = """
        INSERT INTO core.wh_header (race_id, data_kbn, announce_mmddhhmi)
        VALUES (%(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s)
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO NOTHING
    """
    db.execute(
        header_sql,
        {
            "race_id": first.race_id,
            "data_kbn": first.data_kbn,
            "announce_mmddhhmi": first.announce_mmddhhmi,
        },
    )

    count = 0
    detail_sql = """
        INSERT INTO core.wh_detail (
            race_id, data_kbn, announce_mmddhhmi,
            horse_no, body_weight_kg, diff_sign, diff_kg
        ) VALUES (
            %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
            %(horse_no)s, %(body_weight_kg)s,
            %(diff_sign)s, %(diff_kg)s
        )
        ON CONFLICT (
            race_id, data_kbn, announce_mmddhhmi, horse_no
        ) DO UPDATE SET
            body_weight_kg = EXCLUDED.body_weight_kg,
            diff_sign = EXCLUDED.diff_sign,
            diff_kg = EXCLUDED.diff_kg
    """
    for r in records:
        db.execute(
            detail_sql,
            {
                "race_id": r.race_id,
                "data_kbn": r.data_kbn,
                "announce_mmddhhmi": r.announce_mmddhhmi,
                "horse_no": r.horse_no,
                "body_weight_kg": r.body_weight_kg,
                "diff_sign": r.diff_sign,
                "diff_kg": r.diff_kg,
            },
        )
        count += 1

    return count


def _training_params(rec_id: str, record) -> dict:
    if rec_id == "HC":
        return {
            "horse_id": record.horse_id,
            "training_date": record.training_date,
            "data_kbn": record.data_kbn,
            "training_center": record.training_center,
            "training_time": record.training_time,
            "total_4f": record.total_4f,
            "lap_4f": record.lap_4f,
            "total_3f": record.total_3f,
            "lap_3f": record.lap_3f,
            "total_2f": record.total_2f,
            "lap_2f": record.lap_2f,
            "lap_1f": record.lap_1f,
            "payload_raw": record.payload_raw,
        }
    return {
        "horse_id": record.horse_id,
        "training_date": record.training_date,
        "data_kbn": record.data_kbn,
        "training_center": record.training_center,
        "training_time": record.training_time,
        "course": record.course,
        "direction": record.direction,
        "total_10f": record.total_10f,
        "lap_10f": record.lap_10f,
        "total_9f": record.total_9f,
        "lap_9f": record.lap_9f,
        "total_8f": record.total_8f,
        "lap_8f": record.lap_8f,
        "total_7f": record.total_7f,
        "lap_7f": record.lap_7f,
        "total_6f": record.total_6f,
        "lap_6f": record.lap_6f,
        "total_5f": record.total_5f,
        "lap_5f": record.lap_5f,
        "total_4f": record.total_4f,
        "lap_4f": record.lap_4f,
        "total_3f": record.total_3f,
        "lap_3f": record.lap_3f,
        "total_2f": record.total_2f,
        "lap_2f": record.lap_2f,
        "lap_1f": record.lap_1f,
        "payload_raw": record.payload_raw,
    }


def ensure_horse_stubs_batch(db: Database, horse_ids: list[str]) -> None:
    unique_ids = sorted({horse_id for horse_id in horse_ids if horse_id})
    if not unique_ids:
        return
    db.execute_many(HORSE_STUB_SQL, [{"horse_id": horse_id} for horse_id in unique_ids])


def insert_training_records_batch(db: Database, rec_id: str, records: list) -> int:
    """core.training_slop / core.training_wood に調教データをバッチ挿入"""
    if not records:
        return 0
    ensure_horse_stubs_batch(db, [record.horse_id for record in records])
    sql = TRAINING_SLOP_SQL if rec_id == "HC" else TRAINING_WOOD_SQL
    db.execute_many(sql, [_training_params(rec_id, record) for record in records])
    return len(records)


def insert_training_record(db: Database, rec_id: str, record) -> None:
    """互換用: 単一レコード挿入"""
    insert_training_records_batch(db, rec_id, [record])


def _mining_params(record) -> dict:
    return {
        "race_id": record.race_id,
        "horse_no": record.horse_no,
        "data_kbn": record.data_kbn,
        "dm_time_x10": getattr(record, "dm_time_x10", None),
        "dm_rank": getattr(record, "dm_rank", None),
        "tm_score": getattr(record, "tm_score", None),
        "tm_rank": getattr(record, "tm_rank", None),
        "payload_raw": record.payload_raw,
    }


def insert_mining_records_batch(db: Database, rec_id: str, records: list) -> int:
    """core.mining_dm / core.mining_tm にマイニングデータをバッチ挿入"""
    if not records:
        return 0
    sql = MINING_DM_SQL if rec_id == "DM" else MINING_TM_SQL
    db.execute_many(sql, [_mining_params(record) for record in records])
    return len(records)


def insert_mining_record(db: Database, rec_id: str, record) -> None:
    """互換用: 単一レコード挿入"""
    insert_mining_records_batch(db, rec_id, [record])


def insert_event_change(db: Database, record: EventChangeRecord) -> None:
    """core.event_change に当日変更を挿入"""
    # FK (core.event_change -> core.race) 回避のため stub race を用意
    ensure_race_stub(db, record.race_id)
    sql = """
        INSERT INTO core.event_change (
            race_id, record_type, data_create_ymd, announce_mmddhhmi, payload_parsed
        )
        VALUES (
            %(race_id)s, %(record_type)s, %(data_create_ymd)s, %(announce_mmddhhmi)s, %(payload)s
        )
    """
    import json as _json

    payload = dict(record.payload_parsed or {})
    payload.setdefault("raw", record.payload_raw)

    db.execute(
        sql,
        {
            "race_id": record.race_id,
            "record_type": record.record_type,
            "data_create_ymd": record.data_create_ymd,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "payload": _json.dumps(payload, ensure_ascii=False),
        },
    )


def _build_ck_payloads(dataspec: str, record: CKRecord) -> tuple[dict, dict, dict] | None:
    """CKレコードから raw/core/mart の投入パラメータを生成"""
    if record.make_date is None:
        return None

    payload_bytes = record.payload_raw.encode("cp932", errors="replace")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    full_stats = record.get_full_stats()

    def _summary_counts(counts: list[int]) -> tuple[int, int, int, int, int]:
        c = (counts + [0] * 6)[:6]
        c1, c2, c3, c4, c5, cout = c
        starts = c1 + c2 + c3 + c4 + c5 + cout
        wins = c1
        top3 = c1 + c2 + c3
        top5 = c1 + c2 + c3 + c4 + c5
        out = cout
        return starts, wins, top3, top5, out

    def _starts(counts: list[int]) -> int:
        return sum((counts + [0] * 6)[:6])

    fc = full_stats["finish_counts"]
    style = full_stats["style_counts"]
    prize = full_stats["entity_prize"]

    h_total_starts, h_total_wins, h_total_top3, h_total_top5, h_total_out = _summary_counts(
        record.counts_total
    )
    h_central_starts, h_central_wins, h_central_top3, _, _ = _summary_counts(record.counts_central)

    turf_16down = (
        _starts(fc["turf_1200_down"])
        + _starts(fc["turf_1201_1400"])
        + _starts(fc["turf_1401_1600"])
    )
    turf_22down = (
        _starts(fc["turf_1601_1800"])
        + _starts(fc["turf_1801_2000"])
        + _starts(fc["turf_2001_2200"])
    )
    turf_22up = (
        _starts(fc["turf_2201_2400"]) + _starts(fc["turf_2401_2800"]) + _starts(fc["turf_2801_up"])
    )
    dirt_16down = (
        _starts(fc["dirt_1200_down"])
        + _starts(fc["dirt_1201_1400"])
        + _starts(fc["dirt_1401_1600"])
    )
    dirt_22down = (
        _starts(fc["dirt_1601_1800"])
        + _starts(fc["dirt_1801_2000"])
        + _starts(fc["dirt_2001_2200"])
    )
    dirt_22up = (
        _starts(fc["dirt_2201_2400"]) + _starts(fc["dirt_2401_2800"]) + _starts(fc["dirt_2801_up"])
    )

    j_prize = prize.get("jockey", {}) or {}
    t_prize = prize.get("trainer", {}) or {}
    o_prize = prize.get("owner", {}) or {}
    b_prize = prize.get("breeder", {}) or {}

    raw_payload = {
        "dataspec": dataspec,
        "data_kbn": record.data_kbn,
        "data_create_ymd": record.make_date,
        "kaisai_year": record.kaisai_year,
        "kaisai_md": record.kaisai_md,
        "track_cd": record.track_cd,
        "kaisai_kai": record.kaisai_kai,
        "kaisai_nichi": record.kaisai_nichi,
        "race_no": record.race_no,
        "horse_id": record.horse_id,
        "horse_name": record.horse_name,
        "payload": payload_bytes,
        "sha256": payload_sha256,
    }
    core_payload = {
        "dataspec": dataspec,
        "dc_ymd": record.make_date,
        "kaisai_year": record.kaisai_year,
        "kaisai_md": record.kaisai_md,
        "track_cd": record.track_cd,
        "kaisai_kai": record.kaisai_kai,
        "kaisai_nichi": record.kaisai_nichi,
        "race_no": record.race_no,
        "horse_id": record.horse_id,
        "horse_name": record.horse_name,
        "finish_counts": json.dumps(full_stats["finish_counts"]),
        "style_counts": json.dumps(full_stats["style_counts"]),
        "reg_races": full_stats["registered_races"],
        "jockey_cd": record.jockey_code,
        "trainer_cd": record.trainer_code,
        "owner_cd": record.owner_code,
        "breeder_cd": record.breeder_code,
        "entity_prize": json.dumps(full_stats["entity_prize"]),
    }
    feat_payload = {
        "kaisai_year": record.kaisai_year,
        "kaisai_md": record.kaisai_md,
        "track_cd": record.track_cd,
        "kaisai_kai": record.kaisai_kai,
        "kaisai_nichi": record.kaisai_nichi,
        "race_no": record.race_no,
        "horse_id": record.horse_id,
        "dataspec": dataspec,
        "data_create_ymd": record.make_date,
        "h_total_starts": h_total_starts,
        "h_total_wins": h_total_wins,
        "h_total_top3": h_total_top3,
        "h_total_top5": h_total_top5,
        "h_total_out": h_total_out,
        "h_central_starts": h_central_starts,
        "h_central_wins": h_central_wins,
        "h_central_top3": h_central_top3,
        "h_turf_right_starts": _starts(fc["turf_right"]),
        "h_turf_left_starts": _starts(fc["turf_left"]),
        "h_turf_straight_starts": _starts(fc["turf_str"]),
        "h_dirt_right_starts": _starts(fc["dirt_right"]),
        "h_dirt_left_starts": _starts(fc["dirt_left"]),
        "h_dirt_straight_starts": _starts(fc["dirt_str"]),
        "h_turf_good_starts": _starts(fc["turf_good"]),
        "h_turf_soft_starts": _starts(fc["turf_soft"]),
        "h_turf_heavy_starts": _starts(fc["turf_heavy"]),
        "h_turf_bad_starts": _starts(fc["turf_bad"]),
        "h_dirt_good_starts": _starts(fc["dirt_good"]),
        "h_dirt_soft_starts": _starts(fc["dirt_soft"]),
        "h_dirt_heavy_starts": _starts(fc["dirt_heavy"]),
        "h_dirt_bad_starts": _starts(fc["dirt_bad"]),
        "h_turf_16down_starts": turf_16down,
        "h_turf_22down_starts": turf_22down,
        "h_turf_22up_starts": turf_22up,
        "h_dirt_16down_starts": dirt_16down,
        "h_dirt_22down_starts": dirt_22down,
        "h_dirt_22up_starts": dirt_22up,
        "h_style_nige_cnt": int(style.get("nige") or 0),
        "h_style_senko_cnt": int(style.get("senko") or 0),
        "h_style_sashi_cnt": int(style.get("sashi") or 0),
        "h_style_oikomi_cnt": int(style.get("oikomi") or 0),
        "h_registered_races_n": int(full_stats.get("registered_races") or 0),
        "j_year_flat_prize_total": int(j_prize.get("year_flat") or 0),
        "j_year_ob_prize_total": int(j_prize.get("year_obs") or 0),
        "j_cum_flat_prize_total": int(j_prize.get("cum_flat") or 0),
        "j_cum_ob_prize_total": int(j_prize.get("cum_obs") or 0),
        "t_year_flat_prize_total": int(t_prize.get("year_flat") or 0),
        "t_year_ob_prize_total": int(t_prize.get("year_obs") or 0),
        "t_cum_flat_prize_total": int(t_prize.get("cum_flat") or 0),
        "t_cum_ob_prize_total": int(t_prize.get("cum_obs") or 0),
        "o_year_prize_total": int(o_prize.get("year") or 0),
        "o_cum_prize_total": int(o_prize.get("cum") or 0),
        "b_year_prize_total": int(b_prize.get("year") or 0),
        "b_cum_prize_total": int(b_prize.get("cum") or 0),
    }
    return raw_payload, core_payload, feat_payload


def insert_ck_records_batch(
    db: Database,
    raw_payloads: list[dict],
    core_payloads: list[dict],
    feat_payloads: list[dict],
) -> int:
    """raw.jv_ck_event / core.ck_runner_event / mart.feat_ck_win にCKをバッチ挿入"""
    if not raw_payloads:
        return 0
    db.execute_many(CK_RAW_SQL, raw_payloads)
    db.execute_many(CK_CORE_SQL, core_payloads)
    db.execute_many(CK_FEAT_SQL, feat_payloads)
    return len(raw_payloads)


def insert_ck_record(db: Database, dataspec: str, record: CKRecord, filename: str) -> None:
    """互換用: 単一レコード挿入"""
    _ = filename
    payloads = _build_ck_payloads(dataspec, record)
    if payloads is None:
        return
    raw_payload, core_payload, feat_payload = payloads
    insert_ck_records_batch(db, [raw_payload], [core_payload], [feat_payload])


def process_file(db: Database, file_path: Path) -> dict[str, int]:
    """JSONLファイルを処理してDBに投入

    RAレコード（レース情報）を最初に処理してコミットすることで、
    後続のSE/HR/O1レコードでのFK違反（race_id参照エラー）を防ぐ。
    """
    stats = {
        "raw": 0,
        "race": 0,
        "runner": 0,
        "payout": 0,
        "odds": 0,
        "odds_parsed": 0,
        "odds_upserted": 0,
        "odds_missing_runner": 0,
        "odds_skipped_bracket": 0,
        "o1_deferred_records": 0,
        "horse": 0,
        "o1_ts": 0,
        "wh": 0,
        "training": 0,
        "mining": 0,
        "ck": 0,
        "ck_skipped_make_date": 0,
        "event": 0,
        "errors": 0,
    }
    BATCH_SIZE = 1000

    # ファイル名からdataspecを推定
    dataspec = file_path.stem.split("_")[0]
    skip_raw_jv_raw = dataspec in {"SNPN", "SNAP"}

    if dataspec == "0B41":
        logger.info("  0B41 fast path: Processing raw + O1 records in single pass...")
        raw_batch = []
        o1_records_batch: list[OddsTimeSeriesRecord] = []
        line_count = 0
        commit_interval = 10000
        o1_batch_size = 50000
        race_stub_cache: set[int] = set()

        def flush_o1_batch() -> None:
            nonlocal o1_records_batch
            if not o1_records_batch:
                return
            stats["o1_ts"] += upsert_o1_timeseries_bulk(
                db, o1_records_batch, race_stub_cache=race_stub_cache
            )
            o1_records_batch = []

        for record in load_jsonl(file_path):
            line_count += 1
            rec_id = record.get("rec_id", "")
            payload = record.get("payload", "")
            try:
                raw_batch.append(record)
                stats["raw"] += 1

                if rec_id == "O1":
                    ts_records = OddsTimeSeriesRecord.parse(payload)
                    o1_records_batch.extend(ts_records)
                    if len(o1_records_batch) >= o1_batch_size:
                        flush_o1_batch()

                if len(raw_batch) >= BATCH_SIZE:
                    insert_raw_records_batch(db, dataspec, raw_batch)
                    raw_batch = []

                if line_count % commit_interval == 0:
                    flush_o1_batch()
                    if raw_batch:
                        insert_raw_records_batch(db, dataspec, raw_batch)
                        raw_batch = []
                    db.connect().commit()
                    race_stub_cache = set()
                    logger.info(f"  {line_count} 件処理完了...")
            except Exception as e:
                db.connect().rollback()
                raw_batch = []
                o1_records_batch = []
                race_stub_cache = set()
                if stats["errors"] < 20:
                    logger.warning(f"0B41 Error [{rec_id}]: {e}")
                stats["errors"] += 1

        flush_o1_batch()
        if raw_batch:
            insert_raw_records_batch(db, dataspec, raw_batch)
        db.connect().commit()
        logger.info(
            "  0B41 fast path 完了: raw=%s o1_ts=%s errors=%s",
            f"{stats['raw']:,}",
            f"{stats['o1_ts']:,}",
            f"{stats['errors']:,}",
        )
        return stats

    # マスタデータをメモリにロード (FK検証用)
    master_jockeys, master_trainers = prepare_master_data_cache(db)

    if not skip_raw_jv_raw:
        # Pass 1: Race (RA) のみ処理 + rawバッチ挿入
        logger.info("  Pass 1: Processing Race records...")
        raw_batch = []
        for record in load_jsonl(file_path):
            try:
                raw_batch.append(record)
                stats["raw"] += 1

                rec_id = record["rec_id"]
                payload = record["payload"]

                if rec_id == "RA":
                    race = RaceRecord.parse(payload)
                    upsert_race(db, race)
                    stats["race"] += 1

                if len(raw_batch) >= BATCH_SIZE:
                    insert_raw_records_batch(db, dataspec, raw_batch)
                    raw_batch = []
                    db.connect().commit()
                    logger.info(f"    {stats['raw']:,} 件処理...")

            except Exception:
                db.connect().rollback()
                raw_batch = []
                stats["errors"] += 1

        if raw_batch:
            insert_raw_records_batch(db, dataspec, raw_batch)
        db.connect().commit()
        logger.info(f"  Pass 1 完了: {stats['raw']:,} raw, {stats['race']:,} race")
    else:
        logger.info("  Pass 1: skipped raw.jv_raw for CK dataspec (SNPN/SNAP)")

    # Pass 2: Skip RA, process others
    logger.info("  Pass 2: Processing other records...")
    # rawのカウントはPass 1で済ませているのでリセットしない
    # (insert_raw_recordがON CONFLICT DO NOTHINGなので再実行しても安全)
    # ただし stats["raw"] が2倍になるのを防ぐため、Pass 2では raw はスキップ

    pass2_count = 0
    pass2_commit_interval = 10000 if dataspec == "0B41" else 1000
    race_stub_cache: set[int] = set()
    deferred_o1_payloads: list[str] = []
    training_hc_batch: list = []
    training_wc_batch: list = []
    mining_dm_batch: list = []
    mining_tm_batch: list = []
    ck_raw_batch: list[dict] = []
    ck_core_batch: list[dict] = []
    ck_feat_batch: list[dict] = []

    def flush_training(force: bool = False) -> None:
        nonlocal training_hc_batch, training_wc_batch
        if training_hc_batch and (force or len(training_hc_batch) >= TRAINING_BATCH_SIZE):
            stats["training"] += insert_training_records_batch(db, "HC", training_hc_batch)
            training_hc_batch = []
        if training_wc_batch and (force or len(training_wc_batch) >= TRAINING_BATCH_SIZE):
            stats["training"] += insert_training_records_batch(db, "WC", training_wc_batch)
            training_wc_batch = []

    def flush_mining(force: bool = False) -> None:
        nonlocal mining_dm_batch, mining_tm_batch
        if mining_dm_batch and (force or len(mining_dm_batch) >= MINING_BATCH_SIZE):
            stats["mining"] += insert_mining_records_batch(db, "DM", mining_dm_batch)
            mining_dm_batch = []
        if mining_tm_batch and (force or len(mining_tm_batch) >= MINING_BATCH_SIZE):
            stats["mining"] += insert_mining_records_batch(db, "TM", mining_tm_batch)
            mining_tm_batch = []

    def flush_ck(force: bool = False) -> None:
        nonlocal ck_raw_batch, ck_core_batch, ck_feat_batch
        if ck_raw_batch and (force or len(ck_raw_batch) >= CK_BATCH_SIZE):
            stats["ck"] += insert_ck_records_batch(db, ck_raw_batch, ck_core_batch, ck_feat_batch)
            ck_raw_batch = []
            ck_core_batch = []
            ck_feat_batch = []

    def flush_all(force: bool = False) -> None:
        flush_training(force=force)
        flush_mining(force=force)
        flush_ck(force=force)

    for record in load_jsonl(file_path):
        pass2_count += 1
        rec_id = record.get("rec_id", "")
        payload = record.get("payload", "")
        try:
            # rawはPass 1で入れているのでスキップ

            if rec_id == "RA":
                continue  # 処理済み

            elif rec_id == "CK":
                ck = CKRecord.parse(payload)
                payloads = _build_ck_payloads(dataspec, ck)
                if payloads is None:
                    stats["ck_skipped_make_date"] += 1
                else:
                    raw_payload, core_payload, feat_payload = payloads
                    ck_raw_batch.append(raw_payload)
                    ck_core_batch.append(core_payload)
                    ck_feat_batch.append(feat_payload)
                    flush_ck()

            elif rec_id == "SE":
                runner = RunnerRecord.parse(payload)
                upsert_runner(db, runner, master_jockeys, master_trainers)
                stats["runner"] += 1

            elif rec_id == "HR":
                payouts = PayoutRecord.parse(payload)
                for p in payouts:
                    upsert_payout(db, p)
                    stats["payout"] += 1

            elif rec_id == "O1":
                # dataspec により投入先を分岐:
                # - 0B41: 時系列オッズ → core.o1_*
                # - それ以外: 最終オッズ → core.odds_final
                if dataspec == "0B41":
                    ts_records = OddsTimeSeriesRecord.parse(payload)
                    stats["o1_ts"] += upsert_o1_timeseries(
                        db, ts_records, race_stub_cache=race_stub_cache
                    )
                else:
                    # O1がSEより先に出現するファイルがあるため、runner投入後に遅延処理する
                    deferred_o1_payloads.append(payload)
                    stats["o1_deferred_records"] += 1

            elif rec_id == "JG":
                horse = HorseExclusionRecord.parse(payload)
                upsert_horse(db, horse)
                stats["horse"] += 1

            elif rec_id == "UM":
                horse = HorseMasterRecord.parse(payload)
                upsert_horse_master(db, horse)
                stats["horse"] += 1

            elif rec_id == "KS":
                jockey = JockeyRecord.parse(payload)
                upsert_jockey(db, jockey)
                if "jockey" not in stats:
                    stats["jockey"] = 0
                stats["jockey"] += 1

            elif rec_id == "CH":
                trainer = TrainerRecord.parse(payload)
                upsert_trainer(db, trainer)
                if "trainer" not in stats:
                    stats["trainer"] = 0
                stats["trainer"] += 1

            # --- 新規レコード ---
            elif rec_id == "WH":
                wh_list = WHRecord.parse(payload)
                stats["wh"] += upsert_wh_records(db, wh_list)

            elif rec_id in ("HC", "WC"):
                if rec_id == "HC":
                    training = HCRecord.parse(payload)
                    training_hc_batch.append(training)
                else:
                    training = WCRecord.parse(payload)
                    training_wc_batch.append(training)
                flush_training()

            elif rec_id in ("DM", "TM"):
                if rec_id == "DM":
                    mining_list = DMRecord.parse(payload)
                    mining_dm_batch.extend(mining_list)
                else:
                    mining_list = TMRecord.parse(payload)
                    mining_tm_batch.extend(mining_list)
                flush_mining()

            elif rec_id in ("WE", "AV", "JC", "TC", "CC"):
                event = EventChangeRecord.parse(payload)
                insert_event_change(db, event)
                stats["event"] += 1

        except Exception as e:
            # ロールバックしてトランザクションを復旧
            db.connect().rollback()
            training_hc_batch = []
            training_wc_batch = []
            mining_dm_batch = []
            mining_tm_batch = []
            ck_raw_batch = []
            ck_core_batch = []
            ck_feat_batch = []
            race_stub_cache = set()
            if stats["errors"] < 20:
                logger.warning(f"Pass 2 Error [{rec_id}]: {e}")
                # logger.warning(traceback.format_exc())
            stats["errors"] += 1

        if pass2_count % pass2_commit_interval == 0:
            flush_all(force=True)
            db.connect().commit()
            race_stub_cache = set()
            logger.info(f"  {pass2_count} 件スキャン完了...")

    flush_all(force=True)

    if deferred_o1_payloads:
        logger.info(
            f"  Pass 2.5: Processing deferred O1 records... {len(deferred_o1_payloads):,} 件"
        )
        missing_log_limit = 20
        missing_log_count = 0

        for idx, payload in enumerate(deferred_o1_payloads, 1):
            try:
                odds_list = OddsRecord.parse(payload)
                stats["odds_parsed"] += len(odds_list)

                for o in odds_list:
                    if o.bet_type == 3:
                        stats["odds_skipped_bracket"] += 1
                        continue
                    if o.bet_type not in (1, 2):
                        continue

                    affected = upsert_odds(db, o)
                    stats["odds"] += affected
                    stats["odds_upserted"] += affected

                    if affected == 0:
                        stats["odds_missing_runner"] += 1
                        if missing_log_count < missing_log_limit:
                            logger.warning(
                                "O1 odds skipped (runner not found): race_id=%s "
                                "horse_no=%s bet_type=%s",
                                o.race_id,
                                o.horse_no,
                                o.bet_type,
                            )
                            missing_log_count += 1
            except Exception as e:
                db.connect().rollback()
                if stats["errors"] < 20:
                    logger.warning(f"Deferred O1 Error: {e}")
                stats["errors"] += 1

            if idx % 500 == 0:
                db.connect().commit()
                logger.info(f"  deferred O1: {idx:,}/{len(deferred_o1_payloads):,} 件")

        logger.info(
            "  deferred O1 summary: parsed=%s upserted=%s missing_runner=%s skipped_bracket=%s",
            stats["odds_parsed"],
            stats["odds_upserted"],
            stats["odds_missing_runner"],
            stats["odds_skipped_bracket"],
        )

    db.connect().commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description="JSONL → PostgreSQL ロード")
    parser.add_argument(
        "--input",
        type=str,
        help="入力JSONLファイル (ワイルドカード対応: data/RACE_*.jsonl)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="入力ディレクトリ (全JSONLファイルを処理)",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("--input または --input-dir を指定してください")

    files = []
    if args.input:
        # ワイルドカードサポート
        input_path = Path(args.input)
        if "*" in args.input:
            # glob展開
            parent = input_path.parent if input_path.parent.exists() else Path(".")
            pattern = input_path.name
            files.extend(sorted(parent.glob(pattern)))
        else:
            files.append(input_path)
    if args.input_dir:
        files.extend(sorted(args.input_dir.glob("*.jsonl")))

    if not files:
        print("❌ 処理対象ファイルがありません")
        sys.exit(1)

    print("=" * 60)
    print("JSONL → PostgreSQL ロード")
    print("=" * 60)
    print(f"対象ファイル: {len(files)} 件")
    print("=" * 60)

    total_stats: dict[str, int] = {}

    with Database() as db:
        for file_path in files:
            logger.info(f"📂 処理中: {file_path.name}")
            stats = process_file(db, file_path)
            for key, val in stats.items():
                total_stats[key] = total_stats.get(key, 0) + val
            logger.info(f"  → {stats}")

    print("")
    print("=" * 60)
    print("✅ 完了")
    for key, val in total_stats.items():
        if val > 0:
            print(f"   {key}: {val:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
