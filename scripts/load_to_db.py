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


def load_jsonl(file_path: Path):
    """JSONLファイルを読み込み"""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def insert_raw_record(db: Database, dataspec: str, record: dict) -> None:
    """raw.jv_rawにレコードを挿入（単一レコード版）"""
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    db.execute(sql, (dataspec, record["rec_id"], record["filename"], record["payload"]))


def insert_raw_records_batch(db: Database, dataspec: str, records: list[dict]) -> int:
    """raw.jv_rawにレコードをバッチ挿入（高速版）"""
    if not records:
        return 0
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    values = [(dataspec, r["rec_id"], r["filename"], r["payload"]) for r in records]
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

    # race_idから最低限の情報を復元して入れたいところだが、とりあえずrace_idのみ試みる。
    # スキーマのNOT NULL制約に引っかかる場合はデフォルト値を入れる必要がある。
    # ここでは一般的な必須項目（日付、場、回、R）をrace_idから分解して入れたほうが安全。
    # parsers.pyのロジック: race_id = date_int * 10000 + track * 100 + race_no
    # つまり YYYYMMDD + TT + RR = YYYYMMDDTTRR

    try:
        r_id = runner.race_id
        race_no = r_id % 100
        track_code = (r_id // 100) % 100
        date_int = int(r_id // 10000)
        y = date_int // 10000
        m = (date_int // 100) % 100
        d = date_int % 100
        race_date = f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        race_no = 0
        track_code = 0
        race_date = "2000-01-01"

    # 簡易的に race_id のみでInsertしようとするとNOT NULL制約等で失敗し、
    # その際のRollbackで直前のHorse登録まで消えてしまうリスクがあるため、
    # 最初からデフォルト値付きの完全なSQLで実行する。
    race_sql = """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface, distance_m, 
            going, weather, class_code, field_size
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, 0, 0, 0, 0, 0, 0
        ) 
        ON CONFLICT (race_id) DO NOTHING
    """
    db.execute(
        race_sql,
        {
            "race_id": runner.race_id,
            "race_date": race_date,
            "track_code": track_code,
            "race_no": race_no,
        },
    )

    # runner (出走表)
    runner_sql = """
        INSERT INTO core.runner (
            race_id, horse_id, horse_no, gate, jockey_id,
            trainer_id, carried_weight, body_weight, body_weight_diff,
            data_kubun, trainer_code_raw, trainer_name_abbr,
            jockey_code_raw, jockey_name_abbr
        ) VALUES (
            %(race_id)s, %(horse_id)s, %(horse_no)s, %(gate)s, %(jockey_id)s,
            %(trainer_id)s, %(carried_weight)s, %(body_weight)s, %(body_weight_diff)s,
            %(data_kubun)s, %(trainer_code_raw)s, %(trainer_name_abbr)s,
            %(jockey_code_raw)s, %(jockey_name_abbr)s
        )
        ON CONFLICT (race_id, horse_id) DO UPDATE SET
            horse_no = EXCLUDED.horse_no,
            gate = EXCLUDED.gate,
            jockey_id = EXCLUDED.jockey_id,
            trainer_id = EXCLUDED.trainer_id,
            carried_weight = EXCLUDED.carried_weight,
            body_weight = EXCLUDED.body_weight,
            body_weight_diff = EXCLUDED.body_weight_diff,
            data_kubun = EXCLUDED.data_kubun,
            trainer_code_raw = EXCLUDED.trainer_code_raw,
            trainer_name_abbr = EXCLUDED.trainer_name_abbr,
            jockey_code_raw = EXCLUDED.jockey_code_raw,
            jockey_name_abbr = EXCLUDED.jockey_name_abbr,
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
            "data_kubun": runner.data_kubun,
            "trainer_code_raw": runner.trainer_code_raw,
            "trainer_name_abbr": runner.trainer_name_abbr,
            "jockey_code_raw": runner.jockey_code_raw,
            "jockey_name_abbr": runner.jockey_name_abbr,
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


def upsert_odds(db: Database, odds: OddsRecord) -> None:
    """core.odds_finalにデータを挿入/更新

    Note: O1レコードにはhorse_idがないため、core.runnerからhorse_noで引いて解決する
    bet_type=1 (Win), bet_type=2 (Place) のみ処理
    """
    if odds.bet_type == 1:  # Win
        sql = """
            INSERT INTO core.odds_final (race_id, horse_id, odds_win, pop_win)
            SELECT %(race_id)s, horse_id, %(odds_win)s, %(pop_win)s
            FROM core.runner
            WHERE race_id = %(race_id)s AND horse_no = %(horse_no)s
            ON CONFLICT (race_id, horse_id) DO UPDATE SET
                odds_win = EXCLUDED.odds_win,
                pop_win = EXCLUDED.pop_win,
                updated_at = now()
        """
        db.execute(
            sql,
            {
                "race_id": odds.race_id,
                "horse_no": int(odds.horse_no) if isinstance(odds.horse_no, str) else odds.horse_no,
                "odds_win": odds.odds_1,
                "pop_win": odds.popularity,
            },
        )
    elif odds.bet_type == 2:  # Place
        sql = """
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
        """
        db.execute(
            sql,
            {
                "race_id": odds.race_id,
                "horse_no": int(odds.horse_no) if isinstance(odds.horse_no, str) else odds.horse_no,
                "odds_place_low": odds.odds_1,
                "odds_place_high": odds.odds_2,
                "pop_place": odds.popularity,
            },
        )
    # bet_type=3 (Bracket) は現状スキップ（horse_idへの変換が困難）


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


def upsert_o1_timeseries(db: Database, records: list[OddsTimeSeriesRecord]) -> int:
    """core.o1_header / core.o1_win に時系列オッズを挿入/更新"""
    if not records:
        return 0

    # ヘッダー情報は最初のレコードから取得
    first = records[0]
    header_sql = """
        INSERT INTO core.o1_header (race_id, data_kbn, announce_mmddhhmi, win_pool_total_100yen)
        VALUES (%(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s, %(win_pool_total_100yen)s)
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
            win_pool_total_100yen = EXCLUDED.win_pool_total_100yen
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

    count = 0
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
    for r in records:
        db.execute(
            detail_sql,
            {
                "race_id": r.race_id,
                "data_kbn": r.data_kbn,
                "announce_mmddhhmi": r.announce_mmddhhmi,
                "horse_no": r.horse_no,
                "win_odds_x10": r.win_odds_x10,
                "win_popularity": r.win_popularity,
            },
        )
        count += 1

    return count


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


def insert_training_record(db: Database, rec_id: str, record) -> None:
    """core.training_slop / core.training_wood に調教データを挿入"""
    if rec_id == "HC":
        sql = """
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
        db.execute(
            sql,
            {
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
            },
        )
    else:  # WC
        sql = """
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
        db.execute(
            sql,
            {
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
            },
        )


def insert_mining_record(db: Database, rec_id: str, record) -> None:
    """core.mining_dm / core.mining_tm にマイニングデータを挿入"""
    if rec_id == "DM":
        sql = """
            INSERT INTO core.mining_dm (race_id, horse_no, data_kbn, payload_raw)
            VALUES (%(race_id)s, %(horse_no)s, %(data_kbn)s, %(payload_raw)s)
            ON CONFLICT (race_id, horse_no) DO UPDATE SET
                data_kbn = EXCLUDED.data_kbn,
                payload_raw = EXCLUDED.payload_raw
        """
    else:  # TM
        sql = """
            INSERT INTO core.mining_tm (race_id, horse_no, data_kbn, payload_raw)
            VALUES (%(race_id)s, %(horse_no)s, %(data_kbn)s, %(payload_raw)s)
            ON CONFLICT (race_id, horse_no) DO UPDATE SET
                data_kbn = EXCLUDED.data_kbn,
                payload_raw = EXCLUDED.payload_raw
        """
    db.execute(
        sql,
        {
            "race_id": record.race_id,
            "horse_no": record.horse_no,
            "data_kbn": record.data_kbn,
            "payload_raw": record.payload_raw,
        },
    )


def insert_event_change(db: Database, record: EventChangeRecord) -> None:
    """core.event_change に当日変更を挿入"""
    sql = """
        INSERT INTO core.event_change (race_id, record_type, data_create_ymd, payload_parsed)
        VALUES (%(race_id)s, %(record_type)s, NULL, %(payload)s)
    """
    import json as _json

    db.execute(
        sql,
        {
            "race_id": record.race_id,
            "record_type": record.record_type,
            "payload": _json.dumps({"raw": record.payload_raw}, ensure_ascii=False),
        },
    )


def insert_ck_record(db: Database, dataspec: str, record: CKRecord, filename: str) -> None:
    """raw.jv_ck_event / core.ck_runner_event にCKデータを挿入"""
    import json as _json

    # SHA256 calc
    payload_bytes = record.payload_raw.encode("cp932", errors="replace")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()

    # RAW Insert
    raw_sql = """
        INSERT INTO raw.jv_ck_event (
            dataspec, record_type, data_kbn, data_create_ymd,
            kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi,
            race_no, horse_id, horse_name, payload, payload_sha256
        ) VALUES (
            %(dataspec)s, 'CK', 0, %(data_create_ymd)s,
            %(year)s, %(md)s, %(track)s, %(kai)s, %(nichi)s,
            %(race_no)s, %(horse_id)s, %(horse_name)s, %(payload)s, %(sha256)s
        )
        ON CONFLICT (
            dataspec, data_create_ymd, kaisai_year, kaisai_md, 
            track_cd, kaisai_kai, kaisai_nichi, race_no, 
            horse_id, payload_sha256
        )
        DO NOTHING
    """

    # We need to extract parts from RaceID or Record fields?
    # CKRecord stores explicit fields now.
    # make_date is CK_MAKE_DATE (Data Create Date? or Race Date?)
    # Spec: "3 〇 データ作成年月日 4 8" -> CK_MAKE_DATE.
    # So record.make_date IS data_create_ymd.

    # Casting to int for DB
    try:
        # year = int(record.race_id[0:4])  # record.race_id is simplified YYYYMMDD...
        # Wait, CKRecord.race_id construction: `f"{year}{md}{course}{race_no}"`
        # But we have `record.kaisai_kai` etc.
        # Let's use `record.make_date` for data_create_ymd.
        # And parse int from the other fields.

        # We need raw string values for year/md/track/kai/nichi/raceno
        # CKRecord now has them!
        # But race_id is constructed.
        # Let's re-parse from raw if needed or use what we stored.
        # CKRecord has `kaisai_kai`, `kaisai_nichi`, `race_no`, `jockey_code` etc.
        # But `year`, `md`, `course` were local vars in `parse`.
        # I didn't store `year` in CKRecord! I stored `race_id`.
        # I NEED `year`, `md`, `course` (track) in CKRecord or parse them again.

        # I'll rely on `record.race_id` string structure I defined:
        # `f"{year}{md}{course}{race_no}"`
        # Year: 0:4
        # MD: 4:8
        # Course: 8:10
        # RaceNo: 10:12

        r_year = int(record.race_id[0:4])
        r_md_str = record.race_id[4:8]
        r_track_str = record.race_id[8:10]
        # Kai/Nichi are stored in record.
        r_kai = int(record.kaisai_kai)
        r_nichi = int(record.kaisai_nichi)
        r_no = int(record.race_no)

        # data_create_ymd
        # record.make_date is `date` object or None.
        if record.make_date:
            dc_ymd = record.make_date
        else:
            dc_ymd = "2000-01-01"  # Fallback

    except ValueError:
        # If parsing fails, skip or log?
        # logger.warning(f"CK Parse Int Error: {record.race_id}")
        return

    db.execute(
        raw_sql,
        {
            "dataspec": dataspec,
            "data_create_ymd": dc_ymd,
            "year": r_year,
            "md": r_md_str,
            "track": r_track_str,
            "kai": r_kai,
            "nichi": r_nichi,
            "race_no": r_no,
            "horse_id": record.horse_id,
            "horse_name": record.horse_name,
            "payload": record.payload_raw.encode("cp932", errors="replace"),
            "sha256": payload_sha256,
        },
    )

    # Core Insert
    full_stats = record.get_full_stats()
    core_sql = """
        INSERT INTO core.ck_runner_event (
            dataspec, data_create_ymd,
            kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi,
            race_no, horse_id, horse_name,
            finish_counts, style_counts, registered_races_n,
            jockey_cd, trainer_cd, owner_cd, breeder_cd,
            entity_prize
        ) VALUES (
            %(dataspec)s, %(dc_ymd)s,
            %(year)s, %(md)s, %(track)s, %(kai)s, %(nichi)s,
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
        DO NOTHING
    """

    db.execute(
        core_sql,
        {
            "dataspec": dataspec,
            "dc_ymd": dc_ymd,
            "year": r_year,
            "md": r_md_str,
            "track": r_track_str,
            "kai": r_kai,
            "nichi": r_nichi,
            "race_no": r_no,
            "horse_id": record.horse_id,
            "horse_name": record.horse_name,
            "finish_counts": _json.dumps(full_stats["finish_counts"]),
            "style_counts": _json.dumps(full_stats["style_counts"]),
            "reg_races": full_stats["registered_races"],
            "jockey_cd": record.jockey_code,
            "trainer_cd": record.trainer_code,
            "owner_cd": record.owner_cd,
            "breeder_cd": record.breeder_code,
            "entity_prize": _json.dumps(full_stats["entity_prize"]),
        },
    )


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
        "horse": 0,
        "o1_ts": 0,
        "wh": 0,
        "training": 0,
        "mining": 0,
        "ck": 0,
        "event": 0,
        "errors": 0,
    }
    BATCH_SIZE = 1000

    # マスタデータをメモリにロード (FK検証用)
    master_jockeys, master_trainers = prepare_master_data_cache(db)

    # ファイル名からdataspecを推定
    dataspec = file_path.stem.split("_")[0]

    # Pass 1: Race (RA) のみ処理 + rawバッチ挿入
    logger.info("  Pass 1: Processing Race records...")
    raw_batch = []
    for record in load_jsonl(file_path):
        try:
            # rawレコードをバッチに追加
            raw_batch.append(record)
            stats["raw"] += 1

            rec_id = record["rec_id"]
            payload = record["payload"]

            if rec_id == "RA":
                race = RaceRecord.parse(payload)
                upsert_race(db, race)
                stats["race"] += 1

            # バッチサイズに達したらINSERT
            if len(raw_batch) >= BATCH_SIZE:
                insert_raw_records_batch(db, dataspec, raw_batch)
                raw_batch = []
                db.connect().commit()
                logger.info(f"    {stats['raw']:,} 件処理...")

        except Exception:
            db.connect().rollback()
            raw_batch = []  # バッチをクリア
            if stats["errors"] < 5:
                pass
            stats["errors"] += 1

    # 残りのバッチを処理
    if raw_batch:
        insert_raw_records_batch(db, dataspec, raw_batch)
    db.connect().commit()  # Pass 1完了コミット
    logger.info(f"  Pass 1 完了: {stats['raw']:,} raw, {stats['race']:,} race")

    # Pass 2: Skip RA, process others
    logger.info("  Pass 2: Processing other records...")
    # rawのカウントはPass 1で済ませているのでリセットしない
    # (insert_raw_recordがON CONFLICT DO NOTHINGなので再実行しても安全)
    # ただし stats["raw"] が2倍になるのを防ぐため、Pass 2では raw はスキップ

    pass2_count = 0
    for record in load_jsonl(file_path):
        pass2_count += 1
        try:
            # rawはPass 1で入れているのでスキップ

            rec_id = record["rec_id"]
            payload = record["payload"]

            if rec_id == "RA":
                continue  # 処理済み

            elif rec_id == "CK":
                ck = CKRecord.parse(payload)
                insert_ck_record(db, dataspec, ck, record["filename"])
                stats["ck"] += 1

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
                odds_list = OddsRecord.parse(payload)
                for o in odds_list:
                    upsert_odds(db, o)
                    stats["odds"] += 1

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
                else:
                    training = WCRecord.parse(payload)
                insert_training_record(db, rec_id, training)
                stats["training"] += 1

            elif rec_id == "CK":
                CKRecord.parse(payload)  # パース検証のみ
                # CKは簡易版のため payload_raw で保存
                stats.setdefault("ck", 0)
                stats["ck"] += 1

            elif rec_id in ("DM", "TM"):
                if rec_id == "DM":
                    mining = DMRecord.parse(payload)
                else:
                    mining = TMRecord.parse(payload)
                insert_mining_record(db, rec_id, mining)
                stats["mining"] += 1

            elif rec_id in ("WE", "AV", "JC", "TC", "CC"):
                event = EventChangeRecord.parse(payload)
                insert_event_change(db, event)
                stats["event"] += 1

        except Exception as e:
            # ロールバックしてトランザクションを復旧
            db.connect().rollback()
            if stats["errors"] < 20:
                logger.warning(f"Pass 2 Error [{rec_id}]: {e}")
                # logger.warning(traceback.format_exc())
            stats["errors"] += 1

        if pass2_count % 1000 == 0:
            db.connect().commit()
            logger.info(f"  {pass2_count} 件スキャン完了...")

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
