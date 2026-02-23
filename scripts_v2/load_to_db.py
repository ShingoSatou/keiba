"""
JSONL -> PostgreSQL ロードスクリプト（v2）

対象:
- RACE: RA / SE / HR / O1 / O3
- DIFF: UM / KS / CH
- MING: DM / TM
- 0B41: O1（時系列）
- 0B11: WH
- 0B14: WE / AV / JC / TC / CC
- 0B13 / 0B17: DM / TM（速報時系列）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from app.infrastructure.parsers import (  # noqa: E402
    DMRecord,
    EventChangeRecord,
    HorseMasterRecord,
    JockeyRecord,
    O3HeaderRecord,
    O3WideRecord,
    OddsTimeSeriesRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
    TMRecord,
    TrainerRecord,
    WHRecord,
)

logger = logging.getLogger(__name__)


def calc_payload_hash(payload: str) -> bytes:
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).digest()


def race_id_track_code(race_id: int) -> int:
    return (int(race_id) // 100) % 100


def is_central_track(track_code: int) -> bool:
    return 1 <= int(track_code) <= 10


def is_central_race(race_id: int) -> bool:
    return is_central_track(race_id_track_code(race_id))


def ensure_race_stub(db: Database, race_id: int, cache: set[int] | None = None) -> None:
    if not race_id:
        return
    race_id_int = int(race_id)
    if race_id_int <= 0:
        return
    if cache is not None and race_id_int in cache:
        return

    race_no = race_id_int % 100
    track_code = (race_id_int // 100) % 100
    date_int = race_id_int // 10000
    year = date_int // 10000
    month = (date_int // 100) % 100
    day = date_int % 100
    try:
        race_date = date(year, month, day)
    except ValueError:
        race_date = date(2000, 1, 1)

    db.execute(
        """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface, distance_m
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, 0, 0
        )
        ON CONFLICT (race_id) DO NOTHING
        """,
        {
            "race_id": race_id_int,
            "race_date": race_date,
            "track_code": track_code,
            "race_no": race_no,
        },
    )
    if cache is not None:
        cache.add(race_id_int)


def load_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def insert_raw_records_batch(db: Database, dataspec: str, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload, payload_hash)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (dataspec, rec_id, payload_hash) DO NOTHING
    """
    values = [
        (
            dataspec,
            record.get("rec_id"),
            record.get("filename"),
            record.get("payload", ""),
            calc_payload_hash(record.get("payload", "")),
        )
        for record in records
    ]
    db.execute_many(sql, values)
    return len(values)


def upsert_race(db: Database, race: RaceRecord) -> None:
    db.execute(
        """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface,
            distance_m, going, weather, class_code, field_size, start_time,
            turn_dir, course_inout, grade_code, race_type_code,
            weight_type_code, condition_code_min_age
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, %(surface)s,
            %(distance_m)s, %(going)s, %(weather)s,
            %(class_code)s, %(field_size)s, %(start_time)s,
            %(turn_dir)s, %(course_inout)s, %(grade_code)s, %(race_type_code)s,
            %(weight_type_code)s, %(condition_code_min_age)s
        )
        ON CONFLICT (race_id) DO UPDATE SET
            surface = CASE
                WHEN EXCLUDED.surface > 0
                    AND (
                        core.race.surface = 0
                        OR core.race.surface IS NULL
                        OR core.race.distance_m = 0
                        OR core.race.distance_m IS NULL
                        OR core.race.start_time IS NULL
                    )
                THEN EXCLUDED.surface
                ELSE core.race.surface
            END,
            distance_m = CASE
                WHEN EXCLUDED.distance_m > 0
                    AND (
                        core.race.distance_m = 0
                        OR core.race.distance_m IS NULL
                        OR core.race.distance_m < 800
                    )
                THEN EXCLUDED.distance_m
                ELSE core.race.distance_m
            END,
            going = COALESCE(EXCLUDED.going, NULLIF(core.race.going, 0)),
            weather = COALESCE(EXCLUDED.weather, NULLIF(core.race.weather, 0)),
            class_code = COALESCE(
                NULLIF(EXCLUDED.class_code, 0),
                NULLIF(core.race.class_code, 0),
                core.race.class_code,
                0
            ),
            field_size = COALESCE(EXCLUDED.field_size, NULLIF(core.race.field_size, 0)),
            start_time = COALESCE(EXCLUDED.start_time, core.race.start_time),
            turn_dir = COALESCE(core.race.turn_dir, EXCLUDED.turn_dir),
            course_inout = COALESCE(
                NULLIF(core.race.course_inout, 0),
                NULLIF(EXCLUDED.course_inout, 0),
                core.race.course_inout
            ),
            grade_code = COALESCE(core.race.grade_code, EXCLUDED.grade_code),
            race_type_code = COALESCE(
                NULLIF(EXCLUDED.race_type_code, 0),
                NULLIF(core.race.race_type_code, 0),
                core.race.race_type_code
            ),
            weight_type_code = COALESCE(
                NULLIF(EXCLUDED.weight_type_code, 0),
                NULLIF(core.race.weight_type_code, 0),
                core.race.weight_type_code
            ),
            condition_code_min_age = COALESCE(
                NULLIF(EXCLUDED.condition_code_min_age, 0),
                NULLIF(core.race.condition_code_min_age, 0),
                core.race.condition_code_min_age
            ),
            updated_at = now()
        """,
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
            "turn_dir": race.turn_dir,
            "course_inout": race.course_inout,
            "grade_code": race.grade_code,
            "race_type_code": race.race_type_code,
            "weight_type_code": race.weight_type_code,
            "condition_code_min_age": race.condition_code_min_age,
        },
    )


def prepare_master_data_cache(db: Database) -> tuple[set[int], set[int]]:
    jockeys = {int(row["jockey_id"]) for row in db.fetch_all("SELECT jockey_id FROM core.jockey")}
    trainers = {
        int(row["trainer_id"]) for row in db.fetch_all("SELECT trainer_id FROM core.trainer")
    }
    return jockeys, trainers


def upsert_runner(
    db: Database,
    runner: RunnerRecord,
    master_jockeys: set[int],
    master_trainers: set[int],
    race_stub_cache: set[int],
) -> None:
    safe_jockey_id = runner.jockey_id if runner.jockey_id in master_jockeys else None
    safe_trainer_id = runner.trainer_id if runner.trainer_id in master_trainers else None

    if runner.data_kubun in ("A", "B2"):
        safe_jockey_id = None
        safe_trainer_id = None

    db.execute(
        """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = COALESCE(EXCLUDED.horse_name, core.horse.horse_name),
            updated_at = now()
        """,
        {"horse_id": runner.horse_id, "horse_name": runner.horse_name},
    )

    ensure_race_stub(db, runner.race_id, cache=race_stub_cache)

    db.execute(
        """
        INSERT INTO core.runner (
            race_id, horse_id, horse_no, gate, jockey_id,
            trainer_id, carried_weight, body_weight, body_weight_diff,
            sex, data_kubun, trainer_code_raw, trainer_name_abbr,
            jockey_code_raw, jockey_name_abbr
        ) VALUES (
            %(race_id)s,
            COALESCE(
                (
                    SELECT r.horse_id
                    FROM core.runner AS r
                    WHERE r.race_id = %(race_id)s
                      AND r.horse_no = %(horse_no)s
                ),
                %(horse_id)s
            ),
            %(horse_no)s, %(gate)s, %(jockey_id)s,
            %(trainer_id)s, %(carried_weight)s, %(body_weight)s, %(body_weight_diff)s,
            %(sex)s, %(data_kubun)s, %(trainer_code_raw)s, %(trainer_name_abbr)s,
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
            sex = COALESCE(EXCLUDED.sex, core.runner.sex),
            data_kubun = EXCLUDED.data_kubun,
            trainer_code_raw = EXCLUDED.trainer_code_raw,
            trainer_name_abbr = EXCLUDED.trainer_name_abbr,
            jockey_code_raw = EXCLUDED.jockey_code_raw,
            jockey_name_abbr = EXCLUDED.jockey_name_abbr,
            updated_at = now()
        """,
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
            "sex": runner.sex,
            "data_kubun": runner.data_kubun,
            "trainer_code_raw": runner.trainer_code_raw,
            "trainer_name_abbr": runner.trainer_name_abbr,
            "jockey_code_raw": runner.jockey_code_raw,
            "jockey_name_abbr": runner.jockey_name_abbr,
        },
    )

    if runner.finish_pos and runner.finish_pos > 0:
        db.execute(
            """
            INSERT INTO core.result (
                race_id, horse_id, finish_pos, time_sec,
                margin, final3f_sec, corner1_pos, corner2_pos, corner3_pos, corner4_pos
            ) VALUES (
                %(race_id)s,
                COALESCE(
                    (
                        SELECT r.horse_id
                        FROM core.runner AS r
                        WHERE r.race_id = %(race_id)s
                          AND r.horse_no = %(horse_no)s
                    ),
                    %(horse_id)s
                ),
                %(finish_pos)s, %(time_sec)s,
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
            """,
            {
                "race_id": runner.race_id,
                "horse_id": runner.horse_id,
                "horse_no": runner.horse_no,
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


def upsert_payout_records(
    db: Database, payouts: list[PayoutRecord], race_stub_cache: set[int]
) -> int:
    if not payouts:
        return 0
    race_id = payouts[0].race_id
    ensure_race_stub(db, race_id, cache=race_stub_cache)
    sql = """
        INSERT INTO core.payout (race_id, bet_type, selection, payout_yen, popularity)
        VALUES (%(race_id)s, %(bet_type)s, %(selection)s, %(payout_yen)s, %(popularity)s)
        ON CONFLICT (race_id, bet_type, selection) DO UPDATE SET
            payout_yen = EXCLUDED.payout_yen,
            popularity = EXCLUDED.popularity
    """
    rows = [
        {
            "race_id": payout.race_id,
            "bet_type": payout.bet_type,
            "selection": payout.selection,
            "payout_yen": payout.payout_yen,
            "popularity": payout.popularity,
        }
        for payout in payouts
    ]
    db.execute_many(sql, rows)
    return len(rows)


def upsert_horse_master(db: Database, horse: HorseMasterRecord) -> None:
    db.execute(
        """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = COALESCE(EXCLUDED.horse_name, core.horse.horse_name),
            updated_at = now()
        """,
        {"horse_id": horse.horse_id, "horse_name": horse.horse_name},
    )


def upsert_jockey(db: Database, jockey: JockeyRecord) -> None:
    db.execute(
        """
        INSERT INTO core.jockey (jockey_id, jockey_name)
        VALUES (%(jockey_id)s, %(jockey_name)s)
        ON CONFLICT (jockey_id) DO UPDATE SET
            jockey_name = COALESCE(EXCLUDED.jockey_name, core.jockey.jockey_name),
            updated_at = now()
        """,
        {"jockey_id": jockey.jockey_id, "jockey_name": jockey.jockey_name},
    )


def upsert_trainer(db: Database, trainer: TrainerRecord) -> None:
    db.execute(
        """
        INSERT INTO core.trainer (trainer_id, trainer_name)
        VALUES (%(trainer_id)s, %(trainer_name)s)
        ON CONFLICT (trainer_id) DO UPDATE SET
            trainer_name = COALESCE(EXCLUDED.trainer_name, core.trainer.trainer_name),
            updated_at = now()
        """,
        {"trainer_id": trainer.trainer_id, "trainer_name": trainer.trainer_name},
    )


def upsert_o1_timeseries_bulk(
    db: Database, records: list[OddsTimeSeriesRecord], race_stub_cache: set[int]
) -> int:
    if not records:
        return 0

    header_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record.race_id, record.data_kbn, record.announce_mmddhhmi)
        header_map[key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "win_pool_total_100yen": record.win_pool_total_100yen,
        }

    for row in header_map.values():
        ensure_race_stub(db, int(row["race_id"]), cache=race_stub_cache)

    db.execute_many(
        """
        INSERT INTO core.o1_header (race_id, data_kbn, announce_mmddhhmi, win_pool_total_100yen)
        VALUES (%(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s, %(win_pool_total_100yen)s)
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
            win_pool_total_100yen = EXCLUDED.win_pool_total_100yen
        """,
        list(header_map.values()),
    )

    detail_map: dict[tuple[int, int, str, int], dict[str, Any]] = {}
    for record in records:
        detail_key = (
            record.race_id,
            record.data_kbn,
            record.announce_mmddhhmi,
            record.horse_no,
        )
        detail_map[detail_key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "horse_no": record.horse_no,
            "win_odds_x10": record.win_odds_x10,
            "win_popularity": record.win_popularity,
        }
    detail_rows = list(detail_map.values())

    db.execute(
        """
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
        """,
        {"rows_json": json.dumps(detail_rows, ensure_ascii=False)},
    )
    return len(detail_rows)


def upsert_o3_wide_records_bulk(
    db: Database, records: list[O3WideRecord], race_stub_cache: set[int]
) -> int:
    if not records:
        return 0

    header_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record.race_id, record.data_kbn, record.announce_mmddhhmi)
        header_map[key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "wide_pool_total_100yen": record.wide_pool_total_100yen,
            "starters": record.starters,
            "sale_flag_wide": record.sale_flag_wide,
            "data_create_ymd": record.data_create_ymd,
        }

    for row in header_map.values():
        ensure_race_stub(db, int(row["race_id"]), cache=race_stub_cache)

    db.execute_many(
        """
        INSERT INTO core.o3_header (
            race_id, data_kbn, announce_mmddhhmi,
            wide_pool_total_100yen, starters, sale_flag_wide, data_create_ymd
        ) VALUES (
            %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
            %(wide_pool_total_100yen)s, %(starters)s, %(sale_flag_wide)s, %(data_create_ymd)s
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
            wide_pool_total_100yen = EXCLUDED.wide_pool_total_100yen,
            starters = EXCLUDED.starters,
            sale_flag_wide = EXCLUDED.sale_flag_wide,
            data_create_ymd = EXCLUDED.data_create_ymd
        """,
        list(header_map.values()),
    )

    detail_map: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for record in records:
        detail_key = (
            record.race_id,
            record.data_kbn,
            record.announce_mmddhhmi,
            record.kumiban,
        )
        detail_map[detail_key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "kumiban": record.kumiban,
            "min_odds_x10": record.min_odds_x10,
            "max_odds_x10": record.max_odds_x10,
            "popularity": record.popularity,
        }
    detail_rows = list(detail_map.values())

    db.execute(
        """
        INSERT INTO core.o3_wide (
            race_id, data_kbn, announce_mmddhhmi,
            kumiban, min_odds_x10, max_odds_x10, popularity
        )
        SELECT
            x.race_id, x.data_kbn, x.announce_mmddhhmi,
            x.kumiban, x.min_odds_x10, x.max_odds_x10, x.popularity
        FROM jsonb_to_recordset(%(rows_json)s::jsonb) AS x(
            race_id bigint,
            data_kbn integer,
            announce_mmddhhmi text,
            kumiban text,
            min_odds_x10 integer,
            max_odds_x10 integer,
            popularity integer
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, kumiban) DO UPDATE SET
            min_odds_x10 = EXCLUDED.min_odds_x10,
            max_odds_x10 = EXCLUDED.max_odds_x10,
            popularity = EXCLUDED.popularity
        """,
        {"rows_json": json.dumps(detail_rows, ensure_ascii=False)},
    )
    return len(detail_rows)


def upsert_wh_records_bulk(db: Database, records: list[WHRecord], race_stub_cache: set[int]) -> int:
    if not records:
        return 0

    header_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record.race_id, record.data_kbn, record.announce_mmddhhmi)
        header_map[key] = {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "data_create_ymd": None,
        }

    for row in header_map.values():
        ensure_race_stub(db, int(row["race_id"]), cache=race_stub_cache)

    db.execute_many(
        """
        INSERT INTO core.wh_header (race_id, data_kbn, announce_mmddhhmi, data_create_ymd)
        VALUES (%(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s, %(data_create_ymd)s)
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO NOTHING
        """,
        list(header_map.values()),
    )

    detail_rows = [
        {
            "race_id": record.race_id,
            "data_kbn": record.data_kbn,
            "announce_mmddhhmi": record.announce_mmddhhmi,
            "horse_no": record.horse_no,
            "body_weight_kg": record.body_weight_kg,
            "diff_sign": record.diff_sign,
            "diff_kg": record.diff_kg,
        }
        for record in records
    ]
    db.execute_many(
        """
        INSERT INTO core.wh_detail (
            race_id, data_kbn, announce_mmddhhmi,
            horse_no, body_weight_kg, diff_sign, diff_kg
        ) VALUES (
            %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
            %(horse_no)s, %(body_weight_kg)s, %(diff_sign)s, %(diff_kg)s
        )
        ON CONFLICT (race_id, data_kbn, announce_mmddhhmi, horse_no) DO UPDATE SET
            body_weight_kg = EXCLUDED.body_weight_kg,
            diff_sign = EXCLUDED.diff_sign,
            diff_kg = EXCLUDED.diff_kg
        """,
        detail_rows,
    )
    return len(detail_rows)


def _mining_params(record: DMRecord | TMRecord) -> dict[str, Any]:
    return {
        "race_id": record.race_id,
        "horse_no": record.horse_no,
        "data_kbn": record.data_kbn,
        "data_create_ymd": getattr(record, "data_create_ymd", "00000000"),
        "data_create_hm": getattr(record, "data_create_hm", "0000"),
        "dm_time_x10": getattr(record, "dm_time_x10", None),
        "dm_rank": getattr(record, "dm_rank", None),
        "tm_score": getattr(record, "tm_score", None),
        "tm_rank": getattr(record, "tm_rank", None),
        "payload_raw": record.payload_raw,
    }


def _ensure_race_stubs_for_mining_records(
    db: Database, records: list[DMRecord | TMRecord], race_stub_cache: set[int] | None
) -> None:
    if not records:
        return
    race_ids = {int(r.race_id) for r in records if getattr(r, "race_id", 0)}
    for race_id in race_ids:
        ensure_race_stub(db, race_id, cache=race_stub_cache)


def insert_mining_records_batch(
    db: Database,
    rec_id: str,
    records: list[DMRecord | TMRecord],
    race_stub_cache: set[int] | None = None,
) -> int:
    if not records:
        return 0

    _ensure_race_stubs_for_mining_records(db, records, race_stub_cache)
    if rec_id == "DM":
        db.execute_many(
            """
            INSERT INTO core.mining_dm (
                race_id, horse_no, data_kbn, dm_time_x10, dm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(horse_no)s, %(data_kbn)s,
                %(dm_time_x10)s, %(dm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, horse_no) DO UPDATE SET
                data_kbn = EXCLUDED.data_kbn,
                dm_time_x10 = EXCLUDED.dm_time_x10,
                dm_rank = EXCLUDED.dm_rank,
                payload_raw = EXCLUDED.payload_raw
            """,
            [_mining_params(record) for record in records],
        )
    else:
        db.execute_many(
            """
            INSERT INTO core.mining_tm (
                race_id, horse_no, data_kbn, tm_score, tm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(horse_no)s, %(data_kbn)s, %(tm_score)s, %(tm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, horse_no) DO UPDATE SET
                data_kbn = EXCLUDED.data_kbn,
                tm_score = EXCLUDED.tm_score,
                tm_rank = EXCLUDED.tm_rank,
                payload_raw = EXCLUDED.payload_raw
            """,
            [_mining_params(record) for record in records],
        )
    return len(records)


def insert_rt_mining_records_batch(
    db: Database,
    rec_id: str,
    records: list[DMRecord | TMRecord],
    race_stub_cache: set[int] | None = None,
) -> int:
    if not records:
        return 0

    _ensure_race_stubs_for_mining_records(db, records, race_stub_cache)
    if rec_id == "DM":
        db.execute_many(
            """
            INSERT INTO core.rt_mining_dm (
                race_id, data_kbn, data_create_ymd, data_create_hm,
                horse_no, dm_time_x10, dm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(data_kbn)s, %(data_create_ymd)s, %(data_create_hm)s,
                %(horse_no)s, %(dm_time_x10)s, %(dm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no) DO UPDATE SET
                dm_time_x10 = EXCLUDED.dm_time_x10,
                dm_rank = EXCLUDED.dm_rank,
                payload_raw = EXCLUDED.payload_raw,
                ingested_at = now()
            """,
            [_mining_params(record) for record in records],
        )
    else:
        db.execute_many(
            """
            INSERT INTO core.rt_mining_tm (
                race_id, data_kbn, data_create_ymd, data_create_hm,
                horse_no, tm_score, tm_rank, payload_raw
            ) VALUES (
                %(race_id)s, %(data_kbn)s, %(data_create_ymd)s, %(data_create_hm)s,
                %(horse_no)s, %(tm_score)s, %(tm_rank)s, %(payload_raw)s
            )
            ON CONFLICT (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no) DO UPDATE SET
                tm_score = EXCLUDED.tm_score,
                tm_rank = EXCLUDED.tm_rank,
                payload_raw = EXCLUDED.payload_raw,
                ingested_at = now()
            """,
            [_mining_params(record) for record in records],
        )
    return len(records)


def delete_rt_mining_records(
    db: Database, rec_id: str, race_id: int, data_create_ymd: str, data_create_hm: str
) -> int:
    sql = (
        """
        DELETE FROM core.rt_mining_dm
        WHERE race_id = %(race_id)s
          AND data_create_ymd = %(data_create_ymd)s
          AND data_create_hm = %(data_create_hm)s
        """
        if rec_id == "DM"
        else """
        DELETE FROM core.rt_mining_tm
        WHERE race_id = %(race_id)s
          AND data_create_ymd = %(data_create_ymd)s
          AND data_create_hm = %(data_create_hm)s
        """
    )
    db.execute(
        sql,
        {
            "race_id": race_id,
            "data_create_ymd": data_create_ymd,
            "data_create_hm": data_create_hm,
        },
    )
    return 1


def _extract_rt_mining_header(payload: str) -> dict[str, Any] | None:
    try:
        b_payload = payload.encode("cp932")
    except UnicodeEncodeError:
        b_payload = payload.encode("cp932", errors="replace")

    if len(b_payload) < 31:
        b_payload = b_payload.ljust(31, b" ")

    rec_type = b_payload[0:2].decode("cp932", errors="ignore").strip()
    if rec_type not in {"DM", "TM"}:
        return None

    data_kbn_raw = b_payload[2:3].decode("cp932", errors="ignore").strip()
    try:
        data_kbn = int(data_kbn_raw)
    except (TypeError, ValueError):
        data_kbn = -1

    data_create_ymd = b_payload[3:11].decode("cp932", errors="ignore").strip() or "00000000"
    race_key = b_payload[11:27].decode("cp932", errors="ignore")
    data_create_hm = b_payload[27:31].decode("cp932", errors="ignore").strip() or "0000"

    race_id = 0
    if len(race_key) >= 16:
        try:
            year = int(race_key[0:4])
            month = int(race_key[4:6])
            day = int(race_key[6:8])
            track = int(race_key[8:10])
            race_no = int(race_key[14:16])
            date_int = year * 10000 + month * 100 + day
            race_id = date_int * 10000 + track * 100 + race_no
        except ValueError:
            race_id = 0

    return {
        "rec_type": rec_type,
        "race_id": race_id,
        "data_kbn": data_kbn,
        "data_create_ymd": data_create_ymd,
        "data_create_hm": data_create_hm,
    }


def insert_event_change(db: Database, event: EventChangeRecord, race_stub_cache: set[int]) -> None:
    ensure_race_stub(db, event.race_id, cache=race_stub_cache)
    payload = dict(event.payload_parsed or {})
    payload.setdefault("data_kbn", event.data_kbn)
    payload.setdefault("raw", event.payload_raw)
    payload_md5 = hashlib.md5(str(payload.get("raw", "")).encode("utf-8")).hexdigest()
    db.execute(
        """
        INSERT INTO core.event_change (
            race_id, record_type, data_kbn, data_create_ymd,
            announce_mmddhhmi, payload_parsed, payload_md5
        ) VALUES (
            %(race_id)s, %(record_type)s, %(data_kbn)s, %(data_create_ymd)s,
            %(announce_mmddhhmi)s, %(payload_parsed)s, %(payload_md5)s
        )
        ON CONFLICT (race_id, record_type, data_create_ymd, announce_mmddhhmi, payload_md5)
        DO NOTHING
        """,
        {
            "race_id": event.race_id,
            "record_type": event.record_type,
            "data_kbn": event.data_kbn,
            "data_create_ymd": event.data_create_ymd,
            "announce_mmddhhmi": event.announce_mmddhhmi,
            "payload_parsed": json.dumps(payload, ensure_ascii=False),
            "payload_md5": payload_md5,
        },
    )


def process_file(
    db: Database,
    file_path: Path,
    *,
    central_only: bool,
    commit_interval: int = 5000,
    raw_batch_size: int = 1000,
) -> dict[str, int]:
    dataspec = file_path.stem.split("_")[0]
    is_rt_mining = dataspec in {"0B13", "0B17"}

    stats = {
        "raw": 0,
        "race": 0,
        "runner": 0,
        "result": 0,
        "payout": 0,
        "o1": 0,
        "o3": 0,
        "wh": 0,
        "mining": 0,
        "rt_mining_delete": 0,
        "event": 0,
        "horse": 0,
        "jockey": 0,
        "trainer": 0,
        "skipped_non_central": 0,
        "errors": 0,
    }

    master_jockeys, master_trainers = prepare_master_data_cache(db)
    race_stub_cache: set[int] = set()

    raw_batch: list[dict[str, Any]] = []
    o1_batch: list[OddsTimeSeriesRecord] = []
    o3_batch: list[O3WideRecord] = []
    wh_batch: list[WHRecord] = []
    dm_batch: list[DMRecord] = []
    tm_batch: list[TMRecord] = []

    def flush_all(force: bool = False) -> None:
        nonlocal raw_batch, o1_batch, o3_batch, wh_batch, dm_batch, tm_batch
        if raw_batch and (force or len(raw_batch) >= raw_batch_size):
            stats["raw"] += insert_raw_records_batch(db, dataspec, raw_batch)
            raw_batch = []

        if o1_batch and (force or len(o1_batch) >= 50000):
            stats["o1"] += upsert_o1_timeseries_bulk(db, o1_batch, race_stub_cache)
            o1_batch = []

        if o3_batch and (force or len(o3_batch) >= 30000):
            stats["o3"] += upsert_o3_wide_records_bulk(db, o3_batch, race_stub_cache)
            o3_batch = []

        if wh_batch and (force or len(wh_batch) >= 10000):
            stats["wh"] += upsert_wh_records_bulk(db, wh_batch, race_stub_cache)
            wh_batch = []

        if dm_batch and (force or len(dm_batch) >= 10000):
            if is_rt_mining:
                stats["mining"] += insert_rt_mining_records_batch(
                    db, "DM", dm_batch, race_stub_cache
                )
            else:
                stats["mining"] += insert_mining_records_batch(db, "DM", dm_batch, race_stub_cache)
            dm_batch = []

        if tm_batch and (force or len(tm_batch) >= 10000):
            if is_rt_mining:
                stats["mining"] += insert_rt_mining_records_batch(
                    db, "TM", tm_batch, race_stub_cache
                )
            else:
                stats["mining"] += insert_mining_records_batch(db, "TM", tm_batch, race_stub_cache)
            tm_batch = []

    for index, record in enumerate(load_jsonl(file_path), start=1):
        rec_id = str(record.get("rec_id", "")).strip()
        payload = str(record.get("payload", ""))
        try:
            accepted_raw = False

            if rec_id == "RA":
                race = RaceRecord.parse(payload)
                if central_only and not is_central_track(race.track_code):
                    stats["skipped_non_central"] += 1
                else:
                    upsert_race(db, race)
                    stats["race"] += 1
                    accepted_raw = True

            elif rec_id == "SE":
                runner = RunnerRecord.parse(payload)
                if central_only and not is_central_race(runner.race_id):
                    stats["skipped_non_central"] += 1
                else:
                    upsert_runner(db, runner, master_jockeys, master_trainers, race_stub_cache)
                    stats["runner"] += 1
                    if runner.finish_pos and runner.finish_pos > 0:
                        stats["result"] += 1
                    accepted_raw = True

            elif rec_id == "HR":
                payouts = PayoutRecord.parse(payload)
                if not payouts:
                    pass
                elif central_only and not is_central_race(payouts[0].race_id):
                    stats["skipped_non_central"] += 1
                else:
                    stats["payout"] += upsert_payout_records(db, payouts, race_stub_cache)
                    accepted_raw = True

            elif rec_id == "O1":
                records = OddsTimeSeriesRecord.parse(payload)
                if not records:
                    pass
                elif central_only and not is_central_race(records[0].race_id):
                    stats["skipped_non_central"] += 1
                else:
                    o1_batch.extend(records)
                    accepted_raw = True

            elif rec_id == "O3":
                records = O3WideRecord.parse(payload)
                if records:
                    race_id = records[0].race_id
                    if central_only and not is_central_race(race_id):
                        stats["skipped_non_central"] += 1
                    else:
                        o3_batch.extend(records)
                        accepted_raw = True
                else:
                    header = O3HeaderRecord.parse(payload)
                    if not (central_only and not is_central_race(header.race_id)):
                        ensure_race_stub(db, header.race_id, race_stub_cache)
                        db.execute(
                            """
                            INSERT INTO core.o3_header (
                                race_id, data_kbn, announce_mmddhhmi,
                                wide_pool_total_100yen, starters, sale_flag_wide, data_create_ymd
                            ) VALUES (
                                %(race_id)s, %(data_kbn)s, %(announce_mmddhhmi)s,
                                %(wide_pool_total_100yen)s, %(starters)s, %(sale_flag_wide)s,
                                %(data_create_ymd)s
                            )
                            ON CONFLICT (race_id, data_kbn, announce_mmddhhmi) DO UPDATE SET
                                wide_pool_total_100yen = EXCLUDED.wide_pool_total_100yen,
                                starters = EXCLUDED.starters,
                                sale_flag_wide = EXCLUDED.sale_flag_wide,
                                data_create_ymd = EXCLUDED.data_create_ymd
                            """,
                            {
                                "race_id": header.race_id,
                                "data_kbn": header.data_kbn,
                                "announce_mmddhhmi": header.announce_mmddhhmi,
                                "wide_pool_total_100yen": header.wide_pool_total_100yen,
                                "starters": header.starters,
                                "sale_flag_wide": header.sale_flag_wide,
                                "data_create_ymd": header.data_create_ymd,
                            },
                        )
                        accepted_raw = True

            elif rec_id == "WH":
                records = WHRecord.parse(payload)
                if records:
                    if central_only and not is_central_race(records[0].race_id):
                        stats["skipped_non_central"] += 1
                    else:
                        wh_batch.extend(records)
                        accepted_raw = True

            elif rec_id == "DM":
                if is_rt_mining:
                    header = _extract_rt_mining_header(payload)
                    if header and header["data_kbn"] == 0 and header["race_id"] > 0:
                        if not (central_only and not is_central_race(int(header["race_id"]))):
                            flush_all(force=True)
                            stats["rt_mining_delete"] += delete_rt_mining_records(
                                db,
                                rec_id="DM",
                                race_id=int(header["race_id"]),
                                data_create_ymd=str(header["data_create_ymd"]),
                                data_create_hm=str(header["data_create_hm"]),
                            )
                            accepted_raw = True
                    else:
                        records = DMRecord.parse(payload)
                        if records:
                            if central_only and not is_central_race(records[0].race_id):
                                stats["skipped_non_central"] += 1
                            else:
                                dm_batch.extend(records)
                                accepted_raw = True
                else:
                    records = DMRecord.parse(payload)
                    if records:
                        if central_only and not is_central_race(records[0].race_id):
                            stats["skipped_non_central"] += 1
                        else:
                            dm_batch.extend(records)
                            accepted_raw = True

            elif rec_id == "TM":
                if is_rt_mining:
                    header = _extract_rt_mining_header(payload)
                    if header and header["data_kbn"] == 0 and header["race_id"] > 0:
                        if not (central_only and not is_central_race(int(header["race_id"]))):
                            flush_all(force=True)
                            stats["rt_mining_delete"] += delete_rt_mining_records(
                                db,
                                rec_id="TM",
                                race_id=int(header["race_id"]),
                                data_create_ymd=str(header["data_create_ymd"]),
                                data_create_hm=str(header["data_create_hm"]),
                            )
                            accepted_raw = True
                    else:
                        records = TMRecord.parse(payload)
                        if records:
                            if central_only and not is_central_race(records[0].race_id):
                                stats["skipped_non_central"] += 1
                            else:
                                tm_batch.extend(records)
                                accepted_raw = True
                else:
                    records = TMRecord.parse(payload)
                    if records:
                        if central_only and not is_central_race(records[0].race_id):
                            stats["skipped_non_central"] += 1
                        else:
                            tm_batch.extend(records)
                            accepted_raw = True

            elif rec_id in {"WE", "AV", "JC", "TC", "CC"}:
                event = EventChangeRecord.parse(payload)
                if central_only and not is_central_race(event.race_id):
                    stats["skipped_non_central"] += 1
                else:
                    insert_event_change(db, event, race_stub_cache)
                    stats["event"] += 1
                    accepted_raw = True

            elif rec_id == "UM":
                horse = HorseMasterRecord.parse(payload)
                upsert_horse_master(db, horse)
                stats["horse"] += 1
                accepted_raw = True

            elif rec_id == "KS":
                jockey = JockeyRecord.parse(payload)
                upsert_jockey(db, jockey)
                if jockey.jockey_id > 0:
                    master_jockeys.add(jockey.jockey_id)
                stats["jockey"] += 1
                accepted_raw = True

            elif rec_id == "CH":
                trainer = TrainerRecord.parse(payload)
                upsert_trainer(db, trainer)
                if trainer.trainer_id > 0:
                    master_trainers.add(trainer.trainer_id)
                stats["trainer"] += 1
                accepted_raw = True

            if accepted_raw:
                raw_batch.append(record)

            flush_all(force=False)

            if index % commit_interval == 0:
                flush_all(force=True)
                db.connect().commit()
                race_stub_cache = set()
                logger.info(
                    "%s: processed=%s raw=%s skipped_non_central=%s errors=%s",
                    file_path.name,
                    f"{index:,}",
                    f"{stats['raw']:,}",
                    f"{stats['skipped_non_central']:,}",
                    f"{stats['errors']:,}",
                )

        except Exception:
            db.connect().rollback()
            raw_batch = []
            o1_batch = []
            o3_batch = []
            wh_batch = []
            dm_batch = []
            tm_batch = []
            race_stub_cache = set()
            stats["errors"] += 1
            if stats["errors"] <= 20:
                logger.exception("process error file=%s rec_id=%s", file_path.name, rec_id)

    flush_all(force=True)
    db.connect().commit()
    return stats


def collect_input_files(input_pattern: str | None, input_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    if input_pattern:
        input_path = Path(input_pattern)
        if "*" in input_pattern:
            parent = input_path.parent if input_path.parent.exists() else Path(".")
            files.extend(sorted(parent.glob(input_path.name)))
        else:
            files.append(input_path)
    if input_dir:
        files.extend(sorted(input_dir.glob("*.jsonl")))
    seen: set[Path] = set()
    unique_files = []
    for file_path in files:
        if file_path in seen:
            continue
        seen.add(file_path)
        unique_files.append(file_path)
    return unique_files


def main() -> int:
    parser = argparse.ArgumentParser(description="JSONL -> PostgreSQL ロード（v2）")
    parser.add_argument("--input", help="入力JSONL（ワイルドカード対応）")
    parser.add_argument("--input-dir", type=Path, help="入力ディレクトリ（*.jsonl を処理）")
    parser.add_argument(
        "--include-non-central",
        action="store_true",
        help="中央競馬（場コード01-10）以外も取り込む",
    )
    parser.add_argument("--commit-interval", type=int, default=5000, help="コミット間隔")
    parser.add_argument("--raw-batch-size", type=int, default=1000, help="rawバッチ件数")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("--input または --input-dir を指定してください")

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    files = collect_input_files(args.input, args.input_dir)
    if not files:
        print("処理対象ファイルがありません")
        return 1

    total: dict[str, int] = {}
    central_only = not args.include_non_central

    with Database() as db:
        for file_path in files:
            logger.info("start: %s", file_path)
            stats = process_file(
                db,
                file_path,
                central_only=central_only,
                commit_interval=args.commit_interval,
                raw_batch_size=args.raw_batch_size,
            )
            logger.info("done: %s -> %s", file_path.name, stats)
            for key, value in stats.items():
                total[key] = total.get(key, 0) + value

    print("=" * 80)
    print("load summary")
    print("=" * 80)
    for key, value in sorted(total.items()):
        if value:
            print(f"{key:>20}: {value:,}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
