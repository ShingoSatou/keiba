from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from app.infrastructure.parsers import DMRecord, O3WideRecord, OddsTimeSeriesRecord
from scripts_v2 import load_to_db


class _DummyConn:
    def commit(self):
        return None

    def rollback(self):
        return None


class _DummyDB:
    def __init__(self):
        self.conn = _DummyConn()

    def connect(self):
        return self.conn


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_is_central_race():
    assert load_to_db.is_central_race(202602080101)
    assert load_to_db.is_central_race(202602080110)
    assert not load_to_db.is_central_race(202602081101)


def test_insert_event_change_includes_payload_md5_and_data_kbn(monkeypatch):
    captured = {}

    class _CaptureDB:
        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)

    record = SimpleNamespace(
        race_id=202602080101,
        record_type="JC",
        data_kbn=2,
        data_create_ymd="20260208",
        announce_mmddhhmi="02081225",
        payload_parsed={"horse_no": 8},
        payload_raw="raw-payload",
    )
    load_to_db.insert_event_change(_CaptureDB(), record, race_stub_cache=set())

    assert "payload_md5" in captured["sql"]
    payload = json.loads(captured["params"]["payload_parsed"])
    assert payload["data_kbn"] == 2
    assert payload["raw"] == "raw-payload"
    assert captured["params"]["payload_md5"] == hashlib.md5(b"raw-payload").hexdigest()


def test_upsert_o1_timeseries_bulk_allows_null_win_odds(monkeypatch):
    captured = {"header_rows": None, "win_rows": None, "place_rows": None}

    class _CaptureDB:
        def execute_many(self, sql, params):
            if "core.o1_header" in sql:
                captured["header_rows"] = params

        def execute(self, sql, params):
            if "core.o1_win" in sql:
                captured["win_rows"] = json.loads(params["rows_json"])
            elif "core.o1_place" in sql:
                captured["place_rows"] = json.loads(params["rows_json"])

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)

    rows = [
        OddsTimeSeriesRecord(
            race_id=202602080101,
            data_kbn=1,
            announce_mmddhhmi="02080512",
            horse_no=1,
            win_odds_x10=None,
            win_popularity=1,
            win_pool_total_100yen=123456,
            has_win_block=True,
        )
    ]
    inserted = load_to_db.upsert_o1_timeseries_bulk(_CaptureDB(), rows, race_stub_cache=set())

    assert inserted == (1, 0)
    assert captured["header_rows"] is not None
    assert captured["win_rows"] is not None
    assert captured["win_rows"][0]["win_odds_x10"] is None
    assert captured["place_rows"] is None


def test_upsert_o1_timeseries_bulk_deduplicates_conflict_keys(monkeypatch):
    captured = {"win_rows": None}

    class _CaptureDB:
        def execute_many(self, sql, params):
            _ = (sql, params)

        def execute(self, sql, params):
            if "core.o1_win" in sql:
                captured["win_rows"] = json.loads(params["rows_json"])

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)

    rows = [
        OddsTimeSeriesRecord(
            race_id=202602080101,
            data_kbn=1,
            announce_mmddhhmi="02080512",
            horse_no=1,
            win_odds_x10=123,
            win_popularity=2,
            win_pool_total_100yen=123456,
            has_win_block=True,
        ),
        OddsTimeSeriesRecord(
            race_id=202602080101,
            data_kbn=1,
            announce_mmddhhmi="02080512",
            horse_no=1,
            win_odds_x10=145,
            win_popularity=3,
            win_pool_total_100yen=123456,
            has_win_block=True,
        ),
    ]
    inserted = load_to_db.upsert_o1_timeseries_bulk(_CaptureDB(), rows, race_stub_cache=set())

    assert inserted == (1, 0)
    assert captured["win_rows"] is not None
    assert len(captured["win_rows"]) == 1
    assert captured["win_rows"][0]["win_odds_x10"] == 145
    assert captured["win_rows"][0]["win_popularity"] == 3


def test_upsert_o1_timeseries_bulk_writes_place_rows_and_header_fields(monkeypatch):
    captured = {"header_rows": None, "place_rows": None}

    class _CaptureDB:
        def execute_many(self, sql, params):
            if "core.o1_header" in sql:
                captured["header_rows"] = params

        def execute(self, sql, params):
            if "core.o1_place" in sql:
                captured["place_rows"] = json.loads(params["rows_json"])

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)

    rows = [
        OddsTimeSeriesRecord(
            race_id=202602080101,
            data_kbn=4,
            announce_mmddhhmi="00000000",
            horse_no=1,
            win_odds_x10=123,
            win_popularity=2,
            win_pool_total_100yen=123456,
            data_create_ymd="20260208",
            sale_flag_place=7,
            place_pay_key=3,
            place_min_odds_x10=145,
            place_max_odds_x10=215,
            place_popularity=4,
            place_pool_total_100yen=234567,
            has_win_block=True,
            has_place_block=True,
        )
    ]

    inserted = load_to_db.upsert_o1_timeseries_bulk(_CaptureDB(), rows, race_stub_cache=set())

    assert inserted == (1, 1)
    assert captured["header_rows"] is not None
    assert captured["header_rows"][0]["sale_flag_place"] == 7
    assert captured["header_rows"][0]["place_pay_key"] == 3
    assert captured["header_rows"][0]["place_pool_total_100yen"] == 234567
    assert captured["header_rows"][0]["data_create_ymd"] == "20260208"
    assert captured["place_rows"] is not None
    assert captured["place_rows"][0]["min_odds_x10"] == 145
    assert captured["place_rows"][0]["max_odds_x10"] == 215
    assert captured["place_rows"][0]["place_popularity"] == 4


def test_upsert_o3_wide_records_bulk_allows_null_odds(monkeypatch):
    captured = {"detail_rows": None}

    class _CaptureDB:
        def execute_many(self, sql, params):
            _ = (sql, params)

        def execute(self, sql, params):
            if "jsonb_to_recordset" in sql:
                captured["detail_rows"] = json.loads(params["rows_json"])

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)

    rows = [
        O3WideRecord(
            race_id=202602080101,
            data_kbn=4,
            announce_mmddhhmi="00000000",
            data_create_ymd="20260208",
            kumiban="0102",
            min_odds_x10=None,
            max_odds_x10=None,
            popularity=None,
            wide_pool_total_100yen=7654321,
            starters=16,
            sale_flag_wide=7,
        )
    ]
    inserted = load_to_db.upsert_o3_wide_records_bulk(_CaptureDB(), rows, race_stub_cache=set())

    assert inserted == 1
    assert captured["detail_rows"] is not None
    assert captured["detail_rows"][0]["min_odds_x10"] is None
    assert captured["detail_rows"][0]["max_odds_x10"] is None
    assert captured["detail_rows"][0]["popularity"] is None


def test_upsert_o3_wide_records_bulk_deduplicates_conflict_keys(monkeypatch):
    captured = {"detail_rows": None}

    class _CaptureDB:
        def execute_many(self, sql, params):
            _ = (sql, params)

        def execute(self, sql, params):
            if "jsonb_to_recordset" in sql:
                captured["detail_rows"] = json.loads(params["rows_json"])

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)

    rows = [
        O3WideRecord(
            race_id=202602080101,
            data_kbn=4,
            announce_mmddhhmi="00000000",
            data_create_ymd="20260208",
            kumiban="0102",
            min_odds_x10=125,
            max_odds_x10=195,
            popularity=3,
            wide_pool_total_100yen=7654321,
            starters=16,
            sale_flag_wide=7,
        ),
        O3WideRecord(
            race_id=202602080101,
            data_kbn=4,
            announce_mmddhhmi="00000000",
            data_create_ymd="20260208",
            kumiban="0102",
            min_odds_x10=145,
            max_odds_x10=215,
            popularity=2,
            wide_pool_total_100yen=7654321,
            starters=16,
            sale_flag_wide=7,
        ),
    ]
    inserted = load_to_db.upsert_o3_wide_records_bulk(_CaptureDB(), rows, race_stub_cache=set())

    assert inserted == 1
    assert captured["detail_rows"] is not None
    assert len(captured["detail_rows"]) == 1
    assert captured["detail_rows"][0]["min_odds_x10"] == 145
    assert captured["detail_rows"][0]["max_odds_x10"] == 215
    assert captured["detail_rows"][0]["popularity"] == 2


def test_insert_mining_records_batch_ensures_race_stub(monkeypatch):
    called_race_ids: set[int] = set()

    def _ensure_race_stub(db, race_id, cache=None):
        called_race_ids.add(int(race_id))

    monkeypatch.setattr(load_to_db, "ensure_race_stub", _ensure_race_stub)

    class _CaptureDB:
        def execute_many(self, sql, params):
            _ = (sql, params)

    records = [
        DMRecord(
            race_id=202602080101,
            horse_no=1,
            data_kbn=1,
            data_create_ymd="20260208",
            data_create_hm="1200",
            dm_time_x10=None,
            dm_rank=None,
            payload_raw="dm",
        ),
        DMRecord(
            race_id=202602080101,
            horse_no=2,
            data_kbn=1,
            data_create_ymd="20260208",
            data_create_hm="1200",
            dm_time_x10=None,
            dm_rank=None,
            payload_raw="dm",
        ),
        DMRecord(
            race_id=202602080102,
            horse_no=1,
            data_kbn=1,
            data_create_ymd="20260208",
            data_create_hm="1200",
            dm_time_x10=None,
            dm_rank=None,
            payload_raw="dm",
        ),
    ]

    load_to_db.insert_mining_records_batch(_CaptureDB(), "DM", records, race_stub_cache=set())
    assert called_race_ids == {202602080101, 202602080102}


def test_insert_rt_mining_records_batch_ensures_race_stub(monkeypatch):
    called_race_ids: set[int] = set()

    def _ensure_race_stub(db, race_id, cache=None):
        called_race_ids.add(int(race_id))

    monkeypatch.setattr(load_to_db, "ensure_race_stub", _ensure_race_stub)

    class _CaptureDB:
        def execute_many(self, sql, params):
            _ = (sql, params)

    records = [
        DMRecord(
            race_id=202602080101,
            horse_no=1,
            data_kbn=1,
            data_create_ymd="20260208",
            data_create_hm="1205",
            dm_time_x10=None,
            dm_rank=None,
            payload_raw="dm",
        )
    ]

    load_to_db.insert_rt_mining_records_batch(_CaptureDB(), "DM", records, race_stub_cache=set())
    assert called_race_ids == {202602080101}


def test_process_file_applies_rt_mining_delete_for_kbn0(tmp_path, monkeypatch):
    file_path = tmp_path / "0B17_test.jsonl"
    _write_jsonl(file_path, [{"rec_id": "TM", "payload": "tm_payload"}])

    called = {"delete": 0}

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(
        load_to_db, "insert_raw_records_batch", lambda db, dataspec, batch: len(batch)
    )
    monkeypatch.setattr(
        load_to_db,
        "_extract_rt_mining_header",
        lambda payload: {
            "rec_type": "TM",
            "race_id": 202602080101,
            "data_kbn": 0,
            "data_create_ymd": "20260208",
            "data_create_hm": "1225",
        },
    )
    monkeypatch.setattr(
        load_to_db,
        "delete_rt_mining_records",
        lambda db, rec_id, race_id, data_create_ymd, data_create_hm: called.__setitem__(
            "delete", called["delete"] + 1
        )
        or 1,
    )

    stats = load_to_db.process_file(_DummyDB(), file_path, central_only=True, commit_interval=1)

    assert stats["rt_mining_delete"] == 1
    assert called["delete"] == 1


def test_upsert_runner_handles_missing_age_and_sex(monkeypatch):
    class _CaptureDB:
        def execute(self, sql, params):
            _ = sql
            _ = params

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)
    runner = SimpleNamespace(
        race_id=202602080101,
        horse_id="2020100001",
        horse_name="TEST HORSE",
        horse_no=1,
        gate=1,
        jockey_id=12345,
        trainer_id=54321,
        carried_weight=56.0,
        body_weight=480,
        body_weight_diff=2,
        sex=None,
        finish_pos=1,
        time_sec=90.1,
        margin=None,
        final3f_sec=35.2,
        corner1_pos=1,
        corner2_pos=1,
        corner3_pos=1,
        corner4_pos=1,
        data_kubun="1",
        trainer_code_raw="54321",
        trainer_name_abbr="調教師名",
        jockey_code_raw="12345",
        jockey_name_abbr="騎手名",
    )
    load_to_db.upsert_runner(
        _CaptureDB(),
        runner,
        master_jockeys={12345},
        master_trainers={54321},
        race_stub_cache=set(),
    )


def test_upsert_runner_reuses_existing_horse_id_for_same_horse_no(monkeypatch):
    captured = {"runner_sql": None, "result_sql": None}

    class _CaptureDB:
        def execute(self, sql, params):
            _ = params
            if "INSERT INTO core.runner" in sql:
                captured["runner_sql"] = sql
            elif "INSERT INTO core.result" in sql:
                captured["result_sql"] = sql

    monkeypatch.setattr(load_to_db, "ensure_race_stub", lambda db, race_id, cache=None: None)
    runner = SimpleNamespace(
        race_id=202602080101,
        horse_id="new_horse_id",
        horse_name="TEST HORSE",
        horse_no=99,
        gate=8,
        jockey_id=12345,
        trainer_id=54321,
        carried_weight=56.0,
        body_weight=480,
        body_weight_diff=2,
        sex=None,
        finish_pos=1,
        time_sec=90.5,
        margin=None,
        final3f_sec=34.8,
        corner1_pos=1,
        corner2_pos=1,
        corner3_pos=1,
        corner4_pos=1,
        data_kubun="1",
        trainer_code_raw="54321",
        trainer_name_abbr="調教師名",
        jockey_code_raw="12345",
        jockey_name_abbr="騎手名",
    )

    load_to_db.upsert_runner(
        _CaptureDB(),
        runner,
        master_jockeys={12345},
        master_trainers={54321},
        race_stub_cache=set(),
    )

    assert captured["runner_sql"] is not None
    assert "COALESCE(" in captured["runner_sql"]
    assert "SELECT r.horse_id" in captured["runner_sql"]
    assert "r.horse_no = %(horse_no)s" in captured["runner_sql"]
    assert captured["result_sql"] is not None
    assert "COALESCE(" in captured["result_sql"]
