import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from app.infrastructure.parsers import OddsRecord
from scripts import load_to_db


class _DummyConn:
    def commit(self):
        pass

    def rollback(self):
        pass


class _DummyDB:
    def __init__(self):
        self._conn = _DummyConn()

    def connect(self):
        return self._conn


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_process_file_defers_o1_until_runner_ready(tmp_path, monkeypatch):
    file_path = tmp_path / "RACE_test.jsonl"
    _write_jsonl(
        file_path,
        [
            {"rec_id": "O1", "payload": "o1_payload"},
            {"rec_id": "SE", "payload": "se_payload"},
        ],
    )

    call_order: list[str] = []

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(load_to_db, "insert_raw_records_batch", lambda db, dataspec, batch: None)
    monkeypatch.setattr(load_to_db.RunnerRecord, "parse", lambda payload: SimpleNamespace())
    monkeypatch.setattr(
        load_to_db.OddsRecord,
        "parse",
        lambda payload: [
            OddsRecord(
                race_id=202401060601,
                bet_type=1,
                horse_no=1,
                odds_1=2.4,
                odds_2=None,
                popularity=1,
            ),
            OddsRecord(
                race_id=202401060601,
                bet_type=3,
                horse_no="11",
                odds_1=12.3,
                odds_2=None,
                popularity=3,
            ),
        ],
    )

    def fake_upsert_runner(db, runner, master_jockeys=None, master_trainers=None):
        call_order.append("runner")

    def fake_upsert_odds(db, odds):
        call_order.append("odds")
        return 1

    monkeypatch.setattr(load_to_db, "upsert_runner", fake_upsert_runner)
    monkeypatch.setattr(load_to_db, "upsert_odds", fake_upsert_odds)

    stats = load_to_db.process_file(_DummyDB(), file_path)

    assert call_order == ["runner", "odds"]
    assert stats["o1_deferred_records"] == 1
    assert stats["odds_parsed"] == 2
    assert stats["odds"] == 1
    assert stats["odds_upserted"] == 1
    assert stats["odds_skipped_bracket"] == 1
    assert stats["odds_missing_runner"] == 0


def test_process_file_counts_missing_runner_by_rowcount(tmp_path, monkeypatch):
    file_path = tmp_path / "RACE_test.jsonl"
    _write_jsonl(
        file_path,
        [
            {"rec_id": "O1", "payload": "o1_payload"},
            {"rec_id": "SE", "payload": "se_payload"},
        ],
    )

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(load_to_db, "insert_raw_records_batch", lambda db, dataspec, batch: None)
    monkeypatch.setattr(load_to_db.RunnerRecord, "parse", lambda payload: SimpleNamespace())
    monkeypatch.setattr(
        load_to_db.OddsRecord,
        "parse",
        lambda payload: [
            OddsRecord(
                race_id=202401060601,
                bet_type=1,
                horse_no=1,
                odds_1=2.4,
                odds_2=None,
                popularity=1,
            )
        ],
    )
    monkeypatch.setattr(load_to_db, "upsert_runner", lambda db, runner, *args, **kwargs: None)
    monkeypatch.setattr(load_to_db, "upsert_odds", lambda db, odds: 0)

    stats = load_to_db.process_file(_DummyDB(), file_path)

    assert stats["odds"] == 0
    assert stats["odds_upserted"] == 0
    assert stats["odds_missing_runner"] == 1


def test_process_file_skips_raw_jv_raw_for_snpn(tmp_path, monkeypatch):
    file_path = tmp_path / "SNPN_test.jsonl"
    _write_jsonl(
        file_path,
        [
            {"rec_id": "CK", "payload": "ck_payload"},
        ],
    )

    raw_calls: list[int] = []

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(
        load_to_db,
        "insert_raw_records_batch",
        lambda db, dataspec, batch: raw_calls.append(len(batch)),
    )
    monkeypatch.setattr(load_to_db.CKRecord, "parse", lambda payload: SimpleNamespace())
    monkeypatch.setattr(
        load_to_db,
        "_build_ck_payloads",
        lambda dataspec, ck: ({"x": 1}, {"x": 2}, {"x": 3}),
    )
    monkeypatch.setattr(
        load_to_db,
        "insert_ck_records_batch",
        lambda db, raw_payloads, core_payloads, feat_payloads: len(raw_payloads),
    )

    stats = load_to_db.process_file(_DummyDB(), file_path)

    assert raw_calls == []
    assert stats["raw"] == 0
    assert stats["ck"] == 1
    assert stats["ck_skipped_make_date"] == 0


def test_process_file_counts_ck_skipped_make_date(tmp_path, monkeypatch):
    file_path = tmp_path / "SNPN_test.jsonl"
    _write_jsonl(
        file_path,
        [
            {"rec_id": "CK", "payload": "ck_payload"},
        ],
    )

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(load_to_db.CKRecord, "parse", lambda payload: SimpleNamespace())
    monkeypatch.setattr(load_to_db, "_build_ck_payloads", lambda dataspec, ck: None)
    monkeypatch.setattr(
        load_to_db,
        "insert_ck_records_batch",
        lambda db, raw_payloads, core_payloads, feat_payloads: len(raw_payloads),
    )

    stats = load_to_db.process_file(_DummyDB(), file_path)

    assert stats["ck"] == 0
    assert stats["ck_skipped_make_date"] == 1


def test_process_file_routes_rt_mining_for_0b13(tmp_path, monkeypatch):
    file_path = tmp_path / "0B13_test.jsonl"
    _write_jsonl(file_path, [{"rec_id": "DM", "payload": "dm_payload"}])

    called = {"rt": 0, "core": 0}

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(load_to_db, "insert_raw_records_batch", lambda db, dataspec, batch: None)
    monkeypatch.setattr(
        load_to_db.DMRecord,
        "parse",
        lambda payload: [
            SimpleNamespace(
                race_id=202602030501,
                horse_no=1,
                data_kbn=3,
                data_create_ymd="20260203",
                data_create_hm="1230",
                dm_time_x10=900,
                dm_rank=1,
                payload_raw=payload,
            )
        ],
    )
    monkeypatch.setattr(
        load_to_db,
        "insert_rt_mining_records_batch",
        lambda db, rec_id, records: called.__setitem__("rt", called["rt"] + len(records))
        or len(records),
    )
    monkeypatch.setattr(
        load_to_db,
        "insert_mining_records_batch",
        lambda db, rec_id, records: called.__setitem__("core", called["core"] + len(records))
        or len(records),
    )

    stats = load_to_db.process_file(_DummyDB(), file_path)

    assert stats["mining"] == 1
    assert called["rt"] == 1
    assert called["core"] == 0


def test_process_file_applies_rt_mining_delete_for_kbn0(tmp_path, monkeypatch):
    file_path = tmp_path / "0B17_test.jsonl"
    _write_jsonl(file_path, [{"rec_id": "TM", "payload": "tm_payload"}])

    called = {"delete": 0}

    monkeypatch.setattr(load_to_db, "prepare_master_data_cache", lambda db: (set(), set()))
    monkeypatch.setattr(load_to_db, "insert_raw_records_batch", lambda db, dataspec, batch: None)
    monkeypatch.setattr(
        load_to_db,
        "_extract_rt_mining_header",
        lambda payload: {
            "rec_type": "TM",
            "race_id": 202602030501,
            "data_kbn": 0,
            "data_create_ymd": "20260203",
            "data_create_hm": "1230",
        },
    )
    monkeypatch.setattr(
        load_to_db.TMRecord,
        "parse",
        lambda payload: (_ for _ in ()).throw(AssertionError("TMRecord.parse should not run")),
    )
    monkeypatch.setattr(
        load_to_db,
        "delete_rt_mining_records",
        lambda db, rec_id, race_id, data_create_ymd, data_create_hm: called.__setitem__(
            "delete", called["delete"] + 1
        )
        or 1,
    )

    stats = load_to_db.process_file(_DummyDB(), file_path)

    assert stats["rt_mining_delete"] == 1
    assert stats["mining"] == 0
    assert called["delete"] == 1


def test_upsert_race_updates_stub_surface_and_distance():
    class _CaptureDB:
        def __init__(self):
            self.sql = ""
            self.params = {}

        def execute(self, sql, params):
            self.sql = sql
            self.params = params

    db = _CaptureDB()
    race = SimpleNamespace(
        race_id=202602030501,
        race_date="2026-02-03",
        track_code=5,
        race_no=1,
        surface=1,
        distance_m=1600,
        going=1,
        weather=2,
        class_code=0,
        field_size=16,
        start_time=None,
    )

    load_to_db.upsert_race(db, race)

    assert "surface = CASE" in db.sql
    assert "distance_m = CASE" in db.sql
    assert db.params["surface"] == 1
    assert db.params["distance_m"] == 1600


def test_upsert_wh_records_ensures_race_stub(monkeypatch):
    called = {"race_id": None}

    class _CaptureDB:
        def execute(self, sql, params):
            _ = sql
            _ = params

    monkeypatch.setattr(
        load_to_db,
        "ensure_race_stub",
        lambda db, race_id, cache=None: called.__setitem__("race_id", race_id),
    )

    record = SimpleNamespace(
        race_id=202602030501,
        data_kbn=1,
        announce_mmddhhmi="02031230",
        horse_no=1,
        body_weight_kg=480,
        diff_sign="+",
        diff_kg=2,
    )
    inserted = load_to_db.upsert_wh_records(_CaptureDB(), [record])

    assert inserted == 1
    assert called["race_id"] == 202602030501


def test_insert_event_change_includes_payload_md5_and_data_kbn():
    class _CaptureDB:
        def __init__(self):
            self.sql = ""
            self.params = {}

        def execute(self, sql, params):
            self.sql = sql
            self.params = params

    db = _CaptureDB()
    record = SimpleNamespace(
        race_id=202602030501,
        record_type="JC",
        data_kbn=2,
        data_create_ymd="20260203",
        announce_mmddhhmi="02031235",
        payload_parsed={"horse_no": 3},
        payload_raw="raw-payload",
    )

    load_to_db.insert_event_change(db, record)

    assert "payload_md5" in db.sql
    assert "ON CONFLICT" in db.sql
    payload = json.loads(db.params["payload"])
    assert payload["data_kbn"] == 2
    assert payload["raw"] == "raw-payload"
    expected_md5 = hashlib.md5(b"raw-payload").hexdigest()
    assert db.params["payload_md5"] == expected_md5
