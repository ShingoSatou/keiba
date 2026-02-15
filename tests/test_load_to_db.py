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
