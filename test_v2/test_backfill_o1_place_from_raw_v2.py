from __future__ import annotations

from app.infrastructure.parsers import OddsTimeSeriesRecord
from scripts_v2 import backfill_o1_place_from_raw_v2 as backfill


class _DummyConn:
    def __init__(self):
        self.commit_calls = 0

    def commit(self):
        self.commit_calls += 1


class _FakeDB:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.conn = _DummyConn()

    def fetch_all(self, _sql, params):
        last_id = int(params["last_id"])
        limit = int(params["limit"])
        dataspecs = set(params["dataspecs"])
        return [
            row
            for row in self.rows
            if int(row["id"]) > last_id and str(row["dataspec"]) in dataspecs
        ][:limit]

    def connect(self):
        return self.conn


def test_within_date_range_uses_race_id_date():
    assert backfill._within_date_range(202602080101, "20260201", "20260228")
    assert not backfill._within_date_range(202512310101, "20260101", "20261231")


def test_backfill_o1_place_from_raw_pages_and_upserts_place_only(monkeypatch):
    rows = [
        {"id": 1, "dataspec": "0B41", "payload": "keep"},
        {"id": 2, "dataspec": "RACE", "payload": "old"},
        {"id": 3, "dataspec": "0B41", "payload": "noncentral"},
    ]
    db = _FakeDB(rows)

    parsed = {
        "keep": [
            OddsTimeSeriesRecord(
                race_id=202602080101,
                data_kbn=1,
                announce_mmddhhmi="02080512",
                horse_no=1,
                win_odds_x10=123,
                win_popularity=1,
                win_pool_total_100yen=123456,
                place_min_odds_x10=145,
                place_max_odds_x10=215,
                place_popularity=2,
                place_pool_total_100yen=234567,
                has_win_block=True,
                has_place_block=True,
            ),
            OddsTimeSeriesRecord(
                race_id=202602080101,
                data_kbn=1,
                announce_mmddhhmi="02080512",
                horse_no=2,
                win_odds_x10=456,
                win_popularity=3,
                win_pool_total_100yen=123456,
                place_min_odds_x10=300,
                place_max_odds_x10=500,
                place_popularity=4,
                place_pool_total_100yen=234567,
                has_win_block=True,
                has_place_block=True,
            ),
        ],
        "old": [
            OddsTimeSeriesRecord(
                race_id=201512310101,
                data_kbn=4,
                announce_mmddhhmi="00000000",
                horse_no=1,
                win_odds_x10=123,
                win_popularity=1,
                win_pool_total_100yen=123456,
                place_min_odds_x10=145,
                place_max_odds_x10=215,
                place_popularity=2,
                place_pool_total_100yen=234567,
                has_win_block=True,
                has_place_block=True,
            )
        ],
        "noncentral": [
            OddsTimeSeriesRecord(
                race_id=202602081101,
                data_kbn=1,
                announce_mmddhhmi="02080512",
                horse_no=1,
                win_odds_x10=123,
                win_popularity=1,
                win_pool_total_100yen=123456,
                place_min_odds_x10=145,
                place_max_odds_x10=215,
                place_popularity=2,
                place_pool_total_100yen=234567,
                has_win_block=True,
                has_place_block=True,
            )
        ],
    }

    monkeypatch.setattr(backfill.OddsTimeSeriesRecord, "parse", lambda payload: parsed[payload])

    calls: list[dict] = []

    def fake_upsert(
        _db,
        records,
        race_stub_cache,
        *,
        include_win_details=True,
        include_place_details=True,
    ):
        calls.append(
            {
                "len_records": len(records),
                "race_stub_cache": race_stub_cache,
                "include_win_details": include_win_details,
                "include_place_details": include_place_details,
            }
        )
        return (0, len(records))

    monkeypatch.setattr(backfill, "upsert_o1_timeseries_bulk", fake_upsert)

    stats = backfill.backfill_o1_place_from_raw(
        db,
        from_date="20260101",
        to_date="20261231",
        dataspecs=["0B41", "RACE"],
        batch_size=2,
        upsert_batch_size=2,
        include_non_central=False,
        dry_run=False,
    )

    assert len(calls) == 1
    assert calls[0]["len_records"] == 2
    assert calls[0]["race_stub_cache"] == set()
    assert calls[0]["include_win_details"] is False
    assert calls[0]["include_place_details"] is True
    assert db.conn.commit_calls == 1
    assert stats["raw_rows"] == 3
    assert stats["matched_snapshots"] == 1
    assert stats["o1_rows_parsed"] == 2
    assert stats["win_rows_seen"] == 2
    assert stats["place_rows_written"] == 2
    assert stats["skipped_out_of_range"] == 1
    assert stats["skipped_non_central"] == 1
