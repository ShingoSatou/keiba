from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from scripts import predict_t5


class _DummyModel:
    def predict(self, matrix):
        return [0 for _ in range(len(matrix))]


class _DummyModelWithProba:
    def predict(self, matrix):
        return [0 for _ in range(len(matrix))]

    def predict_proba(self, matrix):
        probs = []
        for value in matrix["f1"]:
            p = float(value)
            probs.append([1 - p, p])
        return np.array(probs)


class _StrictNumericModel:
    def predict(self, matrix):
        assert pd.api.types.is_numeric_dtype(matrix["f1"])
        return [0 for _ in range(len(matrix))]

    def predict_proba(self, matrix):
        assert pd.api.types.is_numeric_dtype(matrix["f1"])
        probs = []
        for value in matrix["f1"]:
            p = float(value)
            probs.append([1 - p, p])
        return np.array(probs)


class _DummyCalibrator:
    def predict_proba(self, matrix):
        probs = []
        for value in matrix["f1"]:
            p = float(value)
            probs.append([1 - p, p])
        return np.array(probs)


class _DummyDB:
    def __init__(self, dm_rows, tm_rows):
        self.dm_rows = dm_rows
        self.tm_rows = tm_rows

    def fetch_all(self, query, params):
        if "core.rt_mining_dm" in query:
            return self.dm_rows
        if "core.rt_mining_tm" in query:
            return self.tm_rows
        raise AssertionError("unexpected query")


def test_predict_and_score_skips_missing_and_stale(monkeypatch):
    monkeypatch.setattr(
        predict_t5,
        "load_model",
        lambda: (_DummyModel(), _DummyCalibrator(), ["f1"]),
    )
    frame = pd.DataFrame(
        [
            {
                "f1": 0.9,
                "odds_win_t5": None,
                "odds_snapshot_age_sec": 10,
                "odds_missing_flag": True,
            },
            {
                "f1": 0.9,
                "odds_win_t5": 4.0,
                "odds_snapshot_age_sec": 1000,
                "odds_missing_flag": False,
            },
            {
                "f1": 0.9,
                "odds_win_t5": 4.0,
                "odds_snapshot_age_sec": 10,
                "odds_missing_flag": False,
            },
        ]
    )
    result = predict_t5._predict_and_score(
        frame=frame,
        odds_stale_sec=900,
        slippage=0.15,
        min_prob=0.03,
        bet_amount=500,
    )

    assert result.loc[0, "recommendation"] == "skip"
    assert pd.isna(result.loc[0, "ev"])

    assert result.loc[1, "odds_stale_flag"] == 1
    assert result.loc[1, "recommendation"] == "skip"
    assert result.loc[1, "ev"] > 0

    assert result.loc[2, "odds_stale_flag"] == 0
    assert result.loc[2, "recommendation"] == "buy"
    assert result.loc[2, "bet_amount"] == 500


def test_predict_and_score_works_without_calibrator(monkeypatch):
    monkeypatch.setattr(
        predict_t5,
        "load_model",
        lambda: (_DummyModelWithProba(), None, ["f1"]),
    )
    frame = pd.DataFrame(
        [
            {
                "f1": 0.7,
                "odds_win_t5": 4.0,
                "odds_snapshot_age_sec": 10,
                "odds_missing_flag": False,
            }
        ]
    )
    result = predict_t5._predict_and_score(
        frame=frame,
        odds_stale_sec=900,
        slippage=0.15,
        min_prob=0.03,
        bet_amount=500,
    )

    assert result.loc[0, "p"] == 0.7
    assert result.loc[0, "odds_stale_flag"] == 0


def test_predict_and_score_coerces_object_dtype_for_model(monkeypatch):
    monkeypatch.setattr(
        predict_t5,
        "load_model",
        lambda: (_StrictNumericModel(), None, ["f1"]),
    )
    frame = pd.DataFrame(
        [
            {
                "f1": Decimal("0.7"),
                "odds_win_t5": 4.0,
                "odds_snapshot_age_sec": 10,
                "odds_missing_flag": False,
            }
        ]
    )
    result = predict_t5._predict_and_score(
        frame=frame,
        odds_stale_sec=900,
        slippage=0.15,
        min_prob=0.03,
        bet_amount=500,
    )

    assert result.loc[0, "p"] == 0.7


def test_predict_and_score_skips_non_positive_odds(monkeypatch):
    monkeypatch.setattr(
        predict_t5,
        "load_model",
        lambda: (_DummyModel(), _DummyCalibrator(), ["f1"]),
    )
    frame = pd.DataFrame(
        [
            {
                "f1": 0.9,
                "odds_win_t5": 0,
                "odds_snapshot_age_sec": 10,
                "odds_missing_flag": False,
            }
        ]
    )
    result = predict_t5._predict_and_score(
        frame=frame,
        odds_stale_sec=900,
        slippage=0.15,
        min_prob=0.03,
        bet_amount=500,
    )

    assert bool(result.loc[0, "odds_missing_flag"]) is True
    assert result.loc[0, "recommendation"] == "skip"
    assert pd.isna(result.loc[0, "ev"])


def test_merge_rt_mining_features_uses_snapshot_keys():
    asof_a = datetime(2026, 2, 3, 12, 30)
    asof_b = datetime(2026, 2, 3, 12, 35)
    frame = pd.DataFrame(
        [
            {
                "race_id": 202602030501,
                "horse_no": 1,
                "asof_ts": asof_a,
                "dm_kbn": 3,
                "dm_create_time": "202602031230",
                "tm_kbn": 2,
                "tm_create_time": "202602031231",
            },
            {
                "race_id": 202602030501,
                "horse_no": 1,
                "asof_ts": asof_b,
                "dm_kbn": 3,
                "dm_create_time": "202602031235",
                "tm_kbn": 2,
                "tm_create_time": "202602031236",
            },
        ]
    )
    db = _DummyDB(
        dm_rows=[
            {
                "race_id": 202602030501,
                "horse_no": 1,
                "asof_ts": asof_a,
                "dm_time_x10": 900,
                "dm_rank": 1,
            }
        ],
        tm_rows=[
            {
                "race_id": 202602030501,
                "horse_no": 1,
                "asof_ts": asof_a,
                "tm_score": 88,
                "tm_rank": 2,
            }
        ],
    )
    merged = predict_t5._merge_rt_mining_features(
        db,
        frame=frame,
        race_date="2026-02-03",
        feature_set="realtime",
    )

    assert merged.loc[0, "dm_pred_time_sec"] == 90
    assert merged.loc[0, "dm_missing_flag"] == 0
    assert merged.loc[0, "tm_score"] == 88
    assert merged.loc[0, "tm_missing_flag"] == 0

    assert pd.isna(merged.loc[1, "dm_pred_time_sec"])
    assert merged.loc[1, "dm_missing_flag"] == 1
    assert pd.isna(merged.loc[1, "tm_score"])
    assert merged.loc[1, "tm_missing_flag"] == 1


def test_build_audit_payload_has_required_keys(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "race_id": 202602030501,
                "track_code": 5,
                "race_no": 1,
                "post_time": "12:35:00",
                "asof_ts": datetime(2026, 2, 3, 12, 30),
                "o1_announce_mmddhhmi": "02031220",
                "wh_announce_mmddhhmi": "02031215",
                "event_change_keys": {"we_event_id": 1, "cc_event_id": 2},
                "dm_kbn": 3,
                "dm_create_time": "202602031228",
                "tm_kbn": 2,
                "tm_create_time": "202602031229",
                "code_version": "abc1234",
                "odds_missing_flag": False,
                "odds_stale_flag": 0,
                "recommendation": "buy",
                "horse_no": 3,
                "p": 0.25,
                "ev": 0.12,
                "bet_amount": 500,
            }
        ]
    )
    payload = predict_t5._build_audit_payload(
        frame=frame,
        race_date="2026-02-03",
        feature_set="realtime",
        odds_stale_sec=900,
        slippage=0.15,
        min_prob=0.03,
        bet_amount=500,
        output_dir=tmp_path,
    )

    assert payload["race_date"] == "2026-02-03"
    assert payload["feature_set"] == "realtime"
    assert "predict_code_version" in payload
    assert "files" in payload
    assert len(payload["races"]) == 1

    race = payload["races"][0]
    assert "o1_announce_mmddhhmi" in race
    assert "wh_announce_mmddhhmi" in race
    assert "event_change_keys" in race
    assert "dm_kbn" in race and "dm_create_time" in race
    assert "tm_kbn" in race and "tm_create_time" in race
    assert "code_version" in race
    assert race["counts"]["buy"] == 1
    assert "summary" in payload
    assert "missing_rates_pct" in payload["summary"]
    assert "source_refs" in race
    assert "event_changes" in race["source_refs"]


def test_apply_we_cc_overrides_updates_distance_surface_and_going():
    frame = pd.DataFrame(
        [
            {
                "race_id": 202602030501,
                "surface": 1,
                "distance_m": 1600,
                "going": 1,
                "weather": 1,
                "event_change_keys": {
                    "cc_event_id": 101,
                    "cc_distance_m_after": "1400",
                    "cc_track_type_after": "24",
                    "we_event_id": 201,
                    "we_weather_now": "3",
                    "we_going_turf_now": "2",
                    "we_going_dirt_now": "4",
                },
            }
        ]
    )
    overridden = predict_t5._apply_we_cc_overrides(frame)

    assert int(overridden.loc[0, "distance_m"]) == 1400
    assert int(overridden.loc[0, "surface"]) == 2
    assert int(overridden.loc[0, "going"]) == 4
    assert int(overridden.loc[0, "weather"]) == 3
    assert predict_t5.distance_to_bucket(int(overridden.loc[0, "distance_m"])) == 1400
    assert "distance_m" in overridden.loc[0, "race_field_overrides"]
