from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from scripts_v3.odds_v3_common import (
    assert_asof_no_future_reference,
    merge_odds_features,
    select_place_t10_snapshot,
    select_t10_snapshot,
)


def test_t10_selector_uses_latest_snapshot_before_asof() -> None:
    race_dt = datetime(2025, 1, 5, 15, 0, 0)
    asof = datetime(2025, 1, 5, 14, 50, 0)
    odds_long = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "horse_no": [1, 1, 1],
            "data_kbn": [1, 3, 4],
            "race_datetime": [race_dt, race_dt, race_dt],
            "asof_t10": [asof, asof, asof],
            "announce_datetime": [
                datetime(2025, 1, 5, 14, 30, 0),
                datetime(2025, 1, 5, 14, 49, 0),
                datetime(2025, 1, 5, 14, 55, 0),
            ],
            "odds_win": [5.0, 4.2, 3.8],
        }
    )
    t10 = select_t10_snapshot(odds_long)
    assert len(t10) == 1
    row = t10.iloc[0]
    assert int(row["horse_no"]) == 1
    assert float(row["odds_win_t10"]) == pytest.approx(4.2)
    assert pd.Timestamp(row["odds_t10_announce_dt"]) == pd.Timestamp("2025-01-05 14:49:00")


def test_place_t10_selector_uses_latest_snapshot_before_asof() -> None:
    race_dt = datetime(2025, 1, 5, 15, 0, 0)
    asof = datetime(2025, 1, 5, 14, 50, 0)
    odds_long = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "horse_no": [1, 1, 1],
            "data_kbn": [1, 3, 4],
            "race_datetime": [race_dt, race_dt, race_dt],
            "asof_t10": [asof, asof, asof],
            "announce_datetime": [
                datetime(2025, 1, 5, 14, 30, 0),
                datetime(2025, 1, 5, 14, 49, 0),
                datetime(2025, 1, 5, 14, 55, 0),
            ],
            "odds_place_lower": [1.8, 1.6, 1.4],
            "odds_place_upper": [2.3, 2.1, 1.8],
        }
    )
    t10 = select_place_t10_snapshot(odds_long)
    assert len(t10) == 1
    row = t10.iloc[0]
    assert float(row["odds_place_t10_lower"]) == pytest.approx(1.6)
    assert float(row["odds_place_t10_upper"]) == pytest.approx(2.1)


def test_asof_guard_raises_on_future_announce() -> None:
    frame = pd.DataFrame(
        {
            "odds_t10_announce_dt": [datetime(2025, 1, 5, 14, 51, 0)],
            "odds_t10_asof_dt": [datetime(2025, 1, 5, 14, 50, 0)],
        }
    )
    with pytest.raises(ValueError):
        assert_asof_no_future_reference(frame)


def test_merge_odds_features_builds_place_width_log_ratio() -> None:
    features = pd.DataFrame({"race_id": [1], "horse_no": [1]})
    race_dt = datetime(2025, 1, 5, 15, 0, 0)
    win_long = pd.DataFrame(
        {
            "race_id": [1],
            "horse_no": [1],
            "data_kbn": [3],
            "race_datetime": [race_dt],
            "asof_t20": [datetime(2025, 1, 5, 14, 40, 0)],
            "asof_t15": [datetime(2025, 1, 5, 14, 45, 0)],
            "asof_t10": [datetime(2025, 1, 5, 14, 50, 0)],
            "announce_datetime": [datetime(2025, 1, 5, 14, 49, 0)],
            "odds_win": [3.5],
        }
    )
    place_long = pd.DataFrame(
        {
            "race_id": [1],
            "horse_no": [1],
            "data_kbn": [3],
            "race_datetime": [race_dt],
            "asof_t20": [datetime(2025, 1, 5, 14, 40, 0)],
            "asof_t15": [datetime(2025, 1, 5, 14, 45, 0)],
            "asof_t10": [datetime(2025, 1, 5, 14, 50, 0)],
            "announce_datetime": [datetime(2025, 1, 5, 14, 49, 0)],
            "odds_place_lower": [1.5],
            "odds_place_upper": [2.1],
        }
    )

    merged = merge_odds_features(features, win_long, place_long)
    expected = pytest.approx(math.log(2.1 / 1.5))
    assert float(merged.loc[0, "place_width_log_ratio_t10"]) == expected
    assert float(merged.loc[0, "place_width_log_ratio"]) == expected
