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


def test_merge_odds_features_builds_stacker_market_features() -> None:
    features = pd.DataFrame({"race_id": [1, 1], "horse_no": [1, 2]})
    race_dt = datetime(2025, 1, 5, 15, 0, 0)
    win_long = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1, 1, 1],
            "horse_no": [1, 1, 1, 2, 2, 2],
            "data_kbn": [3, 3, 3, 3, 3, 3],
            "race_datetime": [race_dt] * 6,
            "asof_t20": [datetime(2025, 1, 5, 14, 40, 0)] * 6,
            "asof_t15": [datetime(2025, 1, 5, 14, 45, 0)] * 6,
            "asof_t10": [datetime(2025, 1, 5, 14, 50, 0)] * 6,
            "announce_datetime": [
                datetime(2025, 1, 5, 14, 39, 0),
                datetime(2025, 1, 5, 14, 44, 0),
                datetime(2025, 1, 5, 14, 49, 0),
                datetime(2025, 1, 5, 14, 39, 0),
                datetime(2025, 1, 5, 14, 44, 0),
                datetime(2025, 1, 5, 14, 49, 0),
            ],
            "odds_win": [4.0, 3.0, 2.5, 8.0, 7.0, 6.0],
        }
    )
    place_long = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1, 1, 1],
            "horse_no": [1, 1, 1, 2, 2, 2],
            "data_kbn": [3, 3, 3, 3, 3, 3],
            "race_datetime": [race_dt] * 6,
            "asof_t20": [datetime(2025, 1, 5, 14, 40, 0)] * 6,
            "asof_t15": [datetime(2025, 1, 5, 14, 45, 0)] * 6,
            "asof_t10": [datetime(2025, 1, 5, 14, 50, 0)] * 6,
            "announce_datetime": [
                datetime(2025, 1, 5, 14, 39, 0),
                datetime(2025, 1, 5, 14, 44, 0),
                datetime(2025, 1, 5, 14, 49, 0),
                datetime(2025, 1, 5, 14, 39, 0),
                datetime(2025, 1, 5, 14, 44, 0),
                datetime(2025, 1, 5, 14, 49, 0),
            ],
            "odds_place_lower": [1.8, 1.7, 1.5, 2.8, 2.6, 3.0],
            "odds_place_upper": [2.2, 2.0, 1.9, 3.4, 3.2, 2.4],
        }
    )

    merged = merge_odds_features(features, win_long, place_long)
    horse1 = merged[merged["horse_no"] == 1].iloc[0]
    horse2 = merged[merged["horse_no"] == 2].iloc[0]

    p_t20_h1 = (1.0 / 4.0) / ((1.0 / 4.0) + (1.0 / 8.0))
    p_t15_h1 = (1.0 / 3.0) / ((1.0 / 3.0) + (1.0 / 7.0))
    p_t10_h1 = (1.0 / 2.5) / ((1.0 / 2.5) + (1.0 / 6.0))
    assert float(horse1["p_win_odds_t20_norm"]) == pytest.approx(p_t20_h1)
    assert float(horse1["p_win_odds_t15_norm"]) == pytest.approx(p_t15_h1)
    assert float(horse1["p_win_odds_t10_norm"]) == pytest.approx(p_t10_h1)
    assert float(horse1["d_logit_win_15_20"]) == pytest.approx(
        math.log(p_t15_h1 / (1.0 - p_t15_h1)) - math.log(p_t20_h1 / (1.0 - p_t20_h1))
    )
    assert float(horse1["d_logit_win_10_15"]) == pytest.approx(
        math.log(p_t10_h1 / (1.0 - p_t10_h1)) - math.log(p_t15_h1 / (1.0 - p_t15_h1))
    )
    assert float(horse1["d_logit_win_10_20"]) == pytest.approx(
        math.log(p_t10_h1 / (1.0 - p_t10_h1)) - math.log(p_t20_h1 / (1.0 - p_t20_h1))
    )

    place_mid_t20_h1 = 1.0 / math.sqrt(1.8 * 2.2)
    place_mid_t10_h1 = 1.0 / math.sqrt(1.5 * 1.9)
    place_width_t10_h1 = math.log(1.9 / 1.5)
    place_width_t20_h1 = math.log(2.2 / 1.8)
    assert float(horse1["place_mid_prob_t20"]) == pytest.approx(place_mid_t20_h1)
    assert float(horse1["place_mid_prob_t10"]) == pytest.approx(place_mid_t10_h1)
    assert float(horse1["place_width_log_ratio_t10"]) == pytest.approx(place_width_t10_h1)
    assert float(horse1["place_width_log_ratio"]) == pytest.approx(place_width_t10_h1)
    assert float(horse1["d_place_mid_10_20"]) == pytest.approx(place_mid_t10_h1 - place_mid_t20_h1)
    assert float(horse1["d_place_width_10_20"]) == pytest.approx(
        place_width_t10_h1 - place_width_t20_h1
    )

    assert float(horse2["place_mid_prob_t10"]) == pytest.approx(1.0 / math.sqrt(2.4 * 3.0))
    assert float(horse2["place_width_log_ratio_t10"]) == pytest.approx(math.log(3.0 / 2.4))
