from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts_v2.build_features_v2 import (
    _add_lag_features,
    _apply_segment_filter,
    _compute_recent_entity_target_mean,
    _time_window_stats_by_group,
    assert_no_future_leakage,
    assert_sorted,
)


def test_rolling_stats_excludes_current_row():
    df = pd.DataFrame(
        {
            "race_datetime": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-02 12:00:00"]),
            "track_code": [1, 1],
            "surface": [2, 2],
            "distance_bucket": [1800.0, 1800.0],
            "going_bucket": [1.0, 1.0],
            "time_sec": [100.0, 110.0],
        }
    )
    stats = _time_window_stats_by_group(
        df,
        group_cols=["track_code", "surface", "distance_bucket", "going_bucket"],
        value_col="time_sec",
        prefix="fine",
        window_days=730,
    )
    assert np.isnan(stats.loc[0, "fine_mean"])
    assert stats.loc[1, "fine_mean"] == pytest.approx(100.0)
    assert stats.loc[1, "fine_count"] == pytest.approx(1.0)


def test_lag_features_are_strictly_past():
    df = pd.DataFrame(
        {
            "horse_id": ["H1", "H1", "H2"],
            "race_id": [1, 2, 3],
            "race_datetime": pd.to_datetime(
                ["2024-01-01 12:00:00", "2024-01-05 12:00:00", "2024-01-03 12:00:00"]
            ),
            "finish_pos": [5, 2, 1],
            "time_sec": [101.0, 99.0, 98.0],
            "final3f_sec": [36.0, 35.0, 34.0],
            "perf_speed_index": [70.0, 85.0, 90.0],
            "perf_up3_index": [72.0, 86.0, 88.0],
            "distance_m": [1800, 1800, 1800],
            "surface": [2, 2, 2],
            "jockey_key": [1001, 1002, 1003],
        }
    )
    out = _add_lag_features(df)
    row = out[out["race_id"] == 2].iloc[0]
    assert row["lag1_speed_index"] == pytest.approx(70.0)
    assert row["lag1_up3_index"] == pytest.approx(72.0)
    assert row["days_since_lag1"] == 4
    assert row["is_jockey_change"] == 1
    assert_no_future_leakage(out)


def test_leakage_guard_raises_on_future_lag():
    df = pd.DataFrame(
        {
            "race_datetime": pd.to_datetime(["2024-01-02 12:00:00"]),
            "lag1_race_datetime": pd.to_datetime(["2024-01-03 12:00:00"]),
        }
    )
    with pytest.raises(ValueError):
        assert_no_future_leakage(df)


def test_segment_filter_keeps_only_expected_rows():
    df = pd.DataFrame(
        {
            "track_code": [1, 1, 12],
            "surface": [2, 1, 2],
            "race_type_code": [14, 14, 14],
            "condition_code_min_age": [10, 10, 10],
            "distance_m": [1800, 1800, 1800],
            "field_size": [16, 16, 16],
            "horse_no": [1, 1, 1],
            "finish_pos": [1, 1, 1],
        }
    )
    filtered = _apply_segment_filter(df)
    assert len(filtered) == 1
    assert int(filtered.iloc[0]["track_code"]) == 1
    assert int(filtered.iloc[0]["surface"]) == 2


def test_sorted_guard_detects_order_violation():
    ok = pd.DataFrame({"race_id": [1, 1, 2], "horse_no": [1, 2, 1]})
    assert_sorted(ok)

    ng = pd.DataFrame({"race_id": [1, 1, 2], "horse_no": [2, 1, 1]})
    with pytest.raises(ValueError):
        assert_sorted(ng)


def test_target_encoding_excludes_current_date():
    df = pd.DataFrame(
        {
            "race_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ).date,
            "jockey_key": [100, 100, 100, 200],
            "target_label": [3, 0, 3, 0],
        }
    )
    out = _compute_recent_entity_target_mean(
        df,
        "jockey_key",
        "target_label",
        "jockey_target_label_mean_6m",
        prior_mean=0.5,
    )

    # same day should not see its own target_label (closed='left' on date)
    day1 = out[out["race_date"] == pd.to_datetime("2024-01-01").date()]
    assert day1["jockey_target_label_mean_6m"].to_list() == pytest.approx([0.5, 0.5])

    # next day should incorporate day1's mean for jockey_key=100
    day2_j100 = out[
        (out["race_date"] == pd.to_datetime("2024-01-02").date()) & (out["jockey_key"] == 100)
    ]
    assert len(day2_j100) == 1
    assert float(day2_j100.iloc[0]["jockey_target_label_mean_6m"]) > 0.5
