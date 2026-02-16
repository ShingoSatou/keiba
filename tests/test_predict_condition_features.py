from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from scripts.predict import _add_condition_features


class _DummyDB:
    def fetch_all(self, query, params):
        assert "prev_race_date" in query
        assert params[0] == date(2026, 2, 8)
        assert params[-1] == date(2026, 2, 8)
        assert set(params[1:-1]) == {101, 102}
        return [
            {
                "horse_id": 101,
                "prev_race_date": date(2026, 2, 1),
                "prev_distance_m": Decimal("1600"),
            },
            {
                "horse_id": 102,
                "prev_race_date": "2026-02-05",
                "prev_distance_m": 1400,
            },
        ]


def test_add_condition_features_handles_date_like_prev_race_date_and_decimal_distance():
    frame = pd.DataFrame(
        [
            {"horse_id": 101, "distance_m": 1400},
            {"horse_id": 102, "distance_m": 1200},
        ]
    )
    out = _add_condition_features(_DummyDB(), frame, race_date=date(2026, 2, 8))

    assert out.loc[out["horse_id"] == 101, "days_since_last"].iloc[0] == 7
    assert out.loc[out["horse_id"] == 102, "days_since_last"].iloc[0] == 3
    assert out.loc[out["horse_id"] == 101, "distance_change_m"].iloc[0] == -200
    assert out.loc[out["horse_id"] == 102, "distance_change_m"].iloc[0] == -200
