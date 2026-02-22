from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from scripts import market_features


class _DummyDB:
    pass


def test_add_market_features_builds_12_points_and_derivatives(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "race_id": 202602030501,
                "horse_no": 1,
                "asof_ts": datetime(2026, 2, 3, 12, 30),
                "odds_win_t5": 4.0,
                "odds_rank_t5": 1,
                "odds_missing_flag": False,
                "win_pool_total_100yen_t5": 1200,
            }
        ]
    )

    def _fake_fetch_points(db, source, minutes, chunk_size=400):
        row = {"row_idx": 0, "M_win_pool_total_tminus_60": 800}
        for minute in minutes:
            row[f"M_odds_tminus_{minute}"] = 10.0 - minute / 10.0
        return pd.DataFrame([row])

    monkeypatch.setattr(market_features, "_fetch_market_points", _fake_fetch_points)

    enriched = market_features.add_market_features(_DummyDB(), frame)

    assert "M_odds_tminus_60" in enriched.columns
    assert "M_odds_tminus_5" in enriched.columns
    assert "M_odds_slope_log" in enriched.columns
    assert "M_win_pool_growth_60to5" in enriched.columns
    assert np.isfinite(float(enriched.loc[0, "M_odds_slope_log"]))
    assert float(enriched.loc[0, "M_win_pool_growth_60to5"]) == 400.0
