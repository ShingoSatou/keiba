from __future__ import annotations

import pandas as pd

from scripts.train_walk_forward import _build_outer_folds


def test_build_outer_folds_uses_holdout_year():
    rows = []
    for year in range(2016, 2024):
        rows.append(
            {
                "race_id": year * 10000 + 101,
                "horse_no": 1,
                "race_date": f"{year}-01-10",
                "year": year,
                "is_win": 1,
            }
        )
    frame = pd.DataFrame(rows)
    folds = _build_outer_folds(frame, min_train_years=3, holdout_years=1)

    assert folds[0]["valid_year"] == 2019
    assert folds[-1]["valid_year"] == 2022
