from __future__ import annotations

import pandas as pd

from scripts_v2.train_ranker_v2 import build_rolling_year_folds


def _mock_frame(years: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for year in years:
        for race_idx in (1, 2):
            race_id = year * 100 + race_idx
            for horse_no in (1, 2, 3):
                rows.append(
                    {
                        "race_id": race_id,
                        "year": year,
                        "horse_no": horse_no,
                    }
                )
    return pd.DataFrame(rows)


def test_rolling_window_fold_definition():
    folds = build_rolling_year_folds(
        [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        train_window_years=5,
        holdout_year=2025,
    )
    assert [fold.valid_year for fold in folds] == [2021, 2022, 2023, 2024]
    assert folds[0].train_years == (2016, 2017, 2018, 2019, 2020)
    assert folds[-1].train_years == (2019, 2020, 2021, 2022, 2023)


def test_group_split_has_no_race_overlap():
    frame = _mock_frame([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    folds = build_rolling_year_folds(
        sorted(frame["year"].unique().tolist()),
        train_window_years=5,
        holdout_year=2025,
    )
    for fold in folds:
        train_df = frame[frame["year"].isin(fold.train_years)]
        valid_df = frame[frame["year"] == fold.valid_year]
        train_races = set(train_df["race_id"].unique())
        valid_races = set(valid_df["race_id"].unique())
        assert train_races.isdisjoint(valid_races)


def test_no_temporal_leakage_in_fold_order():
    frame = _mock_frame([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    folds = build_rolling_year_folds(
        sorted(frame["year"].unique().tolist()),
        train_window_years=5,
        holdout_year=2025,
    )
    for fold in folds:
        train_df = frame[frame["year"].isin(fold.train_years)]
        assert int(train_df["year"].max()) < fold.valid_year


def test_holdout_year_and_later_are_not_used_in_folds():
    folds = build_rolling_year_folds(
        [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        train_window_years=5,
        holdout_year=2025,
    )
    used_years = {fold.valid_year for fold in folds}
    for fold in folds:
        used_years.update(fold.train_years)
    assert 2025 not in used_years
    assert 2026 not in used_years
