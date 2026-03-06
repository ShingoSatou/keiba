from __future__ import annotations

from scripts_v3.cv_policy_v3 import build_fixed_window_year_folds
from scripts_v3.v3_common import build_rolling_year_folds


def test_fixed_window_year_folds_uses_previous_four_years_only() -> None:
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    folds = build_fixed_window_year_folds(years, window_years=4, holdout_year=2025)

    target = next(fold for fold in folds if fold.valid_year == 2024)
    assert tuple(target.train_years) == (2020, 2021, 2022, 2023)


def test_fixed_window_year_folds_is_not_expanding() -> None:
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    folds = build_fixed_window_year_folds(years, window_years=4, holdout_year=2025)

    target = next(fold for fold in folds if fold.valid_year == 2024)
    assert 2019 not in target.train_years
    assert len(target.train_years) == 4


def test_fixed_window_year_folds_skips_years_without_enough_history() -> None:
    years = [2021, 2022, 2023, 2024]
    folds = build_fixed_window_year_folds(years, window_years=4, holdout_year=2025)

    assert folds == []


def test_fixed_window_year_folds_excludes_holdout_year_from_cv() -> None:
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    folds = build_fixed_window_year_folds(years, window_years=4, holdout_year=2025)

    assert [fold.valid_year for fold in folds] == [2024]
    assert all(2025 not in fold.train_years for fold in folds)


def test_build_rolling_year_folds_alias_matches_fixed_window_helper() -> None:
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    fixed = build_fixed_window_year_folds(years, window_years=4, holdout_year=2025)
    alias = build_rolling_year_folds(years, train_window_years=4, holdout_year=2025)

    assert alias == fixed
