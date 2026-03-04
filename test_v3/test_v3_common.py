from pathlib import Path

import pandas as pd
import pytest

from scripts_v3.v3_common import (
    BankrollConfig,
    allocate_race_bets,
    assert_fold_integrity,
    assert_sorted,
    build_rolling_year_folds,
    compute_max_drawdown,
    kumiban_from_horse_nos,
    resolve_path,
)


def test_build_rolling_year_folds_basic() -> None:
    years = [2021, 2022, 2023, 2024, 2025]
    folds = build_rolling_year_folds(years, train_window_years=2, holdout_year=2025)
    # Fold 1: train=[2021, 2022], valid=2023
    # Fold 2: train=[2022, 2023], valid=2024
    assert len(folds) == 2
    assert tuple(folds[0].train_years) == (2021, 2022)
    assert folds[0].valid_year == 2023
    assert tuple(folds[1].train_years) == (2022, 2023)
    assert folds[1].valid_year == 2024


def test_fold_integrity_detects_leakage() -> None:
    train_df = pd.DataFrame({"year": [2021, 2022], "race_id": [1, 2]})
    valid_df = pd.DataFrame({"year": [2023, 2021], "race_id": [3, 4]})
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        # max train_year(2022) >= valid_year(2022) -> error expected
        assert_fold_integrity(train_df, valid_df, 2022)


def test_assert_sorted_raises_on_unsorted() -> None:
    df = pd.DataFrame({"race_id": [2, 1], "horse_no": [1, 2]})
    with pytest.raises(ValueError, match="Output sort violation"):
        assert_sorted(df)


def test_kumiban_from_horse_nos() -> None:
    assert kumiban_from_horse_nos(3, 5) == "0305"
    assert kumiban_from_horse_nos(10, 1) == "0110"
    assert kumiban_from_horse_nos(1, 1) == "0101"


def test_resolve_path_relative_and_absolute() -> None:
    # absolute path remains unchanged
    abs_path = Path("/tmp/test.txt")
    assert resolve_path(abs_path) == abs_path

    # relative path is resolved relative to PROJECT_ROOT
    rel_path = "data/test.txt"
    resolved = resolve_path(rel_path)
    assert str(resolved).endswith("data/test.txt")
    assert resolved.is_absolute()


def test_bankroll_config_and_kelly() -> None:
    config = BankrollConfig(
        bankroll_init_yen=100_000,
        kelly_fraction_scale=0.25,
        max_bets_per_race=2,
        race_cap_fraction=0.05,  # 5000 yen race cap
        daily_cap_fraction=0.20,
        bet_unit_yen=100,
        min_bet_yen=100,
        max_bet_yen=None,
    )
    # Dummy candidates:
    # row 1: p_wide = 0.5, odds = 3.0 -> ev_profit = 0.5
    # row 2: p_wide = 0.2, odds = 10.0 -> ev_profit = 1.0 (better EV)
    # row 3: p_wide = 0.1, odds = 5.0 -> ev_profit = -0.5
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "kumiban": ["01-02", "03-04", "05-06"],
            "horse_no_1": [1, 3, 5],
            "horse_no_2": [2, 4, 6],
            "p_wide": [0.5, 0.2, 0.1],
            "odds": [3.0, 10.0, 5.0],
            "ev_profit": [0.5, 1.0, -0.5],
        }
    )
    allocated = allocate_race_bets(df, bankroll_yen=100_000, config=config)
    assert len(allocated) <= config.max_bets_per_race
    assert allocated["bet_yen"].sum() <= 100_000 * config.race_cap_fraction
    assert (allocated["bet_yen"] % config.bet_unit_yen == 0).all()


def test_compute_max_drawdown() -> None:
    equity = [100.0, 120.0, 90.0, 110.0, 80.0, 130.0]
    dd = compute_max_drawdown(equity)
    # max drawdowns:
    # 120 -> 90 = 30
    # 120 -> 80 = 40 (max)
    assert dd == 40.0
