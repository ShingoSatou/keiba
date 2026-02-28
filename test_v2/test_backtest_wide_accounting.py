from __future__ import annotations

import pandas as pd

from scripts_v2.backtest_wide_v2 import apply_remaining_daily_cap
from scripts_v2.bankroll_v2_common import (
    BankrollConfig,
    allocate_race_bets,
    compute_max_drawdown,
)


def test_allocate_race_bets_respects_race_cap_and_rounding():
    candidates = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "horse_no_1": [1, 1, 2],
            "horse_no_2": [2, 3, 3],
            "kumiban": ["0102", "0103", "0203"],
            "p_wide": [0.25, 0.19, 0.14],
            "odds": [8.5, 12.4, 15.0],
            "ev_profit": [1.125, 1.356, 1.10],
        }
    )
    cfg = BankrollConfig(
        bankroll_init_yen=100_000,
        kelly_fraction_scale=0.25,
        max_bets_per_race=3,
        race_cap_fraction=0.03,  # 3,000 yen
        daily_cap_fraction=0.20,
        bet_unit_yen=100,
        min_bet_yen=100,
        max_bet_yen=None,
    )

    out = allocate_race_bets(candidates, bankroll_yen=100_000, config=cfg)

    assert not out.empty
    assert (out["bet_yen"] % 100 == 0).all()
    assert (out["bet_yen"] >= 100).all()
    assert int(out["bet_yen"].sum()) <= 3000


def test_apply_remaining_daily_cap_scales_down_total_bet():
    race_bets = pd.DataFrame(
        {
            "kumiban": ["0102", "0103", "0203"],
            "bet_yen": [1000, 1000, 1000],
        }
    )

    capped = apply_remaining_daily_cap(
        race_bets,
        remaining_cap_yen=1500,
        bet_unit_yen=100,
        min_bet_yen=100,
    )

    assert not capped.empty
    assert int(capped["bet_yen"].sum()) <= 1500
    assert (capped["bet_yen"] % 100 == 0).all()
    assert (capped["bet_yen"] >= 100).all()


def test_accounting_invariants_and_drawdown():
    bets = pd.DataFrame(
        {
            "bet_yen": [100, 200, 300],
            "payout": [0, 700, 0],
            "is_hit": [False, True, False],
        }
    )
    total_bet = int(bets["bet_yen"].sum())
    total_return = int(bets["payout"].sum())
    total_profit = int((bets["payout"] - bets["bet_yen"]).sum())
    roi = total_return / total_bet

    assert total_bet == 600
    assert total_return == 700
    assert total_profit == 100
    assert roi == total_return / total_bet
    assert total_profit == total_return - total_bet

    equity_curve = [1_000_000, 999_000, 1_000_200, 999_100, 1_001_000]
    assert compute_max_drawdown(equity_curve) == 1100.0
