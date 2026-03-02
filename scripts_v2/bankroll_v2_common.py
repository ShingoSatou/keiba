from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BankrollConfig:
    bankroll_init_yen: int = 1_000_000
    kelly_fraction_scale: float = 0.25
    max_bets_per_race: int = 5
    race_cap_fraction: float = 0.05
    daily_cap_fraction: float = 0.2
    bet_unit_yen: int = 100
    min_bet_yen: int = 100
    max_bet_yen: int | None = None


def round_down_to_unit(amount: float, unit: int) -> int:
    if unit <= 0:
        raise ValueError("unit must be > 0")
    if not np.isfinite(amount) or amount <= 0:
        return 0
    return int(np.floor(float(amount) / float(unit)) * float(unit))


def kelly_fraction(probability: float, odds: float) -> float:
    p = float(probability)
    o = float(odds)
    if not np.isfinite(p) or not np.isfinite(o):
        return 0.0
    if p <= 0.0 or p >= 1.0:
        return 0.0
    if o <= 1.0:
        return 0.0
    b = o - 1.0
    value = (p * o - 1.0) / b
    return float(max(value, 0.0))


def fractional_kelly_fraction(probability: float, odds: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(max(float(scale), 0.0) * kelly_fraction(probability, odds))


def allocate_race_bets(
    candidates: pd.DataFrame,
    *,
    bankroll_yen: int,
    config: BankrollConfig,
) -> pd.DataFrame:
    required = {"p_wide", "odds", "ev_profit"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing required candidate columns: {missing}")
    if bankroll_yen <= 0:
        return pd.DataFrame(columns=list(candidates.columns) + ["kelly_f", "bet_yen"])

    work = candidates.copy()
    for column in ("p_wide", "odds", "ev_profit"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["p_wide", "odds", "ev_profit"]).copy()
    if work.empty:
        return pd.DataFrame(columns=list(candidates.columns) + ["kelly_f", "bet_yen"])

    work = work.sort_values(["ev_profit", "p_wide"], ascending=[False, False], kind="mergesort")
    if config.max_bets_per_race > 0:
        work = work.head(int(config.max_bets_per_race)).copy()
    if work.empty:
        return pd.DataFrame(columns=list(candidates.columns) + ["kelly_f", "bet_yen"])

    work["kelly_f"] = [
        fractional_kelly_fraction(p, o, config.kelly_fraction_scale)
        for p, o in zip(work["p_wide"], work["odds"], strict=False)
    ]
    work["bet_raw"] = work["kelly_f"] * float(bankroll_yen)
    if config.max_bet_yen is not None and config.max_bet_yen > 0:
        work["bet_raw"] = np.minimum(work["bet_raw"], float(config.max_bet_yen))

    work["bet_yen"] = work["bet_raw"].map(
        lambda value: round_down_to_unit(float(value), int(config.bet_unit_yen))
    )
    work = work[work["bet_yen"] >= int(config.min_bet_yen)].copy()
    if work.empty:
        return pd.DataFrame(columns=list(candidates.columns) + ["kelly_f", "bet_yen"])

    race_cap_yen = round_down_to_unit(
        float(bankroll_yen) * float(config.race_cap_fraction),
        int(config.bet_unit_yen),
    )
    if race_cap_yen <= 0:
        return pd.DataFrame(columns=list(candidates.columns) + ["kelly_f", "bet_yen"])

    total_bet = int(work["bet_yen"].sum())
    if total_bet > race_cap_yen:
        scale = float(race_cap_yen) / float(total_bet)
        work["bet_yen"] = work["bet_yen"].map(
            lambda value: round_down_to_unit(float(value) * scale, int(config.bet_unit_yen))
        )
        work = work[work["bet_yen"] >= int(config.min_bet_yen)].copy()

    if work.empty:
        return pd.DataFrame(columns=list(candidates.columns) + ["kelly_f", "bet_yen"])

    work["bet_yen"] = work["bet_yen"].astype(int)
    work = work.drop(columns=["bet_raw"])
    return work.reset_index(drop=True)


def apply_daily_cap(
    bet_df: pd.DataFrame,
    *,
    bankroll_yen: int,
    daily_cap_fraction: float,
    bet_unit_yen: int,
    min_bet_yen: int,
) -> pd.DataFrame:
    if bet_df.empty:
        return bet_df.copy()
    cap_yen = round_down_to_unit(float(bankroll_yen) * float(daily_cap_fraction), int(bet_unit_yen))
    if cap_yen <= 0:
        return bet_df.iloc[0:0].copy()

    total_bet = int(pd.to_numeric(bet_df["bet_yen"], errors="coerce").fillna(0).sum())
    if total_bet <= cap_yen:
        return bet_df.copy()

    scale = float(cap_yen) / float(total_bet)
    out = bet_df.copy()
    out["bet_yen"] = (
        pd.to_numeric(out["bet_yen"], errors="coerce")
        .fillna(0.0)
        .map(lambda value: round_down_to_unit(float(value) * scale, int(bet_unit_yen)))
    )
    out = out[out["bet_yen"] >= int(min_bet_yen)].copy()
    out["bet_yen"] = out["bet_yen"].astype(int)
    return out.reset_index(drop=True)


def compute_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = float("-inf")
    max_dd = 0.0
    for value in equity_curve:
        current = float(value)
        if current > peak:
            peak = current
        drawdown = peak - current
        if drawdown > max_dd:
            max_dd = drawdown
    return float(max_dd)
