"""v3 共通ユーティリティ.

v2 から移植した関数 + v3 内で重複していたヘルパーを集約する。
CLI引数・出力形式は v2 と同一を維持。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# パス / ファイル ユーティリティ
# ---------------------------------------------------------------------------


def resolve_path(path_str: str) -> Path:
    """相対パスなら PROJECT_ROOT 基準で解決する。"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """JSON をインデント付きで書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_files(paths: list[Path]) -> str:
    """複数ファイルの SHA-256 ダイジェストを返す。"""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Rolling 年次 CV（v2 train_ranker_v2 由来）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_years: tuple[int, ...]
    valid_year: int


def build_rolling_year_folds(
    years: list[int],
    *,
    train_window_years: int,
    holdout_year: int,
) -> list[FoldSpec]:
    """Rolling 年次 CV の fold を生成する。"""
    if train_window_years <= 0:
        raise ValueError("train_window_years must be > 0")
    trainable_years = sorted(y for y in years if y < holdout_year)
    if len(trainable_years) < train_window_years + 1:
        raise ValueError(
            "Not enough non-holdout years for rolling folds: "
            f"need >= {train_window_years + 1}, got {len(trainable_years)}"
        )
    folds: list[FoldSpec] = []
    for idx in range(train_window_years, len(trainable_years)):
        fold = FoldSpec(
            fold_id=len(folds) + 1,
            train_years=tuple(trainable_years[idx - train_window_years : idx]),
            valid_year=trainable_years[idx],
        )
        folds.append(fold)
    return folds


def assert_fold_integrity(train_df: pd.DataFrame, valid_df: pd.DataFrame, valid_year: int) -> None:
    """train/valid 間の時系列リーク・レースID重複を検出する。"""
    train_year_max = int(train_df["year"].max())
    if train_year_max >= valid_year:
        raise ValueError(
            f"Temporal leakage detected: train max year={train_year_max}, valid year={valid_year}"
        )
    overlap = set(train_df["race_id"].unique()) & set(valid_df["race_id"].unique())
    if overlap:
        raise ValueError(f"Race leakage detected across train/valid: {len(overlap)} races")


# ---------------------------------------------------------------------------
# ソート検証（v2 build_features_v2 由来）
# ---------------------------------------------------------------------------


def assert_sorted(df: pd.DataFrame) -> None:
    """race_id asc / horse_no asc のソート順を検証する。"""
    sorted_df = df.sort_values(["race_id", "horse_no"], kind="mergesort")
    left = df[["race_id", "horse_no"]].reset_index(drop=True)
    right = sorted_df[["race_id", "horse_no"]].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Output sort violation: expected race_id asc / horse_no asc.")


# ---------------------------------------------------------------------------
# レース日時構築（v2 build_features_v2 由来）
# ---------------------------------------------------------------------------


def _time_to_seconds(value: object) -> int:
    """時刻を秒に変換する。パース不能な場合は 23:59:00 を返す。"""
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if value is None or pd.isna(value):
        return 23 * 3600 + 59 * 60
    text = str(value).strip()
    if not text:
        return 23 * 3600 + 59 * 60
    parts = text.split(":")
    if len(parts) >= 2:
        try:
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) >= 3 else 0
            return hour * 3600 + minute * 60 + second
        except ValueError:
            return 23 * 3600 + 59 * 60
    return 23 * 3600 + 59 * 60


def build_race_datetime(
    race_date_series: pd.Series,
    start_time_series: pd.Series,
) -> pd.Series:
    """race_date + start_time から race_datetime を構築する。"""
    race_date = pd.to_datetime(race_date_series, errors="coerce")
    start_seconds = start_time_series.map(_time_to_seconds)
    return race_date + pd.to_timedelta(start_seconds, unit="s")


# ---------------------------------------------------------------------------
# 組番（v2 wide_prob_v2_common 由来）
# ---------------------------------------------------------------------------


def kumiban_from_horse_nos(horse_no_1: int, horse_no_2: int) -> str:
    """2頭の馬番からワイド組番文字列を生成する。"""
    left, right = sorted((int(horse_no_1), int(horse_no_2)))
    return f"{left:02d}{right:02d}"


# ---------------------------------------------------------------------------
# Bankroll / Kelly（v2 bankroll_v2_common 由来）
# ---------------------------------------------------------------------------


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
    """金額を unit 単位に切り捨てる。"""
    if unit <= 0:
        raise ValueError("unit must be > 0")
    if not np.isfinite(amount) or amount <= 0:
        return 0
    return int(np.floor(float(amount) / float(unit)) * float(unit))


def kelly_fraction(probability: float, odds: float) -> float:
    """フル Kelly fraction を計算する。"""
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
    """フラクショナル Kelly fraction を計算する。"""
    if scale <= 0:
        return 0.0
    return float(max(float(scale), 0.0) * kelly_fraction(probability, odds))


def allocate_race_bets(
    candidates: pd.DataFrame,
    *,
    bankroll_yen: int,
    config: BankrollConfig,
) -> pd.DataFrame:
    """レース内の候補から Kelly 基準でベット額を配分する。"""
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


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """エクイティカーブから最大ドローダウンを計算する。"""
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


__all__ = [
    "PROJECT_ROOT",
    # パス / ファイル
    "resolve_path",
    "save_json",
    "hash_files",
    # CV
    "FoldSpec",
    "build_rolling_year_folds",
    "assert_fold_integrity",
    # ソート
    "assert_sorted",
    # 日時
    "build_race_datetime",
    # 組番
    "kumiban_from_horse_nos",
    # Bankroll
    "BankrollConfig",
    "round_down_to_unit",
    "kelly_fraction",
    "fractional_kelly_fraction",
    "allocate_race_bets",
    "compute_max_drawdown",
]
