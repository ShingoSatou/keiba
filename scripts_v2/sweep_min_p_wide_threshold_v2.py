#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from scripts_v2.bankroll_v2_common import (  # noqa: E402
    BankrollConfig,
    allocate_race_bets,
    compute_max_drawdown,
    round_down_to_unit,
)
from scripts_v2.wide_prob_v2_common import (  # noqa: E402
    PLSamplingConfig,
    estimate_wide_probabilities_for_race,
    make_race_rng,
)

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_SEED = 42
MIN_P_WIDE_STAGE_CHOICES = ("candidate", "selected")


@dataclass(frozen=True)
class SweepResult:
    min_p_wide: float
    stage: str
    roi: float
    n_bets: int
    n_hits: int
    total_bet: int
    total_return: int
    max_drawdown: int
    roi_by_year: dict[int, float]
    n_bets_by_year: dict[int, int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep min p_wide threshold on OOF years (v2).")
    parser.add_argument("--input", default="data/oof/top3_convex_oof_cw_none.parquet")
    parser.add_argument(
        "--years",
        default="",
        help="Comma-separated valid_year filter (e.g., '2023,2024'). Empty=all available.",
    )
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--require-years", default="")

    parser.add_argument("--mc-samples", type=int, default=10_000)
    parser.add_argument("--pl-top-k", type=int, default=3)
    parser.add_argument("--pl-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--ev-threshold", type=float, default=0.0)
    parser.add_argument("--max-bets-per-race", type=int, default=5)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--race-cap-fraction", type=float, default=0.05)
    parser.add_argument("--daily-cap-fraction", type=float, default=0.20)
    parser.add_argument("--bankroll-init-yen", type=int, default=1_000_000)
    parser.add_argument("--bet-unit-yen", type=int, default=100)
    parser.add_argument("--min-bet-yen", type=int, default=100)
    parser.add_argument("--max-bet-yen", type=int, default=0)

    parser.add_argument(
        "--min-p-wide-stage",
        choices=list(MIN_P_WIDE_STAGE_CHOICES),
        default="selected",
        help="Where to apply min_p_wide threshold.",
    )
    parser.add_argument(
        "--grid",
        default="",
        help="Comma-separated threshold grid (overrides --grid-start/stop/step).",
    )
    parser.add_argument("--grid-start", type=float, default=0.0)
    parser.add_argument("--grid-stop", type=float, default=0.15)
    parser.add_argument("--grid-step", type=float, default=0.01)

    parser.add_argument("--output", default="data/oof/min_p_wide_sweep.json")
    parser.add_argument("--meta-output", default="data/oof/min_p_wide_sweep_meta.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_years(raw: str) -> list[int]:
    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def check_overwrite(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists. pass --force to overwrite: {joined}")


def load_top3_oof(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    frame = pd.read_parquet(path)

    required = {"race_id", "horse_id", "horse_no", "valid_year", "p_top3"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}")

    out = frame.copy()
    out["race_id"] = pd.to_numeric(out["race_id"], errors="coerce").astype("Int64")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out["valid_year"] = pd.to_numeric(out["valid_year"], errors="coerce").astype("Int64")
    out["p_top3"] = pd.to_numeric(out["p_top3"], errors="coerce")
    out = out.dropna(subset=["race_id", "horse_no", "valid_year", "p_top3"]).copy()
    if out.empty:
        raise SystemExit("Input OOF has no valid rows after coercion.")

    out["race_id"] = out["race_id"].astype(int)
    out["horse_no"] = out["horse_no"].astype(int)
    out["valid_year"] = out["valid_year"].astype(int)
    out["horse_id"] = out["horse_id"].astype(str)
    out = out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)
    return out


def select_years(
    frame: pd.DataFrame,
    *,
    holdout_year: int,
    years_arg: str,
    require_years_arg: str,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    base = frame[frame["valid_year"] < int(holdout_year)].copy()
    if base.empty:
        raise SystemExit(f"No rows remain after holdout filter (valid_year < {int(holdout_year)}).")
    available = sorted(base["valid_year"].unique().tolist())

    required_years = parse_years(require_years_arg)
    if required_years:
        missing = sorted(set(required_years) - set(available))
        if missing:
            raise SystemExit(
                f"required years are missing after holdout filter: {missing}, available={available}"
            )

    selected_years = parse_years(years_arg) if years_arg.strip() else available
    missing_selected = sorted(set(selected_years) - set(available))
    if missing_selected:
        raise SystemExit(
            f"selected years not found in input: {missing_selected}, available={available}"
        )

    out = base[base["valid_year"].isin(selected_years)].copy()
    if out.empty:
        raise SystemExit("No rows left after year selection.")
    return out, selected_years, available


def _build_grid(args: argparse.Namespace) -> list[float]:
    if args.grid.strip():
        values = [float(token.strip()) for token in args.grid.split(",") if token.strip()]
    else:
        start = float(args.grid_start)
        stop = float(args.grid_stop)
        step = float(args.grid_step)
        if step <= 0:
            raise SystemExit("--grid-step must be > 0")
        if stop < start:
            raise SystemExit("--grid-stop must be >= --grid-start")
        count = int(np.floor((stop - start) / step)) + 1
        values = [start + step * idx for idx in range(count)]

    cleaned: list[float] = []
    for value in values:
        if not (0.0 <= float(value) <= 1.0):
            raise SystemExit(f"grid value out of [0,1]: {value}")
        cleaned.append(float(round(float(value), 6)))
    return sorted(set(cleaned))


def apply_remaining_daily_cap(
    race_bets: pd.DataFrame,
    *,
    remaining_cap_yen: int,
    bet_unit_yen: int,
    min_bet_yen: int,
) -> pd.DataFrame:
    if race_bets.empty:
        return race_bets.copy()
    if remaining_cap_yen <= 0:
        return race_bets.iloc[0:0].copy()

    total_bet = int(race_bets["bet_yen"].sum())
    if total_bet <= remaining_cap_yen:
        return race_bets.copy()

    scale = float(remaining_cap_yen) / float(total_bet)
    out = race_bets.copy()
    out["bet_yen"] = out["bet_yen"].map(
        lambda value: round_down_to_unit(float(value) * scale, int(bet_unit_yen))
    )
    out = out[out["bet_yen"] >= int(min_bet_yen)].copy()
    if out.empty:
        return out
    out["bet_yen"] = out["bet_yen"].astype(int)
    return out.reset_index(drop=True)


def fetch_odds_payout_race(
    db: Database,
    race_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    odds_rows = db.fetch_all(
        """
        SELECT DISTINCT ON (w.race_id, w.kumiban)
            w.race_id,
            w.kumiban,
            w.min_odds_x10,
            w.data_kbn,
            w.announce_mmddhhmi
        FROM core.o3_wide AS w
        WHERE w.race_id = ANY(%(race_ids)s)
          AND w.min_odds_x10 IS NOT NULL
          AND w.min_odds_x10 > 0
        ORDER BY
            w.race_id,
            w.kumiban,
            w.data_kbn DESC,
            w.announce_mmddhhmi DESC
        """,
        {"race_ids": race_ids},
    )
    payout_rows = db.fetch_all(
        """
        SELECT
            p.race_id,
            p.selection AS kumiban,
            p.payout_yen
        FROM core.payout AS p
        WHERE p.bet_type = 5
          AND p.race_id = ANY(%(race_ids)s)
        """,
        {"race_ids": race_ids},
    )
    race_rows = db.fetch_all(
        """
        SELECT race_id, race_date
        FROM core.race
        WHERE race_id = ANY(%(race_ids)s)
        """,
        {"race_ids": race_ids},
    )
    return pd.DataFrame(odds_rows), pd.DataFrame(payout_rows), pd.DataFrame(race_rows)


def _prepare_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return pd.DataFrame(columns=["race_id", "kumiban", "odds"])
    work = odds_df.copy()
    work["race_id"] = pd.to_numeric(work["race_id"], errors="coerce").astype("Int64")
    work["min_odds_x10"] = pd.to_numeric(work["min_odds_x10"], errors="coerce")
    work = work.dropna(subset=["race_id", "kumiban", "min_odds_x10"]).copy()
    work["race_id"] = work["race_id"].astype(int)
    work["kumiban"] = work["kumiban"].astype(str)
    work["odds"] = work["min_odds_x10"].astype(float) / 10.0
    work = work[work["odds"] > 1.0].copy()
    return work[["race_id", "kumiban", "odds"]].reset_index(drop=True)


def _prepare_payout(payout_df: pd.DataFrame) -> pd.DataFrame:
    if payout_df.empty:
        return pd.DataFrame(columns=["race_id", "kumiban", "payout_yen"])
    work = payout_df.copy()
    work["race_id"] = pd.to_numeric(work["race_id"], errors="coerce").astype("Int64")
    work["payout_yen"] = pd.to_numeric(work["payout_yen"], errors="coerce")
    work = work.dropna(subset=["race_id", "kumiban", "payout_yen"]).copy()
    work["race_id"] = work["race_id"].astype(int)
    work["kumiban"] = work["kumiban"].astype(str)
    work["payout_yen"] = work["payout_yen"].astype(int)
    return work[["race_id", "kumiban", "payout_yen"]].reset_index(drop=True)


def _prepare_race_dates(race_df: pd.DataFrame) -> dict[int, datetime.date]:
    if race_df.empty:
        return {}
    work = race_df.copy()
    work["race_id"] = pd.to_numeric(work["race_id"], errors="coerce").astype("Int64")
    work["race_date"] = pd.to_datetime(work["race_date"], errors="coerce")
    work = work.dropna(subset=["race_id", "race_date"]).copy()
    work["race_id"] = work["race_id"].astype(int)
    return {int(r["race_id"]): r["race_date"].date() for r in work.to_dict("records")}


def build_candidate_table(
    selected_oof: pd.DataFrame,
    *,
    odds_df: pd.DataFrame,
    payout_df: pd.DataFrame,
    race_date_map: dict[int, datetime.date],
    pl_config: PLSamplingConfig,
) -> pd.DataFrame:
    odds_by_race = {int(rid): sub for rid, sub in odds_df.groupby("race_id", sort=False)}
    payout_key = ["race_id", "kumiban"]
    payout_df = payout_df[payout_key + ["payout_yen"]].copy()

    race_meta = selected_oof[["race_id", "valid_year"]].drop_duplicates().copy()
    race_meta["race_date"] = race_meta["race_id"].map(race_date_map)
    race_meta["race_date"] = pd.to_datetime(race_meta["race_date"], errors="coerce")
    race_meta = race_meta.dropna(subset=["race_date"]).copy()
    race_meta = race_meta.sort_values(["race_date", "race_id"], kind="mergesort").reset_index(
        drop=True
    )

    candidate_rows: list[pd.DataFrame] = []
    for race in race_meta.to_dict("records"):
        race_id = int(race["race_id"])
        if race_id not in odds_by_race:
            continue
        race_input = selected_oof[selected_oof["race_id"] == race_id][
            ["race_id", "horse_no", "p_top3"]
        ].copy()
        if race_input.empty:
            continue

        rng = make_race_rng(pl_config.seed, race_id)
        pair_probs = estimate_wide_probabilities_for_race(
            race_input,
            mc_samples=int(pl_config.mc_samples),
            top_k=int(pl_config.top_k),
            eps=float(pl_config.eps),
            rng=rng,
        )
        if pair_probs.empty:
            continue

        merged = pair_probs.merge(odds_by_race[race_id], on=["race_id", "kumiban"], how="inner")
        if merged.empty:
            continue
        merged = merged.merge(payout_df, on=["race_id", "kumiban"], how="left")
        payout_values = pd.to_numeric(merged["payout_yen"], errors="coerce")
        merged["payout_yen"] = payout_values.fillna(0).astype(int)
        merged["race_date"] = race["race_date"]
        merged["valid_year"] = int(race["valid_year"])
        candidate_rows.append(merged)

    if not candidate_rows:
        return pd.DataFrame()

    out = pd.concat(candidate_rows, axis=0, ignore_index=True)
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    out = out.dropna(subset=["race_date"]).copy()
    out = out.sort_values(["race_date", "race_id"], kind="mergesort").reset_index(drop=True)
    return out


def simulate_threshold(
    candidate_table: pd.DataFrame,
    *,
    min_p_wide: float,
    stage: str,
    ev_threshold: float,
    pl_config: PLSamplingConfig,
    bankroll_config: BankrollConfig,
) -> SweepResult:
    if stage not in MIN_P_WIDE_STAGE_CHOICES:
        raise ValueError(f"invalid stage: {stage}")

    race_meta = (
        candidate_table[["race_id", "race_date", "valid_year"]]
        .drop_duplicates()
        .sort_values(["race_date", "race_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    bankroll = int(bankroll_config.bankroll_init_yen)
    equity_curve = [float(bankroll)]
    day_spent_map: dict[str, int] = {}

    total_bet = 0
    total_return = 0
    n_hits = 0

    by_year_bet: dict[int, int] = {}
    by_year_return: dict[int, int] = {}
    by_year_bets: dict[int, int] = {}

    for race in race_meta.to_dict("records"):
        race_id = int(race["race_id"])
        race_date = pd.Timestamp(race["race_date"]).date()
        race_date_str = race_date.isoformat()
        valid_year = int(race["valid_year"])

        candidates = candidate_table[candidate_table["race_id"] == race_id].copy()
        if candidates.empty:
            continue

        if float(min_p_wide) > 0.0 and stage == "candidate":
            p_wide_values = pd.to_numeric(candidates["p_wide"], errors="coerce")
            candidates = candidates[p_wide_values >= float(min_p_wide)].copy()
            if candidates.empty:
                continue

        candidates["ev_profit"] = candidates["p_wide"] * candidates["odds"] - 1.0
        candidates = candidates[candidates["ev_profit"] >= float(ev_threshold)].copy()
        if candidates.empty:
            continue

        race_bets = allocate_race_bets(candidates, bankroll_yen=bankroll, config=bankroll_config)
        if race_bets.empty:
            continue

        spent_today = int(day_spent_map.get(race_date_str, 0))
        daily_cap = round_down_to_unit(
            float(bankroll) * float(bankroll_config.daily_cap_fraction),
            int(bankroll_config.bet_unit_yen),
        )
        remaining_cap = max(int(daily_cap) - spent_today, 0)
        race_bets = apply_remaining_daily_cap(
            race_bets,
            remaining_cap_yen=int(remaining_cap),
            bet_unit_yen=int(bankroll_config.bet_unit_yen),
            min_bet_yen=int(bankroll_config.min_bet_yen),
        )
        if race_bets.empty:
            continue

        if float(min_p_wide) > 0.0 and stage == "selected":
            p_wide_values = pd.to_numeric(race_bets["p_wide"], errors="coerce")
            race_bets = race_bets[p_wide_values >= float(min_p_wide)].copy()
            if race_bets.empty:
                continue

        race_total_bet = int(race_bets["bet_yen"].sum())
        if race_total_bet <= 0:
            continue
        day_spent_map[race_date_str] = spent_today + race_total_bet

        race_total_return = 0
        race_hits = 0
        for row in race_bets.to_dict("records"):
            bet_yen = int(row["bet_yen"])
            kumiban = str(row["kumiban"])
            payout_yen_per_100 = int(
                candidates.loc[candidates["kumiban"] == kumiban, "payout_yen"].iloc[0]
            )
            is_hit = payout_yen_per_100 > 0
            payout = int((bet_yen // 100) * payout_yen_per_100) if is_hit else 0
            race_total_return += payout
            race_hits += 1 if is_hit else 0

        total_bet += race_total_bet
        total_return += race_total_return
        n_hits += race_hits

        by_year_bet[valid_year] = int(by_year_bet.get(valid_year, 0) + race_total_bet)
        by_year_return[valid_year] = int(by_year_return.get(valid_year, 0) + race_total_return)
        by_year_bets[valid_year] = int(by_year_bets.get(valid_year, 0) + len(race_bets))

        bankroll = int(bankroll + race_total_return - race_total_bet)
        equity_curve.append(float(bankroll))

    roi = float(total_return / total_bet) if total_bet > 0 else 0.0
    max_dd = int(round(compute_max_drawdown(equity_curve)))

    roi_by_year: dict[int, float] = {}
    for year in sorted(by_year_bet):
        bet = int(by_year_bet[year])
        ret = int(by_year_return.get(year, 0))
        roi_by_year[int(year)] = float(ret / bet) if bet > 0 else 0.0

    return SweepResult(
        min_p_wide=float(min_p_wide),
        stage=str(stage),
        roi=float(roi),
        n_bets=int(sum(by_year_bets.values())),
        n_hits=int(n_hits),
        total_bet=int(total_bet),
        total_return=int(total_return),
        max_drawdown=int(max_dd),
        roi_by_year=roi_by_year,
        n_bets_by_year={int(k): int(v) for k, v in by_year_bets.items()},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.holdout_year <= 0:
        raise SystemExit("--holdout-year must be > 0")
    if args.mc_samples <= 0:
        raise SystemExit("--mc-samples must be > 0")
    if args.pl_top_k <= 1:
        raise SystemExit("--pl-top-k must be > 1")
    if not (0.0 < args.pl_eps < 0.5):
        raise SystemExit("--pl-eps must be in (0, 0.5)")
    if args.max_bets_per_race <= 0:
        raise SystemExit("--max-bets-per-race must be > 0")
    if args.kelly_fraction < 0.0:
        raise SystemExit("--kelly-fraction must be >= 0")
    if args.race_cap_fraction <= 0.0:
        raise SystemExit("--race-cap-fraction must be > 0")
    if args.daily_cap_fraction <= 0.0:
        raise SystemExit("--daily-cap-fraction must be > 0")
    if args.bankroll_init_yen <= 0:
        raise SystemExit("--bankroll-init-yen must be > 0")
    if args.bet_unit_yen <= 0:
        raise SystemExit("--bet-unit-yen must be > 0")
    if args.min_bet_yen <= 0:
        raise SystemExit("--min-bet-yen must be > 0")
    if args.min_bet_yen % args.bet_unit_yen != 0:
        raise SystemExit("--min-bet-yen must be a multiple of --bet-unit-yen")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    meta_output_path = resolve_path(args.meta_output)
    check_overwrite([output_path, meta_output_path], force=args.force)

    grid = _build_grid(args)
    logger.info("grid size=%s values=%s", len(grid), grid)

    top3_oof = load_top3_oof(input_path)
    selected_oof, selected_years, available_years = select_years(
        top3_oof,
        holdout_year=int(args.holdout_year),
        years_arg=args.years,
        require_years_arg=args.require_years,
    )

    race_ids = sorted(selected_oof["race_id"].unique().tolist())
    logger.info(
        "selected years=%s races=%s rows=%s (available=%s)",
        selected_years,
        len(race_ids),
        len(selected_oof),
        available_years,
    )

    pl_config = PLSamplingConfig(
        mc_samples=int(args.mc_samples),
        top_k=int(args.pl_top_k),
        eps=float(args.pl_eps),
        seed=int(args.seed),
    )
    max_bet_yen = args.max_bet_yen if int(args.max_bet_yen) > 0 else None
    bankroll_config = BankrollConfig(
        bankroll_init_yen=int(args.bankroll_init_yen),
        kelly_fraction_scale=float(args.kelly_fraction),
        max_bets_per_race=int(args.max_bets_per_race),
        race_cap_fraction=float(args.race_cap_fraction),
        daily_cap_fraction=float(args.daily_cap_fraction),
        bet_unit_yen=int(args.bet_unit_yen),
        min_bet_yen=int(args.min_bet_yen),
        max_bet_yen=max_bet_yen,
    )

    with Database() as db:
        odds_raw, payout_raw, race_raw = fetch_odds_payout_race(db, race_ids)

    odds_df = _prepare_odds(odds_raw)
    payout_df = _prepare_payout(payout_raw)
    race_date_map = _prepare_race_dates(race_raw)

    if odds_df.empty:
        raise SystemExit("No odds rows found in core.o3_wide for selected races.")

    candidate_table = build_candidate_table(
        selected_oof,
        odds_df=odds_df,
        payout_df=payout_df,
        race_date_map=race_date_map,
        pl_config=pl_config,
    )
    if candidate_table.empty:
        raise SystemExit("candidate_table is empty (no races with odds & dates?).")

    logger.info(
        "candidate_table rows=%s races=%s",
        len(candidate_table),
        candidate_table["race_id"].nunique(),
    )

    results: list[SweepResult] = []
    for min_p_wide in grid:
        result = simulate_threshold(
            candidate_table,
            min_p_wide=float(min_p_wide),
            stage=str(args.min_p_wide_stage),
            ev_threshold=float(args.ev_threshold),
            pl_config=pl_config,
            bankroll_config=bankroll_config,
        )
        results.append(result)
        logger.info(
            "min_p_wide=%.4f stage=%s roi=%.4f bets=%s total_bet=%s max_dd=%s",
            result.min_p_wide,
            result.stage,
            result.roi,
            result.n_bets,
            result.total_bet,
            result.max_drawdown,
        )

    ranked = sorted(results, key=lambda item: float(item.roi), reverse=True)
    best = ranked[0]

    payload = {
        "summary": {
            "best_min_p_wide": float(best.min_p_wide),
            "best_stage": str(best.stage),
            "best_roi": float(best.roi),
            "best_n_bets": int(best.n_bets),
            "best_total_bet": int(best.total_bet),
            "best_total_return": int(best.total_return),
            "best_max_drawdown": int(best.max_drawdown),
        },
        "results": [
            {
                "min_p_wide": float(item.min_p_wide),
                "stage": str(item.stage),
                "roi": round(float(item.roi), 6),
                "n_bets": int(item.n_bets),
                "n_hits": int(item.n_hits),
                "total_bet": int(item.total_bet),
                "total_return": int(item.total_return),
                "max_drawdown": int(item.max_drawdown),
                "roi_by_year": {str(k): round(float(v), 6) for k, v in item.roi_by_year.items()},
                "n_bets_by_year": {str(k): int(v) for k, v in item.n_bets_by_year.items()},
            }
            for item in ranked
        ],
    }
    save_json(output_path, payload)

    meta_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input": {
            "path": str(input_path),
            "rows": int(len(top3_oof)),
            "selected_rows": int(len(selected_oof)),
            "selected_races": int(selected_oof["race_id"].nunique()),
            "available_years_after_holdout_filter": available_years,
            "selected_years": selected_years,
            "holdout_year": int(args.holdout_year),
        },
        "grid": {
            "stage": str(args.min_p_wide_stage),
            "values": grid,
        },
        "config": {
            "pl": {
                "mc_samples": int(pl_config.mc_samples),
                "top_k": int(pl_config.top_k),
                "eps": float(pl_config.eps),
                "seed": int(pl_config.seed),
            },
            "selection": {
                "ev_threshold": float(args.ev_threshold),
                "max_bets_per_race": int(args.max_bets_per_race),
            },
            "bankroll": {
                "bankroll_init_yen": int(bankroll_config.bankroll_init_yen),
                "kelly_fraction_scale": float(bankroll_config.kelly_fraction_scale),
                "race_cap_fraction": float(bankroll_config.race_cap_fraction),
                "daily_cap_fraction": float(bankroll_config.daily_cap_fraction),
                "bet_unit_yen": int(bankroll_config.bet_unit_yen),
                "min_bet_yen": int(bankroll_config.min_bet_yen),
                "max_bet_yen": bankroll_config.max_bet_yen,
            },
        },
        "db_sources": {
            "odds": "core.o3_wide(min_odds_x10, latest data_kbn/announce per race+kumiban)",
            "payout": "core.payout(bet_type=5, selection=kumiban)",
            "race_date": "core.race(race_date)",
        },
        "output": {
            "result_path": str(output_path),
            "meta_path": str(meta_output_path),
            "n_grid": int(len(grid)),
        },
    }
    save_json(meta_output_path, meta_payload)

    logger.info("best min_p_wide=%.4f roi=%.4f bets=%s", best.min_p_wide, best.roi, best.n_bets)
    logger.info("wrote %s", output_path)
    logger.info("wrote %s", meta_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
