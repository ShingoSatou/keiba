#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

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
from scripts_v2.wide_prob_v2_common import kumiban_from_horse_nos  # noqa: E402
from scripts_v3.pl_v3_common import estimate_p_wide_by_race  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_YEAR = 2025
DEFAULT_SEED = 42
MIN_P_WIDE_STAGE_CHOICES = ("candidate", "selected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wide backtest for v3 (pl_score -> MC -> p_wide -> EV -> Kelly)."
    )
    parser.add_argument("--input", default="data/oof/pl_v3_oof.parquet")
    parser.add_argument("--output", default="data/backtest_v3/backtest_wide_v3_direct.json")
    parser.add_argument(
        "--meta-output",
        default="data/backtest_v3/backtest_wide_v3_direct_meta.json",
    )
    parser.add_argument("--years", default="", help="Comma-separated valid_year filter.")
    parser.add_argument(
        "--require-years",
        default="",
        help="Comma-separated years that must exist in input after holdout filter.",
    )
    parser.add_argument("--holdout-year", type=int, default=DEFAULT_HOLDOUT_YEAR)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--mc-samples", type=int, default=10_000)
    parser.add_argument("--pl-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--min-p-wide",
        type=float,
        default=0.0,
        help="Minimum estimated p_wide required to consider buying a ticket.",
    )
    parser.add_argument(
        "--min-p-wide-stage",
        choices=list(MIN_P_WIDE_STAGE_CHOICES),
        default="candidate",
        help="Where to apply --min-p-wide filter.",
    )
    parser.add_argument("--ev-threshold", type=float, default=0.0)
    parser.add_argument("--max-bets-per-race", type=int, default=5)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--race-cap-fraction", type=float, default=0.05)
    parser.add_argument("--daily-cap-fraction", type=float, default=0.20)
    parser.add_argument("--bankroll-init-yen", type=int, default=1_000_000)
    parser.add_argument("--bet-unit-yen", type=int, default=100)
    parser.add_argument("--min-bet-yen", type=int, default=100)
    parser.add_argument("--max-bet-yen", type=int, default=0)

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
    return sorted({int(token.strip()) for token in str(raw).split(",") if token.strip()})


def check_overwrite(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists. pass --force to overwrite: {joined}")


def _read_input_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _coerce_required_numeric_column(
    frame: pd.DataFrame,
    *,
    column: str,
    as_int: bool,
    input_mode: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = int(values.isna().sum())
    if invalid > 0:
        raise SystemExit(
            f"Invalid numeric values in column '{column}' for {input_mode} input: {invalid} rows"
        )
    if as_int:
        return values.astype(np.int64)
    return values.astype(float)


def _validate_single_value_per_race(frame: pd.DataFrame, *, column: str) -> None:
    if column not in frame.columns:
        return
    counts = frame.groupby("race_id", sort=False)[column].nunique(dropna=True)
    bad = counts[counts > 1]
    if not bad.empty:
        sample = bad.head(5).to_dict()
        raise SystemExit(f"Column '{column}' must be constant per race_id. sample={sample}")


def _sanitize_pair_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "horse_no_1", "horse_no_2", "kumiban", "p_wide"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in pair-level input: {missing}")

    out = frame.copy()
    out["race_id"] = _coerce_required_numeric_column(
        out,
        column="race_id",
        as_int=True,
        input_mode="pair-level",
    )
    out["horse_no_1"] = _coerce_required_numeric_column(
        out,
        column="horse_no_1",
        as_int=True,
        input_mode="pair-level",
    )
    out["horse_no_2"] = _coerce_required_numeric_column(
        out,
        column="horse_no_2",
        as_int=True,
        input_mode="pair-level",
    )
    out["p_wide"] = _coerce_required_numeric_column(
        out,
        column="p_wide",
        as_int=False,
        input_mode="pair-level",
    )

    if ((out["p_wide"] < 0.0) | (out["p_wide"] > 1.0)).any():
        raise SystemExit("p_wide must be in [0, 1] for pair-level input")

    horse_no_left = out["horse_no_1"].astype(int).to_numpy()
    horse_no_right = out["horse_no_2"].astype(int).to_numpy()
    out["horse_no_1"] = np.minimum(horse_no_left, horse_no_right).astype(int)
    out["horse_no_2"] = np.maximum(horse_no_left, horse_no_right).astype(int)
    if (out["horse_no_1"] == out["horse_no_2"]).any():
        raise SystemExit("horse_no_1 and horse_no_2 must differ in pair-level input")

    out["kumiban"] = out["kumiban"].astype(str).str.strip()
    computed_kumiban = out.apply(
        lambda row: kumiban_from_horse_nos(int(row["horse_no_1"]), int(row["horse_no_2"])),
        axis=1,
    )
    empty_kumiban = out["kumiban"] == ""
    out.loc[empty_kumiban, "kumiban"] = computed_kumiban[empty_kumiban]
    mismatch = out["kumiban"] != computed_kumiban
    if bool(mismatch.any()):
        bad = out.loc[mismatch, ["race_id", "horse_no_1", "horse_no_2", "kumiban"]].head(5)
        raise SystemExit(
            "kumiban mismatch with horse_no pair in pair-level input. "
            f"sample={bad.to_dict('records')}"
        )

    if out.duplicated(["race_id", "horse_no_1", "horse_no_2"]).any():
        dup = out[out.duplicated(["race_id", "horse_no_1", "horse_no_2"], keep=False)].head(5)
        raise SystemExit(
            "Duplicate (race_id, horse_no_1, horse_no_2) in pair-level input: "
            f"{dup.to_dict('records')}"
        )

    if "fold_id" in out.columns:
        out["fold_id"] = _coerce_required_numeric_column(
            out,
            column="fold_id",
            as_int=True,
            input_mode="pair-level",
        )
        _validate_single_value_per_race(out, column="fold_id")

    if "valid_year" in out.columns:
        out["valid_year"] = _coerce_required_numeric_column(
            out,
            column="valid_year",
            as_int=True,
            input_mode="pair-level",
        )
        _validate_single_value_per_race(out, column="valid_year")

    if "p_top3_1" in out.columns:
        out["p_top3_1"] = pd.to_numeric(out["p_top3_1"], errors="coerce")
    else:
        out["p_top3_1"] = np.nan
    if "p_top3_2" in out.columns:
        out["p_top3_2"] = pd.to_numeric(out["p_top3_2"], errors="coerce")
    else:
        out["p_top3_2"] = np.nan

    keep_cols = [
        c
        for c in [
            "race_id",
            "horse_no_1",
            "horse_no_2",
            "kumiban",
            "p_wide",
            "p_top3_1",
            "p_top3_2",
            "fold_id",
            "valid_year",
            "target_label",
            "p_top3",
        ]
        if c in out.columns
    ]
    out = out[keep_cols].copy()
    return out.sort_values(["race_id", "horse_no_1", "horse_no_2"], kind="mergesort").reset_index(
        drop=True
    )


def _sanitize_horse_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "horse_no", "pl_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns in horse-level input: {missing}")

    out = frame.copy()
    out["race_id"] = _coerce_required_numeric_column(
        out,
        column="race_id",
        as_int=True,
        input_mode="horse-level",
    )
    out["horse_no"] = _coerce_required_numeric_column(
        out,
        column="horse_no",
        as_int=True,
        input_mode="horse-level",
    )
    out["pl_score"] = _coerce_required_numeric_column(
        out,
        column="pl_score",
        as_int=False,
        input_mode="horse-level",
    )

    if out.duplicated(["race_id", "horse_no"]).any():
        dup = out[out.duplicated(["race_id", "horse_no"], keep=False)].head(5)
        raise SystemExit(
            f"Duplicate (race_id, horse_no) in horse-level input: {dup.to_dict('records')}"
        )

    if "fold_id" in out.columns:
        out["fold_id"] = _coerce_required_numeric_column(
            out,
            column="fold_id",
            as_int=True,
            input_mode="horse-level",
        )
        _validate_single_value_per_race(out, column="fold_id")

    if "valid_year" in out.columns:
        out["valid_year"] = _coerce_required_numeric_column(
            out,
            column="valid_year",
            as_int=True,
            input_mode="horse-level",
        )
        _validate_single_value_per_race(out, column="valid_year")

    keep_cols = [
        c
        for c in [
            "race_id",
            "horse_no",
            "pl_score",
            "fold_id",
            "valid_year",
            "target_label",
            "p_top3",
        ]
        if c in out.columns
    ]
    out = out[keep_cols].copy()
    return out.sort_values(["race_id", "horse_no"], kind="mergesort").reset_index(drop=True)


def load_backtest_input(path: Path) -> tuple[pd.DataFrame, str, str]:
    frame = _read_input_frame(path)
    columns = set(frame.columns)

    if "p_wide" in columns:
        return _sanitize_pair_input(frame), "pair", "input_pair"

    return _sanitize_horse_input(frame), "horse", "v3_pl_score_mc"


def select_backtest_years(
    frame: pd.DataFrame,
    *,
    holdout_year: int,
    years_arg: str,
    require_years_arg: str,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    years_requested = years_arg.strip()
    years_required = require_years_arg.strip()

    if "valid_year" not in frame.columns:
        if years_requested or years_required:
            raise SystemExit("--years/--require-years requires 'valid_year' column in input")
        return frame.copy(), [], []

    base = frame[frame["valid_year"] < int(holdout_year)].copy()
    if base.empty:
        raise SystemExit(f"No rows remain after holdout filter (valid_year < {int(holdout_year)}).")

    available = sorted(base["valid_year"].unique().tolist())

    required_years = parse_years(years_required)
    if required_years:
        missing = sorted(set(required_years) - set(available))
        if missing:
            raise SystemExit(
                f"required years are missing after holdout filter: {missing}, available={available}"
            )

    selected_years = parse_years(years_requested) if years_requested else available
    missing_selected = sorted(set(selected_years) - set(available))
    if missing_selected:
        raise SystemExit(
            f"selected years not found in input: {missing_selected}, available={available}"
        )

    out = base[base["valid_year"].isin(selected_years)].copy()
    if out.empty:
        raise SystemExit("No rows left after year selection.")
    return out, selected_years, available


def _estimate_pair_probs_from_horse(
    horse_frame: pd.DataFrame,
    *,
    mc_samples: int,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    required = {"race_id", "horse_no", "pl_score"}
    missing = sorted(required - set(horse_frame.columns))
    if missing:
        raise SystemExit(f"Missing required horse columns for p_wide estimation: {missing}")

    outputs: list[pd.DataFrame] = []

    if "fold_id" in horse_frame.columns:
        for fold_id, sub in horse_frame.groupby("fold_id", sort=False):
            fold_seed = int(seed) + int(fold_id)
            pair = estimate_p_wide_by_race(
                sub[["race_id", "horse_no", "pl_score"]],
                score_col="pl_score",
                mc_samples=int(mc_samples),
                seed=int(fold_seed),
                top_k=int(top_k),
            )
            if pair.empty:
                continue
            pair["fold_id"] = int(fold_id)
            if "valid_year" in sub.columns:
                year_map = (
                    sub[["race_id", "valid_year"]]
                    .drop_duplicates(["race_id"])
                    .set_index("race_id")["valid_year"]
                    .to_dict()
                )
                pair["valid_year"] = pair["race_id"].map(year_map).astype(int)
            outputs.append(pair)
    else:
        pair = estimate_p_wide_by_race(
            horse_frame[["race_id", "horse_no", "pl_score"]],
            score_col="pl_score",
            mc_samples=int(mc_samples),
            seed=int(seed),
            top_k=int(top_k),
        )
        if not pair.empty:
            if "valid_year" in horse_frame.columns:
                year_map = (
                    horse_frame[["race_id", "valid_year"]]
                    .drop_duplicates(["race_id"])
                    .set_index("race_id")["valid_year"]
                    .to_dict()
                )
                pair["valid_year"] = pair["race_id"].map(year_map).astype(int)
            outputs.append(pair)

    if not outputs:
        base_cols = [
            "race_id",
            "horse_no_1",
            "horse_no_2",
            "kumiban",
            "p_wide",
            "p_top3_1",
            "p_top3_2",
        ]
        if "fold_id" in horse_frame.columns:
            base_cols.append("fold_id")
        if "valid_year" in horse_frame.columns:
            base_cols.append("valid_year")
        return pd.DataFrame(columns=base_cols)

    out = pd.concat(outputs, axis=0, ignore_index=True)
    if out.duplicated(["race_id", "horse_no_1", "horse_no_2"]).any():
        dup = out[out.duplicated(["race_id", "horse_no_1", "horse_no_2"], keep=False)].head(5)
        raise SystemExit(
            "Duplicate pair rows generated from horse-level input. "
            f"sample={dup.to_dict('records')}"
        )
    return out.sort_values(["race_id", "horse_no_1", "horse_no_2"], kind="mergesort").reset_index(
        drop=True
    )


def fetch_db_tables(
    db: Database,
    race_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    runner_rows = db.fetch_all(
        """
        SELECT race_id, horse_no, horse_name
        FROM core.runner
        WHERE race_id = ANY(%(race_ids)s)
        """,
        {"race_ids": race_ids},
    )
    return (
        pd.DataFrame(odds_rows),
        pd.DataFrame(payout_rows),
        pd.DataFrame(race_rows),
        pd.DataFrame(runner_rows),
    )


def quality_metrics(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if "target_label" not in frame.columns or "p_top3" not in frame.columns:
        return None, None
    y_true = (
        pd.to_numeric(frame["target_label"], errors="coerce").fillna(0).astype(int) > 0
    ).astype(int)
    p_pred = pd.to_numeric(frame["p_top3"], errors="coerce")
    valid = y_true.notna() & p_pred.notna()
    if int(valid.sum()) == 0:
        return None, None
    y = y_true[valid].to_numpy(dtype=int)
    p = np.clip(p_pred[valid].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
    logloss_value: float | None
    auc_value: float | None
    try:
        logloss_value = float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        logloss_value = None
    try:
        auc_value = float(roc_auc_score(y, p))
    except ValueError:
        auc_value = None
    return logloss_value, auc_value


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


def _round_or_none(value: Any, digits: int) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.holdout_year <= 0:
        raise SystemExit("--holdout-year must be > 0")
    if args.mc_samples <= 0:
        raise SystemExit("--mc-samples must be > 0")
    if args.pl_top_k <= 1:
        raise SystemExit("--pl-top-k must be > 1")
    if not (0.0 <= float(args.min_p_wide) <= 1.0):
        raise SystemExit("--min-p-wide must be in [0, 1]")
    if args.min_p_wide_stage not in MIN_P_WIDE_STAGE_CHOICES:
        raise SystemExit(
            f"--min-p-wide-stage must be one of {MIN_P_WIDE_STAGE_CHOICES}. "
            f"got={args.min_p_wide_stage}"
        )
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

    max_bet_yen = args.max_bet_yen if args.max_bet_yen > 0 else None
    output_path = resolve_path(args.output)
    meta_output_path = resolve_path(args.meta_output)
    check_overwrite([output_path, meta_output_path], force=args.force)

    input_path = resolve_path(args.input)
    loaded_input, input_mode, p_wide_source = load_backtest_input(input_path)
    selected_input, selected_years, available_years = select_backtest_years(
        loaded_input,
        holdout_year=int(args.holdout_year),
        years_arg=args.years,
        require_years_arg=args.require_years,
    )

    if input_mode == "pair":
        pair_probs = selected_input.copy()
        metric_frame = pd.DataFrame(columns=["target_label", "p_top3"])
    else:
        metric_frame = selected_input.copy()
        pair_probs = _estimate_pair_probs_from_horse(
            selected_input,
            mc_samples=int(args.mc_samples),
            top_k=int(args.pl_top_k),
            seed=int(args.seed),
        )

    if pair_probs.empty:
        raise SystemExit("No pair probabilities available for backtest after preprocessing.")

    race_ids = sorted(pair_probs["race_id"].astype(int).unique().tolist())
    logger.info(
        "input_mode=%s p_wide_source=%s selected_years=%s races=%s input_rows=%s pair_rows=%s",
        input_mode,
        p_wide_source,
        selected_years,
        len(race_ids),
        len(selected_input),
        len(pair_probs),
    )

    with Database() as db:
        odds_df, payout_df, race_df, runner_df = fetch_db_tables(db, race_ids)

    if odds_df.empty:
        raise SystemExit("No odds rows found in core.o3_wide for selected races.")
    odds_df["race_id"] = pd.to_numeric(odds_df["race_id"], errors="coerce").astype("Int64")
    odds_df["min_odds_x10"] = pd.to_numeric(odds_df["min_odds_x10"], errors="coerce")
    odds_df = odds_df.dropna(subset=["race_id", "kumiban", "min_odds_x10"]).copy()
    odds_df["race_id"] = odds_df["race_id"].astype(int)
    odds_df["kumiban"] = odds_df["kumiban"].astype(str)
    odds_df["odds"] = odds_df["min_odds_x10"].astype(float) / 10.0
    odds_df = odds_df[odds_df["odds"] > 1.0].copy()
    odds_df = odds_df[["race_id", "kumiban", "odds"]]

    payout_df = payout_df.copy()
    if not payout_df.empty:
        payout_df["race_id"] = pd.to_numeric(payout_df["race_id"], errors="coerce").astype("Int64")
        payout_df["payout_yen"] = pd.to_numeric(payout_df["payout_yen"], errors="coerce")
        payout_df = payout_df.dropna(subset=["race_id", "kumiban", "payout_yen"]).copy()
        payout_df["race_id"] = payout_df["race_id"].astype(int)
        payout_df["kumiban"] = payout_df["kumiban"].astype(str)
        payout_df["payout_yen"] = payout_df["payout_yen"].astype(int)
        payout_df = payout_df[["race_id", "kumiban", "payout_yen"]]
    else:
        payout_df = pd.DataFrame(columns=["race_id", "kumiban", "payout_yen"])

    race_df["race_id"] = pd.to_numeric(race_df["race_id"], errors="coerce").astype("Int64")
    race_df["race_date"] = pd.to_datetime(race_df["race_date"], errors="coerce")
    race_df = race_df.dropna(subset=["race_id", "race_date"]).copy()
    race_df["race_id"] = race_df["race_id"].astype(int)
    race_date_map = {int(r["race_id"]): r["race_date"].date() for r in race_df.to_dict("records")}

    runner_df["race_id"] = pd.to_numeric(runner_df["race_id"], errors="coerce").astype("Int64")
    runner_df["horse_no"] = pd.to_numeric(runner_df["horse_no"], errors="coerce").astype("Int64")
    runner_df = runner_df.dropna(subset=["race_id", "horse_no"]).copy()
    runner_df["race_id"] = runner_df["race_id"].astype(int)
    runner_df["horse_no"] = runner_df["horse_no"].astype(int)
    runner_df["horse_name"] = runner_df["horse_name"].fillna("").astype(str)
    runner_name_map = {}
    for row in runner_df.to_dict("records"):
        race_id_key = int(row["race_id"])
        horse_no_key = int(row["horse_no"])
        horse_name = str(row["horse_name"]).strip() or f"馬番{horse_no_key}"
        runner_name_map[(race_id_key, horse_no_key)] = horse_name

    odds_by_race = {int(rid): sub.copy() for rid, sub in odds_df.groupby("race_id", sort=False)}
    payout_map = {
        (int(row["race_id"]), str(row["kumiban"])): int(row["payout_yen"])
        for row in payout_df.to_dict("records")
    }

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

    race_rows = pair_probs[["race_id"]].drop_duplicates().copy()
    if "valid_year" in pair_probs.columns:
        year_map = pair_probs[["race_id", "valid_year"]].drop_duplicates("race_id")
        race_rows = race_rows.merge(year_map, on="race_id", how="left")
    else:
        race_rows["valid_year"] = pd.NA
    race_rows = race_rows.assign(race_date=lambda df: df["race_id"].map(race_date_map))
    race_rows["race_date"] = pd.to_datetime(race_rows["race_date"], errors="coerce")
    race_rows = race_rows.sort_values(["race_date", "race_id"], kind="mergesort")
    race_rows = race_rows.reset_index(drop=True)

    bankroll = int(bankroll_config.bankroll_init_yen)
    equity_curve = [float(bankroll)]
    day_spent_map: dict[str, int] = {}

    total_bet = 0
    total_return = 0
    n_hits = 0
    bet_records: list[dict[str, Any]] = []
    monthly_map: dict[str, dict[str, float | int]] = {}

    for race in race_rows.to_dict("records"):
        race_id = int(race["race_id"])
        race_date_obj = race.get("race_date")
        if pd.isna(race_date_obj):
            continue
        race_date = pd.Timestamp(race_date_obj).date()
        race_date_str = race_date.isoformat()
        race_month = race_date.strftime("%Y-%m")

        race_input = pair_probs[pair_probs["race_id"] == race_id].copy()
        if race_input.empty:
            continue
        if race_id not in odds_by_race:
            continue

        candidates = race_input.merge(odds_by_race[race_id], on=["race_id", "kumiban"], how="inner")
        if candidates.empty:
            continue

        if float(args.min_p_wide) > 0.0 and args.min_p_wide_stage == "candidate":
            p_wide_values = pd.to_numeric(candidates["p_wide"], errors="coerce")
            candidates = candidates[p_wide_values >= float(args.min_p_wide)].copy()
            if candidates.empty:
                continue

        candidates["ev_profit"] = candidates["p_wide"] * candidates["odds"] - 1.0
        candidates = candidates[candidates["ev_profit"] >= float(args.ev_threshold)].copy()
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

        if float(args.min_p_wide) > 0.0 and args.min_p_wide_stage == "selected":
            p_wide_values = pd.to_numeric(race_bets["p_wide"], errors="coerce")
            race_bets = race_bets[p_wide_values >= float(args.min_p_wide)].copy()
            if race_bets.empty:
                continue

        race_total_bet = int(race_bets["bet_yen"].sum())
        if race_total_bet <= 0:
            continue
        day_spent_map[race_date_str] = spent_today + race_total_bet

        race_total_return = 0
        race_hit_count = 0
        for row in race_bets.to_dict("records"):
            horse_no_1 = int(row["horse_no_1"])
            horse_no_2 = int(row["horse_no_2"])
            kumiban = str(row["kumiban"]) or kumiban_from_horse_nos(horse_no_1, horse_no_2)
            bet_yen = int(row["bet_yen"])
            payout_yen_per_100 = int(payout_map.get((race_id, kumiban), 0))
            is_hit = payout_yen_per_100 > 0
            payout = int((bet_yen // 100) * payout_yen_per_100) if is_hit else 0
            profit = int(payout - bet_yen)

            race_total_return += payout
            race_hit_count += 1 if is_hit else 0

            horse_name_1 = runner_name_map.get((race_id, horse_no_1), f"馬番{horse_no_1}")
            horse_name_2 = runner_name_map.get((race_id, horse_no_2), f"馬番{horse_no_2}")
            pair_display_name = f"{horse_name_1} / {horse_name_2}"

            valid_year_raw = row.get("valid_year", race.get("valid_year"))
            valid_year = int(valid_year_raw) if pd.notna(valid_year_raw) else None

            bet_records.append(
                {
                    "race_date": race_date_str,
                    "race_id": race_id,
                    "valid_year": valid_year,
                    "horse_name": pair_display_name,
                    "horse_no": horse_no_1,
                    "p_win": _round_or_none(row.get("p_wide"), 6),
                    "odds_final": _round_or_none(row.get("odds"), 1),
                    "ev_profit": _round_or_none(row.get("ev_profit"), 6),
                    "is_hit": bool(is_hit),
                    "payout": int(payout),
                    "profit": int(profit),
                    "horse_no_1": horse_no_1,
                    "horse_no_2": horse_no_2,
                    "horse_name_1": horse_name_1,
                    "horse_name_2": horse_name_2,
                    "kumiban": kumiban,
                    "p_wide": _round_or_none(row.get("p_wide"), 6),
                    "p_top3_1": _round_or_none(row.get("p_top3_1"), 6),
                    "p_top3_2": _round_or_none(row.get("p_top3_2"), 6),
                    "bet_yen": bet_yen,
                    "payout_yen_per_100": payout_yen_per_100,
                    "kelly_f": _round_or_none(row.get("kelly_f"), 8),
                }
            )

        total_bet += race_total_bet
        total_return += race_total_return
        n_hits += race_hit_count
        bankroll = int(bankroll + race_total_return - race_total_bet)
        equity_curve.append(float(bankroll))

        month_item = monthly_map.setdefault(
            race_month,
            {"month": race_month, "n_bets": 0, "n_hits": 0, "total_bet": 0, "total_return": 0},
        )
        month_item["n_bets"] += int(len(race_bets))
        month_item["n_hits"] += int(race_hit_count)
        month_item["total_bet"] += int(race_total_bet)
        month_item["total_return"] += int(race_total_return)

    monthly_rows: list[dict[str, Any]] = []
    for month in sorted(monthly_map):
        item = monthly_map[month]
        month_bet = int(item["total_bet"])
        month_return = int(item["total_return"])
        month_roi = float(month_return / month_bet) if month_bet > 0 else 0.0
        monthly_rows.append(
            {
                "month": month,
                "n_bets": int(item["n_bets"]),
                "n_hits": int(item["n_hits"]),
                "roi": round(month_roi, 4),
            }
        )

    n_bets = len(bet_records)
    roi = float(total_return / total_bet) if total_bet > 0 else 0.0
    hit_rate = float(n_hits / n_bets) if n_bets > 0 else 0.0
    max_dd = compute_max_drawdown(equity_curve)
    period_from = min((row["race_date"] for row in bet_records), default="")
    period_to = max((row["race_date"] for row in bet_records), default="")
    logloss_value, auc_value = quality_metrics(metric_frame)

    summary = {
        "period_from": period_from,
        "period_to": period_to,
        "n_races": int(race_rows["race_id"].nunique()),
        "n_bets": int(n_bets),
        "n_hits": int(n_hits),
        "hit_rate": round(hit_rate, 4),
        "total_bet": int(total_bet),
        "total_return": int(total_return),
        "roi": round(roi, 4),
        "max_drawdown": int(round(max_dd)),
        "logloss": round(float(logloss_value), 4) if logloss_value is not None else None,
        "auc": round(float(auc_value), 4) if auc_value is not None else None,
    }

    payload = {
        "summary": summary,
        "monthly": monthly_rows,
        "bets": bet_records,
    }
    save_json(output_path, payload)

    meta_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input": {
            "path": str(input_path),
            "input_mode": input_mode,
            "p_wide_source": p_wide_source,
            "rows": int(len(loaded_input)),
            "selected_rows": int(len(selected_input)),
            "pair_rows_for_backtest": int(len(pair_probs)),
            "selected_races": int(pair_probs["race_id"].nunique()),
            "available_years_after_holdout_filter": available_years,
            "selected_years": selected_years,
            "holdout_year": int(args.holdout_year),
        },
        "config": {
            "pl": {
                "mc_samples": int(args.mc_samples),
                "top_k": int(args.pl_top_k),
                "seed": int(args.seed),
            },
            "selection": {
                "min_p_wide": float(args.min_p_wide),
                "min_p_wide_stage": str(args.min_p_wide_stage),
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
            "horse_name": "core.runner(horse_name)",
        },
        "output": {
            "result_path": str(output_path),
            "meta_path": str(meta_output_path),
            "n_bets": int(n_bets),
            "n_months": int(len(monthly_rows)),
        },
    }
    save_json(meta_output_path, meta_payload)

    logger.info(
        "finished races=%s bets=%s roi=%.4f max_dd=%s",
        summary["n_races"],
        summary["n_bets"],
        summary["roi"],
        summary["max_drawdown"],
    )
    logger.info("wrote %s", output_path)
    logger.info("wrote %s", meta_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
