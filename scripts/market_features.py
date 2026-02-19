from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.infrastructure.database import Database
from scripts.t5_modeling import MARKET_SERIES_MINUTES

logger = logging.getLogger(__name__)


def _chunk_indices(size: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < size:
        end = min(start + chunk_size, size)
        yield start, end
        start = end


def _build_chunk_query(minutes: list[int], value_rows: list[str]) -> str:
    odds_subqueries = []
    for minute in minutes:
        odds_subqueries.append(
            f"""
            (
                SELECT
                    CASE
                        WHEN w.win_odds_x10 IS NULL OR w.win_odds_x10 <= 0 THEN NULL
                        ELSE ROUND(w.win_odds_x10::numeric / 10.0, 2)
                    END
                FROM core.o1_header h
                JOIN core.o1_win w
                  ON w.race_id = h.race_id
                 AND w.data_kbn = h.data_kbn
                 AND w.announce_mmddhhmi = h.announce_mmddhhmi
                WHERE h.race_id = t.race_id
                  AND w.horse_no = t.horse_no
                  AND h.data_kbn = 1
                  AND h.announce_mmddhhmi <= to_char(
                      t.asof_ts - INTERVAL '{minute} minutes',
                      'MMDDHH24MI'
                  )
                ORDER BY h.announce_mmddhhmi DESC
                LIMIT 1
            ) AS M_odds_tminus_{minute}
            """
        )
    return f"""
    WITH target(row_idx, race_id, horse_no, asof_ts) AS (
        VALUES
        {",".join(value_rows)}
    )
    SELECT
        t.row_idx,
        t.race_id,
        t.horse_no,
        t.asof_ts,
        (
            SELECT h.win_pool_total_100yen
            FROM core.o1_header h
            WHERE h.race_id = t.race_id
              AND h.data_kbn = 1
              AND h.announce_mmddhhmi <= to_char(
                  t.asof_ts - INTERVAL '60 minutes',
                  'MMDDHH24MI'
              )
            ORDER BY h.announce_mmddhhmi DESC
            LIMIT 1
        ) AS M_win_pool_total_tminus_60,
        {",".join(odds_subqueries)}
    FROM target t
    """


def _fetch_market_points(
    db: Database,
    frame: pd.DataFrame,
    minutes: list[int],
    chunk_size: int = 400,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    required = {"race_id", "horse_no", "asof_ts"}
    if not required.issubset(set(frame.columns)):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"market feature input missing columns: {missing}")

    targets = frame[["race_id", "horse_no", "asof_ts"]].copy().reset_index(drop=True)
    targets["row_idx"] = targets.index.astype(int)
    targets["asof_ts"] = pd.to_datetime(targets["asof_ts"], errors="coerce")
    targets = targets[targets["asof_ts"].notna()].copy()
    if targets.empty:
        return pd.DataFrame()

    merged_parts: list[pd.DataFrame] = []
    for start, end in _chunk_indices(len(targets), chunk_size):
        chunk = targets.iloc[start:end]
        params: dict[str, object] = {}
        value_rows: list[str] = []
        for local_pos, row in enumerate(chunk.itertuples(index=False)):
            suffix = f"{start}_{local_pos}"
            key_row = f"row_{suffix}"
            key_race = f"race_{suffix}"
            key_horse = f"horse_{suffix}"
            key_asof = f"asof_{suffix}"
            params[key_row] = int(row.row_idx)
            params[key_race] = int(row.race_id)
            params[key_horse] = int(row.horse_no)
            params[key_asof] = row.asof_ts
            value_rows.append(f"(%({key_row})s, %({key_race})s, %({key_horse})s, %({key_asof})s)")

        query = _build_chunk_query(minutes=minutes, value_rows=value_rows)
        rows = db.fetch_all(query, params)
        merged_parts.append(pd.DataFrame(rows))

    points = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    if points.empty:
        return points
    points["asof_ts"] = pd.to_datetime(points["asof_ts"], errors="coerce")
    return points


def _calc_entropy(frame: pd.DataFrame, prob_col: str) -> pd.Series:
    if frame.empty or prob_col not in frame.columns:
        return pd.Series(dtype=float)

    out = pd.Series(index=frame.index, dtype=float)
    group_cols = ["race_id"]
    if "asof_ts" in frame.columns:
        group_cols.append("asof_ts")
    for _, group in frame.groupby(group_cols, sort=False):
        probs = pd.to_numeric(group[prob_col], errors="coerce").to_numpy(dtype=float).copy()
        probs[~np.isfinite(probs)] = 0.0
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            entropy = math.nan
        else:
            norm = probs / total
            norm = np.clip(norm, 1e-12, 1.0)
            entropy = float(-(norm * np.log(norm)).sum())
        out.loc[group.index] = entropy
    return out


def add_market_features(
    db: Database,
    frame: pd.DataFrame,
    minutes: list[int] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    minutes = list(minutes or MARKET_SERIES_MINUTES)
    output = frame.copy()
    output["asof_ts"] = pd.to_datetime(output["asof_ts"], errors="coerce")

    output["M_odds_win_t5"] = pd.to_numeric(output.get("odds_win_t5"), errors="coerce")
    output["M_imp_prob_t5"] = np.where(
        output["M_odds_win_t5"] > 0,
        1.0 / output["M_odds_win_t5"],
        np.nan,
    )
    output["M_log_odds_t5"] = np.log(output["M_odds_win_t5"])
    output.loc[~np.isfinite(output["M_log_odds_t5"]), "M_log_odds_t5"] = np.nan
    output["M_odds_rank_t5"] = pd.to_numeric(output.get("odds_rank_t5"), errors="coerce")
    output["M_odds_missing_flag"] = (
        output.get("odds_missing_flag", False).fillna(False).astype(bool).astype(int)
    )
    output["M_win_pool_total_t5"] = pd.to_numeric(
        output.get("win_pool_total_100yen_t5"), errors="coerce"
    )

    output = output.reset_index(drop=True).copy()
    output["row_idx"] = output.index.astype(int)
    points = _fetch_market_points(db, output, minutes=minutes)
    if points.empty:
        logger.warning("market points unavailable; derived features will be NaN")
        for minute in minutes:
            output[f"M_odds_tminus_{minute}"] = np.nan
        output["M_win_pool_total_tminus_60"] = np.nan
    else:
        points = points.sort_values("row_idx").reset_index(drop=True)
        join_columns = [f"M_odds_tminus_{minute}" for minute in minutes]
        lower_name_map = {str(name).lower(): str(name) for name in points.columns}
        expected_columns = ["row_idx", "M_win_pool_total_tminus_60", *join_columns]
        rename_map: dict[str, str] = {}
        for expected in expected_columns:
            source = lower_name_map.get(expected.lower())
            if source is not None and source != expected:
                rename_map[source] = expected
        if rename_map:
            points = points.rename(columns=rename_map)
        for expected in expected_columns:
            if expected not in points.columns:
                points[expected] = np.nan
        output = output.merge(
            points[["row_idx", "M_win_pool_total_tminus_60", *join_columns]],
            on="row_idx",
            how="left",
        )
        output["M_win_pool_total_tminus_60"] = pd.to_numeric(
            output["M_win_pool_total_tminus_60"], errors="coerce"
        )
        for column in join_columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    point_columns = [f"M_odds_tminus_{minute}" for minute in minutes]
    point_raw = output[point_columns].copy()
    point_filled = point_raw.ffill(axis=1)
    output[point_columns] = point_filled
    output["M_odds_series_missing_points"] = point_raw.isna().sum(axis=1).astype(int)

    x_axis = np.array([-minute for minute in minutes], dtype=float)
    slope_values: list[float] = []
    vol_values: list[float] = []
    min_values: list[float] = []
    max_values: list[float] = []
    for row in point_filled.itertuples(index=False):
        odds = np.asarray(row, dtype=float)
        mask = np.isfinite(odds) & (odds > 0)
        if mask.sum() == 0:
            slope_values.append(np.nan)
            vol_values.append(np.nan)
            min_values.append(np.nan)
            max_values.append(np.nan)
            continue
        log_values = np.log(odds[mask])
        x_valid = x_axis[mask]
        if mask.sum() >= 2:
            slope_values.append(float(np.polyfit(x_valid, log_values, 1)[0]))
            vol_values.append(float(np.std(log_values)))
        else:
            slope_values.append(np.nan)
            vol_values.append(0.0)
        min_values.append(float(np.min(odds[mask])))
        max_values.append(float(np.max(odds[mask])))

    output["M_odds_slope_log"] = slope_values
    output["M_odds_volatility_log"] = vol_values
    output["M_odds_min"] = min_values
    output["M_odds_max"] = max_values

    last_col = "M_odds_tminus_5"
    prev_col = "M_odds_tminus_10"
    last_values = pd.to_numeric(output[last_col], errors="coerce")
    prev_values = pd.to_numeric(output[prev_col], errors="coerce")
    output["M_odds_last_change"] = (last_values - prev_values) / prev_values
    output["M_odds_last_change"] = output["M_odds_last_change"].replace([np.inf, -np.inf], np.nan)
    output["M_odds_jump_flag"] = (
        pd.to_numeric(output["M_odds_last_change"], errors="coerce").abs() > 0.15
    ).astype(int)

    output["M_win_pool_growth_60to5"] = (
        output["M_win_pool_total_t5"] - output["M_win_pool_total_tminus_60"]
    )
    output["M_win_pool_growth_ratio_60to5"] = output["M_win_pool_growth_60to5"] / output[
        "M_win_pool_total_tminus_60"
    ].replace(0, np.nan)
    output["M_market_entropy_t5"] = _calc_entropy(output, prob_col="M_imp_prob_t5")
    return output.drop(columns=["row_idx"], errors="ignore")
