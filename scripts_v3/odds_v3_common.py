from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from scripts_v2.build_features_v2 import build_race_datetime

FINAL_KBN_PRIORITY = {4: 0, 3: 1, 2: 2, 1: 3}


@dataclass(frozen=True)
class OddsSnapshotSpec:
    mode: str
    odds_col: str
    raw_col: str
    norm_col: str


FINAL_SPEC = OddsSnapshotSpec(
    mode="final",
    odds_col="odds_win_final",
    raw_col="p_win_odds_final_raw",
    norm_col="p_win_odds_final_norm",
)
T10_SPEC = OddsSnapshotSpec(
    mode="t10",
    odds_col="odds_win_t10",
    raw_col="p_win_odds_t10_raw",
    norm_col="p_win_odds_t10_norm",
)


def _parse_announce_datetime(
    race_datetime: pd.Series,
    announce_mmddhhmi: pd.Series,
) -> pd.Series:
    race_dt = pd.to_datetime(race_datetime, errors="coerce")
    ann = announce_mmddhhmi.fillna("").astype(str).str.zfill(8)

    mm = pd.to_numeric(ann.str.slice(0, 2), errors="coerce")
    dd = pd.to_numeric(ann.str.slice(2, 4), errors="coerce")
    hh = pd.to_numeric(ann.str.slice(4, 6), errors="coerce")
    mi = pd.to_numeric(ann.str.slice(6, 8), errors="coerce")

    year = race_dt.dt.year
    date_str = (
        year.astype("Int64").astype(str)
        + "-"
        + mm.astype("Int64").astype(str).str.zfill(2)
        + "-"
        + dd.astype("Int64").astype(str).str.zfill(2)
        + " "
        + hh.astype("Int64").astype(str).str.zfill(2)
        + ":"
        + mi.astype("Int64").astype(str).str.zfill(2)
        + ":00"
    )
    announce_dt = pd.to_datetime(date_str, errors="coerce")

    race_day = race_dt.dt.normalize()
    ann_day = announce_dt.dt.normalize()
    delta_days = (ann_day - race_day).dt.days

    over = delta_days > 180
    under = delta_days < -180
    if over.any():
        announce_dt.loc[over] = announce_dt.loc[over] - pd.DateOffset(years=1)
    if under.any():
        announce_dt.loc[under] = announce_dt.loc[under] + pd.DateOffset(years=1)

    return announce_dt


def load_o1_odds_long(
    db,
    race_ids: list[int],
    *,
    allowed_data_kbn: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()

    query = """
    SELECT
        w.race_id,
        w.horse_no,
        w.data_kbn,
        w.announce_mmddhhmi,
        w.win_odds_x10,
        r.race_date,
        r.start_time
    FROM core.o1_win w
    JOIN core.race r
      ON r.race_id = w.race_id
    WHERE w.race_id = ANY(%(race_ids)s)
      AND w.data_kbn = ANY(%(allowed_data_kbn)s)
    """
    rows = db.fetch_all(
        query,
        {
            "race_ids": race_ids,
            "allowed_data_kbn": list(allowed_data_kbn),
        },
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["race_id"] = pd.to_numeric(frame["race_id"], errors="coerce").astype("Int64")
    frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64")
    frame["data_kbn"] = pd.to_numeric(frame["data_kbn"], errors="coerce").astype("Int64")
    frame["win_odds_x10"] = pd.to_numeric(frame["win_odds_x10"], errors="coerce")
    frame = frame.dropna(subset=["race_id", "horse_no", "data_kbn"]).copy()
    frame["race_id"] = frame["race_id"].astype(int)
    frame["horse_no"] = frame["horse_no"].astype(int)
    frame["data_kbn"] = frame["data_kbn"].astype(int)

    frame["race_datetime"] = build_race_datetime(frame["race_date"], frame["start_time"])
    frame["asof_t10"] = frame["race_datetime"] - timedelta(minutes=10)
    frame["announce_datetime"] = _parse_announce_datetime(
        frame["race_datetime"],
        frame["announce_mmddhhmi"],
    )
    frame["odds_win"] = frame["win_odds_x10"] / 10.0
    return frame


def select_final_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    if odds_long.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "odds_win_final",
                "odds_final_data_kbn",
                "odds_final_announce_dt",
            ]
        )

    work = odds_long.copy()
    work = work[
        work["announce_datetime"].notna()
        & work["race_datetime"].notna()
        & (work["announce_datetime"] <= work["race_datetime"])
        & work["odds_win"].notna()
        & (work["odds_win"] > 0.0)
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "odds_win_final",
                "odds_final_data_kbn",
                "odds_final_announce_dt",
            ]
        )

    work["_priority"] = work["data_kbn"].map(FINAL_KBN_PRIORITY).fillna(999).astype(int)
    work = work.sort_values(
        ["race_id", "horse_no", "_priority", "announce_datetime"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    chosen = work.drop_duplicates(["race_id", "horse_no"], keep="first")
    return chosen[["race_id", "horse_no", "odds_win", "data_kbn", "announce_datetime"]].rename(
        columns={
            "odds_win": "odds_win_final",
            "data_kbn": "odds_final_data_kbn",
            "announce_datetime": "odds_final_announce_dt",
        }
    )


def select_t10_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    if odds_long.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "odds_win_t10",
                "odds_t10_data_kbn",
                "odds_t10_announce_dt",
                "odds_t10_asof_dt",
            ]
        )

    work = odds_long.copy()
    work = work[
        work["announce_datetime"].notna()
        & work["asof_t10"].notna()
        & (work["announce_datetime"] <= work["asof_t10"])
        & work["odds_win"].notna()
        & (work["odds_win"] > 0.0)
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "odds_win_t10",
                "odds_t10_data_kbn",
                "odds_t10_announce_dt",
                "odds_t10_asof_dt",
            ]
        )

    work = work.sort_values(
        ["race_id", "horse_no", "announce_datetime", "data_kbn"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    chosen = work.drop_duplicates(["race_id", "horse_no"], keep="first")
    return chosen[
        ["race_id", "horse_no", "odds_win", "data_kbn", "announce_datetime", "asof_t10"]
    ].rename(
        columns={
            "odds_win": "odds_win_t10",
            "data_kbn": "odds_t10_data_kbn",
            "announce_datetime": "odds_t10_announce_dt",
            "asof_t10": "odds_t10_asof_dt",
        }
    )


def add_implied_probability_columns(df: pd.DataFrame, spec: OddsSnapshotSpec) -> pd.DataFrame:
    out = df.copy()
    odds = pd.to_numeric(out[spec.odds_col], errors="coerce")
    out[spec.raw_col] = np.where(odds > 0.0, 1.0 / odds, np.nan)
    race_sum = out.groupby("race_id", sort=False)[spec.raw_col].transform(
        lambda s: float(s.sum(min_count=1))
    )
    out[spec.norm_col] = np.where(race_sum > 0.0, out[spec.raw_col] / race_sum, np.nan)
    return out


def merge_odds_features(features: pd.DataFrame, odds_long: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    final_df = select_final_snapshot(odds_long)
    t10_df = select_t10_snapshot(odds_long)

    if not final_df.empty:
        out = out.merge(final_df, on=["race_id", "horse_no"], how="left")
    else:
        out["odds_win_final"] = np.nan
        out["odds_final_data_kbn"] = np.nan
        out["odds_final_announce_dt"] = pd.NaT

    if not t10_df.empty:
        out = out.merge(t10_df, on=["race_id", "horse_no"], how="left")
    else:
        out["odds_win_t10"] = np.nan
        out["odds_t10_data_kbn"] = np.nan
        out["odds_t10_announce_dt"] = pd.NaT
        out["odds_t10_asof_dt"] = pd.NaT

    out = add_implied_probability_columns(out, FINAL_SPEC)
    out = add_implied_probability_columns(out, T10_SPEC)
    return out


def assert_t10_no_future_reference(
    frame: pd.DataFrame,
    *,
    announce_col: str = "odds_t10_announce_dt",
    asof_col: str = "odds_t10_asof_dt",
) -> None:
    if announce_col not in frame.columns or asof_col not in frame.columns:
        return
    announce = pd.to_datetime(frame[announce_col], errors="coerce")
    asof = pd.to_datetime(frame[asof_col], errors="coerce")
    invalid = announce.notna() & asof.notna() & (announce > asof)
    if invalid.any():
        raise ValueError("as-of violation: t10 odds references future announce time")
