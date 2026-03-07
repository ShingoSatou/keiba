from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from scripts_v3.metrics_benter_v3_common import logit_clip
from scripts_v3.v3_common import build_race_datetime

FINAL_KBN_PRIORITY = {4: 0, 3: 1, 2: 2, 1: 3}
DEFAULT_ALLOWED_DATA_KBN = (1, 2, 3, 4)
SNAPSHOT_MINUTES = (20, 15, 10)


@dataclass(frozen=True)
class OddsSnapshotSpec:
    mode: str
    odds_col: str
    raw_col: str | None
    norm_col: str


FINAL_SPEC = OddsSnapshotSpec(
    mode="final",
    odds_col="odds_win_final",
    raw_col="p_win_odds_final_raw",
    norm_col="p_win_odds_final_norm",
)
T20_SPEC = OddsSnapshotSpec(
    mode="t20",
    odds_col="odds_win_t20",
    raw_col=None,
    norm_col="p_win_odds_t20_norm",
)
T15_SPEC = OddsSnapshotSpec(
    mode="t15",
    odds_col="odds_win_t15",
    raw_col=None,
    norm_col="p_win_odds_t15_norm",
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


def _build_asof_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["race_datetime"] = build_race_datetime(out["race_date"], out["start_time"])
    for minutes in SNAPSHOT_MINUTES:
        out[f"asof_t{minutes}"] = out["race_datetime"] - timedelta(minutes=int(minutes))
    out["announce_datetime"] = _parse_announce_datetime(
        out["race_datetime"],
        out["announce_mmddhhmi"],
    )
    return out


def load_o1_win_odds_long(
    db,
    race_ids: list[int],
    *,
    allowed_data_kbn: tuple[int, ...] = DEFAULT_ALLOWED_DATA_KBN,
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
    frame = _build_asof_columns(frame)
    frame["odds_win"] = frame["win_odds_x10"] / 10.0
    return frame


def load_o1_odds_long(
    db,
    race_ids: list[int],
    *,
    allowed_data_kbn: tuple[int, ...] = DEFAULT_ALLOWED_DATA_KBN,
) -> pd.DataFrame:
    return load_o1_win_odds_long(
        db,
        race_ids,
        allowed_data_kbn=allowed_data_kbn,
    )


def load_o1_place_odds_long(
    db,
    race_ids: list[int],
    *,
    allowed_data_kbn: tuple[int, ...] = DEFAULT_ALLOWED_DATA_KBN,
) -> pd.DataFrame:
    if not race_ids:
        return pd.DataFrame()

    query = """
    SELECT
        p.race_id,
        p.horse_no,
        p.data_kbn,
        p.announce_mmddhhmi,
        p.min_odds_x10,
        p.max_odds_x10,
        r.race_date,
        r.start_time
    FROM core.o1_place p
    JOIN core.race r
      ON r.race_id = p.race_id
    WHERE p.race_id = ANY(%(race_ids)s)
      AND p.data_kbn = ANY(%(allowed_data_kbn)s)
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
    frame["min_odds_x10"] = pd.to_numeric(frame["min_odds_x10"], errors="coerce")
    frame["max_odds_x10"] = pd.to_numeric(frame["max_odds_x10"], errors="coerce")
    frame = frame.dropna(subset=["race_id", "horse_no", "data_kbn"]).copy()
    frame["race_id"] = frame["race_id"].astype(int)
    frame["horse_no"] = frame["horse_no"].astype(int)
    frame["data_kbn"] = frame["data_kbn"].astype(int)
    frame = _build_asof_columns(frame)
    frame["odds_place_lower"] = frame["min_odds_x10"] / 10.0
    frame["odds_place_upper"] = frame["max_odds_x10"] / 10.0
    return frame


def _select_final_snapshot(
    odds_long: pd.DataFrame,
    *,
    value_cols: list[str],
    output_names: dict[str, str],
) -> pd.DataFrame:
    if odds_long.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", *output_names.values()])

    work = odds_long.copy()
    positive_mask = np.ones(len(work), dtype=bool)
    for col in value_cols:
        values = pd.to_numeric(work[col], errors="coerce")
        positive_mask &= values.notna() & (values > 0.0)
    work = work[
        work["announce_datetime"].notna()
        & work["race_datetime"].notna()
        & (work["announce_datetime"] <= work["race_datetime"])
        & positive_mask
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", *output_names.values()])

    work["_priority"] = work["data_kbn"].map(FINAL_KBN_PRIORITY).fillna(999).astype(int)
    work = work.sort_values(
        ["race_id", "horse_no", "_priority", "announce_datetime"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    chosen = work.drop_duplicates(["race_id", "horse_no"], keep="first")
    return chosen[["race_id", "horse_no", *value_cols, "data_kbn", "announce_datetime"]].rename(
        columns=output_names
    )


def _select_asof_snapshot(
    odds_long: pd.DataFrame,
    *,
    minutes: int,
    value_cols: list[str],
    output_names: dict[str, str],
) -> pd.DataFrame:
    asof_col = f"asof_t{int(minutes)}"
    if odds_long.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", *output_names.values()])

    work = odds_long.copy()
    positive_mask = np.ones(len(work), dtype=bool)
    for col in value_cols:
        values = pd.to_numeric(work[col], errors="coerce")
        positive_mask &= values.notna() & (values > 0.0)
    work = work[
        work["announce_datetime"].notna()
        & work[asof_col].notna()
        & (work["announce_datetime"] <= work[asof_col])
        & positive_mask
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", *output_names.values()])

    work = work.sort_values(
        ["race_id", "horse_no", "announce_datetime", "data_kbn"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    chosen = work.drop_duplicates(["race_id", "horse_no"], keep="first")
    keep_cols = [
        "race_id",
        "horse_no",
        *value_cols,
        "data_kbn",
        "announce_datetime",
        asof_col,
    ]
    return chosen[keep_cols].rename(columns=output_names)


def select_final_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_final_snapshot(
        odds_long,
        value_cols=["odds_win"],
        output_names={
            "odds_win": "odds_win_final",
            "data_kbn": "odds_final_data_kbn",
            "announce_datetime": "odds_final_announce_dt",
        },
    )


def select_t20_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_asof_snapshot(
        odds_long,
        minutes=20,
        value_cols=["odds_win"],
        output_names={
            "odds_win": "odds_win_t20",
            "data_kbn": "odds_win_t20_data_kbn",
            "announce_datetime": "odds_win_t20_announce_dt",
            "asof_t20": "odds_win_t20_asof_dt",
        },
    )


def select_t15_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_asof_snapshot(
        odds_long,
        minutes=15,
        value_cols=["odds_win"],
        output_names={
            "odds_win": "odds_win_t15",
            "data_kbn": "odds_win_t15_data_kbn",
            "announce_datetime": "odds_win_t15_announce_dt",
            "asof_t15": "odds_win_t15_asof_dt",
        },
    )


def select_t10_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_asof_snapshot(
        odds_long,
        minutes=10,
        value_cols=["odds_win"],
        output_names={
            "odds_win": "odds_win_t10",
            "data_kbn": "odds_t10_data_kbn",
            "announce_datetime": "odds_t10_announce_dt",
            "asof_t10": "odds_t10_asof_dt",
        },
    )


def select_place_t20_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_asof_snapshot(
        odds_long,
        minutes=20,
        value_cols=["odds_place_lower", "odds_place_upper"],
        output_names={
            "odds_place_lower": "odds_place_t20_lower",
            "odds_place_upper": "odds_place_t20_upper",
            "data_kbn": "odds_place_t20_data_kbn",
            "announce_datetime": "odds_place_t20_announce_dt",
            "asof_t20": "odds_place_t20_asof_dt",
        },
    )


def select_place_t15_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_asof_snapshot(
        odds_long,
        minutes=15,
        value_cols=["odds_place_lower", "odds_place_upper"],
        output_names={
            "odds_place_lower": "odds_place_t15_lower",
            "odds_place_upper": "odds_place_t15_upper",
            "data_kbn": "odds_place_t15_data_kbn",
            "announce_datetime": "odds_place_t15_announce_dt",
            "asof_t15": "odds_place_t15_asof_dt",
        },
    )


def select_place_t10_snapshot(odds_long: pd.DataFrame) -> pd.DataFrame:
    return _select_asof_snapshot(
        odds_long,
        minutes=10,
        value_cols=["odds_place_lower", "odds_place_upper"],
        output_names={
            "odds_place_lower": "odds_place_t10_lower",
            "odds_place_upper": "odds_place_t10_upper",
            "data_kbn": "odds_place_t10_data_kbn",
            "announce_datetime": "odds_place_t10_announce_dt",
            "asof_t10": "odds_place_t10_asof_dt",
        },
    )


def add_implied_probability_columns(df: pd.DataFrame, spec: OddsSnapshotSpec) -> pd.DataFrame:
    out = df.copy()
    odds = pd.to_numeric(out[spec.odds_col], errors="coerce")
    implied = pd.Series(np.where(odds > 0.0, 1.0 / odds, np.nan), index=out.index, dtype=float)
    if spec.raw_col is not None:
        out[spec.raw_col] = implied
    race_sum = implied.groupby(out["race_id"], sort=False).transform(
        lambda s: float(s.sum(min_count=1))
    )
    out[spec.norm_col] = np.where(race_sum > 0.0, implied / race_sum, np.nan)
    return out


def _ensure_columns(frame: pd.DataFrame, columns: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    for col, default in columns.items():
        if col not in out.columns:
            out[col] = default
    return out


def _compute_place_width_log_ratio(
    frame: pd.DataFrame,
    lower_col: str,
    upper_col: str,
) -> pd.Series:
    lower = pd.to_numeric(frame[lower_col], errors="coerce")
    upper = pd.to_numeric(frame[upper_col], errors="coerce")
    low = np.minimum(lower, upper)
    high = np.maximum(lower, upper)
    valid = low.notna() & high.notna() & (low > 0.0) & (high > 0.0)
    values = np.full(len(frame), np.nan, dtype=float)
    values[valid.to_numpy()] = np.log(
        high[valid].to_numpy(dtype=float) / low[valid].to_numpy(dtype=float)
    )
    return pd.Series(values, index=frame.index, dtype=float)


def _compute_place_mid_prob(
    frame: pd.DataFrame,
    lower_col: str,
    upper_col: str,
) -> pd.Series:
    lower = pd.to_numeric(frame[lower_col], errors="coerce")
    upper = pd.to_numeric(frame[upper_col], errors="coerce")
    low = np.minimum(lower, upper)
    high = np.maximum(lower, upper)
    valid = low.notna() & high.notna() & (low > 0.0) & (high > 0.0)
    values = np.full(len(frame), np.nan, dtype=float)
    values[valid.to_numpy()] = 1.0 / np.sqrt(
        low[valid].to_numpy(dtype=float) * high[valid].to_numpy(dtype=float)
    )
    return pd.Series(values, index=frame.index, dtype=float)


def _compute_logit_delta(
    frame: pd.DataFrame,
    newer_col: str,
    older_col: str,
    *,
    eps: float = 1e-6,
) -> pd.Series:
    newer = pd.to_numeric(frame[newer_col], errors="coerce")
    older = pd.to_numeric(frame[older_col], errors="coerce")
    newer_logit = pd.Series(logit_clip(newer.to_numpy(dtype=float), eps=eps), index=frame.index)
    older_logit = pd.Series(logit_clip(older.to_numpy(dtype=float), eps=eps), index=frame.index)
    return (newer_logit - older_logit).astype(float)


def merge_odds_features(
    features: pd.DataFrame,
    win_odds_long: pd.DataFrame,
    place_odds_long: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = features.copy()

    final_df = select_final_snapshot(win_odds_long)
    win_t20_df = select_t20_snapshot(win_odds_long)
    win_t15_df = select_t15_snapshot(win_odds_long)
    win_t10_df = select_t10_snapshot(win_odds_long)

    place_long = place_odds_long if place_odds_long is not None else pd.DataFrame()
    place_t20_df = select_place_t20_snapshot(place_long)
    place_t15_df = select_place_t15_snapshot(place_long)
    place_t10_df = select_place_t10_snapshot(place_long)

    for snapshot_df in (
        final_df,
        win_t20_df,
        win_t15_df,
        win_t10_df,
        place_t20_df,
        place_t15_df,
        place_t10_df,
    ):
        if snapshot_df.empty:
            continue
        out = out.merge(snapshot_df, on=["race_id", "horse_no"], how="left")

    out = _ensure_columns(
        out,
        {
            "odds_win_final": np.nan,
            "odds_final_data_kbn": np.nan,
            "odds_final_announce_dt": pd.NaT,
            "odds_win_t20": np.nan,
            "odds_win_t20_data_kbn": np.nan,
            "odds_win_t20_announce_dt": pd.NaT,
            "odds_win_t20_asof_dt": pd.NaT,
            "odds_win_t15": np.nan,
            "odds_win_t15_data_kbn": np.nan,
            "odds_win_t15_announce_dt": pd.NaT,
            "odds_win_t15_asof_dt": pd.NaT,
            "odds_win_t10": np.nan,
            "odds_t10_data_kbn": np.nan,
            "odds_t10_announce_dt": pd.NaT,
            "odds_t10_asof_dt": pd.NaT,
            "odds_place_t20_lower": np.nan,
            "odds_place_t20_upper": np.nan,
            "odds_place_t20_data_kbn": np.nan,
            "odds_place_t20_announce_dt": pd.NaT,
            "odds_place_t20_asof_dt": pd.NaT,
            "odds_place_t15_lower": np.nan,
            "odds_place_t15_upper": np.nan,
            "odds_place_t15_data_kbn": np.nan,
            "odds_place_t15_announce_dt": pd.NaT,
            "odds_place_t15_asof_dt": pd.NaT,
            "odds_place_t10_lower": np.nan,
            "odds_place_t10_upper": np.nan,
            "odds_place_t10_data_kbn": np.nan,
            "odds_place_t10_announce_dt": pd.NaT,
            "odds_place_t10_asof_dt": pd.NaT,
        },
    )

    out = add_implied_probability_columns(out, FINAL_SPEC)
    out = add_implied_probability_columns(out, T20_SPEC)
    out = add_implied_probability_columns(out, T15_SPEC)
    out = add_implied_probability_columns(out, T10_SPEC)

    out["d_logit_win_15_20"] = _compute_logit_delta(
        out,
        "p_win_odds_t15_norm",
        "p_win_odds_t20_norm",
    )
    out["d_logit_win_10_15"] = _compute_logit_delta(
        out,
        "p_win_odds_t10_norm",
        "p_win_odds_t15_norm",
    )
    out["d_logit_win_10_20"] = _compute_logit_delta(
        out,
        "p_win_odds_t10_norm",
        "p_win_odds_t20_norm",
    )

    for minutes in SNAPSHOT_MINUTES:
        out[f"place_mid_prob_t{minutes}"] = _compute_place_mid_prob(
            out,
            f"odds_place_t{minutes}_lower",
            f"odds_place_t{minutes}_upper",
        )
        out[f"place_width_log_ratio_t{minutes}"] = _compute_place_width_log_ratio(
            out,
            f"odds_place_t{minutes}_lower",
            f"odds_place_t{minutes}_upper",
        )
    out["place_width_log_ratio"] = out["place_width_log_ratio_t10"]
    out["d_place_mid_10_20"] = (
        pd.to_numeric(out["place_mid_prob_t10"], errors="coerce")
        - pd.to_numeric(out["place_mid_prob_t20"], errors="coerce")
    )
    out["d_place_width_10_20"] = (
        pd.to_numeric(out["place_width_log_ratio_t10"], errors="coerce")
        - pd.to_numeric(out["place_width_log_ratio_t20"], errors="coerce")
    )
    return out


def _assert_announce_not_future(
    frame: pd.DataFrame,
    *,
    announce_col: str,
    asof_col: str,
) -> None:
    if announce_col not in frame.columns or asof_col not in frame.columns:
        return
    announce = pd.to_datetime(frame[announce_col], errors="coerce")
    asof = pd.to_datetime(frame[asof_col], errors="coerce")
    invalid = announce.notna() & asof.notna() & (announce > asof)
    if invalid.any():
        raise ValueError(
            "as-of violation: "
            f"{announce_col} references future announce time relative to {asof_col}"
        )


def assert_asof_no_future_reference(frame: pd.DataFrame) -> None:
    for announce_col, asof_col in (
        ("odds_win_t20_announce_dt", "odds_win_t20_asof_dt"),
        ("odds_win_t15_announce_dt", "odds_win_t15_asof_dt"),
        ("odds_t10_announce_dt", "odds_t10_asof_dt"),
        ("odds_place_t20_announce_dt", "odds_place_t20_asof_dt"),
        ("odds_place_t15_announce_dt", "odds_place_t15_asof_dt"),
        ("odds_place_t10_announce_dt", "odds_place_t10_asof_dt"),
    ):
        _assert_announce_not_future(
            frame,
            announce_col=announce_col,
            asof_col=asof_col,
        )


def assert_t10_no_future_reference(
    frame: pd.DataFrame,
    *,
    announce_col: str = "odds_t10_announce_dt",
    asof_col: str = "odds_t10_asof_dt",
) -> None:
    _assert_announce_not_future(
        frame,
        announce_col=announce_col,
        asof_col=asof_col,
    )


__all__ = [
    "FINAL_SPEC",
    "SNAPSHOT_MINUTES",
    "T15_SPEC",
    "T20_SPEC",
    "T10_SPEC",
    "add_implied_probability_columns",
    "assert_asof_no_future_reference",
    "assert_t10_no_future_reference",
    "load_o1_odds_long",
    "load_o1_place_odds_long",
    "load_o1_win_odds_long",
    "merge_odds_features",
    "select_final_snapshot",
    "select_place_t10_snapshot",
    "select_place_t15_snapshot",
    "select_place_t20_snapshot",
    "select_t10_snapshot",
    "select_t15_snapshot",
    "select_t20_snapshot",
]
