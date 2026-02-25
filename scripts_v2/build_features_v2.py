#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import deque
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402

logger = logging.getLogger(__name__)

DISTANCE_BUCKETS = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 3000, 3200, 3600]
BASE_DISTANCE_M = 1440.0
WINDOW_2Y_DAYS = 730
WINDOW_6M_DAYS = 183
PRIOR_TOP3_RATE = 0.17
PRIOR_N = 20.0
SHRINKAGE_K = 20.0
DEFAULT_HISTORY_DAYS = 730
DEFAULT_ALLOWED_CONDITION_CODES = {5, 10, 16, 999}


def distance_to_bucket(distance_m: float | int | None) -> float:
    if distance_m is None or pd.isna(distance_m):
        return np.nan
    value = float(distance_m)
    if value <= 0:
        return np.nan
    for bucket in DISTANCE_BUCKETS:
        if value <= bucket:
            return float(bucket)
    return float(DISTANCE_BUCKETS[-1])


def going_to_bucket(going: float | int | None) -> float:
    if going is None or pd.isna(going):
        return np.nan
    return 1.0 if float(going) <= 2 else 2.0


def _time_to_seconds(value: object) -> int:
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
    race_date = pd.to_datetime(race_date_series, errors="coerce")
    start_seconds = start_time_series.map(_time_to_seconds)
    return race_date + pd.to_timedelta(start_seconds, unit="s")


def _numeric_code(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_code(raw_series: pd.Series) -> pd.Series:
    extracted = raw_series.fillna("").astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    std = valid.std(ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        zeros = pd.Series(0.0, index=series.index, dtype=float)
        return zeros.where(series.notna(), np.nan)
    mean = valid.mean()
    out = (series - mean) / std
    return out.where(series.notna(), np.nan)


def _time_window_stats_by_group(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
    window_days: int,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result[f"{prefix}_mean"] = np.nan
    result[f"{prefix}_std"] = np.nan
    result[f"{prefix}_count"] = np.nan
    window = f"{window_days}D"

    for idx in frame.groupby(group_cols, dropna=False).groups.values():
        sub = frame.loc[list(idx)].sort_values("race_datetime")
        series = pd.to_numeric(sub[value_col], errors="coerce")
        rolling = series.set_axis(sub["race_datetime"]).rolling(window, closed="left")
        result.loc[sub.index, f"{prefix}_mean"] = rolling.mean().to_numpy()
        result.loc[sub.index, f"{prefix}_std"] = rolling.std(ddof=0).to_numpy()
        result.loc[sub.index, f"{prefix}_count"] = rolling.count().to_numpy()

    return result


def _shrinkage_mean(
    fine_mean: pd.Series,
    fine_count: pd.Series,
    coarse_mean: pd.Series,
    k: float,
) -> pd.Series:
    count = fine_count.fillna(0.0)
    weight = count / (count + k)
    fine_filled = fine_mean.fillna(coarse_mean)
    return weight * fine_filled + (1.0 - weight) * coarse_mean


def _daily_bias_shifted_mean(
    winners: pd.DataFrame,
    bias_col: str,
    out_col: str,
) -> pd.DataFrame:
    winners[out_col] = np.nan
    group_cols = ["race_date", "track_code", "surface"]
    for idx in winners.groupby(group_cols, dropna=False).groups.values():
        sub = winners.loc[list(idx)].sort_values(["race_datetime", "race_id"])
        shifted = sub[bias_col].expanding().mean().shift(1)
        winners.loc[sub.index, out_col] = shifted.to_numpy()
    return winners


def _compute_speed_baseline(df: pd.DataFrame) -> pd.DataFrame:
    race_cols = ["race_id", "race_datetime", "race_date", "track_code", "surface"]
    winners = df[df["finish_pos"] == 1].copy()
    if winners.empty:
        df["baseline_time_sec"] = np.nan
        df["baseline_3f_sec"] = np.nan
        df["daily_time_bias"] = np.nan
        df["daily_up3_bias"] = np.nan
        df["perf_speed_index"] = np.nan
        df["perf_up3_index"] = np.nan
        return df

    winners = winners.sort_values(["race_datetime", "race_id", "horse_no"])
    winners = winners.drop_duplicates("race_id", keep="first")
    winners["time_sec"] = pd.to_numeric(winners["time_sec"], errors="coerce")
    winners["final3f_sec"] = pd.to_numeric(winners["final3f_sec"], errors="coerce")
    winners["distance_bucket"] = winners["distance_m"].map(distance_to_bucket)
    winners["going_bucket"] = winners["going"].map(going_to_bucket)

    fine_groups = ["track_code", "surface", "distance_bucket", "going_bucket"]
    coarse_groups = ["surface", "distance_bucket"]

    fine_time = _time_window_stats_by_group(
        winners, fine_groups, "time_sec", "fine_time", WINDOW_2Y_DAYS
    )
    coarse_time = _time_window_stats_by_group(
        winners, coarse_groups, "time_sec", "coarse_time", WINDOW_2Y_DAYS
    )
    fine_3f = _time_window_stats_by_group(
        winners, fine_groups, "final3f_sec", "fine_3f", WINDOW_2Y_DAYS
    )
    coarse_3f = _time_window_stats_by_group(
        winners, coarse_groups, "final3f_sec", "coarse_3f", WINDOW_2Y_DAYS
    )
    winners = winners.join(fine_time).join(coarse_time).join(fine_3f).join(coarse_3f)

    winners["baseline_time_sec"] = _shrinkage_mean(
        winners["fine_time_mean"],
        winners["fine_time_count"],
        winners["coarse_time_mean"],
        SHRINKAGE_K,
    )
    winners["baseline_3f_sec"] = _shrinkage_mean(
        winners["fine_3f_mean"],
        winners["fine_3f_count"],
        winners["coarse_3f_mean"],
        SHRINKAGE_K,
    )
    winners["time_bias"] = winners["time_sec"] - winners["baseline_time_sec"]
    winners["up3_bias"] = winners["final3f_sec"] - winners["baseline_3f_sec"]
    winners = _daily_bias_shifted_mean(winners, "time_bias", "daily_time_bias")
    winners = _daily_bias_shifted_mean(winners, "up3_bias", "daily_up3_bias")

    race_level = winners[
        race_cols + ["baseline_time_sec", "baseline_3f_sec", "daily_time_bias", "daily_up3_bias"]
    ]
    df = df.merge(race_level, on=race_cols, how="left")
    df["base_time_sec"] = df["baseline_time_sec"] + df["daily_time_bias"].fillna(0.0)
    df["base_3f_sec"] = df["baseline_3f_sec"] + df["daily_up3_bias"].fillna(0.0)

    distance = _numeric_code(df["distance_m"]).replace(0, np.nan)
    distance_correction = BASE_DISTANCE_M / distance
    time_sec = _numeric_code(df["time_sec"])
    final3f_sec = _numeric_code(df["final3f_sec"])
    df["perf_speed_index"] = (df["base_time_sec"] - time_sec) * distance_correction + 80.0
    df["perf_up3_index"] = (df["base_3f_sec"] - final3f_sec) * distance_correction + 80.0
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["horse_id", "race_datetime", "race_id"]).copy()
    horse_group = df.groupby("horse_id", sort=False)

    for lag in (1, 2, 3):
        df[f"lag{lag}_finish_pos"] = horse_group["finish_pos"].shift(lag)
        df[f"lag{lag}_time_sec"] = horse_group["time_sec"].shift(lag)
        df[f"lag{lag}_final3f_sec"] = horse_group["final3f_sec"].shift(lag)
        df[f"lag{lag}_speed_index"] = horse_group["perf_speed_index"].shift(lag)
        df[f"lag{lag}_up3_index"] = horse_group["perf_up3_index"].shift(lag)
        df[f"lag{lag}_distance_m"] = horse_group["distance_m"].shift(lag)
        df[f"lag{lag}_surface"] = horse_group["surface"].shift(lag)

    df["lag1_race_datetime"] = horse_group["race_datetime"].shift(1)
    df["lag1_jockey_key"] = horse_group["jockey_key"].shift(1)
    df["days_since_lag1"] = (
        pd.to_datetime(df["race_datetime"]) - pd.to_datetime(df["lag1_race_datetime"])
    ).dt.days
    df["lag1_distance_diff"] = _numeric_code(df["distance_m"]) - _numeric_code(
        df["lag1_distance_m"]
    )
    df["lag1_course_type_match"] = (
        _numeric_code(df["surface"]) == _numeric_code(df["lag1_surface"])
    ).astype(float)
    df.loc[df["lag1_surface"].isna(), "lag1_course_type_match"] = np.nan
    df["is_jockey_change"] = 0
    valid = df["lag1_jockey_key"].notna() & (_numeric_code(df["jockey_key"]) > 0)
    df.loc[valid, "is_jockey_change"] = (
        _numeric_code(df.loc[valid, "jockey_key"])
        != _numeric_code(df.loc[valid, "lag1_jockey_key"])
    ).astype(int)
    return df


def _compute_aptitude_features(df: pd.DataFrame) -> pd.DataFrame:
    same_distance_rate = pd.Series(np.nan, index=df.index, dtype=float)
    same_condition_rate = pd.Series(np.nan, index=df.index, dtype=float)
    top3 = (_numeric_code(df["finish_pos"]) <= 3).astype(int).fillna(0)

    for idx in df.groupby("horse_id", sort=False).groups.values():
        sub = df.loc[list(idx)].sort_values("race_datetime")
        history: deque[tuple[pd.Timestamp, float, float, int]] = deque()
        for row in sub.itertuples():
            race_dt = pd.Timestamp(row.race_datetime)
            min_dt = race_dt - timedelta(days=WINDOW_2Y_DAYS)
            while history and history[0][0] < min_dt:
                history.popleft()

            dist_count = 0
            dist_top3 = 0
            cond_count = 0
            cond_top3 = 0
            cur_distance = float(row.distance_m) if pd.notna(row.distance_m) else np.nan
            cur_going_bucket = float(row.going_bucket) if pd.notna(row.going_bucket) else np.nan
            for hist_dt, hist_distance, hist_going_bucket, hist_top3 in history:
                _ = hist_dt
                if pd.notna(cur_distance) and hist_distance == cur_distance:
                    dist_count += 1
                    dist_top3 += hist_top3
                if pd.notna(cur_going_bucket) and hist_going_bucket == cur_going_bucket:
                    cond_count += 1
                    cond_top3 += hist_top3

            same_distance_rate.loc[row.Index] = (dist_top3 + PRIOR_N * PRIOR_TOP3_RATE) / (
                dist_count + PRIOR_N
            )
            same_condition_rate.loc[row.Index] = (cond_top3 + PRIOR_N * PRIOR_TOP3_RATE) / (
                cond_count + PRIOR_N
            )

            hist_distance = float(row.distance_m) if pd.notna(row.distance_m) else np.nan
            hist_going_bucket = float(row.going_bucket) if pd.notna(row.going_bucket) else np.nan
            history.append((race_dt, hist_distance, hist_going_bucket, int(top3.loc[row.Index])))

    df["apt_same_distance_top3_rate_2y"] = same_distance_rate
    df["apt_same_going_top3_rate_2y"] = same_condition_rate
    return df


def _compute_recent_entity_rate(df: pd.DataFrame, key_col: str, out_col: str) -> pd.DataFrame:
    key_numeric = _numeric_code(df[key_col]).fillna(0).astype(int)
    data = df.copy()
    data[key_col] = key_numeric
    data["race_date_dt"] = pd.to_datetime(data["race_date"], errors="coerce")
    valid = data[key_col] > 0

    if not valid.any():
        data[out_col] = PRIOR_TOP3_RATE
        return data.drop(columns=["race_date_dt"])

    daily = (
        data.loc[valid, [key_col, "race_date_dt", "is_top3"]]
        .groupby([key_col, "race_date_dt"], as_index=False)
        .agg(top3_sum=("is_top3", "sum"), starts=("is_top3", "size"))
    )
    daily["rate"] = np.nan
    for idx in daily.groupby(key_col, sort=False).groups.values():
        sub = daily.loc[list(idx)].sort_values("race_date_dt")
        sum_series = sub["top3_sum"].set_axis(sub["race_date_dt"])
        cnt_series = sub["starts"].set_axis(sub["race_date_dt"])
        roll_sum = sum_series.rolling(f"{WINDOW_6M_DAYS}D", closed="left").sum()
        roll_cnt = cnt_series.rolling(f"{WINDOW_6M_DAYS}D", closed="left").sum()
        rate = (roll_sum + PRIOR_N * PRIOR_TOP3_RATE) / (roll_cnt + PRIOR_N)
        daily.loc[sub.index, "rate"] = rate.to_numpy()

    data = data.merge(
        daily[[key_col, "race_date_dt", "rate"]], on=[key_col, "race_date_dt"], how="left"
    )
    data[out_col] = data["rate"].fillna(PRIOR_TOP3_RATE)
    return data.drop(columns=["race_date_dt", "rate"])


def _compute_recent_entity_target_mean(
    df: pd.DataFrame,
    key_col: str,
    target_col: str,
    out_col: str,
    *,
    prior_mean: float,
) -> pd.DataFrame:
    key_numeric = _numeric_code(df[key_col]).fillna(0).astype(int)
    data = df.copy()
    data[key_col] = key_numeric
    data["race_date_dt"] = pd.to_datetime(data["race_date"], errors="coerce")
    valid = data[key_col] > 0

    if not valid.any():
        data[out_col] = float(prior_mean)
        return data.drop(columns=["race_date_dt"])

    daily = (
        data.loc[valid, [key_col, "race_date_dt", target_col]]
        .groupby([key_col, "race_date_dt"], as_index=False)
        .agg(target_sum=(target_col, "sum"), starts=(target_col, "size"))
    )
    daily["mean"] = np.nan
    for idx in daily.groupby(key_col, sort=False).groups.values():
        sub = daily.loc[list(idx)].sort_values("race_date_dt")
        sum_series = pd.to_numeric(sub["target_sum"], errors="coerce").set_axis(sub["race_date_dt"])
        cnt_series = pd.to_numeric(sub["starts"], errors="coerce").set_axis(sub["race_date_dt"])
        roll_sum = sum_series.rolling(f"{WINDOW_6M_DAYS}D", closed="left").sum()
        roll_cnt = cnt_series.rolling(f"{WINDOW_6M_DAYS}D", closed="left").sum()
        mean = (roll_sum + PRIOR_N * float(prior_mean)) / (roll_cnt + PRIOR_N)
        daily.loc[sub.index, "mean"] = mean.to_numpy()

    data = data.merge(
        daily[[key_col, "race_date_dt", "mean"]], on=[key_col, "race_date_dt"], how="left"
    )
    data[out_col] = data["mean"].fillna(float(prior_mean))
    return data.drop(columns=["race_date_dt", "mean"])


def _add_relative_features(df: pd.DataFrame, *, with_te: bool) -> pd.DataFrame:
    race_group = df.groupby("race_id", sort=False)
    df["rel_lag1_speed_index_z"] = race_group["lag1_speed_index"].transform(_zscore)
    df["rel_lag1_speed_index_rank"] = race_group["lag1_speed_index"].rank(
        method="average",
        ascending=False,
    )
    df["rel_lag1_speed_index_pct"] = race_group["lag1_speed_index"].rank(
        method="average",
        ascending=False,
        pct=True,
    )
    df["rel_carried_weight_z"] = race_group["carried_weight"].transform(_zscore)
    df["rel_jockey_top3_rate_z"] = race_group["jockey_top3_rate_6m"].transform(_zscore)
    if with_te and "jockey_target_label_mean_6m" in df.columns:
        df["rel_jockey_target_label_mean_z"] = race_group["jockey_target_label_mean_6m"].transform(
            _zscore
        )
    df["rel_meta_tm_score_z"] = race_group["meta_tm_score"].transform(_zscore)
    return df


def _apply_segment_filter(df: pd.DataFrame) -> pd.DataFrame:
    race_type = _numeric_code(df["race_type_code"])
    condition = _numeric_code(df["condition_code_min_age"])
    field_size = _numeric_code(df["field_size"])
    distance = _numeric_code(df["distance_m"])
    horse_no = _numeric_code(df["horse_no"])
    finish_pos = _numeric_code(df["finish_pos"])

    mask = (
        (_numeric_code(df["track_code"]).between(1, 10))
        & (_numeric_code(df["surface"]) == 2)
        & (race_type == 14)
        & (~condition.isin([701, 702, 703]))
        & (condition.isin(DEFAULT_ALLOWED_CONDITION_CODES))
        & (distance > 0)
        & (field_size > 0)
        & (horse_no.between(1, 18))
        & (finish_pos.notna())
    )
    return df.loc[mask].copy()


def assert_no_future_leakage(df: pd.DataFrame) -> None:
    if "lag1_race_datetime" in df.columns:
        current = pd.to_datetime(df["race_datetime"])
        lag1 = pd.to_datetime(df["lag1_race_datetime"])
        invalid = lag1.notna() & (lag1 >= current)
        if invalid.any():
            raise ValueError("Leakage detected: lag1_race_datetime is not strictly past.")


def assert_sorted(df: pd.DataFrame) -> None:
    sorted_df = df.sort_values(["race_id", "horse_no"], kind="mergesort")
    left = df[["race_id", "horse_no"]].reset_index(drop=True)
    right = sorted_df[["race_id", "horse_no"]].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Output sort violation: expected race_id asc / horse_no asc.")


def _load_base_data(
    db: Database,
    history_from: date,
    to_date: date,
) -> pd.DataFrame:
    query = """
    SELECT
        r.race_id,
        r.race_date,
        r.track_code,
        r.race_no,
        r.surface,
        r.distance_m,
        r.going,
        r.weather,
        r.field_size,
        r.start_time,
        r.class_code,
        r.grade_code,
        r.race_type_code,
        r.weight_type_code,
        r.condition_code_min_age,
        run.horse_id,
        run.horse_no,
        run.gate,
        run.jockey_id,
        run.trainer_id,
        run.carried_weight,
        run.body_weight,
        run.body_weight_diff,
        run.jockey_code_raw,
        run.trainer_code_raw,
        run.sex,
        res.finish_pos,
        res.time_sec,
        res.final3f_sec
    FROM core.race r
    JOIN core.runner run
      ON r.race_id = run.race_id
    JOIN core.result res
      ON run.race_id = res.race_id
     AND run.horse_id = res.horse_id
    WHERE r.track_code BETWEEN 1 AND 10
      AND r.race_date BETWEEN %(history_from)s AND %(to_date)s
    ORDER BY r.race_date, r.race_id, run.horse_no
    """
    rows = db.fetch_all(query, {"history_from": history_from, "to_date": to_date})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["race_datetime"] = build_race_datetime(frame["race_date"], frame["start_time"])
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.date
    frame["distance_bucket"] = frame["distance_m"].map(distance_to_bucket)
    frame["going_bucket"] = frame["going"].map(going_to_bucket)
    return frame


def _load_latest_rt_mining(
    db: Database,
    table_name: str,
    value_cols: list[str],
    history_from: date,
    to_date: date,
) -> pd.DataFrame:
    cols_sql = ", ".join(f"m.{col}" for col in value_cols)
    query = f"""
    SELECT
        m.race_id,
        m.horse_no,
        m.data_create_ymd,
        m.data_create_hm,
        {cols_sql},
        r.race_date,
        r.start_time
    FROM core.{table_name} m
    JOIN core.race r
      ON r.race_id = m.race_id
    WHERE r.race_date BETWEEN %(history_from)s AND %(to_date)s
    """
    rows = db.fetch_all(query, {"history_from": history_from, "to_date": to_date})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["race_datetime"] = build_race_datetime(frame["race_date"], frame["start_time"])
    stamp = frame["data_create_ymd"].fillna("").astype(str) + frame["data_create_hm"].fillna(
        ""
    ).astype(str).str.zfill(4)
    frame["create_datetime"] = pd.to_datetime(stamp, format="%Y%m%d%H%M", errors="coerce")
    frame = frame[frame["create_datetime"].notna()]
    frame = frame[frame["create_datetime"] <= frame["race_datetime"]]
    if frame.empty:
        return frame

    frame = frame.sort_values(
        ["race_id", "horse_no", "create_datetime"], ascending=[True, True, False]
    )
    frame = frame.drop_duplicates(["race_id", "horse_no"], keep="first")
    keep_cols = ["race_id", "horse_no", *value_cols]
    return frame[keep_cols]


def _load_mining_fallback(db: Database, table_name: str, value_cols: list[str]) -> pd.DataFrame:
    cols_sql = ", ".join(value_cols)
    query = f"SELECT race_id, horse_no, {cols_sql} FROM core.{table_name}"
    rows = db.fetch_all(query)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame[["race_id", "horse_no", *value_cols]]


def _merge_mining_features(
    df: pd.DataFrame,
    rt_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    merge_cols: list[str],
) -> pd.DataFrame:
    if rt_df.empty and fallback_df.empty:
        for col in merge_cols:
            df[col] = np.nan
        return df

    merged = df.copy()
    if not rt_df.empty:
        merged = merged.merge(rt_df, on=["race_id", "horse_no"], how="left")
    else:
        for col in merge_cols:
            merged[col] = np.nan

    if not fallback_df.empty:
        fallback = fallback_df.rename(columns={col: f"{col}_fallback" for col in merge_cols})
        merged = merged.merge(fallback, on=["race_id", "horse_no"], how="left")
        for col in merge_cols:
            merged[col] = merged[col].combine_first(merged[f"{col}_fallback"])
            merged = merged.drop(columns=[f"{col}_fallback"])
    return merged


def _target_label(series: pd.Series) -> pd.Series:
    finish = _numeric_code(series)
    label = np.where(finish == 1, 3, np.where(finish == 2, 2, np.where(finish == 3, 1, 0)))
    return pd.Series(label, index=series.index, dtype=int)


def _feature_columns(*, with_te: bool) -> list[str]:
    cols = [
        "race_id",
        "horse_id",
        "horse_no",
        "race_date",
        "track_code",
        "surface",
        "distance_m",
        "going",
        "weather",
        "field_size",
        "grade_code",
        "race_type_code",
        "weight_type_code",
        "condition_code_min_age",
        "target_label",
        "age",
        "sex",
        "carried_weight",
        "body_weight",
        "body_weight_diff",
        "jockey_key",
        "trainer_key",
        "is_jockey_change",
        "days_since_lag1",
        "lag1_distance_diff",
        "lag1_course_type_match",
        "lag1_finish_pos",
        "lag2_finish_pos",
        "lag3_finish_pos",
        "lag1_speed_index",
        "lag2_speed_index",
        "lag3_speed_index",
        "lag1_up3_index",
        "lag2_up3_index",
        "lag3_up3_index",
        "apt_same_distance_top3_rate_2y",
        "apt_same_going_top3_rate_2y",
        "meta_dm_time_x10",
        "meta_dm_rank",
        "meta_tm_score",
        "meta_tm_rank",
        "jockey_top3_rate_6m",
        "trainer_top3_rate_6m",
        "rel_lag1_speed_index_z",
        "rel_lag1_speed_index_rank",
        "rel_lag1_speed_index_pct",
        "rel_carried_weight_z",
        "rel_jockey_top3_rate_z",
        "rel_meta_tm_score_z",
    ]
    if with_te:
        cols.extend(
            [
                "jockey_target_label_mean_6m",
                "trainer_target_label_mean_6m",
                "rel_jockey_target_label_mean_z",
            ]
        )
    return cols


def build_features_dataframe(
    db: Database,
    from_date: date,
    to_date: date,
    history_days: int,
    *,
    with_te: bool,
) -> pd.DataFrame:
    history_from = from_date - timedelta(days=history_days)
    df = _load_base_data(db, history_from=history_from, to_date=to_date)
    if df.empty:
        return df

    df = df.sort_values(["race_datetime", "race_id", "horse_no"]).copy()
    numeric_cols = [
        "distance_m",
        "going",
        "weather",
        "field_size",
        "grade_code",
        "race_type_code",
        "weight_type_code",
        "condition_code_min_age",
        "horse_no",
        "carried_weight",
        "body_weight",
        "body_weight_diff",
        "sex",
        "finish_pos",
        "time_sec",
        "final3f_sec",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target_label"] = _target_label(df["finish_pos"])
    df["is_top3"] = (_numeric_code(df["finish_pos"]) <= 3).fillna(False).astype(int)
    df["jockey_key"] = _numeric_code(df["jockey_id"]).combine_first(
        _extract_code(df["jockey_code_raw"])
    )
    df["trainer_key"] = _numeric_code(df["trainer_id"]).combine_first(
        _extract_code(df["trainer_code_raw"])
    )
    df["jockey_key"] = df["jockey_key"].fillna(0).astype(int)
    df["trainer_key"] = df["trainer_key"].fillna(0).astype(int)

    race_year = pd.to_datetime(df["race_date"], errors="coerce").dt.year
    birth_year = pd.to_numeric(df["horse_id"].astype(str).str.slice(0, 4), errors="coerce")
    df["age"] = race_year - birth_year
    df.loc[(df["age"] < 0) | (df["age"] > 20), "age"] = np.nan
    df["sex"] = _numeric_code(df["sex"])

    rt_dm = _load_latest_rt_mining(
        db,
        table_name="rt_mining_dm",
        value_cols=["dm_time_x10", "dm_rank"],
        history_from=history_from,
        to_date=to_date,
    )
    core_dm = _load_mining_fallback(db, "mining_dm", ["dm_time_x10", "dm_rank"])
    df = _merge_mining_features(df, rt_dm, core_dm, ["dm_time_x10", "dm_rank"])
    df = df.rename(columns={"dm_time_x10": "meta_dm_time_x10", "dm_rank": "meta_dm_rank"})

    rt_tm = _load_latest_rt_mining(
        db,
        table_name="rt_mining_tm",
        value_cols=["tm_score", "tm_rank"],
        history_from=history_from,
        to_date=to_date,
    )
    core_tm = _load_mining_fallback(db, "mining_tm", ["tm_score", "tm_rank"])
    df = _merge_mining_features(df, rt_tm, core_tm, ["tm_score", "tm_rank"])
    df = df.rename(columns={"tm_score": "meta_tm_score", "tm_rank": "meta_tm_rank"})

    df = _compute_speed_baseline(df)
    df = _add_lag_features(df)
    assert_no_future_leakage(df)
    df = _compute_aptitude_features(df)
    df = _compute_recent_entity_rate(df, "jockey_key", "jockey_top3_rate_6m")
    df = _compute_recent_entity_rate(df, "trainer_key", "trainer_top3_rate_6m")
    if with_te:
        prior_label_mean = float(pd.to_numeric(df["target_label"], errors="coerce").mean())
        df = _compute_recent_entity_target_mean(
            df,
            "jockey_key",
            "target_label",
            "jockey_target_label_mean_6m",
            prior_mean=prior_label_mean,
        )
        df = _compute_recent_entity_target_mean(
            df,
            "trainer_key",
            "target_label",
            "trainer_target_label_mean_6m",
            prior_mean=prior_label_mean,
        )

    output_mask = (pd.to_datetime(df["race_date"]) >= pd.to_datetime(from_date)) & (
        pd.to_datetime(df["race_date"]) <= pd.to_datetime(to_date)
    )
    output_df = df.loc[output_mask].copy()
    output_df = _apply_segment_filter(output_df)
    output_df = _add_relative_features(output_df, with_te=with_te)
    output_df = output_df.sort_values(["race_id", "horse_no"], kind="mergesort")
    output_df = output_df[_feature_columns(with_te=with_te)].copy()
    assert_sorted(output_df)
    return output_df


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_outputs(
    features: pd.DataFrame,
    output_path: Path,
    meta_path: Path,
    from_date: date,
    to_date: date,
    history_days: int,
    *,
    with_te: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    missing_rate = {
        col: float(features[col].isna().mean())
        for col in ["lag1_speed_index", "lag1_up3_index", "meta_tm_score", "meta_dm_time_x10"]
        if col in features.columns
    }
    meta = {
        "from_date": str(from_date),
        "to_date": str(to_date),
        "history_days": history_days,
        "with_te": bool(with_te),
        "rows": int(len(features)),
        "races": int(features["race_id"].nunique()) if not features.empty else 0,
        "columns": features.columns.tolist(),
        "segment_filter": {
            "track_code": "01-10",
            "surface": 2,
            "race_type_code": 14,
            "condition_code_excluded": [701, 702, 703],
            "condition_code_allowed": sorted(DEFAULT_ALLOWED_CONDITION_CODES),
        },
        "speed_index": {
            "base_distance_m": BASE_DISTANCE_M,
            "window_days": WINDOW_2Y_DAYS,
            "shrinkage_k": SHRINKAGE_K,
        },
        "missing_rate": missing_rate,
        "code_hash": _hash_files([Path(__file__)]),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 features from core tables.")
    parser.add_argument("--from-date", required=True, help="Output start date (YYYY-MM-DD).")
    parser.add_argument("--to-date", required=True, help="Output end date (YYYY-MM-DD).")
    parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    parser.add_argument(
        "--with-te",
        action="store_true",
        help="Include target encoding features (jockey/trainer rolling target_label mean).",
    )
    parser.add_argument("--output", default="data/features_v2.parquet")
    parser.add_argument("--meta-output", default="data/features_v2_meta.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)
    if from_date > to_date:
        raise SystemExit("--from-date must be <= --to-date")

    output_path = (PROJECT_ROOT / args.output).resolve()
    meta_path = (PROJECT_ROOT / args.meta_output).resolve()

    with Database() as db:
        features = build_features_dataframe(
            db=db,
            from_date=from_date,
            to_date=to_date,
            history_days=args.history_days,
            with_te=bool(args.with_te),
        )

    if features.empty:
        logger.warning("No rows produced for the given period/segment.")
    logger.info("features rows=%s races=%s", len(features), features["race_id"].nunique())
    write_outputs(
        features=features,
        output_path=output_path,
        meta_path=meta_path,
        from_date=from_date,
        to_date=to_date,
        history_days=args.history_days,
        with_te=bool(args.with_te),
    )
    logger.info("wrote %s", output_path)
    logger.info("wrote %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
