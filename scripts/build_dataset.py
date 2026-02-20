"""
学習データセット生成スクリプト

mart 層の特徴量を結合し、レース内相対特徴量・ペース圧特徴量を付与して
学習用データセットを生成する。

使用方法:
    uv run python scripts/build_dataset.py

出力:
    data/train.parquet

前提:
    build_features.py が実行済みであること
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FROM_DATE = date(2016, 1, 1)

CK_FEATURE_COLUMNS = [
    "h_total_starts",
    "h_total_wins",
    "h_total_top3",
    "h_total_top5",
    "h_total_out",
    "h_central_starts",
    "h_central_wins",
    "h_central_top3",
    "h_turf_right_starts",
    "h_turf_left_starts",
    "h_turf_straight_starts",
    "h_dirt_right_starts",
    "h_dirt_left_starts",
    "h_dirt_straight_starts",
    "h_turf_good_starts",
    "h_turf_soft_starts",
    "h_turf_heavy_starts",
    "h_turf_bad_starts",
    "h_dirt_good_starts",
    "h_dirt_soft_starts",
    "h_dirt_heavy_starts",
    "h_dirt_bad_starts",
    "h_turf_16down_starts",
    "h_turf_22down_starts",
    "h_turf_22up_starts",
    "h_dirt_16down_starts",
    "h_dirt_22down_starts",
    "h_dirt_22up_starts",
    "h_style_nige_cnt",
    "h_style_senko_cnt",
    "h_style_sashi_cnt",
    "h_style_oikomi_cnt",
    "h_registered_races_n",
    "j_year_flat_prize_total",
    "j_year_ob_prize_total",
    "j_cum_flat_prize_total",
    "j_cum_ob_prize_total",
    "t_year_flat_prize_total",
    "t_year_ob_prize_total",
    "t_cum_flat_prize_total",
    "t_cum_ob_prize_total",
    "o_year_prize_total",
    "o_cum_prize_total",
    "b_year_prize_total",
    "b_cum_prize_total",
]


# =============================================================================
# ユーティリティ関数
# =============================================================================


def distance_to_bucket(distance_m: int) -> int:
    """距離をバケットに変換"""
    buckets = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 3000, 3200, 3600]
    for bucket in buckets:
        if distance_m <= bucket:
            return bucket
    return buckets[-1]


def going_to_bucket(going: int | None) -> int:
    """馬場状態をバケットに変換"""
    if going is None or pd.isna(going) or going <= 2:
        return 1  # 良系
    return 2  # 道悪系


# =============================================================================
# メインのデータセット生成
# =============================================================================


def build_dataset(
    db: Database, from_date: date | None = None, to_date: date | None = None
) -> pd.DataFrame:
    """学習用データセットを生成"""
    logger.info("データセット生成開始...")

    # 日付フィルタ
    date_filter = ""
    if from_date:
        date_filter += f" AND r.race_date >= '{from_date}'"
    if to_date:
        date_filter += f" AND r.race_date <= '{to_date}'"

    # 基本データ取得 (runner + result + race)
    base_query = f"""
    SELECT
        r.race_id,
        r.race_date,
        r.track_code,
        r.surface,
        r.distance_m,
        r.going,
        r.class_code,
        r.field_size,
        run.horse_id,
        run.horse_no,
        run.gate,
        run.jockey_id,
        run.trainer_id,
        run.carried_weight,
        run.body_weight,
        run.body_weight_diff,
        res.finish_pos,
        res.time_sec,
        res.final3f_sec
    FROM core.race r
    JOIN core.runner run ON r.race_id = run.race_id
    JOIN core.result res ON run.race_id = res.race_id AND run.horse_id = res.horse_id
    WHERE r.track_code BETWEEN 1 AND 10
      AND r.surface IN (1, 2, 3)
      AND r.distance_m > 0
      AND run.scratch_flag = FALSE
      AND res.finish_pos IS NOT NULL
      AND EXISTS (
        SELECT 1
        FROM core.result res2
        WHERE res2.race_id = r.race_id
          AND res2.finish_pos = 1
      )
      {date_filter}
    ORDER BY r.race_date, r.race_id, run.horse_no
    """
    base_df = pd.DataFrame(db.fetch_all(base_query))

    if base_df.empty:
        logger.warning("対象データがありません")
        return pd.DataFrame()

    logger.info(f"基本データ: {len(base_df)} 件")

    # Decimal を float に変換
    numeric_base_cols = [
        "carried_weight",
        "body_weight",
        "body_weight_diff",
        "time_sec",
        "final3f_sec",
    ]
    for col in numeric_base_cols:
        if col in base_df.columns:
            base_df[col] = pd.to_numeric(base_df[col], errors="coerce")

    # バケット追加
    base_df["distance_bucket"] = base_df["distance_m"].apply(distance_to_bucket)
    base_df["going_bucket"] = base_df["going"].apply(going_to_bucket)

    # horse_stats を結合
    base_df = _join_horse_stats(db, base_df)

    # person_stats を結合
    base_df = _join_person_stats(db, base_df)

    # MING(DM/TM) を結合
    base_df = _join_mining_features(db, base_df)

    # 調教特徴量を結合
    base_df = _join_training_features(db, base_df, from_date=from_date, to_date=to_date)

    # CK特徴量を結合
    base_df = _join_ck_features(db, base_df)

    # レース内相対特徴量を計算
    base_df = _add_inrace_features(base_df)

    # ペース圧特徴量を計算
    base_df = _add_pace_features(base_df)

    # 当日コンディション特徴量を計算
    base_df = _add_condition_features(db, base_df)

    # 目的変数
    base_df["is_win"] = (base_df["finish_pos"] == 1).astype(int)

    logger.info(f"最終データセット: {len(base_df)} 件")
    return base_df


def _filter_small_field_races(df: pd.DataFrame, min_horses: int) -> pd.DataFrame:
    """最低頭数を満たさないレースを除外"""
    if min_horses <= 1 or df.empty:
        return df

    race_sizes = df.groupby("race_id").size()
    valid_races = race_sizes[race_sizes >= min_horses].index
    dropped_races = len(race_sizes) - len(valid_races)
    if dropped_races > 0:
        logger.info(
            "最低頭数フィルタ: %s 頭未満のレースを %s 件除外",
            min_horses,
            dropped_races,
        )

    return df[df["race_id"].isin(valid_races)].copy()


def _join_horse_stats(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """horse_stats を結合"""
    logger.info("horse_stats を結合中...")

    # horse_stats を取得
    stats_query = """
    SELECT
        calc_date,
        horse_id,
        target_surface,
        target_distance_bucket,
        target_going_bucket,
        speed_last, speed_mean_3, speed_best_5, speed_std_5, speed_trend_3,
        closing_last, closing_mean_3, closing_best_5,
        early_mean_3, early_best_5, position_gain_mean_3,
        finish_mean_3, finish_best_5, n_runs_5,
        speed_sim_mean_3, speed_sim_best_5,
        closing_sim_mean_3, closing_sim_best_5,
        early_sim_mean_3, n_sim_runs_5
    FROM mart.horse_stats
    """
    stats_df = pd.DataFrame(db.fetch_all(stats_query))

    if stats_df.empty:
        logger.warning("horse_stats がありません")
        return df

    # Decimal を float に変換
    numeric_cols = [
        "speed_last",
        "speed_mean_3",
        "speed_best_5",
        "speed_std_5",
        "speed_trend_3",
        "closing_last",
        "closing_mean_3",
        "closing_best_5",
        "early_mean_3",
        "early_best_5",
        "position_gain_mean_3",
        "finish_mean_3",
        "finish_best_5",
        "speed_sim_mean_3",
        "speed_sim_best_5",
        "closing_sim_mean_3",
        "closing_sim_best_5",
        "early_sim_mean_3",
    ]
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    # 結合
    df = df.merge(
        stats_df,
        left_on=["race_date", "horse_id", "surface", "distance_bucket", "going_bucket"],
        right_on=[
            "calc_date",
            "horse_id",
            "target_surface",
            "target_distance_bucket",
            "target_going_bucket",
        ],
        how="left",
    )

    # 不要カラム削除
    drop_cols = ["calc_date", "target_surface", "target_distance_bucket", "target_going_bucket"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def _join_person_stats(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """person_stats を結合"""
    logger.info("person_stats を結合中...")

    # 騎手
    jockey_query = """
    SELECT calc_date, person_id as jockey_id,
           win_rate_1y as jockey_win_rate_1y,
           place_rate_1y as jockey_place_rate_1y
    FROM mart.person_stats
    WHERE person_type = 'jockey'
    """
    jockey_df = pd.DataFrame(db.fetch_all(jockey_query))

    if not jockey_df.empty:
        # Decimal を float に変換
        jockey_df["jockey_win_rate_1y"] = pd.to_numeric(
            jockey_df["jockey_win_rate_1y"], errors="coerce"
        )
        jockey_df["jockey_place_rate_1y"] = pd.to_numeric(
            jockey_df["jockey_place_rate_1y"], errors="coerce"
        )
        df = df.merge(
            jockey_df,
            left_on=["race_date", "jockey_id"],
            right_on=["calc_date", "jockey_id"],
            how="left",
        )
        df = df.drop(columns=["calc_date"], errors="ignore")

    # 調教師
    trainer_query = """
    SELECT calc_date, person_id as trainer_id,
           win_rate_1y as trainer_win_rate_1y,
           place_rate_1y as trainer_place_rate_1y
    FROM mart.person_stats
    WHERE person_type = 'trainer'
    """
    trainer_df = pd.DataFrame(db.fetch_all(trainer_query))

    if not trainer_df.empty:
        # Decimal を float に変換
        trainer_df["trainer_win_rate_1y"] = pd.to_numeric(
            trainer_df["trainer_win_rate_1y"], errors="coerce"
        )
        trainer_df["trainer_place_rate_1y"] = pd.to_numeric(
            trainer_df["trainer_place_rate_1y"], errors="coerce"
        )
        df = df.merge(
            trainer_df,
            left_on=["race_date", "trainer_id"],
            right_on=["calc_date", "trainer_id"],
            how="left",
        )
        df = df.drop(columns=["calc_date"], errors="ignore")

    return df


def _join_mining_features(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """MING(DM/TM) 特徴量を結合"""
    logger.info("MING(DM/TM) を結合中...")

    dm_df = pd.DataFrame(
        db.fetch_all(
            """
            SELECT race_id, horse_no, data_kbn AS dm_data_kbn, dm_time_x10, dm_rank
            FROM core.mining_dm
            """
        )
    )
    tm_df = pd.DataFrame(
        db.fetch_all(
            """
            SELECT race_id, horse_no, data_kbn AS tm_data_kbn, tm_score, tm_rank
            FROM core.mining_tm
            """
        )
    )

    if not dm_df.empty:
        df = df.merge(dm_df, on=["race_id", "horse_no"], how="left")
    else:
        df["dm_data_kbn"] = None
        df["dm_time_x10"] = None
        df["dm_rank"] = None

    if not tm_df.empty:
        df = df.merge(tm_df, on=["race_id", "horse_no"], how="left")
    else:
        df["tm_data_kbn"] = None
        df["tm_score"] = None
        df["tm_rank"] = None

    for col in ["dm_data_kbn", "dm_time_x10", "dm_rank", "tm_data_kbn", "tm_score", "tm_rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dm_pred_time_sec"] = df["dm_time_x10"] / 10.0
    df["tm_score"] = df["tm_score"] / 10.0
    df["dm_missing_flag"] = df["dm_pred_time_sec"].isna().astype(int)
    df["tm_missing_flag"] = df["tm_score"].isna().astype(int)

    return df


def _join_training_features(
    db: Database, df: pd.DataFrame, from_date: date | None = None, to_date: date | None = None
) -> pd.DataFrame:
    """調教(SLOP/WOOD) 特徴量を結合"""
    logger.info("調教(SLOP/WOOD) 特徴量を結合中...")

    where_clauses = [
        "r.track_code BETWEEN 1 AND 10",
        "run.scratch_flag = FALSE",
        "res.finish_pos IS NOT NULL",
    ]
    params: list = []
    if from_date:
        where_clauses.append("r.race_date >= %s")
        params.append(from_date)
    if to_date:
        where_clauses.append("r.race_date <= %s")
        params.append(to_date)

    where_sql = " AND ".join(where_clauses)
    query = f"""
    WITH base AS (
        SELECT DISTINCT r.race_id, r.race_date, run.horse_id
        FROM core.race r
        JOIN core.runner run ON r.race_id = run.race_id
        JOIN core.result res ON run.race_id = res.race_id AND run.horse_id = res.horse_id
        WHERE {where_sql}
    )
    SELECT
        b.race_id,
        b.horse_id,
        sl.total_4f AS slop_last_total_4f_sec,
        sl.lap_4f AS slop_last_lap_4f_sec,
        sl.lap_1f AS slop_last_lap_1f_sec,
        CASE
            WHEN sl.training_date IS NULL THEN NULL
            ELSE (b.race_date - sl.training_date)
        END AS slop_days_since_last,
        COALESCE(sl28.slop_count_28d, 0) AS slop_count_28d,
        wd.total_4f AS wood_last_total_4f_sec,
        wd.lap_4f AS wood_last_lap_4f_sec,
        wd.lap_1f AS wood_last_lap_1f_sec,
        wd.course AS wood_course,
        wd.direction AS wood_direction,
        CASE
            WHEN wd.training_date IS NULL THEN NULL
            ELSE (b.race_date - wd.training_date)
        END AS wood_days_since_last,
        COALESCE(wd28.wood_count_28d, 0) AS wood_count_28d
    FROM base b
    LEFT JOIN LATERAL (
        SELECT ts.training_date, ts.training_time, ts.total_4f, ts.lap_4f, ts.lap_1f
        FROM core.training_slop ts
        WHERE ts.horse_id = b.horse_id
          AND ts.training_date <= b.race_date
          AND ts.training_date >= (b.race_date - INTERVAL '60 days')::date
        ORDER BY ts.training_date DESC, ts.training_time DESC NULLS LAST
        LIMIT 1
    ) sl ON TRUE
    LEFT JOIN LATERAL (
        SELECT COUNT(*)::INT AS slop_count_28d
        FROM core.training_slop ts
        WHERE ts.horse_id = b.horse_id
          AND ts.training_date <= b.race_date
          AND ts.training_date >= (b.race_date - INTERVAL '28 days')::date
    ) sl28 ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            tw.training_date,
            tw.training_time,
            tw.total_4f,
            tw.lap_4f,
            tw.lap_1f,
            tw.course,
            tw.direction
        FROM core.training_wood tw
        WHERE tw.horse_id = b.horse_id
          AND tw.training_date <= b.race_date
          AND tw.training_date >= (b.race_date - INTERVAL '60 days')::date
        ORDER BY tw.training_date DESC, tw.training_time DESC NULLS LAST
        LIMIT 1
    ) wd ON TRUE
    LEFT JOIN LATERAL (
        SELECT COUNT(*)::INT AS wood_count_28d
        FROM core.training_wood tw
        WHERE tw.horse_id = b.horse_id
          AND tw.training_date <= b.race_date
          AND tw.training_date >= (b.race_date - INTERVAL '28 days')::date
    ) wd28 ON TRUE
    """

    training_df = pd.DataFrame(db.fetch_all(query, tuple(params) if params else None))
    if training_df.empty:
        df["slop_missing_flag"] = 1
        df["wood_missing_flag"] = 1
        return df

    df = df.merge(training_df, on=["race_id", "horse_id"], how="left")
    numeric_cols = [
        "slop_last_total_4f_sec",
        "slop_last_lap_4f_sec",
        "slop_last_lap_1f_sec",
        "slop_days_since_last",
        "slop_count_28d",
        "wood_last_total_4f_sec",
        "wood_last_lap_4f_sec",
        "wood_last_lap_1f_sec",
        "wood_course",
        "wood_direction",
        "wood_days_since_last",
        "wood_count_28d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["slop_count_28d"] = df["slop_count_28d"].fillna(0).astype(int)
    df["wood_count_28d"] = df["wood_count_28d"].fillna(0).astype(int)
    df["slop_missing_flag"] = df["slop_last_total_4f_sec"].isna().astype(int)
    df["wood_missing_flag"] = df["wood_last_total_4f_sec"].isna().astype(int)
    return df


def _join_ck_features(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """CK特徴量を結合"""
    logger.info("CK 特徴量を結合中...")

    ck_columns_sql = ",\n        ".join(CK_FEATURE_COLUMNS)
    query = f"""
    WITH ck_with_race AS (
        SELECT
            (
                (
                    (kaisai_year::BIGINT * 10000)
                    + kaisai_md::BIGINT
                ) * 10000
                + (track_cd::BIGINT * 100)
                + race_no::BIGINT
            ) AS race_id,
            horse_id,
            dataspec,
            data_create_ymd,
            {ck_columns_sql}
        FROM mart.feat_ck_win
    ),
    ck_latest AS (
        SELECT DISTINCT ON (race_id, horse_id)
            race_id,
            horse_id,
            {ck_columns_sql}
        FROM ck_with_race
        ORDER BY
            race_id,
            horse_id,
            CASE WHEN dataspec = 'SNPN' THEN 0 ELSE 1 END,
            data_create_ymd DESC
    )
    SELECT * FROM ck_latest
    """
    ck_df = pd.DataFrame(db.fetch_all(query))

    if ck_df.empty:
        df["ck_missing_flag"] = 1
        return df

    for col in CK_FEATURE_COLUMNS:
        ck_df[col] = pd.to_numeric(ck_df[col], errors="coerce")
    ck_df = ck_df.rename(columns={col: f"ck_{col}" for col in CK_FEATURE_COLUMNS})

    df = df.merge(ck_df, on=["race_id", "horse_id"], how="left")
    df["ck_missing_flag"] = df["ck_h_total_starts"].isna().astype(int)

    df["ck_overall_win_rate_sm"] = (
        (df["ck_h_total_wins"] + 1) / (df["ck_h_total_starts"] + 10)
    ).where(df["ck_h_total_starts"].notna())
    df["ck_overall_top3_rate_sm"] = (
        (df["ck_h_total_top3"] + 1) / (df["ck_h_total_starts"] + 10)
    ).where(df["ck_h_total_starts"].notna())
    df["ck_overall_top5_rate_sm"] = (
        (df["ck_h_total_top5"] + 1) / (df["ck_h_total_starts"] + 10)
    ).where(df["ck_h_total_starts"].notna())
    df["ck_central_win_rate_sm"] = (
        (df["ck_h_central_wins"] + 1) / (df["ck_h_central_starts"] + 10)
    ).where(df["ck_h_central_starts"].notna())
    df["ck_central_top3_rate_sm"] = (
        (df["ck_h_central_top3"] + 1) / (df["ck_h_central_starts"] + 10)
    ).where(df["ck_h_central_starts"].notna())

    return df


def _add_inrace_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量を計算"""
    logger.info("レース内相対特徴量を計算中...")

    # 対象とする特徴量
    target_cols = ["speed_best_5", "closing_best_5", "early_mean_3"]

    for col in target_cols:
        if col not in df.columns:
            continue

        # レース内平均・標準偏差
        race_stats = df.groupby("race_id")[col].agg(["mean", "std"]).reset_index()
        race_stats.columns = ["race_id", f"{col}_race_mean", f"{col}_race_std"]

        df = df.merge(race_stats, on="race_id", how="left")

        # z-score
        std_col = f"{col}_race_std"
        df[f"{col}_z_inrace"] = (df[col] - df[f"{col}_race_mean"]) / df[std_col].replace(0, 1)

        # rank (1が最強)
        df[f"{col}_rank"] = df.groupby("race_id")[col].rank(ascending=False, method="min")

        # 一時カラム削除
        df = df.drop(columns=[f"{col}_race_mean", f"{col}_race_std"], errors="ignore")

    return df


def _add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """ペース圧特徴量を計算"""
    logger.info("ペース圧特徴量を計算中...")

    if "early_mean_3_rank" not in df.columns:
        return df

    # pace_front_runner_cnt: レース内で先行候補 (early_mean_3_rank <= 3) の数
    front_count = (
        df[df["early_mean_3_rank"] <= 3]
        .groupby("race_id")
        .size()
        .reset_index(name="pace_front_runner_cnt")
    )
    df = df.merge(front_count, on="race_id", how="left")
    df["pace_front_runner_cnt"] = df["pace_front_runner_cnt"].fillna(0).astype(int)

    # num_faster_early: 自分より early_mean_3 が大きい馬の数
    # rank が小さいほど先行力が高いので、(rank - 1) がそれより前にいる馬の数
    df["num_faster_early"] = (df["early_mean_3_rank"] - 1).fillna(0).astype(int)

    return df


def _add_condition_features(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """当日コンディション特徴量を計算"""
    logger.info("当日コンディション特徴量を計算中...")

    # 前走情報を取得
    prev_race_query = """
    SELECT
        run.horse_id,
        r.race_date,
        LAG(r.distance_m) OVER (PARTITION BY run.horse_id ORDER BY r.race_date, run.race_id) as prev_distance_m,
        LAG(r.race_date) OVER (PARTITION BY run.horse_id ORDER BY r.race_date, run.race_id) as prev_race_date
    FROM core.runner run
    JOIN core.race r ON run.race_id = r.race_id
    WHERE r.track_code BETWEEN 1 AND 10
      AND run.scratch_flag = FALSE
    """
    prev_df = pd.DataFrame(db.fetch_all(prev_race_query))

    if prev_df.empty:
        df["days_since_last"] = None
        df["distance_change_m"] = None
        return df

    # 前走日を取得
    prev_df = prev_df.dropna(subset=["prev_race_date"])

    # date型をdatetimeに変換して日数計算
    prev_df["race_date_dt"] = pd.to_datetime(prev_df["race_date"])
    prev_df["prev_race_date_dt"] = pd.to_datetime(prev_df["prev_race_date"])
    prev_df["days_since_last"] = (prev_df["race_date_dt"] - prev_df["prev_race_date_dt"]).dt.days

    # 結合 (race_dateはそのまま使用)
    df = df.merge(
        prev_df[["horse_id", "race_date", "days_since_last", "prev_distance_m"]],
        on=["horse_id", "race_date"],
        how="left",
    )

    # 距離変化
    df["distance_change_m"] = df["distance_m"] - df["prev_distance_m"].fillna(df["distance_m"])
    df = df.drop(columns=["prev_distance_m"], errors="ignore")

    return df


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="学習データセット生成")
    parser.add_argument(
        "--from-date",
        type=str,
        default=DEFAULT_FROM_DATE.isoformat(),
        help=f"開始日 (YYYY-MM-DD, default: {DEFAULT_FROM_DATE.isoformat()})",
    )
    parser.add_argument("--to-date", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/train.parquet", help="出力ファイル")
    parser.add_argument("--min-horses", type=int, default=5, help="最低出走頭数")
    args = parser.parse_args()

    from_date = None
    to_date = None
    if args.from_date:
        from datetime import datetime

        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    if args.to_date:
        from datetime import datetime

        to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    # 出力ディレクトリ作成
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Database() as db:
        df = build_dataset(db, from_date=from_date, to_date=to_date)

    if df.empty:
        logger.error("データセットが空です")
        return

    df = _filter_small_field_races(df, args.min_horses)
    if df.empty:
        logger.error("最低頭数フィルタ後にデータセットが空になりました")
        return

    if "class_code" in df.columns:
        class_code = pd.to_numeric(df["class_code"], errors="coerce")
        df["class_code"] = class_code.fillna(0).astype(int)
    if "going" in df.columns:
        going = pd.to_numeric(df["going"], errors="coerce")
        going = going.where(going != 0)
        df["going"] = going
    if "field_size" in df.columns:
        field_size = pd.to_numeric(df["field_size"], errors="coerce")
        field_size = field_size.where(field_size != 0)
        df["field_size"] = field_size

    # 特徴量カラムを選択
    feature_cols = [
        # メタ情報
        "race_id",
        "race_date",
        "horse_id",
        "horse_no",
        # レース条件
        "track_code",
        "surface",
        "distance_m",
        "going",
        "class_code",
        "field_size",
        # 当日コンディション
        "gate",
        "carried_weight",
        "body_weight",
        "body_weight_diff",
        "days_since_last",
        "distance_change_m",
        # 馬の能力 (全条件版)
        "speed_last",
        "speed_mean_3",
        "speed_best_5",
        "speed_std_5",
        "speed_trend_3",
        "closing_last",
        "closing_mean_3",
        "closing_best_5",
        "early_mean_3",
        "early_best_5",
        "position_gain_mean_3",
        "finish_mean_3",
        "finish_best_5",
        "n_runs_5",
        # 馬の適性 (条件近似版)
        "speed_sim_mean_3",
        "speed_sim_best_5",
        "closing_sim_mean_3",
        "closing_sim_best_5",
        "early_sim_mean_3",
        "n_sim_runs_5",
        # レース内相対
        "speed_best_5_z_inrace",
        "closing_best_5_z_inrace",
        "early_mean_3_z_inrace",
        "speed_best_5_rank",
        "closing_best_5_rank",
        "early_mean_3_rank",
        # ペース圧
        "pace_front_runner_cnt",
        "num_faster_early",
        # 人の実績
        "jockey_win_rate_1y",
        "jockey_place_rate_1y",
        "trainer_win_rate_1y",
        "trainer_place_rate_1y",
        # MING (DM/TM)
        "dm_data_kbn",
        "dm_pred_time_sec",
        "dm_rank",
        "dm_missing_flag",
        "tm_data_kbn",
        "tm_score",
        "tm_rank",
        "tm_missing_flag",
        # 調教
        "slop_last_total_4f_sec",
        "slop_last_lap_4f_sec",
        "slop_last_lap_1f_sec",
        "slop_days_since_last",
        "slop_count_28d",
        "slop_missing_flag",
        "wood_last_total_4f_sec",
        "wood_last_lap_4f_sec",
        "wood_last_lap_1f_sec",
        "wood_course",
        "wood_direction",
        "wood_days_since_last",
        "wood_count_28d",
        "wood_missing_flag",
        # CK
        *[f"ck_{col}" for col in CK_FEATURE_COLUMNS],
        "ck_missing_flag",
        "ck_overall_win_rate_sm",
        "ck_overall_top3_rate_sm",
        "ck_overall_top5_rate_sm",
        "ck_central_win_rate_sm",
        "ck_central_top3_rate_sm",
        # 目的変数
        "is_win",
        "finish_pos",
    ]

    # 存在するカラムのみ選択
    available_cols = [c for c in feature_cols if c in df.columns]
    df = df[available_cols]

    # 保存
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"データセット保存完了: {output_path} ({len(df)} 件)")

    # 統計情報
    logger.info(f"レース数: {df['race_id'].nunique()}")
    logger.info(f"勝率: {df['is_win'].mean():.4f}")
    logger.info(f"欠損率: n_runs_5=0 の割合 = {(df['n_runs_5'] == 0).mean():.4f}")


if __name__ == "__main__":
    main()
