"""
特徴量生成バッチスクリプト

core テーブルから mart テーブルに特徴量を生成する。
以下のステップを順次実行:
1. グループ統計更新 (mart.time_stats)
2. run_index計算 (mart.run_index)
3. horse_stats計算 (mart.horse_stats)
4. person_stats計算 (mart.person_stats)

使用方法:
    uv run python scripts/build_features.py

オプション:
    --rebuild    全データを再構築 (デフォルト: 増分更新)
    --date       特定日のみ処理 (例: --date 2026-01-01)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 定数
# =============================================================================

# 中央競馬の競馬場コード (01:札幌 - 10:小倉)
CENTRAL_TRACK_CODES = tuple(range(1, 11))

# 距離バケット (m)
DISTANCE_BUCKETS = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 3000, 3200, 3600]

# グループ統計の最小サンプル数
MIN_SAMPLE_COUNT = 200


# =============================================================================
# ユーティリティ関数
# =============================================================================


def distance_to_bucket(distance_m: int) -> int:
    """距離をバケットに変換"""
    for bucket in DISTANCE_BUCKETS:
        if distance_m <= bucket:
            return bucket
    return DISTANCE_BUCKETS[-1]


def going_to_bucket(going: int | None) -> int:
    """馬場状態をバケットに変換 (1:良系, 2:道悪系)"""
    if going is None or going <= 2:
        return 1  # 良系 (良, 稍重)
    return 2  # 道悪系 (重, 不良)


def is_similar_condition(
    target_surface: int,
    target_distance_bucket: int,
    target_going_bucket: int,
    past_surface: int,
    past_distance_m: int,
    past_going: int | None,
) -> bool:
    """条件近似かどうかを判定"""
    # 1) surface: 完全一致必須
    if past_surface != target_surface:
        return False

    # 2) distance: ±200m 以内
    if abs(past_distance_m - target_distance_bucket) > 200:
        return False

    # 3) going: 良系/道悪系 で一致
    if going_to_bucket(past_going) != target_going_bucket:
        return False

    return True


# =============================================================================
# Helper Functions for Feature Engineering
# =============================================================================


def calculate_benchmark_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    span: float = 730,  # 2 years approx
    min_periods: int = 1,
) -> pd.DataFrame:
    """西暦順にソートされたDFに対して、EWM統計を計算する (Shift(1)でリーク防止)"""
    # 1. Calculate EWM Mean/Std
    ewm_stats = (
        df.groupby(group_cols)["time_sec"]
        .ewm(span=span, min_periods=min_periods)
        .agg(["mean", "std"])
    )

    # 2. Calculate Count (Expanding)
    count_stats = df.groupby(group_cols)["time_sec"].expanding().count().rename("count")

    # Merge
    stats = pd.concat([ewm_stats, count_stats], axis=1)

    # Reset index to sort by original index
    stats = stats.reset_index(level=list(range(len(group_cols))), drop=True)
    stats = stats.sort_index()

    # Add group columns back
    for col in group_cols:
        stats[col] = df[col]

    # Shift(1) to avoid leakage
    stats_shifted = stats.groupby(group_cols)[["mean", "std", "count"]].shift(1)

    return stats_shifted


def apply_shrinkage(
    fine_stats: pd.DataFrame,
    coarse_stats: pd.DataFrame,
    shrinkage_k: float = 20.0,
) -> pd.DataFrame:
    """階層型縮約 (Bayesian Shrinkage)"""
    n = fine_stats["count"].fillna(0)
    mu_fine = fine_stats["mean"]
    mu_coarse = coarse_stats["mean"]

    # 欠損は Coarse で埋める
    mu_fine = mu_fine.fillna(mu_coarse)

    # Weight
    w = n / (n + shrinkage_k)

    # Shrinkage
    mu_est = w * mu_fine + (1 - w) * mu_coarse
    sigma_est = w * fine_stats["std"].fillna(0) + (1 - w) * coarse_stats["std"].fillna(0)

    return pd.DataFrame(
        {
            "mean": mu_est,
            "std": sigma_est,
            "count": n,
            "weight_fine": w,
        }
    )


# =============================================================================
# Step 1: 廃止 (互換性のため関数名のみ残すが何もしない)
# =============================================================================


def build_time_stats(db: Database) -> int:
    """以前の統計ロジック。現在は使われていません (Step 2 で動的に計算されます)"""
    logger.info("Step 1: build_time_stats は廃止されました。Step 2 に統合されています。")
    return 0


# =============================================================================
# Step 2: run_index 計算
# =============================================================================


# =============================================================================
# Step 2: run_index 計算 (Feature Leakage修正版: EWM + Shrinkage)
# =============================================================================


def build_run_index(
    db: Database,
    target_date: date | None = None,
    rebuild: bool = False,
    from_date: date | None = None,
    to_date: date | None = None,
) -> int:
    """1走ごとの基礎指標を計算して mart.run_index に保存 (時系列シーケンシャル更新)"""
    logger.info("Step 2: run_index を計算中 (EWM + Shrinkage)...")

    # 1. 全レースデータを取得 (時系列計算のため全件ロードが必要)
    # ※ target_date が指定されていても、過去からの推移が必要なため全件計算する
    #   (ただし保存処理は対象日以降のみにするなどで最適化可能だが、今回はシンプルに全件UPSERTする)

    where_clauses = [
        "r.track_code BETWEEN 1 AND 10",
        "r.surface IN (1, 2, 3)",
        "r.distance_m > 0",
        "res.finish_pos IS NOT NULL",
        "res.time_sec IS NOT NULL",
    ]
    params: list[date] = []
    if from_date:
        where_clauses.append("r.race_date >= %s")
        params.append(from_date)
    if to_date:
        where_clauses.append("r.race_date <= %s")
        params.append(to_date)
    where_sql = " AND ".join(where_clauses)

    query = f"""
    SELECT
        r.race_id,
        r.race_date,
        r.track_code,
        r.surface,
        r.distance_m,
        r.going,
        res.horse_id,
        res.finish_pos,
        res.time_sec,
        res.final3f_sec,
        res.corner4_pos
    FROM core.race r
    JOIN core.result res ON r.race_id = res.race_id
    WHERE {where_sql}
    ORDER BY r.race_date, r.race_id
    """
    rows = db.fetch_all(query, tuple(params) if params else None)

    if not rows:
        logger.warning("Step 2: レースデータがありません")
        return 0

    df = pd.DataFrame(rows)
    # Decimal型が含まれる可能性があるためfloatに変換
    for col in ["time_sec", "final3f_sec"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df["distance_bucket"] = df["distance_m"].apply(distance_to_bucket)
    df["going_bucket"] = df["going"].apply(going_to_bucket)

    # 日付型変換
    # df["race_date"] = pd.to_datetime(df["race_date"]) # DBからdate型で返るなら不要だが念のため
    # 既にsortされているが、indexをリセットしておく
    df = df.reset_index(drop=True)

    logger.info(f"データロード完了: {len(df)} 件. 基準タイム計算開始...")

    # 2. 基準タイムの計算 (Speed Index)
    # Coarse: track, surface, distance
    coarse_groups = ["track_code", "surface", "distance_bucket"]
    # Fine: track, surface, distance, going
    fine_groups = ["track_code", "surface", "distance_bucket", "going_bucket"]

    # Calculate Time Stats
    time_coarse = calculate_benchmark_stats(df, coarse_groups, span=365 * 2)
    time_fine = calculate_benchmark_stats(df, fine_groups, span=365 * 2)

    # Shrinkage
    # time_coarse/fine は df と同じ Index (sort済み) を持つ
    time_bm = apply_shrinkage(time_fine, time_coarse, shrinkage_k=20.0)

    # 3. 基準上がり3Fの計算 (Closing Index)
    # 欠損がある場合は計算対象から外すが、EWMはNaNを無視して計算してくれる
    # ただし count の計算で少しズレる可能性はあるが許容する

    # calculate_benchmark_stats は現状 "time_sec" 固定になっているので、
    # 一時的にカラム名を変更するか、関数を汎用化する必要がある。
    # ここでは関数呼び出し前に rename して対応する (副作用なしコピーで)

    df_3f = df.copy()
    df_3f["time_sec"] = df_3f["final3f_sec"]  # Hack: final3fをtime_secとして渡す

    f3f_coarse = calculate_benchmark_stats(df_3f, coarse_groups, span=365 * 2)
    f3f_fine = calculate_benchmark_stats(df_3f, fine_groups, span=365 * 2)

    f3f_bm = apply_shrinkage(f3f_fine, f3f_coarse, shrinkage_k=20.0)

    # 4. 指標計算
    # speed_index = (benchmark_mean - time_sec) / benchmark_std
    # std が 0 または NaN の場合は 1.0 にする (偏差値計算できないため)

    # 基準タイム (平均)
    df["bm_time_mean"] = time_bm["mean"]
    df["bm_time_std"] = time_bm["std"].fillna(0).replace(0, 1.0)  # Avoid div by zero

    df["speed_index"] = (df["bm_time_mean"] - df["time_sec"]) / df["bm_time_std"]

    # 上がり指標
    df["bm_3f_mean"] = f3f_bm["mean"]
    df["bm_3f_std"] = f3f_bm["std"].fillna(0).replace(0, 1.0)

    df["closing_index"] = (df["bm_3f_mean"] - df["final3f_sec"]) / df["bm_3f_std"]
    df["closing_missing"] = df["final3f_sec"].isna()

    # early_index, position_gain
    df["early_index"] = -df["corner4_pos"].fillna(10)
    df["position_gain"] = df["corner4_pos"] - df["finish_pos"]

    # 5. 保存対象のフィルタリング
    # rebuild=True なら全件保存
    # rebuild=False なら、本来は「新規レースのみ」だが、
    # 過去の集計値が変わる可能性があるため、本来は全件更新すべき。
    # しかしパフォーマンスを考慮し、target_date があればその日以降、なければ全件とする。

    save_df = df
    if target_date:
        save_df = save_df[save_df["race_date"] >= target_date]
    if from_date:
        save_df = save_df[save_df["race_date"] >= from_date]
    if to_date:
        save_df = save_df[save_df["race_date"] <= to_date]

    logger.info(f"保存対象: {len(save_df)} 件")

    if save_df.empty:
        return 0

    # リビルド時はテーブルクリア
    if rebuild and not target_date:
        logger.info("古いデータを削除中...")
        db.execute("DELETE FROM mart.run_index")
        db.connect().commit()

    # DBに保存
    insert_query = """
    INSERT INTO mart.run_index
        (race_id, horse_id, speed_index, closing_index, early_index,
         position_gain, closing_missing)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (race_id, horse_id) DO UPDATE SET
        speed_index = EXCLUDED.speed_index,
        closing_index = EXCLUDED.closing_index,
        early_index = EXCLUDED.early_index,
        position_gain = EXCLUDED.position_gain,
        closing_missing = EXCLUDED.closing_missing
    """

    # execute_batch を使うと速いが、ここでは既存のループで実装する（件数が多い場合は要検討）
    # 数万件なら execute_batch (psycopg2.extras) が望ましいが、infrastructure.database の実装次第。
    # ここでは既存コードにならってループするが、件数が多いので batch commit する

    count = 0
    batch_size = 1000

    for _, row in save_df.iterrows():
        db.execute(
            insert_query,
            (
                int(row["race_id"]),
                row["horse_id"],
                float(row["speed_index"]) if pd.notna(row["speed_index"]) else None,
                float(row["closing_index"]) if pd.notna(row["closing_index"]) else None,
                float(row["early_index"]) if pd.notna(row["early_index"]) else None,
                int(row["position_gain"]) if pd.notna(row["position_gain"]) else None,
                bool(row["closing_missing"]),
            ),
        )
        count += 1
        if count % batch_size == 0:
            db.connect().commit()

    db.connect().commit()
    logger.info(f"Step 2: 完了 - {count} 件")
    return count


# =============================================================================
# Step 3: horse_stats 計算
# =============================================================================


def calc_trend(values: list) -> float | None:
    """直近N走の傾き (線形回帰)"""
    if len(values) < 2:
        return None

    x = np.arange(len(values))
    # decimal.Decimal 対応: floatに変換
    y = np.array([float(v) for v in values], dtype=np.float64)
    # 古い方から新しい方に並んでいる想定
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def build_horse_stats(
    db: Database,
    target_date: date | None = None,
    rebuild: bool = False,
    from_date: date | None = None,
    to_date: date | None = None,
) -> int:
    """馬単位の近走集計を計算して mart.horse_stats に保存"""
    logger.info("Step 3: horse_stats を計算中...")

    # 対象日リストを取得
    if target_date:
        target_dates = [target_date]
    else:
        # 全レース日を取得
        where_clauses = [
            "track_code BETWEEN 1 AND 10",
            "surface IN (1, 2, 3)",
            "distance_m > 0",
        ]
        params: list[date] = []
        if from_date:
            where_clauses.append("race_date >= %s")
            params.append(from_date)
        if to_date:
            where_clauses.append("race_date <= %s")
            params.append(to_date)
        where_sql = " AND ".join(where_clauses)

        dates_query = f"""
        SELECT DISTINCT race_date
        FROM core.race
        WHERE {where_sql}
        ORDER BY race_date
        """
        target_dates = [row["race_date"] for row in db.fetch_all(dates_query, tuple(params))]

    if not target_dates:
        logger.warning("対象日がありません")
        return 0

    logger.info(f"対象日数: {len(target_dates)} 日")

    # リビルド時は既存データ削除
    if rebuild:
        db.execute("DELETE FROM mart.horse_stats")
        db.connect().commit()

    count = 0
    for i, calc_date in enumerate(target_dates):
        count += _build_horse_stats_for_date(db, calc_date)
        # 進捗ログ (100日ごと)
        if (i + 1) % 100 == 0:
            logger.info(f"  進捗: {i + 1}/{len(target_dates)} 日完了 ({count:,} 件)")

    logger.info(f"Step 3: 完了 - {count} 件")
    return count


def _build_horse_stats_for_date(db: Database, calc_date: date) -> int:
    """特定日の horse_stats を計算"""
    # その日に出走する馬とレース条件を取得
    runners_query = """
    SELECT DISTINCT
        run.horse_id,
        r.surface,
        r.distance_m,
        r.going
    FROM core.runner run
    JOIN core.race r ON run.race_id = r.race_id
    WHERE r.race_date = %s
      AND r.track_code BETWEEN 1 AND 10
      AND r.surface IN (1, 2, 3)
      AND r.distance_m > 0
      AND run.scratch_flag = FALSE
    """
    runners = db.fetch_all(runners_query, (calc_date,))

    if not runners:
        return 0

    # 対象馬IDリスト
    horse_ids = [r["horse_id"] for r in runners]

    # 過去走データを取得 (対象馬のみ、直近5走に限定)
    placeholders = ",".join(["%s"] * len(horse_ids))
    history_query = f"""
    SELECT
        run.horse_id,
        r.race_date,
        r.surface,
        r.distance_m,
        r.going,
        res.finish_pos,
        ri.speed_index,
        ri.closing_index,
        ri.early_index,
        ri.position_gain,
        ROW_NUMBER() OVER (PARTITION BY run.horse_id ORDER BY r.race_date DESC) as rn
    FROM core.runner run
    JOIN core.race r ON run.race_id = r.race_id
    JOIN core.result res ON run.race_id = res.race_id AND run.horse_id = res.horse_id
    LEFT JOIN mart.run_index ri ON run.race_id = ri.race_id AND run.horse_id = ri.horse_id
    WHERE r.race_date < %s
      AND r.track_code BETWEEN 1 AND 10
      AND r.surface IN (1, 2, 3)
      AND r.distance_m > 0
      AND res.finish_pos IS NOT NULL
      AND run.horse_id IN ({placeholders})
    """
    history = db.fetch_all(history_query, (calc_date, *horse_ids))

    # 直近5走のみにフィルタ
    history = [h for h in history if h["rn"] <= 5]

    # horse_id でグループ化 (dict形式で高速化)
    horse_history: dict[str, list[dict]] = {}
    for h in history:
        hid = h["horse_id"]
        if hid not in horse_history:
            horse_history[hid] = []
        horse_history[hid].append(h)

    count = 0
    for runner in runners:
        horse_id = runner["horse_id"]
        target_surface = runner["surface"]
        target_distance_bucket = distance_to_bucket(runner["distance_m"])
        target_going_bucket = going_to_bucket(runner["going"])

        past_runs = horse_history.get(horse_id, [])

        # 条件近似走を抽出
        sim_runs = [
            r
            for r in past_runs
            if is_similar_condition(
                target_surface,
                target_distance_bucket,
                target_going_bucket,
                r["surface"],
                r["distance_m"],
                r["going"],
            )
        ][:5]

        _insert_horse_stats(
            db,
            calc_date,
            horse_id,
            target_surface,
            target_distance_bucket,
            target_going_bucket,
            past_runs,
            sim_runs,
        )
        count += 1

    db.connect().commit()
    return count


def _insert_horse_stats(
    db: Database,
    calc_date: date,
    horse_id: str,
    target_surface: int,
    target_distance_bucket: int,
    target_going_bucket: int,
    past_runs: list[dict],
    sim_runs: list[dict],
) -> None:
    """horse_stats を1件挿入"""
    # 全条件版の集計 (decimal.Decimal 対応: floatに変換)
    n_runs = len(past_runs)
    speed_list = [
        float(r["speed_index"])
        for r in past_runs
        if r["speed_index"] is not None and pd.notna(r["speed_index"])
    ]
    closing_list = [
        float(r["closing_index"])
        for r in past_runs
        if r["closing_index"] is not None and pd.notna(r["closing_index"])
    ]
    early_list = [
        float(r["early_index"])
        for r in past_runs
        if r["early_index"] is not None and pd.notna(r["early_index"])
    ]
    pg_list = [
        int(r["position_gain"])
        for r in past_runs
        if r["position_gain"] is not None and pd.notna(r["position_gain"])
    ]
    finish_list = [
        int(r["finish_pos"])
        for r in past_runs
        if r["finish_pos"] is not None and pd.notna(r["finish_pos"])
    ]

    # 条件近似版の集計
    n_sim = len(sim_runs)
    speed_sim_list = [
        float(r["speed_index"])
        for r in sim_runs
        if r["speed_index"] is not None and pd.notna(r["speed_index"])
    ]
    closing_sim_list = [
        float(r["closing_index"])
        for r in sim_runs
        if r["closing_index"] is not None and pd.notna(r["closing_index"])
    ]
    early_sim_list = [
        float(r["early_index"])
        for r in sim_runs
        if r["early_index"] is not None and pd.notna(r["early_index"])
    ]

    def mean_n(lst: list, n: int) -> float | None:
        vals = lst[:n]
        return sum(vals) / len(vals) if vals else None

    def max_n(lst: list, n: int) -> float | None:
        vals = lst[:n]
        return max(vals) if vals else None

    def std_n(lst: list, n: int) -> float | None:
        vals = lst[:n]
        if len(vals) < 2:
            return None

        return float(np.std(vals, ddof=1))

    def min_n(lst: list, n: int) -> float | None:
        vals = lst[:n]
        return min(vals) if vals else None

    query = """
    INSERT INTO mart.horse_stats (
        calc_date, horse_id,
        speed_last, speed_mean_3, speed_best_5, speed_std_5, speed_trend_3,
        closing_last, closing_mean_3, closing_best_5,
        early_mean_3, early_best_5, position_gain_mean_3,
        finish_mean_3, finish_best_5, n_runs_5,
        target_surface, target_distance_bucket, target_going_bucket,
        speed_sim_mean_3, speed_sim_best_5,
        closing_sim_mean_3, closing_sim_best_5,
        early_sim_mean_3, n_sim_runs_5
    ) VALUES (
        %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s
    )
    ON CONFLICT (calc_date, horse_id, target_surface, target_distance_bucket, target_going_bucket)
    DO UPDATE SET
        speed_last = EXCLUDED.speed_last,
        speed_mean_3 = EXCLUDED.speed_mean_3,
        speed_best_5 = EXCLUDED.speed_best_5,
        speed_std_5 = EXCLUDED.speed_std_5,
        speed_trend_3 = EXCLUDED.speed_trend_3,
        closing_last = EXCLUDED.closing_last,
        closing_mean_3 = EXCLUDED.closing_mean_3,
        closing_best_5 = EXCLUDED.closing_best_5,
        early_mean_3 = EXCLUDED.early_mean_3,
        early_best_5 = EXCLUDED.early_best_5,
        position_gain_mean_3 = EXCLUDED.position_gain_mean_3,
        finish_mean_3 = EXCLUDED.finish_mean_3,
        finish_best_5 = EXCLUDED.finish_best_5,
        n_runs_5 = EXCLUDED.n_runs_5,
        speed_sim_mean_3 = EXCLUDED.speed_sim_mean_3,
        speed_sim_best_5 = EXCLUDED.speed_sim_best_5,
        closing_sim_mean_3 = EXCLUDED.closing_sim_mean_3,
        closing_sim_best_5 = EXCLUDED.closing_sim_best_5,
        early_sim_mean_3 = EXCLUDED.early_sim_mean_3,
        n_sim_runs_5 = EXCLUDED.n_sim_runs_5
    """

    db.execute(
        query,
        (
            calc_date,
            horse_id,
            speed_list[0] if speed_list else None,
            mean_n(speed_list, 3),
            max_n(speed_list, 5),
            std_n(speed_list, 5),
            calc_trend(speed_list[:3][::-1]) if len(speed_list) >= 2 else None,  # 古い順に
            closing_list[0] if closing_list else None,
            mean_n(closing_list, 3),
            max_n(closing_list, 5),
            mean_n(early_list, 3),
            max_n(early_list, 5),
            mean_n(pg_list, 3),
            mean_n(finish_list, 3),
            int(min_n(finish_list, 5)) if finish_list else None,
            n_runs,
            target_surface,
            target_distance_bucket,
            target_going_bucket,
            mean_n(speed_sim_list, 3) if len(speed_sim_list) >= 3 else None,
            max_n(speed_sim_list, 5) if speed_sim_list else None,
            mean_n(closing_sim_list, 3) if len(closing_sim_list) >= 3 else None,
            max_n(closing_sim_list, 5) if closing_sim_list else None,
            mean_n(early_sim_list, 3) if len(early_sim_list) >= 3 else None,
            n_sim,
        ),
    )


# =============================================================================
# Step 4: person_stats 計算
# =============================================================================


def build_person_stats(
    db: Database,
    target_date: date | None = None,
    rebuild: bool = False,
    from_date: date | None = None,
    to_date: date | None = None,
) -> int:
    """騎手・調教師の過去実績を計算して mart.person_stats に保存"""
    logger.info("Step 4: person_stats を計算中...")

    # 対象日リストを取得
    if target_date:
        target_dates = [target_date]
    else:
        where_clauses = [
            "track_code BETWEEN 1 AND 10",
            "surface IN (1, 2, 3)",
            "distance_m > 0",
        ]
        params: list[date] = []
        if from_date:
            where_clauses.append("race_date >= %s")
            params.append(from_date)
        if to_date:
            where_clauses.append("race_date <= %s")
            params.append(to_date)
        where_sql = " AND ".join(where_clauses)

        dates_query = f"""
        SELECT DISTINCT race_date
        FROM core.race
        WHERE {where_sql}
        ORDER BY race_date
        """
        target_dates = [row["race_date"] for row in db.fetch_all(dates_query, tuple(params))]

    if not target_dates:
        logger.warning("対象日がありません")
        return 0

    logger.info(f"対象日数: {len(target_dates)} 日")

    if rebuild:
        db.execute("DELETE FROM mart.person_stats")
        db.connect().commit()

    # 一括で過去1年実績を取得（日付範囲を限定）
    min_date = min(target_dates) - timedelta(days=365)
    max_date = max(target_dates)

    logger.info(f"実績データ取得中: {min_date} ~ {max_date}")

    # 騎手の日別成績を一括取得
    jockey_daily_query = """
    SELECT
        r.race_date,
        run.jockey_id as person_id,
        COUNT(*) as total,
        SUM(CASE WHEN res.finish_pos = 1 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN res.finish_pos <= 3 THEN 1 ELSE 0 END) as places
    FROM core.runner run
    JOIN core.race r ON run.race_id = r.race_id
    JOIN core.result res ON run.race_id = res.race_id AND run.horse_id = res.horse_id
    WHERE r.race_date >= %s AND r.race_date < %s
      AND r.track_code BETWEEN 1 AND 10
      AND r.surface IN (1, 2, 3)
      AND r.distance_m > 0
      AND run.jockey_id IS NOT NULL
      AND res.finish_pos IS NOT NULL
    GROUP BY r.race_date, run.jockey_id
    """
    jockey_daily = db.fetch_all(jockey_daily_query, (min_date, max_date))
    logger.info(f"騎手日別成績: {len(jockey_daily)} 件")

    # 調教師の日別成績を一括取得
    trainer_daily_query = """
    SELECT
        r.race_date,
        run.trainer_id as person_id,
        COUNT(*) as total,
        SUM(CASE WHEN res.finish_pos = 1 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN res.finish_pos <= 3 THEN 1 ELSE 0 END) as places
    FROM core.runner run
    JOIN core.race r ON run.race_id = r.race_id
    JOIN core.result res ON run.race_id = res.race_id AND run.horse_id = res.horse_id
    WHERE r.race_date >= %s AND r.race_date < %s
      AND r.track_code BETWEEN 1 AND 10
      AND r.surface IN (1, 2, 3)
      AND r.distance_m > 0
      AND run.trainer_id IS NOT NULL
      AND res.finish_pos IS NOT NULL
    GROUP BY r.race_date, run.trainer_id
    """
    trainer_daily = db.fetch_all(trainer_daily_query, (min_date, max_date))
    logger.info(f"調教師日別成績: {len(trainer_daily)} 件")

    # DataFrameで効率的に処理
    jockey_df = pd.DataFrame(jockey_daily) if jockey_daily else pd.DataFrame()
    trainer_df = pd.DataFrame(trainer_daily) if trainer_daily else pd.DataFrame()

    insert_query = """
    INSERT INTO mart.person_stats
        (calc_date, person_type, person_id, win_rate_1y, place_rate_1y, sample_count_1y)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (calc_date, person_type, person_id)
    DO UPDATE SET
        win_rate_1y = EXCLUDED.win_rate_1y,
        place_rate_1y = EXCLUDED.place_rate_1y,
        sample_count_1y = EXCLUDED.sample_count_1y
    """

    count = 0
    for i, calc_date in enumerate(target_dates):
        from_date = calc_date - timedelta(days=365)

        # 騎手の1年実績を集計
        if not jockey_df.empty:
            mask = (jockey_df["race_date"] >= from_date) & (jockey_df["race_date"] < calc_date)
            jockey_agg = (
                jockey_df[mask]
                .groupby("person_id")
                .agg({"total": "sum", "wins": "sum", "places": "sum"})
                .reset_index()
            )
            for _, row in jockey_agg.iterrows():
                win_rate = row["wins"] / row["total"] if row["total"] > 0 else 0
                place_rate = row["places"] / row["total"] if row["total"] > 0 else 0
                db.execute(
                    insert_query,
                    (
                        calc_date,
                        "jockey",
                        row["person_id"],
                        float(win_rate),
                        float(place_rate),
                        int(row["total"]),
                    ),
                )
                count += 1

        # 調教師の1年実績を集計
        if not trainer_df.empty:
            mask = (trainer_df["race_date"] >= from_date) & (trainer_df["race_date"] < calc_date)
            trainer_agg = (
                trainer_df[mask]
                .groupby("person_id")
                .agg({"total": "sum", "wins": "sum", "places": "sum"})
                .reset_index()
            )
            for _, row in trainer_agg.iterrows():
                win_rate = row["wins"] / row["total"] if row["total"] > 0 else 0
                place_rate = row["places"] / row["total"] if row["total"] > 0 else 0
                db.execute(
                    insert_query,
                    (
                        calc_date,
                        "trainer",
                        row["person_id"],
                        float(win_rate),
                        float(place_rate),
                        int(row["total"]),
                    ),
                )
                count += 1

        db.connect().commit()

        # 進捗ログ (100日ごと)
        if (i + 1) % 100 == 0:
            logger.info(f"  進捗: {i + 1}/{len(target_dates)} 日完了 ({count:,} 件)")

    logger.info(f"Step 4: 完了 - {count} 件")
    return count


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="特徴量生成バッチ")
    parser.add_argument("--rebuild", action="store_true", help="全データを再構築")
    parser.add_argument("--date", type=str, help="特定日のみ処理 (YYYY-MM-DD)")
    parser.add_argument(
        "--from-date",
        type=str,
        default="2016-01-01",
        help="開始日 (YYYY-MM-DD, default: 2016-01-01)",
    )
    parser.add_argument("--to-date", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--step", type=int, help="特定ステップのみ実行 (1-4)")
    args = parser.parse_args()

    target_date = None
    from_date = None
    to_date = None
    from datetime import datetime

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    if args.from_date:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    if args.to_date:
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    with Database() as db:
        if args.step is None or args.step == 1:
            build_time_stats(db)

        if args.step is None or args.step == 2:
            build_run_index(
                db,
                target_date=target_date,
                rebuild=args.rebuild,
                from_date=from_date,
                to_date=to_date,
            )

        if args.step is None or args.step == 3:
            build_horse_stats(
                db,
                target_date=target_date,
                rebuild=args.rebuild,
                from_date=from_date,
                to_date=to_date,
            )

        if args.step is None or args.step == 4:
            build_person_stats(
                db,
                target_date=target_date,
                rebuild=args.rebuild,
                from_date=from_date,
                to_date=to_date,
            )

    logger.info("全ステップ完了")


if __name__ == "__main__":
    main()
