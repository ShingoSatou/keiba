"""
CLI推論スクリプト

レースIDと10分前オッズを入力して、買い/見送りの推奨を出力する。

使用方法:
    uv run python scripts/predict.py --race_id 202602030505 --odds "1:5.2,2:12.0,3:6.5"

オプション:
    --race_id    レースID (必須)
    --odds       10分前オッズ (馬番:オッズ のカンマ区切り)
    --slippage   スリッページ率 (デフォルト: 0.15)
    --min-prob   最低確率閾値 (デフォルト: 0.03)

出力例:
    ================================================================================
    RACE: 202602030505 (東京 5R)
    ================================================================================
    馬番  馬名                p(win)  O10    EV利益   推奨
    --------------------------------------------------------------------------------
      3  ホースネーム         0.180   6.5    +0.006  ★ BUY
      1  アナザーホース       0.150   5.2    -0.087  SKIP
      2  サードホース         0.080  12.0    -0.184  SKIP
    ================================================================================
    推奨: 3番 ホースネーム を 500円 で購入
    ================================================================================
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from app.services.ev_service import EVService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"


# =============================================================================
# Pickle互換クラス
# =============================================================================


class LGBMClassifierWrapper:
    """旧pickle互換: __main__.LGBMClassifierWrapper を復元するための最小クラス。"""

    def __init__(self, params: dict | None = None, num_boost_round: int = 1000):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names: list[str] = []
        self.classes_ = [0, 1]

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        import numpy as np

        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds])


class _CompatUnpickler(pickle.Unpickler):
    """古い __main__ 参照pickleを読み込むための互換Unpickler。"""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "LGBMClassifierWrapper":
            return LGBMClassifierWrapper
        return super().find_class(module, name)


def _load_pickle_with_compat(path: Path):
    """pickleロード。互換クラス不足時は __main__ 参照を補正して再読込する。"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except AttributeError as exc:
        if "LGBMClassifierWrapper" not in str(exc):
            raise
        logger.warning(
            "pickle互換ロードを実行: %s (__main__.LGBMClassifierWrapper を補正)",
            path,
        )
        with open(path, "rb") as f:
            return _CompatUnpickler(f).load()


def predict_with_optional_calibrator(model, calibrator, matrix):
    """確率予測を返す。calibratorが無ければmodel出力をそのまま利用する。"""
    if calibrator is not None:
        return calibrator.predict_proba(matrix)[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    return model.predict(matrix)


def coerce_model_matrix(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """推論入力を学習時feature順に揃え、数値列へ正規化する。"""
    matrix = df.reindex(columns=feature_names).copy()
    for column in matrix.columns:
        if pd.api.types.is_numeric_dtype(matrix[column]):
            continue
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


# =============================================================================
# 特徴量定義 (train.py と同じ)
# =============================================================================

FEATURE_COLS = [
    "track_code",
    "surface",
    "distance_m",
    "going",
    "class_code",
    "field_size",
    "gate",
    "carried_weight",
    "body_weight",
    "body_weight_diff",
    "days_since_last",
    "distance_change_m",
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
    "speed_sim_mean_3",
    "speed_sim_best_5",
    "closing_sim_mean_3",
    "closing_sim_best_5",
    "early_sim_mean_3",
    "n_sim_runs_5",
    "speed_best_5_z_inrace",
    "closing_best_5_z_inrace",
    "early_mean_3_z_inrace",
    "speed_best_5_rank",
    "closing_best_5_rank",
    "early_mean_3_rank",
    "pace_front_runner_cnt",
    "num_faster_early",
    "jockey_win_rate_1y",
    "jockey_place_rate_1y",
    "trainer_win_rate_1y",
    "trainer_place_rate_1y",
    "dm_data_kbn",
    "dm_pred_time_sec",
    "dm_rank",
    "dm_missing_flag",
    "tm_data_kbn",
    "tm_score",
    "tm_rank",
    "tm_missing_flag",
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
    "ck_h_total_starts",
    "ck_h_total_wins",
    "ck_h_total_top3",
    "ck_h_total_top5",
    "ck_h_total_out",
    "ck_h_central_starts",
    "ck_h_central_wins",
    "ck_h_central_top3",
    "ck_h_turf_right_starts",
    "ck_h_turf_left_starts",
    "ck_h_turf_straight_starts",
    "ck_h_dirt_right_starts",
    "ck_h_dirt_left_starts",
    "ck_h_dirt_straight_starts",
    "ck_h_turf_good_starts",
    "ck_h_turf_soft_starts",
    "ck_h_turf_heavy_starts",
    "ck_h_turf_bad_starts",
    "ck_h_dirt_good_starts",
    "ck_h_dirt_soft_starts",
    "ck_h_dirt_heavy_starts",
    "ck_h_dirt_bad_starts",
    "ck_h_turf_16down_starts",
    "ck_h_turf_22down_starts",
    "ck_h_turf_22up_starts",
    "ck_h_dirt_16down_starts",
    "ck_h_dirt_22down_starts",
    "ck_h_dirt_22up_starts",
    "ck_h_style_nige_cnt",
    "ck_h_style_senko_cnt",
    "ck_h_style_sashi_cnt",
    "ck_h_style_oikomi_cnt",
    "ck_h_registered_races_n",
    "ck_j_year_flat_prize_total",
    "ck_j_year_ob_prize_total",
    "ck_j_cum_flat_prize_total",
    "ck_j_cum_ob_prize_total",
    "ck_t_year_flat_prize_total",
    "ck_t_year_ob_prize_total",
    "ck_t_cum_flat_prize_total",
    "ck_t_cum_ob_prize_total",
    "ck_o_year_prize_total",
    "ck_o_cum_prize_total",
    "ck_b_year_prize_total",
    "ck_b_cum_prize_total",
    "ck_missing_flag",
    "ck_overall_win_rate_sm",
    "ck_overall_top3_rate_sm",
    "ck_overall_top5_rate_sm",
    "ck_central_win_rate_sm",
    "ck_central_top3_rate_sm",
]


# =============================================================================
# ユーティリティ
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
    if going is None or going <= 2:
        return 1
    return 2


def parse_odds(odds_str: str) -> dict[int, float]:
    """オッズ文字列をパース"""
    odds = {}
    for pair in odds_str.split(","):
        if ":" not in pair:
            continue
        horse_no, odd = pair.split(":")
        odds[int(horse_no.strip())] = float(odd.strip())
    return odds


# =============================================================================
# モデルロード
# =============================================================================


def load_model():
    """モデルと校正器をロード"""
    model_path = MODEL_DIR / "lgb_model.pkl"
    calibrator_path = MODEL_DIR / "calibrator.pkl"

    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        logger.error("先に train.py を実行してください")
        sys.exit(1)

    model_data = _load_pickle_with_compat(model_path)

    calibrator = None
    if calibrator_path.exists():
        calibrator = _load_pickle_with_compat(calibrator_path)
    else:
        logger.warning("校正器が見つかりません。モデル生確率で推論します: %s", calibrator_path)

    return model_data["model"], calibrator, model_data["feature_names"]


# =============================================================================
# 特徴量生成
# =============================================================================


def get_race_features(db: Database, race_id: int) -> pd.DataFrame:
    """レースの特徴量を取得"""
    # レース情報
    race_query = """
    SELECT race_id, race_date, track_code, surface, distance_m, going, class_code, field_size
    FROM core.race
    WHERE race_id = %s
    """
    race = db.fetch_one(race_query, (race_id,))
    if not race:
        raise ValueError(f"レースが見つかりません: {race_id}")

    race_date = race["race_date"]
    distance_bucket = distance_to_bucket(race["distance_m"])
    going_bucket = going_to_bucket(race["going"])

    # 出走馬情報
    runners_query = """
    SELECT
        run.horse_id,
        run.horse_no,
        run.gate,
        run.jockey_id,
        run.trainer_id,
        run.carried_weight,
        run.body_weight,
        run.body_weight_diff,
        h.horse_name
    FROM core.runner run
    LEFT JOIN core.horse h ON run.horse_id = h.horse_id
    WHERE run.race_id = %s AND run.scratch_flag = FALSE
    ORDER BY run.horse_no
    """
    runners = db.fetch_all(runners_query, (race_id,))

    if not runners:
        raise ValueError(f"出走馬が見つかりません: {race_id}")

    df = pd.DataFrame(runners)

    # レース情報を追加
    df["race_id"] = race_id
    df["race_date"] = race_date
    df["track_code"] = race["track_code"]
    df["surface"] = race["surface"]
    df["distance_m"] = race["distance_m"]
    df["going"] = race["going"]
    df["class_code"] = race["class_code"]
    df["field_size"] = race["field_size"]
    df["distance_bucket"] = distance_bucket
    df["going_bucket"] = going_bucket

    # horse_stats を結合
    df = _join_horse_stats(db, df, race_date)

    # person_stats を結合
    df = _join_person_stats(db, df, race_date)

    # レース内相対特徴量
    df = _add_inrace_features(df)

    # ペース圧特徴量
    df = _add_pace_features(df)

    # 当日コンディション
    df = _add_condition_features(db, df, race_date)

    return df


def _join_horse_stats(db: Database, df: pd.DataFrame, race_date) -> pd.DataFrame:
    """horse_stats を結合"""
    stats_query = """
    SELECT
        horse_id, target_surface, target_distance_bucket, target_going_bucket,
        speed_last, speed_mean_3, speed_best_5, speed_std_5, speed_trend_3,
        closing_last, closing_mean_3, closing_best_5,
        early_mean_3, early_best_5, position_gain_mean_3,
        finish_mean_3, finish_best_5, n_runs_5,
        speed_sim_mean_3, speed_sim_best_5,
        closing_sim_mean_3, closing_sim_best_5,
        early_sim_mean_3, n_sim_runs_5
    FROM mart.horse_stats
    WHERE calc_date = %s
    """
    stats_df = pd.DataFrame(db.fetch_all(stats_query, (race_date,)))

    if stats_df.empty:
        return df

    df = df.merge(
        stats_df,
        left_on=["horse_id", "surface", "distance_bucket", "going_bucket"],
        right_on=["horse_id", "target_surface", "target_distance_bucket", "target_going_bucket"],
        how="left",
    )

    drop_cols = ["target_surface", "target_distance_bucket", "target_going_bucket"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def _join_person_stats(db: Database, df: pd.DataFrame, race_date) -> pd.DataFrame:
    """person_stats を結合"""
    jockey_query = """
    SELECT person_id as jockey_id,
           win_rate_1y as jockey_win_rate_1y,
           place_rate_1y as jockey_place_rate_1y
    FROM mart.person_stats
    WHERE calc_date = %s AND person_type = 'jockey'
    """
    jockey_df = pd.DataFrame(db.fetch_all(jockey_query, (race_date,)))
    if not jockey_df.empty:
        df = df.merge(jockey_df, on="jockey_id", how="left")

    trainer_query = """
    SELECT person_id as trainer_id,
           win_rate_1y as trainer_win_rate_1y,
           place_rate_1y as trainer_place_rate_1y
    FROM mart.person_stats
    WHERE calc_date = %s AND person_type = 'trainer'
    """
    trainer_df = pd.DataFrame(db.fetch_all(trainer_query, (race_date,)))
    if not trainer_df.empty:
        df = df.merge(trainer_df, on="trainer_id", how="left")

    return df


def _add_inrace_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量"""
    target_cols = ["speed_best_5", "closing_best_5", "early_mean_3"]

    for col in target_cols:
        if col not in df.columns:
            continue

        # psycopg may return Decimal for NUMERIC; coerce to float for stable stats.
        values = pd.to_numeric(df[col], errors="coerce").astype(float)
        race_mean = float(values.mean())
        race_std = float(values.std())
        if race_std == 0 or pd.isna(race_std):
            race_std = 1

        df[f"{col}_z_inrace"] = (values - race_mean) / race_std
        df[f"{col}_rank"] = values.rank(ascending=False, method="min")

    return df


def _add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """ペース圧特徴量"""
    if "early_mean_3_rank" not in df.columns:
        return df

    front_count = (df["early_mean_3_rank"] <= 3).sum()
    df["pace_front_runner_cnt"] = front_count
    df["num_faster_early"] = (df["early_mean_3_rank"] - 1).fillna(0).astype(int)

    return df


def _add_condition_features(db: Database, df: pd.DataFrame, race_date) -> pd.DataFrame:
    """当日コンディション特徴量"""
    horse_ids = df["horse_id"].tolist()
    if not horse_ids:
        df["days_since_last"] = None
        df["distance_change_m"] = None
        return df

    placeholders = ",".join(["%s"] * len(horse_ids))
    prev_query = f"""
    SELECT
        run.horse_id,
        MAX(r.race_date) as prev_race_date,
        (SELECT r2.distance_m FROM core.race r2
         JOIN core.runner run2 ON r2.race_id = run2.race_id
         WHERE run2.horse_id = run.horse_id AND r2.race_date < %s
         ORDER BY r2.race_date DESC LIMIT 1) as prev_distance_m
    FROM core.runner run
    JOIN core.race r ON run.race_id = r.race_id
    WHERE run.horse_id IN ({placeholders})
      AND r.race_date < %s
    GROUP BY run.horse_id
    """
    prev_df = pd.DataFrame(db.fetch_all(prev_query, [race_date] + horse_ids + [race_date]))

    if prev_df.empty:
        df["days_since_last"] = None
        df["distance_change_m"] = None
        return df

    prev_df["prev_race_date"] = pd.to_datetime(prev_df["prev_race_date"], errors="coerce")
    race_ts = pd.to_datetime(race_date)
    prev_df["days_since_last"] = (race_ts - prev_df["prev_race_date"]).dt.days

    df = df.merge(
        prev_df[["horse_id", "days_since_last", "prev_distance_m"]], on="horse_id", how="left"
    )
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["prev_distance_m"] = pd.to_numeric(df["prev_distance_m"], errors="coerce")
    df["distance_change_m"] = df["distance_m"] - df["prev_distance_m"].fillna(df["distance_m"])
    df = df.drop(columns=["prev_distance_m"], errors="ignore")

    return df


# =============================================================================
# 推論
# =============================================================================


def predict_race(
    db: Database,
    race_id: int,
    odds_dict: dict[int, float],
    slippage: float = 0.15,
    min_prob: float = 0.03,
) -> None:
    """レースの予測を実行して結果を出力"""
    # モデルロード
    model, calibrator, feature_names = load_model()

    # 特徴量取得
    df = get_race_features(db, race_id)

    # 学習時のfeature_namesに合わせて列順を固定（不足列はNaNで補完）
    missing_features = [c for c in feature_names if c not in df.columns]
    if missing_features:
        logger.warning(f"推論時に欠損している特徴量: {missing_features}")
    X = coerce_model_matrix(df, feature_names)

    # 予測
    _ = model.predict(X)  # raw_probaは未使用
    calibrated_proba = predict_with_optional_calibrator(model, calibrator, X)

    df["p_win"] = calibrated_proba

    # オッズを追加
    df["odds_10min"] = df["horse_no"].map(odds_dict)

    # オッズがない馬は除外
    df_with_odds = df[df["odds_10min"].notna()].copy()

    if df_with_odds.empty:
        print("オッズが入力された馬がいません")
        return

    # EV計算
    ev_service = EVService(slippage=slippage, min_prob=min_prob)

    candidates = df_with_odds[["horse_no", "horse_name", "p_win", "odds_10min"]].to_dict("records")

    # 結果出力
    print()
    print(f"RACE: {race_id}")
    print(ev_service.format_recommendation(candidates))

    # ログ保存
    _save_prediction_log(db, race_id, candidates, ev_service)


def _save_prediction_log(
    db: Database,
    race_id: int,
    candidates: list[dict],
    ev_service: EVService,
) -> None:
    """予測ログを保存"""
    predicted_at = datetime.now()

    insert_query = """
    INSERT INTO mart.prediction_log
        (predicted_at, race_id, horse_no, horse_name, p_win, odds_10min,
         odds_effective, ev_profit, recommendation, bet_amount)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for c in candidates:
        ev_result = ev_service.calculate_ev(c["p_win"], c["odds_10min"])
        recommendation = "buy" if ev_result.is_buy else "skip"
        bet_amount = ev_service.bet_amount if ev_result.is_buy else None

        db.execute(
            insert_query,
            (
                predicted_at,
                race_id,
                c["horse_no"],
                c.get("horse_name", ""),
                c["p_win"],
                c["odds_10min"],
                ev_result.odds_effective,
                ev_result.ev_profit,
                recommendation,
                bet_amount,
            ),
        )

    db.connect().commit()
    logger.info(f"予測ログ保存: {len(candidates)} 件")


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="CLI推論")
    parser.add_argument("--race-id", type=int, required=True, help="レースID")
    parser.add_argument(
        "--odds", type=str, required=True, help="10分前オッズ (馬番:オッズ のカンマ区切り)"
    )
    parser.add_argument("--slippage", type=float, default=0.15, help="スリッページ率")
    parser.add_argument("--min-prob", type=float, default=0.03, help="最低確率閾値")
    args = parser.parse_args()

    odds_dict = parse_odds(args.odds)

    with Database() as db:
        predict_race(
            db,
            race_id=args.race_id,
            odds_dict=odds_dict,
            slippage=args.slippage,
            min_prob=args.min_prob,
        )


if __name__ == "__main__":
    main()
