"""
モデル学習スクリプト

LightGBMで勝率予測モデルを学習し、確率校正を適用する。
W&Bで学習メトリクスを記録する。

使用方法:
    uv run python scripts/train.py

オプション:
    --no-wandb   W&B記録を無効化

出力:
    models/lgb_model.pkl
    models/calibrator.pkl

前提:
    build_dataset.py が実行済みで data/train.parquet が存在すること
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# .env から環境変数を読み込み
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


# =============================================================================
# 特徴量定義
# =============================================================================

# モデルに使用する特徴量
FEATURE_COLS = [
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

# カテゴリカル特徴量
CATEGORICAL_COLS = ["track_code", "surface", "going", "class_code", "wood_course", "wood_direction"]

# 目的変数
TARGET_COL = "is_win"

# LightGBM デフォルトパラメータ
DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# LightGBM GPU用パラメータ (追加・上書き)
GPU_LGB_PARAMS = {
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}


# =============================================================================
# W&B 初期化
# =============================================================================


def init_wandb(enabled: bool = True, config: dict | None = None):
    """W&B を初期化"""
    if not enabled:
        return None

    try:
        import wandb

        project = os.getenv("WANDB_PROJECT", "keiba-prediction")
        run = wandb.init(
            project=project,
            config=config or {},
            tags=["training", "lightgbm"],
        )
        logger.info(f"W&B 初期化: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"W&B 初期化失敗: {e}")
        return None


def log_to_wandb(run, metrics: dict, step: int | None = None):
    """W&B にメトリクスを記録"""
    if run is None:
        return

    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"W&B ログ失敗: {e}")


def log_feature_importance_to_wandb(run, importance_df: pd.DataFrame):
    """特徴量重要度を W&B に記録"""
    if run is None:
        return

    try:
        import wandb

        # 棒グラフとして記録
        table = wandb.Table(dataframe=importance_df.head(30))
        wandb.log(
            {
                "feature_importance": wandb.plot.bar(
                    table, "feature", "importance", title="Feature Importance (Top 30)"
                )
            }
        )

        # アーティファクトとしても保存
        artifact = wandb.Artifact("feature_importance", type="dataset")
        artifact.add(table, "importance")
        run.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"W&B 特徴量重要度ログ失敗: {e}")


def log_calibration_to_wandb(run, prob_pred: list, prob_true: list):
    """校正曲線を W&B に記録"""
    if run is None:
        return

    try:
        import wandb

        # 校正曲線のデータ
        data = [[p, t] for p, t in zip(prob_pred, prob_true, strict=True)]
        table = wandb.Table(data=data, columns=["predicted", "actual"])
        wandb.log(
            {
                "calibration_curve": wandb.plot.scatter(
                    table, "predicted", "actual", title="Calibration Curve"
                )
            }
        )
    except Exception as e:
        logger.warning(f"W&B 校正曲線ログ失敗: {e}")


def finish_wandb(run):
    """W&B を終了"""
    if run is None:
        return

    try:
        import wandb

        wandb.finish()
    except Exception as e:
        logger.warning(f"W&B 終了失敗: {e}")


# =============================================================================
# LightGBMラッパー (sklearn互換)
# =============================================================================


def wandb_callback(wandb_run):
    """W&B に学習曲線を記録するコールバック"""

    def callback(env):
        if wandb_run is None:
            return
        metrics = {}
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            metrics[f"train/{data_name}_{metric_name}"] = value
        wandb_run.log(metrics, step=env.iteration)

    return callback


class LGBMClassifierWrapper:
    """LightGBM を sklearn 互換で使うためのラッパー"""

    def __init__(self, params: dict | None = None, num_boost_round: int = 1000):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names = None
        self.best_iteration = None

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50, wandb_run=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)

        # validation set
        valid_sets = [train_data]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        params = {**DEFAULT_LGB_PARAMS, **self.params}

        # コールバック設定
        callbacks = []
        if X_val is not None and early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
            )
        callbacks.append(lgb.log_evaluation(period=100))

        # W&B コールバック
        if wandb_run:
            callbacks.append(wandb_callback(wandb_run))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.best_iteration = (
            self.model.best_iteration
            if hasattr(self.model, "best_iteration")
            else self.num_boost_round
        )
        return self

    def predict(self, X):
        """確率を返す (CalibratedClassifierCV 用)"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """sklearn形式の確率を返す (2列)"""
        prob = self.predict(X)
        return np.column_stack([1 - prob, prob])

    def get_params(self, deep=True):
        return {"params": self.params, "num_boost_round": self.num_boost_round}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# =============================================================================
# データ分割
# =============================================================================


def _split_by_race_id(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    es_val_size: float = 0.1,
) -> tuple:
    """
    race_id 境界で時系列分割 (Train / ES-Val / Test)

    同一レースが複数のセットに跨がらないよう、レース単位で分割する。

    Args:
        test_size: テストデータの割合 (0.0 の場合は本番学習モード)
        es_val_size: Early Stopping 用検証データの割合

    Returns:
        (X_train, y_train, X_es_val, y_es_val, X_test, y_test)
    """
    if "race_id" not in df.columns or "race_date" not in df.columns:
        raise ValueError("race_id と race_date が必要です")

    # レース単位でソート
    unique_races = df[["race_id", "race_date"]].drop_duplicates().sort_values("race_date")
    n_races = len(unique_races)

    # 割合計算
    # test_size が 0.0 (本番モード) の場合も意図通り動作する
    n_test = int(n_races * test_size)
    n_es = int(n_races * es_val_size)
    n_train = n_races - n_test - n_es

    if n_train <= 0:
        raise ValueError("学習データが少なすぎます")

    # 分割インデックス
    train_end = n_train
    es_val_end = n_train + n_es

    # race_id のセットを作成
    train_races = set(unique_races.iloc[:train_end]["race_id"])
    es_val_races = set(unique_races.iloc[train_end:es_val_end]["race_id"])
    test_races = set(unique_races.iloc[es_val_end:]["race_id"])

    # 行インデックスにマップ
    train_idx = df[df["race_id"].isin(train_races)].index
    es_val_idx = df[df["race_id"].isin(es_val_races)].index
    test_idx = df[df["race_id"].isin(test_races)].index

    return (
        X.loc[train_idx],
        y.loc[train_idx],
        X.loc[es_val_idx],
        y.loc[es_val_idx],
        X.loc[test_idx],
        y.loc[test_idx],
    )


# =============================================================================
# 学習
# =============================================================================


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    num_boost_round: int = 500,
    use_gpu: bool = False,
    wandb_run=None,
) -> tuple:
    """モデルを学習"""
    logger.info("モデル学習開始...")

    # 特徴量が存在するかチェック
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = set(FEATURE_COLS) - set(available_features)
    if missing_features:
        logger.warning(f"欠損している特徴量: {missing_features}")

    X = df[available_features]
    y = df[TARGET_COL]

    logger.info(f"データサイズ: {len(X)} 件, 特徴量数: {len(available_features)}")
    logger.info(f"正例率: {y.mean():.4f}")

    # W&B にデータ統計を記録
    log_to_wandb(
        wandb_run,
        {
            "data/n_samples": len(X),
            "data/n_features": len(available_features),
            "data/positive_rate": y.mean(),
        },
    )

    # 時系列分割: race_id境界で Train/ES-Val/Test に分割
    if "race_id" in df.columns and "race_date" in df.columns:
        # test_size が 0.0 なら本番モード (Testなし)
        X_train, y_train, X_es_val, y_es_val, X_test, y_test = _split_by_race_id(
            df, X, y, test_size=test_size
        )
        logger.info(f"分割: Train={len(X_train)}, ES-Val={len(X_es_val)}, Test={len(X_test)}")
    else:
        # race_id がない場合はフォールバック（レガシー）
        logger.warning("race_id がないため、従来の2分割を使用 (リークあり)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        # ES-Val は Test と同じにする (簡易的)
        X_es_val, y_es_val = X_test, y_test

    log_to_wandb(
        wandb_run,
        {
            "data/n_train": len(X_train),
            "data/n_es_val": len(X_es_val),
            "data/n_test": len(X_test),
        },
    )

    # LightGBM学習 (early stopping は ES-Val のみ)
    extra_params = GPU_LGB_PARAMS if use_gpu else {}
    if use_gpu:
        logger.info("GPUモードで学習します")
    lgb_model = LGBMClassifierWrapper(params=extra_params, num_boost_round=num_boost_round)
    lgb_model.fit(
        X_train,
        y_train,
        X_val=X_es_val,
        y_val=y_es_val,
        early_stopping_rounds=50,
        wandb_run=wandb_run,
    )
    logger.info(f"Best iteration: {lgb_model.best_iteration}")

    # テストデータでの評価
    if len(X_test) > 0:
        y_pred_test_raw = lgb_model.predict_proba(X_test)[:, 1]
        logloss_raw = log_loss(y_test, y_pred_test_raw)
        auc = roc_auc_score(y_test, y_pred_test_raw)
        logger.info(f"テスト Logloss: {logloss_raw:.4f}, AUC: {auc:.4f}")

        # ECE の計算
        ece = _calculate_ece(y_test, y_pred_test_raw)
        logger.info(f"Test ECE: {ece:.4f}")

        log_to_wandb(
            wandb_run,
            {
                "test/logloss": logloss_raw,
                "test/auc": auc,
                "test/ece": ece,
            },
        )

        # 校正チェック (W&B記録)
        prob_pred_list, prob_true_list = _check_calibration(y_test, y_pred_test_raw)
        log_calibration_to_wandb(wandb_run, prob_pred_list, prob_true_list)
    else:
        logger.info("Testセットがないため評価をスキップします (本番学習モード)")

    # 特徴量重要度
    importance = pd.DataFrame(
        {
            "feature": lgb_model.feature_names,
            "importance": lgb_model.model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    logger.info("特徴量重要度 Top 10:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    log_feature_importance_to_wandb(wandb_run, importance)

    # 校正なし (None)
    calibrator = None

    return lgb_model, calibrator, available_features


def _calculate_ece(y_true, y_pred, n_bins=10):
    """Expected Calibration Error を計算"""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    # ビンごとのサンプル数を計算 (簡易実装)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    n_total = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=True):
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            accuracy = np.mean(y_true[in_bin])
            avg_confidence = np.mean(y_pred[in_bin])
            ece += np.abs(accuracy - avg_confidence) * (n_in_bin / n_total)

    return ece


def _check_calibration(y_true: pd.Series, y_pred: np.ndarray) -> tuple[list, list]:
    """校正をチェック"""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

    logger.info("校正チェック (予測確率 vs 実際の勝率):")
    for pt, pp in zip(prob_pred, prob_true, strict=True):
        logger.info(f"  予測 {pp:.3f} → 実際 {pt:.3f}")

    return prob_pred.tolist(), prob_true.tolist()


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="モデル学習")
    parser.add_argument("--input", type=str, default="data/train.parquet", help="入力ファイル")
    parser.add_argument("--test-size", type=float, default=0.2, help="テストサイズ")
    parser.add_argument("--num-boost-round", type=int, default=500, help="ブーストラウンド数")
    parser.add_argument("--gpu", action="store_true", help="GPUを使用して学習")
    parser.add_argument("--no-wandb", action="store_true", help="W&B記録を無効化")
    parser.add_argument("--production", action="store_true", help="本番学習モード (全データ使用)")
    args = parser.parse_args()

    # データ読み込み
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not input_path.exists():
        logger.error(f"データファイルが見つかりません: {input_path}")
        logger.error("先に build_dataset.py を実行してください")
        return

    df = pd.read_parquet(input_path)
    logger.info(f"データ読み込み: {len(df)} 件")

    # W&B 設定
    config = {
        "model": "lightgbm",
        "test_size": args.test_size,
        "num_boost_round": args.num_boost_round,
        "use_gpu": args.gpu,
        "n_features": len(FEATURE_COLS),
        **DEFAULT_LGB_PARAMS,
        **(GPU_LGB_PARAMS if args.gpu else {}),
    }

    wandb_run = init_wandb(enabled=not args.no_wandb, config=config)

    try:
        # 本番モードなら test_size=0.0
        test_size = 0.0 if args.production else args.test_size

        if args.production:
            logger.info("=== 本番学習モード (Production Mode) ===")
            logger.info("Testセットを作成せず、直近データまで含めて学習します")

        # 学習
        lgb_model, calibrator, feature_names = train_model(
            df,
            test_size=test_size,
            num_boost_round=args.num_boost_round,
            use_gpu=args.gpu,
            wandb_run=wandb_run,
        )

        # モデル保存
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODEL_DIR / "lgb_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": lgb_model, "feature_names": feature_names}, f)
        logger.info(f"モデル保存: {model_path}")

        calibrator_path = MODEL_DIR / "calibrator.pkl"
        if calibrator is not None:
            with open(calibrator_path, "wb") as f:
                pickle.dump(calibrator, f)
            logger.info(f"校正器保存: {calibrator_path}")
        else:
            # 校正器なしの場合はファイルを削除（もしあれば）
            if calibrator_path.exists():
                calibrator_path.unlink()
            logger.info("校正器は保存されません (None)")

        # W&B にモデルをアーティファクトとして保存
        if wandb_run:
            try:
                import wandb

                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(str(model_path))
                if calibrator is not None and calibrator_path.exists():
                    artifact.add_file(str(calibrator_path))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"W&B アーティファクト保存失敗: {e}")

        logger.info("学習完了")

    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
