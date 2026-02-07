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
]

# カテゴリカル特徴量
CATEGORICAL_COLS = ["track_code", "surface", "going", "class_code"]

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

    # 時系列分割: 日付でソートして後半をテスト
    if "race_date" in df.columns:
        df_sorted = df.sort_values("race_date")
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        logger.info(f"時系列分割: 学習={len(X_train)}, テスト={len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    log_to_wandb(
        wandb_run,
        {
            "data/n_train": len(X_train),
            "data/n_test": len(X_test),
        },
    )

    # LightGBM学習 (early stopping 付き)
    extra_params = GPU_LGB_PARAMS if use_gpu else {}
    if use_gpu:
        logger.info("GPUモードで学習します")
    lgb_model = LGBMClassifierWrapper(params=extra_params, num_boost_round=num_boost_round)
    lgb_model.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        early_stopping_rounds=50,
        wandb_run=wandb_run,
    )
    logger.info(f"Best iteration: {lgb_model.best_iteration}")

    # テストデータでの評価
    y_pred_prob = lgb_model.predict(X_test)
    logloss = log_loss(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    logger.info(f"テスト Logloss: {logloss:.4f}, AUC: {auc:.4f}")

    log_to_wandb(
        wandb_run,
        {
            "test/logloss_raw": logloss,
            "test/auc": auc,
        },
    )

    # 確率校正 (sklearn 1.6+ 互換: IsotonicRegressionを直接使用)
    logger.info("確率校正中...")
    from sklearn.isotonic import IsotonicRegression

    y_pred_raw = lgb_model.predict_proba(X_test)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_raw, y_test)

    # 校正後の評価
    y_pred_calibrated = calibrator.predict(y_pred_raw)
    logloss_cal = log_loss(y_test, y_pred_calibrated)
    logger.info(f"校正後 Logloss: {logloss_cal:.4f}")

    log_to_wandb(
        wandb_run,
        {
            "test/logloss_calibrated": logloss_cal,
            "test/logloss_improvement": logloss - logloss_cal,
        },
    )

    # 校正の確認 (予測確率 vs 実際の勝率)
    prob_pred_list, prob_true_list = _check_calibration(y_test, y_pred_calibrated)
    log_calibration_to_wandb(wandb_run, prob_pred_list, prob_true_list)

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

    return lgb_model, calibrator, available_features


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
        # 学習
        lgb_model, calibrator, feature_names = train_model(
            df,
            test_size=args.test_size,
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
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)
        logger.info(f"校正器保存: {calibrator_path}")

        # W&B にモデルをアーティファクトとして保存
        if wandb_run:
            try:
                import wandb

                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(str(model_path))
                artifact.add_file(str(calibrator_path))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"W&B アーティファクト保存失敗: {e}")

        logger.info("学習完了")

    finally:
        finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
