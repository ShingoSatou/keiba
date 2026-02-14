"""
バックテストスクリプト

過去データに対してモデル推論→EV計算→収支シミュレーションを実行する。

使用方法:
    uv run python scripts/backtest.py

オプション:
    --from-date    開始日 (YYYY-MM-DD)
    --to-date      終了日 (YYYY-MM-DD)
    --slippage     スリッページ率 (デフォルト: 0.15)
    --min-prob     最低確率閾値 (デフォルト: 0.03)

出力:
    - 購入レース数
    - 回収率 (ROI)
    - 最大ドローダウン
    - 校正 (予測確率 vs 実際の勝率)
    - スリッページ統計
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database
from app.services.ev_service import EVService
from scripts.train import TARGET_COL, _split_by_race_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# LGBMClassifierWrapper (pickleロード用)
# =============================================================================


class LGBMClassifierWrapper:
    """LightGBM を scikit-learn 風にラップ (pickleロード用)"""

    def __init__(self, params: dict | None = None, num_boost_round: int = 1000):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.model = None
        self.feature_names: list[str] = []
        self.classes_ = [0, 1]

    def predict(self, X):
        """生の予測値を返す"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """確率を返す"""
        import numpy as np

        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds])


# =============================================================================
# モデルロード
# =============================================================================


def load_model():
    """モデルと校正器をロード"""
    model_path = MODEL_DIR / "lgb_model.pkl"
    calibrator_path = MODEL_DIR / "calibrator.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    calibrator = None
    if calibrator_path.exists():
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)

    return model_data["model"], calibrator, model_data["feature_names"]


# =============================================================================
# バックテスト
# =============================================================================


def run_backtest(
    db: Database,
    from_date: date | None = None,
    to_date: date | None = None,
    slippage: float = 0.15,
    min_prob: float = 0.03,
    bet_amount: int = 500,
    use_test_split: bool = False,
) -> pd.DataFrame:
    """バックテストを実行"""
    logger.info("バックテスト開始...")

    # モデルロード
    model, calibrator, feature_names = load_model()

    # データセット読み込み
    data_path = DATA_DIR / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"データ読み込み: {len(df)} 件")

    # race_date を datetime に変換
    df["race_date"] = pd.to_datetime(df["race_date"])

    # 日付フィルタ
    if from_date:
        df = df[df["race_date"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["race_date"] <= pd.Timestamp(to_date)]

    # Testセットのみを使用する場合
    if use_test_split:
        logger.info("学習データのリークを防ぐため、Testセットのみ抽出します (--use-test-split)")
        # train.py と同じロジックで分割
        # X, y は分割ロジックの内部で index 抽出に使われるだけなので、df をそのまま渡す
        if TARGET_COL not in df.columns:
            logger.warning(f"{TARGET_COL} カラムがないため、Testセット抽出ができません")
        else:
            try:
                # _split_by_race_id は (X_train, y_train, X_es_val, y_es_val, X_test, y_test) を返す
                # test_size=0.2 (デフォルト) を使用
                _, _, _, _, df_test, _ = _split_by_race_id(df, df, df[TARGET_COL], test_size=0.2)
                df = df_test
                logger.info(f"Testセット抽出完了: {len(df)} 件")
            except Exception as e:
                logger.error(f"Testセット抽出に失敗しました: {e}")
                return pd.DataFrame()

    if df.empty:
        logger.warning("対象データがありません")
        return pd.DataFrame()

    logger.info(f"対象期間: {df['race_date'].min()} ~ {df['race_date'].max()}")
    logger.info(f"対象レース数: {df['race_id'].nunique()}")

    # 最終オッズを取得
    df = _join_final_odds(db, df)

    # 予測
    available_features = [c for c in feature_names if c in df.columns]
    X = df[available_features]

    raw_proba = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        # calibrator は IsotonicRegression なので predict() を使用
        calibrated_proba = calibrator.predict(raw_proba)
    else:
        calibrated_proba = raw_proba

    df["p_win"] = calibrated_proba
    df["p_win_raw"] = raw_proba

    # EV計算
    _ = EVService(slippage=slippage, min_prob=min_prob, bet_amount=bet_amount)

    df["odds_effective"] = df["odds_final"] * (1 - slippage)
    df["ev_profit"] = df["p_win"] * df["odds_effective"] - 1
    df["is_buy"] = (df["ev_profit"] > 0) & (df["p_win"] > min_prob)

    # レースごとにEV最大の馬を選択
    bet_df = _select_bets(df)

    # 評価指標を計算
    results = _calculate_metrics(df, bet_df, bet_amount)

    return results


def _join_final_odds(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """最終オッズを結合"""
    odds_query = """
    SELECT race_id, horse_id, odds_win as odds_final
    FROM core.odds_final
    WHERE odds_win IS NOT NULL
    """
    odds_df = pd.DataFrame(db.fetch_all(odds_query))

    if odds_df.empty:
        logger.warning("オッズデータがありません。バックテストにはオッズが必要です。")
        df["odds_final"] = None
        return df

    # Decimal を float に変換
    odds_df["odds_final"] = pd.to_numeric(odds_df["odds_final"], errors="coerce")

    df = df.merge(odds_df, on=["race_id", "horse_id"], how="left")

    # オッズがない行を除外
    missing = df["odds_final"].isna().sum()
    if missing > 0:
        logger.warning(f"オッズ欠損: {missing} 件を除外")
        df = df[df["odds_final"].notna()]

    return df


def _select_bets(df: pd.DataFrame) -> pd.DataFrame:
    """レースごとにEV最大の馬を選択"""
    buy_df = df[df["is_buy"]].copy()

    if buy_df.empty:
        return pd.DataFrame()

    # レースごとにEV最大を選択
    idx = buy_df.groupby("race_id")["ev_profit"].idxmax()
    bet_df = buy_df.loc[idx].copy()

    return bet_df


def _calculate_metrics(
    df: pd.DataFrame,
    bet_df: pd.DataFrame,
    bet_amount: int,
) -> dict:
    """評価指標を計算"""
    results = {}

    # === 予測品質 ===
    y_true = df["is_win"]
    y_pred = df["p_win"]

    results["logloss"] = log_loss(y_true, y_pred)
    results["auc"] = roc_auc_score(y_true, y_pred)

    logger.info(f"Logloss: {results['logloss']:.4f}")
    logger.info(f"AUC: {results['auc']:.4f}")

    # 校正
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    results["calibration"] = list(zip(prob_pred.tolist(), prob_true.tolist(), strict=True))

    logger.info("校正チェック:")
    for pp, pt in results["calibration"]:
        logger.info(f"  予測 {pp:.3f} → 実際 {pt:.3f}")

    # === 運用成績 ===
    if bet_df.empty:
        results["n_bets"] = 0
        results["total_bet"] = 0
        results["total_return"] = 0
        results["roi"] = 0.0
        results["max_drawdown"] = 0.0
        logger.warning("買い推奨がありません")
        return results

    # 勝ち: finish_pos == 1
    bet_df["is_hit"] = bet_df["finish_pos"] == 1
    bet_df["payout"] = bet_df.apply(
        lambda r: r["odds_final"] * bet_amount if r["is_hit"] else 0,
        axis=1,
    )
    bet_df["profit"] = bet_df["payout"] - bet_amount

    results["n_races"] = df["race_id"].nunique()
    results["n_bets"] = len(bet_df)
    results["n_hits"] = bet_df["is_hit"].sum()
    results["hit_rate"] = results["n_hits"] / results["n_bets"]
    results["total_bet"] = results["n_bets"] * bet_amount
    results["total_return"] = bet_df["payout"].sum()
    results["roi"] = results["total_return"] / results["total_bet"]

    logger.info(f"総レース数: {results['n_races']}")
    logger.info(f"購入レース数: {results['n_bets']}")
    logger.info(f"的中数: {results['n_hits']}")
    logger.info(f"的中率: {results['hit_rate']:.4f}")
    logger.info(f"総投資: {results['total_bet']:,}円")
    logger.info(f"総回収: {results['total_return']:,.0f}円")
    logger.info(f"回収率 (ROI): {results['roi']:.4f} ({results['roi'] * 100:.1f}%)")

    # 最大ドローダウン
    bet_df = bet_df.sort_values("race_date")
    bet_df["cumsum"] = bet_df["profit"].cumsum()
    bet_df["running_max"] = bet_df["cumsum"].cummax()
    bet_df["drawdown"] = bet_df["running_max"] - bet_df["cumsum"]
    results["max_drawdown"] = bet_df["drawdown"].max()

    logger.info(f"最大ドローダウン: {results['max_drawdown']:,.0f}円")

    # スリッページ統計（10分前オッズがあれば）
    if "odds_10min" in bet_df.columns and bet_df["odds_10min"].notna().any():
        slip = bet_df["odds_10min"] - bet_df["odds_final"]
        results["slippage_mean"] = slip.mean()
        results["slippage_std"] = slip.std()
        logger.info(
            f"スリッページ: 平均 {results['slippage_mean']:.2f}, "
            f"標準偏差 {results['slippage_std']:.2f}"
        )

    # 月別ROI
    bet_df["month"] = bet_df["race_date"].dt.to_period("M")
    monthly = bet_df.groupby("month").agg(
        {
            "profit": "sum",
            "is_hit": ["count", "sum"],
        }
    )
    monthly.columns = ["profit", "n_bets", "n_hits"]
    monthly["bet_total"] = monthly["n_bets"] * bet_amount
    monthly["roi"] = (monthly["profit"] + monthly["bet_total"]) / monthly["bet_total"]

    logger.info("月別ROI:")
    for month, row in monthly.iterrows():
        logger.info(f"  {month}: ROI {row['roi']:.4f} ({row['n_hits']}/{row['n_bets']}的中)")

    results["monthly_roi"] = monthly["roi"].to_dict()

    return results


# =============================================================================
# メイン
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="バックテスト")
    parser.add_argument("--from-date", type=str, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--slippage", type=float, default=0.15, help="スリッページ率")
    parser.add_argument("--min-prob", type=float, default=0.03, help="最低確率閾値")
    parser.add_argument("--bet-amount", type=int, default=500, help="賭け金")
    parser.add_argument(
        "--use-test-split", action="store_true", help="学習に使っていないTestデータのみを使用"
    )
    args = parser.parse_args()

    from_date = None
    to_date = None
    if args.from_date:
        from datetime import datetime

        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    if args.to_date:
        from datetime import datetime

        to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    with Database() as db:
        results = run_backtest(
            db,
            from_date=from_date,
            to_date=to_date,
            slippage=args.slippage,
            min_prob=args.min_prob,
            bet_amount=args.bet_amount,
            use_test_split=args.use_test_split,
        )

    if results:
        logger.info("バックテスト完了")
    else:
        logger.error("バックテスト失敗")


if __name__ == "__main__":
    main()
