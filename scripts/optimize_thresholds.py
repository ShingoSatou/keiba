"""
閾値最適化スクリプト

min_prob (最低勝率) と min_ev (最低期待値) のグリッドサーチを行い、
ROIを最大化するパラメータを探索する。
"""

from __future__ import annotations

import argparse
import logging
import pickle
from itertools import product
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from app.infrastructure.database import Database
from app.services.betting_strategy import FixedBetStrategy
from scripts.train import TARGET_COL, LGBMClassifierWrapper, _split_by_race_id  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# ヘルパー関数
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


def _join_final_odds(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """最終オッズを結合 (scripts/backtest.py から流用)"""
    odds_query = """
    SELECT race_id, horse_id, odds_win as odds_final
    FROM core.odds_final
    WHERE odds_win IS NOT NULL
    """
    odds_df = pd.DataFrame(db.fetch_all(odds_query))

    if odds_df.empty:
        logger.warning("オッズデータがありません。")
        df["odds_final"] = None
        return df

    odds_df["odds_final"] = pd.to_numeric(odds_df["odds_final"], errors="coerce")
    df = df.merge(odds_df, on=["race_id", "horse_id"], how="left")

    # オッズがない行を除外
    df = df[df["odds_final"].notna()]
    return df


def prepare_data(from_date=None, to_date=None, use_test_split=False):
    """データセットの読み込みと前処理"""
    logger.info("データセット準備中...")

    # モデルロード
    model, calibrator, feature_names = load_model()

    # データ読み込み
    data_path = DATA_DIR / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    df = pd.read_parquet(data_path)
    df["race_date"] = pd.to_datetime(df["race_date"])

    # 日付フィルタ
    if from_date:
        df = df[df["race_date"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["race_date"] <= pd.Timestamp(to_date)]

    # Testセット抽出
    if use_test_split and TARGET_COL in df.columns:
        _, _, _, _, df_test, _ = _split_by_race_id(df, df, df[TARGET_COL], test_size=0.2)
        df = df_test
        logger.info(f"Testセット抽出完了: {len(df)} 件")

    if df.empty:
        return pd.DataFrame()

    # オッズ結合
    with Database() as db:
        df = _join_final_odds(db, df)

    if df.empty:
        return pd.DataFrame()

    # 予測実行
    available_features = [c for c in feature_names if c in df.columns]
    X = df[available_features]
    raw_proba = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        calibrated_proba = calibrator.predict(raw_proba)
    else:
        calibrated_proba = raw_proba

    df["p_win"] = calibrated_proba

    return df


def evaluate_strategy(df: pd.DataFrame, strategy: FixedBetStrategy, slippage: float) -> dict:
    """単一の戦略設定での評価"""
    # 計算高速化のため、ループを使わずベクタライズ処理する
    # 固定額ベットなので、applyよりも単純なフィルタリングで良い

    # 実効オッズ
    odds_effective = df["odds_final"] * (1 - slippage)

    # 期待値
    ev_profit = df["p_win"] * odds_effective - 1.0

    # 購入条件 (FixedBetStrategyのロジック: prob >= min_prob AND ev > min_ev)
    # Strategyクラスを使わずに直接計算して高速化
    is_buy = (df["p_win"] >= strategy.min_prob) & (ev_profit > strategy.min_ev)

    buy_candidates = df[is_buy].copy()

    if buy_candidates.empty:
        return {
            "min_prob": strategy.min_prob,
            "min_ev": strategy.min_ev,
            "n_bets": 0,
            "roi": 0.0,
            "profit": 0,
            "hit_rate": 0.0,
        }

    # レースごとにEV最大を選択
    buy_candidates["ev_val"] = ev_profit[is_buy]  # ソート用
    # idx = buy_candidates.groupby("race_id")["ev_val"].idxmax()
    # 遅いのでsort_values + drop_duplicatesを使う

    bets = buy_candidates.sort_values("ev_val", ascending=False).drop_duplicates("race_id")

    # 成績計算
    n_bets = len(bets)
    n_hits = (bets["finish_pos"] == 1).sum()
    total_bet = n_bets * strategy.bet_amount

    # 払戻金
    # (オッズはスリッページ前のもので計算すべきだが、バックテスト仕様に合わせる)
    # ここではユーザーのリクエスト通り "EV判定" にはスリッページ後を使うが、
    # "払戻" は実際のオッズ(odds_final)で計算するのが正しい。
    # ただし今回は簡易的に odds_final を使う。
    payout = bets[bets["finish_pos"] == 1]["odds_final"].sum() * strategy.bet_amount

    profit = payout - total_bet
    roi = payout / total_bet if total_bet > 0 else 0.0

    return {
        "min_prob": strategy.min_prob,
        "min_ev": strategy.min_ev,
        "n_bets": n_bets,
        "roi": roi,
        "profit": int(profit),
        "hit_rate": n_hits / n_bets if n_bets > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="閾値グリッドサーチ")
    parser.add_argument(
        "--from-date", type=str, default=None, help="開始日 (use_test_split時は無視)"
    )
    parser.add_argument("--to-date", type=str, default=None, help="終了日 (use_test_split時は無視)")
    parser.add_argument("--slippage", type=float, default=0.15)
    parser.add_argument(
        "--use-test-split", action="store_true", help="学習に使っていないTestデータのみを使用"
    )
    args = parser.parse_args()

    # 1. データ準備
    # use_test_split が有効な場合、日付フィルタは無視してTestセットを使用
    if args.use_test_split:
        df = prepare_data(from_date=None, to_date=None, use_test_split=True)
    else:
        df = prepare_data(from_date=args.from_date, to_date=args.to_date, use_test_split=False)
    if df.empty:
        logger.error("データがありません")
        return

    logger.info(f"データ準備完了: {len(df)} 行 (レース数: {df['race_id'].nunique()})")

    # 2. グリッド定義
    # ユーザー計画書に基づいて範囲設定
    min_prob_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    min_ev_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    results = []

    logger.info(f"グリッドサーチ開始: {len(min_prob_range) * len(min_ev_range)} 通り")

    for min_prob, min_ev in product(min_prob_range, min_ev_range):
        strategy = FixedBetStrategy(bet_amount=500, min_prob=min_prob, min_ev=min_ev)
        res = evaluate_strategy(df, strategy, slippage=args.slippage)
        results.append(res)

    # 3. 集計と表示
    res_df = pd.DataFrame(results)

    # n_bets が少なすぎるもの (例えば50件未満) は信頼性が低いので
    # 除外したランキングも見たいかもしれないが、ここでは単純に ROI でソート
    res_df = res_df.sort_values("roi", ascending=False)

    print("\n=== Grid Search Results (Top 20 by ROI) ===")
    print(
        tabulate(
            res_df.head(20), headers="keys", tablefmt="github", floatfmt=".4f", showindex=False
        )
    )

    # 利益順
    res_df_profit = res_df.sort_values("profit", ascending=False)
    print("\n=== Grid Search Results (Top 10 by Profit) ===")
    print(
        tabulate(
            res_df_profit.head(10),
            headers="keys",
            tablefmt="github",
            floatfmt=".4f",
            showindex=False,
        )
    )

    # 最適パラメータの保存（簡易的にログ出力のみ）
    best_roi = res_df.iloc[0]
    logger.info(
        f"Best ROI Parameters: min_prob={best_roi['min_prob']}, "
        f"min_ev={best_roi['min_ev']} -> ROI {best_roi['roi']:.4f}"
    )


if __name__ == "__main__":
    main()
