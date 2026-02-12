"""
ケリー基準バックテストスクリプト

戦略A (ケリーフィルタ + 固定額) と 戦略B (ケリー変動ベット + 資金管理) を評価する。
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from app.infrastructure.database import Database
from app.services.betting_strategy import KellyStrategy
from scripts.train import TARGET_COL, LGBMClassifierWrapper, _split_by_race_id  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# ヘルパー関数 (scripts/optimize_thresholds.py と共通化すべきだが独立させる)
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
    """最終オッズを結合"""
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

    model, calibrator, feature_names = load_model()

    data_path = DATA_DIR / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    df = pd.read_parquet(data_path)
    df["race_date"] = pd.to_datetime(df["race_date"])

    if from_date:
        df = df[df["race_date"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["race_date"] <= pd.Timestamp(to_date)]

    if use_test_split and TARGET_COL in df.columns:
        _, _, _, _, df_test, _ = _split_by_race_id(df, df, df[TARGET_COL], test_size=0.2)
        df = df_test
        logger.info(f"Testセット抽出完了: {len(df)} 件")

    if df.empty:
        return pd.DataFrame()

    with Database() as db:
        df = _join_final_odds(db, df)

    if df.empty:
        return pd.DataFrame()

    available_features = [c for c in feature_names if c in df.columns]
    X = df[available_features]
    raw_proba = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        calibrated_proba = calibrator.predict(raw_proba)
    else:
        calibrated_proba = raw_proba

    df["p_win"] = calibrated_proba

    # race_date順にソート (時系列シミュレーションのため必須)
    df = df.sort_values(["race_date", "race_id"])

    return df


# =============================================================================
# シミュレーション関数
# =============================================================================


def run_simulation(
    df: pd.DataFrame, strategy: KellyStrategy, initial_bankroll: int, slippage: float
) -> dict:
    """
    時系列シミュレーションを実行

    Args:
        strategy: KellyStrategyインスタンス
    """
    bankroll = initial_bankroll
    current_funds = bankroll

    bets_history = []

    # レースごとに処理する
    # race_date でソート済みであることを前提とする
    # unique race_ids を使う
    race_ids = df["race_id"].unique()

    min_funds = bankroll
    max_funds = bankroll
    bankruptcy = False

    for race_id in race_ids:
        # 破産チェック
        if current_funds <= 0:
            bankruptcy = True
            break

        # 当該レースのデータ（全馬）
        # boolean indexingで抽出（簡易実装）
        race_df = df[df["race_id"] == race_id]

        # 各馬についてベット判断
        # Kelly基準では、複数馬への同時ベットは数理的に複雑 (Multi-horse Kelly)
        # ここでは「最もf*が高い1頭のみに賭ける」単一ベット戦略とする
        # scripts/backtest.py でも「EV最大の馬を選択」しているので、それに合わせる

        # まず候補全頭のf*を計算し、最大のものを探す

        candidates = []
        for _, row in race_df.iterrows():
            # スリッページ後のオッズで判断するのがユーザー要件
            odds_effective = row["odds_final"] * (1 - slippage)
            prob = row["p_win"]

            # strategy.calculate_kelly_f を直接は使えない (decide_bet 内で計算される)
            # decide_bet を呼ぶ
            decision = strategy.decide_bet(prob=prob, odds=odds_effective, bankroll=current_funds)

            if decision.should_bet:
                candidates.append(
                    {
                        "row": row,
                        "amount": decision.amount,
                        "f_star": strategy.calculate_kelly_f(prob, odds_effective),  # ログ用
                    }
                )

        if not candidates:
            continue

        # 複数候補がいる場合、f_star が最大のものを1つ選ぶ (単一ベット制約)
        # あるいは期待値最大を選ぶ？ -> Kelly戦略なら f* 最大を選ぶのが自然
        best_bet = max(candidates, key=lambda x: x["f_star"])

        # ベット実行
        bet_amount = best_bet["amount"]
        row = best_bet["row"]

        # 資金拘束（レース前に減らす）
        current_funds -= bet_amount

        # 結果判定
        is_hit = row["finish_pos"] == 1
        payout = 0
        if is_hit:
            payout = int(bet_amount * row["odds_final"])  # 払戻は実際のオッズ

        current_funds += payout

        # 記録
        bets_history.append(
            {
                "race_id": race_id,
                "bet_amount": bet_amount,
                "payout": payout,
                "profit": payout - bet_amount,
                "is_hit": is_hit,
                "bankroll_after": current_funds,
                "f_star": best_bet["f_star"],
                "race_date": row["race_date"],
            }
        )

        min_funds = min(min_funds, current_funds)
        max_funds = max(max_funds, current_funds)

    # 集計
    n_bets = len(bets_history)
    if n_bets == 0:
        return {
            "n_bets": 0,
            "roi": 0.0,
            "final_bankroll": initial_bankroll,
            "profit": 0,
            "max_drawdown": 0,
            "bankruptcy": False,
        }

    history_df = pd.DataFrame(bets_history)
    total_bet = history_df["bet_amount"].sum()
    total_return = history_df["payout"].sum()
    profit = total_return - total_bet
    roi = total_return / total_bet if total_bet > 0 else 0.0

    # ドローダウン計算
    # 資金推移からのドローダウン (Peak - Current)
    # history_dfには "bankroll_after" がある
    # 開始時資金も含める
    funds_series = [initial_bankroll] + history_df["bankroll_after"].tolist()
    funds_series = pd.Series(funds_series)
    running_max = funds_series.cummax()
    drawdowns = running_max - funds_series
    max_drawdown = drawdowns.max()

    return {
        "n_bets": n_bets,
        "roi": roi,
        "final_bankroll": int(current_funds),
        "profit": int(profit),
        "max_drawdown": int(max_drawdown),
        "bankruptcy": bankruptcy,
    }


def main():
    parser = argparse.ArgumentParser(description="ケリー基準バックテスト")
    parser.add_argument(
        "--from-date", type=str, default=None, help="開始日 (use_test_split時は無視)"
    )
    parser.add_argument("--to-date", type=str, default=None, help="終了日 (use_test_split時は無視)")
    parser.add_argument("--slippage", type=float, default=0.15)
    parser.add_argument("--initial-bankroll", type=int, default=50000)
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

    results = []

    # =============================================================================
    # 戦略A: ケリー基準フィルタ (固定額)
    # =============================================================================
    logger.info("=== Strategy A: Kelly Filter (Fixed 500 yen) ===")
    fractions_a = [
        0.5,
        1.0,
    ]  # フィルタモードでも係数を変えて境界付近での挙動確認

    for frac in fractions_a:
        strategy = KellyStrategy(
            kelly_fraction=frac,
            min_bet=500,
            max_bet=3000,
            fixed_bet_mode=True,  # 固定額モード (amount=fixed_bet_amount)
            fixed_bet_amount=500,
        )

        # 初期資金は十分にないと破産して検証にならないので多めに設定するか、
        # あるいは「固定額」なので資金枯渇を無視して全レース賭けるロジックにするか。
        # run_simulation は資金枯渇で停止するので、十分な資金を与える。
        res = run_simulation(df, strategy, initial_bankroll=1_000_000, slippage=args.slippage)

        res["strategy"] = "A (Fixed)"
        res["fraction"] = frac
        results.append(res)

    # =============================================================================
    # 戦略B: ケリー変動ベット
    # =============================================================================
    logger.info("=== Strategy B: Kelly Variable Bet ===")
    fractions_b = [0.1, 0.25, 0.5, 1.0]

    for frac in fractions_b:
        strategy = KellyStrategy(
            kelly_fraction=frac, min_bet=500, max_bet=3000, fixed_bet_mode=False
        )

        # こちらはユーザー指定の初期資金で検証
        res = run_simulation(
            df, strategy, initial_bankroll=args.initial_bankroll, slippage=args.slippage
        )

        res["strategy"] = "B (Variable)"
        res["fraction"] = frac
        results.append(res)

    # =============================================================================
    # 結果表示
    # =============================================================================
    res_df = pd.DataFrame(results)

    cols = [
        "strategy",
        "fraction",
        "roi",
        "profit",
        "final_bankroll",
        "max_drawdown",
        "n_bets",
        "bankruptcy",
    ]
    print("\n=== Backtest Results ===")
    print(
        tabulate(res_df[cols], headers="keys", tablefmt="github", floatfmt=".4f", showindex=False)
    )


if __name__ == "__main__":
    main()
