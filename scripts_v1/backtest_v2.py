"""
バックテスト v2 スクリプト

縮小確率（市場確率との混合）を使用したバリュー投資バックテスト。
- 縮小確率 p' = α×p + (1-α)×q
- EV条件: p' × O >= threshold (1.05 or 1.10)
- 1レース1頭（EV最大）
- 1/4ケリー + レース上限2%
- スリッページなし（楽観的評価）
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from app.infrastructure.database import Database  # noqa: E402
from scripts.train import (  # noqa: E402
    TARGET_COL,
    LGBMClassifierWrapper,  # noqa: F401
    _split_by_race_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def load_model():
    """モデルをロード"""
    model_path = MODEL_DIR / "lgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data["model"], model_data.get("calibrator"), model_data["feature_names"]


def _join_final_odds(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """最終オッズを結合"""
    odds_query = """
    SELECT race_id, horse_id, odds_win as odds_final
    FROM core.odds_final
    WHERE odds_win IS NOT NULL
    """
    odds_df = pd.DataFrame(db.fetch_all(odds_query))

    if odds_df.empty:
        logger.warning("オッズデータがありません")
        df["odds_final"] = None
        return df

    odds_df["odds_final"] = pd.to_numeric(odds_df["odds_final"], errors="coerce")
    df = df.merge(odds_df, on=["race_id", "horse_id"], how="left")
    df = df[df["odds_final"].notna()]
    return df


def prepare_data(use_test_split: bool = False) -> pd.DataFrame:
    """データセットの読み込みと前処理"""
    logger.info("データセット準備中...")

    model, calibrator, feature_names = load_model()

    data_path = DATA_DIR / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    df = pd.read_parquet(data_path)
    df["race_date"] = pd.to_datetime(df["race_date"])

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
        df["p_win"] = calibrator.predict(raw_proba)
    else:
        df["p_win"] = raw_proba

    # 市場確率 q の計算
    df["inv_odds"] = 1 / df["odds_final"]
    df["q_market"] = df.groupby("race_id")["inv_odds"].transform(lambda x: x / x.sum())

    return df


def run_backtest(
    df: pd.DataFrame,
    alpha: float = 0.7,
    ev_threshold: float = 1.05,
    kelly_fraction: float = 0.25,
    max_bet_ratio: float = 0.02,
    initial_bankroll: int = 50000,
    fixed_bet: int = 0,  # 0以外なら固定ベット
    min_prob: float = 0.0,  # 最低勝率閾値
    max_prob: float = 1.0,  # 最高勝率閾値
) -> dict:
    """バックテスト実行"""
    # 縮小なし (alpha=1.0) を推奨
    # p_shrunk は alpha=1.0 なら p_win そのまま
    df = df.copy()
    df["p_shrunk"] = alpha * df["p_win"] + (1 - alpha) * df["q_market"]

    # EV計算（スリッページなし）
    df["ev_return"] = df["p_shrunk"] * df["odds_final"]

    # 購入条件: EV閾値 + 勝率レンジ
    is_buy = (
        (df["ev_return"] >= ev_threshold)
        & (df["p_shrunk"] >= min_prob)
        & (df["p_shrunk"] <= max_prob)
    )

    buy_candidates = df[is_buy].copy()

    if buy_candidates.empty:
        return {
            "ev_threshold": ev_threshold,
            "min_prob": min_prob,
            "max_prob": max_prob,
            "alpha": alpha,
            "n_bets": 0,
            "roi": 0.0,
            "profit": 0,
            "hit_rate": 0.0,
            "final_bankroll": initial_bankroll,
            "max_drawdown": 0,
        }

    # 1レース1頭（p_shrunk最大を選択）
    # 注: EV最大を選ぶと大穴に偏るため、p'最大を選ぶ
    bets = buy_candidates.sort_values("p_shrunk", ascending=False).drop_duplicates("race_id")
    bets = bets.sort_values(["race_date", "race_id"])

    # シミュレーション
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_drawdown = 0

    total_bet = 0
    total_payout = 0
    n_hits = 0

    for _, row in bets.iterrows():
        p = row["p_shrunk"]
        odds = row["odds_final"]

        # 賭け金計算
        if fixed_bet > 0:
            # 固定ベットモード
            bet_amount = fixed_bet
        else:
            # 1/4ケリー
            b = odds - 1
            if b <= 0:
                continue

            f_star = (p * odds - 1) / b
            if f_star <= 0:
                continue

            bet_ratio = min(kelly_fraction * f_star, max_bet_ratio)
            bet_amount = int(bankroll * bet_ratio)

            # 最低100円
            if bet_amount < 100:
                bet_amount = 100
            # 100円単位に丸め
            bet_amount = (bet_amount // 100) * 100

        if bet_amount > bankroll:
            bet_amount = int(bankroll // 100) * 100

        if bet_amount <= 0:
            continue

        total_bet += bet_amount
        bankroll -= bet_amount

        # 的中判定
        if row["finish_pos"] == 1:
            payout = int(bet_amount * odds)
            total_payout += payout
            bankroll += payout
            n_hits += 1

        # ドローダウン計算
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        drawdown = peak_bankroll - bankroll
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # 破産チェック
        if bankroll < 100:
            break

    roi = total_payout / total_bet if total_bet > 0 else 0.0
    profit = total_payout - total_bet

    return {
        "ev_threshold": ev_threshold,
        "alpha": alpha,
        "n_bets": len(bets),
        "n_actual_bets": n_hits + (len(bets) - n_hits),
        "roi": roi,
        "profit": profit,
        "hit_rate": n_hits / len(bets) if len(bets) > 0 else 0.0,
        "final_bankroll": bankroll,
        "max_drawdown": max_drawdown,
        "total_bet": total_bet,
        "total_payout": total_payout,
    }


def main():
    parser = argparse.ArgumentParser(description="バックテスト v2（縮小確率 + バリュー投資）")
    parser.add_argument("--use-test-split", action="store_true", help="Testセットのみ使用")
    parser.add_argument("--ev-threshold", type=float, default=1.05, help="EV閾値")
    parser.add_argument("--alpha", type=float, default=0.7, help="縮小係数 (0.0-1.0)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="ケリー係数")
    parser.add_argument("--max-bet-ratio", type=float, default=0.02, help="レース上限")
    parser.add_argument("--initial-bankroll", type=int, default=50000, help="初期資金")
    parser.add_argument("--min-prob", type=float, default=0.04, help="最低勝率閾値")
    parser.add_argument("--max-prob", type=float, default=0.15, help="最高勝率閾値")
    args = parser.parse_args()

    df = prepare_data(use_test_split=args.use_test_split)
    if df.empty:
        logger.error("データがありません")
        return

    logger.info(f"データ準備完了: {len(df)} 行 (レース数: {df['race_id'].nunique()})")

    # 複数の閾値でテスト
    thresholds = [1.00, 1.05, 1.10, 1.15, 1.20]

    # 固定ベット (500円) での結果
    print("\n=== 固定ベット (500円) ===")
    print(
        f"縮小係数 α = {args.alpha} (1.0=縮小なし), "
        f"勝率レンジ = {args.min_prob}-{args.max_prob}, スリッページ = なし\n"
    )
    print("| EV閾値 | ベット数 |    ROI | 利益 | 的中率 |")
    print("|--------|----------|--------|------|--------|")

    for ev_th in thresholds:
        res = run_backtest(
            df,
            alpha=args.alpha,
            ev_threshold=ev_th,
            initial_bankroll=10_000_000,
            fixed_bet=500,
            min_prob=args.min_prob,
            max_prob=args.max_prob,
        )
        status = "✓" if res["roi"] >= 1.0 else "✗"
        print(
            f"| {ev_th:.2f}   | {res['n_bets']:>8} | {res['roi']:.4f} | "
            f"{res['profit']:>+10,} | {res['hit_rate']:.4f} | {status}"
        )

    # 1/4ケリーでの結果
    print(f"\n=== 1/4ケリー (初期資金 {args.initial_bankroll:,}円) ===")
    print(
        f"縮小係数 α = {args.alpha}, 最低勝率 = {args.min_prob}, "
        f"レース上限 = {args.max_bet_ratio}\n"
    )
    print("| EV閾値 | ベット数 |    ROI | 利益 | 最終資金 | 最大DD | 的中率 |")
    print("|--------|----------|--------|------|----------|--------|--------|")

    for ev_th in thresholds:
        res = run_backtest(
            df,
            alpha=args.alpha,
            ev_threshold=ev_th,
            kelly_fraction=args.kelly_fraction,
            max_bet_ratio=args.max_bet_ratio,
            initial_bankroll=args.initial_bankroll,
            fixed_bet=0,
            min_prob=args.min_prob,
            max_prob=args.max_prob,
        )
        status = "✓" if res["roi"] >= 1.0 else "✗"
        print(
            f"| {res['ev_threshold']:.2f}   | {res['n_bets']:>8} | {res['roi']:.4f} | "
            f"{res['profit']:>+8,} | {res['final_bankroll']:>8,} | {res['max_drawdown']:>6,} | "
            f"{res['hit_rate']:.4f} | {status}"
        )


if __name__ == "__main__":
    main()
