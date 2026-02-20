"""
バックテスト結果出力サービス

DBから実データを引き、バックテストを実行し、
Web UI が読み込める JSON（summary / monthly / bets）を出力する。

使用方法:
    uv run python -m app.services.backtest_runner --output data/backtest_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from app.infrastructure.database import Database

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ターゲットカラム名（train.py と同じ）
TARGET_COL = "is_win"


# =============================================================================
# pickle ロード用: LGBMClassifierWrapper (train.py で保存したモデルの復元に必要)
# =============================================================================


class LGBMClassifierWrapper:
    """LightGBM を scikit-learn 風にラップ (pickle ロード互換用)"""

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


def _load_model():
    """モデルと校正器をロード"""
    model_path = MODEL_DIR / "lgb_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)  # noqa: S301

    return model_data["model"], model_data.get("calibrator"), model_data["feature_names"]


# =============================================================================
# データ準備
# =============================================================================


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


def _join_horse_info(db: Database, df: pd.DataFrame) -> pd.DataFrame:
    """馬名をDB JOINで取得（horse_no は train.parquet に既に含まれる）"""
    # 馬名を core.horse から取得
    horse_query = """
    SELECT horse_id, horse_name
    FROM core.horse
    WHERE horse_name IS NOT NULL
    """
    horse_df = pd.DataFrame(db.fetch_all(horse_query))

    if not horse_df.empty:
        df = df.merge(horse_df, on="horse_id", how="left")
    else:
        df["horse_name"] = "不明"

    # horse_no が DataFrame に存在しない場合のみ DB から取得
    if "horse_no" not in df.columns:
        runner_query = """
        SELECT race_id, horse_id, horse_no
        FROM core.runner
        """
        runner_df = pd.DataFrame(db.fetch_all(runner_query))

        if not runner_df.empty:
            df = df.merge(runner_df, on=["race_id", "horse_id"], how="left")
        else:
            df["horse_no"] = 0

    return df


def _split_test_by_race_id(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """race_id 境界で時系列分割し、Testセットのみ返す"""
    if "race_id" not in df.columns or "race_date" not in df.columns:
        raise ValueError("race_id と race_date が必要です")

    unique_races = df[["race_id", "race_date"]].drop_duplicates().sort_values("race_date")
    n_races = len(unique_races)
    n_test = int(n_races * test_size)
    es_val_end = n_races - n_test

    test_races = set(unique_races.iloc[es_val_end:]["race_id"])
    return df[df["race_id"].isin(test_races)].copy()


def prepare_data(db: Database, *, use_test_split: bool = True) -> pd.DataFrame:
    """データセットの読み込みと前処理"""
    logger.info("データセット準備中...")

    model, calibrator, feature_names = _load_model()

    data_path = DATA_DIR / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    df = pd.read_parquet(data_path)
    df["race_date"] = pd.to_datetime(df["race_date"])

    # Testセットのみ使用（学習データのリーク防止）
    if use_test_split and TARGET_COL in df.columns:
        df = _split_test_by_race_id(df, test_size=0.2)
        logger.info(f"Testセット抽出完了: {len(df)} 件")

    if df.empty:
        return pd.DataFrame()

    # オッズ結合
    df = _join_final_odds(db, df)

    if df.empty:
        return pd.DataFrame()

    # 馬名・馬番を結合
    df = _join_horse_info(db, df)

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

    logger.info(f"データ準備完了: {len(df)} 行 (レース数: {df['race_id'].nunique()})")
    return df


# =============================================================================
# バックテスト実行（UI出力用）
# =============================================================================


def run_backtest_for_ui(
    df: pd.DataFrame,
    *,
    alpha: float = 0.7,
    ev_threshold: float = 1.05,
    fixed_bet: int = 500,
    min_prob: float = 0.04,
    max_prob: float = 0.15,
) -> dict:
    """
    バックテスト実行し、UI用フォーマットの辞書を返す。

    Returns:
        {"summary": {...}, "monthly": [...], "bets": [...]}
    """
    df = df.copy()

    # 縮小確率
    df["p_shrunk"] = alpha * df["p_win"] + (1 - alpha) * df["q_market"]

    # EV計算
    df["ev_return"] = df["p_shrunk"] * df["odds_final"]

    # 購入条件
    is_buy = (
        (df["ev_return"] >= ev_threshold)
        & (df["p_shrunk"] >= min_prob)
        & (df["p_shrunk"] <= max_prob)
    )

    buy_candidates = df[is_buy].copy()

    if buy_candidates.empty:
        return _empty_result()

    # 1レース1頭（p_shrunk最大を選択）
    bets = buy_candidates.sort_values("p_shrunk", ascending=False).drop_duplicates("race_id")
    bets = bets.sort_values(["race_date", "race_id"])

    # --- 個別ベット情報を構築 ---
    bet_records = []
    total_bet = 0
    total_payout = 0
    n_hits = 0
    cumsum = 0
    peak = 0
    max_drawdown = 0

    for _, row in bets.iterrows():
        is_hit = row.get("finish_pos") == 1
        payout = int(fixed_bet * row["odds_final"]) if is_hit else 0
        profit = payout - fixed_bet

        total_bet += fixed_bet
        total_payout += payout
        if is_hit:
            n_hits += 1

        # ドローダウン
        cumsum += profit
        if cumsum > peak:
            peak = cumsum
        dd = peak - cumsum
        if dd > max_drawdown:
            max_drawdown = dd

        # ev_profit = p_shrunk * odds - 1 （UIの期待する値）
        ev_profit = row["p_shrunk"] * row["odds_final"] - 1.0

        # 馬名のフォールバック
        horse_name = row.get("horse_name", "不明")
        if pd.isna(horse_name) or horse_name == "":
            horse_name = "不明"

        horse_no = row.get("horse_no", 0)
        if pd.isna(horse_no):
            horse_no = 0

        bet_records.append(
            {
                "race_date": str(row["race_date"].date()),
                "race_id": int(row["race_id"]),
                "horse_name": str(horse_name).strip(),
                "horse_no": int(horse_no),
                "p_win": round(float(row["p_shrunk"]), 4),
                "odds_final": round(float(row["odds_final"]), 1),
                "ev_profit": round(float(ev_profit), 4),
                "is_hit": bool(is_hit),
                "payout": payout,
                "profit": profit,
            }
        )

    # --- 月別集計 ---
    monthly_data = _aggregate_monthly(bet_records, fixed_bet)

    # --- 予測品質指標（全データ対象） ---
    logloss_val = None
    auc_val = None
    y_true = df.get("is_win") if "is_win" in df.columns else df.get("finish_pos")
    if y_true is not None and "p_win" in df.columns:
        if "is_win" in df.columns:
            y_binary = df["is_win"]
        else:
            y_binary = (df["finish_pos"] == 1).astype(int)
        try:
            logloss_val = round(log_loss(y_binary, df["p_win"]), 4)
            auc_val = round(roc_auc_score(y_binary, df["p_win"]), 4)
        except Exception:
            pass

    # --- サマリ ---
    roi = total_payout / total_bet if total_bet > 0 else 0.0
    hit_rate = n_hits / len(bet_records) if bet_records else 0.0

    period_dates = [r["race_date"] for r in bet_records]
    summary = {
        "period_from": min(period_dates) if period_dates else "",
        "period_to": max(period_dates) if period_dates else "",
        "n_races": int(df["race_id"].nunique()),
        "n_bets": len(bet_records),
        "n_hits": n_hits,
        "hit_rate": round(hit_rate, 4),
        "total_bet": total_bet,
        "total_return": total_payout,
        "roi": round(roi, 4),
        "max_drawdown": max_drawdown,
        "logloss": logloss_val,
        "auc": auc_val,
    }

    return {
        "summary": summary,
        "monthly": monthly_data,
        "bets": bet_records,
    }


def _empty_result() -> dict:
    """ベットがない場合の空結果"""
    return {
        "summary": {
            "period_from": "",
            "period_to": "",
            "n_races": 0,
            "n_bets": 0,
            "n_hits": 0,
            "hit_rate": 0.0,
            "total_bet": 0,
            "total_return": 0,
            "roi": 0.0,
            "max_drawdown": 0,
            "logloss": None,
            "auc": None,
        },
        "monthly": [],
        "bets": [],
    }


def _aggregate_monthly(bet_records: list[dict], bet_amount: int) -> list[dict]:
    """ベットレコードから月別集計を生成"""
    monthly: dict[str, dict] = defaultdict(lambda: {"n_bets": 0, "n_hits": 0, "total_payout": 0})

    for bet in bet_records:
        # "2023-01-08" → "2023-01"
        month_key = bet["race_date"][:7]
        monthly[month_key]["n_bets"] += 1
        if bet["is_hit"]:
            monthly[month_key]["n_hits"] += 1
        monthly[month_key]["total_payout"] += bet["payout"]

    result = []
    for month_key in sorted(monthly.keys()):
        m = monthly[month_key]
        total_bet_month = m["n_bets"] * bet_amount
        roi = m["total_payout"] / total_bet_month if total_bet_month > 0 else 0.0
        result.append(
            {
                "month": month_key,
                "n_bets": m["n_bets"],
                "n_hits": m["n_hits"],
                "roi": round(roi, 4),
            }
        )

    return result


# =============================================================================
# JSON 出力
# =============================================================================


def export_json(result: dict, output_path: Path) -> None:
    """バックテスト結果をJSONファイルに書き出す"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON出力完了: {output_path}")


# =============================================================================
# CLI エントリーポイント
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="バックテスト実行 → UI用JSON出力")
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest_result.json",
        help="出力先JSONパス (デフォルト: data/backtest_result.json)",
    )
    parser.add_argument("--alpha", type=float, default=0.7, help="縮小係数 (0.0-1.0)")
    parser.add_argument("--ev-threshold", type=float, default=1.05, help="EV閾値")
    parser.add_argument("--fixed-bet", type=int, default=500, help="固定ベット額")
    parser.add_argument("--min-prob", type=float, default=0.04, help="最低勝率閾値")
    parser.add_argument("--max-prob", type=float, default=0.15, help="最高勝率閾値")
    parser.add_argument(
        "--no-test-split",
        action="store_true",
        help="全データでバックテスト（Testセット分割しない）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    with Database() as db:
        df = prepare_data(db, use_test_split=not args.no_test_split)

    if df.empty:
        logger.error("データがありません。バックテストを中止します。")
        sys.exit(1)

    result = run_backtest_for_ui(
        df,
        alpha=args.alpha,
        ev_threshold=args.ev_threshold,
        fixed_bet=args.fixed_bet,
        min_prob=args.min_prob,
        max_prob=args.max_prob,
    )

    logger.info(
        f"バックテスト完了: {result['summary']['n_bets']} ベット, "
        f"ROI={result['summary']['roi']:.4f}"
    )

    export_json(result, output_path)


if __name__ == "__main__":
    main()
