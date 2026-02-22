"""
学習データ分析スクリプト

train.parquet の内容（ラベル分布、欠損値、外れ値、相関）を分析し、
可視化グラフと統計情報を docs/reports/figures/ に出力する。

使用方法:
    uv run python scripts/analyze_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 設定
DATA_PATH = Path("data/train.parquet")
OUTPUT_DIR = Path("docs/reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_labels(df: pd.DataFrame) -> None:
    """ラベル分布を分析する"""
    logger.info("=== Label Distribution ===")
    logger.info(f"Total samples: {len(df)}")

    if "is_win" in df.columns:
        logger.info(f"Win ratio (is_win=1): {df['is_win'].mean():.4f}")

    if "finish_pos" in df.columns:
        # 着順分布
        plt.figure(figsize=(10, 6))
        sns.histplot(df["finish_pos"], bins=18, discrete=True)
        plt.title("Finish Position Distribution")
        plt.savefig(OUTPUT_DIR / "finish_pos_dist.png")
        plt.close()
        logger.info("Saved finish_pos_dist.png")


def analyze_missing(df: pd.DataFrame) -> None:
    """欠損値を分析する"""
    logger.info("=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_info = pd.concat([missing, missing_pct], axis=1, keys=["Count", "Percent"])
    missing_info = missing_info[missing_info["Count"] > 0].sort_values("Count", ascending=False)

    if missing_info.empty:
        logger.info("No missing values found.")
    else:
        print(missing_info)

        # 欠損値の可視化
        plt.figure(figsize=(12, 6))
        missing_info["Percent"].plot(kind="bar")
        plt.title("Missing Value Percentage")
        plt.ylabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "missing_values.png")
        plt.close()
        logger.info("Saved missing_values.png")


def analyze_numeric_distributions(df: pd.DataFrame) -> None:
    """数値特徴量の分布を分析し、外れ値を検出する"""
    logger.info("=== Numeric Distributions ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # ID系やフラグを除外
    exclude_cols = [
        "race_id",
        "horse_id",
        "jockey_id",
        "trainer_id",
        "is_win",
        "finish_pos",
        "horse_no",
        "gate",
        "track_code",
        "surface",
        "going",
        "distance_bucket",
    ]
    target_cols = [c for c in numeric_cols if c not in exclude_cols]

    if target_cols:
        print(df[target_cols].describe().T)

    # 主要特徴量のヒストグラム
    key_features = [
        "speed_best_5",
        "closing_best_5",
        "early_mean_3",
        "speed_best_5_z_inrace",
        "body_weight",
    ]

    for col in key_features:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"dist_{col}.png")
            plt.close()

    # 正規化した箱ひげ図（外れ値チェック）
    valid_features = [c for c in key_features if c in df.columns]
    if valid_features:
        plt.figure(figsize=(12, 6))
        data_norm = (df[valid_features] - df[valid_features].mean()) / df[valid_features].std()
        sns.boxplot(data=data_norm)
        plt.title("Normalized Distributions (Outlier Check)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "outliers_boxplot.png")
        plt.close()
        logger.info("Saved distribution plots.")


def analyze_correlations(df: pd.DataFrame) -> None:
    """相関関係を分析する"""
    logger.info("=== Correlations ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # ID系やコードを除外
    relevant_cols = [
        c
        for c in numeric_cols
        if "id" not in c and "code" not in c and "no" not in c and "bucket" not in c
    ]

    if not relevant_cols:
        return

    corr = df[relevant_cols].corr()

    # is_win との相関に注目
    if "is_win" in corr.index:
        logger.info("\nTop 10 features correlated with is_win:")
        iso_corr = (
            corr["is_win"]
            .drop("is_win", errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )
        print(iso_corr)

        # テキストファイルに保存
        with open(OUTPUT_DIR / "top_correlations.txt", "w") as f:
            f.write(str(iso_corr))

    # ヒートマップ
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
    plt.close()
    logger.info("Saved correlation_matrix.png")


def main() -> None:
    """メイン処理"""
    if not DATA_PATH.exists():
        logger.error(f"Error: {DATA_PATH} not found.")
        return

    logger.info(f"Loading {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as e:
        logger.error(f"Error loading parquet: {e}")
        return

    analyze_labels(df)
    analyze_missing(df)
    analyze_numeric_distributions(df)
    analyze_correlations(df)

    logger.info("\nAnalysis complete.")


if __name__ == "__main__":
    main()
