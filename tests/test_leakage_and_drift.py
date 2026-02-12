"""
Feature Leakage & Drift/Shrinkage Tests

build_features.py の新しい計算ロジック（EWM + Hierarchical Shrinkage）をテストする。
実際のDB接続は行わず、Pandas DataFrame上でロジックを検証する。
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# モック用ロジック (実装予定のロジックと同じものをここで定義してテストする)
# 本番実装時はこの関数を scripts/build_features.py に移動/統合する想定
# -------------------------------------------------------------------------


def calculate_benchmark_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    span: float = 730,  # 2 years approx
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    指定グループごとの指数加重移動平均(EWM)を計算する。
    時系列順に並んでいる前提。shift(1) して「当該レース以前」の統計にする。
    """
    # 1. Calculate EWM Mean/Std
    ewm_stats = (
        df.groupby(group_cols)["time_sec"]
        .ewm(span=span, min_periods=min_periods)
        .agg(["mean", "std"])
    )

    # 2. Calculate Count (Expanding) - EWM doesn't support count in the same way
    # count is just the number of observations, weighted or not doesn't matter for "N samples"
    count_stats = df.groupby(group_cols)["time_sec"].expanding().count().rename("count")

    # Merge
    stats = pd.concat([ewm_stats, count_stats], axis=1)

    # Reset index to handle multi-index from groupby operation
    # The result has index: (group_cols..., original_index)
    # We want to sort by original_index to align with df
    stats = stats.reset_index(level=list(range(len(group_cols))), drop=True)
    stats = stats.sort_index()

    # Add group columns back to be able to group by them again for shifting
    # Since stats is aligned with df, we can just assign the values
    for col in group_cols:
        stats[col] = df[col]

    # Shift(1) to avoid leakage (未来のデータを使わない)
    # グループごとに shift する必要がある
    stats_shifted = stats.groupby(group_cols)[["mean", "std", "count"]].shift(1)

    return stats_shifted


def apply_shrinkage(
    fine_stats: pd.DataFrame, coarse_stats: pd.DataFrame, shrinkage_k: float = 10.0
) -> pd.DataFrame:
    """
    Fine統計とCoarse統計をブレンドする (Bayesian Shrinkage)。
    mu_est = (n * mu_fine + K * mu_coarse) / (n + K)
    """
    # index を揃える (呼び出し元で保証されている前提だが念のため)
    # fine_stats, coarse_stats は同じ長さで、同じ行が対応していると仮定

    n = fine_stats["count"].fillna(0)
    mu_fine = fine_stats["mean"]
    mu_coarse = coarse_stats["mean"]

    # 欠損（サンプル0）の場合は Coarse をそのまま使う
    # mu_fine が NaN の場合も考慮
    mu_fine = mu_fine.fillna(mu_coarse)

    # count が NaN の場合は 0 に

    # Shrinkage Formula
    # weight for fine: w = n / (n + K)
    w = n / (n + shrinkage_k)

    mu_est = w * mu_fine + (1 - w) * mu_coarse

    return pd.DataFrame({"mean": mu_est, "weight_fine": w})


# -------------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------------


class TestFeatureLogic:
    def test_ewm_leakage_prevention(self):
        """未来のデータが過去の推計に含まれていないか確認"""
        # シナリオ: 特定コースでレースが続く
        data = {
            "track_code": [1, 1, 1, 1],
            "surface": [1, 1, 1, 1],
            "distance_bucket": [1600, 1600, 1600, 1600],
            "going_bucket": [1, 1, 1, 1],
            "race_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "time_sec": [96.0, 95.0, 94.0, 100.0],  # 徐々に速くなる -> 最後遅い
        }
        df = pd.DataFrame(data).sort_values("race_date")

        # Coarse Stats 計算 (track, surface, distance)
        group_cols = ["track_code", "surface", "distance_bucket"]
        stats = calculate_benchmark_stats(df, group_cols, span=10, min_periods=1)

        # 1戦目 (index 0): 過去データなし -> NaN
        assert pd.isna(stats.iloc[0]["mean"])

        # 2戦目 (index 1): 1戦目のデータ (96.0) のみが反映されるはず
        assert stats.iloc[1]["mean"] == 96.0

        # 3戦目: 1戦目(96.0)と2戦目(95.0)のEWM
        # span=10なので、単純平均に近いが少し新しい方が重い
        # 95.0 < mean < 96.0
        assert 95.0 < stats.iloc[2]["mean"] < 96.0

        # 4戦目: 3戦目(94.0)まで反映。未来(100.0)は含まれない
        prev_mean = stats.iloc[2]["mean"]
        curr_mean = stats.iloc[3]["mean"]
        # 94.0 が入るので平均は下がるはず
        assert curr_mean < prev_mean

        # 4戦目の値(100.0)はどこにも使われていないこと
        # (statsはshift(1)されているため、index 3 の行には index 0-2 の集計が入る)

    def test_hierarchical_shrinkage(self):
        """サンプル不足時にCoarse統計にフォールバックするか確認"""
        # シナリオ:
        # Coarse: 芝1600m (データ多数, 平均96.0秒)
        # Fine 1: 芝1600m 良 (データ多数, 平均95.0秒) -> そのまま95.0に近くなるべき
        # Fine 2: 芝1600m 重 (データ1件, タイム100.0秒)
        # -> 信頼できないので Coarse(96.0) に引っ張られるべき

        # 簡易的なデータ生成
        stats_coarse = pd.DataFrame({"mean": [96.0, 96.0], "count": [1000, 1000]})

        stats_fine = pd.DataFrame(
            {
                "mean": [95.0, 100.0],
                "count": [1000, 1],  # 1件目は十分、2件目は不足
            }
        )

        # K=10 で縮約
        res = apply_shrinkage(stats_fine, stats_coarse, shrinkage_k=10.0)

        # Case 1: 十分なサンプル (n=1000, K=10)
        # w = 1000 / 1010 ≈ 0.99
        # result ≈ 0.99*95 + 0.01*96 ≈ 95.01
        assert 0.98 < res.iloc[0]["weight_fine"] < 1.0
        assert 95.0 <= res.iloc[0]["mean"] < 95.1

        # Case 2: サンプル不足 (n=1, K=10)
        # w = 1 / 11 ≈ 0.09
        # result = 0.09*100 + 0.91*96 = 9.09 + 87.36 = 96.45
        # 100秒(Fine)よりも96秒(Coarse)に近い
        assert 0.08 < res.iloc[1]["weight_fine"] < 0.10
        assert 96.0 < res.iloc[1]["mean"] < 97.0

    def test_missing_fine_data(self):
        """Fineデータが全くない(初登場の条件)場合"""
        stats_coarse = pd.DataFrame({"mean": [96.0], "count": [100]})
        stats_fine = pd.DataFrame({"mean": [np.nan], "count": [0]})

        res = apply_shrinkage(stats_fine, stats_coarse, shrinkage_k=10.0)

        # w = 0 / 10 = 0
        # result = 0*NaN + 1*96.0 = 96.0
        assert res.iloc[0]["mean"] == 96.0
        assert res.iloc[0]["weight_fine"] == 0.0
