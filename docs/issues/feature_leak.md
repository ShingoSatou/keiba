# 特徴量リーク問題 (Feature Leak)

## 発見日
2026-02-07

## 概要

集計特徴量（騎手勝率、トレーナー勝率など）が **全期間で一括計算** されており、
将来のデータが過去のレコードに混入している可能性がある。

---

## 対象となる特徴量

### 集計系 (リークリスクあり)

| 特徴量 | 計算方法 | リスク |
|--------|----------|--------|
| `jockey_win_rate_1y` | 過去1年の騎手勝率 | 計算時点によっては未来混入 |
| `jockey_place_rate_1y` | 過去1年の騎手複勝率 | 同上 |
| `trainer_win_rate_1y` | 過去1年の調教師勝率 | 同上 |
| `trainer_place_rate_1y` | 過去1年の調教師複勝率 | 同上 |

### 過去走系 (リスク低)

- `speed_mean_3`, `closing_mean_3` など
- これらは各馬の過去レース結果を参照するため、正しく実装されていればリスクは低い
- ただし、**同一レース内の相対特徴量** (z_inrace, rank) の計算ロジックは要確認

---

## 問題のパターン

### パターン1: 全期間での一括計算

```python
# NG: 2024年のデータが2023年のレコードに影響
df["jockey_win_rate"] = df.groupby("jockey_id")["is_win"].transform("mean")
```

### パターン2: Rolling での未来混入

```python
# NG: sort されていない、または未来方向を含む
df["speed_mean_3"] = df.groupby("horse_id")["speed"].rolling(3).mean()
```

---

## 修正方針

### 案A: fit/transform 分離 (推奨)

```python
class JockeyStatEncoder:
    def fit(self, df_train):
        """学習データのみで統計を計算"""
        self.stats_ = df_train.groupby("jockey_id")["is_win"].mean()
        
    def transform(self, df):
        """学習済み統計をマップ"""
        return df["jockey_id"].map(self.stats_).fillna(self.stats_.mean())
```

### 案B: 累積統計 (時系列順)

```python
# 各行の時点までの累積で計算
df = df.sort_values("race_date")
df["jockey_win_rate"] = (
    df.groupby("jockey_id")["is_win"]
    .expanding()
    .mean()
    .shift(1)  # 当該レースを除外
    .values
)
```

---

## 確認ポイント

- [ ] `build_features.py` の集計ロジックを確認
- [ ] 時系列順でのみ計算されているか
- [ ] shift(1) で当該レースを除外しているか
- [ ] Train/Val/Test 分割後に fit/transform しているか

---

## 関連ファイル

- `scripts/build_features.py`: 特徴量生成
- `scripts/train.py`: 学習パイプライン

---

## 解決策 (2026-02-07 実装)

**EWM (Exponential Weighted Moving Average) と Hierarchical Shrinkage** を組み合わせた `ExponentialTargetEncoder` を実装しました。

1. **時系列順の処理**:
   - データを `race_date` でソートし、過去のデータのみから統計を計算。
   - `shift(1)` 相当の処理により、当該レースの情報が混入することを防止。

2. **ベイジアン平滑化 (Hierarchical Shrinkage)**:
   - データ数が少ないカテゴリ（新人騎手など）の統計値が暴れるのを防ぐため、全体の平均値（事前分布）との加重平均を使用。
   - $ \hat{\mu}_i = \lambda \mu_i + (1 - \lambda) \mu_{global} $
   - 信頼度 $\lambda$ はサンプル数（重み付き）に基づいて動的に決定。

## ステータス

- [x] 現状の計算ロジック確認
- [x] fit/transform 分離の実装 (EWM+Shrinkageによる動的計算へ変更)
- [x] 修正後の指標再計測 (テストによる検証完了)
- [x] **解決済み**

