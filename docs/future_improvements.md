# 今後の改善予定

## 概要

現在のMVPパイプライン完成後の改善アイデアをまとめる。

---

## 1. モデルアーキテクチャ改善

### 1.1 複勝分類モデル

**現状**: 二値分類 (`is_win` = 1着かどうか)  
**改善**: `is_place` (3着以内) を追加

```python
# 目的変数を複勝に変更
df["is_place"] = (df["finish_pos"] <= 3).astype(int)
```

**メリット**:
- 正例率が約7% → 約21%に増加
- クラス不均衡が緩和
- 複勝馬券のEV計算に直接使用可能

### 1.2 LambdaRank (ランキング学習)

**現状**: 各馬を独立に分類  
**改善**: レース内の相対順位を学習

```python
DEFAULT_LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
}
```

**メリット**:
- 「2着」と「10着」を区別できる
- レース内相対性を自然にモデル化

### 1.3 マルチタスク学習

勝利・複勝・着順を同時に学習するアプローチ。

---

## 2. データ品質改善

### 2.1 外れ値除去

```python
# 走破タイム異常値
df = df[(df["time_sec"] > 60) & (df["time_sec"] < 300)]

# オッズ異常値
df = df[df["odds_final"] < 1000]

# 体重異常値
df = df[(df["body_weight"] > 300) & (df["body_weight"] < 600)]
```

### 2.2 欠損値補完

```python
# グループ平均で補完
df["speed_mean_3"].fillna(df.groupby("class_code")["speed_mean_3"].transform("mean"))
```

### 2.3 特徴量クリッピング

```python
# percentile でクリップ
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)
```

---

## 3. 特徴量追加

### 3.1 血統特徴量

- 父馬の産駒勝率
- 母父の距離適性スコア
- BMS (母父) の芝/ダート成績

### 3.2 馬場特徴量

- 当日の馬場バイアス (内外、前残り/差し有利)
- 天候・馬場状態の変化予測

### 3.3 オッズ特徴量

- オッズの過去推移 (10分前 → 締切直前)
- 単勝支持率 vs 複勝支持率の乖離

---

## 4. ベット戦略改善

### 4.1 閾値最適化

```python
# グリッドサーチでROI最大化
for min_prob in [0.02, 0.03, 0.05, 0.10]:
    for min_ev in [0.0, 0.05, 0.10, 0.15]:
        roi = run_backtest(min_prob=min_prob, min_ev=min_ev)
```

### 4.2 ケリー基準

期待値と分散を考慮した最適ベット額計算。

### 4.3 複数馬券対応

- 複勝: `is_place` モデル
- 馬連・ワイド: 2頭の組み合わせスコア
- 三連複/三連単: 確率の積で近似

---

## 5. インフラ改善

### 5.1 ハイパーパラメータチューニング

Optuna による自動チューニング。

```python
import optuna

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        ...
    }
    return cross_val_auc(params)
```

### 5.2 GPU学習環境

WSL2 + OpenCL 設定、または クラウドGPUインスタンス。

### 5.3 自動再学習パイプライン

新規レースデータ取得 → 特徴量更新 → モデル再学習 の自動化。

---

## 優先順位

| 優先度 | 項目 | 期待効果 |
|--------|------|----------|
| 高 | 外れ値除去 | 精度安定化 |
| 高 | 閾値最適化 | ROI改善 |
| 中 | 複勝分類モデル | 新馬券種対応 |
| 中 | 血統特徴量 | 精度向上 |
| 低 | LambdaRank | アーキテクチャ改善 |
| 低 | Optuna | 自動チューニング |
