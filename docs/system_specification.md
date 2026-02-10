# 競馬予測システム仕様書

## 1. システム概要

### 1.1 目的
JRA-VAN Data Lab のデータを用いた機械学習ベースの競馬予測システム。
単勝馬券の期待値 (EV) を計算し、プラス期待値の馬を推奨する。

### 1.2 アーキテクチャ概要
```
┌─────────────────────────────────────────────────────────────────┐
│                     JRA-VAN Data Lab                            │
│                    (Windows側で取得)                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │ TCP/IP
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PostgreSQL (WSL2)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ core.race    │  │ core.runner  │  │ core.result  │          │
│  │ core.odds_*  │  │ core.horse   │  │ core.jockey  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 特徴量生成パイプライン                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ time_stats   │→ │ run_index    │→ │ horse_stats  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                       ┌──────────────┐          │
│                                       │ person_stats │          │
│                                       └──────────────┘          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    学習・推論パイプライン                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ train.py     │→ │ lgb_model    │→ │ backtest.py  │          │
│  │              │  │ calibrator   │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 技術スタック
| カテゴリ | 技術 | バージョン |
|----------|------|------------|
| 言語 | Python | 3.11 |
| パッケージ管理 | uv | - |
| Web Framework | FastAPI | - |
| DB | PostgreSQL | 15+ |
| ML | LightGBM | 4.x |
| データ処理 | pandas, numpy | - |
| 実験管理 | Weights & Biases | - |

---

## 2. ディレクトリ構成

```
keiba/
├── app/                          # FastAPI アプリケーション
│   ├── main.py                   # エントリーポイント
│   ├── services/
│   │   └── ev_service.py         # EV計算サービス
│   └── infrastructure/
│       └── database.py           # PostgreSQL接続管理
│
├── scripts/                      # バッチスクリプト
│   ├── build_features.py         # 特徴量生成 (Step 1-4)
│   ├── build_dataset.py          # 学習データセット生成
│   ├── train.py                  # モデル学習
│   └── backtest.py               # バックテスト
│
├── models/                       # 学習済みモデル
│   ├── lgb_model.pkl             # LightGBMモデル
│   └── calibrator.pkl            # 確率校正器 (IsotonicRegression)
│
├── data/
│   └── train.parquet             # 学習データセット
│
├── docs/                         # ドキュメント
│   ├── system_specification.md   # 本ドキュメント
│   ├── future_improvements.md    # 今後の改善予定
│   └── data_design.md            # データ設計
│
├── tests/                        # テスト
│   ├── test_ev_service.py        # EV計算テスト
│   └── test_features.py          # 特徴量計算テスト
│
├── sql/
│   └── init_mart.sql             # mart スキーマ初期化
│
└── .env                          # 環境変数 (W&B API Key等)
```

---

## 3. データベース設計

### 3.1 スキーマ構成
- **core**: JRA-VAN から取得した生データ
- **mart**: 特徴量計算後の集計データ

### 3.2 core スキーマ (主要テーブル)

#### core.race
| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | VARCHAR(16) PK | レースID (YYYYPPDDRRNN形式) |
| race_date | DATE | 開催日 |
| track_code | INT | 競馬場コード (1-10: 中央競馬) |
| surface | INT | 馬場 (1:芝, 2:ダート) |
| distance_m | INT | 距離 (メートル) |
| going | INT | 馬場状態 (1:良, 2:稍重, 3:重, 4:不良) |
| class_code | INT | クラスコード |
| field_size | INT | 出走頭数 |

#### core.runner
| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | VARCHAR(16) PK | レースID |
| horse_id | VARCHAR(10) PK | 馬ID |
| horse_no | INT | 馬番 |
| gate | INT | 枠番 |
| jockey_id | VARCHAR(5) | 騎手ID |
| trainer_id | VARCHAR(5) | 調教師ID |
| carried_weight | DECIMAL(4,1) | 斤量 |
| body_weight | INT | 馬体重 |
| body_weight_diff | INT | 馬体重増減 |
| scratch_flag | BOOLEAN | 取消フラグ |

#### core.result
| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | VARCHAR(16) PK | レースID |
| horse_id | VARCHAR(10) PK | 馬ID |
| finish_pos | INT | 着順 |
| time_sec | DECIMAL(5,1) | 走破タイム (秒) |
| final3f_sec | DECIMAL(4,1) | 上がり3F (秒) |
| corner_pos | VARCHAR(20) | コーナー通過順 |

#### core.odds_final
| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | VARCHAR(16) PK | レースID |
| horse_id | VARCHAR(10) PK | 馬ID |
| odds_win | DECIMAL(6,1) | 単勝オッズ (確定) |

### 3.3 mart スキーマ

#### mart.time_stats
コース×距離×馬場状態ごとの基準タイム。

| カラム | 型 | 説明 |
|--------|-----|------|
| track_code | INT PK | 競馬場コード |
| surface | INT PK | 馬場 |
| distance_m | INT PK | 距離 |
| going | INT PK | 馬場状態 |
| time_median | DECIMAL(5,1) | 走破タイム中央値 |
| time_iqr | DECIMAL(5,2) | 四分位範囲 |
| final3f_median | DECIMAL(4,1) | 上がり3F中央値 |
| sample_count | INT | サンプル数 |

#### mart.run_index
1走ごとの走破指数。

| カラム | 型 | 説明 |
|--------|-----|------|
| race_id | VARCHAR(16) PK | レースID |
| horse_id | VARCHAR(10) PK | 馬ID |
| speed_index | DECIMAL(5,1) | スピード指数 |
| closing_index | DECIMAL(5,1) | 末脚指数 |
| early_index | DECIMAL(5,1) | 先行指数 |
| position_gain | INT | 位置取り変化 |

#### mart.horse_stats
馬の近走成績集計。

| カラム | 型 | 説明 |
|--------|-----|------|
| calc_date | DATE PK | 計算基準日 |
| horse_id | VARCHAR(10) PK | 馬ID |
| target_surface | INT PK | 対象馬場 |
| target_distance_bucket | INT PK | 対象距離バケット |
| target_going_bucket | INT PK | 対象馬場状態バケット |
| speed_last | DECIMAL | 直近のスピード指数 |
| speed_mean_3 | DECIMAL | 直近3走の平均 |
| speed_best_5 | DECIMAL | 直近5走のベスト |
| speed_std_5 | DECIMAL | 直近5走の標準偏差 |
| speed_trend_3 | DECIMAL | 直近3走のトレンド (傾き) |
| ... | ... | (他の指数も同様) |
| n_runs_5 | INT | 直近5走の出走数 |
| n_sim_runs_5 | INT | 条件近似走の出走数 |

#### mart.person_stats
騎手・調教師の成績。

| カラム | 型 | 説明 |
|--------|-----|------|
| calc_date | DATE PK | 計算基準日 |
| person_type | VARCHAR(10) PK | 'jockey' or 'trainer' |
| person_id | VARCHAR(5) PK | ID |
| win_rate_1y | DECIMAL(5,4) | 過去1年勝率 |
| place_rate_1y | DECIMAL(5,4) | 過去1年複勝率 |
| sample_count_1y | INT | サンプル数 |

---

## 4. 特徴量生成パイプライン

### 4.1 Step 2: run_index (走破指数計算)

**目的**: 1走ごとのパフォーマンスを数値化。

**入力**: `core.result`, `core.race` (全期間)

**計算手法 (Feature Leakage修正版)**:
1. **基準タイム計算**:
   - 全レースを時系列順に処理。
   - **EWM (Exponential Weighted Moving Average)** を用いて過去のレースタイムから基準タイムを動的に生成 (`span=730`日)。
   - **Hierarchical Shrinkage**: サンプル不足時は、細かい条件 (`track, surface, distance, going`) から広域条件 (`track, surface, distance`) へ自動的にフォールバックして数値を安定させる。
2. **標準化**:
   - `speed_index = (benchmark_mean - time_sec) / benchmark_std`

**出力**: `mart.run_index`

---

### 4.3 Step 3: horse_stats (馬の近走成績)

**目的**: 各馬の能力を直近の成績から推定。

**入力**: `mart.run_index`, `core.runner`, `core.race`

**集計項目**:
| 項目 | 説明 | 窓 |
|------|------|-----|
| `*_last` | 直近1走の値 | 1 |
| `*_mean_3` | 直近3走の平均 | 3 |
| `*_best_5` | 直近5走の最高値 | 5 |
| `*_std_5` | 直近5走の標準偏差 | 5 |
| `*_trend_3` | 直近3走の線形トレンド | 3 |

**条件近似走**:
同じ馬場 (芝/ダート) × 類似距離 × 類似馬場状態の過去走のみを使用。

```python
def is_similar_condition(target_surface, target_distance_bucket, target_going_bucket,
                          past_surface, past_distance_m, past_going):
    if target_surface != past_surface:
        return False
    if distance_to_bucket(past_distance_m) != target_distance_bucket:
        return False
    if going_to_bucket(past_going) != target_going_bucket:
        return False
    return True
```

**出力**: `mart.horse_stats`

---

### 4.4 Step 4: person_stats (騎手・調教師成績)

**目的**: 騎手・調教師の能力指標を計算。

**入力**: `core.runner`, `core.result`, `core.race`

**集計**:
```sql
SELECT
    jockey_id,
    COUNT(*) as total,
    SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) / COUNT(*) as win_rate_1y,
    SUM(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) / COUNT(*) as place_rate_1y
FROM core.runner run
JOIN core.result res ...
WHERE race_date >= calc_date - INTERVAL '1 year'
GROUP BY jockey_id
```

**出力**: `mart.person_stats`

---

## 5. モデル仕様

### 5.1 アルゴリズム
**LightGBM (Light Gradient Boosting Machine)**

- 勾配ブースティング決定木
- カテゴリ変数の効率的な処理
- 高速な学習・推論

### 5.2 目的関数
```python
objective = "binary"  # 二値分類
metric = "binary_logloss"  # 対数損失
```

**ターゲット変数**: `is_win` (1着なら1、それ以外0)

### 5.3 ハイパーパラメータ
```python
DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,          # 葉ノード数 (複雑さ)
    "learning_rate": 0.05,     # 学習率
    "feature_fraction": 0.8,   # 使用特徴量の割合
    "bagging_fraction": 0.8,   # 使用サンプルの割合
    "bagging_freq": 5,         # バギング頻度
    "verbose": -1,
    "seed": 42,
}

num_boost_round = 1000  # 最大反復回数
```

### 5.4 Early Stopping
過学習を防ぐため、validation loss が50ラウンド改善しなければ停止。

```python
lgb.early_stopping(stopping_rounds=50, verbose=True)
```

### 5.5 データ分割
時系列分割を使用。未来のデータでテストすることで現実的な評価を行う。

```python
# 日付順にソート → 前半80%を学習、後半20%をテスト
df_sorted = df.sort_values("race_date")
split_idx = int(len(df_sorted) * 0.8)
train_idx = df_sorted.index[:split_idx]
test_idx = df_sorted.index[split_idx:]
```

### 5.6 確率校正
LightGBM の出力確率は実際の勝率とズレることがあるため、
`IsotonicRegression` で校正。

```python
from sklearn.isotonic import IsotonicRegression

# 生確率を実際の勝率に合わせる
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(y_pred_raw, y_test)

# 推論時
calibrated_prob = calibrator.predict(raw_prob)
```

---

## 6. 特徴量一覧 (44次元)

### 6.1 メタ情報 (4)
| 特徴量 | 説明 | 型 |
|--------|------|-----|
| `race_id` | レースID | カテゴリ |
| `race_date` | 開催日 | 日付 |
| `horse_id` | 馬ID | カテゴリ |
| `horse_no` | 馬番 | 数値 |

### 6.2 レース条件 (6)
| 特徴量 | 説明 | 型 |
|--------|------|-----|
| `track_code` | 競馬場 (1-10) | カテゴリ |
| `surface` | 芝(1)/ダート(2) | カテゴリ |
| `distance_m` | 距離 (m) | 数値 |
| `going` | 馬場状態 (1-4) | カテゴリ |
| `class_code` | クラス | カテゴリ |
| `field_size` | 出走頭数 | 数値 |

### 6.3 当日コンディション (6)
| 特徴量 | 説明 | 型 |
|--------|------|-----|
| `gate` | 枠番 | 数値 |
| `carried_weight` | 斤量 (kg) | 数値 |
| `body_weight` | 馬体重 (kg) | 数値 |
| `body_weight_diff` | 馬体重増減 | 数値 |
| `days_since_last` | 前走からの日数 | 数値 |
| `distance_change_m` | 距離変化 (m) | 数値 |

### 6.4 馬の能力 - 全条件版 (14)
| 特徴量 | 説明 |
|--------|------|
| `speed_last` | 直近1走のスピード指数 |
| `speed_mean_3` | 直近3走のスピード指数平均 |
| `speed_best_5` | 直近5走のスピード指数最高値 |
| `speed_std_5` | 直近5走のスピード指数標準偏差 |
| `speed_trend_3` | 直近3走のスピード指数トレンド |
| `closing_last` | 直近1走の末脚指数 |
| `closing_mean_3` | 直近3走の末脚指数平均 |
| `closing_best_5` | 直近5走の末脚指数最高値 |
| `early_mean_3` | 直近3走の先行指数平均 |
| `early_best_5` | 直近5走の先行指数最高値 |
| `position_gain_mean_3` | 直近3走の位置取り変化平均 |
| `finish_mean_3` | 直近3走の平均着順 |
| `finish_best_5` | 直近5走のベスト着順 |
| `n_runs_5` | 直近5走の出走数 (0=新馬) |

### 6.5 馬の適性 - 条件近似版 (6)
| 特徴量 | 説明 |
|--------|------|
| `speed_sim_mean_3` | 条件近似走3走のスピード指数平均 |
| `speed_sim_best_5` | 条件近似走5走のスピード指数最高値 |
| `closing_sim_mean_3` | 条件近似走3走の末脚指数平均 |
| `closing_sim_best_5` | 条件近似走5走の末脚指数最高値 |
| `early_sim_mean_3` | 条件近似走3走の先行指数平均 |
| `n_sim_runs_5` | 条件近似走の出走数 |

### 6.6 レース内相対特徴量 (6)
| 特徴量 | 説明 |
|--------|------|
| `speed_best_5_z_inrace` | レース内でのスピード指数z-score |
| `closing_best_5_z_inrace` | レース内での末脚指数z-score |
| `early_mean_3_z_inrace` | レース内での先行指数z-score |
| `speed_best_5_rank` | レース内でのスピード指数順位 |
| `closing_best_5_rank` | レース内での末脚指数順位 |
| `early_mean_3_rank` | レース内での先行指数順位 |

### 6.7 ペース圧特徴量 (2)
| 特徴量 | 説明 |
|--------|------|
| `pace_front_runner_cnt` | レース内の先行候補馬数 |
| `num_faster_early` | 自分より先行力が高い馬の数 |

### 6.8 人の実績 (4)
| 特徴量 | 説明 |
|--------|------|
| `jockey_win_rate_1y` | 騎手の過去1年勝率 |
| `jockey_place_rate_1y` | 騎手の過去1年複勝率 |
| `trainer_win_rate_1y` | 調教師の過去1年勝率 |
| `trainer_place_rate_1y` | 調教師の過去1年複勝率 |

---

## 7. ベット戦略

### 7.1 期待値 (EV) 計算

```python
# 有効オッズ = 確定オッズ × (1 - スリッページ率)
odds_effective = odds_final * (1 - slippage)

# 期待値利益 = 勝率 × 有効オッズ - 1
# EV > 0 なら購入価値あり
ev_profit = p_win * odds_effective - 1
```

### 7.2 購入条件
```python
is_buy = (ev_profit > 0) AND (p_win > min_prob)
```

| 条件 | 理由 |
|------|------|
| `ev_profit > 0` | 期待値がプラスであること |
| `p_win > min_prob` | 極端な大穴を除外 (モデル誤差対策) |

### 7.3 馬券選択
```python
# 各レースでEV最大の1頭を購入
bet_df = buy_df.groupby("race_id")["ev_profit"].idxmax()
```

### 7.4 パラメータ設定
| パラメータ | デフォルト値 | 説明 |
|------------|--------------|------|
| `slippage` | 0.15 (15%) | オッズ低下リスク。購入によるオッズ変動を保守的に見積もる |
| `min_prob` | 0.03 (3%) | 最低勝率閾値。これ以下は購入しない |
| `bet_amount` | 500円 | 1レースあたりの購入額 (均等買い) |

### 7.5 ケリー基準 (未実装)
最適ベット額を計算する手法。将来実装予定。

```python
kelly_fraction = (p_win * odds - 1) / (odds - 1)
bet_amount = bankroll * kelly_fraction * fraction_multiplier
```

---

## 8. 評価指標

### 8.1 モデル性能指標

#### AUC (Area Under ROC Curve)
- **意味**: 識別能力。ランダムな勝ち馬と負け馬のペアで、勝ち馬に高い確率を与える割合
- **現在値**: 0.7703
- **解釈**: 0.5=ランダム、0.7=普通、0.8+=優秀

#### Logloss (対数損失)
- **意味**: 予測確率の正確さ。低いほど良い
- **現在値**: 0.2265 (生) → 0.2256 (校正後)
- **ベースライン**: 勝率7%で常に0.07と予測 = 約0.265

#### Calibration (校正)
- **意味**: 予測確率が実際の勝率と一致しているか
- **確認方法**: 予測確率0.10の馬が実際に10%勝っているか
- **現状**: ほぼ完璧に校正済み

### 8.2 バックテスト指標

#### ROI (Return On Investment)
- **計算**: 総回収額 / 総投資額
- **現在値**: 74.6%
- **解釈**: 100%で±0、74.6%は25.4%の損失

#### 的中率
- **計算**: 的中数 / 購入レース数
- **現在値**: 2.0% (70/3550)

#### 最大ドローダウン
- **意味**: 累積収支が最高点から最大何円下落したか
- **現在値**: 593,750円

---

## 9. 実行方法

### 9.1 環境準備
```bash
# 仮想環境作成
uv venv
source .venv/bin/activate

# 依存関係インストール
uv sync
```

### 9.2 特徴量生成
```bash
# 全ステップ実行
PYTHONPATH=. uv run python scripts/build_features.py

# 特定ステップのみ
PYTHONPATH=. uv run python scripts/build_features.py --step 3

# 特定日のみ
PYTHONPATH=. uv run python scripts/build_features.py --date 2025-02-01

# 再構築 (既存データ削除)
PYTHONPATH=. uv run python scripts/build_features.py --rebuild
```

### 9.3 データセット生成
```bash
# 2020年以降のデータでデータセット作成
PYTHONPATH=. uv run python scripts/build_dataset.py --from-date 2020-01-01

# 期間指定
PYTHONPATH=. uv run python scripts/build_dataset.py --from-date 2020-01-01 --to-date 2024-12-31
```

### 9.4 モデル学習
```bash
# 通常学習
PYTHONPATH=. uv run python scripts/train.py

# ラウンド数指定
PYTHONPATH=. uv run python scripts/train.py --num-boost-round 2000

# W&B無効
PYTHONPATH=. uv run python scripts/train.py --no-wandb

# GPU使用 (要OpenCL)
PYTHONPATH=. uv run python scripts/train.py --gpu
```

### 9.5 バックテスト
```bash
# 2025年以降でバックテスト
PYTHONPATH=. uv run python scripts/backtest.py --from-date 2025-01-01

# パラメータ調整
PYTHONPATH=. uv run python scripts/backtest.py \
    --from-date 2025-01-01 \
    --slippage 0.10 \
    --min-prob 0.05 \
    --bet-amount 1000
```

---

## 10. W&B 連携

### 10.1 セットアップ
```bash
# .env に設定
WANDB_API_KEY=your_api_key
WANDB_PROJECT=keiba-prediction
```

### 10.2 記録内容
- データ統計 (サンプル数、特徴量数、正例率)
- 学習曲線 (train/valid の logloss)
- テスト指標 (AUC, Logloss)
- 校正曲線
- 特徴量重要度
- モデル artifact

### 10.3 ダッシュボード
https://wandb.ai/uu13234-none/keiba-prediction

---

## 11. 既知の制限事項と課題

### 11.1 評価データのリーク (Critical)
現在の `train.py` には、以下の評価データリークが存在する疑いがある：

1. **Validation Set の再利用**
   - Early Stopping 用の `X_val` と、最終評価用の `X_test` が同一である。
   - これにより、モデルがテストデータに過適合している可能性がある。

2. **Calibration の過学習**
   - 確率校正器 (`IsotonicRegression`) の学習に `X_test` の予測値を使用し、**同じ `X_test` で評価** している。
   - これにより、校正プロットや Logloss が実力以上に良く見えている（"完璧すぎる校正"）。

**影響**:
- `train.py` が出力する AUC (0.77) と Calibration 結果は信頼できない（過大評価）。
- ただし、`backtest.py` は完全に未来のデータ (2025年以降) を使用しているため、バックテスト結果 (ROI 74.6%) はリークの影響を受けていない（ただし、モデル選択バイアスの影響はある）。

**今後の対応方針**:
- データ分割を `Train` / `Validation` (Early Stopping) / `Calibration` / `Test` (Final Eval) の4分割に変更する。
- または、Cross-Validation を導入して Out-of-Fold 予測値で校正を行う。

### 11.2 その他の制限
1. **単勝のみ対応**: 複勝・馬連・三連複は未対応
2. **中央競馬のみ**: 地方競馬は除外
3. **リアルタイム推論なし**: バッチ処理のみ
4. **血統特徴量なし**: 父馬・母父の成績は未使用
5. **馬場バイアスなし**: 当日の馬場傾向は未反映

---

## 12. 今後の改善予定

詳細は [docs/future_improvements.md](./future_improvements.md) を参照。

主な項目:
- 複勝分類モデル追加
- LambdaRank (ランキング学習)
- 外れ値除去・欠損値補完
- 血統特徴量
- ベット閾値最適化
- Optuna ハイパーパラメータチューニング
