# Keiba Prediction

JRA-VANデータを用いた競馬予測モデル (LightGBM)

## 環境構築

```bash
uv sync
```

## 実行フロー

### 1. データベース構築
```bash
bash setup_postgres.sh
```

### 2. データ取得・ロード
```bash
# JV-Linkからデータを取得してDBにロード
uv run python scripts/extract_jvlink.py
```

### 3. 特徴量生成
```bash
# 特徴量を計算してDB (martスキーマ) に保存
uv run python scripts/build_features.py
```

### 4. 学習データセット作成
```bash
# DBからデータを取得し、学習用ファイル (data/train.parquet) を生成
uv run python scripts/build_dataset.py
```

### 5. モデル学習
```bash
# 通常モード (Train 70% / ES-Val 10% / Test 20%)
uv run python scripts/train.py

# 本番学習モード (Train 90% / ES-Val 10% / Testなし)
# 未来の予測に使用するモデルを作成する場合
uv run python scripts/train.py --production
```

## 分析・評価ツール

### バックテスト (標準)
学習に使用していないTestデータ（時系列で最後の20%）を使用してモデル性能を評価します。

```bash
# データリークを防ぐため --use-test-split 推奨
uv run python scripts/backtest.py --use-test-split
```

### 閾値最適化グリッドサーチ
ROIを最大化する「最低勝率」と「最低EV」の閾値を探索します。

```bash
# Testデータのみ使用して探索
uv run python scripts/optimize_thresholds.py --use-test-split
```

### ケリー基準シミュレーション
資金管理戦略（ケリー基準）を用いた長期シミュレーションを行います。

```bash
# 初期資金5万円、スリッページ15% (デフォルト)
uv run python scripts/backtest_kelly.py --use-test-split --initial-bankroll 50000
```

### [実験的] バリュー投資バックテスト (v2)
市場オッズとの乖離（EV）を重視し、確率縮小（Shrinkage）や勝率レンジ指定を行う実験的なスクリプトです。

```bash
# 縮小なし(alpha=1.0)、勝率4%〜15%に限定してテスト
uv run python scripts/backtest_v2.py --use-test-split --alpha 1.0 --min-prob 0.04 --max-prob 0.15
```

- `--alpha`: 縮小係数 (1.0=縮小なし, 0.0=市場オッズのみ)
- `--min-prob` / `--max-prob`: 勝率レンジフィルタ
- `--ev-threshold`: 期待値閾値 (例: 1.05)

## ドキュメント

- [Features](docs/data_design.md)
- [System Spec](docs/system_specification.md)
- [Reports](docs/reports/)