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

### 6. バックテスト
```bash
# 学習に使っていないTestデータのみで評価 (データリーク防止)
uv run python scripts/backtest.py --use-test-split
```

## ドキュメント

- [Features](docs/data_design.md)
- [System Spec](docs/system_specification.md)