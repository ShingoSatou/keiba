# Keiba Prediction

JRA-VANデータを用いた競馬予測モデル (LightGBM)

## 非開発者向け（運用者）クイックスタート

当日（開催日）の **T-5 推論**までを最短で回す手順です。詳細は `docs/ops/運用手順.md` を参照してください。

```bash
# 1) 依存導入
uv sync

# 2) DB作成 + migrations
bash setup_postgres_multi.sh keiba_cp
uv run python scripts/migrate.py

# 3) 当日運用（取得→投入→スナップ→推論→出力）
uv run python scripts/ops_t5_dryrun.py --race-date 20260214 --fetch-rt --fail-on-empty
```

前提:
- `.env` に `DATABASE_URL` を設定
- JV-Link 取得は Windows + JV-Link + 32bit Python が必要（WSL からは `ops_*` が Windows 側を呼び出します）
- `models/t5_bundle.pkl` が必要（なければ学習が先）

## 環境構築

```bash
uv sync

# W&B を使う場合のみ追加
uv sync --extra wandb
```

## 実行フロー

### 1. データベース構築
```bash
bash setup_postgres_multi.sh keiba_cp

# DDL / migrations 適用（新規DBはこれでスキーマ作成まで完了）
uv run python scripts/migrate.py
```

#### Migrations 運用

このリポジトリのDBスキーマは `migrations/*.sql` を正とし、`scripts/migrate.py` で適用します。

```bash
# 状態確認（適用済み/未適用）
uv run python scripts/migrate.py --list

# 新規DB: 未適用の migration を順に実行して適用
uv run python scripts/migrate.py

# 既存DB（すでに手動でDDL適用済み）:
# SQLは実行せず、適用済みとして記録だけ開始（baseline）
uv run python scripts/migrate.py --baseline

# 途中まで適用したい場合（ファイル名で指定、含む）
uv run python scripts/migrate.py --to 0002_init_mart.sql
```

運用ルール（最低限）:
- 適用済みの migration（すでに本番/共有DBで実行済みの `.sql`）は編集しない（**追加のみ**）。
- 適用順は「ファイル名の昇順」（例: `0001_...` → `0002_...` → `YYYYMMDD_...`）。
- 適用履歴は `public.schema_migrations` に記録されます。

### 2. データ取得・ロード
```bash
# JSONL → DB投入（64bit Python）
uv run python scripts/load_to_db.py --input-dir data/
```

> 注: `scripts/extract_jvlink.py` / `scripts/extract_rt_jvlink.py` は **JV-Link (Windows COM) のため Windows + 32bit Python 必須**です。  
> 取得（JV-Link→JSONL）は Windows 側で実行し、DB投入は WSL/Linux（`uv`）で実行します。  
> 当日運用は `scripts/ops_t5_dryrun.py` / `scripts/ops_rt.py` が WSL→Windows32bit をオーケストレーションします。

### data/ のSQL取り込み対象

- **SQLに取り込む**: `RACE/DIFF/MING/SNPN/SLOP/WOOD/0B41/0B11/0B14/0B13/0B17` の各 `*.jsonl`（`scripts/load_to_db.py`）
- **いまは取り込まない（未整備）**: `COMM/YSCH`（取得はできるが core化/特徴量利用が未整備）
- **取り込まない**: `train*.parquet` / `data/ops/<date>/*`（学習/運用の成果物）

表（取り込み先テーブル含む）は `docs/ops/運用手順.md` にまとめています。

### 3. 特徴量生成
```bash
# 特徴量を計算してDB (martスキーマ) に保存
# デフォルト: 障害レース(surface=3)は除外（含める場合は --include-obstacles）
uv run python scripts/build_features.py --rebuild --from-date 2016-01-01
```

### 4. 学習データセット作成
```bash
# DBからデータを取得し、学習用ファイル (data/train.parquet) を生成
# デフォルト: 障害レース(surface=3)は除外（含める場合は --include-obstacles）
uv run python scripts/build_dataset_t5.py --from-date 2016-01-01 --min-horses 5 --output data/train.parquet

# 品質ゲート（閾値違反時は exit 1）
uv run python scripts/check_dataset_quality.py --input data/train.parquet --gate --json-output data/quality/train_latest.json
```

### 5. モデル学習
```bash
# 通常モード (Walk-Forward CV で評価)
uv run python scripts/train_walk_forward.py

# W&B を使わない場合（依存未導入でも実行可能）
uv run python scripts/train_walk_forward.py --no-wandb

# 本番モデル (Bundle) セーブ用
# 未来の予測に使用するデプロイモデルを作成する場合
uv run python scripts/train_production_bundle.py
```

## 分析・評価ツール

### 本番運用バックテスト
学習に使用していないデータを用いて、実際の運用で「過去買っていたらどう推移したか」をT-5スナップとCV予測値を用いてシミュレートします。

```bash
uv run python scripts/backtest_v2.py
```

### 閾値最適化
バックテスト出力をもとに、EV閾値等のベターなパラメータを探します。

```bash
uv run python scripts/optimize_thresholds.py
```

## ドキュメント

- [スクリプトリファレンス](docs/ops/スクリプトリファレンス.md)
- [運用手順（非開発者向け）](docs/ops/運用手順.md)
- [運用仕様（T-5の採用ルール等）](docs/specs/50_運用仕様書.md)
- [データ取得仕様（JV-Link/JVRTOpen）](docs/specs/10_データ取得仕様書.md)
- [ダウンロードリスト（推奨dataspec）](docs/specs/11_ダウンロードリスト.md)
- [進捗状況（現状の制約/次にやること）](docs/ops/進捗状況.md)
- [DB設計仕様書（参考）](docs/specs/21_DB設計仕様書.md)
