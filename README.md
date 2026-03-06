# Keiba Prediction

JRA-VANデータを用いた競馬予測（**v2: ワイド / Listwise / NDCG@3** の PoC）

## v2（Wide PoC）クイックスタート

前提:
- `.env` に `DATABASE_URL` を設定（例: `postgresql://jv_ingest:...@127.0.0.1:5432/keiba_v2`）
- PostgreSQL が動いていること（ローカル構築の補助として `setup_postgres_multi.sh` を利用可能）

```bash
# 1) 依存導入
uv sync

# 2) DB作成（必要な場合のみ）
bash setup_postgres_multi.sh keiba_v2

# 3) migrations 適用（既定dir=migrations_v2）
uv run python scripts_v2/migrate.py

# 4) JSONL → DB投入（v2）
uv run python scripts_v2/load_to_db.py --input-dir data/
```

## v2ドキュメント

- 入口（毎回参照）: `docs/specs_v2/競馬予測システム 課題設定・要件定義書.md`
- 実装計画（全体）: `docs/ops_v2/実装計画.md`
- TODO（進捗管理）: `docs/ops_v2/TODO.md`
- スクリプトリファレンス: `docs/ops_v2/スクリプトリファレンス.md`

## v3（追加）

- 仕様書インデックス: `docs/specs_v3/v3_システム仕様書.md`
- 全体構成 / データフロー: `docs/specs_v3/v3_01_全体アーキテクチャ.md`
- 特徴量生成 / odds: `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- binary / 校正: `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- PL / 推論 / ROI: `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- 共通基盤 / テスト / CLI: `docs/specs_v3/v3_05_共通基盤と付録.md`
- 運用手順: `docs/ops_v3/スクリプトリファレンス.md`
- 推定前提: `docs/ops_v3/Assumptions.md`
- 実行結果: `docs/ops_v3/v3_run_report_*.md`
- default PL contract は `meta_default`（`p_win_meta`, `p_place_meta`, `p_win_odds_t10_norm`, `PL_CONTEXT_FEATURES_SMALL`）
- v3 の実行順と canonical CLI は `docs/ops_v3/スクリプトリファレンス.md` を source of truth とする

## v1について（参照専用）

v1（旧）は参照用として残しています（保守・編集しない）。

- v1: `docs/*_v1/`, `scripts_v1/`, `tests_v1/`, `migrations_v1/`
- v2: `docs/specs_v2/`, `docs/ops_v2/`, `scripts_v2/`, `test_v2/`, `migrations_v2/`
