## 目的 / 背景
- v2（ワイド / Listwise / NDCG@3）実装を進めるため、Phase 1（DB構築 + JSONL投入）を実運用可能な形にする
- v1/v2混在状態のまま誤参照しないよう、v2ディレクトリ命名とドキュメントを整理する

## 変更内容
- v2ディレクトリを明確化
  - `migrations` -> `migrations_v2`
  - `scripts` -> `scripts_v2`
  - `tests` -> `test_v2`
- Phase 1実装を追加
  - `migrations_v2/0001`〜`0005` を追加
  - `scripts_v2/migrate.py` / `scripts_v2/load_to_db.py` を追加
  - `app/infrastructure/parsers.py` に O3（ワイド確定オッズ）パーサを追加
  - WH（馬体重）パース定義を仕様に合わせて修正
- テスト追加
  - `test_v2/test_parsers.py`
  - `test_v2/test_load_to_db.py`
  - `test_v2/test_migrate_discover.py`
- 設定/ドキュメント更新
  - `pyproject.toml` の pytest/ruff 対象を v2 構成へ更新
  - `README.md` を v2 PoC 前提に刷新
  - `docs/ops_v2/実装計画.md` を進捗分離 + フェーズ詳細化
  - `docs/ops_v2/TODO.md` に品質メモ追記
  - `docs/ops_v2/スクリプトリファレンス.md` を新規作成
- 不要スクリプト整理
  - `setup_postgres.sh` を削除（`setup_postgres_multi.sh` に一本化）

## 影響範囲
- v2 の DBマイグレーション/投入手順（`scripts_v2/*`）
- v2 テスト実行パス（`test_v2`）
- README と運用ドキュメント
- 既存 v1 実装（`scripts_v1`, `tests_v1`, `migrations_v1`）には変更なし

## 動作確認（実施済み）
- [x] `uv run ruff format .`
- [x] `uv run ruff check .`
- [x] `uv run pytest -q`

## リスク / 注意点
- `core.race` の `class_code/条件コード` は Phase 2 で特徴量抽出条件に使うため、必要に応じて RA パース拡張が必要
- 品質メモ（stub race/オッズ欠損/取消horse_no=99）は `docs/ops_v2/TODO.md` を参照
