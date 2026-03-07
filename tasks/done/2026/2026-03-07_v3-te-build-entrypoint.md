# 2026-03-07_v3-te-build-entrypoint

## タイトル

- v3 TE 生成入口の固定化

## ステータス

- `done`

## 対象範囲

- `scripts_v3/build_features_v3_te.py`
- `scripts_v3/feature_registry_v3.py`
- `test_v3/*`
- `docs/specs_v3/*`
- `docs/ops_v3/*`

## 対象バージョン

- `v3`

## 前提・仮定

- current `features_v3_te*.parquet` は `features_v3*` に safe TE extra 列を結合した派生物として扱う
- safe TE extra 列の定義は `feature_registry_v3.py` の binary contract に合わせる

## チェックリスト

- [x] 既存 TE 生成前提を調査する
- [x] 専用 build 入口を実装する
- [x] テストと CLI smoke を更新する
- [x] docs / ops を更新する

## 確認結果

- `scripts_v3/build_features_v3_te.py` を追加し、current `features_v3*` に safe TE extra 列だけを strict merge する fixed entrypoint を実装した
- safe TE extra 列は `feature_registry_v3.py` の `get_binary_safe_te_feature_columns()` を source of truth とした
- join key は `race_id`, `horse_id`, `horse_no` に固定し、base / source どちらかで duplicate や unmatched rows があれば fail-fast する
- real data で次を再生成できた
  - `data/features_v3_te.parquet`
  - `data/features_v3_te_2025.parquet`
- safe TE extra 列は現状 3 列で固定された
  - `jockey_target_label_mean_6m`
  - `trainer_target_label_mean_6m`
  - `rel_jockey_target_label_mean_z`
- `pytest` 22 件、`ruff check` が通った

## 実行コマンド

- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3_te.py --base-input data/features_v3.parquet --te-source-input data/features_v2_te.parquet --output data/features_v3_te.parquet --meta-output data/features_v3_te_meta.json`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3_te.py --base-input data/features_v3_2025.parquet --te-source-input data/features_v2_te_2025.parquet --output data/features_v3_te_2025.parquet --meta-output data/features_v3_te_2025_meta.json`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -s test_v3/test_build_features_v3_te.py test_v3/test_binary_feature_contract_v3.py test_v3/test_v3_cli_smoke.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts_v3/build_features_v3_te.py test_v3/test_build_features_v3_te.py test_v3/test_v3_cli_smoke.py`

## 関連ドキュメント

- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_05_共通基盤と付録.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- TE extra 列の source は現状 `features_v2_te*.parquet` 依存であり、raw から TE を再計算する入口は別途必要
