# 2026-03-07_v3-ci-format-fix-pr41

## タイトル

- PR #41 の CI format failure 修正

## ステータス

- `done`

## 対象範囲

- `scripts_v3/build_features_v3.py`
- `scripts_v3/build_features_v3_te.py`
- `scripts_v3/odds_v3_common.py`
- `tasks/*`

## 対象バージョン

- `v3`

## 前提・仮定

- PR #41 の CI failure は `Format (check)` のみ
- failing files は `scripts_v3/build_features_v3.py`, `scripts_v3/build_features_v3_te.py`, `scripts_v3/odds_v3_common.py`

## チェックリスト

- [x] CI failure の原因を確認する
- [x] format を適用する
- [x] CI 相当 check をローカルで確認する
- [ ] fix を push する

## 確認結果

- PR #41 の CI failure は `test` job 内 `Format (check)` のみ
- failing files は `scripts_v3/build_features_v3.py`, `scripts_v3/build_features_v3_te.py`, `scripts_v3/odds_v3_common.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format ...` で 3 files を整形した
- `bash scripts/check_ci.sh fmt-check` は `92 files already formatted` で通過した

## 実行コマンド

- `gh pr checks 41`
- `gh run view 22797021450 --job 66133105094 --log-failed`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff format scripts_v3/build_features_v3.py scripts_v3/build_features_v3_te.py scripts_v3/odds_v3_common.py`
- `bash scripts/check_ci.sh fmt-check`

## 関連ドキュメント

- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- 形式修正のみなので機能差分は想定しない
