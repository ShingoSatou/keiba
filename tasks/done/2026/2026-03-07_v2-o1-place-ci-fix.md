# 2026-03-07_v2-o1-place-ci-fix

## タイトル

- v2 O1 複勝時系列 PR の CI format failure 修正

## ステータス

- `done`

## 対象範囲

- `app/infrastructure/parsers.py`
- `tasks/*`

## 対象バージョン

- `v2`

## 前提・仮定

- 対象 PR は `https://github.com/ShingoSatou/keiba/pull/36`。
- 失敗チェックは GitHub Actions `CI / test` の `Format (check)` のみ。

## チェックリスト

- [x] failing check とログを確認する
- [x] ローカルで format failure を再現する
- [x] 必要最小限の修正を入れて CI 相当チェックを通す
- [x] task を閉じて PR branch に push する

## 確認結果

- `gh run view 22788532256 --log` で失敗を確認した。
- 失敗箇所は `bash scripts/check_ci.sh fmt-check` で、`Would reformat: app/infrastructure/parsers.py`。
- ローカルでも `bash scripts/check_ci.sh fmt-check` で同じ failure を再現した。
- `uv run ruff format app/infrastructure/parsers.py` 実行後、`bash scripts/check_ci.sh fmt-check` は通過した。
- `bash scripts/check_ci.sh lint` は通過した。
- `uv run pytest -s test_v2/test_parsers.py test_v2/test_load_to_db.py test_v2/test_backfill_o1_place_from_raw_v2.py test_v2/test_migrate_discover.py` は `36 passed`。

## 実行コマンド

- `gh pr view 36 --json number,title,url,headRefName,baseRefName,state,statusCheckRollup`
- `gh run view 22788532256 --log`
- `bash scripts/check_ci.sh fmt-check`
- `uv run ruff format app/infrastructure/parsers.py`
- `bash scripts/check_ci.sh fmt-check`
- `bash scripts/check_ci.sh lint`
- `uv run pytest -s test_v2/test_parsers.py test_v2/test_load_to_db.py test_v2/test_backfill_o1_place_from_raw_v2.py test_v2/test_migrate_discover.py`

## 関連ドキュメント

- `tasks/done/2026/2026-03-07_v2-o1-place-pr.md`

## 残リスク

- format 以外の新規失敗が後続チェックで出る可能性は未確認。
