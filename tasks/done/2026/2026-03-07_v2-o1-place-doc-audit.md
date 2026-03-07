# 2026-03-07_v2-o1-place-doc-audit

## タイトル

- v2 O1 複勝時系列対応のドキュメント反映確認

## ステータス

- `done`

## 対象範囲

- `docs/ops_v2/実装計画.md`
- `docs/ops_v2/TODO.md`
- `tasks/*`

## 対象バージョン

- `v2`

## 前提・仮定

- 実装・migration・DB backfill は完了済みで、今回は docs の追従漏れ確認に限定する。
- スクリプトリファレンスは前タスクで更新済みなので、未反映箇所があれば最小限だけ補う。

## チェックリスト

- [x] v2 docs に残る O1 複勝未反映箇所を洗い出す
- [x] 必要な docs のみ更新する
- [x] 変更内容を確認して task ledger を更新する

## 確認結果

- `docs/ops_v2/実装計画.md` の Phase 1 現状整理が `0005` / `parsers/loader/migrate` 止まりで、今回の `0006` と backfill 反映が漏れていた。
- `docs/ops_v2/TODO.md` は Phase 1 実装・テスト・DB確認に今回の O1 複勝対応実績が未記載だった。
- `docs/ops_v2/スクリプトリファレンス.md` は `load_to_db.py` の `core.o1_place` 展開と `backfill_o1_place_from_raw_v2.py` が既に反映済みで、追加更新は不要だった。
- `core.o1_win` と `core.o1_place` の欠損件数メモを DB 実測で確認し、`docs/ops_v2/TODO.md` に追記した。

## 実行コマンド

- `rg -n "0B41|O1|o1_place|o1_win|o1_header|複勝時系列|時系列オッズ" docs/ops_v2 docs/specs_v2`
- `uv run python - <<'PY' ...`
- `git diff -- docs/ops_v2/実装計画.md docs/ops_v2/TODO.md tasks/active/2026-03-07_v2-o1-place-doc-audit.md tasks/todo.md`

## 関連ドキュメント

- `docs/ops_v2/実装計画.md`
- `docs/ops_v2/TODO.md`
- `docs/ops_v2/スクリプトリファレンス.md`

## 残リスク

- v2 docs は運用メモ中心のため、将来別の O1 下流利用を追加した際は再度 docs を点検する必要がある。
