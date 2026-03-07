# 2026-03-07_v2-o1-place-timeseries-load

## タイトル

- v2 O1 複勝時系列オッズの DB 取り込み拡張

## ステータス

- `done`

## 対象範囲

- `app/infrastructure/parsers.py`
- `scripts_v2/load_to_db.py`
- `scripts_v2/backfill_o1_place_from_raw_v2.py`
- `migrations_v2/`
- `test_v2/`
- `docs/ops_v2/`

## 対象バージョン

- `v2`

## 前提・仮定

- 既存 `raw.jv_raw` の `0B41` / `RACE` `O1` payload には複勝ブロックが含まれる。
- 履歴実行は既存 raw の再利用を優先し、追加 JV-Link 取得は行わない。
- 対象期間は既存 `core.o1_header` のカバレッジに合わせて `2016-01-05` から `2026-02-15` とする。

## チェックリスト

- [x] O1 parser に複勝項目を追加する
- [x] O1 header / place detail の migration を追加する
- [x] `load_to_db.py` で複勝も upsert する
- [x] `raw.jv_raw` から複勝を再正規化する backfill script を追加する
- [x] parser / loader / backfill のテストを追加する
- [x] migration 適用、backfill 実行、DB 確認を行う

## 確認結果

- 着手前確認: 既存 DB には `core.o1_header` / `core.o1_win` が 2016-01-05〜2026-02-15 で投入済み。
- `uv run pytest -s test_v2/test_parsers.py test_v2/test_load_to_db.py test_v2/test_backfill_o1_place_from_raw_v2.py` は `33 passed`。
- `uv run python scripts_v2/migrate.py` で `0006_o1_place_timeseries.sql` を適用した。
- `uv run python scripts_v2/backfill_o1_place_from_raw_v2.py --from-date 20160105 --to-date 20260215 --dataspecs 0B41,RACE --batch-size 5000 --upsert-batch-size 50000` を完走した。
- backfill 実績: `raw_rows=5,396,481`, `matched_snapshots=5,396,481`, `place_rows_written=75,485,950`（upsert 試行件数）, `skipped_non_central=0`, `skipped_out_of_range=0`。
- DB 実測: `core.o1_place=75,485,925 行`, `core.o1_win=75,485,925 行`, `core.o1_place` の日付範囲は `2016-01-05`〜`2026-02-15`。
- `core.o1_place` の `data_kbn` 分布は `1/2/3/4/5/9` で、主要区分 `1/3/4/5` が存在する。
- `min_odds_x10 > max_odds_x10` かつ両方正値の行は `0` 件だった。
- サンプル `race_id=201601050601`, `announce_mmddhhmi=01041831`, `data_kbn=1` は raw payload 再パース結果と `core.o1_header` / `core.o1_place` が一致した。

## 実行コマンド

- `git status --short`
- `uv run pytest -s test_v2/test_parsers.py test_v2/test_load_to_db.py test_v2/test_backfill_o1_place_from_raw_v2.py`
- `uv run python scripts_v2/migrate.py`
- `uv run python scripts_v2/backfill_o1_place_from_raw_v2.py --from-date 20160105 --to-date 20260215 --dataspecs 0B41,RACE --batch-size 5000 --upsert-batch-size 50000`
- `uv run python - <<'PY' ...`

## 関連ドキュメント

- `docs/JV-link/JV-Data仕様書_4.9.0.1.docling.md`
- `docs/specs_v2/競馬予測システム 課題設定・要件定義書.md`
- `docs/ops_v2/スクリプトリファレンス.md`

## 残リスク

- `core.o1_place` を新設したため、DB 使用量は大きく増える。
- 今回の全量実行は既存 `raw.jv_raw` を再利用したもので、JV-Link からの再取得フロー自体は別途未検証。
