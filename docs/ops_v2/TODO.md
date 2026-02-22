# 競馬予測システム v2 TODO（バックログ）

## 運用ルール

- すべてチェックボックスで管理する
- 優先度タグ:
  - `[P0]` 今すぐ実施
  - `[P1]` 次フェーズ実施
  - `[P2]` 改善枠

## Phase 0: 基盤準備

- [x] [P0] `migrations_v2/0001_init_db.sql` を作成
- [x] [P0] `scripts_v2/migrate.py` を作成
- [x] [P0] `test_v2/test_migrate_discover.py` を追加

## Phase 1: DB構築 + 取り込み（JSONL -> DB）

### 実装
- [x] [P0] `migrations_v2/0002_expand_ingest_phase1.sql` を追加
- [x] [P0] `migrations_v2/0003_event_change_add_data_kbn.sql` を追加
- [x] [P0] `migrations_v2/0004_ensure_o3_tables.sql` を追加
- [x] [P0] `migrations_v2/0005_runner_add_code_columns.sql` を追加
- [x] [P0] `app/infrastructure/parsers.py` に `O3` パーサを追加
- [x] [P0] `scripts_v2/load_to_db.py` を実装
- [x] [P0] `RACE/DIFF/MING/0B41/0B11/0B14/0B13/0B17` ルートを実装
- [x] [P0] 中央競馬フィルタ（01-10）を取り込み時に適用

### テスト
- [x] [P0] `test_v2/test_parsers.py` を追加（O3含む）
- [x] [P0] `test_v2/test_load_to_db.py` を追加
- [x] [P0] `uv run pytest -q test_v2/test_parsers.py test_v2/test_load_to_db.py test_v2/test_migrate_discover.py`

### DB確認
- [x] [P0] `.env` を `keiba_v2` に切り替え
- [x] [P0] `keiba_v2` をクリーン再構築（schema reset + migrations `0001`〜`0005` 適用 ※`migrations_v2/`）
- [x] [P0] 実データロード（`RACE/DIFF/MING/0B41(merged)/0B11/0B14/0B13/0B17`）
- [x] [P0] 品質チェック（非中央=0 / stub race=709 / WH異常=0 / horse_no=99が12件）
- [ ] [P2] （メモ）stub race（`distance_m<=0` かつ/または `surface<=0`）が 709 件（2016以降: 354 / 2016以前: 355）。Phase 2 で除外/補完方針が必要
- [ ] [P2] （メモ）`core.o1_win.win_odds_x10` は「単勝オッズ×10（整数）」で、欠損が多い（`NULL`=106,071 / `0`=951,416）。下流では `NULL/0` を欠損扱いに統一する想定
- [ ] [P2] （メモ）`core.o3_wide.min_odds_x10` は「ワイド確定オッズ（下限）×10」で、`NULL`=29,215。JV側の `*****` マスク（発売なし/発売停止/取消等）に注意
- [ ] [P2] （メモ）`core.runner.horse_no=99`（取消/除外）=12件。馬番 1-18 前提処理は除外ロジックを入れる

## Phase 2: 特徴量生成

- [ ] [P1] `scripts_v2/build_features_v2.py` を作成
- [ ] [P1] 1行=1頭の学習行列を生成
- [ ] [P1] レース内相対特徴量（z/rank）を実装
- [ ] [P1] as-ofリーク検査を実装
- [ ] [P1] `data/features_v2.parquet` 出力仕様を固定

## Phase 3: Ranker学習

- [ ] [P1] `scripts_v2/train_ranker_v2.py` を作成
- [ ] [P1] Walk-forward + Group(`Race_ID`)分割を実装
- [ ] [P1] OOFスコアとCV指標を保存
- [ ] [P1] 2025 Hold-out封印ルールを実装

## Phase 4: Calibration

- [ ] [P1] `scripts_v2/train_calibrator_v2.py` を作成
- [ ] [P1] OOF由来特徴のみでTop3確率を学習
- [ ] [P1] Logloss/Brier/Reliabilityを出力
- [ ] [P1] 校正済みOOFを保存

## Phase 5: Wide確率 + バックテスト

- [ ] [P1] `scripts_v2/backtest_wide_v2.py` を作成
- [ ] [P1] PL + Monte Carloでwide同時確率を推定
- [ ] [P1] EVフィルタとfractional Kellyを実装
- [ ] [P1] ROI / MaxDD / 月次ROIを算出
- [ ] [P1] `data/backtest_result.json` をUI互換で出力

## Phase 6: 運用強化（T-5）

- [ ] [P2] T-5 as-ofデータ整備（速報オッズ/速報馬体重）
- [ ] [P2] オッズドロップ推定モデル実装
- [ ] [P2] リアルタイム監視と監査ログ整備
- [ ] [P2] 封印期間ワンショット評価運用の整備

## 横断タスク

- [ ] [P1] 主要スクリプトのCLIヘルプ/エラーメッセージ整備
- [ ] [P1] API/UI型定義をwide仕様へ更新
- [ ] [P1] 再現性メタデータ（seed/version/hash）を成果物へ付与
- [ ] [P2] 障害時リカバリ手順の文書化
