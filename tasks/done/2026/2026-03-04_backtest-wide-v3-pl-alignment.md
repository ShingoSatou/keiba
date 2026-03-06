# 2026-03-04_backtest-wide-v3-pl-alignment

## タイトル

- v3ワイドROIをPL出力整合にする

## ステータス

- `done`

## 対象範囲

- `scripts_v3/backtest_wide_v3.py` 新規追加
- `test_v3/test_pl_v3_wide_invariants.py` 新規追加
- `test_v3/test_backtest_wide_v3_smoke.py` 新規追加
- `docs/ops_v3/Assumptions.md` 更新
- `docs/ops_v3/スクリプトリファレンス.md` 更新
- `docs/ops_v3/v3_run_report_2026-03-04.md` 更新

## 対象バージョン

- `v3`
- `v2` は参照のみ

## 前提・仮定

- v3 の ROI default 経路は `pl_score -> MC -> p_wide`
- 混在入力では `p_wide` 優先
- `valid_year` がない入力で `--years/--require-years` 指定時はエラー
- 旧 v2 近似は docs 上の参考扱い

## チェックリスト

- [x] `backtest_wide_v3.py` 実装
- [x] wide 推定スモークテスト追加
- [x] backtest CLI / 入力バリデーションスモークテスト追加
- [x] docs 更新
- [x] `train_pl_v3 --emit-wide-oof` 実行
- [x] `backtest_wide_v3` OOF(2023,2024) 実行
- [x] `backtest_wide_v3` holdout(2025) 実行
- [x] pytest / `--help` / ruff 確認

## 確認結果

- `test_v3/test_pl_v3_wide_invariants.py`, `test_v3/test_backtest_wide_v3_smoke.py` が通過
- `backtest_wide_v3.py --help` が正常動作
- `train_pl_v3 --emit-wide-oof` 実行で `data/oof/pl_v3_wide_oof_w2.parquet` を生成
- 推奨経路の再計測結果:
  - OOF(2023,2024): ROI `0.0948` / bets `1513`
  - holdout(2025): ROI `0.0617` / bets `1427`

## 実行コマンド

- `uv run pytest test_v3/test_pl_v3_wide_invariants.py test_v3/test_backtest_wide_v3_smoke.py`
- `uv run python scripts_v3/backtest_wide_v3.py --help`
- `uv run python scripts_v3/train_pl_v3.py --train-window-years 2 --oof-output data/oof/pl_v3_oof_w2.parquet --wide-oof-output data/oof/pl_v3_wide_oof_w2.parquet --emit-wide-oof --metrics-output data/oof/pl_v3_cv_metrics_w2.json --model-output models/pl_v3_recent_window_w2.joblib`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_wide_oof_w2.parquet --years 2023,2024 --holdout-year 2025 --output data/backtest_v3/backtest_wide_v3_direct_w2_2023_2024.json --meta-output data/backtest_v3/backtest_wide_v3_direct_w2_2023_2024_meta.json --force`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_direct_holdout_2025.json --meta-output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_meta.json --force`
- `uv run ruff check scripts_v3/backtest_wide_v3.py test_v3/test_pl_v3_wide_invariants.py test_v3/test_backtest_wide_v3_smoke.py`

## 関連ドキュメント

- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-04.md`

## 残リスク

- `backtest_wide_v3.py` の挙動は DB データに依存するため、ROI は再計測で変動しうる
- 参考経路の閾値スイープ結果は過去計測値であり、同一データ更新下で再現したい場合は再実行が必要

## 移行メモ

- 2026-03-06 に旧 monolithic `tasks/todo.md` から分割移行
