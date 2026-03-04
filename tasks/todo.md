# TODO: v3ワイドROIをPL出力整合にする

## 対象範囲
- `scripts_v3/backtest_wide_v3.py` 新規追加
- `test_v3/test_pl_v3_wide_invariants.py` 新規追加
- `test_v3/test_backtest_wide_v3_smoke.py` 新規追加
- `docs/ops_v3/Assumptions.md` 更新
- `docs/ops_v3/スクリプトリファレンス.md` 更新
- `docs/ops_v3/v3_run_report_2026-03-04.md` 更新

## 対象バージョン
- v3（新規実装）
- v2は参照のみ（bankroll/backtestロジック流用）

## 前提・仮定
- v3のROIデフォルト経路は `pl_score -> MC -> p_wide`
- 混在入力（`pl_score` と `p_wide` の両方あり）は `p_wide` 優先（pair-level）
- `valid_year` がない入力で `--years/--require-years` 指定時はエラー
- 旧v2近似は docs 上の参考扱い（デフォルト経路にしない）

## チェックリスト
- [x] `backtest_wide_v3.py` 実装（horse/pair両対応、入力バリデーション、meta拡張）
- [x] v3 wide推定スモークテスト追加
- [x] backtest v3 CLI/入力バリデーションスモークテスト追加
- [x] docs 3本更新（推奨=v3 / 参考=旧v2近似）
- [x] `train_pl_v3 --emit-wide-oof` 実行
- [x] `backtest_wide_v3` OOF(2023,2024) 実行
- [x] `backtest_wide_v3` holdout(2025) 実行
- [x] pytest/--help など最終確認

## 確認結果
- 追加テスト: `test_v3/test_pl_v3_wide_invariants.py`, `test_v3/test_backtest_wide_v3_smoke.py` が通過
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

## 残リスク
- `backtest_wide_v3.py` の本体挙動はDBデータ（`core.o3_wide`, `core.payout`）に依存するため、DB内容更新時にはROI値が再計測で変動する。
- 参考経路（旧v2近似）の閾値スイープ結果は過去計測値のため、同一データ更新下で再現したい場合は再実行が必要。
