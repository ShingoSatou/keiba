# 2026-03-06_v3-fixed-4y-retrain-eval

## タイトル

- v3 4年固定条件での再学習・検証・評価

## ステータス

- `done`

## 対象範囲

- `data/features_v3.parquet`
- `data/features_v3_2025.parquet`
- `scripts_v3/train_win_{lgbm,xgb,cat}_v3.py`
- `scripts_v3/train_place_{lgbm,xgb,cat}_v3.py`
- `scripts_v3/train_odds_calibrator_v3.py`
- `scripts_v3/train_pl_v3.py`
- `scripts_v3/predict_race_v3.py` 相当の holdout 一括推論
- `scripts_v3/backtest_wide_v3.py`
- `docs/ops_v3/v3_run_report_2026-03-06_feature_governance_retrain.md`

## 対象バージョン

- `v3`

## 前提・仮定

- 標準比較条件は `train_window_years=4` / `cv_window_policy=fixed_sliding`
- 入力特徴量 `data/features_v3.parquet` と `data/features_v3_2025.parquet` は既存成果物をそのまま利用する
- backtest は DB 接続が有効である前提で実行する

## チェックリスト

- [x] 実行前状態を確認
- [x] binary 6本を再学習
- [x] odds calibration / PL を再学習
- [x] holdout 2025 の PL 予測を生成
- [x] OOF / holdout の backtest を実行
- [x] 指標を集計して report を更新

## 確認結果

- binary 6本、odds calibration、PL 学習、OOF / holdout backtest まで完了
- binary は `valid_year=2020-2024` の 5 fold、PL は OOF 制約込みで `valid_year=2024` の 1 fold
- holdout 2025 の PL 予測を `data/oof/pl_v3_holdout_2025_pred.parquet` として再生成
- backtest は OOF fixed4 で ROI `0.0000`、holdout 2025 fixed4 で ROI `0.1863`
- `cv_policy` が metrics / meta / backtest meta に保存されることを確認

## 実行コマンド

- `uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_odds_calibrator_v3.py`
- `uv run python scripts_v3/train_pl_v3.py --emit-wide-oof`
- `uv run python - <<'PY' ...  # predict_race_v3 helper chain で data/features_v3_2025.parquet -> data/oof/pl_v3_holdout_2025_pred.parquet を生成`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_wide_oof.parquet --holdout-year 2025 --output data/backtest_v3/backtest_wide_v3_direct_oof_fixed4.json --meta-output data/backtest_v3/backtest_wide_v3_direct_oof_fixed4_meta.json --force`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_fixed4.json --meta-output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_fixed4_meta.json --force`
- `uv run python - <<'PY' ...  # metrics / json / parquet を集計`

## 関連ドキュメント

- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-06_feature_governance_retrain.md`

## 残リスク

- holdout PL 一括推論は専用 CLI がないため、既存モジュールを使った one-shot 実行になる
