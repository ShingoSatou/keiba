# 2026-03-07_v3-feature-contract-retrain-eval

## タイトル

- v3 特徴量再生成と stacker / PL 再学習・評価

## ステータス

- `done`

## 対象範囲

- `data/features_v3.parquet`
- `data/features_v3_meta.json`
- `data/features_v3_2025.parquet`
- `data/features_v3_2025_meta.json`
- `data/oof/win_stack_oof.parquet`
- `data/oof/place_stack_oof.parquet`
- `data/oof/pl_v3_oof.parquet`
- `data/oof/pl_v3_holdout_2025_pred.parquet`
- `data/oof/pl_v3_cv_metrics.json`
- `data/oof/v3_pipeline_year_coverage.json`
- `data/holdout/win_stack_holdout_pred_v3.parquet`
- `data/holdout/place_stack_holdout_pred_v3.parquet`
- `models/win_stack_bundle_meta_v3.json`
- `models/place_stack_bundle_meta_v3.json`
- `models/pl_v3_recent_window.joblib`
- `models/pl_v3_all_years.joblib`
- `models/pl_v3_bundle_meta.json`
- `data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor.json`
- `docs/ops_v3/v3_run_report_2026-03-07_feature_contract_refactor_retrain.md`

## 対象バージョン

- `v3`

## 前提・仮定

- 今回はユーザー依頼どおり、既存 binary OOF / holdout を前提に stacker / PL を再学習した。
- binary 自体の再学習は別スコープとし、今回の評価では現行 binary 予測を固定入力として扱った。
- holdout は `2025`、stacker は strict temporal `capped_expanding`、PL は `fixed_sliding(train_window_years=3)` を維持した。

## チェックリスト

- [x] 学習 / holdout 用 features_v3 を再生成する
- [x] win/place stacker を再学習する
- [x] PL `stack_default` を再学習する
- [x] holdout / year coverage / backtest を確認する
- [x] 結果と残リスクをまとめる

## 確認結果

- `features_v3`, `features_v3_2025` を再生成し、新市場派生列の coverage は train / holdout とも `notna_rate=1.0`。
- `win_stack` の CV summary は `logloss=0.211415`, `auc=0.814222`。
- `place_stack` の CV summary は `logloss=0.419472`, `auc=0.796010`。
- `PL stack_default` holdout 2025 は `top3_logloss=0.435280`, `top3_auc=0.773852`。
- holdout wide backtest は `roi=0.1314`。
- `predict_race_v3.py` は、base 予測をマージした 1 レース入力 + `--skip-base-model-inference` で stacker / PL path を通過確認した。
- full default path の smoke input は current binary artifact が TE 列を要求するため未通過だった。

## 実行コマンド

- `set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3.py --input data/features_base.parquet --output data/features_v3.parquet --meta-output data/features_v3_meta.json`
- `set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3.py --input data/features_v2_2025.parquet --output data/features_v3_2025.parquet --meta-output data/features_v3_2025_meta.json`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_win_stack_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --holdout-year 2025`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_place_stack_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --holdout-year 2025`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_pl_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --pl-feature-profile stack_default --train-window-years 3 --holdout-year 2025 --emit-wide-oof`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor.json --meta-output data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor_meta.json --force`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ...  # 1レース holdout 入力 + base holdout pred merge で smoke parquet 作成`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/predict_race_v3.py --input data/inference/race_features_smoke_2025_with_base_preds.parquet --pl-model models/pl_v3_recent_window.joblib --skip-base-model-inference --output data/predictions/race_v3_pred_feature_contract_refactor.parquet --emit-wide --wide-output data/predictions/race_v3_wide_feature_contract_refactor.parquet`

## 関連ドキュメント

- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-07_feature_contract_refactor_retrain.md`

## 残リスク

- binary を再学習していないため、今回の評価は「新 stack/PL contract の評価」であり、「binary 責務変更込みの end-to-end 再学習」ではない。
- full default inference smoke は current binary artifact と smoke input の TE 列不一致により未確認。
