# 2026-03-07_v3-binary6-retrain-end-to-end

## タイトル

- v3 binary 6本再学習と end-to-end 検証

## ステータス

- `done`

## 対象範囲

- `scripts_v3/train_*_v3.py`
- `scripts_v3/train_win_stack_v3.py`
- `scripts_v3/train_place_stack_v3.py`
- `scripts_v3/train_pl_v3.py`
- `scripts_v3/backtest_wide_v3.py`
- `scripts_v3/predict_race_v3.py`
- `docs/ops_v3/v3_run_report_*.md`

## 対象バージョン

- `v3`

## 前提・仮定

- `features_v3.parquet` と `features_v3_2025.parquet` は feature contract refactor 後の内容で再生成済み
- binary default contract は市場オッズを含まない
- current tuned replay は `data/oof/binary_v3_{task}_{model}_best_params.json` を自動読込する
- `win_xgb` は tuned input が `data/features_v3_te.parquet` 系である可能性がある

## チェックリスト

- [x] binary 6本の入力条件を確認する
- [x] binary 6本を再学習する
- [x] stacker / PL / backtest を再実行する
- [x] end-to-end 推論確認を行う
- [x] run report と task を更新する

## 確認結果

- `win_xgb` だけが `data/features_v3_te.parquet` / `data/features_v3_te_2025.parquet` を使う tuned replay だった
- binary 6本の tuned replay を完了し、`data/oof/*_oof.parquet`, `data/holdout/*_holdout_pred_v3.parquet`, `models/*_bundle_meta_v3.json` を更新した
- 再学習後の downstream 指標:
  - `win_stack logloss=0.212096, auc=0.810551`
  - `place_stack logloss=0.423053, auc=0.790955`
  - `PL holdout top3_logloss=0.436695, top3_auc=0.770620`
  - `wide holdout roi=0.1040`
- 初回の full default inference は stale `features_v3_te_2025.parquet` が新 stacker 市場派生列を持たず失敗した
- `features_v3_te.parquet` / `features_v3_te_2025.parquet` を current `features_v3*` + TE 3列で refresh した後、full default inference が通過した
- direct smoke output:
  - `data/predictions/race_v3_pred_binary6_end_to_end_direct.parquet` `rows=16 cols=25`
  - `data/predictions/race_v3_wide_binary6_end_to_end_direct.parquet` `rows=120 cols=12`

## 実行コマンド

- `export UV_CACHE_DIR=/tmp/uv-cache`
- `export V3_MODEL_THREADS=4`
- `uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_xgb_v3.py`
- `uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_stack_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --holdout-year 2025`
- `uv run python scripts_v3/train_place_stack_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --holdout-year 2025`
- `uv run python scripts_v3/train_pl_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --pl-feature-profile stack_default --train-window-years 3 --holdout-year 2025 --emit-wide-oof`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_holdout_2025_binary6_end_to_end.json --meta-output data/backtest_v3/backtest_wide_v3_holdout_2025_binary6_end_to_end_meta.json --force`
- `uv run python scripts_v3/predict_race_v3.py --input data/inference/race_features_smoke_2025_end_to_end.parquet --pl-model models/pl_v3_recent_window.joblib --output data/predictions/race_v3_pred_binary6_end_to_end.parquet --emit-wide --wide-output data/predictions/race_v3_wide_binary6_end_to_end.parquet`
- `uv run python scripts_v3/predict_race_v3.py --input data/inference/race_features_smoke_2025_te.parquet --pl-model models/pl_v3_recent_window.joblib --output data/predictions/race_v3_pred_binary6_end_to_end_direct.parquet --emit-wide --wide-output data/predictions/race_v3_wide_binary6_end_to_end_direct.parquet`

## 関連ドキュメント

- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-07_binary6_end_to_end.md`

## 残リスク

- PL fixed3 OOF fold は current year coverage 上 0 件のまま
- `features_v3_te` 系は build script が無く、今回は current `features_v3*` に TE 3列を再結合して refresh した
