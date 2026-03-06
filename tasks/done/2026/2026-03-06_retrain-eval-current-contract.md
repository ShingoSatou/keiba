# 2026-03-06_retrain-eval-current-contract

## タイトル

- current contract の再学習・検証・評価

## ステータス

- `done`

## 対象範囲

- `data/features_v3.parquet`, `data/features_v3_meta.json`
- `data/features_v3_2025.parquet`, `data/features_v3_2025_meta.json`
- `data/oof/win_*`, `data/oof/place_*`, `data/oof/odds_win_calibration_*`, `data/oof/pl_v3_*`
- `data/holdout/win_*`, `data/holdout/place_*`
- `data/backtest_v3/backtest_wide_v3_direct_oof_current.json`
- `data/backtest_v3/backtest_wide_v3_direct_holdout_2025_current.json`
- `models/win_*_v3.*`, `models/place_*_v3.*`, `models/*_bundle_meta_v3.json`, `models/*_feature_manifest_v3.json`
- `models/odds_win_calibrators_v3.pkl`, `models/pl_v3_*.joblib`

## 対象バージョン

- `v3`

## 前提・仮定

- 再学習は current default contract (`t10_only`, entity ID OFF, PL explicit list) で実施
- binary holdout 評価は `data/features_v3_2025.parquet` を入力に利用
- PL holdout 評価は binary holdout 予測 + calibrator + `pl_v3_recent_window.joblib` から `pl_v3_holdout_2025_pred.parquet` を再生成
- `torch` は未導入のため、PL は NumPy optimizer fallback

## チェックリスト

- [x] `build_features_v3.py` を train / 2025 holdout で再実行
- [x] binary 6 本を再学習し、holdout 予測も更新
- [x] odds calibrator を再学習
- [x] PL を再学習し、OOF / wide OOF を更新
- [x] PL holdout 2025 予測を再生成
- [x] OOF / holdout の wide ROI を再計測
- [x] manifest / meta の contract を確認
- [x] 結果・評価・考察を別ドキュメントへ整理

## 確認結果

- `features_v3_meta.json`, `features_v3_2025_meta.json` ともに `operational_default=t10_only`, `contains_final_odds_columns=true`, `contains_t10_odds_columns=true`
- binary manifest は 6 本すべて `operational_mode=t10_only`, `include_entity_id_features=false`, `feature_count=46`
- binary manifest 6 本すべてで final odds / entity raw ID は未混入
- PL meta は `operational_mode=t10_only`, `feature_count=20`
- PL feature columns は `required OOF + t10 odds + small context` のみで、final odds / entity raw ID は未混入
- PL 学習は `torch` 未導入のため NumPy fallback で完了
- 再学習結果・評価・考察を `docs/ops_v3/v3_run_report_2026-03-06_feature_governance_retrain.md` に分離記録

主要評価値:
- win CV:
  - `win_lgbm` logloss `0.20979`, auc `0.81874`
  - `win_xgb` logloss `0.20916`, auc `0.81990`
  - `win_cat` logloss `0.20757`, auc `0.82392`
- win 2025 holdout:
  - `win_lgbm` logloss `0.20807`, auc `0.80719`
  - `win_xgb` logloss `0.20806`, auc `0.80660`
  - `win_cat` logloss `0.20754`, auc `0.80713`
- place CV:
  - `place_lgbm` logloss `0.42084`, auc `0.79424`
  - `place_xgb` logloss `0.42071`, auc `0.79439`
  - `place_cat` logloss `0.41772`, auc `0.79793`
- place 2025 holdout:
  - `place_lgbm` logloss `0.43235`, auc `0.76918`
  - `place_xgb` logloss `0.42955`, auc `0.77307`
  - `place_cat` logloss `0.42857`, auc `0.77476`
- PL:
  - CV `pl_nll_valid(mean)=23.73331`, `top3_logloss(mean)=0.43174`, `top3_auc(mean)=0.79298`
  - 2025 holdout `top3_logloss=0.43718`, `top3_auc=0.77005`
- wide ROI:
  - OOF current (`valid_year=2024`): ROI `0.0050`, bets `1317`
  - holdout 2025 current: ROI `0.3816`, bets `1705`

## 実行コマンド

- `uv run python scripts_v3/build_features_v3.py`
- `uv run python scripts_v3/build_features_v3.py --input data/features_v2_2025.parquet --output data/features_v3_2025.parquet --meta-output data/features_v3_2025_meta.json`
- `uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_odds_calibrator_v3.py`
- `uv run python scripts_v3/train_pl_v3.py --emit-wide-oof`
- `uv run python - <<'PY' ...` で `data/oof/pl_v3_holdout_2025_pred.parquet` を再生成
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_wide_oof.parquet --holdout-year 2025 --output data/backtest_v3/backtest_wide_v3_direct_oof_current.json --meta-output data/backtest_v3/backtest_wide_v3_direct_oof_current_meta.json --force`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_current.json --meta-output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_current_meta.json --force`

## 関連ドキュメント

- `docs/ops_v3/v3_run_report_2026-03-06_feature_governance_retrain.md`
- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-04.md`

## 残リスク

- PL の OOF backtest は default `train-window-years=3` のため、対象年が `2024` のみ
- wide ROI は DB の `core.o3_wide` / `core.payout` 内容に依存するため再計測で変動しうる
- `docs/ops_v3/v3_run_report_2026-03-04.md` は履歴文書として残しており、current run の数値には更新していない

## 移行メモ

- 2026-03-06 に旧 monolithic `tasks/todo.md` から分割移行
