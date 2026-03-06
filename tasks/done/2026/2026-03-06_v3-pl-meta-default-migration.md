# 2026-03-06_v3-pl-meta-default-migration

## タイトル

- v3 PL meta_default 契約移行

## ステータス

- `done`

## 対象範囲

- `scripts_v3/train_meta_v3_common.py`
- `scripts_v3/train_win_meta_v3.py`
- `scripts_v3/train_place_meta_v3.py`
- `scripts_v3/feature_registry_v3.py`
- `scripts_v3/train_pl_v3.py`
- `scripts_v3/predict_race_v3.py`
- `test_v3/*`
- `README.md`
- `docs/specs_v3/*`
- `docs/ops_v3/*`

## 対象バージョン

- `v3`

## 前提・仮定

- binary / PL の outer evaluation policy は `train_window_years=4`, `holdout_year=2025`, `cv_window_policy=fixed_sliding` を維持する。
- `p_win_meta` / `p_place_meta` は strict temporal OOF ではなく `race_id` grouped CV 由来の convenience feature として扱う。
- default operational path は引き続き `t10_only` とし、default PL contract で final odds は使わない。
- canonical artifact path は `meta_default` に残し、比較用 `raw_legacy` のみ suffix 付き path を使う。

## チェックリスト

- [x] meta 学習スクリプトを追加する
- [x] PL feature registry を profile 化する
- [x] `train_pl_v3.py` を `meta_default` / `raw_legacy` 対応にする
- [x] `predict_race_v3.py` を meta 推論対応にする
- [x] tests を追加・更新する
- [x] docs / run report を更新する
- [x] targeted test を実行する
- [x] full retrain を実行する
- [x] holdout / wide backtest を実行する

## 確認結果

- `train_meta_v3_common.py` と thin wrapper 2 本を追加し、`race_id` grouped CV の logistic regression meta OOF / holdout / artifact 出力を実装した。
- `feature_registry_v3.py` に `meta_default` / `raw_legacy` profile を追加し、default PL contract を `p_win_meta`, `p_place_meta`, `p_win_odds_t10_norm`, `PL_CONTEXT_FEATURES_SMALL` に切り替えた。
- `train_pl_v3.py` は profile-aware output path、meta/raw legacy merge、holdout scoring、artifact meta 拡張に対応した。
- `predict_race_v3.py` は PL artifact の profile を見て meta model 適用有無を切り替えるようにした。
- docs / run report / README を更新し、meta OOF が strict temporal ではなく reference-only であることを明記した。
- targeted pytest を 2 回実行し、最終確認では `24 passed` を確認した。
- full retrain を実行し、binary 6 本・meta 2 本・PL 2 profile・wide backtest 4 本を生成した。
- 2025 holdout では `raw_legacy` が `meta_default` を上回った。
  - PL holdout: `meta_default` logloss `0.439790`, auc `0.764378`
  - PL holdout: `raw_legacy` logloss `0.437219`, auc `0.769534`
  - wide holdout ROI: `meta_default=0.1179`, `raw_legacy=0.1829`
- `meta_default` 実装は成立したが、今回の実測では運用採否を支持する材料は不足している。

## 実行コマンド

- `git status --short`
- `uv run pytest test_v3/test_meta_v3_common.py test_v3/test_feature_registry_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_predict_race_v3.py test_v3/test_v3_cli_smoke.py`
- `uv run pytest test_v3/test_meta_v3_common.py test_v3/test_feature_registry_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_predict_race_v3.py test_v3/test_v3_cli_smoke.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3.py --input data/features_v2.parquet --output data/features_v3.parquet --meta-output data/features_v3_meta.json`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3.py --input data/features_v2_2025.parquet --output data/features_v3_2025.parquet --meta-output data/features_v3_2025_meta.json`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_win_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_win_meta_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_place_meta_v3.py --holdout-input data/features_v3_2025.parquet`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_pl_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --pl-feature-profile meta_default --train-window-years 4 --holdout-year 2025 --emit-wide-oof`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_pl_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --pl-feature-profile raw_legacy --train-window-years 4 --holdout-year 2025 --emit-wide-oof`
- `/bin/bash -lc 'cd /home/sato/projects/keibas/REPO-cp && set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_wide_oof.parquet --years 2024 --holdout-year 2025 --output data/backtest_v3/backtest_wide_v3_meta_default_oof_2024.json --meta-output data/backtest_v3/backtest_wide_v3_meta_default_oof_2024_meta.json --force'`
- `/bin/bash -lc 'cd /home/sato/projects/keibas/REPO-cp && set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_meta_default_holdout_2025.json --meta-output data/backtest_v3/backtest_wide_v3_meta_default_holdout_2025_meta.json --force'`
- `/bin/bash -lc 'cd /home/sato/projects/keibas/REPO-cp && set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_wide_oof_raw_legacy.parquet --years 2024 --holdout-year 2025 --output data/backtest_v3/backtest_wide_v3_raw_legacy_oof_2024.json --meta-output data/backtest_v3/backtest_wide_v3_raw_legacy_oof_2024_meta.json --force'`
- `/bin/bash -lc 'cd /home/sato/projects/keibas/REPO-cp && set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred_raw_legacy.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_raw_legacy_holdout_2025.json --meta-output data/backtest_v3/backtest_wide_v3_raw_legacy_holdout_2025_meta.json --force'`

## 関連ドキュメント

- `docs/specs_v3/v3_システム仕様書.md`
- `docs/specs_v3/v3_01_全体アーキテクチャ.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- meta 層は grouped CV convenience feature であり、strict temporal OOF と同列比較できない。
- `raw_legacy` で calibrated odds 列を明示 opt-in した場合、holdout scoring 側では対応 holdout 列の事前生成が必要になる。
- `meta_default` はコード実装としては完了したが、2025 holdout / wide holdout の実測では `raw_legacy` 優勢であり、運用 default としての採否は保留が妥当。
