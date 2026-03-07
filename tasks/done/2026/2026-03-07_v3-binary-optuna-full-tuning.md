# 2026-03-07_v3-binary-optuna-full-tuning

## タイトル

- v3 binary full tuning 実行と tuned default 反映

## ステータス

- `done`

## 対象範囲

- `data/features_v3_te.parquet`
- `data/features_v3_te_2025.parquet`
- `data/oof/binary_v3_*`
- `data/optuna/binary_v3_*`
- `scripts_v3/train_binary_model_v3.py`
- `docs/specs_v3/`
- `docs/ops_v3/`

## 対象バージョン

- `v3`

## 前提・仮定

- full tuning は `win/place x lgbm/xgb/cat` の 6 study を対象とする。
- 各 study は `scripts_v3/tune_binary_optuna_v3.py` の default `n_trials=300` を使う。
- tuned default 反映には best feature_set も含める。

## チェックリスト

- [x] v3 TE 入力を生成する
- [x] 6 study の full tuning を完了する
- [x] best 結果を集約する
- [x] tuned default を学習側へ反映する
- [x] 反映後の確認を実行する

## 確認結果

- 着手時点で `data/features_v3_te.parquet` は未生成、`data/features_v2_te.parquet` は存在する。
- `data/features_v3_te.parquet` / `data/features_v3_te_2025.parquet` を生成済み。
- `train_binary_model_v3.py` は `data/oof/binary_v3_{task}_{model}_best_params.json` を自動適用し、TE tuned input の場合は対応 holdout parquet も自動補完するように更新済み。
- full tuning は 2026-03-07 13:54 JST に開始し、2026-03-07 15:15 JST に 6 study 完了。
- best 集約:
  - `win_lgbm`: `feature_set=base`, mean CV logloss `0.208939076045`
  - `win_xgb`: `feature_set=te`, mean CV logloss `0.208569606492`
  - `win_cat`: `feature_set=base`, mean CV logloss `0.208773043869`
  - `place_lgbm`: `feature_set=base`, mean CV logloss `0.420708670257`
  - `place_xgb`: `feature_set=base`, mean CV logloss `0.420306583377`
  - `place_cat`: `feature_set=base`, mean CV logloss `0.420168474693`
- tuned best の再学習を 6 wrapper で実行し、`*_cv_metrics.json` / `*_bundle_meta_v3.json` / `*_holdout_pred_v3.parquet` が更新され、`params_json` と `input_path` が best と一致することを確認済み。

## 実行コマンド

- `rg --files data | rg 'features_v3(_te)?\\.parquet|features_v3_2025(_te)?\\.parquet'`
- `rg -n \"with-te|features_v3_te|target_label_mean|build_features_v3|features_v2_te\" scripts_v3 docs/ops_v3 docs/specs_v3 README.md`
- `sed -n '1,260p' scripts_v3/build_features_v3.py`
- `uv run python scripts_v3/build_features_v3.py --input data/features_v2_te.parquet --output data/features_v3_te.parquet --meta-output data/features_v3_te_meta.json`
- `uv run python scripts_v3/build_features_v3.py --input data/features_v2_te_2025.parquet --output data/features_v3_te_2025.parquet --meta-output data/features_v3_te_2025_meta.json`
- `V3_MODEL_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /tmp/run_binary_v3_parallel.sh`
- `uv run pytest -q -s test_v3/test_binary_feature_contract_v3.py test_v3/test_v3_cli_smoke.py test_v3/test_tune_binary_optuna_v3.py`
- `V3_MODEL_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /tmp/retrain_binary_v3_tuned.sh`

## 関連ドキュメント

- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- `binary_v3_win_lgbm.sqlite3` には中断由来の stale `RUNNING` trial が 1 本残るため、study の `total_trials` は 301 になっている。ただし `best/best_params` は regenerate 済みで、再学習成果物への影響はない。
