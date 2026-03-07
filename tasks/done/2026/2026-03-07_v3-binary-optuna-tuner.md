# 2026-03-07_v3-binary-optuna-tuner

## タイトル

- v3 binary Optuna tuner 追加

## ステータス

- `done`

## 対象範囲

- `scripts_v3/tune_binary_optuna_v3.py`
- `scripts_v3/train_binary_model_v3.py`
- `scripts_v3/feature_registry_v3.py`
- `docs/specs_v3/`
- `docs/ops_v3/`
- `test_v3/`

## 対象バージョン

- `v3`

## 前提・仮定

- binary のみを対象とし、`task in {win, place}`、`model in {lgbm, xgb, cat}` を 1 実行 1 study で扱う。
- tuner の固定契約は `train_window_years=4`、`operational_mode=t10_only`、`include_entity_id_features=false`。
- TE 再現は入力ファイル名の basename に `_te` を含む場合に有効とする。
- `best_params_output` は再学習条件の JSON を出力し、`train_binary_model_v3.py` の CLI 自体は増やさない。

## チェックリスト

- [x] task 台帳を更新する
- [x] binary feature selection の TE replay を共通化する
- [x] Optuna tuner を追加する
- [x] best trial 制約選抜と baseline を実装する
- [x] tests を追加・更新する
- [x] docs を更新する
- [x] targeted verification を実行する

## 確認結果

- 着手前に `docs/specs_v3/v3_システム仕様書.md`、`docs/specs_v3/v3_03_二値分類と校正仕様.md`、`docs/ops_v3/スクリプトリファレンス.md` を確認した。
- 既存 v3 binary は fixed-sliding 4 年 CV、`t10_only` default、entity raw ID default OFF、win のみ Benter R* を fold valid で計算する。
- 既存 `train_binary_model_v3.py` には `--params-json` はなく、再現は input path と個別 CLI 引数で行う前提。
- `uv run ruff check` は今回の Python 変更ファイル群で通過した。
- `uv run pytest -s test_v3/test_binary_feature_contract_v3.py test_v3/test_tune_binary_optuna_v3.py test_v3/test_v3_cli_smoke.py` は `18 passed` だった。
- synthetic parquet で `scripts_v3/tune_binary_optuna_v3.py --task win --model lgbm --n-trials 1` の smoke 実行に成功し、trials / best / best_params の 3 出力を `/tmp/repo_cp_optuna_smoke/` に生成できた。
- synthetic smoke では極小データのため LightGBM が `There are no meaningful features` 警告を出したが、study 完走と出力生成は確認できた。

## 実行コマンド

- `git status --short`
- `rg --files -g 'AGENTS.md' -g 'AGENTS.override.md' -g 'train_binary_model_v3.py' -g 'tune_ranker_optuna_v2.py' -g 'v3_システム仕様書.md' -g 'v3_03_二値分類と校正仕様.md' -g 'スクリプトリファレンス.md' -g 'todo.md' -g 'lessons.md'`
- `sed -n '1,260p' docs/specs_v3/v3_システム仕様書.md`
- `sed -n '1,320p' docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `sed -n '1,260p' docs/ops_v3/スクリプトリファレンス.md`
- `sed -n '1,220p' tasks/todo.md`
- `sed -n '1,220p' tasks/lessons.md`
- `sed -n '1,1160p' scripts_v3/train_binary_model_v3.py`
- `sed -n '1,520p' scripts_v2/tune_ranker_optuna_v2.py`
- `sed -n '1,320p' scripts_v3/train_binary_v3_common.py`
- `sed -n '1,420p' scripts_v3/feature_registry_v3.py`
- `sed -n '1,220p' scripts_v3/metrics_benter_v3_common.py`
- `sed -n '1,220p' scripts_v3/v3_common.py`
- `sed -n '1,220p' test_v3/test_binary_feature_contract_v3.py`
- `sed -n '1,220p' test_v3/test_benter_r2_v3.py`
- `sed -n '1,220p' test_v3/test_cv_window_policy_v3.py`
- `sed -n '1,240p' README.md`
- `uv run ruff check scripts_v3/feature_registry_v3.py scripts_v3/train_binary_model_v3.py scripts_v3/tune_binary_optuna_v3.py test_v3/test_binary_feature_contract_v3.py test_v3/test_tune_binary_optuna_v3.py test_v3/test_v3_cli_smoke.py`
- `uv run pytest -s test_v3/test_binary_feature_contract_v3.py test_v3/test_tune_binary_optuna_v3.py test_v3/test_v3_cli_smoke.py`
- `uv run python - <<'PY' ... create synthetic /tmp/repo_cp_optuna_smoke parquet ... PY`
- `uv run python scripts_v3/tune_binary_optuna_v3.py --task win --model lgbm --input-base /tmp/repo_cp_optuna_smoke/features_v3.parquet --input-te /tmp/repo_cp_optuna_smoke/features_v3_te.parquet --holdout-year 2025 --n-trials 1 --study-name binary_v3_win_lgbm_smoke --storage /tmp/repo_cp_optuna_smoke/binary_v3_win_lgbm_smoke.sqlite3 --trials-output /tmp/repo_cp_optuna_smoke/binary_v3_win_lgbm_smoke_trials.parquet --best-output /tmp/repo_cp_optuna_smoke/binary_v3_win_lgbm_smoke_best.json --best-params-output /tmp/repo_cp_optuna_smoke/binary_v3_win_lgbm_smoke_best_params.json`

## 関連ドキュメント

- `docs/specs_v3/v3_システム仕様書.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_05_共通基盤と付録.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- v3 の TE 列名仕様はまだ厳密に文書化されておらず、safe TE extra 抽出は current naming convention（`target` / `te_` 系）に依存する。
- 実データの `data/features_v3_te.parquet` を使った full tuning は未実施で、実運用時の探索時間や trial 分布は未確認。
