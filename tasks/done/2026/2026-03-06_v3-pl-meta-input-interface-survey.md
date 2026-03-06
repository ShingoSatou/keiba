# 2026-03-06_v3-pl-meta-input-interface-survey

## タイトル

- v3 PL 学習・推論・bundle/meta interface 調査

## ステータス

- `done`

## 対象範囲

- `scripts_v3/train_pl_v3.py`
- `scripts_v3/predict_race_v3.py`
- `scripts_v3/feature_registry_v3.py`
- `scripts_v3/train_binary_model_v3.py`
- `scripts_v3/train_odds_calibrator_v3.py`
- `scripts_v3/v3_common.py`
- 上記が直接 import する obvious helper

## 対象バージョン

- `v3`

## 前提・仮定

- ユーザー依頼は現状調査と要約であり、実装変更は行わない。
- `docs/` は import 参照の確認を除き調査対象に含めない。

## チェックリスト

- [x] 対象スクリプトの CLI/interface を把握する
- [x] OOF の保存と merge フローを把握する
- [x] bundle meta JSON の構造を把握する
- [x] 推論時の予測生成フローを把握する
- [x] meta inputs 対応に必要な具体的変更点を整理する

## 確認結果

- binary 学習は `scripts_v3/train_binary_model_v3.py` 1 本に集約され、`task` x `model` の wrapper は thin entrypoint。
- binary OOF は `build_oof_frame()` で fold ごとに作成し concat、PL 側は 6 本の binary OOF と 1 本の odds calibration OOF を `race_id` x `horse_no` で left merge する。
- PL 最終 artifact は joblib 辞書で、推論は別 meta JSON ではなくこの artifact の `feature_columns` / `weights` / `preprocess` を直接読む。
- binary bundle meta JSON は推論時に使われるが、実際に参照されるのは `pred_col` / `feature_columns` / `output_paths.main_model` が中心。
- odds calibrator の meta JSON は情報用途で、推論時は `models/odds_win_calibrators_v3.pkl` の joblib bundle が直接使われる。
- `predict_race_v3.py` は 1 レース入力を受け、binary meta 群から base prediction を作り、必要なら odds calibrator を適用し、PL artifact で `pl_score` を出し、MC sampling で `p_top3` と任意で `p_wide` を推定する。
- meta inputs を追加する場合、feature registry の定数追加だけでは不十分で、PL 学習の OOF merge 入口、inference 時の feature materialization、必要なら bundle meta/artifact の input contract 拡張が必要。

## 実行コマンド

- `rg --files -g 'AGENTS.md' -g 'AGENTS.override.md' -g 'tasks/todo.md' -g 'tasks/lessons.md' -g 'scripts_v3/*.py'`
- `sed -n '1,220p' tasks/todo.md`
- `sed -n '1,220p' tasks/lessons.md`
- `sed -n '1,220p' tasks/templates/task.md`
- `sed -n '1,260p' scripts_v3/train_pl_v3.py`
- `sed -n '261,620p' scripts_v3/train_pl_v3.py`
- `sed -n '621,1040p' scripts_v3/train_pl_v3.py`
- `sed -n '1,320p' scripts_v3/predict_race_v3.py`
- `sed -n '321,640p' scripts_v3/predict_race_v3.py`
- `sed -n '1,260p' scripts_v3/feature_registry_v3.py`
- `sed -n '1,340p' scripts_v3/train_binary_model_v3.py`
- `sed -n '341,760p' scripts_v3/train_binary_model_v3.py`
- `sed -n '761,1120p' scripts_v3/train_binary_model_v3.py`
- `sed -n '1,340p' scripts_v3/train_odds_calibrator_v3.py`
- `sed -n '341,520p' scripts_v3/train_odds_calibrator_v3.py`
- `sed -n '1,340p' scripts_v3/v3_common.py`
- `sed -n '1,360p' scripts_v3/train_binary_v3_common.py`
- `sed -n '1,360p' scripts_v3/pl_v3_common.py`
- `sed -n '1,260p' scripts_v3/odds_v3_common.py`
- `sed -n '240,340p' scripts_v3/odds_v3_common.py`
- `sed -n '1,260p' scripts_v3/cv_policy_v3.py`
- `sed -n '1,120p' scripts_v3/train_win_lgbm_v3.py`
- `sed -n '1,120p' scripts_v3/train_place_lgbm_v3.py`
- `sed -n '1,120p' scripts_v3/train_win_xgb_v3.py`
- `sed -n '1,120p' scripts_v3/train_win_cat_v3.py`
- `rg -n "^def parse_args|^def _merge_prediction_features|^def _artifact_from_fit|^def _build_pl_meta_payload|^def main\\(" scripts_v3/train_pl_v3.py`
- `rg -n "^def parse_args|^def _predict_with_meta|^def _apply_base_models|^def _apply_odds_calibrators|^def _score_with_pl|^def main\\(" scripts_v3/predict_race_v3.py`
- `rg -n "^def get_binary_feature_columns|^def get_pl_feature_columns|^def validate_feature_contract|^BINARY_BASE_FEATURES =|^PL_REQUIRED_PRED_FEATURES =|^PL_CONTEXT_FEATURES_SMALL =" scripts_v3/feature_registry_v3.py`
- `rg -n "^def parse_args|^def _resolve_output_paths|^def _build_feature_manifest_payload|^def _build_meta_payload|^def main\\(" scripts_v3/train_binary_model_v3.py`
- `rg -n "^def prepare_binary_frame|^def coerce_feature_matrix|^def compute_binary_metrics|^def build_oof_frame" scripts_v3/train_binary_v3_common.py`
- `rg -n "^def build_group_indices|^def fit_pl_linear_torch|^def predict_linear_scores|^def estimate_p_top3_by_race|^def estimate_p_wide_by_race" scripts_v3/pl_v3_common.py`
- `rg -n "^def assert_t10_no_future_reference|^def merge_odds_features|^def add_implied_probability_columns" scripts_v3/odds_v3_common.py`
- `rg -n "^def build_fixed_window_year_folds|^def select_recent_window_years|^def build_cv_policy_payload|^def attach_cv_policy_columns" scripts_v3/cv_policy_v3.py`
- `rg -n "^def resolve_path|^def save_json|^def hash_files|^def assert_fold_integrity|^def build_race_datetime" scripts_v3/v3_common.py`

## 関連ドキュメント

- `AGENTS.md`
- `tasks/todo.md`

## 残リスク

- 実コードと生成済み artifact/json の実物一致までは確認していない。
- `meta inputs` の厳密な要件定義がないため、変更案は「追加特徴量を training/inference contract に通す」前提の整理。
