# v3 再学習レポート（2026-03-07, feature contract refactor）

## 0. 目的

`binary=能力推定 / stack=市場統合 / PL=順位分布化` の責務整理を反映した後、
`features_v3` を再生成し、`stack_win` / `stack_place` / `PL stack_default` を再学習して holdout 2025 で評価する。

今回のスコープ:

- `features_v3`, `features_v3_2025` の再生成
- `stack_win`, `stack_place` の再学習
- `PL stack_default` の再学習
- holdout `top3` / wide backtest / inference smoke の確認

非スコープ:

- binary 6 本の再学習
  - 今回は既存 binary OOF / holdout を固定入力として利用した

## 1. features 再生成

### 1.1 train

- `data/features_v3.parquet`
  - `rows=45444`
  - `races=3188`

### 1.2 holdout

- `data/features_v3_2025.parquet`
  - `rows=4971`
  - `races=344`

### 1.3 新市場派生の coverage

`data/features_v3_meta.json`, `data/features_v3_2025_meta.json` より:

- train / holdout とも `notna_rate=1.0`
  - `p_win_odds_t20_norm`
  - `p_win_odds_t15_norm`
  - `p_win_odds_t10_norm`
  - `d_logit_win_15_20`
  - `d_logit_win_10_15`
  - `d_logit_win_10_20`
  - `place_mid_prob_t20`
  - `place_mid_prob_t15`
  - `place_mid_prob_t10`
  - `d_place_mid_10_20`
  - `d_place_width_10_20`
  - `place_width_log_ratio`

## 2. stacker 再学習

### 2.1 win stack

- feature columns:
  - `p_win_{lgbm,xgb,cat}`
  - `p_win_odds_t20_norm`, `p_win_odds_t15_norm`, `p_win_odds_t10_norm`
  - `d_logit_win_15_20`, `d_logit_win_10_15`, `d_logit_win_10_20`
  - context 10 列
- OOF:
  - `data/oof/win_stack_oof.parquet`
  - `rows=14819`
  - `valid_years=[2022, 2023, 2024]`
- CV summary:
  - `logloss=0.211415`
  - `brier=0.058777`
  - `auc=0.814222`
  - `ece=0.005626`

前回 report（`strict_stacker_retrain`）比:

- logloss: `0.211616 -> 0.211415`
- auc: `0.813708 -> 0.814222`

### 2.2 place stack

- feature columns:
  - `p_place_{lgbm,xgb,cat}`
  - `place_mid_prob_t20/t15/t10`
  - `place_width_log_ratio_t20/t15/t10`
  - `d_place_mid_10_20`
  - `d_place_width_10_20`
  - context 10 列
- OOF:
  - `data/oof/place_stack_oof.parquet`
  - `rows=14819`
  - `valid_years=[2022, 2023, 2024]`
- CV summary:
  - `logloss=0.419472`
  - `brier=0.134120`
  - `auc=0.796010`
  - `ece=0.010986`

前回 report 比:

- logloss: `0.420166 -> 0.419472`
- auc: `0.795184 -> 0.796010`

## 3. PL 再学習

- profile: `stack_default`
- required pred cols:
  - `p_win_stack`
  - `p_place_stack`
- feature columns:
  - `z_win_stack`
  - `z_place_stack`
  - `place_width_log_ratio`
  - `z_win_stack_x_z_place_stack`
  - `z_win_stack_x_place_width_log_ratio`
  - `z_place_stack_x_place_width_log_ratio`
  - `z_win_stack_x_field_size`
  - `z_place_stack_x_field_size`
  - `z_win_stack_x_distance_m`
  - `z_place_stack_x_distance_m`
  - `z_win_stack_race_centered`
  - `z_place_stack_race_centered`
  - `place_width_log_ratio_race_centered`
  - `z_win_stack_rank_pct`
  - `z_place_stack_rank_pct`
  - `place_width_log_ratio_rank_pct`

### 3.1 year coverage

`data/oof/v3_pipeline_year_coverage.json`:

- `base_oof_years=[2020, 2021, 2022, 2023, 2024]`
- `stacker_oof_years=[2022, 2023, 2024]`
- `pl_eligible_years=[2022, 2023, 2024]`
- `pl_oof_valid_years=[]`
- `pl_holdout_train_years=[2022, 2023, 2024]`
- `pl_fixed_window_oof_feasible=false`

今回も strict temporal stacker coverage の都合で PL fixed3 OOF fold は 0 件。
仕様どおり、空 OOF を保存しつつ holdout / final artifact を生成した。

### 3.2 holdout 2025

`data/oof/pl_v3_cv_metrics.json` より:

- `top3_logloss=0.435280`
- `top3_brier=0.138199`
- `top3_auc=0.773852`
- `top3_ece=0.036172`

前回 report 比:

- `top3_logloss: 0.436188 -> 0.435280`
- `top3_auc: 0.772327 -> 0.773852`

## 4. wide holdout backtest

入力:

- `data/oof/pl_v3_holdout_2025_pred.parquet`
- 出力:
  - `data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor.json`
  - `data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor_meta.json`

summary:

- `n_races=344`
- `n_bets=1569`
- `n_hits=5`
- `hit_rate=0.0032`
- `total_bet=1091300`
- `total_return=143440`
- `roi=0.1314`
- `max_drawdown=947860`
- `logloss=0.4353`
- `auc=0.7739`

前回 report 比:

- `roi: 0.0984 -> 0.1314`

## 5. inference smoke

### 5.1 full default path

- `data/inference/race_features_smoke_2025.parquet` に対する full default path は、
  base model artifact が TE 列
  - `jockey_target_label_mean_6m`
  - `trainer_target_label_mean_6m`
  - `rel_jockey_target_label_mean_z`
  を要求するため未通過。

これは今回の stacker / PL contract 変更ではなく、smoke input と current binary artifact の整合問題。

### 5.2 stacker / PL path verification

- holdout 2025 の 1 レースに base holdout 予測をマージした入力を用意し、
  `--skip-base-model-inference` で stacker / PL path を検証した。
- 出力:
  - `data/predictions/race_v3_pred_feature_contract_refactor.parquet`
  - `data/predictions/race_v3_wide_feature_contract_refactor.parquet`

確認:

- updated stacker manifest を使って `p_win_stack`, `p_place_stack` を生成できる
- updated PL artifact で `pl_score`, `p_top3`, `p_wide` を生成できる

## 6. 総合所見

1. 新市場派生列は train / holdout とも欠損なく生成された。
2. stacker は win / place ともに前回 report よりわずかに改善した。
3. PL holdout `top3_logloss` / `top3_auc` も改善した。
4. holdout 2025 wide ROI は `0.1314` で、前回 `0.0984` を上回った。
5. ただし今回は binary を再学習していないため、評価は「新 stack/PL contract の改善効果」に限定される。

## 7. 実行コマンド

```bash
set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3.py \
  --input data/features_base.parquet \
  --output data/features_v3.parquet \
  --meta-output data/features_v3_meta.json

set -a && source .env && set +a && UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/build_features_v3.py \
  --input data/features_v2_2025.parquet \
  --output data/features_v3_2025.parquet \
  --meta-output data/features_v3_2025_meta.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_win_stack_v3.py \
  --features-input data/features_v3.parquet \
  --holdout-input data/features_v3_2025.parquet \
  --holdout-year 2025

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_place_stack_v3.py \
  --features-input data/features_v3.parquet \
  --holdout-input data/features_v3_2025.parquet \
  --holdout-year 2025

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/train_pl_v3.py \
  --features-input data/features_v3.parquet \
  --holdout-input data/features_v3_2025.parquet \
  --pl-feature-profile stack_default \
  --train-window-years 3 \
  --holdout-year 2025 \
  --emit-wide-oof

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_holdout_2025_pred.parquet \
  --years 2025 \
  --holdout-year 2026 \
  --output data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor.json \
  --meta-output data/backtest_v3/backtest_wide_v3_holdout_2025_feature_contract_refactor_meta.json \
  --force

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts_v3/predict_race_v3.py \
  --input data/inference/race_features_smoke_2025_with_base_preds.parquet \
  --pl-model models/pl_v3_recent_window.joblib \
  --skip-base-model-inference \
  --output data/predictions/race_v3_pred_feature_contract_refactor.parquet \
  --emit-wide \
  --wide-output data/predictions/race_v3_wide_feature_contract_refactor.parquet
```
