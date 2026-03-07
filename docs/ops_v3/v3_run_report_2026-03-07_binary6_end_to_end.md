# v3 再学習レポート（2026-03-07, binary6 end-to-end）

## 0. 目的

feature contract refactor 後の `v3` について、binary 6 本も含めて再学習し、
`base -> stacker -> PL -> wide backtest -> 1race inference` まで end-to-end で再確認する。

今回のスコープ:

- binary 6 本の tuned replay 再学習
- `win_stack` / `place_stack` の再学習
- `PL stack_default` の再学習
- holdout 2025 wide backtest
- full default inference smoke
- `features_v3_te` / `features_v3_te_2025` の current base features 追随

## 1. binary 6 本再学習

### 1.1 replay 条件

- `train_binary_model_v3.py` は `data/oof/binary_v3_{task}_{model}_best_params.json` を自動読込
- 5 本は `data/features_v3.parquet`
- `win_xgb` のみ `data/features_v3_te.parquet`
- holdout は対応する `*_2025.parquet` を自動補完

### 1.2 CV summary

`data/oof/*_cv_metrics.json` より:

- `win_lgbm`
  - `logloss=0.222053`
  - `brier=0.060630`
  - `auc=0.778855`
  - `ece=0.006672`
  - `benter_r2_valid=0.160515`
  - `final_iterations=195`
- `win_xgb`
  - `logloss=0.222543`
  - `brier=0.060702`
  - `auc=0.776979`
  - `ece=0.006257`
  - `benter_r2_valid=0.165341`
  - `final_iterations=181`
- `win_cat`
  - `logloss=0.222132`
  - `brier=0.060642`
  - `auc=0.779630`
  - `ece=0.007485`
  - `benter_r2_valid=0.165765`
  - `final_iterations=248`
- `place_lgbm`
  - `logloss=0.440187`
  - `brier=0.141051`
  - `auc=0.765670`
  - `ece=0.010820`
  - `final_iterations=351`
- `place_xgb`
  - `logloss=0.441017`
  - `brier=0.141128`
  - `auc=0.764622`
  - `ece=0.015438`
  - `final_iterations=391`
- `place_cat`
  - `logloss=0.440815`
  - `brier=0.141173`
  - `auc=0.764791`
  - `ece=0.016322`
  - `final_iterations=759`

主出力:

- `data/oof/{win,place}_{lgbm,xgb,cat}_oof.parquet`
- `data/holdout/{win,place}_{lgbm,xgb,cat}_holdout_pred_v3.parquet`
- `models/{win,place}_{lgbm,xgb,cat}_{...}_v3.*`

## 2. downstream 再学習

### 2.1 win stack

- `logloss=0.212096`
- `brier=0.058989`
- `auc=0.810551`
- `ece=0.007188`
- `feature_count=19`
- `final_iterations=55`

### 2.2 place stack

- `logloss=0.423053`
- `brier=0.135445`
- `auc=0.790955`
- `ece=0.013110`
- `feature_count=21`
- `final_iterations=61`

### 2.3 PL stack_default

- `feature_count=16`
- `oof_folds=0`
- `holdout_top3_logloss=0.436695`
- `holdout_top3_brier=0.138704`
- `holdout_top3_auc=0.770620`
- `holdout_top3_ece=0.033861`

`data/oof/v3_pipeline_year_coverage.json`:

- `base_oof_years=[2020, 2021, 2022, 2023, 2024]`
- `stacker_oof_years=[2022, 2023, 2024]`
- `pl_oof_valid_years=[]`
- `pl_holdout_train_years=[2022, 2023, 2024]`

今回も仕様どおり、PL fixed3 OOF fold は 0 件のまま。

## 3. wide holdout backtest

入力:

- `data/oof/pl_v3_holdout_2025_pred.parquet`

出力:

- `data/backtest_v3/backtest_wide_v3_holdout_2025_binary6_end_to_end.json`
- `data/backtest_v3/backtest_wide_v3_holdout_2025_binary6_end_to_end_meta.json`

summary:

- `n_races=344`
- `n_bets=1570`
- `n_hits=3`
- `total_bet=1064100`
- `total_return=110710`
- `roi=0.1040`
- `max_drawdown=953390`

## 4. end-to-end inference

### 4.1 発見したギャップ

初回の full default inference は、
`features_v3_te_2025.parquet` が `win_xgb` 用 TE 列は持つ一方で、
新 stacker 市場派生

- `p_win_odds_t20_norm`
- `p_win_odds_t15_norm`
- `d_logit_win_15_20`
- `d_logit_win_10_15`
- `d_logit_win_10_20`
- `place_mid_prob_t20/t15/t10`
- `d_place_mid_10_20`
- `d_place_width_10_20`

を持っていなかったため失敗した。

### 4.2 対応

`features_v3.parquet` / `features_v3_2025.parquet` の current base columns に、
既存 TE 3 列

- `jockey_target_label_mean_6m`
- `trainer_target_label_mean_6m`
- `rel_jockey_target_label_mean_z`

だけを結合し、`features_v3_te.parquet` / `features_v3_te_2025.parquet` を refresh した。

refresh 後:

- `data/features_v3_te.parquet`
  - `rows=45444`
  - `cols=103`
- `data/features_v3_te_2025.parquet`
  - `rows=4971`
  - `cols=103`

### 4.3 direct smoke

`data/inference/race_features_smoke_2025_te.parquet` を
refresh 後 `features_v3_te_2025.parquet` から 1 レース切り出して作成し、
full default path を通した。

出力:

- `data/predictions/race_v3_pred_binary6_end_to_end_direct.parquet`
  - `rows=16`
  - `cols=25`
- `data/predictions/race_v3_wide_binary6_end_to_end_direct.parquet`
  - `rows=120`
  - `cols=12`

確認できたこと:

1. base model 6 本の artifact が現行 contract で推論できる
2. strict temporal stacker が current market features で `p_win_stack` / `p_place_stack` を生成できる
3. `PL stack_default` が `pl_score`, `p_top3`, `p_wide` を生成できる

## 5. 総合所見

1. binary 6 本の再学習から stacker / PL / wide backtest / 1race inference まで一通り更新できた。
2. full default inference を通すには、TE input も current base features に追随している必要がある。今回は artifact refresh で揃えた。
3. holdout 2025 wide ROI は `0.1040`。
4. PL fixed3 OOF fold 0 件は年範囲由来であり、今回も仕様どおりの挙動。

## 6. 実行コマンド

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export V3_MODEL_THREADS=4

uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_win_xgb_v3.py
uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet

uv run python scripts_v3/train_win_stack_v3.py \
  --features-input data/features_v3.parquet \
  --holdout-input data/features_v3_2025.parquet \
  --holdout-year 2025

uv run python scripts_v3/train_place_stack_v3.py \
  --features-input data/features_v3.parquet \
  --holdout-input data/features_v3_2025.parquet \
  --holdout-year 2025

uv run python scripts_v3/train_pl_v3.py \
  --features-input data/features_v3.parquet \
  --holdout-input data/features_v3_2025.parquet \
  --pl-feature-profile stack_default \
  --train-window-years 3 \
  --holdout-year 2025 \
  --emit-wide-oof

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_holdout_2025_pred.parquet \
  --years 2025 \
  --holdout-year 2026 \
  --output data/backtest_v3/backtest_wide_v3_holdout_2025_binary6_end_to_end.json \
  --meta-output data/backtest_v3/backtest_wide_v3_holdout_2025_binary6_end_to_end_meta.json \
  --force

uv run python scripts_v3/predict_race_v3.py \
  --input data/inference/race_features_smoke_2025_te.parquet \
  --pl-model models/pl_v3_recent_window.joblib \
  --output data/predictions/race_v3_pred_binary6_end_to_end_direct.parquet \
  --emit-wide \
  --wide-output data/predictions/race_v3_wide_binary6_end_to_end_direct.parquet
```
