# v3 再学習レポート（2026-03-07, strict temporal stacker / PL stack_default）

## 0. 目的

v3 の default main path を `base -> strict temporal stacker -> PL` に切り替えた後、
full retrain を end-to-end で実行し、holdout 2025 と wide backtest で評価する。

対象:

- features: `features_v3` の stacker 用 snapshot 拡張
- binary: `win/place x {lgbm,xgb,cat}`
- stacker: `stack_win`, `stack_place`
- PL: `stack_default`
- downstream: `p_top3`, `p_wide`, wide backtest

## 1. 実行条件

- train / CV 対象年: `2016-2024`
- holdout: `2025`
- binary CV: `fixed_sliding`, `train_window_years=4`
- stacker CV: `capped_expanding`, `min_train_years=2`, `max_train_years=4`
- PL CV: `fixed_sliding`, `train_window_years=3`
- PL profile: `stack_default`
- PL backend: `torch` 未導入のため NumPy optimizer fallback

## 2. 生成確認

### 2.1 features

- `data/features_v3.parquet`
  - `rows=45444`, `races=3188`
- `data/features_v3_2025.parquet`
  - `rows=4971`, `races=344`
- 追加列の存在を確認:
  - `odds_win_t20`, `odds_win_t15`, `odds_win_t10`
  - `odds_place_t20_lower`, `odds_place_t20_upper`
  - `odds_place_t15_lower`, `odds_place_t15_upper`
  - `odds_place_t10_lower`, `odds_place_t10_upper`
  - `place_width_log_ratio`
- meta 確認:
  - `data/features_v3_meta.json`
  - `data/features_v3_2025_meta.json`
  - `contains_stacker_timeseries_columns=true`

### 2.2 year coverage

`data/oof/v3_pipeline_year_coverage.json`

| item | years |
|---|---|
| `base_oof_years` | `2020, 2021, 2022, 2023, 2024` |
| `stacker_oof_years` | `2022, 2023, 2024` |
| `pl_eligible_years` | `2022, 2023, 2024` |
| `pl_oof_valid_years` | `[]` |
| `pl_holdout_train_years` | `2022, 2023, 2024` |

所見:

- strict temporal stacker 導入後、PL fixed3 の OOF fold は current year coverage では成立しない。
- 仕様どおり、PL は空 OOF を保存しつつ holdout / final 学習を継続した。

## 3. binary 結果

### 3.1 単勝

| model | CV logloss | CV auc | 2025 holdout logloss | 2025 holdout brier | 2025 holdout auc |
|---|---:|---:|---:|---:|---:|
| `win_lgbm` | 0.212963 | 0.811932 | 0.210448 | 0.057095 | 0.799010 |
| `win_xgb` | 0.211333 | 0.814915 | 0.209372 | 0.056803 | 0.801939 |
| `win_cat` | 0.210247 | 0.817261 | 0.207773 | 0.056507 | 0.805295 |

### 3.2 複勝

| model | CV logloss | CV auc | 2025 holdout logloss | 2025 holdout brier | 2025 holdout auc |
|---|---:|---:|---:|---:|---:|
| `place_lgbm` | 0.423491 | 0.791200 | 0.434271 | 0.138870 | 0.766301 |
| `place_xgb` | 0.423003 | 0.791684 | 0.431576 | 0.137904 | 0.769599 |
| `place_cat` | 0.421505 | 0.793757 | 0.428915 | 0.136796 | 0.774581 |

所見:

- binary は win / place ともに `cat` が最良。
- holdout 2025 では place の劣化幅が win より大きい。

## 4. stacker 結果

### 4.1 OOF 成果物

- `data/oof/win_stack_oof.parquet`
  - `rows=14819`, `valid_years=[2022, 2023, 2024]`
- `data/oof/place_stack_oof.parquet`
  - `rows=14819`, `valid_years=[2022, 2023, 2024]`

### 4.2 指標

| model | CV logloss | CV auc | 2025 holdout logloss | 2025 holdout brier | 2025 holdout auc |
|---|---:|---:|---:|---:|---:|
| `win_stack` | 0.211616 | 0.813708 | 0.211822 | 0.057332 | 0.796023 |
| `place_stack` | 0.420166 | 0.795184 | 0.429378 | 0.136964 | 0.773059 |

所見:

- `stack_win` / `stack_place` ともに strict temporal OOF は `2022-2024` で成立。
- holdout 2025 では binary best 単体を明確に上回ってはいないが、PL への責務分離用の主入力として契約どおり生成できている。

## 5. PL / top3 結果

- `data/oof/pl_v3_oof.parquet`
  - `rows=0`, `cols=21`
- `data/oof/pl_v3_holdout_2025_pred.parquet`
  - `rows=4971`, `races=344`, `valid_years=[2025]`

### 5.1 holdout 2025

| metric | value |
|---|---:|
| `top3_logloss` | 0.436188 |
| `top3_brier` | 0.138378 |
| `top3_auc` | 0.772327 |
| `top3_ece` | 0.036356 |

### 5.2 feature contract

PL の実使用列は以下で確認した。

- 主入力
  - `z_win_stack`
  - `z_place_stack`
  - `place_width_log_ratio`
- interaction block
  - `track_code`
  - `surface`
  - `distance_m`
  - `going`
  - `weather`
  - `field_size`
  - `grade_code`
  - `race_type_code`
  - `weight_type_code`
  - `condition_code_min_age`
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

## 6. wide holdout backtest

入力:

- `data/oof/pl_v3_holdout_2025_pred.parquet`
- `data/backtest_v3/backtest_wide_v3_holdout_2025_strict_stack.json`

| metric | value |
|---|---:|
| `period_from` | `2025-01-05` |
| `period_to` | `2025-06-01` |
| `n_races` | 344 |
| `n_bets` | 1533 |
| `n_hits` | 6 |
| `hit_rate` | 0.0039 |
| `total_bet` | 1061500 |
| `total_return` | 104490 |
| `roi` | 0.0984 |
| `max_drawdown` | 957010 |
| `logloss` | 0.4362 |
| `auc` | 0.7723 |

所見:

- strict temporal stacker + PL stack_default の holdout 2025 wide ROI は `0.0984`。
- ROI は単年・payout DB 依存でぶれやすいため、binary / top3 指標と分けて扱う。

## 7. inference smoke

single-race 推論を trained artifact で確認した。

- 入力: `data/inference/race_features_smoke_2025.parquet`
- 出力:
  - `data/predictions/race_v3_pred_smoke_2025.parquet`
    - `rows=16`, `cols=25`
  - `data/predictions/race_v3_wide_smoke_2025.parquet`
    - `rows=120`, `cols=12`

確認:

- default 順序 `base -> stacker -> PL` で推論が通る
- `p_win_stack`, `p_place_stack`, `p_top3`, `p_wide` が生成される

## 8. 総合評価

1. full retrain は `base -> stacker -> PL` の 3 層で end-to-end 完了した。
2. strict temporal stacker OOF は `2022-2024` で成立し、PL には同年 fitted 値を混入していない。
3. current year coverage では PL fixed3 OOF は空だが、これは仕様どおりの結果であり `v3_pipeline_year_coverage.json` で確認できる。
4. holdout 2025 の main 指標は `top3_logloss=0.436188`, `top3_auc=0.772327`, wide `roi=0.0984`。
5. 現時点の次の改善候補は、stacker 特徴量の質改善か、PL interaction block の再設計であり、リーク防止と year coverage 契約は維持する。

## 9. 実行コマンド

```bash
uv run python scripts_v3/build_features_v3.py \
  --input data/features_v2.parquet \
  --output data/features_v3.parquet \
  --meta-output data/features_v3_meta.json

uv run python scripts_v3/build_features_v3.py \
  --input data/features_v2_2025.parquet \
  --output data/features_v3_2025.parquet \
  --meta-output data/features_v3_2025_meta.json

uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_win_xgb_v3.py --holdout-input data/features_v3_2025.parquet
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
  --output data/backtest_v3/backtest_wide_v3_holdout_2025_strict_stack.json \
  --meta-output data/backtest_v3/backtest_wide_v3_holdout_2025_strict_stack_meta.json \
  --force

uv run python scripts_v3/predict_race_v3.py \
  --input data/inference/race_features_smoke_2025.parquet \
  --pl-model models/pl_v3_recent_window.joblib \
  --output data/predictions/race_v3_pred_smoke_2025.parquet \
  --emit-wide \
  --wide-output data/predictions/race_v3_wide_smoke_2025.parquet
```

## 10. 注意点

- `torch` 未導入のため、PL は NumPy fallback で学習した。
- `pl_v3_oof.parquet` が空なのは異常ではなく、strict temporal stacker coverage と PL fixed3 の組み合わせによる仕様結果。
- wide ROI は単年・DB依存なので、モデル採否は `top3_*` 指標と year coverage と合わせて判断する。
