# v3 再学習レポート（2026-03-06, feature governance hardening 後）

## 0. 目的

feature governance hardening 後の current v3 contract で再学習し、
結果、評価、考察を 2026-03-04 の旧 run report とは分離して記録する。

対象変更:
- binary: `t10_only` default、entity raw ID default OFF、feature manifest 出力
- PL: `required OOF + t10 odds + small context` の explicit list 化
- build_features meta: `operational_default`, odds presence の明示

## 1. 実行条件

- 学習/CV対象: 2016-2024（`holdout_year=2025`）
- holdout 評価: 2025（`data/features_v3_2025.parquet`）
- binary / PL ともに feature registry 契約を適用
- PL は `torch` 未導入のため NumPy optimizer fallback で学習

## 2. Contract 検証結果

- `data/features_v3_meta.json`, `data/features_v3_2025_meta.json`
  - `operational_default=t10_only`
  - `contains_final_odds_columns=true`
  - `contains_t10_odds_columns=true`
- binary manifest 6 本
  - `operational_mode=t10_only`
  - `include_entity_id_features=false`
  - `feature_count=46`
  - final odds / `jockey_key` / `trainer_key` は未混入
- PL meta
  - `operational_mode=t10_only`
  - `feature_count=20`
  - final odds / entity raw ID は未混入

PL で実際に使った特徴量:

```text
p_win_lgbm
p_win_xgb
p_win_cat
p_place_lgbm
p_place_xgb
p_place_cat
p_win_odds_t10_norm_cal_isotonic
p_win_odds_t10_norm_cal_logreg
odds_win_t10
odds_t10_data_kbn
p_win_odds_t10_raw
p_win_odds_t10_norm
field_size
surface
distance_m
going
apt_same_distance_top3_rate_2y
apt_same_going_top3_rate_2y
rel_lag1_speed_index_z
rel_meta_tm_score_z
```

## 3. binary 再学習結果

### 3.1 単勝

| model | CV logloss | CV auc | 2025 holdout logloss | 2025 holdout auc |
|---|---:|---:|---:|---:|
| win_lgbm | 0.20979 | 0.81874 | 0.20807 | 0.80719 |
| win_xgb  | 0.20916 | 0.81990 | 0.20806 | 0.80660 |
| win_cat  | 0.20757 | 0.82392 | 0.20754 | 0.80713 |

### 3.2 複勝

| model | CV logloss | CV auc | 2025 holdout logloss | 2025 holdout auc |
|---|---:|---:|---:|---:|
| place_lgbm | 0.42084 | 0.79424 | 0.43235 | 0.76918 |
| place_xgb  | 0.42071 | 0.79439 | 0.42955 | 0.77307 |
| place_cat  | 0.41772 | 0.79793 | 0.42857 | 0.77476 |

### 3.3 所見

- win は final odds / entity raw ID を外した current contract でも CV AUC が `0.819-0.824` を維持し、holdout AUC も `0.806-0.807` で大崩れしていない。
- place は CV から holdout で AUC が `0.02` 前後落ちており、2025 shift は win より強い。
- current default では win/place ともに `cat` が最良。holdout では place `xgb` も近い。

## 4. odds 校正結果

| variant | logloss | brier | auc | ece |
|---|---:|---:|---:|---:|
| p_win_odds_t10_norm_cal_isotonic | 0.20779 | 0.05807 | 0.82533 | 0.00487 |
| p_win_odds_t10_norm_cal_logreg   | 0.21807 | 0.05941 | 0.82582 | 0.01366 |
| p_win_odds_final_norm_cal_isotonic | 0.20611 | 0.05770 | 0.82806 | 0.00503 |
| p_win_odds_final_norm_cal_logreg   | 0.21568 | 0.05870 | 0.82856 | 0.01237 |

所見:
- operational t10 path では `p_win_odds_t10_norm_cal_isotonic` が最有力。
- final odds 校正は数値上は良いが、validation-only であり運用既定には使わない。

## 5. PL / top3 結果

| 指標 | 値 |
|---|---:|
| CV `pl_nll_valid(mean)` | 23.73331 |
| CV `top3_logloss(mean)` | 0.43174 |
| CV `top3_auc(mean)` | 0.79298 |
| 2025 holdout `top3_logloss` | 0.43718 |
| 2025 holdout `top3_auc` | 0.77005 |

所見:
- current contract の 20 列でも PL は成立している。
- holdout top3 AUC は CV より落ちるが、binary と同様に 2025 側の分布変化が見える。
- `torch` 未導入で NumPy fallback なので、PL の絶対値は学習 backend 依存の余地がある。

## 6. wide ROI 結果

| 評価 | 入力 | 年 | ROI | bets |
|---|---|---:|---:|---:|
| OOF current | `pl_v3_wide_oof.parquet` | 2024 | 0.0050 | 1317 |
| holdout current | `pl_v3_holdout_2025_pred.parquet` | 2025 | 0.3816 | 1705 |

所見:
- default `train-window-years=3` の PL OOF は `valid_year=2024` のみで、旧 report の `w2` OOF（2023-2024）と直接比較しない方がよい。
- holdout ROI `0.3816` は旧 baseline より良いが、単年・thresholdなし・DB依存のため過信しない。
- ROI を主評価にするなら、`train-window-years=2` を current contract でも再計測し、2023-2024 OOF と横並び比較するのが次の自然な手順。

## 7. 総合考察

1. feature governance hardening により、train-serve mismatch と accidental leakage のリスクを下げつつ、binary の識別性能は大きく崩れていない。
2. operational default としては、binary は `cat`、odds 校正は t10 isotonic、PL は current explicit-list のまま維持してよい。
3. ただし ROI は PL の学習窓と評価年の取り方に敏感で、default `w=3` の OOF だけで良否を決めない方がよい。
4. 次の改善候補は feature 数追加ではなく、`train-window-years` の再検討、`torch` 導入時の PL 再学習、ROI 閾値設計の再検証。

## 8. 実行コマンド

```bash
uv run python scripts_v3/build_features_v3.py
uv run python scripts_v3/build_features_v3.py --input data/features_v2_2025.parquet --output data/features_v3_2025.parquet --meta-output data/features_v3_2025_meta.json

uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_win_xgb_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet

uv run python scripts_v3/train_odds_calibrator_v3.py
uv run python scripts_v3/train_pl_v3.py --emit-wide-oof

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_wide_oof.parquet \
  --holdout-year 2025 \
  --output data/backtest_v3/backtest_wide_v3_direct_oof_current.json \
  --meta-output data/backtest_v3/backtest_wide_v3_direct_oof_current_meta.json \
  --force

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_holdout_2025_pred.parquet \
  --years 2025 \
  --holdout-year 2026 \
  --output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_current.json \
  --meta-output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_current_meta.json \
  --force
```

## 9. 注意点

- 2026-03-04 の旧 report とは feature contract も一部評価設定も違うため、数値をそのまま横比較しない。
- `docs/ops_v3/v3_run_report_2026-03-04.md` は履歴として残し、この文書を current contract の結果として扱う。
- ROI 系の値は DB の odds / payout データ更新で再計測時に変動しうる。
