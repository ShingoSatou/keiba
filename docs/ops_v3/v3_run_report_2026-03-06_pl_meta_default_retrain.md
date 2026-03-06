# v3 再学習レポート（2026-03-06, PL meta_default 契約移行）

## 0. 目的

PL の default 入力を raw 6 本直結から `meta_default` へ切り替えた。

- 旧 default:
  - `p_win_lgbm`
  - `p_win_xgb`
  - `p_win_cat`
  - `p_place_lgbm`
  - `p_place_xgb`
  - `p_place_cat`
- 新 default:
  - `p_win_meta`
  - `p_place_meta`
  - `p_win_odds_t10_norm`
  - `PL_CONTEXT_FEATURES_SMALL`

狙い:

1. PL 入力の冗長性を下げる
2. raw 6 本の多重共線性リスクを下げる
3. 前段で stacking した整理済み信号を PL に渡す
4. `raw_legacy` profile で旧ルートとの比較可能性を残す

## 1. 評価ポリシー

- binary / PL の outer evaluation policy は現行どおり維持する。
  - `train_window_years=4`
  - `holdout_year=2025`
  - `cv_window_policy=fixed_sliding`
- operational default は引き続き `t10_only`
- final odds は default では使わない

本実装の `p_win_meta` / `p_place_meta` は、base 予測の結合器として grouped 通常CVで生成した OOF を用いている。  
これらは strict temporal OOF ではないため、meta 層の内部CV指標は参考値であり、fixed4 時系列CVの指標と同列には扱わない。  
最終的なモデル採否は、2025 holdout・PL holdout・wide holdout の結果を最重視する。  
binary / PL の outer evaluation policy と t10_only operational constraint は維持する。

## 2. 実行手順

```bash
uv run python scripts_v3/build_features_v3.py
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

uv run python scripts_v3/train_win_meta_v3.py --holdout-input data/features_v3_2025.parquet
uv run python scripts_v3/train_place_meta_v3.py --holdout-input data/features_v3_2025.parquet

uv run python scripts_v3/train_pl_v3.py \
  --pl-feature-profile meta_default \
  --holdout-input data/features_v3_2025.parquet \
  --emit-wide-oof

uv run python scripts_v3/train_pl_v3.py \
  --pl-feature-profile raw_legacy \
  --holdout-input data/features_v3_2025.parquet \
  --emit-wide-oof

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_wide_oof.parquet \
  --years 2024 \
  --holdout-year 2025 \
  --output data/backtest_v3/backtest_wide_v3_meta_default_oof_2024.json \
  --meta-output data/backtest_v3/backtest_wide_v3_meta_default_oof_2024_meta.json \
  --force

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_holdout_2025_pred.parquet \
  --years 2025 \
  --holdout-year 2026 \
  --output data/backtest_v3/backtest_wide_v3_meta_default_holdout_2025.json \
  --meta-output data/backtest_v3/backtest_wide_v3_meta_default_holdout_2025_meta.json \
  --force

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_wide_oof_raw_legacy.parquet \
  --years 2024 \
  --holdout-year 2025 \
  --output data/backtest_v3/backtest_wide_v3_raw_legacy_oof_2024.json \
  --meta-output data/backtest_v3/backtest_wide_v3_raw_legacy_oof_2024_meta.json \
  --force

uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_holdout_2025_pred_raw_legacy.parquet \
  --years 2025 \
  --holdout-year 2026 \
  --output data/backtest_v3/backtest_wide_v3_raw_legacy_holdout_2025.json \
  --meta-output data/backtest_v3/backtest_wide_v3_raw_legacy_holdout_2025_meta.json \
  --force
```

## 3. 生成される主成果物

- `data/oof/win_meta_oof.parquet`
- `data/oof/place_meta_oof.parquet`
- `data/holdout/win_meta_holdout_pred_v3.parquet`
- `data/holdout/place_meta_holdout_pred_v3.parquet`
- `models/win_meta_v3.pkl`
- `models/place_meta_v3.pkl`
- `data/oof/pl_v3_oof.parquet`
- `data/oof/pl_v3_holdout_2025_pred.parquet`
- `models/pl_v3_bundle_meta.json`
- `models/pl_v3_bundle_meta_raw_legacy.json`
- `data/backtest_v3/backtest_wide_v3_meta_default_oof_2024.json`
- `data/backtest_v3/backtest_wide_v3_meta_default_holdout_2025.json`
- `data/backtest_v3/backtest_wide_v3_raw_legacy_oof_2024.json`
- `data/backtest_v3/backtest_wide_v3_raw_legacy_holdout_2025.json`

## 4. 特徴量・binary・meta の検証結果

### 4.1 binary 2025 holdout

| task | model | logloss | brier | auc | ece |
|---|---|---:|---:|---:|---:|
| win | lgbm | 0.210448 | 0.057095 | 0.799010 | 0.010110 |
| win | xgb | 0.209372 | 0.056803 | 0.801939 | 0.007685 |
| win | cat | **0.207773** | **0.056507** | 0.805295 | **0.004350** |
| place | lgbm | 0.434271 | 0.138870 | 0.766301 | 0.019770 |
| place | xgb | 0.431576 | 0.137904 | 0.769599 | **0.013979** |
| place | cat | **0.428915** | **0.136796** | **0.774581** | 0.016381 |

### 4.2 meta reference / 2025 holdout

- `p_win_meta`
  - reference CV mean: logloss `0.218189`, brier `0.059756`, auc `0.818218`, ece `0.012250`
  - 2025 holdout: logloss `0.211562`, brier `0.056939`, auc `0.806996`, ece `0.009180`
- `p_place_meta`
  - reference CV mean: logloss `0.425855`, brier `0.135680`, auc `0.794301`, ece `0.029273`
  - 2025 holdout: logloss `0.434501`, brier `0.138312`, auc `0.773034`, ece `0.024975`

補足:
- win meta は win-cat より AUC は僅かに高いが、holdout logloss は win-cat に届かなかった。
- place meta は holdout で place-cat / place-xgb を上回れなかった。

## 5. PL 比較結果

PL では `meta_default` / `raw_legacy` とも、利用可能な strict yearly OOF の都合で fixed4 outer CV は `2024` の 1 fold のみになった。

### 5.1 OOF 2024

| profile | pl_nll | top3_logloss | top3_brier | top3_auc | top3_ece |
|---|---:|---:|---:|---:|---:|
| meta_default | 23.765847 | 0.434817 | 0.138116 | 0.788664 | 0.045190 |
| raw_legacy | **23.730307** | **0.431616** | **0.137545** | **0.795112** | 0.048282 |

### 5.2 2025 holdout

| profile | top3_logloss | top3_brier | top3_auc | top3_ece |
|---|---:|---:|---:|---:|
| meta_default | 0.439790 | 0.139563 | 0.764378 | 0.033317 |
| raw_legacy | **0.437219** | **0.139015** | **0.769534** | **0.030912** |

## 6. Wide ROI 比較

### 6.1 OOF 2024

| profile | races | bets | hits | roi | max_drawdown |
|---|---:|---:|---:|---:|---:|
| meta_default | 342 | 1167 | 2 | **0.0038** | 983020 |
| raw_legacy | 342 | 1331 | 0 | 0.0000 | 974200 |

### 6.2 2025 holdout

| profile | races | bets | hits | roi | max_drawdown |
|---|---:|---:|---:|---:|---:|
| meta_default | 344 | 1411 | 4 | 0.1179 | 975620 |
| raw_legacy | 344 | 1612 | 6 | **0.1829** | 949960 |

## 7. 評価判断

- 実装自体は成立し、`meta_default` / `raw_legacy` の両 profile で full retrain と holdout / wide 評価を完了した。
- ただし今回の 2025 holdout では、PL 指標・wide ROI ともに `raw_legacy` が `meta_default` を上回った。
- したがって、コード上の default contract 切替は実装済みだが、運用採否の観点では `meta_default` を即採用と判断する材料は不足している。
- 次に見るべき論点は、`p_place_meta` の改善、meta 学習器の再検討、あるいは meta 入力と odds/context の組み合わせ見直しである。

## 8. 注意点

- meta JSON / metrics JSON には `cv_is_temporal=false`, `group_key=race_id`, `meta_metrics_are_reference_only=true` を残した。
- meta OOF は strict temporal OOF ではないため、reference-only 指標として扱う。
- `torch` 未導入環境のため、PL 学習は仕様どおり NumPy optimizer fallback で実行した。
