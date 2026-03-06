# v3 実行結果レポート（2026-03-04）

## 0. 目的

v3の実装（単勝/複勝分類、odds確率・校正、PL(uなし)、ワイドROI評価）について、
実行結果（指標・ROI）と再現コマンドを記録する。

> [!IMPORTANT]
> このレポートの数値結果は 2026-03-04 時点の計測であり、feature governance hardening 前の run を記録している。
> 現在の contract（binary/PL の feature registry, binary default `t10_only`, entity ID default OFF, PL explicit-list 化）で同等の数値を確認するには再学習が必要。

## 1. データ範囲

- 学習/CV対象（trainable）: 2016–2024（`holdout_year=2025` で除外）
  - `data/features_v3.parquet`: 45,444行 / 3,188レース
- 2025評価用:
  - `data/features_v3_2025.parquet`: 4,971行 / 344レース

## 2. 実装コンポーネント（概要）

- v3特徴量: `scripts_v3/build_features_v3.py`
- 単勝分類（`y_win`）: `scripts_v3/train_win_{lgbm,xgb,cat}_v3.py`
- 複勝分類（`y_place`）: `scripts_v3/train_place_{lgbm,xgb,cat}_v3.py`
- odds校正（任意）: `scripts_v3/train_odds_calibrator_v3.py`
- PL(uなし): `scripts_v3/train_pl_v3.py`
- 1レース推論: `scripts_v3/predict_race_v3.py`
- feature contract: `scripts_v3/feature_registry_v3.py`

## 3. 単勝分類（OOF）

CV設定:
- Rolling年次（v2踏襲）
- `train_window_years=5`, `holdout_year=2025`

指標（平均, valid=2021–2024）:

| model | logloss | brier | auc | ece | Benter R* (β̂) | Benter R* (β=1) | β̂ mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| win_lgbm | 0.21494 | 0.05946 | 0.80556 | 0.00823 | -0.20369 | 0.19779 | 2.690 |
| win_xgb  | 0.20876 | 0.05849 | 0.82143 | 0.00616 | 0.16043 | 0.22766 | 1.609 |
| win_cat  | 0.20679 | 0.05792 | 0.82600 | 0.00471 | 0.21002 | 0.23441 | 1.290 |

出力:
- `data/oof/win_{lgbm,xgb,cat}_oof.parquet`
- `data/oof/win_{lgbm,xgb,cat}_cv_metrics.json`
- `models/win_{lgbm,xgb,cat}_bundle_meta_v3.json`
- `models/win_{lgbm,xgb,cat}_feature_manifest_v3.json`

現仕様メモ:
- binary は feature registry の whitelist を使用
- default operational profile は `t10_only`
- `jockey_key` / `trainer_key` は default では学習に入らない
- final odds は `--operational-mode includes_final` 指定時だけ学習投入する

## 4. 複勝分類（OOF）

CV設定:
- Rolling年次（v2踏襲）
- `train_window_years=5`, `holdout_year=2025`

指標（平均, valid=2021–2024）:

| model | logloss | brier | auc | ece |
|---|---:|---:|---:|---:|
| place_lgbm | 0.42791 | 0.13693 | 0.78495 | 0.01273 |
| place_xgb  | 0.41932 | 0.13407 | 0.79628 | 0.01242 |
| place_cat  | 0.41640 | 0.13321 | 0.80007 | 0.01110 |

出力:
- `data/oof/place_{lgbm,xgb,cat}_oof.parquet`
- `data/oof/place_{lgbm,xgb,cat}_cv_metrics.json`
- `models/place_{lgbm,xgb,cat}_bundle_meta_v3.json`
- `models/place_{lgbm,xgb,cat}_feature_manifest_v3.json`

## 5. odds校正（OOF, 任意）

対象:
- `p_win_odds_t10_norm` / `p_win_odds_final_norm` を `y_win` に対して校正
- 手法: `logreg` / `isotonic`

指標（平均）:

| variant | logloss | brier | auc | ece |
|---|---:|---:|---:|---:|
| p_win_odds_t10_norm_cal_isotonic | 0.20779 | 0.05807 | 0.82533 | 0.00487 |
| p_win_odds_t10_norm_cal_logreg   | 0.21807 | 0.05941 | 0.82582 | 0.01366 |
| p_win_odds_final_norm_cal_isotonic | 0.20611 | 0.05770 | 0.82806 | 0.00503 |
| p_win_odds_final_norm_cal_logreg   | 0.21568 | 0.05870 | 0.82856 | 0.01237 |

出力:
- `data/oof/odds_win_calibration_oof.parquet`
- `data/oof/odds_win_calibration_cv_metrics.json`
- `models/odds_win_calibrators_v3.pkl`

## 6. PL(uなし)（OOF）

入力（概略）:
- 単勝/複勝のOOF予測（`p_win_*`, `p_place_*`）
- `feature_registry_v3.py` で定義した `t10 odds + small context`
- final odds は `--include-final-odds-features` 指定時のみ追加

備考:
- `torch` 未導入のため、学習は NumPy最適化フォールバックで実行（損失関数は同一）。
- PL は `features_v3` の numeric 列を自動走査しない。

`train_window_years=2`（valid=2023–2024 のOOFを確保）:
- `data/oof/pl_v3_oof_w2.parquet`: 9,805行 / 694レース

指標（平均）:
- `pl_nll_valid(mean)=23.83174`
- `top3_logloss(mean)=0.43200` / `top3_auc(mean)=0.78759`

出力:
- `data/oof/pl_v3_oof_w2.parquet`
- `data/oof/pl_v3_wide_oof_w2.parquet`
- `data/oof/pl_v3_cv_metrics_w2.json`
- `models/pl_v3_recent_window_w2.joblib`

## 7. ワイドROI

### 7.1 推奨: v3 backtest（`pl_score -> MC -> p_wide` / `p_wide` 直接入力）

ROI算出は v3の `scripts_v3/backtest_wide_v3.py` を利用。
- ワイドのオッズ: `core.o3_wide`
- ワイドの払戻: `core.payout (bet_type=5)`

OOF（2023–2024, pair-level `data/oof/pl_v3_wide_oof_w2.parquet`）:
- ROI `0.0948`（bets=1513）
- `data/backtest_v3/backtest_wide_v3_direct_w2_2023_2024.json`

holdout（2025, horse-level `data/oof/pl_v3_holdout_2025_pred.parquet`）:
- ROI `0.0617`（bets=1427）
- `data/backtest_v3/backtest_wide_v3_direct_holdout_2025.json`

同一期間内での `min_p_wide` スイープ（参考, holdout内最適化のため過学習リスクあり）:
- 対象年は 2025 のみ（`selected_years=[2025]`, `--years 2025`, `--holdout-year 2026`）
- `candidate min_p_wide=0.11`: ROI `0.3624`（bets=77）
- `candidate min_p_wide=0.12`: ROI `0.4064`（bets=59）
- `candidate min_p_wide=0.14`: ROI `0.4516`（bets=34）
- `candidate min_p_wide=0.15`: ROI `0.5952`（bets=27）
- `selected min_p_wide=0.11`: ROI `0.8206`（bets=11）
- `selected min_p_wide=0.12`: ROI `1.1355`（bets=8）
- `selected min_p_wide=0.14`: ROI `1.2287`（bets=7）
- `selected min_p_wide=0.15`: ROI `1.6151`（bets=5）

### 7.2 参考: 旧v2近似（`p_top3 -> p/(1-p) -> p_wide`）

ROI算出は v2の `scripts_v2/backtest_wide_v2.py` を利用。

OOF（2023–2024, `pl_v3_oof_w2.parquet`）:

閾値なし:
- ROI `0.0810`（bets=1406）
- `data/backtest_v3/backtest_wide_v3_pl_w2_2023_2024.json`

同一期間内での閾値スイープ（参考, 期待値過学習リスクあり）:
- `candidate min_p_wide=0.14`: ROI `1.0430`（bets=373）
- `selected min_p_wide=0.15`: ROI `1.1743`（bets=48）

よりリークを抑えた参考（2023で閾値決定→2024で評価）:
- `candidate min_p_wide=0.12`: ROI `0.8880`（bets=263）
- `selected min_p_wide=0.15`: ROI `0.8581`（bets=24）

2025評価（holdout）:

2025の `p_top3` は以下を作成して入力:
- `data/oof/pl_v3_holdout_2025_pred.parquet`（4,971行 / 344レース）

結果（全てROI<1）:
- 閾値なし: ROI `0.0273`（bets=1101）
- `candidate min_p_wide=0.11`: ROI `0.3687`（bets=240）
- `candidate min_p_wide=0.14`: ROI `0.4559`（bets=138）

## 8. 再現コマンド（抜粋）

```bash
# 1) v3 features（2016-2024）
uv run python scripts_v3/build_features_v3.py

# 2) win/place（6本）
uv run python scripts_v3/train_win_lgbm_v3.py
uv run python scripts_v3/train_win_xgb_v3.py
uv run python scripts_v3/train_win_cat_v3.py
uv run python scripts_v3/train_place_lgbm_v3.py
uv run python scripts_v3/train_place_xgb_v3.py
uv run python scripts_v3/train_place_cat_v3.py

# binary defaults:
# - operational_mode=t10_only
# - entity raw ID features are OFF unless --include-entity-id-features

# 3) odds校正（任意）
uv run python scripts_v3/train_odds_calibrator_v3.py

# 4) PL（OOF=2023-2024）
uv run python scripts_v3/train_pl_v3.py --train-window-years 2 \
  --oof-output data/oof/pl_v3_oof_w2.parquet \
  --wide-oof-output data/oof/pl_v3_wide_oof_w2.parquet \
  --emit-wide-oof \
  --metrics-output data/oof/pl_v3_cv_metrics_w2.json \
  --model-output models/pl_v3_recent_window_w2.joblib

# PL defaults:
# - required OOF + t10 odds + small context only
# - final odds are OFF unless --include-final-odds-features

# 5) ワイドROI（推奨: v3 backtest, OOF 2023-2024）
uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_wide_oof_w2.parquet \
  --years 2023,2024 \
  --holdout-year 2025 \
  --output data/backtest_v3/backtest_wide_v3_direct_w2_2023_2024.json \
  --meta-output data/backtest_v3/backtest_wide_v3_direct_w2_2023_2024_meta.json \
  --force

# 5b) ワイドROI（推奨: v3 backtest, holdout 2025 baseline）
uv run python scripts_v3/backtest_wide_v3.py \
  --input data/oof/pl_v3_holdout_2025_pred.parquet \
  --years 2025 \
  --holdout-year 2026 \
  --output data/backtest_v3/backtest_wide_v3_direct_holdout_2025.json \
  --meta-output data/backtest_v3/backtest_wide_v3_direct_holdout_2025_meta.json \
  --force

# 5c) ワイドROI（推奨: v3 backtest, holdout 2025 min_p_wide sweep）
for stage in candidate selected; do
  for thr in 0.11 0.12 0.14 0.15; do
    tag=$(echo "$thr" | tr -d '.')
    uv run python scripts_v3/backtest_wide_v3.py \
      --input data/oof/pl_v3_holdout_2025_pred.parquet \
      --years 2025 \
      --holdout-year 2026 \
      --min-p-wide "$thr" \
      --min-p-wide-stage "$stage" \
      --output "data/backtest_v3/backtest_wide_v3_direct_holdout_2025_minp${tag}_${stage}.json" \
      --meta-output "data/backtest_v3/backtest_wide_v3_direct_holdout_2025_minp${tag}_${stage}_meta.json" \
      --force
  done
done

# 6) ワイドROI（参考: 旧v2近似）
uv run python scripts_v2/backtest_wide_v2.py \
  --input data/oof/pl_v3_oof_w2.parquet \
  --years 2023,2024 \
  --holdout-year 2025 \
  --output data/backtest_v3/backtest_wide_v3_pl_w2_2023_2024.json \
  --meta-output data/backtest_v3/backtest_wide_v3_pl_w2_2023_2024_meta.json \
  --force
```

## 9. 注意点（重要）

- `predict_race_v3.py` は運用 t10 path を前提とし、PL 側で final-odds 特徴量を禁止している。
- binary / PL ともに feature registry を使うため、新しい numeric 列を `features_v3` に追加しても学習列は自動では増えない。
- final odds は `features_v3` に保持されるが、明示 opt-in なしでは学習投入しない。
- 2025評価は実施済みのため、「封印holdout（one-shot）」としての性質は以後弱まる（仕様書運用に従う場合は封印年を更新する）。
- contract test は `test_v3/test_feature_registry_v3.py`, `test_v3/test_binary_feature_contract_v3.py`, `test_v3/test_pl_feature_contract_v3.py` を参照。
