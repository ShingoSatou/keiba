# Ranker v2 stacking：2025 holdout（one-shot）評価メモ

## 1. 目的

- Ranker v2 を 3モデル（LightGBM / XGBoost / CatBoost）で stacking したときの、2025 holdout（NDCG@3）を確認する
- 「どのメタ方式を使うべきか」を、**one-shot の結果を踏まえて意思決定できる状態にする**

> 注意：2025 は封印（holdout）であり、原則として 1 回だけ開封する。  
> 2025 を見て設計を変えた場合は、封印期間を 2026 以降へ更新して再度 one-shot を行う（評価汚染回避）。

## 2. 前提（評価条件）

- 指標: **NDCG@3**
- 分割単位: `race_id`（同一レースが跨らない）
- base 3モデルは Rolling CV（valid=2021–2024）で作成した OOF を使用
- holdout: **2025**
- 特徴量セット: `base` / `te`
  - v2の実装では `models/*_bundle_meta.json` の `input_path` を参照し、モデルごとに `base/te` を切り替える
    - LightGBM: `te`
    - XGBoost: `te`
    - CatBoost: `base`

## 3. 実行手順（2025 one-shot）

### 3.1 2025特徴量（base / te）

```bash
uv run python scripts_v2/build_features_v2.py \
  --from-date 2025-01-01 \
  --to-date 2025-12-31 \
  --output data/features_v2_2025.parquet \
  --meta-output data/features_v2_2025_meta.json

uv run python scripts_v2/build_features_v2.py \
  --from-date 2025-01-01 \
  --to-date 2025-12-31 \
  --with-te \
  --output data/features_v2_te_2025.parquet \
  --meta-output data/features_v2_te_2025_meta.json
```

### 3.2 メタ方式の横並び比較（推奨）

```bash
uv run python scripts_v2/eval_ranker_stacker_holdout_compare_v2.py \
  --year 2025 \
  --features-base data/features_v2_2025.parquet \
  --features-te data/features_v2_te_2025.parquet \
  --preds-output data/holdout/ranker_stack_2025_compare_preds.parquet \
  --metrics-output data/holdout/ranker_stack_2025_compare_metrics.json
```

## 4. 結果（2026-02-27）

2025 holdout（`344` レース / `4,971` 行）の NDCG@3。

### 4.1 base（単体）

- `xgb`: **0.481491**
- `lgbm`: 0.477957
- `cat`: 0.475639

### 4.2 stacking（メタ方式別）

- `convex`: **0.483300**（best）
- `ridge`: 0.479149
- `logreg_multiclass`: 0.478234
- `lgbm_ranker`: 0.478096

convex weights（`[lgbm,xgb,cat]`）:
- `[0.020040, 0.529749, 0.450210]`

出力ファイル:
- 指標: `data/holdout/ranker_stack_2025_compare_metrics.json`
- 予測: `data/holdout/ranker_stack_2025_compare_preds.parquet`

## 5. 考察（なぜ convex が良かったか）

- **単体ベストは XGB**だが、`cat` を混ぜた `convex` が上回った  
  - CatBoost は `base` 特徴量、XGB/LGBM は `te` 特徴量で学習しており、学習条件が異なるため多様性が出やすい
- **metaの入力を percentile に寄せた**ことで、スコアスケールの差や年次のスケール変動に頑健になりやすい
- ridge/logreg/lgbm_ranker が伸びなかったのは、meta特徴量が少数（percentile派生）である一方、
  - 過度な当て込み（特にlgbm_ranker）や
  - 回帰/分類の目的（`target_label`）と NDCG の整合性のズレ
  が影響した可能性がある

## 6. 推奨（どれを使うべきか）

**結論: `convex` を推奨**（2025 holdout で最高、かつ実装がシンプルで過学習リスクが小さい）。

- 下流（校正・ワイド確率）の入力となる「大元スコア」は、まず `convex` stacking をデフォルトとする
- 追加の改善（特徴量/モデル/メタ方式変更など）を行う場合は、評価汚染を避けるため **封印期間を 2026 以降へ更新**して one-shot をやり直す

### 6.1 運用上のメモ（convex を採用する場合）

`train_ranker_stacker_v2.py` は `--method` で方式を固定できるため、選抜方式に依存せず `convex` のメタモデル（+メタ情報）を作成できる。

```bash
uv run python scripts_v2/train_ranker_stacker_v2.py \
  --best-config data/oof/ranker_stack_optuna_best.json \
  --method convex \
  --train-years 2021,2022,2023,2024 \
  --model-output models/ranker_stack_meta.model \
  --meta-output models/ranker_stack_bundle_meta.json
```

※`models/` はGit管理外のため、PRには含めない。

