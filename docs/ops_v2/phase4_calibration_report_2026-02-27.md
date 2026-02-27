# Phase 4 Calibration レポート（2026-02-27）

本ドキュメントは、Phase 4（Top3確率化＝Calibration）の実験結果と、運用上の推奨設定を記録する。

## 目的・範囲

- 目的: ranker v2 のスコア（OOFのみ）を `p_top3 = P(3着以内)` に変換し、下流（ワイド確率 / EV / 資金配分）へ渡す。
- 制約:
  - 時系列 walk-forward（未来学習禁止）
  - 分割単位は `race_id`（同一レースが train/valid に跨らない）
  - 入力は OOF由来のスコアのみ（生特徴量からの再学習は禁止）
  - 2025は holdout（one-shot評価）。チューニングや運用調整には使わない

## データ / 入力

- stacking OOF（本命）: `data/oof/ranker_stack_oof.parquet`（valid_year=2022-2024, method=lgbm_ranker）
- convex比較用 OOF: `data/oof/ranker_stack_convex_oof.parquet`（valid_year=2022-2024, method=convex）
- holdout preds（stack方式の比較出力）: `data/holdout/ranker_stack_2025_compare_preds.parquet`
  - 評価に使うスコア列:
    - lgbm_ranker系 stacking score: `stack_lgbm_ranker_score`
    - convex系 stacking score: `stack_convex_score`

## 校正モデル（Calibration）

- 目的変数: `is_top3 = 1(target_label > 0) else 0`
- 入力特徴量（レース内集計。OOFスコアが揃ってから算出）:
  - `percentile_rank`, `z_score`, `field_size`, `score_diff_from_top`, `gap_1st_to_3rd`
- モデル: LogisticRegression + StandardScaler（特徴量エンジニアリング付き Platt Scaling）
- walk-forward校正OOF:
  - OOF年が [2022, 2023, 2024] の場合、校正OOF年は [2023, 2024]（2022は学習用）

## 実験条件

比較軸は2つ:

1. 入力スコア（stacking後スコア）
   - lgbm_ranker系 stacking score
   - convex系 stacking score
2. `class_weight`
   - `balanced`
   - `None`（クラス重みなし）

注: `class_weight='balanced'` は「クラス事前確率を重み付けした世界」での推定になり、出力をそのまま確率として読むと経験的な base_rate と整合しない（事前確率補正が必要になりやすい）。

## 結果（OOF 2023-2024）

データ: 694レース / 9,805行

| 入力スコア | class_weight | Logloss | Brier | ECE | mean(p_top3) | mean(レース内 Σp_top3) |
|---|---:|---:|---:|---:|---:|---:|
| lgbm_ranker | balanced | 0.590771 | 0.201264 | 0.239834 | 0.4523 | 6.3899 |
| lgbm_ranker | None | 0.439083 | 0.140470 | 0.014475 | 0.2172 | 3.0682 |
| convex | balanced | 0.577557 | 0.197912 | 0.226153 | 0.4386 | 6.1966 |
| convex | None | 0.438091 | 0.140578 | 0.011246 | 0.2127 | 3.0045 |

解釈:

- `class_weight=None` で Logloss/Brier/ECE が大幅改善し、確率スケールが自然になる。
- `mean(レース内 Σp_top3) ~ 3` は sanity check として有効（各レースのTop3枠は必ず3つ）。
- 今回は convex が僅差で良いが、差は小さい。

## 結果（2025 holdout / one-shot）

データ: 344レース / 4,971行（base_rate=0.2080）

| 入力スコア | class_weight | Logloss | Brier | ECE | mean(p_top3) | mean(レース内 Σp_top3) |
|---|---:|---:|---:|---:|---:|---:|
| lgbm_ranker | balanced | 0.587622 | 0.201551 | 0.229383 | 0.4374 | 6.3205 |
| lgbm_ranker | None | 0.445237 | 0.141529 | 0.017351 | 0.2085 | 3.0123 |
| convex | balanced | 0.579583 | 0.199206 | 0.222485 | 0.4305 | 6.2209 |
| convex | None | 0.443197 | 0.141616 | 0.015074 | 0.2073 | 2.9955 |

解釈:

- OOFと同様に、`balanced` は過大確率になり、Reliabilityが悪化しやすい。
- `class_weight=None` は 2025でも確率スケールが自然（mean(p_top3) が base_rate に整合し、レース内合計も `~3`）。
- convex + `None` が僅差で良いが、差は小さい。

## Reliability（2025のスナップショット）

ビン別のテーブルは JSON 出力を参照（`summary.reliability`）。要点:

- `class_weight=balanced`: 多くのビンで `mean_pred >> frac_pos`（体系的な過大化）。
- `class_weight=None`: 対角線に近く、全体として良好。

## 推奨（運用デフォルト）

- `p_top3` を確率として下流（EV/Kelly）で使うなら、デフォルトは `class_weight=None`。
- `balanced` を使う場合は、確率として扱う前に事前確率補正（prior correction）を明示的に挟むこと。
- 入力スコアは一貫させる:
  - convexを採用するなら、convex OOFで校正器を学習し、holdoutもconvexスコアで評価する。
  - 選抜した meta method（例: lgbm_ranker）を採用するなら、そのスコアで校正器を統一する。
- 現状は、convexを採用する。

## 再現コマンド（参考）

```bash
# 校正OOF作成（推奨: class_weight=None）
uv run python scripts_v2/train_calibrator_v2.py --class-weight none

# holdout年の評価（one-shot）
uv run python scripts_v2/eval_calibrator_holdout_v2.py --model models/calibrator_cw_none.pkl

# convex比較（出力を分けて実行）
uv run python scripts_v2/train_ranker_stacker_v2.py --method convex \
  --oof-output data/oof/ranker_stack_convex_oof.parquet
uv run python scripts_v2/train_calibrator_v2.py \
  --input data/oof/ranker_stack_convex_oof.parquet --score-col stack_score --class-weight none \
  --oof-output data/oof/top3_convex_oof_cw_none.parquet \
  --metrics-output data/oof/calibration_convex_metrics_cw_none.json \
  --model-output models/calibrator_convex_cw_none.pkl
uv run python scripts_v2/eval_calibrator_holdout_v2.py \
  --model models/calibrator_convex_cw_none.pkl --score-col stack_convex_score
```

## 成果物（パス）

OOF出力:

- lgbm_ranker:
  - `data/oof/top3_oof.parquet` (balanced)
  - `data/oof/top3_oof_cw_none.parquet` (None)
- convex:
  - `data/oof/top3_convex_oof.parquet` (balanced)
  - `data/oof/top3_convex_oof_cw_none.parquet` (None)

2025 holdout出力:

- lgbm_ranker:
  - `data/holdout/calibration_2025_metrics.json` (balanced)
  - `data/holdout/calibration_2025_metrics_cw_none.json` (None)
- convex:
  - `data/holdout/calibration_convex_2025_metrics.json` (balanced)
  - `data/holdout/calibration_convex_2025_metrics_cw_none.json` (None)
