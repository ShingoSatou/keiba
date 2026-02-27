## 目的 / 背景
- Ranker v2 の推論強化のため、3モデル（LightGBM / XGBoost / CatBoost）の stacking を導入する
- Rolling CV（valid=2021-2024, holdout=2025）とリークなしのメタ学習手順を固定し、2025で one-shot 評価する

## 変更内容
### 1) Base ranker（XGB/Cat）学習スクリプト追加
- `scripts_v2/train_ranker_xgb_v2.py` / `scripts_v2/train_ranker_cat_v2.py`
  - LightGBM と同一の Rolling 設計で OOF を生成
  - OOF列仕様を揃え、stacking 入力（score/rank/percentile）に利用可能にする

### 2) Stacking（メタ学習）実装
- `scripts_v2/ranker_stacking_v2_common.py`
  - 3モデルOOFの結合、メタ特徴量（percentile派生）生成、NDCG@3評価の共通処理
- `scripts_v2/tune_ranker_stacker_optuna_v2.py`
  - tune_years（既定: 2021-2023）で各メタ方式を Optuna tuning
  - select_year（既定: 2024）で方式選抜
- `scripts_v2/train_ranker_stacker_v2.py`
  - 選抜方式で固定パラメータの walk-forward OOF（2022-2024）を生成し、最終メタモデルを保存

### 3) Holdout 評価（2025）
- `scripts_v2/eval_ranker_stacker_holdout_v2.py`
  - `models/ranker_stack_meta.model`（選抜済みメタモデル）で holdout を評価
- `scripts_v2/eval_ranker_stacker_holdout_compare_v2.py`
  - holdout 年でメタ方式を横並び比較（one-shot、上書きは `--force` が必要）

### 4) テスト追加
- `test_v2/test_ranker_stacker_v2_common.py`

### 5) ドキュメント更新
- `docs/ops_v2/スクリプトリファレンス.md`
  - stacking 学習/評価スクリプト、2025評価手順を追記
- `docs/ops_v2/Rankerスタッキング_2025評価.md`
  - 2025 holdout（one-shot）の結果・考察・推奨方式を整理
- `docs/specs_v2/時系列交差検証およびモデル評価仕様書.md`
  - holdout成果物と2025 one-shot（NDCG@3）結果を追記
- `docs/ops_v2/TODO.md` / `docs/ops_v2/実装計画.md`
  - stacking 実装分を反映

## 2025 holdout NDCG@3（one-shot, 2026-02-27）
- base: `xgb=0.481491` / `lgbm=0.477957` / `cat=0.475639`
- stacking: `convex=0.483300`（best） / `ridge=0.479149` / `logreg_multiclass=0.478234` / `lgbm_ranker=0.478096`

## 動作確認（実施済み）
- [x] `uv run ruff check .`
- [x] `uv run ruff format --check .`
- [x] `uv run pytest -q`

## 影響範囲
- v2 の Ranker 学習/評価（stacking含む）：`scripts_v2/*`
- v2 テスト：`test_v2/*`
- v2 ドキュメント：`docs/ops_v2/*`, `docs/specs_v2/*`

## 注意点
- `data/` / `models/` はGit管理外（評価結果は `data/holdout/` に保存）
- holdout（2025）は one-shot 運用前提のため、`--force` による再実行は慎重に扱う
