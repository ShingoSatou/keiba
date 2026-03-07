# 2026-03-07_v3-strict-stacker-retrain-eval

## タイトル

- v3 strict temporal stacker の full retrain と評価

## ステータス

- `done`

## 対象範囲

- `data/features_v3*.parquet`
- `data/oof/*`
- `data/holdout/*`
- `data/backtest_v3/*`
- `models/*`
- `docs/ops_v3/v3_run_report_*.md`

## 対象バージョン

- `v3`

## 前提・仮定

- default main path は `base -> stacker -> PL`
- PL fixed3 OOF fold は現行年数だと 0 件のため、評価の主眼は holdout 2025 と stage metrics に置く

## チェックリスト

- [x] features_v3 / features_v3_2025 を再生成する
- [x] base binary 6 本を再学習する
- [x] strict temporal stacker 2 本を再学習する
- [x] PL default path を再学習する
- [x] holdout 2025 の wide backtest を実行する
- [x] metrics / artifact / year coverage を確認する
- [x] 実行結果を run report にまとめる

## 確認結果

- `features_v3.parquet`: `45444` 行 / `3188` レース
- `features_v3_2025.parquet`: `4971` 行 / `344` レース
- binary 6 本、stacker 2 本、PL 1 本の full retrain を完了
- `win_stack_oof.parquet` / `place_stack_oof.parquet` は `valid_year=2022,2023,2024`
- `pl_v3_oof.parquet` は `0` 行、`pl_v3_holdout_2025_pred.parquet` は `4971` 行
- year coverage は `base=2020-2024`, `stacker=2022-2024`, `PL fixed3 OOF=[]`, `PL holdout train=2022-2024`
- holdout 2025 の主要指標:
  - binary best `win_cat`: logloss `0.207773`, auc `0.805295`
  - binary best `place_cat`: logloss `0.428915`, auc `0.774581`
  - `win_stack`: logloss `0.211822`, auc `0.796023`
  - `place_stack`: logloss `0.429378`, auc `0.773059`
  - PL `top3_logloss=0.436188`, `top3_auc=0.772327`
  - wide holdout ROI `0.0984` (`344` races / `1533` bets)
- single-race inference smoke も通過
  - horse-level `16` 行
  - wide-level `120` 行

## 実行コマンド

- `uv run python scripts_v3/build_features_v3.py --input data/features_v2.parquet --output data/features_v3.parquet --meta-output data/features_v3_meta.json`
- `uv run python scripts_v3/build_features_v3.py --input data/features_v2_2025.parquet --output data/features_v3_2025.parquet --meta-output data/features_v3_2025_meta.json`
- `uv run python scripts_v3/train_win_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_lgbm_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_xgb_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_place_cat_v3.py --holdout-input data/features_v3_2025.parquet`
- `uv run python scripts_v3/train_win_stack_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --holdout-year 2025`
- `uv run python scripts_v3/train_place_stack_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --holdout-year 2025`
- `uv run python scripts_v3/train_pl_v3.py --features-input data/features_v3.parquet --holdout-input data/features_v3_2025.parquet --pl-feature-profile stack_default --train-window-years 3 --holdout-year 2025 --emit-wide-oof`
- `uv run python scripts_v3/backtest_wide_v3.py --input data/oof/pl_v3_holdout_2025_pred.parquet --years 2025 --holdout-year 2026 --output data/backtest_v3/backtest_wide_v3_holdout_2025_strict_stack.json --meta-output data/backtest_v3/backtest_wide_v3_holdout_2025_strict_stack_meta.json --force`
- `uv run python scripts_v3/predict_race_v3.py --input data/inference/race_features_smoke_2025.parquet --pl-model models/pl_v3_recent_window.joblib --output data/predictions/race_v3_pred_smoke_2025.parquet --emit-wide --wide-output data/predictions/race_v3_wide_smoke_2025.parquet`

## 残リスク

- `torch` 未導入環境では PL が NumPy fallback になるため、backend 差分の影響余地が残る
- PL fixed3 OOF は current year coverage では空のままで、CV 比較は holdout 主体になる
- wide ROI は単年・DB依存なので、モデル採否は top3 指標と併せて判断する
