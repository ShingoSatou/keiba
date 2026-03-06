# 2026-03-06_v3-fixed-4y-eval-policy

## タイトル

- v3評価条件の4年固定sliding window統一

## ステータス

- `done`

## 対象範囲

- `scripts_v3/cv_policy_v3.py` 新規追加
- `scripts_v3/v3_common.py` 更新
- `scripts_v3/train_binary_v3_common.py` 更新
- `scripts_v3/train_binary_model_v3.py` 更新
- `scripts_v3/train_odds_calibrator_v3.py` 更新
- `scripts_v3/train_pl_v3.py` 更新
- `scripts_v3/predict_race_v3.py` 更新
- `scripts_v3/backtest_wide_v3.py` 更新
- `test_v3/test_cv_window_policy_v3.py` 新規追加
- `test_v3/test_v3_common.py` 更新
- `test_v3/test_binary_feature_contract_v3.py` 更新
- `test_v3/test_pl_feature_contract_v3.py` 更新
- `test_v3/test_backtest_wide_v3_smoke.py` 更新
- `docs/ops_v3/Assumptions.md` 更新
- `docs/ops_v3/スクリプトリファレンス.md` 更新
- `docs/ops_v3/v3_run_report_2026-03-06_feature_governance_retrain.md` 更新
- `docs/specs_v3/v3_01_全体アーキテクチャ.md` 更新
- `docs/specs_v3/v3_03_二値分類と校正仕様.md` 更新
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md` 更新
- `docs/specs_v3/v3_05_共通基盤と付録.md` 更新

## 対象バージョン

- `v3`

## 前提・仮定

- v3 の標準比較条件は `train_window_years=4` と `cv_window_policy=fixed_sliding`
- holdout 評価の標準モデルは holdout 年直前4年のみで学習する
- `all_years_model` は互換維持のため残すが、標準比較条件では使わない
- backtest meta の `holdout_year` は model policy の holdout 年として保存し、CLI の `--holdout-year` は filter 境界として別管理する

## チェックリスト

- [x] task ledger を更新
- [x] 共通 CV policy utility を追加
- [x] binary / odds calibration / PL の default と metadata を更新
- [x] predict / backtest へ policy metadata を伝搬
- [x] docs / specs を4年固定へ更新
- [x] pytest / CLI help / diff check を確認

## 確認結果

- `scripts_v3/cv_policy_v3.py` を追加し、fixed-length sliding fold 生成・`cv_policy` payload・parquet 列付与を共通化
- binary / odds calibration / PL の default を 4 年に統一し、OOF / holdout / metrics / meta / PL artifact に `cv_policy` を保存
- binary / PL の標準 `main_model` と odds calibrator の最終学習を holdout 年直前4年のみに変更
- `predict_race_v3.py` が `valid_year` と `cv_policy` を出力し、`backtest_wide_v3.py` がそれを保持して meta に転記するよう更新
- docs / specs を 4 年固定 standard に更新し、2026-03-06 report に policy update 注記を追加

## 実行コマンド

- `uv run ruff check scripts_v3/cv_policy_v3.py scripts_v3/v3_common.py scripts_v3/train_binary_v3_common.py scripts_v3/train_binary_model_v3.py scripts_v3/train_odds_calibrator_v3.py scripts_v3/train_pl_v3.py scripts_v3/predict_race_v3.py scripts_v3/backtest_wide_v3.py test_v3/test_cv_window_policy_v3.py test_v3/test_v3_common.py test_v3/test_binary_feature_contract_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_backtest_wide_v3_smoke.py`
- `uv run pytest test_v3/test_cv_window_policy_v3.py test_v3/test_v3_common.py -q`
- `uv run pytest test_v3/test_binary_feature_contract_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_backtest_wide_v3_smoke.py -q`
- `uv run pytest test_v3/test_v3_cli_smoke.py -q`
- `git diff --check`

## 関連ドキュメント

- `docs/specs_v3/v3_システム仕様書.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/specs_v3/v3_05_共通基盤と付録.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-06_feature_governance_retrain.md`

## 残リスク

- historical artifact / report（例: 2026-03-04）の窓条件そのものは書き換えていないため、履歴比較時は引き続き注記が必要
- 実データでの full retrain / holdout 再計測は未実施のため、数値面の比較基準は次回 run report で更新する必要がある
