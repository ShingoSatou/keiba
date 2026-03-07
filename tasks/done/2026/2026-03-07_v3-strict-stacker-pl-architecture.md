# 2026-03-07_v3-strict-stacker-pl-architecture

## タイトル

- v3 strict temporal stacker 導入と PL 責務分離

## ステータス

- `done`

## 対象範囲

- `scripts_v3/build_features_v3.py`
- `scripts_v3/odds_v3_common.py`
- `scripts_v3/feature_registry_v3.py`
- `scripts_v3/cv_policy_v3.py`
- `scripts_v3/train_stacker_v3_common.py`
- `scripts_v3/train_win_stack_v3.py`
- `scripts_v3/train_place_stack_v3.py`
- `scripts_v3/pl_v3_common.py`
- `scripts_v3/train_pl_v3.py`
- `scripts_v3/predict_race_v3.py`
- `test_v3/*`
- `docs/specs_v3/*`
- `docs/ops_v3/*`

## 対象バージョン

- `v3`

## 前提・仮定

- base binary 6 本の fixed 4-year sliding policy は維持する。
- stacker は strict temporal OOF の capped expanding (`min=2`, `max=4`) を採用する。
- PL default main path は stacker 出力を主入力に使う。
- 現行データ年数では PL fixed 3-year sliding OOF fold が成立しないため、holdout-only を許容し、年対応表で明示する。

## チェックリスト

- [x] features_v3 に win/place t20/t15/t10 snapshot と place width を追加する
- [x] stacker 学習と artifact 契約を実装する
- [x] PL default contract を stacker ベースへ更新する
- [x] `predict_race_v3.py` を base → stacker → PL に更新する
- [x] year coverage 出力を実装する
- [x] tests を追加・更新する
- [x] docs/specs_v3 と docs/ops_v3 を更新する
- [x] targeted test / CLI help を実行する

## 確認結果

- `features_v3` に単勝 `t20/t15/t10`、複勝 `t20/t15/t10`、`place_width_log_ratio` を追加した。
- stacker 用 CV helper に capped expanding (`min=2`, `max=4`) を追加した。
- `train_win_stack_v3.py` / `train_place_stack_v3.py` を追加し、strict temporal OOF / holdout / bundle meta を出力するようにした。
- PL default profile を `stack_default` に切り替え、`z_win_stack`, `z_place_stack`, `place_width_log_ratio`, interaction block を使うようにした。
- `predict_race_v3.py` を `base -> stacker -> PL` 順へ更新した。
- `v3_pipeline_year_coverage.json` を追加し、base / stacker / PL の OOF 年対応を保存するようにした。
- specs / ops docs を stacker default 前提に更新した。

## 実行コマンド

- `git status --short`
- `uv run python -m py_compile scripts_v3/train_pl_v3.py scripts_v3/predict_race_v3.py scripts_v3/train_stacker_v3_common.py scripts_v3/feature_registry_v3.py scripts_v3/odds_v3_common.py scripts_v3/build_features_v3.py scripts_v3/pl_v3_common.py scripts_v3/cv_policy_v3.py`
- `uv run pytest -s test_v3/test_cv_window_policy_v3.py test_v3/test_features_v3_no_leakage.py test_v3/test_feature_registry_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_predict_race_v3.py test_v3/test_v3_cli_smoke.py`
- `uv run ruff check scripts_v3/train_pl_v3.py scripts_v3/predict_race_v3.py scripts_v3/train_stacker_v3_common.py scripts_v3/feature_registry_v3.py scripts_v3/odds_v3_common.py scripts_v3/build_features_v3.py scripts_v3/pl_v3_common.py scripts_v3/cv_policy_v3.py test_v3/test_cv_window_policy_v3.py test_v3/test_features_v3_no_leakage.py test_v3/test_feature_registry_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_predict_race_v3.py test_v3/test_v3_cli_smoke.py`

## 関連ドキュメント

- `docs/specs_v3/v3_システム仕様書.md`
- `docs/specs_v3/v3_01_全体アーキテクチャ.md`
- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/specs_v3/v3_05_共通基盤と付録.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- place / win snapshot coverage によっては holdout で欠損除外が増える可能性がある。
- current repo の年範囲では PL fixed3 OOF fold は 0 件であり、評価は holdout/final artifact 側で見る必要がある。
