# 2026-03-06_v3-pl-inference-test-gap-analysis

## タイトル

- v3 PL/inference テストギャップ調査

## ステータス

- `done`

## 対象範囲

- `test_v3/test_feature_registry_v3.py`
- `test_v3/test_pl_feature_contract_v3.py`
- `test_v3/test_features_v3_no_leakage.py`
- `test_v3/test_v3_cli_smoke.py`
- `test_v3/test_binary_feature_contract_v3.py`
- `test_v3/test_pl_v3_wide_invariants.py`
- `scripts_v3/feature_registry_v3.py`
- `scripts_v3/train_pl_v3.py`
- `scripts_v3/predict_race_v3.py`
- `scripts_v3/odds_v3_common.py`

## 対象バージョン

- `v3`

## 前提・仮定

- ユーザー依頼は実装ではなく既存テスト調査と追加候補整理
- docs は test 名から必要性が出ない限り読まない

## チェックリスト

- [x] 関連テストとユーティリティの現状把握
- [x] train_pl_v3 / predict_race_v3 / odds 制約の未テスト境界整理
- [x] 現行スタイルに沿う追加テスト案の要約

## 確認結果

- `test_v3` には共有 `conftest.py` / fixture module はなく、各 test file が `_sample_frame()` や `tmp_path` をローカルに持つ
- `feature_registry_v3` / `train_pl_v3` は feature contract と meta payload まではテスト済みだが、odds calibration OOF merge や matrix 前処理は未テスト
- `predict_race_v3` は CLI help のみテストされており、input validation / odds calibrator 適用 / PL artifact scoring / CV metadata 伝播は未テスト
- odds 系は `select_t10_snapshot()` と `assert_t10_no_future_reference()` のみテスト済みで、`select_final_snapshot()` と `merge_odds_features()` は未カバー
- 関連 pytest を実行し、対象 18 件は全件 pass した

## 実行コマンド

- `rg --files test_v3 scripts_v3`
- `rg -n "train_pl_v3|predict_race_v3|feature_registry_v3|odds_v3_common|constraint|cap_prob|clip|odds" test_v3 scripts_v3`
- `uv run pytest test_v3/test_feature_registry_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_features_v3_no_leakage.py test_v3/test_v3_cli_smoke.py test_v3/test_backtest_wide_v3_smoke.py test_v3/test_pl_v3_wide_invariants.py`

## 関連ドキュメント

- なし

## 残リスク

- `predict_race_v3` の追加テストは現状ゼロに近いため、meta-layer を実装しても回帰が main 経路で見逃されやすい
- `train_pl_v3` の odds calibration 列取り込み規約はコード上は明確だが、現時点ではファイル境界込みで固定化されていない
