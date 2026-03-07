# 2026-03-07_v3-feature-contract-refactor

## タイトル

- v3 特徴量契約再設計: binary/stack/PL の層責務整理

## ステータス

- `done`

## 対象範囲

- `scripts_v3/feature_registry_v3.py`
- `scripts_v3/odds_v3_common.py`
- `scripts_v3/build_features_v3.py`
- `scripts_v3/pl_v3_common.py`
- `test_v3/test_binary_feature_contract_v3.py`
- `test_v3/test_feature_registry_v3.py`
- `test_v3/test_features_v3_no_leakage.py`
- `test_v3/test_pl_feature_contract_v3.py`
- `test_v3/test_predict_race_v3.py`
- `docs/specs_v3/v3_01_全体アーキテクチャ.md`
- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 対象バージョン

- `v3`

## 前提・仮定

- binary / stacker / PL の CV 方針、窓長、学習年は維持する。
- `t10_only` で final odds を禁止する既存制約は維持する。
- `features_v3` の raw odds / raw race context 列は物理削除しない。
- `place_mid_prob_t*` は `sqrt(lower * upper)` ベースの raw implied probability とした。

## チェックリスト

- [x] binary default contract から `BINARY_T10_ODDS_FEATURES` を外す
- [x] stack win/place 向け市場派生列を `merge_odds_features()` で追加する
- [x] stacker / PL contract を新設計へ更新する
- [x] 推論経路と manifest/meta の required feature 列を新 contract に追従させる
- [x] 対応テストを更新する
- [x] v3 specs / ops docs を更新する
- [x] targeted pytest / ruff check を実行する

## 確認結果

- binary default contract は `BINARY_BASE_FEATURES` のみとなり、`BINARY_T10_ODDS_FEATURES` は default 入力から外れた。
- `merge_odds_features()` で `p_win_odds_t20/t15/t10_norm`, `d_logit_win_*`, `place_mid_prob_t20/t15/t10`, `d_place_mid_10_20`, `d_place_width_10_20` を一元生成するようにした。
- `stack_win` は probability + logit diff 主体、`stack_place` は mid + width + diff 主体の contract へ更新した。
- `stack_default` は raw race context 単体列を contract から外し、interaction / race-relative のみを残した。
- stacker / PL contract は missing 列を黙って落とさず、明示的 `ValueError` を返すようにした。
- v3 specs / ops docs を実装に追従させた。

## 実行コマンド

- `uv run pytest -s test_v3/test_binary_feature_contract_v3.py test_v3/test_feature_registry_v3.py test_v3/test_features_v3_no_leakage.py test_v3/test_pl_feature_contract_v3.py test_v3/test_predict_race_v3.py`
- `uv run ruff check scripts_v3/feature_registry_v3.py scripts_v3/odds_v3_common.py scripts_v3/build_features_v3.py scripts_v3/pl_v3_common.py scripts_v3/train_stacker_v3_common.py scripts_v3/train_pl_v3.py scripts_v3/predict_race_v3.py test_v3/test_binary_feature_contract_v3.py test_v3/test_feature_registry_v3.py test_v3/test_features_v3_no_leakage.py test_v3/test_pl_feature_contract_v3.py test_v3/test_predict_race_v3.py`
- `uv run python -c "from scripts_v3.feature_registry_v3 import BINARY_BASE_FEATURES,BINARY_T10_ODDS_FEATURES,STACKER_WIN_ODDS_FEATURES,STACKER_PLACE_ODDS_FEATURES,PL_STACK_CORE_FEATURES,PL_STACK_INTERACTION_FEATURES; print('binary_default', BINARY_BASE_FEATURES); print('binary_t10_odds', BINARY_T10_ODDS_FEATURES); print('stack_win', STACKER_WIN_ODDS_FEATURES); print('stack_place', STACKER_PLACE_ODDS_FEATURES); print('pl_stack_default', PL_STACK_CORE_FEATURES + PL_STACK_INTERACTION_FEATURES)"`

## 関連ドキュメント

- `docs/specs_v3/v3_システム仕様書.md`
- `docs/specs_v3/v3_01_全体アーキテクチャ.md`
- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- 新 contract を実運用に反映するには `features_v3` 再生成と stacker / PL 再学習が必要。
- 既存 unsuffixed artifact は再学習前は旧 contract のまま残る。
