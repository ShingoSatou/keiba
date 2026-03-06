# 2026-03-06_feature-governance-hardening

## タイトル

- v3特徴量ガバナンス hardening

## ステータス

- `done`

## 対象範囲

- `scripts_v3/feature_registry_v3.py` 新規追加
- `scripts_v3/train_binary_v3_common.py` 更新
- `scripts_v3/train_binary_model_v3.py` 更新
- `scripts_v3/train_pl_v3.py` 更新
- `scripts_v3/build_features_v3.py` 更新
- `test_v3/test_feature_registry_v3.py` 新規追加
- `test_v3/test_binary_feature_contract_v3.py` 新規追加
- `test_v3/test_pl_feature_contract_v3.py` 新規追加
- `docs/ops_v3/Assumptions.md` 更新
- `docs/ops_v3/v3_run_report_2026-03-04.md` 更新
- `docs/ops_v3/スクリプトリファレンス.md` 更新
- `docs/specs_v3/v3_システム仕様書.md` 更新

## 対象バージョン

- `v3`
- `v2` / `v1` は参照のみ

## 前提・仮定

- binary の default operational profile は `t10_only`
- entity raw ID は binary default OFF
- PL は `required OOF + t10 odds + small context` の explicit list に固定する
- final odds 列は `features_v3` に保持するが、学習投入は opt-in のみとする
- full retrain / 指標再計測はこの task 自体の対象外

## チェックリスト

- [x] feature registry 新設
- [x] binary を whitelist + feature manifest 出力へ変更
- [x] binary CLI を `--include-entity-id-features` / `--operational-mode` ベースへ更新
- [x] PL を explicit-list feature contract へ変更
- [x] build_features meta に operational default を追加
- [x] contract tests 3 本を追加
- [x] docs / spec を現仕様へ更新
- [x] pytest / ruff / CLI help / `git diff --check` を確認

## 確認結果

- `scripts_v3/feature_registry_v3.py` を追加し、binary / PL の whitelist と forbidden contract を一元化
- binary default contract は `t10_only`、entity raw ID default OFF に変更
- PL は `required OOF + t10 odds + small context` の明示列のみを採用し、extra numeric を自動収集しない
- build_features meta に `operational_default`, `contains_final_odds_columns`, `contains_t10_odds_columns` を追加
- 追加テスト 3 本と既存 CLI smoke を含めて `13 passed`

## 実行コマンド

- `uv run ruff check scripts_v3/feature_registry_v3.py scripts_v3/train_binary_v3_common.py scripts_v3/train_binary_model_v3.py scripts_v3/train_pl_v3.py scripts_v3/build_features_v3.py test_v3/test_feature_registry_v3.py test_v3/test_binary_feature_contract_v3.py test_v3/test_pl_feature_contract_v3.py`
- `uv run pytest test_v3/test_feature_registry_v3.py test_v3/test_binary_feature_contract_v3.py test_v3/test_pl_feature_contract_v3.py test_v3/test_v3_cli_smoke.py`
- `uv run python scripts_v3/train_binary_model_v3.py --help`
- `uv run python scripts_v3/train_pl_v3.py --help`
- `git diff --check`

## 関連ドキュメント

- `docs/ops_v3/Assumptions.md`
- `docs/ops_v3/スクリプトリファレンス.md`
- `docs/ops_v3/v3_run_report_2026-03-04.md`
- `docs/specs_v3/v3_システム仕様書.md`

## 残リスク

- この task 完了時点では run report の数値は hardening 前の記録のままだった
- `--operational-mode includes_final` で学習した binary モデルは、運用 t10 入力では final odds 列不足でそのまま推論できない

## 移行メモ

- 旧 monolithic `tasks/todo.md` にあった feature governance 系の重複 3 エントリを 1 file に統合
- hardening 後の再学習・評価は別 task `2026-03-06_retrain-eval-current-contract` へ分離
