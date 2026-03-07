# 2026-03-07_v3-optuna-boundary-doc

## タイトル

- v3 binary Optuna 境界張り付き所見のドキュメント化

## ステータス

- `done`

## 対象範囲

- `docs/ops_v3/スクリプトリファレンス.md`
- `tasks/todo.md`

## 対象バージョン

- `v3`

## 前提・仮定

- 現時点では feature 追加も検討中のため、Optuna の再 tuning は実施しない。
- 後続の再 tuning で探索空間を見直せるよう、境界付近の所見だけを残す。

## チェックリスト

- [x] 既存 docs の適切な追記先を選ぶ
- [x] 境界張り付き所見と再 tuning defer の理由を追記する
- [x] 反映内容を確認する

## 確認結果

- `docs/ops_v3/スクリプトリファレンス.md` の `3.1 binary Optuna tuning` に、2026-03-07 時点の運用メモを追記した。
- メモには、TE leak 修正後でも `win_xgb` のスコア差分が小さいため即時再 tuning は見送ること、上限張り付きは見られず下限側に圧力がある候補だけを残すことを記載した。
- 次回の feature 追加後に widening を検討する候補として、`xgb`, `win_cat`, 一部 `lgbm` の下限見直し案を明記した。

## 実行コマンド

- `rg -n "tune_binary_optuna_v3|Optuna" docs/ops_v3 docs/specs_v3`
- `sed -n '72,150p' docs/ops_v3/スクリプトリファレンス.md`

## 関連ドキュメント

- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- 実際に feature set を追加した後は、ここに書いた探索空間候補を再点検する必要がある
