# 2026-03-07_v3-pl-meta-default-pr

## タイトル

- v3 PL meta_default 契約移行の PR 作成

## ステータス

- `active`

## 対象範囲

- `scripts_v3/*`
- `test_v3/*`
- `docs/specs_v3/*`
- `docs/ops_v3/*`
- `README.md`
- `tasks/*`

## 対象バージョン

- `v3`

## 前提・仮定

- 実装・full retrain・holdout / wide backtest は完了済み。
- PR では実装差分と評価結果を同時にレビュー可能な形でまとめる。
- 2025 holdout では `raw_legacy` が `meta_default` を上回っているため、その評価結果を PR に明記する。

## チェックリスト

- [ ] PR 用 branch を切る
- [ ] task ledger を更新する
- [ ] 変更差分を commit する
- [ ] remote へ push する
- [ ] PR を作成する

## 確認結果

- 未記入

## 実行コマンド

- 未記入

## 関連ドキュメント

- `docs/ops_v3/v3_run_report_2026-03-06_pl_meta_default_retrain.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/ops_v3/スクリプトリファレンス.md`

## 残リスク

- PR は code default を `meta_default` に切り替えるが、実測評価では `raw_legacy` 優勢という結論を含む。
