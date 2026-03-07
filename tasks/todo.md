# Task Ledger

## 使い方

- `tasks/todo.md` は台帳専用とし、1画面で見渡せる分量を維持する。
- 複数ステップの作業は 1 task = 1 file で管理する。
- 新規 task は `tasks/templates/task.md` を元に `tasks/active/` に作成する。
- 完了した task は `tasks/done/YYYY/` に移動し、この台帳のリンクを更新する。
- 詳細な経緯、確認結果、実行コマンドは各 task file に書き、この台帳には要約だけ残す。
- 再発防止ルールは `tasks/lessons.md` に記録する。

## Active

| id | title | version | status | file |
|---|---|---|---|---|
| `2026-03-07_v3-docs-pr` | v3 docs 更新と PR 作成 | v3 | `active` | [task](./active/2026-03-07_v3-docs-pr.md) |
| `2026-03-07_v3-refactor-remove-v2-dependency` | v3 リファクタリング: v2 依存排除・legacy 削除・ドキュメント整備 | v3 | `in-progress` | [task](./active/2026-03-07_v3-refactor-remove-v2-dependency.md) |

## Done

| id | title | version | status | file |
|---|---|---|---|---|
| `2026-03-07_v3-te-build-entrypoint` | v3 TE 生成入口の固定化 | v3 | `done` | [task](./done/2026/2026-03-07_v3-te-build-entrypoint.md) |
| `2026-03-07_v3-binary6-retrain-end-to-end` | v3 binary 6本再学習と end-to-end 検証 | v3 | `done` | [task](./done/2026/2026-03-07_v3-binary6-retrain-end-to-end.md) |
| `2026-03-07_v3-feature-contract-retrain-eval` | v3 特徴量再生成と stacker / PL 再学習・評価 | v3 | `done` | [task](./done/2026/2026-03-07_v3-feature-contract-retrain-eval.md) |
| `2026-03-07_v3-feature-contract-refactor` | v3 特徴量契約再設計: binary/stack/PL の層責務整理 | v3 | `done` | [task](./done/2026/2026-03-07_v3-feature-contract-refactor.md) |
| `2026-03-07_v3-optuna-boundary-doc` | v3 binary Optuna 境界張り付き所見のドキュメント化 | v3 | `done` | [task](./done/2026/2026-03-07_v3-optuna-boundary-doc.md) |
| `2026-03-07_v3-te-leakage-audit-fix` | v3 TE リーク監査と必要時の修正・xgb 再学習 | v3 | `done` | [task](./done/2026/2026-03-07_v3-te-leakage-audit-fix.md) |
| `2026-03-07_v3-binary-optuna-tuner` | v3 binary Optuna tuner 追加 | v3 | `done` | [task](./done/2026/2026-03-07_v3-binary-optuna-tuner.md) |
| `2026-03-07_v3-binary-optuna-full-tuning` | v3 binary full tuning 実行と tuned default 反映 | v3 | `done` | [task](./done/2026/2026-03-07_v3-binary-optuna-full-tuning.md) |
| `2026-03-07_v3-strict-stacker-retrain-eval` | v3 strict temporal stacker の full retrain と評価 | v3 | `done` | [task](./done/2026/2026-03-07_v3-strict-stacker-retrain-eval.md) |
| `2026-03-07_v3-strict-stacker-pl-architecture` | v3 strict temporal stacker 導入と PL 責務分離 | v3 | `done` | [task](./done/2026/2026-03-07_v3-strict-stacker-pl-architecture.md) |
| `2026-03-07_v2-o1-place-ci-fix` | v2 O1 複勝時系列 PR の CI format failure 修正 | v2 | `done` | [task](./done/2026/2026-03-07_v2-o1-place-ci-fix.md) |
| `2026-03-07_v2-o1-place-pr` | v2 O1 複勝時系列対応の PR 作成 | v2 | `done` | [task](./done/2026/2026-03-07_v2-o1-place-pr.md) |
| `2026-03-07_v2-o1-place-doc-audit` | v2 O1 複勝時系列対応のドキュメント反映確認 | v2 | `done` | [task](./done/2026/2026-03-07_v2-o1-place-doc-audit.md) |
| `2026-03-07_v2-o1-place-timeseries-load` | v2 O1 複勝時系列オッズの DB 取り込み拡張 | v2 | `done` | [task](./done/2026/2026-03-07_v2-o1-place-timeseries-load.md) |
| `2026-03-07_v3-pl-meta-default-pr` | v3 PL meta_default 契約移行の PR 作成 | v3 | `done` | [task](./done/2026/2026-03-07_v3-pl-meta-default-pr.md) |
| `2026-03-04_backtest-wide-v3-pl-alignment` | v3ワイドROIをPL出力整合にする | v3 | `done` | [task](./done/2026/2026-03-04_backtest-wide-v3-pl-alignment.md) |
| `2026-03-06_v3-pl-inference-test-gap-analysis` | v3 PL/inference テストギャップ調査 | v3 | `done` | [task](./done/2026/2026-03-06_v3-pl-inference-test-gap-analysis.md) |
| `2026-03-06_v3-pl-meta-input-interface-survey` | v3 PL 学習・推論・bundle/meta interface 調査 | v3 | `done` | [task](./done/2026/2026-03-06_v3-pl-meta-input-interface-survey.md) |
| `2026-03-06_v3-pl-meta-default-migration` | v3 PL meta_default 契約移行 | v3 | `done` | [task](./done/2026/2026-03-06_v3-pl-meta-default-migration.md) |
| `2026-03-06_feature-governance-hardening` | v3特徴量ガバナンス hardening | v3 | `done` | [task](./done/2026/2026-03-06_feature-governance-hardening.md) |
| `2026-03-06_fix-ci-for-pr34` | PR #34 の CI 失敗修正と再発防止 | v3 | `done` | [task](./done/2026/2026-03-06_fix-ci-for-pr34.md) |
| `2026-03-06_v3-fixed-4y-pr` | v3 4年固定評価条件変更の PR 作成 | v3 | `done` | [task](./done/2026/2026-03-06_v3-fixed-4y-pr.md) |
| `2026-03-06_v3-fixed-4y-eval-policy` | v3評価条件の4年固定sliding window統一 | v3 | `done` | [task](./done/2026/2026-03-06_v3-fixed-4y-eval-policy.md) |
| `2026-03-06_v3-fixed-4y-retrain-eval` | v3 4年固定条件での再学習・検証・評価 | v3 | `done` | [task](./done/2026/2026-03-06_v3-fixed-4y-retrain-eval.md) |
| `2026-03-06_retrain-eval-current-contract` | current contract の再学習・検証・評価 | v3 | `done` | [task](./done/2026/2026-03-06_retrain-eval-current-contract.md) |
| `2026-03-06_v3-spec-split-and-ledger-update` | v3仕様書分割と AGENTS の仕様書台帳化 | v3 | `done` | [task](./done/2026/2026-03-06_v3-spec-split-and-ledger-update.md) |
| `2026-03-06_fix-ci-for-pr33` | PR #33 の CI 失敗修正 | v3 | `done` | [task](./done/2026/2026-03-06_fix-ci-for-pr33.md) |

## 移行メモ

- 2026-03-06 に旧 monolithic `tasks/todo.md` から履歴を分割移行した。
- feature governance 系の重複エントリは 1 task file に統合した。
