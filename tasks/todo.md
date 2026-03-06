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
| - | 進行中タスクなし | - | - | - |

## Done

| id | title | version | status | file |
|---|---|---|---|---|
| `2026-03-04_backtest-wide-v3-pl-alignment` | v3ワイドROIをPL出力整合にする | v3 | `done` | [task](./done/2026/2026-03-04_backtest-wide-v3-pl-alignment.md) |
| `2026-03-06_feature-governance-hardening` | v3特徴量ガバナンス hardening | v3 | `done` | [task](./done/2026/2026-03-06_feature-governance-hardening.md) |
| `2026-03-06_retrain-eval-current-contract` | current contract の再学習・検証・評価 | v3 | `done` | [task](./done/2026/2026-03-06_retrain-eval-current-contract.md) |
| `2026-03-06_v3-spec-split-and-ledger-update` | v3仕様書分割と AGENTS の仕様書台帳化 | v3 | `done` | [task](./done/2026/2026-03-06_v3-spec-split-and-ledger-update.md) |
| `2026-03-06_fix-ci-for-pr33` | PR #33 の CI 失敗修正 | v3 | `done` | [task](./done/2026/2026-03-06_fix-ci-for-pr33.md) |

## 移行メモ

- 2026-03-06 に旧 monolithic `tasks/todo.md` から履歴を分割移行した。
- feature governance 系の重複エントリは 1 task file に統合した。
