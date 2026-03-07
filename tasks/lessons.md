# Lessons

## 使い方

- ユーザーからの修正指摘が入ったとき、または同じミスが再発したときだけ追記する。
- 1項目は短い再発防止ルールにする。
- task の詳細な経緯は `tasks/todo.md` や各 task file に書き、ここには一般化した学びだけ残す。

## Entries

- PR 前の確認は CI と同じ入口に寄せる。repo では `bash scripts/check_ci.sh check` を使い、format 漏れは `bash scripts/check_ci.sh fix` で先に潰す。
- stacker / feature contract の設計変更では、指定された入力列以外を convenience で足さず、snapshot 種別と odds 変換の指示をそのまま固定する。
