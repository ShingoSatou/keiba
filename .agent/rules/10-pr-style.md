# PR style rules

## PRの粒度
- 1PR = 1目的（小さく）
- 大規模リファクタと機能変更は分ける

## PR本文に必ず書く
- 目的 / 背景
- 変更内容（箇条書き）
- 影響範囲（API/挙動/設定）
- 検証（実行したコマンドと結果）
  - uv run ruff format .
  - uv run ruff check .
  - uv run pytest -q

## CIが落ちたとき
1) 原因を1行で要約
2) 最小の修正で直す
3) 同じ検証を再実行し、結果をPRに追記する
