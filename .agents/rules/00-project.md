---
trigger: always_on
---

# Project Rules (FastAPI + uv + Python 3.11)

## Tech
- Python 3.11 固定（.python-version を正とする）
- 依存管理は uv のみ（pip/poetry は使わない）
- アプリは FastAPI を使用。起動は uvicorn を想定。

## Coding
- app/main.py は薄く保つ（ルーティングやロジックは分割して拡張してよい）
- 例外・入力バリデーション・レスポンス形式は明示する
- 変更には可能な範囲で pytest を追加/更新する（最低1ケース）

## Verification loop (必須)
変更後は必ず以下を通す（ローカル → PR/CI でも同等）:
- uv run ruff format .
- uv run ruff check .
- uv run pytest -q

## Safety
- rm / del / 破壊的な一括操作は、実行前に必ず確認する
- DBは絶対に変更しない。別ユーザーが管理しています。