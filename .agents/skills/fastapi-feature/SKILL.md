---
name: fastapi-feature
description: FastAPIのエンドポイント追加・変更を行い、ruff/pytestまで通す
---

# FastAPI Feature Skill
- 変更は小さく、テストを必ず追加/更新する
- 実装後に必ず以下を実行する:
  - uv run ruff format .
  - uv run ruff check .
  - uv run pytest -q
