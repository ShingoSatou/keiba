# Keiba Prediction（決め事）

## 方針（最重要）
- 本リポジトリは **v1（旧）** と **v2（新）** が同居
- **新規開発は v2 のみ**
- **v1 は参照専用（保守・編集しない）**
- v2の実装/変更は毎回 @docs/specs_v2/競馬予測システム 課題設定・要件定義書.md を確認（該当箇所のみでOK）
- それ以外の仕様はタスクに応じて `docs/specs_v2/` を参照

## 対象の目安
- v1: `docs/*_v1/`, `scripts_v1/`, `tests_v1/`, `migrations_v1/`
- v2: `docs/specs_v2/`, `app/`, `frontend/`, `scripts_v2/`, `test_v2/`, `migrations_v2/`

## 開発コマンド（v2）
- 依存: `uv sync`
- API: `uv run uvicorn app.main:app --reload`
- UI: `cd frontend && npm ci && npm run dev`

## メモ
- `data/`, `models/`, `.env` はGit管理しない（`.gitignore`）
