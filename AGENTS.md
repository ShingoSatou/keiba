# Keiba Prediction（決め事）

## 方針（最重要）
- 本リポジトリは **v1（旧）/ v2（安定）/ v3（最新）** が同居
- **新規開発は原則 v3**（v2は互換維持・保守が中心）
- **v1 は参照専用（保守・編集しない）**
- v3の実装/変更は毎回 `docs/specs_v3/v3_実装仕様.md` と `docs/ops_v3/` を確認
- v2を触る場合は毎回 `docs/specs_v2/競馬予測システム 課題設定・要件定義書.md` を確認（該当箇所のみでOK）
- v3は v2 の作法（Rolling年次CV、OOF保存、as-ofリーク防止）を踏襲する前提のため、必要に応じて `docs/specs_v2/` も参照

## 対象の目安
- v1: `docs/*_v1/`, `scripts_v1/`, `tests_v1/`, `migrations_v1/`
- v2: `docs/specs_v2/`, `app/`, `frontend/`, `scripts_v2/`, `test_v2/`, `migrations_v2/`
- v3: `docs/specs_v3/`, `docs/ops_v3/`, `scripts_v3/`, `test_v3/`

## 開発コマンド

### v2
- 依存: `uv sync`
- API: `uv run uvicorn app.main:app --reload`
- UI: `cd frontend && npm ci && npm run dev`

### v3
- 依存: `uv sync`
- 追加依存（必要に応じて）: `uv sync --extra xgboost --extra catboost`（PL学習のtorchはoptional）
- v3特徴量生成: `uv run python scripts_v3/build_features_v3.py`
- 学習/評価の入口: `docs/ops_v3/スクリプトリファレンス.md`
- 実行結果レポート: `docs/ops_v3/v3_run_report_2026-03-04.md`

## メモ
- `data/`, `models/`, `.env` はGit管理しない（`.gitignore`）
