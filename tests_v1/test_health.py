def test_health():
    # NOTE:
    # starlette/fastapi の TestClient は環境によってはブロッキングすることがあるため、
    # ここではハンドラを直接呼び出してヘルスチェックの契約を検証する。
    from app.main import app, health

    assert any(getattr(route, "path", None) == "/health" for route in app.router.routes)
    assert health() == {"status": "ok"}
