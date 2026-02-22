"""
UIエンドポイントのテスト

テスト用のJSONファイルを一時生成し、TestClientでAPIを叩いて動作を確認する。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

_DUMMY = {
    "summary": {
        "period_from": "2023-01-01",
        "period_to": "2024-12-31",
        "n_races": 100,
        "n_bets": 20,
        "n_hits": 3,
        "hit_rate": 0.15,
        "total_bet": 10000,
        "total_return": 9500,
        "roi": 0.95,
        "max_drawdown": 2000,
        "logloss": 0.32,
        "auc": 0.73,
    },
    "monthly": [
        {"month": "2023-01", "n_bets": 5, "n_hits": 1, "roi": 1.10},
        {"month": "2023-02", "n_bets": 5, "n_hits": 0, "roi": 0.80},
    ],
    "bets": [
        {
            "race_date": "2023-01-08",
            "race_id": 202301080601,
            "horse_name": "テストホース",
            "horse_no": 3,
            "p_win": 0.18,
            "odds_final": 6.0,
            "ev_profit": 0.08,
            "is_hit": True,
            "payout": 3000,
            "profit": 2500,
        },
        {
            "race_date": "2023-01-15",
            "race_id": 202301150601,
            "horse_name": "サブホース",
            "horse_no": 5,
            "p_win": 0.15,
            "odds_final": 7.5,
            "ev_profit": 0.12,
            "is_hit": False,
            "payout": 0,
            "profit": -500,
        },
    ],
}


@pytest.fixture
def client(tmp_path: Path):
    """テスト用JSONを一時ファイルに書き出し、_DATA_PATHをパッチしてクライアントを返す"""
    json_path = tmp_path / "backtest_result.json"
    json_path.write_text(json.dumps(_DUMMY), encoding="utf-8")

    # _load_result のキャッシュをリセット
    import app.routers.ui as ui_module

    ui_module._load_result.cache_clear()

    with patch("app.routers.ui._DATA_PATH", json_path):
        yield TestClient(app)

    # テスト後にキャッシュをリセット
    ui_module._load_result.cache_clear()


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------


def test_summary_ok(client):
    """GET /ui/summary → 200 + 必要キーを含む"""
    resp = client.get("/ui/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "roi" in data
    assert "n_bets" in data
    assert "hit_rate" in data
    assert data["n_bets"] == 20


def test_monthly_ok(client):
    """GET /ui/monthly → 200 + リスト形式"""
    resp = client.get("/ui/monthly")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["month"] == "2023-01"


def test_bets_all(client):
    """GET /ui/bets → 全件返す"""
    resp = client.get("/ui/bets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


def test_bets_filter_hit(client):
    """GET /ui/bets?hit=hit → 的中のみ"""
    resp = client.get("/ui/bets?hit=hit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["is_hit"] is True


def test_bets_filter_miss(client):
    """GET /ui/bets?hit=miss → 外れのみ"""
    resp = client.get("/ui/bets?hit=miss")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["is_hit"] is False


def test_bets_pagination(client):
    """GET /ui/bets?page=2 → ページ2（データ数が少なければ空）"""
    resp = client.get("/ui/bets?page=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2
    assert isinstance(data["total_pages"], int)


def test_summary_503_when_file_missing():
    """JSONファイルが存在しない場合 → 503"""
    import app.routers.ui as ui_module

    ui_module._load_result.cache_clear()

    with patch("app.routers.ui._DATA_PATH", Path("/nonexistent/path.json")):
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get("/ui/summary")
    assert resp.status_code == 503

    ui_module._load_result.cache_clear()
