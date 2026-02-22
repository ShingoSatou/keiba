"""
backtest_runner サービスのテスト

DBアクセスを使わず、run_backtest_for_ui / _aggregate_monthly / export_json を
モックデータで検証する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.services.backtest_runner import (
    _aggregate_monthly,
    export_json,
    run_backtest_for_ui,
)

# ---------------------------------------------------------------------------
# テストデータ生成ヘルパー
# ---------------------------------------------------------------------------


def _make_test_df(n_races: int = 5, horses_per_race: int = 8) -> pd.DataFrame:
    """バックテスト用のDataFrameを生成（run_backtest_for_ui に渡せる形式）"""
    import numpy as np

    rng = np.random.RandomState(42)
    rows = []

    for r in range(n_races):
        race_id = 202301010101 + r
        race_date = pd.Timestamp("2023-01-08") + pd.Timedelta(days=7 * r)

        for h in range(1, horses_per_race + 1):
            p_win = rng.uniform(0.02, 0.25)
            odds = round(1 / p_win * rng.uniform(0.8, 1.2), 1)
            finish = h  # 馬番1が1着

            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "horse_id": f"horse_{r}_{h}",
                    "horse_name": f"テスト馬{r}_{h}",
                    "horse_no": h,
                    "p_win": p_win,
                    "odds_final": max(odds, 1.1),
                    "inv_odds": 1 / max(odds, 1.1),
                    "finish_pos": finish,
                    "is_win": 1 if finish == 1 else 0,
                }
            )

    df = pd.DataFrame(rows)
    df["q_market"] = df.groupby("race_id")["inv_odds"].transform(lambda x: x / x.sum())
    return df


# ---------------------------------------------------------------------------
# テスト: run_backtest_for_ui
# ---------------------------------------------------------------------------


class TestRunBacktestForUi:
    """run_backtest_for_ui の出力構造テスト"""

    def test_output_has_summary_monthly_bets(self):
        """出力に summary / monthly / bets の3キーが含まれる"""
        df = _make_test_df()
        result = run_backtest_for_ui(df, alpha=1.0, ev_threshold=1.0, min_prob=0.0, max_prob=1.0)

        assert "summary" in result
        assert "monthly" in result
        assert "bets" in result

    def test_summary_has_required_keys(self):
        """summary に UI が要求する全キーが含まれる"""
        df = _make_test_df()
        result = run_backtest_for_ui(df, alpha=1.0, ev_threshold=1.0, min_prob=0.0, max_prob=1.0)
        summary = result["summary"]

        required_keys = [
            "period_from",
            "period_to",
            "n_races",
            "n_bets",
            "n_hits",
            "hit_rate",
            "total_bet",
            "total_return",
            "roi",
            "max_drawdown",
            "logloss",
            "auc",
        ]
        for key in required_keys:
            assert key in summary, f"summary に {key} がありません"

    def test_bets_have_required_fields(self):
        """bets の各行に UI が要求するフィールドが揃っている"""
        df = _make_test_df()
        result = run_backtest_for_ui(df, alpha=1.0, ev_threshold=1.0, min_prob=0.0, max_prob=1.0)

        if not result["bets"]:
            pytest.skip("ベットが生成されなかった（閾値厳しすぎ）")

        required_fields = [
            "race_date",
            "race_id",
            "horse_name",
            "horse_no",
            "p_win",
            "odds_final",
            "ev_profit",
            "is_hit",
            "payout",
            "profit",
        ]
        for bet in result["bets"]:
            for field in required_fields:
                assert field in bet, f"bets に {field} がありません"

    def test_monthly_aggregation(self):
        """monthly が月別に正しく集計されている"""
        df = _make_test_df(n_races=10)
        result = run_backtest_for_ui(df, alpha=1.0, ev_threshold=1.0, min_prob=0.0, max_prob=1.0)
        monthly = result["monthly"]

        if not monthly:
            pytest.skip("月別データが生成されなかった")

        for m in monthly:
            assert "month" in m
            assert "n_bets" in m
            assert "n_hits" in m
            assert "roi" in m
            # month の形式チェック
            assert len(m["month"]) == 7  # "YYYY-MM"

    def test_empty_when_no_candidates(self):
        """ベット候補がない場合、空のサマリを返す"""
        df = _make_test_df()
        # 極端に高い閾値で候補をゼロにする
        result = run_backtest_for_ui(df, alpha=1.0, ev_threshold=100.0, min_prob=0.0, max_prob=1.0)

        assert result["summary"]["n_bets"] == 0
        assert result["bets"] == []
        assert result["monthly"] == []

    def test_roi_calculation(self):
        """ROI が total_return / total_bet で正しく計算される"""
        df = _make_test_df()
        result = run_backtest_for_ui(df, alpha=1.0, ev_threshold=1.0, min_prob=0.0, max_prob=1.0)
        s = result["summary"]

        if s["total_bet"] > 0:
            expected_roi = round(s["total_return"] / s["total_bet"], 4)
            assert s["roi"] == expected_roi


# ---------------------------------------------------------------------------
# テスト: _aggregate_monthly
# ---------------------------------------------------------------------------


class TestAggregateMonthly:
    """月別集計ロジックのテスト"""

    def test_basic_aggregation(self):
        """2ヶ月分のベットが正しく集計される"""
        records = [
            {"race_date": "2023-01-08", "is_hit": True, "payout": 3000},
            {"race_date": "2023-01-15", "is_hit": False, "payout": 0},
            {"race_date": "2023-02-05", "is_hit": True, "payout": 2500},
        ]
        result = _aggregate_monthly(records, bet_amount=500)

        assert len(result) == 2
        assert result[0]["month"] == "2023-01"
        assert result[0]["n_bets"] == 2
        assert result[0]["n_hits"] == 1
        assert result[0]["roi"] == round(3000 / 1000, 4)

        assert result[1]["month"] == "2023-02"
        assert result[1]["n_bets"] == 1
        assert result[1]["n_hits"] == 1

    def test_empty_input(self):
        """空のベットリストで空の結果を返す"""
        result = _aggregate_monthly([], bet_amount=500)
        assert result == []


# ---------------------------------------------------------------------------
# テスト: export_json
# ---------------------------------------------------------------------------


class TestExportJson:
    """JSON出力のテスト"""

    def test_writes_valid_json(self, tmp_path: Path):
        """有効なJSONファイルが書き出される"""
        data = {"summary": {"roi": 0.95}, "monthly": [], "bets": []}
        out = tmp_path / "test_result.json"

        export_json(data, out)

        assert out.exists()
        content = json.loads(out.read_text(encoding="utf-8"))
        assert content["summary"]["roi"] == 0.95

    def test_creates_parent_dirs(self, tmp_path: Path):
        """親ディレクトリが存在しない場合、自動生成される"""
        out = tmp_path / "subdir" / "result.json"
        export_json({"summary": {}}, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# テスト: POST /ui/reload
# ---------------------------------------------------------------------------


class TestReloadEndpoint:
    """POST /ui/reload エンドポイントのテスト"""

    def test_reload_ok(self, tmp_path: Path):
        """JSONファイルが存在する場合、reload が成功する"""
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        import app.routers.ui as ui_module
        from app.main import app

        ui_module._load_result.cache_clear()

        json_path = tmp_path / "backtest_result.json"
        json_path.write_text(
            json.dumps({"summary": {}, "monthly": [], "bets": []}),
            encoding="utf-8",
        )

        with patch("app.routers.ui._DATA_PATH", json_path):
            client = TestClient(app)
            resp = client.post("/ui/reload")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        ui_module._load_result.cache_clear()

    def test_reload_missing_file(self):
        """JSONが存在しない場合、warn を返す"""
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        import app.routers.ui as ui_module
        from app.main import app

        ui_module._load_result.cache_clear()

        with patch("app.routers.ui._DATA_PATH", Path("/nonexistent/path.json")):
            client = TestClient(app)
            resp = client.post("/ui/reload")

        assert resp.status_code == 200
        assert resp.json()["status"] == "warn"
        ui_module._load_result.cache_clear()
