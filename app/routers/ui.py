"""
UIバックエンドAPIルーター

data/backtest_result.json を読み込み、以下のエンドポイントを提供する:
  GET /ui/summary  - サマリKPI
  GET /ui/monthly  - 月別ROI一覧
  GET /ui/bets     - ベット一覧（フィルタ・ページネーション付き）
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"])

# backtest_result.json のパス（プロジェクトルート/data/）
_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_result.json"

# ページネーションのデフォルト件数
_PAGE_SIZE = 30


@lru_cache(maxsize=1)
def _load_result() -> dict:
    """バックテスト結果JSONを読み込む（起動後は初回のみ）"""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(f"バックテスト結果ファイルが見つかりません: {_DATA_PATH}")
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _get_result() -> dict:
    """結果取得（ファイル不在は503で返す）"""
    try:
        return _load_result()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise HTTPException(
            status_code=503,
            detail=(
                "バックテスト結果ファイルが見つかりません。"
                " `data/backtest_result.json` を生成してから再度アクセスしてください。"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# GET /ui/summary
# ---------------------------------------------------------------------------


@router.get("/summary")
def get_summary() -> dict:
    """KPIカード用サマリを返す"""
    data = _get_result()
    return data.get("summary", {})


# ---------------------------------------------------------------------------
# GET /ui/monthly
# ---------------------------------------------------------------------------


@router.get("/monthly")
def get_monthly() -> list:
    """月別ROI一覧を返す"""
    data = _get_result()
    return data.get("monthly", [])


# ---------------------------------------------------------------------------
# GET /ui/bets
# ---------------------------------------------------------------------------


@router.post("/reload")
def reload_result() -> dict:
    """キャッシュをクリアして最新のJSONを再読み込みする"""
    _load_result.cache_clear()
    try:
        _load_result()
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return {"status": "warn", "message": "ファイルが見つかりません"}
    return {"status": "ok", "message": "再読み込み完了"}


# ---------------------------------------------------------------------------
# GET /ui/bets
# ---------------------------------------------------------------------------


@router.get("/bets")
def get_bets(
    page: int = Query(default=1, ge=1, description="ページ番号（1始まり）"),
    hit: Literal["all", "hit", "miss"] = Query(
        default="all", description="フィルタ: all=全件, hit=的中, miss=外れ"
    ),
) -> dict:
    """ベット一覧を返す（ページネーション・フィルタ付き）"""
    data = _get_result()
    bets: list[dict] = data.get("bets", [])

    # フィルタ
    if hit == "hit":
        bets = [b for b in bets if b.get("is_hit")]
    elif hit == "miss":
        bets = [b for b in bets if not b.get("is_hit")]

    # ページネーション
    total = len(bets)
    start = (page - 1) * _PAGE_SIZE
    end = start + _PAGE_SIZE
    paged = bets[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": _PAGE_SIZE,
        "total_pages": max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE),
        "items": paged,
    }
