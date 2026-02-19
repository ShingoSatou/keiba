"""
取り込みサービスモジュール

JV-Linkからデータを取得し、PostgreSQLに保存するサービス。
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from app.infrastructure.database import Database
from app.infrastructure.parsers import RaceRecord

if TYPE_CHECKING:
    from app.infrastructure.jvlink import JVLinkClient

logger = logging.getLogger(__name__)


class IngestService:
    """
    JV-Linkからのデータ取り込みサービス

    使用例:
        with JVLinkClient() as jv:
            service = IngestService(jv, db)
            service.ingest_race_data("20260101", "20260131")
    """

    def __init__(self, jvlink: JVLinkClient, database: Database):
        self.jv = jvlink
        self.db = database

    def ingest_raw(self, dataspec: str, from_date: str, to_date: str | None = None) -> int:
        """
        生データをraw.jv_rawテーブルに保存

        Args:
            dataspec: データ種別 (例: "RACE")
            from_date: 開始日 (YYYYMMDD)
            to_date: 終了日 (未使用、将来拡張用)

        Returns:
            取り込んだレコード数
        """
        self.jv.open(dataspec, from_date)
        count = 0

        insert_sql = """
            INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload, payload_hash)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (dataspec, rec_id, payload_hash) DO NOTHING
        """

        for record in self.jv.read_all():
            payload_hash = hashlib.sha256(record.payload.encode("utf-8", errors="replace")).digest()
            self.db.execute(
                insert_sql,
                (dataspec, record.rec_id, record.filename, record.payload, payload_hash),
            )
            count += 1

            if count % 1000 == 0:
                logger.info(f"Ingested {count} records...")
                self.db.connect().commit()

        self.db.connect().commit()
        logger.info(f"Ingested total {count} records for {dataspec}")
        return count

    def process_raw_to_core(self, batch_size: int = 1000) -> dict[str, int]:
        """
        raw.jv_rawからcoreスキーマのテーブルへ変換・保存

        Returns:
            各テーブルへの挿入件数
        """
        stats = {"race": 0, "runner": 0, "result": 0, "payout": 0}

        # RAレコードの処理
        rows = self.db.fetch_all(
            "SELECT id, payload FROM raw.jv_raw WHERE rec_id = 'RA' ORDER BY id LIMIT %s",
            (batch_size,),
        )

        for row in rows:
            try:
                race = RaceRecord.parse(row["payload"])
                self._upsert_race(race)
                stats["race"] += 1
            except Exception as e:
                logger.warning(f"Failed to parse RA record id={row['id']}: {e}")

        self.db.connect().commit()
        return stats

    def _upsert_race(self, race: RaceRecord) -> None:
        """レースをcoreスキーマに保存"""
        sql = """
            INSERT INTO core.race (
                race_id, race_date, track_code, race_no, surface,
                distance_m, going, weather, class_code, field_size, start_time
            ) VALUES (
                %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, %(surface)s,
                %(distance_m)s, %(going)s, %(weather)s,
                %(class_code)s, %(field_size)s, %(start_time)s
            )
            ON CONFLICT (race_id) DO UPDATE SET
                surface = CASE
                    WHEN EXCLUDED.surface > 0
                        AND (
                            core.race.surface = 0
                            OR core.race.surface IS NULL
                            OR core.race.distance_m = 0
                            OR core.race.distance_m IS NULL
                            OR core.race.start_time IS NULL
                        )
                    THEN EXCLUDED.surface
                    ELSE core.race.surface
                END,
                distance_m = CASE
                    WHEN EXCLUDED.distance_m > 0
                        AND (core.race.distance_m = 0 OR core.race.distance_m IS NULL)
                    THEN EXCLUDED.distance_m
                    ELSE core.race.distance_m
                END,
                going = EXCLUDED.going,
                weather = EXCLUDED.weather,
                field_size = EXCLUDED.field_size,
                start_time = COALESCE(EXCLUDED.start_time, core.race.start_time),
                updated_at = now()
        """
        self.db.execute(
            sql,
            {
                "race_id": race.race_id,
                "race_date": race.race_date,
                "track_code": race.track_code,
                "race_no": race.race_no,
                "surface": race.surface,
                "distance_m": race.distance_m,
                "going": race.going,
                "weather": race.weather,
                "class_code": race.class_code,
                "field_size": race.field_size,
                "start_time": race.start_time,
            },
        )


def run_daily_ingest(from_date: str) -> None:
    """
    日次取り込みを実行するエントリーポイント

    Args:
        from_date: 取り込み開始日 (YYYYMMDD)
    """
    # Windows環境チェック
    import sys

    if sys.platform != "win32":
        raise RuntimeError("This function requires Windows with JV-Link installed")

    from app.infrastructure.jvlink import JVLinkClient

    with JVLinkClient() as jv:
        with Database() as db:
            service = IngestService(jv, db)

            # 生データ取り込み
            logger.info(f"Starting raw data ingestion from {from_date}")
            service.ingest_raw("RACE", from_date)

            # Core変換
            logger.info("Processing raw data to core tables")
            stats = service.process_raw_to_core()
            logger.info(f"Processing complete: {stats}")
