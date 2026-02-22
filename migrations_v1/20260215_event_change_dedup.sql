-- =============================================================================
-- Migration: core.event_change の冪等化（重複排除）
-- Date: 2026-02-15
--
-- 目的:
--   - RT再投入時に core.event_change が重複増殖しないようにする
--   - payload_md5 を導入し、同一イベントの重複を排除する
-- =============================================================================

BEGIN;

ALTER TABLE core.event_change
    ADD COLUMN IF NOT EXISTS payload_md5 CHAR(32);

UPDATE core.event_change
SET payload_md5 = md5(COALESCE(payload_parsed->>'raw', ''))
WHERE payload_md5 IS NULL;

DELETE FROM core.event_change t
USING (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY race_id, record_type, data_create_ymd, announce_mmddhhmi, payload_md5
            ORDER BY id
        ) AS rn
    FROM core.event_change
) d
WHERE t.id = d.id
  AND d.rn > 1;

ALTER TABLE core.event_change
    ALTER COLUMN payload_md5 SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_event_change_dedup
    ON core.event_change (
        race_id, record_type, data_create_ymd, announce_mmddhhmi, payload_md5
    );

COMMIT;
