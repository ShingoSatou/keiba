-- =============================================================================
-- Migration: raw.jv_raw の冪等化 (payload_hash 運用)
-- Date: 2026-02-13
--
-- 目的:
--   - payload_hash(sha256) を全行に埋める
--   - (dataspec, rec_id, payload_hash) で重複排除
--   - NOT NULL + UNIQUE 制約を追加
--
-- 注意:
--   - 大量データがある環境では時間/ロックが発生します。オフピーク推奨。
--   - sha256 は「payload(TEXT) を UTF-8 バイト列にしたもの」を対象にします。
-- =============================================================================

BEGIN;

-- sha256(digest) 用
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 旧スキーマ互換: カラムがない場合は追加
ALTER TABLE raw.jv_raw
  ADD COLUMN IF NOT EXISTS payload_hash BYTEA;

-- 1) payload_hash を全行埋める（NULLのみ）
UPDATE raw.jv_raw
SET payload_hash = digest(convert_to(payload, 'UTF8'), 'sha256')
WHERE payload_hash IS NULL;

-- 2) (dataspec, rec_id, payload_hash) で重複削除（最小idを残す）
WITH ranked AS (
  SELECT
    id,
    row_number() OVER (
      PARTITION BY dataspec, rec_id, payload_hash
      ORDER BY id
    ) AS rn
  FROM raw.jv_raw
)
DELETE FROM raw.jv_raw r
USING ranked d
WHERE r.id = d.id
  AND d.rn > 1;

-- 3) NOT NULL 化
ALTER TABLE raw.jv_raw
  ALTER COLUMN payload_hash SET NOT NULL;

-- 4) UNIQUE 制約追加（既にある場合はスキップ）
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'uq_jv_raw_dedup'
      AND conrelid = 'raw.jv_raw'::regclass
  ) THEN
    ALTER TABLE raw.jv_raw
      ADD CONSTRAINT uq_jv_raw_dedup UNIQUE (dataspec, rec_id, payload_hash);
  END IF;
END $$;

COMMIT;

