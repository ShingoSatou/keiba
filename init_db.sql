-- Create table for raw JV-Link data
CREATE TABLE IF NOT EXISTS jv_raw (
  id            BIGSERIAL PRIMARY KEY,
  rec_id        CHAR(2) NOT NULL,
  filename      TEXT,
  dataspec      TEXT,
  payload       TEXT NOT NULL,
  ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_jv_raw_recid ON jv_raw(rec_id);
CREATE INDEX IF NOT EXISTS idx_jv_raw_ingested ON jv_raw(ingested_at);
