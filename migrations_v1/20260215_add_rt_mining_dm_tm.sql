-- =============================================================================
-- Migration: リアルタイム DM/TM 履歴テーブル追加
-- Date: 2026-02-15
--
-- 目的:
--   - 速報DM(0B13) / 速報TM(0B17) を履歴保持する
--   - T-5 as-of で kbn優先(3>2>1) + 作成時刻最大を再現可能にする
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS core.rt_mining_dm (
    race_id          BIGINT NOT NULL,
    data_kbn         SMALLINT NOT NULL,
    data_create_ymd  CHAR(8) NOT NULL,
    data_create_hm   CHAR(4) NOT NULL,
    horse_no         SMALLINT NOT NULL,
    dm_time_x10      INTEGER,
    dm_rank          SMALLINT,
    payload_raw      TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_rt_mining_dm_race
    ON core.rt_mining_dm (race_id);

CREATE INDEX IF NOT EXISTS idx_rt_mining_dm_pick
    ON core.rt_mining_dm (race_id, data_kbn, data_create_ymd, data_create_hm);

CREATE INDEX IF NOT EXISTS idx_rt_mining_dm_race_horse
    ON core.rt_mining_dm (race_id, horse_no);

CREATE TABLE IF NOT EXISTS core.rt_mining_tm (
    race_id          BIGINT NOT NULL,
    data_kbn         SMALLINT NOT NULL,
    data_create_ymd  CHAR(8) NOT NULL,
    data_create_hm   CHAR(4) NOT NULL,
    horse_no         SMALLINT NOT NULL,
    tm_score         INTEGER,
    tm_rank          SMALLINT,
    payload_raw      TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_rt_mining_tm_race
    ON core.rt_mining_tm (race_id);

CREATE INDEX IF NOT EXISTS idx_rt_mining_tm_pick
    ON core.rt_mining_tm (race_id, data_kbn, data_create_ymd, data_create_hm);

CREATE INDEX IF NOT EXISTS idx_rt_mining_tm_race_horse
    ON core.rt_mining_tm (race_id, horse_no);

COMMIT;
