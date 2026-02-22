BEGIN;

-- =============================================================================
-- O1: 単勝オッズ時系列（0B41 / RACE O1）
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.o1_header (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    win_pool_total_100yen BIGINT,
    data_create_ymd CHAR(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi)
);

CREATE INDEX IF NOT EXISTS idx_o1_header_pick
    ON core.o1_header (race_id, data_kbn, announce_mmddhhmi);

CREATE TABLE IF NOT EXISTS core.o1_win (
    race_id BIGINT NOT NULL,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    horse_no SMALLINT NOT NULL,
    win_odds_x10 INTEGER,
    win_popularity SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi, horse_no),
    FOREIGN KEY (race_id, data_kbn, announce_mmddhhmi)
        REFERENCES core.o1_header (race_id, data_kbn, announce_mmddhhmi) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_o1_win_race ON core.o1_win (race_id);
CREATE INDEX IF NOT EXISTS idx_o1_win_pick ON core.o1_win (race_id, horse_no);

-- =============================================================================
-- WH: 馬体重速報（0B11）
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.wh_header (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    data_create_ymd CHAR(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi)
);

CREATE TABLE IF NOT EXISTS core.wh_detail (
    race_id BIGINT NOT NULL,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    horse_no SMALLINT NOT NULL,
    body_weight_kg SMALLINT,
    diff_sign CHAR(1),
    diff_kg SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi, horse_no),
    FOREIGN KEY (race_id, data_kbn, announce_mmddhhmi)
        REFERENCES core.wh_header (race_id, data_kbn, announce_mmddhhmi) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_wh_detail_race ON core.wh_detail (race_id);
CREATE INDEX IF NOT EXISTS idx_wh_detail_pick ON core.wh_detail (race_id, horse_no);

-- =============================================================================
-- 速報/確定データマイニング（MING / 0B13 / 0B17）
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.mining_dm (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    horse_no SMALLINT NOT NULL,
    data_kbn SMALLINT NOT NULL,
    dm_time_x10 INTEGER,
    dm_rank SMALLINT,
    payload_raw TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_mining_dm_race ON core.mining_dm (race_id);

CREATE TABLE IF NOT EXISTS core.mining_tm (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    horse_no SMALLINT NOT NULL,
    data_kbn SMALLINT NOT NULL,
    tm_score INTEGER,
    tm_rank SMALLINT,
    payload_raw TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_mining_tm_race ON core.mining_tm (race_id);

CREATE TABLE IF NOT EXISTS core.rt_mining_dm (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    data_kbn SMALLINT NOT NULL,
    data_create_ymd CHAR(8) NOT NULL,
    data_create_hm CHAR(4) NOT NULL,
    horse_no SMALLINT NOT NULL,
    dm_time_x10 INTEGER,
    dm_rank SMALLINT,
    payload_raw TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_rt_mining_dm_race ON core.rt_mining_dm (race_id);
CREATE INDEX IF NOT EXISTS idx_rt_mining_dm_time
    ON core.rt_mining_dm (race_id, data_create_ymd, data_create_hm);

CREATE TABLE IF NOT EXISTS core.rt_mining_tm (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    data_kbn SMALLINT NOT NULL,
    data_create_ymd CHAR(8) NOT NULL,
    data_create_hm CHAR(4) NOT NULL,
    horse_no SMALLINT NOT NULL,
    tm_score INTEGER,
    tm_rank SMALLINT,
    payload_raw TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, data_create_ymd, data_create_hm, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_rt_mining_tm_race ON core.rt_mining_tm (race_id);
CREATE INDEX IF NOT EXISTS idx_rt_mining_tm_time
    ON core.rt_mining_tm (race_id, data_create_ymd, data_create_hm);

-- =============================================================================
-- 当日変更（0B14: WE / AV / JC / TC / CC）
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.event_change (
    id BIGSERIAL PRIMARY KEY,
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    record_type CHAR(2) NOT NULL,
    data_kbn SMALLINT,
    data_create_ymd CHAR(8) NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    payload_parsed JSONB NOT NULL,
    payload_md5 CHAR(32) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (race_id, record_type, data_create_ymd, announce_mmddhhmi, payload_md5)
);

CREATE INDEX IF NOT EXISTS idx_event_change_race
    ON core.event_change (race_id, record_type, announce_mmddhhmi);

COMMIT;
