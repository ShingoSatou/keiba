-- =============================================================================
-- v2 Initial Schema (raw/core/mart)
-- =============================================================================

BEGIN;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS mart;

-- =============================================================================
-- RAW
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw.jv_raw (
    id BIGSERIAL PRIMARY KEY,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataspec TEXT NOT NULL,
    rec_id CHAR(2) NOT NULL,
    filename TEXT,
    payload TEXT NOT NULL,
    payload_hash BYTEA NOT NULL,
    CONSTRAINT uq_jv_raw_dedup UNIQUE (dataspec, rec_id, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_jv_raw_ingested_at ON raw.jv_raw (ingested_at);
CREATE INDEX IF NOT EXISTS idx_jv_raw_dataspec_recid ON raw.jv_raw (dataspec, rec_id);

-- =============================================================================
-- CORE
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.race (
    race_id BIGINT PRIMARY KEY,
    race_date DATE NOT NULL,
    track_code SMALLINT NOT NULL,
    race_no SMALLINT NOT NULL,
    surface SMALLINT NOT NULL,
    distance_m SMALLINT NOT NULL,
    going SMALLINT,
    weather SMALLINT,
    class_code SMALLINT,
    turn_dir SMALLINT,
    course_inout SMALLINT,
    field_size SMALLINT,
    start_time TIME,
    grade_code SMALLINT,
    race_type_code SMALLINT,
    weight_type_code SMALLINT,
    condition_code_min_age SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (race_date, track_code, race_no)
);

CREATE INDEX IF NOT EXISTS idx_race_date_track ON core.race (race_date, track_code);

CREATE TABLE IF NOT EXISTS core.horse (
    horse_id TEXT PRIMARY KEY,
    horse_name TEXT,
    sex SMALLINT,
    birth_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS core.jockey (
    jockey_id BIGINT PRIMARY KEY,
    jockey_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS core.trainer (
    trainer_id BIGINT PRIMARY KEY,
    trainer_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS core.runner (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    horse_id TEXT NOT NULL REFERENCES core.horse (horse_id),
    horse_no SMALLINT NOT NULL,
    gate SMALLINT,
    jockey_id BIGINT REFERENCES core.jockey (jockey_id),
    trainer_id BIGINT REFERENCES core.trainer (trainer_id),
    carried_weight NUMERIC(4,1),
    body_weight SMALLINT,
    body_weight_diff SMALLINT,
    scratch_flag BOOLEAN NOT NULL DEFAULT FALSE,
    data_kubun CHAR(2),
    age SMALLINT,
    sex SMALLINT,
    horse_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_id),
    UNIQUE (race_id, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_runner_race ON core.runner (race_id);
CREATE INDEX IF NOT EXISTS idx_runner_horse ON core.runner (horse_id);

CREATE TABLE IF NOT EXISTS core.result (
    race_id BIGINT NOT NULL,
    horse_id TEXT NOT NULL,
    finish_pos SMALLINT,
    time_sec NUMERIC(6,2),
    margin TEXT,
    final3f_sec NUMERIC(5,2),
    corner1_pos SMALLINT,
    corner2_pos SMALLINT,
    corner3_pos SMALLINT,
    corner4_pos SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id, horse_id)
        REFERENCES core.runner (race_id, horse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_result_finish ON core.result (race_id, finish_pos);

CREATE TABLE IF NOT EXISTS core.payout (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    bet_type SMALLINT NOT NULL,
    selection TEXT NOT NULL,
    payout_yen INTEGER NOT NULL,
    popularity SMALLINT,
    PRIMARY KEY (race_id, bet_type, selection)
);

CREATE INDEX IF NOT EXISTS idx_payout_race_bet ON core.payout (race_id, bet_type);

CREATE TABLE IF NOT EXISTS core.o3_header (
    race_id BIGINT NOT NULL REFERENCES core.race (race_id) ON DELETE CASCADE,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    wide_pool_total_100yen BIGINT,
    starters SMALLINT,
    sale_flag_wide SMALLINT,
    data_create_ymd CHAR(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi)
);

CREATE INDEX IF NOT EXISTS idx_o3_header_pick ON core.o3_header (race_id, data_kbn, announce_mmddhhmi);

CREATE TABLE IF NOT EXISTS core.o3_wide (
    race_id BIGINT NOT NULL,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    kumiban CHAR(4) NOT NULL,
    min_odds_x10 INTEGER,
    max_odds_x10 INTEGER,
    popularity SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi, kumiban),
    FOREIGN KEY (race_id, data_kbn, announce_mmddhhmi)
        REFERENCES core.o3_header (race_id, data_kbn, announce_mmddhhmi) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_o3_wide_race ON core.o3_wide (race_id);
CREATE INDEX IF NOT EXISTS idx_o3_wide_kumiban ON core.o3_wide (race_id, kumiban);

COMMIT;
