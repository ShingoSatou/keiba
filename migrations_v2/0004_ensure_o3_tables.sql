BEGIN;

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
