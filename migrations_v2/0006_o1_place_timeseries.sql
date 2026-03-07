BEGIN;

ALTER TABLE core.o1_header
    ADD COLUMN IF NOT EXISTS sale_flag_place SMALLINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS place_pay_key SMALLINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS place_pool_total_100yen BIGINT;

CREATE TABLE IF NOT EXISTS core.o1_place (
    race_id BIGINT NOT NULL,
    data_kbn SMALLINT NOT NULL,
    announce_mmddhhmi CHAR(8) NOT NULL,
    horse_no SMALLINT NOT NULL,
    min_odds_x10 INTEGER,
    max_odds_x10 INTEGER,
    place_popularity SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi, horse_no),
    FOREIGN KEY (race_id, data_kbn, announce_mmddhhmi)
        REFERENCES core.o1_header (race_id, data_kbn, announce_mmddhhmi) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_o1_place_race ON core.o1_place (race_id);
CREATE INDEX IF NOT EXISTS idx_o1_place_pick ON core.o1_place (race_id, horse_no);

COMMIT;
