-- =============================================================================
-- Migration: mart.t5_runner_snapshot 追加
-- Date: 2026-02-14
--
-- 目的:
--   - T-5(as-of) 入力を「1行=1頭」で再現可能にする
--   - 採用キーを含む監査可能な土台を mart に用意する
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS mart.t5_runner_snapshot (
    race_id BIGINT NOT NULL,
    race_date DATE NOT NULL,
    track_code SMALLINT NOT NULL,
    race_no SMALLINT NOT NULL,
    horse_id TEXT NOT NULL,
    horse_no SMALLINT NOT NULL,
    gate SMALLINT,
    jockey_id BIGINT,
    trainer_id BIGINT,
    carried_weight NUMERIC(4,1),
    body_weight_asof SMALLINT,
    body_weight_diff_asof SMALLINT,
    bw_source TEXT NOT NULL,
    post_time TIME NOT NULL,
    asof_ts TIMESTAMP NOT NULL,
    feature_set TEXT NOT NULL,
    o1_data_kbn SMALLINT,
    o1_announce_mmddhhmi CHAR(8),
    odds_win_t5 NUMERIC(8,2),
    pop_win_t5 SMALLINT,
    odds_rank_t5 SMALLINT,
    win_pool_total_100yen_t5 BIGINT,
    odds_snapshot_age_sec INT,
    odds_missing_flag BOOLEAN NOT NULL DEFAULT FALSE,
    odds_win_final NUMERIC(8,2),
    pop_win_final SMALLINT,
    wh_announce_mmddhhmi CHAR(8),
    event_change_keys JSONB,
    dm_kbn SMALLINT,
    dm_create_time CHAR(12),
    tm_kbn SMALLINT,
    tm_create_time CHAR(12),
    code_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_no, asof_ts)
);

CREATE INDEX IF NOT EXISTS idx_t5_snapshot_race_date ON mart.t5_runner_snapshot(race_date, race_id);
CREATE INDEX IF NOT EXISTS idx_t5_snapshot_asof ON mart.t5_runner_snapshot(asof_ts);
CREATE INDEX IF NOT EXISTS idx_t5_snapshot_feature_set ON mart.t5_runner_snapshot(feature_set);

COMMIT;
