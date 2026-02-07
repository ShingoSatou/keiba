-- =============================================================================
-- MART層: 特徴量テーブル定義
-- 
-- 学習・推論で使用する特徴量を格納するスキーマ。
-- core層のデータを加工して生成する。
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 条件別タイム統計（グループ統計）
-- speed_index, closing_index 計算の基準値
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.time_stats (
    track_code SMALLINT NOT NULL,
    surface SMALLINT NOT NULL,           -- 1:芝, 2:ダート
    distance_bucket SMALLINT NOT NULL,   -- 1000,1200,1400,1600,...
    going_bucket SMALLINT NOT NULL,      -- 1:良系(良/稍重), 2:道悪系(重/不良)
    mu_time NUMERIC(6,2),                -- 走破タイム平均 (秒)
    sd_time NUMERIC(6,2),                -- 走破タイム標準偏差
    mu_final3f NUMERIC(5,2),             -- 上がり3F平均 (秒)
    sd_final3f NUMERIC(5,2),             -- 上がり3F標準偏差
    sample_count INT NOT NULL,           -- サンプル数
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (track_code, surface, distance_bucket, going_bucket)
);

COMMENT ON TABLE mart.time_stats IS 'グループ別タイム統計。speed_index計算の基準値。';

-- -----------------------------------------------------------------------------
-- 1走ごとの基礎指標
-- 過去レース（馬×レース）ごとに計算した指標
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.run_index (
    race_id BIGINT NOT NULL,
    horse_id TEXT NOT NULL,
    speed_index NUMERIC(6,3),            -- 標準化スピード (大きいほど速い)
    closing_index NUMERIC(6,3),          -- 標準化上がり (大きいほど速い)
    early_index NUMERIC(6,3),            -- 先行力 (大きいほど前に行く)
    position_gain SMALLINT,              -- 4角→着順の改善 (マイナス=差しが決まる)
    closing_missing BOOLEAN NOT NULL DEFAULT FALSE,  -- 上がりデータ欠損フラグ
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id, horse_id) REFERENCES core.runner(race_id, horse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_index_horse ON mart.run_index(horse_id);

COMMENT ON TABLE mart.run_index IS '1走ごとの基礎指標。近走集計のベース。';

-- -----------------------------------------------------------------------------
-- 馬単位の近走集計（全条件版 + 条件近似版）
-- calc_date + horse_id + 条件キー でユニーク
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.horse_stats (
    calc_date DATE NOT NULL,              -- 計算基準日（この日のレースで使う特徴量）
    horse_id TEXT NOT NULL,
    -- 全条件版
    speed_last NUMERIC(6,3),
    speed_mean_3 NUMERIC(6,3),
    speed_best_5 NUMERIC(6,3),
    speed_std_5 NUMERIC(6,3),
    speed_trend_3 NUMERIC(6,3),           -- 直近3走の傾き
    closing_last NUMERIC(6,3),
    closing_mean_3 NUMERIC(6,3),
    closing_best_5 NUMERIC(6,3),
    early_mean_3 NUMERIC(6,3),
    early_best_5 NUMERIC(6,3),
    position_gain_mean_3 NUMERIC(6,3),
    finish_mean_3 NUMERIC(6,3),
    finish_best_5 SMALLINT,
    n_runs_5 SMALLINT NOT NULL,           -- 過去5走の有効走数 (0-5)
    -- 条件近似版 (target条件)
    target_surface SMALLINT NOT NULL,     -- 対象レースの芝/ダ
    target_distance_bucket SMALLINT NOT NULL,  -- 対象レースの距離バケット
    target_going_bucket SMALLINT NOT NULL,     -- 対象レースの馬場バケット
    speed_sim_mean_3 NUMERIC(6,3),
    speed_sim_best_5 NUMERIC(6,3),
    closing_sim_mean_3 NUMERIC(6,3),
    closing_sim_best_5 NUMERIC(6,3),
    early_sim_mean_3 NUMERIC(6,3),
    n_sim_runs_5 SMALLINT NOT NULL,       -- 条件近似走の有効走数 (0-5)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (calc_date, horse_id, target_surface, target_distance_bucket, target_going_bucket)
);

CREATE INDEX IF NOT EXISTS idx_horse_stats_horse ON mart.horse_stats(horse_id);
CREATE INDEX IF NOT EXISTS idx_horse_stats_date ON mart.horse_stats(calc_date);

COMMENT ON TABLE mart.horse_stats IS '馬単位の近走集計。全条件版と条件近似版を含む。';

-- -----------------------------------------------------------------------------
-- 騎手・調教師の過去実績
-- calc_date + person_type + person_id でユニーク
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.person_stats (
    calc_date DATE NOT NULL,
    person_type TEXT NOT NULL,            -- 'jockey' or 'trainer'
    person_id BIGINT NOT NULL,
    win_rate_1y NUMERIC(5,4),             -- 過去1年勝率
    place_rate_1y NUMERIC(5,4),           -- 過去1年3着内率
    sample_count_1y INT NOT NULL,         -- 過去1年の騎乗/管理数
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (calc_date, person_type, person_id)
);

CREATE INDEX IF NOT EXISTS idx_person_stats_person ON mart.person_stats(person_type, person_id);

COMMENT ON TABLE mart.person_stats IS '騎手・調教師の過去実績統計。';

-- -----------------------------------------------------------------------------
-- 予測ログ
-- 推奨結果と実際の結果を蓄積
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mart.prediction_log (
    log_id BIGSERIAL PRIMARY KEY,
    predicted_at TIMESTAMPTZ NOT NULL,    -- 予測実行時刻
    race_id BIGINT NOT NULL,
    horse_no SMALLINT NOT NULL,
    horse_name TEXT,
    p_win NUMERIC(5,4),                   -- 予測勝率
    odds_10min NUMERIC(8,2),              -- 10分前オッズ (入力値)
    odds_effective NUMERIC(8,2),          -- スリッページ考慮後オッズ
    ev_profit NUMERIC(8,4),               -- 期待利益
    recommendation TEXT NOT NULL,         -- 'buy' or 'skip'
    bet_amount INT,                       -- 賭け金 (buyの場合)
    -- 結果（レース終了後に埋める）
    odds_final NUMERIC(8,2),              -- 最終オッズ
    finish_pos SMALLINT,                  -- 着順
    payout_yen INT,                       -- 払戻金
    profit_yen INT,                       -- 損益
    result_updated_at TIMESTAMPTZ         -- 結果更新時刻
);

CREATE INDEX IF NOT EXISTS idx_prediction_log_race ON mart.prediction_log(race_id);
CREATE INDEX IF NOT EXISTS idx_prediction_log_date ON mart.prediction_log(predicted_at);

COMMENT ON TABLE mart.prediction_log IS '予測ログ。バックテスト・運用評価に使用。';

-- =============================================================================
-- 完了メッセージ
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'mart スキーマの特徴量テーブル初期化完了';
END $$;
