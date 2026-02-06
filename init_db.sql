-- =============================================================================
-- JRA-VAN データ格納用 PostgreSQL スキーマ (MVP + 拡張対応)
-- 
-- 3層構造:
--   raw  : JV-Linkから取得した生データ（再パース可能な保険）
--   core : 正規化された事実テーブル＋マスタ
--   mart : 学習・分析用（後で追加）
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema定義
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS mart;

-- =============================================================================
-- RAW層: JV-Linkの生データをそのまま格納
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw.jv_raw (
    id           BIGSERIAL PRIMARY KEY,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataspec     TEXT NOT NULL,               -- JVOpenで指定したDataSpec (例: "RACE", "DIFF")
    rec_id       CHAR(2) NOT NULL,            -- レコード種別 (payload先頭2文字: RA, SE, HR等)
    filename     TEXT,                        -- JVReadから取得したファイル名
    payload      TEXT NOT NULL,               -- 固定長文字列（Shift_JIS→UTF-8変換後）
    payload_hash BYTEA                        -- 重複排除用ハッシュ（任意）
);

CREATE INDEX IF NOT EXISTS idx_jv_raw_ingested_at ON raw.jv_raw(ingested_at);
CREATE INDEX IF NOT EXISTS idx_jv_raw_dataspec_recid ON raw.jv_raw(dataspec, rec_id);

-- =============================================================================
-- CORE層: 正規化された事実テーブル
-- =============================================================================

-- -----------------------------------------------------------------------------
-- レース (RA)
-- race_id = YYYYMMDD * 10000 + track_code * 100 + race_no で一意生成
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.race (
    race_id      BIGINT PRIMARY KEY,
    race_date    DATE NOT NULL,
    track_code   SMALLINT NOT NULL,           -- 競馬場コード (01:札幌, 02:函館, ...)
    race_no      SMALLINT NOT NULL,           -- R番号 (1-12)
    surface      SMALLINT NOT NULL,           -- 1:芝, 2:ダート, 3:障害
    distance_m   SMALLINT NOT NULL,           -- 距離(m)
    going        SMALLINT,                    -- 馬場状態 (1:良, 2:稍, 3:重, 4:不良)
    weather      SMALLINT,                    -- 天候コード
    class_code   SMALLINT,                    -- クラス (新馬, 未勝利, 1勝C, OP, 重賞等)
    turn_dir     SMALLINT,                    -- 1:右, 2:左, 3:直線
    course_inout SMALLINT,                    -- コース (1:A, 2:B, 3:C等)
    field_size   SMALLINT,                    -- 頭数
    start_time   TIME,                        -- 発走時刻
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (race_date, track_code, race_no)
);

CREATE INDEX IF NOT EXISTS idx_race_date_track ON core.race(race_date, track_code);

-- -----------------------------------------------------------------------------
-- 馬マスタ (UM)
-- horse_id = JRA-VANの血統登録番号（旧8桁/新10桁混在のためTEXT型）
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.horse (
    horse_id     TEXT PRIMARY KEY,            -- 血統登録番号 (8-10桁, ゼロ埋め保持)
    horse_name   TEXT,
    sex          SMALLINT,                    -- 1:牡, 2:牝, 3:騸
    birth_date   DATE,
    coat_color   SMALLINT,                    -- 毛色コード（任意）
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 騎手マスタ (KS) - 拡張用
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.jockey (
    jockey_id    BIGINT PRIMARY KEY,
    jockey_name  TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 調教師マスタ (CH) - 拡張用
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.trainer (
    trainer_id   BIGINT PRIMARY KEY,
    trainer_name TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 出走馬 (SE: 出走表部分)
-- レース確定前から存在するデータ。結果はresultテーブルへ分離。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.runner (
    race_id          BIGINT NOT NULL REFERENCES core.race(race_id) ON DELETE CASCADE,
    horse_id         TEXT NOT NULL REFERENCES core.horse(horse_id),
    horse_no         SMALLINT NOT NULL,       -- 馬番
    gate             SMALLINT,                -- 枠番
    jockey_id        BIGINT REFERENCES core.jockey(jockey_id),
    trainer_id       BIGINT REFERENCES core.trainer(trainer_id),
    carried_weight   NUMERIC(4,1),            -- 斤量 (kg)
    body_weight      SMALLINT,                -- 馬体重 (kg)
    body_weight_diff SMALLINT,                -- 増減 (kg)
    scratch_flag     BOOLEAN NOT NULL DEFAULT FALSE,  -- 出走取消フラグ
    entry_status     SMALLINT,                -- 出走確定/除外 コード
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_id),
    UNIQUE (race_id, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_runner_horse ON core.runner(horse_id);
CREATE INDEX IF NOT EXISTS idx_runner_race ON core.runner(race_id);

-- -----------------------------------------------------------------------------
-- 結果 (SE: 結果部分)
-- 着順・タイム・上がり3F・通過順など、MVP特徴量の核
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.result (
    race_id        BIGINT NOT NULL,
    horse_id       TEXT NOT NULL,
    finish_pos     SMALLINT,                  -- 着順 (取消/中止等はNULL)
    time_sec       NUMERIC(6,2),              -- 走破タイム (秒)  例: 118.50
    margin         TEXT,                      -- 着差 (文字列で保持、後で数値変換)
    final3f_sec    NUMERIC(5,2),              -- 上がり3F (秒)
    corner1_pos    SMALLINT,                  -- 1コーナー通過順
    corner2_pos    SMALLINT,                  -- 2コーナー通過順
    corner3_pos    SMALLINT,                  -- 3コーナー通過順
    corner4_pos    SMALLINT,                  -- 4コーナー通過順
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id, horse_id) REFERENCES core.runner(race_id, horse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_result_finish_pos ON core.result(race_id, finish_pos);

-- -----------------------------------------------------------------------------
-- ラップタイム (拡張用 - ペース分析)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.race_lap (
    race_id     BIGINT NOT NULL REFERENCES core.race(race_id) ON DELETE CASCADE,
    lap_index   SMALLINT NOT NULL,            -- ラップ番号 (1, 2, 3...)
    lap_sec     NUMERIC(5,2) NOT NULL,        -- ラップタイム (秒)
    PRIMARY KEY (race_id, lap_index)
);

-- -----------------------------------------------------------------------------
-- 払戻金 (HR)
-- MVP必須: 単勝払戻。縦持ちで全券種対応。
-- -----------------------------------------------------------------------------
-- bet_type: 1=単勝, 2=複勝, 3=枠連, 4=馬連, 5=ワイド, 6=馬単, 7=三連複, 8=三連単
CREATE TABLE IF NOT EXISTS core.payout (
    race_id     BIGINT NOT NULL REFERENCES core.race(race_id) ON DELETE CASCADE,
    bet_type    SMALLINT NOT NULL,
    selection   TEXT NOT NULL,                -- 単勝: "3", 馬連: "3-7", 三連単: "3-7-1"
    payout_yen  INTEGER NOT NULL,             -- 払戻額 (円)
    popularity  SMALLINT,                     -- 人気
    PRIMARY KEY (race_id, bet_type, selection)
);

CREATE INDEX IF NOT EXISTS idx_payout_race_bet ON core.payout(race_id, bet_type);

-- -----------------------------------------------------------------------------
-- 最終オッズ (O1確定)
-- MVP必須: 単勝確定オッズ
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.odds_final (
    race_id         BIGINT NOT NULL,
    horse_id        TEXT NOT NULL,
    odds_win        NUMERIC(8,2),             -- 単勝オッズ
    pop_win         SMALLINT,                 -- 単勝人気
    odds_place_low  NUMERIC(8,2),             -- 複勝下限
    odds_place_high NUMERIC(8,2),             -- 複勝上限
    pop_place       SMALLINT,                 -- 複勝人気
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id, horse_id) REFERENCES core.runner(race_id, horse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_odds_final_race ON core.odds_final(race_id);

-- -----------------------------------------------------------------------------
-- オッズスナップショット / 時系列 (拡張用)
-- 発走10分前オッズ、時系列オッズなど
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.odds_snapshot (
    race_id       BIGINT NOT NULL,
    horse_id      TEXT NOT NULL,
    collected_at  TIMESTAMPTZ NOT NULL,       -- 観測時刻
    odds_win      NUMERIC(8,2),
    pop_win       SMALLINT,
    vote_count    INTEGER,                    -- 票数
    pool_yen      BIGINT,                     -- 売上
    PRIMARY KEY (race_id, horse_id, collected_at),
    FOREIGN KEY (race_id, horse_id) REFERENCES core.runner(race_id, horse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshot_race_time ON core.odds_snapshot(race_id, collected_at);

-- -----------------------------------------------------------------------------
-- 変更・訂正イベントログ (DIFF)
-- 運用安定化: 出走取消、騎手変更、降着など
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.diff_event (
    diff_id       BIGSERIAL PRIMARY KEY,
    race_id       BIGINT REFERENCES core.race(race_id),
    event_time    TIMESTAMPTZ NOT NULL DEFAULT now(),
    diff_type     TEXT NOT NULL,              -- "SCRATCH", "JOCKEY_CHANGE", "WEIGHT_CHANGE" 等
    subject_key   TEXT,                       -- horse_id, horse_no などを文字で保持
    old_value     TEXT,
    new_value     TEXT,
    source_raw_id BIGINT REFERENCES raw.jv_raw(id)  -- 元レコードへのリンク
);

CREATE INDEX IF NOT EXISTS idx_diff_race_time ON core.diff_event(race_id, event_time);

-- =============================================================================
-- 完了メッセージ
-- =============================================================================
-- 実行完了後の確認用
DO $$
BEGIN
    RAISE NOTICE 'スキーマ初期化完了: raw, core (mart は後で追加)';
END $$;
