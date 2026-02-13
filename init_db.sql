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
    payload_hash BYTEA NOT NULL,              -- 重複排除用ハッシュ（sha256, 常に埋める）
    CONSTRAINT uq_jv_raw_dedup UNIQUE (dataspec, rec_id, payload_hash)
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
-- CORE層: 時系列オッズ (O1) - 5分足の全データを格納
-- =============================================================================

-- O1ヘッダー（スナップショット単位）
CREATE TABLE IF NOT EXISTS core.o1_header (
    race_id               BIGINT NOT NULL REFERENCES core.race(race_id) ON DELETE CASCADE,
    data_kbn              SMALLINT NOT NULL,        -- 1=中間, 2=前日最終, 3=最終, 4=確定
    announce_mmddhhmi     CHAR(8) NOT NULL,         -- 発表月日時分 (中間のみキー、他は'00000000')
    win_pool_total_100yen BIGINT,                   -- 単勝票数合計（単位百円）
    place_pool_total_100yen BIGINT,                 -- 複勝票数合計
    data_create_ymd       CHAR(8),                  -- データ作成年月日
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi)
);

CREATE INDEX IF NOT EXISTS idx_o1_header_race_time ON core.o1_header(race_id, data_kbn, announce_mmddhhmi);

-- O1明細（馬番単位）
CREATE TABLE IF NOT EXISTS core.o1_win (
    race_id               BIGINT NOT NULL,
    data_kbn              SMALLINT NOT NULL,
    announce_mmddhhmi     CHAR(8) NOT NULL,
    horse_no              SMALLINT NOT NULL,
    win_odds_x10          INTEGER,                  -- オッズ×10 (12.3倍 → 123)
    win_popularity        SMALLINT,
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi, horse_no),
    FOREIGN KEY (race_id, data_kbn, announce_mmddhhmi)
        REFERENCES core.o1_header(race_id, data_kbn, announce_mmddhhmi) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_o1_win_race_horse ON core.o1_win(race_id, horse_no);

-- =============================================================================
-- CORE層: 馬体重速報 (WH)
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.wh_header (
    race_id               BIGINT NOT NULL REFERENCES core.race(race_id) ON DELETE CASCADE,
    data_kbn              SMALLINT NOT NULL,
    announce_mmddhhmi     CHAR(8) NOT NULL,
    data_create_ymd       CHAR(8),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi)
);

CREATE TABLE IF NOT EXISTS core.wh_detail (
    race_id               BIGINT NOT NULL,
    data_kbn              SMALLINT NOT NULL,
    announce_mmddhhmi     CHAR(8) NOT NULL,
    horse_no              SMALLINT NOT NULL,
    body_weight_kg        SMALLINT,
    diff_sign             CHAR(1),                  -- '+', '-', ' '
    diff_kg               SMALLINT,
    PRIMARY KEY (race_id, data_kbn, announce_mmddhhmi, horse_no),
    FOREIGN KEY (race_id, data_kbn, announce_mmddhhmi)
        REFERENCES core.wh_header(race_id, data_kbn, announce_mmddhhmi) ON DELETE CASCADE
);

-- =============================================================================
-- CORE層: 当日変更 (WE/AV/JC/TC/CC)
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.event_change (
    id                    BIGSERIAL PRIMARY KEY,
    race_id               BIGINT NOT NULL REFERENCES core.race(race_id) ON DELETE CASCADE,
    record_type           CHAR(2) NOT NULL,         -- 'WE','AV','JC','TC','CC'
    data_create_ymd       CHAR(8),
    announce_mmddhhmi     CHAR(8),
    payload_parsed        JSONB,                    -- 変更内容（種類で構造が変わるためJSON）
    received_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_event_change_race_type ON core.event_change(race_id, record_type, announce_mmddhhmi);

-- =============================================================================
-- CORE層: 坂路調教 (HC - SLOP)
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.training_slop (
    id                    BIGSERIAL PRIMARY KEY,
    horse_id              TEXT NOT NULL REFERENCES core.horse(horse_id),
    training_date         DATE NOT NULL,
    data_kbn              SMALLINT,
    training_center       CHAR(1),                  -- 0:美浦 1:栗東
    training_time         CHAR(4),                  -- hhmm
    total_4f              NUMERIC(4,1),
    lap_4f                NUMERIC(4,1),
    total_3f              NUMERIC(4,1),
    lap_3f                NUMERIC(4,1),
    total_2f              NUMERIC(4,1),
    lap_2f                NUMERIC(4,1),
    lap_1f                NUMERIC(4,1),             -- 1Fはラップのみ
    payload_raw           TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (horse_id, training_date, training_center, total_4f)
);

CREATE INDEX IF NOT EXISTS idx_training_slop_horse ON core.training_slop(horse_id, training_date);

-- =============================================================================
-- CORE層: ウッド調教 (WC - WOOD)
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.training_wood (
    id                    BIGSERIAL PRIMARY KEY,
    horse_id              TEXT NOT NULL REFERENCES core.horse(horse_id),
    training_date         DATE NOT NULL,
    data_kbn              SMALLINT,
    training_center       CHAR(1),                  -- 0:美浦 1:栗東
    training_time         CHAR(4),                  -- hhmm
    course                CHAR(1),                  -- 0:A 1:B 2:C 3:D 4:E
    direction             CHAR(1),                  -- 0:右 1:左

    total_10f             NUMERIC(4,1),
    lap_10f               NUMERIC(3,1),
    total_9f              NUMERIC(4,1),
    lap_9f                NUMERIC(3,1),
    total_8f              NUMERIC(4,1),
    lap_8f                NUMERIC(3,1),
    total_7f              NUMERIC(4,1),
    lap_7f                NUMERIC(3,1),
    total_6f              NUMERIC(4,1),
    lap_6f                NUMERIC(3,1),
    total_5f              NUMERIC(4,1),
    lap_5f                NUMERIC(3,1),
    total_4f              NUMERIC(4,1),
    lap_4f                NUMERIC(3,1),
    total_3f              NUMERIC(4,1),
    lap_3f                NUMERIC(3,1),
    total_2f              NUMERIC(4,1),
    lap_2f                NUMERIC(3,1),
    lap_1f                NUMERIC(3,1),

    payload_raw           TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (horse_id, training_date, training_center, training_time)
);

CREATE INDEX IF NOT EXISTS idx_training_wood_horse ON core.training_wood(horse_id, training_date);

-- =============================================================================
-- RAW層: CK (出走別着度数 - SNAP/SNPN)
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw.jv_ck_event (
  ingest_id          BIGSERIAL PRIMARY KEY,
  ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

  dataspec           TEXT NOT NULL CHECK (dataspec IN ('SNAP','SNPN')),
  record_type        CHAR(2) NOT NULL DEFAULT 'CK',

  data_kbn           SMALLINT NOT NULL,   -- データ区分
  data_create_ymd    DATE NOT NULL,       -- データ作成年月日

  -- CKキー
  kaisai_year        SMALLINT NOT NULL,   -- 開催年 yyyy
  kaisai_md          CHAR(4)  NOT NULL,   -- 開催月日 mmdd
  track_cd           CHAR(2)  NOT NULL,   -- 競馬場コード
  kaisai_kai         SMALLINT NOT NULL,   -- 開催回
  kaisai_nichi       SMALLINT NOT NULL,   -- 開催日目
  race_no            SMALLINT NOT NULL,   -- レース番号
  horse_id           TEXT NOT NULL,       -- 血統登録番号 (CKは馬番を持たない)

  horse_name         TEXT,
  payload            BYTEA NOT NULL,       -- 生レコード（Shift-JIS/JIS8混在のまま）
  payload_sha256     CHAR(64) NOT NULL,    -- 重複排除用ハッシュ

  UNIQUE (dataspec, data_create_ymd, kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id, payload_sha256)
);

CREATE INDEX IF NOT EXISTS ix_raw_ck_key
  ON raw.jv_ck_event (kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id);

CREATE INDEX IF NOT EXISTS ix_raw_ck_created
  ON raw.jv_ck_event (data_create_ymd);


-- =============================================================================
-- CORE層: CK (出走時点情報) - 解析用正規化テーブル
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.ck_runner_event (
  dataspec           TEXT NOT NULL CHECK (dataspec IN ('SNAP','SNPN')),
  data_create_ymd    DATE NOT NULL,

  kaisai_year        SMALLINT NOT NULL,
  kaisai_md          CHAR(4)  NOT NULL,
  track_cd           CHAR(2)  NOT NULL,
  kaisai_kai         SMALLINT NOT NULL,
  kaisai_nichi       SMALLINT NOT NULL,
  race_no            SMALLINT NOT NULL,
  horse_id           TEXT NOT NULL,

  horse_name         TEXT,

  -- 主要ブロック（JSONB配列で柔軟に保持）
  finish_counts      JSONB NOT NULL,  -- {"overall": [...], "central": [...], ...}
  style_counts       JSONB,           -- {"nige": N, "senko": N, ...}
  registered_races_n INTEGER,

  -- 人的リソースコード (Key codes only)
  jockey_cd          CHAR(5),
  trainer_cd         CHAR(5),
  owner_cd           CHAR(6),
  breeder_cd         CHAR(8),

  entity_prize       JSONB,           -- {"jockey": {...}, "trainer": {...}}

  PRIMARY KEY (dataspec, data_create_ymd, kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id)
);


-- =============================================================================
-- MART層: CK特徴量 (45列確定モデル投入用)
-- =============================================================================

CREATE TABLE IF NOT EXISTS mart.feat_ck_win (
  -- キー
  kaisai_year        SMALLINT NOT NULL,
  kaisai_md          CHAR(4)  NOT NULL,
  track_cd           CHAR(2)  NOT NULL,
  kaisai_kai         SMALLINT NOT NULL,
  kaisai_nichi       SMALLINT NOT NULL,
  race_no            SMALLINT NOT NULL,
  horse_id           TEXT NOT NULL,

  -- メタ
  dataspec           TEXT NOT NULL CHECK (dataspec IN ('SNAP','SNPN')),
  data_create_ymd    DATE NOT NULL,

  -- 1) 総合（全成績）
  h_total_starts     INTEGER NOT NULL,
  h_total_wins       INTEGER NOT NULL,
  h_total_top3       INTEGER NOT NULL,
  h_total_top5       INTEGER NOT NULL,
  h_total_out        INTEGER NOT NULL,

  -- 2) 中央合計
  h_central_starts   INTEGER NOT NULL,
  h_central_wins     INTEGER NOT NULL,
  h_central_top3     INTEGER NOT NULL,

  -- 3) 回り
  h_turf_right_starts    INTEGER NOT NULL,
  h_turf_left_starts     INTEGER NOT NULL,
  h_turf_straight_starts INTEGER NOT NULL,
  h_dirt_right_starts    INTEGER NOT NULL,
  h_dirt_left_starts     INTEGER NOT NULL,
  h_dirt_straight_starts INTEGER NOT NULL,

  -- 4) 馬場状態
  h_turf_good_starts  INTEGER NOT NULL,
  h_turf_soft_starts  INTEGER NOT NULL,
  h_turf_heavy_starts INTEGER NOT NULL,
  h_turf_bad_starts   INTEGER NOT NULL,
  h_dirt_good_starts  INTEGER NOT NULL,
  h_dirt_soft_starts  INTEGER NOT NULL,
  h_dirt_heavy_starts INTEGER NOT NULL,
  h_dirt_bad_starts   INTEGER NOT NULL,

  -- 5) 距離
  h_turf_16down_starts INTEGER NOT NULL,
  h_turf_22down_starts INTEGER NOT NULL,
  h_turf_22up_starts   INTEGER NOT NULL,
  h_dirt_16down_starts INTEGER NOT NULL,
  h_dirt_22down_starts INTEGER NOT NULL,
  h_dirt_22up_starts   INTEGER NOT NULL,

  -- 6) 脚質傾向
  h_style_nige_cnt   INTEGER NOT NULL,
  h_style_senko_cnt  INTEGER NOT NULL,
  h_style_sashi_cnt  INTEGER NOT NULL,
  h_style_oikomi_cnt INTEGER NOT NULL,

  -- 7) 登録レース数
  h_registered_races_n INTEGER NOT NULL,

  -- 8) 賞金（騎手/調教師/馬主/生産者）
  j_year_flat_prize_total BIGINT NOT NULL,
  j_year_ob_prize_total   BIGINT NOT NULL,
  j_cum_flat_prize_total  BIGINT NOT NULL,
  j_cum_ob_prize_total    BIGINT NOT NULL,

  t_year_flat_prize_total BIGINT NOT NULL,
  t_year_ob_prize_total   BIGINT NOT NULL,
  t_cum_flat_prize_total  BIGINT NOT NULL,
  t_cum_ob_prize_total    BIGINT NOT NULL,

  o_year_prize_total      BIGINT NOT NULL,
  o_cum_prize_total       BIGINT NOT NULL,
  b_year_prize_total      BIGINT NOT NULL,
  b_cum_prize_total       BIGINT NOT NULL,

  PRIMARY KEY (kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id, dataspec, data_create_ymd)
);

CREATE INDEX IF NOT EXISTS ix_feat_ck_race
  ON mart.feat_ck_win (kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no);


-- =============================================================================
-- CORE層: マイニングデータ (DM/TM - MING)
-- =============================================================================

-- タイム型マイニング (DM)
CREATE TABLE IF NOT EXISTS core.mining_dm (
    race_id               BIGINT NOT NULL,
    horse_no              SMALLINT NOT NULL,
    data_kbn              SMALLINT,
    dm_time_x10           INTEGER,                  -- 基準タイム×10
    dm_deviation          SMALLINT,                 -- 偏差
    dm_rank               SMALLINT,                 -- 順位
    payload_raw           TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_no)
);

-- 対戦型マイニング (TM)
CREATE TABLE IF NOT EXISTS core.mining_tm (
    race_id               BIGINT NOT NULL,
    horse_no              SMALLINT NOT NULL,
    data_kbn              SMALLINT,
    tm_score              INTEGER,                  -- 対戦スコア
    tm_rank               SMALLINT,
    payload_raw           TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (race_id, horse_no)
);

CREATE INDEX IF NOT EXISTS idx_mining_dm_race ON core.mining_dm(race_id);
CREATE INDEX IF NOT EXISTS idx_mining_tm_race ON core.mining_tm(race_id);

-- =============================================================================
-- CORE層: 開催スケジュール (YS - YSCH)
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.schedule (
    schedule_id           BIGSERIAL PRIMARY KEY,
    race_date             DATE NOT NULL,
    track_code            SMALLINT NOT NULL,
    kai                   SMALLINT,                 -- 開催回
    nichi                 SMALLINT,                 -- 開催日目
    race_count            SMALLINT,                 -- レース数
    grade_info            TEXT,                     -- グレード情報
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (race_date, track_code)
);

CREATE INDEX IF NOT EXISTS idx_schedule_date ON core.schedule(race_date);

-- =============================================================================
-- CORE層: コース情報 (CS - COMM)
-- =============================================================================

CREATE TABLE IF NOT EXISTS core.course (
    course_id             BIGSERIAL PRIMARY KEY,
    track_code            SMALLINT NOT NULL,
    surface               SMALLINT NOT NULL,        -- 1=芝, 2=ダート
    distance_m            SMALLINT NOT NULL,
    course_name           TEXT,
    turn_dir              SMALLINT,                 -- 1=右, 2=左, 3=直線
    course_inout          SMALLINT,                 -- A/B/C コース
    width_m               NUMERIC(4,1),             -- コース幅
    straight_m            SMALLINT,                 -- 直線距離
    slope_info            TEXT,                     -- 勾配情報
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (track_code, surface, distance_m, course_inout)
);

CREATE INDEX IF NOT EXISTS idx_course_track ON core.course(track_code, surface, distance_m);

-- =============================================================================
-- 完了メッセージ
-- =============================================================================
-- 実行完了後の確認用
DO $$
BEGIN
    RAISE NOTICE 'スキーマ初期化完了: raw, core (mart は後で追加)';
END $$;
