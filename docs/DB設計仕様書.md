# DB設計仕様書（PostgreSQL / Raw・Core・Mart）v0.1

* 対象プロジェクト: `keiba`
* 対象DB: PostgreSQL
* スキーマ定義（一次ソース）: `init_db.sql`
* 最終更新: 2026-02-12

---

## 1. 目的 / 非目的

### 目的

* 本プロジェクトで利用するDBの **スキーマ（raw/core/mart）** を定義し、設計意図・キー設計・制約・索引を明文化する
* 実装（ETL/アプリ）から参照する際に、テーブルの責務と境界がブレないようにする

### 非目的

* レコード（JV-Data）→カラムの対応表（参照: `docs/データ辞書・正規化仕様書.md`）
* 取得仕様（JVOpen/JVRTOpenの引数・実行例）（参照: `docs/データ取得仕様書.md`）
* 特徴量/検証/運用（参照: `docs/特徴量仕様.md`, `docs/検証設計仕様書.md`, `docs/運用仕様書.md`）

---

## 2. 全体アーキテクチャ

### 2.1 スキーマ（3層）

`init_db.sql` で以下スキーマを作成する。

* `raw`: 取得した生データ（監査・再処理の保険）
* `core`: 正規化された事実テーブル＋マスタ
* `mart`: 学習・分析用（特徴量マート）

### 2.2 設計方針（要点）

* **再現性**: raw に生データ（またはその代替）を残す
* **正規化**: core は race_id / horse_id を中心に、参照整合性と分析しやすさを優先
* **拡張性**: oddsの時系列、調教、当日変更、CK（SNPN）等を後から追加できる設計
* **運用耐性**: 外部キーが欠ける/仕様変更がある場合に備えて、最小限のstub登録やNULL許容を許す（ただし方針は別途実装と揃える）

---

## 3. 共通キー・命名規則

### 3.1 race_id

`init_db.sql` のコメントに従い、`race_id` は以下で一意生成する。

```text
race_id = YYYYMMDD * 10000 + track_code * 100 + race_no
```

* `track_code`: 競馬場コード（例: 05=東京）
* `race_no`: 1〜12

### 3.2 horse_id

* `horse_id` は JRA-VAN の血統登録番号（旧8桁/新10桁が混在し得る）を想定し、**TEXT** を採用する。

### 3.3 タイムゾーン

* 監査/イベント等の時刻は `TIMESTAMPTZ` を使用する（DB側のタイムゾーン設定に依存しない）。

---

## 4. ER（主要な参照関係）

* `core.race (race_id)` ← `core.runner (race_id)`（ON DELETE CASCADE）
* `core.horse (horse_id)` ← `core.runner (horse_id)`（ON DELETE CASCADE）
* `core.runner (race_id, horse_id)` ← `core.result (race_id, horse_id)`（ON DELETE CASCADE）
* `core.runner (race_id, horse_id)` ← `core.odds_final (race_id, horse_id)`（ON DELETE CASCADE）
* `core.o1_header (race_id, data_kbn, announce_mmddhhmi)` ← `core.o1_win (...)`（ON DELETE CASCADE）
* `core.wh_header (race_id, data_kbn, announce_mmddhhmi)` ← `core.wh_detail (...)`（ON DELETE CASCADE）
* `core.horse (horse_id)` ← `core.training_slop/wood (horse_id)`
* `raw.jv_raw (id)` ← `core.diff_event (source_raw_id)`（参照のみ）
* `raw.jv_ck_event`（CK raw）→ `core.ck_runner_event`（キーは同一だがFKは定義していない）

---

## 5. スキーマ定義（テーブル別）

以降のカラム定義は `init_db.sql` に準拠する（本書は「設計書」であり、DDLの一次ソースは `init_db.sql`）。

---

## 5A. RAW層（raw）

## 5A.1 raw.jv_raw

JV-Link から取得した生レコード（UTF-8文字列化後）を保持する汎用テーブル。

* 主キー: `id (BIGSERIAL)`
* 主要索引:
  * `idx_jv_raw_ingested_at (ingested_at)`
  * `idx_jv_raw_dataspec_recid (dataspec, rec_id)`

| カラム | 型 | NULL | 説明 |
|---|---|---:|---|
| `id` | BIGSERIAL | NO | PK |
| `ingested_at` | TIMESTAMPTZ | NO | 取り込み時刻（default now） |
| `dataspec` | TEXT | NO | JVOpenで指定した dataspec（例: RACE, DIFF） |
| `rec_id` | CHAR(2) | NO | レコード種別（payload先頭2文字） |
| `filename` | TEXT | YES | JVReadのファイル名 |
| `payload` | TEXT | NO | 固定長文字列（Shift_JIS→UTF-8変換後） |
| `payload_hash` | BYTEA | YES | 重複排除用（任意、現状は未運用） |

## 5A.2 raw.jv_ck_event（SNAP/SNPN）

CK（出走別着度数）の生レコード（BYTEA）を、ハッシュ付きで保持する専用raw。

* 制約:
  * `dataspec IN ('SNAP','SNPN')`
  * UNIQUE: `(dataspec, data_create_ymd, kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id, payload_sha256)`
* 主要索引:
  * `ix_raw_ck_key (kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id)`
  * `ix_raw_ck_created (data_create_ymd)`

---

## 5B. CORE層（core）: マスタ

## 5B.1 core.horse（UM）

* 主キー: `horse_id (TEXT)`

| カラム | 型 | NULL | 説明 |
|---|---|---:|---|
| `horse_id` | TEXT | NO | 血統登録番号（8〜10桁想定） |
| `horse_name` | TEXT | YES | 馬名 |
| `sex` | SMALLINT | YES | 1=牡,2=牝,3=騸 |
| `birth_date` | DATE | YES | 生年月日 |
| `coat_color` | SMALLINT | YES | 毛色（任意） |
| `created_at` | TIMESTAMPTZ | NO | default now |
| `updated_at` | TIMESTAMPTZ | NO | default now |

## 5B.2 core.jockey（KS）

* 主キー: `jockey_id (BIGINT)`

## 5B.3 core.trainer（CH）

* 主キー: `trainer_id (BIGINT)`

---

## 5C. CORE層（core）: 事実テーブル（レース中心）

## 5C.1 core.race（RA）

* 主キー: `race_id (BIGINT)`
* ユニーク: `(race_date, track_code, race_no)`
* 主要索引: `idx_race_date_track (race_date, track_code)`

| カラム | 型 | NULL | 説明 |
|---|---|---:|---|
| `race_id` | BIGINT | NO | 3.1 の式で生成 |
| `race_date` | DATE | NO | 開催年月日 |
| `track_code` | SMALLINT | NO | 競馬場コード |
| `race_no` | SMALLINT | NO | レース番号 |
| `surface` | SMALLINT | NO | 1=芝,2=ダート,3=障害 |
| `distance_m` | SMALLINT | NO | 距離(m) |
| `going` | SMALLINT | YES | 馬場状態 |
| `weather` | SMALLINT | YES | 天候 |
| `class_code` | SMALLINT | YES | クラス（拡張） |
| `turn_dir` | SMALLINT | YES | 右/左/直線（拡張） |
| `course_inout` | SMALLINT | YES | コース区分（拡張） |
| `field_size` | SMALLINT | YES | 頭数 |
| `start_time` | TIME | YES | 発走時刻 |
| `created_at` | TIMESTAMPTZ | NO | default now |
| `updated_at` | TIMESTAMPTZ | NO | default now |

## 5C.2 core.runner（SE: 出走表）

* 主キー: `(race_id, horse_id)`
* ユニーク: `(race_id, horse_no)`（馬番の一意）
* 索引:
  * `idx_runner_horse (horse_id)`
  * `idx_runner_race (race_id)`

| カラム | 型 | NULL | 説明 |
|---|---|---:|---|
| `race_id` | BIGINT | NO | FK→core.race（CASCADE） |
| `horse_id` | TEXT | NO | FK→core.horse（CASCADE） |
| `horse_no` | SMALLINT | NO | 馬番 |
| `gate` | SMALLINT | YES | 枠番 |
| `jockey_id` | BIGINT | YES | FK→core.jockey |
| `trainer_id` | BIGINT | YES | FK→core.trainer |
| `carried_weight` | NUMERIC(4,1) | YES | 斤量 |
| `body_weight` | SMALLINT | YES | 馬体重 |
| `body_weight_diff` | SMALLINT | YES | 増減 |
| `scratch_flag` | BOOLEAN | NO | default false |
| `entry_status` | SMALLINT | YES | 出走確定/除外コード（拡張） |
| `created_at` | TIMESTAMPTZ | NO | default now |
| `updated_at` | TIMESTAMPTZ | NO | default now |

## 5C.3 core.result（SE: 結果）

* 主キー: `(race_id, horse_id)`
* FK: `(race_id, horse_id) → core.runner`（CASCADE）
* 索引: `idx_result_finish_pos (race_id, finish_pos)`

| カラム | 型 | NULL | 説明 |
|---|---|---:|---|
| `finish_pos` | SMALLINT | YES | 着順 |
| `time_sec` | NUMERIC(6,2) | YES | 走破タイム（秒） |
| `margin` | TEXT | YES | 着差（文字列保持） |
| `final3f_sec` | NUMERIC(5,2) | YES | 上がり3F |
| `corner1_pos` | SMALLINT | YES | 1角 |
| `corner2_pos` | SMALLINT | YES | 2角 |
| `corner3_pos` | SMALLINT | YES | 3角 |
| `corner4_pos` | SMALLINT | YES | 4角 |
| `created_at` | TIMESTAMPTZ | NO | default now |
| `updated_at` | TIMESTAMPTZ | NO | default now |

## 5C.4 core.payout（HR）

* 主キー: `(race_id, bet_type, selection)`
* 索引: `idx_payout_race_bet (race_id, bet_type)`

## 5C.5 core.odds_final（O1: 最終/確定）

* 主キー: `(race_id, horse_id)`
* FK: `(race_id, horse_id) → core.runner`（CASCADE）
* 索引: `idx_odds_final_race (race_id)`

## 5C.6 core.odds_snapshot（拡張用）

* 主キー: `(race_id, horse_id, collected_at)`
* FK: `(race_id, horse_id) → core.runner`（CASCADE）
* 索引: `idx_odds_snapshot_race_time (race_id, collected_at)`

## 5C.7 core.race_lap（拡張用）

* 主キー: `(race_id, lap_index)`

## 5C.8 core.diff_event（DIFF: 変更・訂正ログ）

* 主キー: `diff_id (BIGSERIAL)`
* FK:
  * `race_id → core.race`
  * `source_raw_id → raw.jv_raw(id)`
* 索引: `idx_diff_race_time (race_id, event_time)`

---

## 5D. CORE層（core）: 速報・時系列

## 5D.1 core.o1_header / core.o1_win（O1: 時系列オッズ）

* header 主キー: `(race_id, data_kbn, announce_mmddhhmi)`
* win 主キー: `(race_id, data_kbn, announce_mmddhhmi, horse_no)`
* 索引:
  * `idx_o1_header_race_time (race_id, data_kbn, announce_mmddhhmi)`
  * `idx_o1_win_race_horse (race_id, horse_no)`

## 5D.2 core.wh_header / core.wh_detail（WH: 馬体重速報）

* header 主キー: `(race_id, data_kbn, announce_mmddhhmi)`
* detail 主キー: `(race_id, data_kbn, announce_mmddhhmi, horse_no)`

## 5D.3 core.event_change（WE/AV/JC/TC/CC）

* 主キー: `id (BIGSERIAL)`
* 索引: `idx_event_change_race_type (race_id, record_type, announce_mmddhhmi)`
* 備考: `payload_parsed (JSONB)` は種類ごとに構造が変わる前提の箱

---

## 5E. CORE層（core）: 調教

## 5E.1 core.training_slop（HC）

* ユニーク: `(horse_id, training_date, training_center, total_4f)`
* 索引: `idx_training_slop_horse (horse_id, training_date)`

## 5E.2 core.training_wood（WC）

* ユニーク: `(horse_id, training_date, training_center, training_time)`
* 索引: `idx_training_wood_horse (horse_id, training_date)`

---

## 5F. CORE層（core）: CK（SNPN）・マイニング・その他

## 5F.1 core.ck_runner_event（CK正規化）

* 主キー: `(dataspec, data_create_ymd, kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id)`
* 主要カラム:
  * `finish_counts (JSONB)` / `style_counts (JSONB)` / `entity_prize (JSONB)`（柔軟保持）

## 5F.2 core.mining_dm / core.mining_tm（DM/TM）

* 主キー: `(race_id, horse_no)`
* 索引:
  * `idx_mining_dm_race (race_id)`
  * `idx_mining_tm_race (race_id)`

## 5F.3 core.schedule（YSCH）

* ユニーク: `(race_date, track_code)`
* 索引: `idx_schedule_date (race_date)`

## 5F.4 core.course（COMM: CS）

* ユニーク: `(track_code, surface, distance_m, course_inout)`
* 索引: `idx_course_track (track_code, surface, distance_m)`

---

## 5G. MART層（mart）

## 5G.1 mart.feat_ck_win（CK特徴量）

* 主キー: `(kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no, horse_id, dataspec, data_create_ymd)`
* 索引: `ix_feat_ck_race (kaisai_year, kaisai_md, track_cd, kaisai_kai, kaisai_nichi, race_no)`

> 本テーブルは「モデル投入用に列を確定した特徴量マート」の位置付けで、生成ロジックは別途定義・実装する（本書はDB設計のみ）。

---

## 6. 実装整合性メモ（設計と実装を揃えるための論点）

* `core.runner` のカラムは `init_db.sql` を一次ソースとする。
  * `scripts/load_to_db.py` が追加カラム（`data_kubun` 等）を参照している場合、スキーマ拡張かローダ修正のいずれかで整合させる。
* `raw.jv_raw` で重複排除を行う場合は、`payload_hash` の運用（計算・ユニーク制約）を設計に追加する必要がある。

---

## 7. 参照

* DBスキーマ（DDL一次ソース）: `init_db.sql`
* データ辞書（レコード→カラム）: `docs/データ辞書・正規化仕様書.md`
* データ設計メモ: `docs/data_design.md`
* DBセットアップ: `setup_postgres.sh`

