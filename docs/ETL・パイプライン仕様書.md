# ETL・パイプライン仕様書（JV-Link → JSONL → PostgreSQL）v0.1

* 対象プロジェクト: `keiba`
* 対象範囲: 取得済みJSONLを PostgreSQL（raw/core/mart）へ投入するETL
* 実装（一次ソース）:
  * ローダ: `scripts/load_to_db.py`
  * パーサ: `app/infrastructure/parsers.py`
  * DBスキーマ: `migrations/0001_init_db.sql`（+ `migrations/*.sql`）
* 最終更新: 2026-02-15

---

## 1. 目的 / 非目的

### 目的

* 取得済みデータ（`data/*.jsonl`）を PostgreSQL に **再現可能**に投入する
* 1回限りのバックフィル（2016〜）と、継続的な増分更新（週次/開催後）を同じ枠組みで扱う
* 失敗時に「どこからリトライするか」「重複/整合性をどう担保するか」を明確にする

### 非目的（本書では扱わない）

* JV-Linkからの取得仕様（参照: `docs/データ取得仕様書.md`）
* レコード→カラムのデータ辞書（参照: `docs/データ辞書・正規化仕様書.md`）
* DB設計（参照: `docs/DB設計仕様書.md` / 一次ソース `migrations/0001_init_db.sql`）
* 特徴量生成・学習・推論（参照: `docs/特徴量仕様.md`）

---

## 2. パイプライン全体像

```text
（取得）  JV-Link → JSONL（data/*.jsonl）
（ETL）   JSONL → raw/core（+一部mart）
（利用）  API/分析/特徴量生成（別仕様）
```

本書の対象は ETL（JSONL→DB）部分。

---

## 3. コンポーネント

### 3.1 入力（JSONL）

* 入力ディレクトリ: `data/`
* 入力ファイル命名例:
  * `DIFF_20260207_091006.jsonl`
  * `RACE_20160101-20171231_20260207_095054.jsonl`
  * `0B41_2025122806010101_20260208_151112.jsonl`
* 1行1JSON。最低限以下のキーを含むことを前提とする（実装準拠）。
  * `rec_id`, `filename`, `payload`, `size`

### 3.2 パース（固定長 → dataclass）

* `app/infrastructure/parsers.py` が固定長レコード（payload）をパースし、`RaceRecord` 等の dataclass に変換する。

### 3.3 ロード（DB投入）

* `scripts/load_to_db.py` が JSONL を読み込み、`raw`/`core` に投入する。

### 3.4 DB

* PostgreSQL
* 接続設定は環境変数（`app/infrastructure/database.py`）で上書き可能。

---

## 4. 実行環境 / 設定

### 4.1 DB接続（環境変数）

`app/infrastructure/database.py` の既定値に従う。

| env | default | 説明 |
|---|---|---|
| `DATABASE_URL` | `-` | PostgreSQL接続URL（指定時は最優先） |
| `DB_HOST` | `127.0.0.1` | PostgreSQLホスト |
| `DB_PORT` | `5432` | ポート |
| `DB_NAME` | `keiba` | DB名 |
| `DB_USER` | `jv_ingest` | ユーザー |
| `DB_PASSWORD` | `keiba_pass` | パスワード |

### 4.2 DBスキーマ適用

* DDL一次ソース: `migrations/0001_init_db.sql`（新規DBは `scripts/migrate.py` で `migrations/*.sql` を適用）
* 本ETLは「テーブルが存在する」前提で動く。

### 4.3 実行コマンド（ETL）

```powershell
# 64bit Python（uv）で実行
uv run python scripts/load_to_db.py --input-dir data/

# ワイルドカード指定
uv run python scripts/load_to_db.py --input "data/RACE_*.jsonl"
uv run python scripts/load_to_db.py --input "data/DIFF_*.jsonl"
```

---

## 5. ロード仕様（`scripts/load_to_db.py`）

### 5.1 ファイル選択・処理順

* `--input-dir` 指定時:
  * `data/*.jsonl` を **ファイル名昇順**で処理する
  * 期待する効果: `DIFF_*` が `RACE_*` より先に処理され、マスタが先に入りやすい
* `--input` 指定時:
  * `*` を含む場合は glob 展開し、**ファイル名昇順**で処理する

### 5.2 dataspec 推定

* `dataspec = file_path.stem.split("_")[0]`
  * 例: `RACE_20160101-20171231_...` → `RACE`
  * 例: `0B41_2025...` → `0B41`

### 5.3 2パス処理（同一ファイルを2回読む）

`process_file()` は同一JSONLを2回走査する。

#### Pass 1（RA優先 + rawバッチ挿入）

* 目的:
  * `core.race` を先に満たし、後続のFK参照エラーを減らす
  * `raw.jv_raw` へバッチ投入してスキャンを確保する
* 動作:
  * 全レコードを読み、バッチ（1000件）で `raw.jv_raw` に `executemany` INSERT
  * `rec_id == "RA"` のみ `core.race` に UPSERT
  * バッチごとに commit

#### Pass 2（RA以外の正規化）

* 目的: RA以外（SE/HR/O1/UM/…）を `core`（および一部 `raw`/`mart`）へ投入
* 動作:
  * 同じJSONLを再走査し、`rec_id` で分岐して投入
  * 1000レコードごとに commit
  * 例外は rollback して継続（一定件数まで warning を出す）

#### Pass 2.5（O1遅延処理: `dataspec != "0B41"`）

* 背景: `RACE` の `O1` は入力ファイルによって `SE` より先に出現することがある。`O1` は `core.runner (race_id+horse_no)` で `horse_id` を解決するため、順序依存で 0 行投入になり得る。
* 動作:
  * Pass 2 では `O1`（最終オッズ）は payload を一旦バッファする
  * `SE` の投入が終わった後に、バッファした `O1` をまとめて `core.odds_final` へUPSERTする
  * `0B41`（時系列O1）は `core.o1_*` への投入で runner 依存が無いため、従来どおり即時処理する

### 5.4 マスタキャッシュ（FK回避）

* 起動時に `core.jockey` と `core.trainer` の ID を `SELECT` して set 化する。
* `SE` の投入では、この集合に存在しないIDを `NULL` に落としてFK違反を回避する。

---

## 6. 正規化先（テーブル別の投入方針：要点）

詳細なレコード→カラム対応は `docs/データ辞書・正規化仕様書.md` を正とする。

### 6.1 raw.jv_raw

* Pass 1 でバッチ投入（`INSERT ... ON CONFLICT DO NOTHING`）

### 6.2 core.race（RA）

* Pass 1 で先行UPSERT

### 6.3 core.runner / core.result（SE）

* `core.horse` / `core.race` に欠損がある場合は stub 登録で落ちないようにする（現行ローダの方針）
* `finish_pos` が存在する場合のみ `core.result` に投入する

### 6.4 core.payout（HR）

* 縦持ち（券種ごと）でUPSERT

### 6.5 core.odds_final（O1）

* horse_id が O1 に無いため `core.runner` を `race_id+horse_no` で引いて解決する

### 6.6 core.wh_header / core.wh_detail（WH）

* 速報馬体重をスナップ単位（header）＋馬番別（detail）で保持

### 6.7 core.training_*（HC/WC）

* 走査時にそのまま投入（ユニーク制約で重複を抑制）

### 6.8 core.event_change（WE/AV/JC/TC/CC）

* 現状は `payload_parsed`（監査キー＋最小構造化）を JSONB として保存し、`raw` も同梱する（詳細は `docs/データ辞書・正規化仕様書.md`）
* T-5スナップ（`mart.t5_runner_snapshot`）への反映は後段バッチで実施する（現状: `TC/AV/JC` は `scripts/build_t5_snapshot.py` で反映、`WE/CC` はTODO）

### 6.9 core.mining_* / core.rt_mining_*（DM/TM）

* Backfillable（`dataspec="MING"`）:
  * `DM/TM` は `core.mining_dm` / `core.mining_tm` に UPSERT（主キー: `(race_id, horse_no)` の上書き）
* Realtime-only（`dataspec in {"0B13","0B17"}`）:
  * 速報 `DM/TM` は **履歴保持**として `core.rt_mining_dm` / `core.rt_mining_tm` に投入（主キー: `(race_id, data_kbn, data_create_ymd, data_create_hm, horse_no)`）
  * `data_kbn=0`（削除）の場合は、payloadヘッダーから `(race_id, data_create_ymd, data_create_hm)` を抽出し、該当 create_time の行を削除する
  * DDL: `migrations/20260215_add_rt_mining_dm_tm.sql`

---

## 7. 冪等性（再実行）と重複の考え方

### 7.1 core系（推奨：冪等）

* `core.race` / `core.runner` / `core.result` / `core.payout` / `core.odds_final` 等は主キーに対して UPSERT を行う設計で、同一入力の再投入は基本的に冪等にできる。

### 7.2 raw.jv_raw（注意）

* `raw.jv_raw` は `(dataspec, rec_id, payload_hash)` のUNIQUE制約（`uq_jv_raw_dedup`）で重複排除する設計。
* `scripts/load_to_db.py` は `payload_hash = sha256(payload)` を計算して INSERT するため、同一payloadの再投入は冪等になる。
* 既存DBが古いDDLで作成されている場合は、制約の有無を確認してから投入すること。

### 7.3 raw.jv_ck_event（CK）

* `payload_sha256` を含む UNIQUE で重複排除できる設計（E2E投入確認済）

---

## 8. エラー処理 / リトライ

### 8.1 単位

* 原則: **ファイル単位**でリトライする（`--input` でそのファイルだけ再投入）
* 例: `RACE_20180101-20191231_....jsonl` が落ちたら、そのファイルだけやり直す

### 8.2 例外時の動作

* Pass 1:
  * 例外発生時は rollback し、バッチをクリアして継続する
* Pass 2:
  * 例外発生時は rollback し、次レコードへ進む

### 8.3 失敗の再現に使う情報

* 入力ファイル名
* `rec_id`
* 例外メッセージ（スタックトレースは現在抑制されることがある）

---

## 9. 品質チェック（最小）

### 9.1 パーサのスモーク

```powershell
uv run pytest -q
uv run python scripts/verify_parsers.py
```

### 9.2 DBの簡易サニティ（例）

* 年別レース数が極端に欠けていないか
* `core.runner` と `core.result` の整合（結果があるのに出走がない等）がないか

> 実際のSQLチェック項目は運用に合わせて追加する（本書では例示に留める）。

---

## 10. 既知の制約 / TODO（重要）

以下は「ETL仕様として確定していない/未整備」なため、実装修正が必要。

* `core.runner` のスキーマとローダのINSERT列が不一致になり得る（スキーマ/ローダの統一が必要）
* CK（SNPN/CK）の投入:
  * パーサ/ローダ/DDLは揃っており、E2E投入確認済（参照: `docs/DB構築ToDo.md`）
* 時系列オッズ（0B41）の投入:
  * `scripts/load_to_db.py` は `dataspec=="0B41"` の `O1` を `core.o1_*` に投入する実装を持つ
  * E2E確認済（`core.odds_final` への誤ルーティング無し）
* DM/TM（MING/0B13/0B17）:
  * `horse_no`/`dm_time_x10`/`tm_score` の抽出は実装済み。運用（0B13/0B17）は `core.rt_mining_*` に履歴保持する（as-of採用用）

---

## 11. 参照

* 取得仕様: `docs/データ取得仕様書.md`
* データ辞書・正規化: `docs/データ辞書・正規化仕様書.md`
* DB設計: `docs/DB設計仕様書.md`
* ワークフロー（運用手順）: `.agent/workflows/batch-download.md`
