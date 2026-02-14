# DB構築・ロード実装状況とToDo

## 1. 現状概要

主要なテーブル定義（Schema）とデータ投入ロジック（Loader）は概ね実装が完了しています。
2026-02-13 の仕様整理（Backfillable / Realtime-only）に合わせて、本書もその前提で整理します。

### 重要：Backfillable / Realtime-only（2016〜学習との整合）

* **Backfillable（2016〜再現可能）**：学習DB（2016〜現在）を再現できるデータ
  * JVOpen（蓄積系）が中心（例：RACE（RA/SE/HR/O1）、DIFF（UM/KS/CH）、MING（DM/TM）、SNPN（CK）、SLOP/WOOD（HC/WC））
  * **0B41（時系列オッズ）は JVRTOpen の racekey 指定でバックフィル可能**（2003年〜、参照: `docs/jvlink_handover.md`）
* **Realtime-only（運用開始日以降のみ）**：運用当日の監視/スナップ用途で取得するが、長期バックフィルは前提にしないデータ
  * 例：0B11（WH）、0B16（WE/AV/JC/TC/CC）、0B13/0B17（速報DM/TM）

> DB構築のブロッカー判定は「Backfillable で 2016〜学習DBが作れるか」を主軸にし、Realtime-only は運用開始以降の蓄積（JSONL→DB）が整った段階で順次埋める。

### 実装状況サマリ

| データ種別 | RecID | 区分 | DB Schema | Parser | Loader | 状態 / 備考 |
|---|---|---|---|---|---|---|
| レース詳細 | RA | Backfillable | ✅ | ✅ | ✅ | 実装完了 |
| 出走馬・結果 | SE | Backfillable | ✅ | ✅ | ✅ | 実装完了 |
| 払戻金 | HR | Backfillable | ✅ | ✅ | ✅ | 実装完了 |
| 確定オッズ | O1 | Backfillable | ✅ | ✅ | ✅ | 実装完了 |
| 時系列オッズ | O1(0B41) | Backfillable | ✅ | ✅ | ✅ | ローダ実装済み。**2003年〜取得可能**（backfill可、参照: `docs/jvlink_handover.md`） |
| マスタ（馬/騎手/調教師） | UM/KS/CH | Backfillable | ✅ | ✅ | ✅ | 実装完了 |
| 競走馬除外 | JG | Backfillable | ✅ | ✅ | ✅ | 実装完了（馬名更新のみ） |
| 馬体重速報 | WH | Realtime-only | ✅ | ✅ | ✅ | 実装完了（運用開始以降に蓄積） |
| 調教（坂路/ウッド） | HC/WC | Backfillable | ✅ | ✅ | ✅ | 実装完了 |
| マイニング | DM/TM | 両方 | ✅ | ✅ | ✅ | **学習= MING（Backfillable）**はOK。**運用= 0B13/0B17（Realtime-only）**は採用キー/履歴の扱いを別途設計 |
| 出走別着度数（CK） | CK | Backfillable | ✅ | ✅ | ⚠️ | 実装済・**E2E検証待ち**（`raw.jv_ck_event`/`core.ck_runner_event`/`mart.feat_ck_win`） |
| 当日変更情報 | WE/AV等 | Realtime-only | ✅ | ✅ | ✅ | **監査キー+最小構造化（payload_parsed）まで投入**。mart反映（取消/変更フラグ）は別ToDo |
| 速報オッズ | O1(0B15) | Realtime-only | ✅ | ✅ | ⚠️ | **0B41以外は上書き保存**（時系列保持なし）。現状はMVP外 |
| 速報票数 | H1/H6(0B12) | Realtime-only | ❌ | ❌ | ❌ | **完全未実装**（必要なら追加） |

## 2. 実装ギャップとToDoリスト

DB構築を完了し、信頼できる状態にするためのタスクリストです。

### [DB] DDL / Migration（既存DBをアップデートする場合）

- [ ] **raw.jv_raw の冪等化（payload_hash）を適用** <!-- id: raw-dedup -->
    - 既に `init_db.sql` では `payload_hash` + UNIQUE が前提だが、既存DBを使う場合は移行が必要。
    - 参照: `docs/migrations/20260213_raw_dedup.sql`

### [Code] コード修正・整理

- [x] **JVRTOpen（0Bxx）の取得オーケストレーション整備** <!-- id: rt-tool-update -->
    - `scripts/extract_rt_jvlink.py` は **racekey × dataspec** の単発取得。
    - ✅ `scripts/backfill_rt.py`: 0B41 バックフィル（racekey ループ、途中再開・重複排除対応）
    - ✅ `scripts/ops_rt.py`: 当日運用一括取得（0B41/0B11/0B16/0B13/0B17）
    - ✅ `scripts/rt_common.py`: 共通ユーティリティ（WSL→Win32 subprocess 連携）

- [ ] **CK（SNPN）ロードのE2E検証** <!-- id: ck-e2e -->
    - `SNPN_*.jsonl` を投入して `raw.jv_ck_event` / `core.ck_runner_event` / `mart.feat_ck_win` が増えることを確認する。
    - 同一ファイルの再投入が冪等（UPSERT/UNIQUEで破綻しない）であることを確認する。

### [Doc] ドキュメント更新

- [x] **仕様書のステータス更新（反映済）** <!-- id: doc-update -->
    - `0B41` はバックフィル可能（2003年〜）として、学習・検証（Model M/O）に使える前提に整理した。
    - `0B11/0B16/0B13/0B17` は現時点では Realtime-only として扱う（必要なら別途方針を追加）。
    - `event_change` は「監査キー+最小構造化（payload_parsed）」まで投入する前提に更新した（mart反映はTODO）。

### [Test] 検証

- [ ] **全データのロードテスト**
    - 全データスペックを含むJSONLファイルを用いて、`load_to_db.py` がエラーなく完走することを確認する。
    - **注記**:
      - Backfillable（JVOpen + 0B41バックフィル）で **2016〜現在の学習DBが作れる**ことをまず検証する。
      - Realtime-only（0B11/0B16/0B13/0B17）は運用開始以降のデータが揃い次第、別途E2Eで検証する。

### [Future] 将来的な改善（Optional）

- [ ] **当日変更（0B16）の mart 反映（T-5スナップ）**
    - `core.event_change` から as-of（<=T-5）で
      * 取消（AV）→ 推論対象外（または `scratch_flag`）
      * 騎手変更（JC）→ jockey置換＋フラグ
      * 発走時刻変更（TC）→ post_time更新＋asof再計算
      * コース変更（CC）→ 条件特徴量更新＋フラグ
      を反映する。
- [ ] **event_change の追加構造化（必要なら）**
    - 現状は「監査キー+最小構造化」まで。分析/可視化用途が必要になったら詳細項目のパースを増やす。

## 3. 次のアクション

1. Backfillable（JVOpen + 0B41バックフィル）で 2016〜現在の学習DBを構築できることを先に検証する。
2. CK（SNPN）のE2E検証を通し、`mart.feat_ck_win` まで増えることを確認する。
3. Realtime-only（0B11/0B16/0B13/0B17）は「運用開始以降に蓄積する」前提で、当日取得のオーケストレーションを整備する。
