# JV-Link データ取り込みパイプライン - 引継ぎ資料

## 現在のステータス (2026-02-09 更新)

| 項目 | 状態 | 備考 |
|------|------|------|
| JV-Link接続 | ✅ 完了 | 32bit Python + COM |
| データ取得 (JVOpen) | ✅ 完了 | `--to-date` オプション追加済 |
| データ取得 (JVRTOpen) | ✅ 完了 | 時系列オッズ対応 |
| パーサー | ✅ 完了 | 仕様書4.9.0.1準拠（オフセット修正済） |
| DB保存 | ✅ 完了 | FK違反対策実装済 |
| 時系列オッズ | ✅ 確認済 | 2003年〜現在 取得可能 |

---

## JV-Link / JV-Data 注意点

### 1. バイトオフセット (重要)
JV-Data仕様書は **1-indexed**、Pythonコードは **0-indexed**。

```python
# 仕様書「位置12」 → コード「11」
RA_YEAR_START = 11  # 仕様書: 12
```

### 2. トラックコード (Surface判定)
コード表2009を正しく読むこと。
| 範囲 | Surface |
|------|---------|
| 10-22 | 芝 |
| **23-29** | ダート (27,28はサンド) |
| 51-59 | 障害 |

**誤った範囲 `30-49` をダートと判定しないこと！**

### 3. RAレコードのオフセット
以下は仕様書4.9.0.1準拠の正しい値（0-indexed）:

| 項目 | 位置(0-idx) | 仕様書(1-idx) |
|------|------------|--------------|
| 開催年 | 11 | 12 |
| 開催月日 | 15 | 16 |
| 競馬場コード | 19 | 20 |
| レース番号 | 25 | 26 |
| 距離 | **697** | 698 |
| トラックコード | **705** | 706 |
| 天候 | **887** | 888 |

### 4. FK制約違反対策
地方競馬(data_kubun='A')や海外(data_kubun='B2')のデータは、
マスタ(horse, jockey, trainer)が欠落している場合がある。

**実装済み対策:**
- `jockey_id`, `trainer_id` をNULL許容
- `core.horse`, `core.race` への自動Stub登録
- 生値カラム (`xxx_code_raw`, `xxx_name_abbr`) で元データ保持

### 5. JVRTOpen (リアルタイム系API) - 重要

#### 戻り値の解釈
| 戻り値 | 意味 |
|--------|------|
| `0` | **成功** (データ有無はJVReadで確認) |
| 負の値 | エラー (-1: dataspec不正, -503: ファイルなし等) |

> [!CAUTION]
> `JVRTOpen` の戻り値 `0` は「成功」であり「データなし」ではない！
> データの有無は `JVRead` を呼んで確認する必要がある。

#### 時系列オッズ (0B41/0B42)

| 項目 | 値 |
|------|-----|
| dataspec | `0B41` (単複枠), `0B42` (馬連) |
| 取得可能期間 | **2003年 〜 現在** |
| 時系列粒度 | 約5分間隔 |
| レコード種別 | O1, O2 |

> [!IMPORTANT]
> 仕様書には「提供期間1年間」と記載されているが、実際には **2003年以降の全データが取得可能**。

#### レースキー形式
```
YYYYMMDDJJKKHHRR  または  YYYYMMDDJJRR
```
- YYYY: 年, MM: 月, DD: 日
- JJ: 競馬場コード (01=札幌〜10=小倉)
- KK: 開催回 (第N回)
- HH: 開催日目
- RR: レース番号

#### 時系列の「発表月日時分」
O1/O2レコード内の「発表月日時分 (MMDDhhmm)」フィールドで各時点を識別。
同一レースキーで取得したレコードを発表月日時分でソートすると時系列になる。

---

## 主な実装ファイル

| ファイル | 役割 |
|----------|------|
| `scripts/extract_jvlink.py` | JV-Linkからデータ取得 (32bit専用) |
| `scripts/extract_rt_jvlink.py` | JVRTOpen経由でリアルタイム系データ取得 (32bit専用) |
| `scripts/load_to_db.py` | JSONLからDB投入 |
| `app/infrastructure/parsers.py` | レコードパーサー群 |
| `.agent/workflows/batch-download.md` | 2年刻みバッチ取得手順 |

---

## コマンド一覧

### データ取得 (32bit Python)
```powershell
# 2年刻みで取得 (推奨)
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py \
    --from-date 20160101 --to-date 20171231 --option 4

# マスタデータ (DIFF) 取得
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py \
    --dataspec DIFF --from-date 20160101 --option 4
```

### 時系列オッズ取得 (32bit Python)
```powershell
# データ有無を確認 (dry-run)
.\.venv32\Scripts\python.exe scripts/extract_rt_jvlink.py \
    --dataspec 0B41 --racekey 2016122506010101 --dry-run

# 実際に取得
.\.venv32\Scripts\python.exe scripts/extract_rt_jvlink.py \
    --dataspec 0B41 --racekey 2016122506010101
```

### DBロード (64bit Python)
```powershell
# 全ファイルを一括ロード (推奨)
uv run python scripts/load_to_db.py --input-dir data/

# ワイルドカードで指定
uv run python scripts/load_to_db.py --input "data/RACE_*.jsonl"
uv run python scripts/load_to_db.py --input "data/DIFF_*.jsonl"

# 単一ファイル指定
uv run python scripts/load_to_db.py --input data/RACE_20260207_123456.jsonl
```

> **Note**: `--input-dir` はアルファベット順で処理されるため、  
> DIFF_* → RACE_* の順で自動的にマスタが先にロードされます。

### テスト
```powershell
uv run pytest -q
```

---

## 次のステップ

1. `/batch-download` ワークフローで過去10年分を取得
2. 修正済みパーサーでデータ再ロード
3. 時系列オッズ (0B41/0B42) のバッチ取得・DB投入
4. 分析・可視化API実装
