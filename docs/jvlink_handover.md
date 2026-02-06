# JV-Link データ取り込みパイプライン - 引継ぎ資料

## 現在のステータス (2026-02-07 更新)

| 項目 | 状態 | 備考 |
|------|------|------|
| JV-Link接続 | ✅ 完了 | 32bit Python + COM |
| データ取得 | ✅ 完了 | `--to-date` オプション追加済 |
| パーサー | ✅ 完了 | 仕様書4.9.0.1準拠（オフセット修正済） |
| DB保存 | ✅ 完了 | FK違反対策実装済 |

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

---

## 主な実装ファイル

| ファイル | 役割 |
|----------|------|
| `scripts/extract_jvlink.py` | JV-Linkからデータ取得 (32bit専用) |
| `scripts/load_to_db.py` | JSONLからDB投入 |
| `app/infrastructure/parsers.py` | レコードパーサー群 |
| `.agent/workflows/batch-download.md` | 2年刻みバッチ取得手順 |

---

## コマンド一覧

```powershell
# データ取得 (32bit Python)
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py \
    --from-date 20160101 --to-date 20171231 --option 4

# DBロード
uv run python scripts/load_to_db.py --input data/RACE_*.jsonl

# テスト
uv run pytest -q
```

---

## 次のステップ

1. `/batch-download` ワークフローで過去10年分を取得
2. 修正済みパーサーでデータ再ロード
3. 分析・可視化API実装
